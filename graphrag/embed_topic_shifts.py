#!/usr/bin/env python3
"""Detect topic shifts in cross-edition index using embedding similarity.

Embeds the opening text (first 500 words) of each article-edition pair,
then computes pairwise cosine similarity within each cross-edition entry.
Entries with low similarity between editions are flagged as topic shifts.

Requires: sentence-transformers, numpy, scipy

Usage:
    # Full run on GPU (HPC):
    python graphrag/embed_topic_shifts.py

    # Analyze pre-computed embeddings (local, CPU):
    python graphrag/embed_topic_shifts.py --analyze-only

    # Tune threshold against known topic shifts:
    python graphrag/embed_topic_shifts.py --analyze-only --threshold 0.5

    # Incremental: only re-embed articles that changed:
    python graphrag/embed_topic_shifts.py --incremental
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
INDEX_PATH = REPO_ROOT / "data" / "cross_edition_index.jsonl"
MANIFEST_DIFF_PATH = REPO_ROOT / "data" / "article_manifest.diff.json"
TOPIC_SHIFT_REPORT_PATH = REPO_ROOT / "data" / "topic_shift_report.md"

EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "topic_shift_embeddings.npz"
METADATA_PATH = EMBEDDINGS_DIR / "topic_shift_metadata.jsonl"
ANALYSIS_PATH = EMBEDDINGS_DIR / "topic_shift_analysis.jsonl"
DETECTIONS_PATH = EMBEDDINGS_DIR / "topic_shift_detections.md"

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
OPENING_WORDS = 500
DEFAULT_THRESHOLD = 0.75
EMBED_DIM = 1024
BATCH_SIZE = 64

# Known topic shifts from manual analysis (data/topic_shift_report.md)
KNOWN_SHIFTS = {
    "BLACK", "TEMPLE", "DOUGLAS", "BARRY", "WOOD", "JOHN",
    "BULL", "MOORE", "PASSION", "CLARKE", "PHILIP", "SIGN",
}


def load_cross_edition_index() -> list[dict]:
    with open(INDEX_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_export_articles() -> dict[str, dict]:
    """Load all export articles keyed by (title, edition_year).

    The cross-edition index uses article_ids from the articles/ dir (e.g.,
    eb_1st_1771_v01_3524) but export files use sequential IDs (eb_1st_1771_000001).
    So we key by (title, edition_year) for reliable lookup.
    """
    articles = {}
    for fpath in sorted(EXPORT_DIR.glob("eb_*_*.jsonl")):
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                key = (art["title"], art["edition_year"])
                # Keep the longest version if duplicates exist
                if key not in articles or art.get("word_count", 0) > articles[key].get("word_count", 0):
                    articles[key] = art
    return articles


def get_opening_text(text: str, n_words: int = OPENING_WORDS) -> str:
    """Extract the first n_words from article text."""
    words = text.split()
    return " ".join(words[:n_words])


def load_changed_article_ids() -> set[str] | None:
    """Load changed article IDs from manifest diff. Returns None if no diff."""
    if not MANIFEST_DIFF_PATH.exists():
        return None
    with open(MANIFEST_DIFF_PATH) as f:
        diff = json.load(f)
    return set(diff.get("added", [])) | set(diff.get("changed", []))


def build_embedding_inputs(index: list[dict], articles: dict[str, dict],
                           incremental_ids: set[str] | None = None
                           ) -> tuple[list[str], list[dict]]:
    """Build list of texts to embed and their metadata.

    Returns (texts, metadata) where each text is prefixed for nomic-embed.
    If incremental_ids is provided, only include articles with those IDs.
    """
    texts = []
    metadata = []

    for entry in index:
        for year_str, ed_info in entry["editions"].items():
            title = ed_info.get("title", entry["id"])
            year = int(year_str)
            article_id = ed_info.get("article_id", "")

            if incremental_ids is not None and article_id not in incremental_ids:
                continue

            # Look up by (title, edition_year) since export uses different IDs
            art = articles.get((title, year))
            if not art or not art.get("text"):
                continue

            opening = get_opening_text(art["text"])
            # Qwen3-Embedding instruction-aware prefix
            texts.append(
                "Instruct: Represent this opening passage from an "
                "18th-19th century Encyclopedia Britannica article\n"
                f"Query: {opening}"
            )
            metadata.append({
                "article_id": article_id,
                "cross_edition_id": entry["id"],
                "edition_year": year,
                "title": title,
                "word_count": ed_info.get("word_count", 0),
                "text_preview": opening[:200],
            })

    return texts, metadata


def embed_texts(texts: list[str], model_name: str = MODEL_NAME,
                device: str = None) -> np.ndarray:
    """Embed texts using sentence-transformers. Returns (N, dim) array."""
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        if torch.cuda.is_available():
            # Check VRAM — need ~2GB for model + batch
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            device = "cuda" if vram >= 6 else "cpu"
        else:
            device = "cpu"

    print(f"Loading model {model_name} on {device}...")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    batch_size = BATCH_SIZE if device != "cpu" else 32
    print(f"Embedding {len(texts):,} texts in batches of {batch_size} on {device}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine = dot product
        truncate_dim=EMBED_DIM,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
    return embeddings


def save_embeddings(embeddings: np.ndarray, metadata: list[dict]):
    """Save embeddings to npz + metadata to jsonl."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_PATH, embeddings=embeddings)
    with open(METADATA_PATH, "w") as f:
        for rec in metadata:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(metadata):,} embeddings to {EMBEDDINGS_PATH}")


def load_embeddings() -> tuple[np.ndarray, list[dict]]:
    """Load pre-computed embeddings."""
    data = np.load(EMBEDDINGS_PATH)
    embeddings = data["embeddings"]
    metadata = []
    with open(METADATA_PATH) as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    return embeddings, metadata


def merge_incremental(old_emb: np.ndarray, old_meta: list[dict],
                      new_emb: np.ndarray, new_meta: list[dict]
                      ) -> tuple[np.ndarray, list[dict]]:
    """Merge new embeddings into existing set, replacing changed articles."""
    new_ids = {m["article_id"] for m in new_meta}
    # Keep old entries that weren't re-embedded
    keep_indices = [i for i, m in enumerate(old_meta) if m["article_id"] not in new_ids]
    if keep_indices:
        kept_emb = old_emb[keep_indices]
        kept_meta = [old_meta[i] for i in keep_indices]
        merged_emb = np.vstack([kept_emb, new_emb])
        merged_meta = kept_meta + new_meta
    else:
        merged_emb = new_emb
        merged_meta = new_meta
    return merged_emb, merged_meta


def analyze_topic_shifts(embeddings: np.ndarray, metadata: list[dict],
                         threshold: float) -> list[dict]:
    """Compute pairwise similarity within each cross-edition entry.

    Returns analysis records sorted by min_similarity (most shifted first).
    """
    # Group by cross_edition_id
    groups = defaultdict(list)
    for i, meta in enumerate(metadata):
        groups[meta["cross_edition_id"]].append((i, meta))

    results = []
    for ce_id, members in groups.items():
        if len(members) < 2:
            continue

        indices = [m[0] for m in members]
        years = [m[1]["edition_year"] for m in members]
        embs = embeddings[indices]  # (K, dim) for K editions

        # Pairwise cosine similarity (embeddings are L2-normalized)
        sim_matrix = embs @ embs.T

        pairs = []
        min_sim = 1.0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                s = float(sim_matrix[i, j])
                pairs.append({
                    "a": years[i], "b": years[j],
                    "sim": round(s, 4),
                    "title_a": members[i][1]["title"],
                    "title_b": members[j][1]["title"],
                })
                if s < min_sim:
                    min_sim = s

        needs_split = min_sim < threshold

        # Classify the type of shift
        shift_type = "ok"
        if needs_split:
            shift_type = classify_shift(pairs, members, threshold)

        # Cluster editions by similarity if flagged
        clusters = []
        if needs_split and len(members) >= 2:
            clusters = cluster_editions(sim_matrix, years, threshold)

        results.append({
            "cross_edition_id": ce_id,
            "edition_count": len(members),
            "min_similarity": round(min_sim, 4),
            "needs_split": needs_split,
            "shift_type": shift_type,
            "clusters": clusters,
            "pairs": sorted(pairs, key=lambda p: p["sim"]),
            "editions": {m[1]["edition_year"]: {
                "title": m[1]["title"],
                "word_count": m[1]["word_count"],
                "preview": m[1]["text_preview"][:100],
            } for m in members},
        })

    results.sort(key=lambda r: r["min_similarity"])
    return results


SHORT_DEFINITION_THRESHOLD = 200  # words — 1771 entries below this are just definitions


def classify_shift(pairs: list[dict], members: list[tuple],
                   threshold: float) -> str:
    """Classify a flagged entry into one of three categories.

    Returns:
        "topic_shift"    — multiple editions have genuinely different topics
        "single_outlier" — one edition has different content (swallowed/misparse)
        "short_expansion" — short definition (typically 1771) vs later long article
    """
    from collections import Counter

    low_pairs = [p for p in pairs if p["sim"] < threshold]
    if not low_pairs:
        return "ok"

    edition_count = len(members)

    # Count which editions appear in low-similarity pairs
    ed_counts = Counter()
    for p in low_pairs:
        ed_counts[p["a"]] += 1
        ed_counts[p["b"]] += 1

    most_common_ed, mc_count = ed_counts.most_common(1)[0]
    is_single = mc_count / len(low_pairs) > 0.8 and edition_count >= 3

    if is_single:
        # Check if the outlier is a short definition
        outlier_wc = 0
        for _, meta in members:
            if meta["edition_year"] == most_common_ed:
                outlier_wc = meta["word_count"]
                break
        if outlier_wc < SHORT_DEFINITION_THRESHOLD:
            return "short_expansion"
        return "single_outlier"

    return "topic_shift"


def cluster_editions(sim_matrix: np.ndarray, years: list[int],
                     threshold: float) -> list[list[int]]:
    """Cluster editions by similarity using agglomerative clustering."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(years)
    if n < 2:
        return [years]

    # Convert similarity to distance
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)
    # Ensure symmetry and non-negativity
    dist_matrix = np.maximum(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters = defaultdict(list)
    for year, label in zip(years, labels):
        clusters[label].append(year)
    return [sorted(v) for v in clusters.values()]


def validate_against_known(results: list[dict]) -> dict:
    """Check detection results against known topic shifts."""
    flagged_ids = {r["cross_edition_id"] for r in results if r["needs_split"]}
    true_pos = KNOWN_SHIFTS & flagged_ids
    false_neg = KNOWN_SHIFTS - flagged_ids
    false_pos_count = len(flagged_ids) - len(true_pos)

    return {
        "known": len(KNOWN_SHIFTS),
        "flagged": len(flagged_ids),
        "true_positives": len(true_pos),
        "false_negatives": len(false_neg),
        "false_positive_candidates": false_pos_count,
        "recall": len(true_pos) / len(KNOWN_SHIFTS) if KNOWN_SHIFTS else 0,
        "missed": sorted(false_neg),
    }


def write_analysis(results: list[dict]):
    """Write analysis JSONL."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Analysis written: {ANALYSIS_PATH}")


def _render_entry(r: dict, threshold: float) -> list[str]:
    """Render a single flagged entry as markdown lines."""
    lines = []
    known_marker = " **[KNOWN]**" if r["cross_edition_id"] in KNOWN_SHIFTS else ""
    lines.append(f"### {r['cross_edition_id']}{known_marker}")
    lines.append("")
    lines.append(f"Min similarity: **{r['min_similarity']:.3f}** "
                  f"({r['edition_count']} editions)")
    if r["clusters"]:
        cluster_strs = [str(c) for c in r["clusters"]]
        lines.append(f"Suggested clusters: {' | '.join(cluster_strs)}")
    lines.append("")

    lines.append("| Year | Title | Words | Opening |")
    lines.append("|------|-------|-------|---------|")
    for year in sorted(r["editions"].keys()):
        ed = r["editions"][year]
        preview = ed["preview"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {year} | {ed['title']} | {ed['word_count']:,} | {preview}... |")
    lines.append("")

    low_pairs = [p for p in r["pairs"] if p["sim"] < threshold][:3]
    if low_pairs:
        lines.append("Lowest similarity pairs:")
        for p in low_pairs:
            lines.append(f"- {p['a']} vs {p['b']}: **{p['sim']:.3f}**")
        lines.append("")
    return lines


def write_detections_report(results: list[dict], threshold: float,
                            validation: dict):
    """Write human-readable markdown report of detected topic shifts."""
    flagged = [r for r in results if r["needs_split"]]
    topic_shifts = [r for r in flagged if r["shift_type"] == "topic_shift"]
    single_outliers = [r for r in flagged if r["shift_type"] == "single_outlier"]
    short_expansions = [r for r in flagged if r["shift_type"] == "short_expansion"]

    lines = [
        "# Detected Topic Shifts in Cross-Edition Index",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Model:** {MODEL_NAME}",
        f"**Method:** First {OPENING_WORDS} words embedded, "
        f"pairwise cosine similarity, threshold={threshold}",
        f"**Entries analyzed:** {len(results):,}",
        "",
        "## Summary",
        "",
        f"| Category | Count | Description |",
        f"|----------|-------|-------------|",
        f"| **Topic shifts** | {len(topic_shifts)} | Multiple editions cover different topics — need index splits |",
        f"| **Single-edition outliers** | {len(single_outliers)} | One edition has different content — likely swallowed/misparse |",
        f"| **Short expansions** | {len(short_expansions)} | Short definition expanded in later editions — noise |",
        f"| **Total flagged** | {len(flagged)} | |",
        "",
        "## Validation Against Known Shifts",
        "",
        f"- Known topic shifts: {validation['known']}",
        f"- Detected (true positives): {validation['true_positives']}",
        f"- Missed (false negatives): {validation['false_negatives']} — {validation['missed']}",
        f"- Recall: {validation['recall']:.0%}",
        "",
        "---",
        "",
        f"## 1. Topic Shifts ({len(topic_shifts)} entries)",
        "",
        "These entries have multiple editions with genuinely different topics.",
        "Each should be split into separate cross-edition index entries.",
        "",
    ]

    for r in topic_shifts:
        lines.extend(_render_entry(r, threshold))

    lines.extend([
        "---",
        "",
        f"## 2. Single-Edition Outliers ({len(single_outliers)} entries)",
        "",
        "One edition has different content, likely a swallowed article or parser error.",
        "The outlier edition should be investigated and potentially removed or fixed.",
        "",
    ])

    for r in single_outliers:
        lines.extend(_render_entry(r, threshold))

    lines.extend([
        "---",
        "",
        f"## 3. Short Expansions ({len(short_expansions)} entries)",
        "",
        "Short definitions (typically 1771, <200 words) that expanded into full articles",
        "in later editions. These are not topic shifts — just editorial growth. Listed",
        "here for completeness but no action needed.",
        "",
    ])

    # Just list these as a compact table, no full detail
    lines.append("| Entry | Min Sim | 1st Ed Words | Max Words |")
    lines.append("|-------|---------|-------------|-----------|")
    for r in short_expansions:
        eds = r["editions"]
        wcs = {y: eds[y]["word_count"] for y in eds}
        min_year = min(eds.keys())
        lines.append(f"| {r['cross_edition_id']} | {r['min_similarity']:.3f} | "
                      f"{wcs.get(min_year, 0):,} | {max(wcs.values()):,} |")
    lines.append("")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DETECTIONS_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written:   {DETECTIONS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Detect topic shifts via embeddings")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip embedding, analyze pre-computed embeddings")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-embed articles in manifest diff")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold for flagging (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Embedding model (default: {MODEL_NAME})")
    args = parser.parse_args()

    if args.analyze_only:
        if not EMBEDDINGS_PATH.exists():
            sys.exit(f"No embeddings found at {EMBEDDINGS_PATH}. Run without --analyze-only first.")
        print("Loading pre-computed embeddings...")
        embeddings, metadata = load_embeddings()
        print(f"  {len(metadata):,} embeddings loaded")
    else:
        print("Loading cross-edition index...")
        index = load_cross_edition_index()
        print(f"  {len(index):,} entries")

        print("Loading export articles...")
        articles = load_export_articles()
        print(f"  {len(articles):,} articles")

        incremental_ids = None
        if args.incremental:
            incremental_ids = load_changed_article_ids()
            if incremental_ids is not None:
                print(f"Incremental mode: {len(incremental_ids):,} changed articles")
                if not incremental_ids:
                    print("No changes detected, nothing to embed.")
                    # Still run analysis on existing embeddings
                    if EMBEDDINGS_PATH.exists():
                        embeddings, metadata = load_embeddings()
                    else:
                        return
            else:
                print("No manifest diff found, embedding all articles.")

        texts, metadata = build_embedding_inputs(index, articles, incremental_ids)
        print(f"  {len(texts):,} article-edition openings to embed")

        if not texts:
            print("Nothing to embed.")
            return

        embeddings = embed_texts(texts, args.model)

        # Merge with existing if incremental
        if args.incremental and EMBEDDINGS_PATH.exists():
            old_emb, old_meta = load_embeddings()
            embeddings, metadata = merge_incremental(old_emb, old_meta, embeddings, metadata)
            print(f"  Merged: {len(metadata):,} total embeddings")

        save_embeddings(embeddings, metadata)

    # Analysis
    print(f"\nAnalyzing topic shifts (threshold={args.threshold})...")
    results = analyze_topic_shifts(embeddings, metadata, args.threshold)

    flagged = [r for r in results if r["needs_split"]]
    topic_shifts = [r for r in flagged if r["shift_type"] == "topic_shift"]
    single_outliers = [r for r in flagged if r["shift_type"] == "single_outlier"]
    short_expansions = [r for r in flagged if r["shift_type"] == "short_expansion"]

    print(f"  {len(results):,} entries analyzed")
    print(f"  {len(topic_shifts):,} topic shifts (need index splits)")
    print(f"  {len(single_outliers):,} single-edition outliers (likely swallowed/misparse)")
    print(f"  {len(short_expansions):,} short expansions (noise)")

    validation = validate_against_known(results)
    print(f"\nValidation: {validation['true_positives']}/{validation['known']} "
          f"known shifts detected (recall={validation['recall']:.0%})")
    if validation["missed"]:
        print(f"  Missed: {', '.join(validation['missed'])}")

    write_analysis(results)
    write_detections_report(results, args.threshold, validation)

    # Quick summary of top topic shifts
    print(f"\nTop 15 topic shifts:")
    for r in topic_shifts[:15]:
        marker = " *" if r["cross_edition_id"] in KNOWN_SHIFTS else ""
        clusters = " | ".join(str(c) for c in r["clusters"]) if r["clusters"] else ""
        print(f"  {r['cross_edition_id']:30s} min_sim={r['min_similarity']:.3f} "
              f"({r['edition_count']} eds) {clusters}{marker}")


if __name__ == "__main__":
    main()
