#!/usr/bin/env python3
"""Detect swallowed articles via paragraph-level embedding similarity.

Loads paragraph embeddings and finds articles where consecutive paragraphs
have sharp topic breaks — indicating a missed headword boundary.

Usage:
    python graphrag/detect_swallowed.py                      # all editions
    python graphrag/detect_swallowed.py --edition-year 1810  # single edition
    python graphrag/detect_swallowed.py --threshold 0.25     # adjust sensitivity
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
OUTPUT_JSONL = REPO_ROOT / "data" / "swallowed_detections.jsonl"
OUTPUT_MD = REPO_ROOT / "data" / "swallowed_detections.md"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]
MIN_ARTICLE_PARAS = 3    # need >= 3 paragraphs to detect internal breaks
DROP_THRESHOLD = 0.25    # min drop from rolling average to flag
ABS_THRESHOLD = 0.35     # absolute similarity below this is suspicious
ROLLING_WINDOW = 5       # paragraphs for rolling average

# Per-category similarity thresholds (from spot-checking)
# mid_word/mid_sentence: structural signals, always keep
# new_headword: ALLCAPS is strong evidence, ~80% real even at 0.35
# person_bio/topic_change: classifier less reliable, need stronger signal
CATEGORY_THRESHOLDS = {
    "mid_word": 0.50,
    "mid_sentence": 0.50,
    "new_headword": 0.35,
    "person_bio": 0.20,
    "topic_change": 0.20,
}


def load_paragraph_embeddings(edition_year: int) -> tuple[np.ndarray, list[dict]]:
    """Load paragraph embeddings for one edition."""
    fp = EMBEDDINGS_DIR / f"eb_{edition_year}.paragraphs.jsonl"
    if not fp.exists():
        return np.array([]), []

    embeddings = []
    metadata = []
    with open(fp) as f:
        for line in f:
            rec = json.loads(line)
            embeddings.append(rec["embedding"])
            metadata.append({k: v for k, v in rec.items() if k != "embedding"})

    return np.array(embeddings, dtype=np.float32), metadata


def detect_breaks(emb_matrix: np.ndarray, meta: list[dict],
                  drop_threshold: float = DROP_THRESHOLD,
                  abs_threshold: float = ABS_THRESHOLD) -> list[dict]:
    """Find topic breaks within articles."""
    # Group paragraphs by article
    articles = defaultdict(list)
    for i, m in enumerate(meta):
        articles[m["article_id"]].append((i, m.get("para_index", 0)))

    detections = []

    for aid, paras in articles.items():
        if len(paras) < MIN_ARTICLE_PARAS:
            continue

        paras.sort(key=lambda x: x[1])  # sort by paragraph index
        indices = [p[0] for p in paras]

        # Compute consecutive similarities
        sims = []
        for j in range(len(indices) - 1):
            sim = float(emb_matrix[indices[j]] @ emb_matrix[indices[j + 1]])
            sims.append(sim)

        if not sims:
            continue

        # Rolling average
        avg_sims = []
        for j in range(len(sims)):
            window_start = max(0, j - ROLLING_WINDOW // 2)
            window_end = min(len(sims), j + ROLLING_WINDOW // 2 + 1)
            avg_sims.append(np.mean(sims[window_start:window_end]))

        # Find breaks
        for j, sim in enumerate(sims):
            drop = avg_sims[j] - sim
            is_break = (drop > drop_threshold and sim < 0.5) or sim < abs_threshold

            if is_break:
                before_meta = meta[indices[j]]
                after_meta = meta[indices[j + 1]]

                # Get text snippets
                before_text = before_meta.get("text", "")
                after_text = after_meta.get("text", "")

                # Classify the break
                classification = classify_break(after_text)

                detections.append({
                    "article_id": aid,
                    "title": before_meta["title"],
                    "edition_year": before_meta["edition_year"],
                    "para_before": paras[j][1],
                    "para_after": paras[j + 1][1],
                    "total_paras": len(paras),
                    "similarity": round(sim, 3),
                    "avg_similarity": round(avg_sims[j], 3),
                    "drop": round(drop, 3),
                    "classification": classification,
                    "before_end": before_text[-100:] if before_text else "",
                    "after_start": after_text[:150] if after_text else "",
                })

    detections.sort(key=lambda d: d["similarity"])
    return detections


def classify_break(after_text: str) -> str:
    """Classify the type of break based on the text that follows."""
    if not after_text:
        return "unknown"

    text = after_text.strip()

    # Mid-word: starts with lowercase fragment (1-3 chars, not a common word)
    first_word = text.split()[0] if text.split() else ""
    common_starts = {
        "in", "a", "an", "or", "of", "is", "the", "by", "on", "to", "as",
        "no", "so", "if", "it", "he", "we", "at", "be", "do", "up", "and",
        "for", "are", "but", "not", "was", "one", "has", "had", "its", "may",
        "new", "old", "see", "any", "all", "can", "how", "our", "out", "few",
    }
    fw_lower = first_word.lower().rstrip(".,;:!?")
    if len(fw_lower) <= 3 and fw_lower not in common_starts and text[0].islower():
        return "mid_word"

    # Mid-sentence: starts lowercase (but not a dictionary-style opening)
    if text[0].islower() and fw_lower not in common_starts:
        return "mid_sentence"

    # Starts with ALLCAPS heading — likely a new article headword
    if re.match(r'^[A-Z]{3,}', text):
        return "new_headword"

    # Starts with a person name pattern (Capitalized, Capitalized)
    if re.match(r'^[A-Z][a-z]+,?\s+[A-Z][a-z]', text):
        return "person_bio"

    return "topic_change"


def apply_thresholds(detections: list[dict]) -> list[dict]:
    """Filter detections by per-category similarity thresholds."""
    kept = []
    for d in detections:
        thresh = CATEGORY_THRESHOLDS.get(d["classification"], 0.20)
        if d["similarity"] < thresh:
            kept.append(d)
    return kept


def extract_break_headword(after_text: str) -> str:
    """Try to extract a headword from the text after the break.

    Returns a normalized headword for cross-edition matching, or empty string
    if no plausible headword can be extracted.
    """
    text = after_text.strip().replace("\n", " ")
    if not text:
        return ""

    # Bold markdown headword: **WORD** or **Word**
    m = re.match(r'\*\*([A-Za-z][A-Za-z\s\-\'\.]+?)\*\*', text)
    if m:
        hw = m.group(1).strip().upper()
        if len(hw) >= 2:
            return hw

    # ALLCAPS headword (3+ chars, not common words)
    m = re.match(r'([A-Z][A-Z\s\-\']{2,})', text)
    if m:
        hw = m.group(1).strip()
        # Filter out common words that aren't headwords
        if hw not in {"THE", "AND", "FOR", "BUT", "NOT", "THIS", "THAT",
                      "WITH", "FROM", "HAVE", "BEEN", "WERE", "THEY",
                      "WHICH", "THEIR", "THERE", "WHEN", "WHAT", "SOME",
                      "THESE", "THOSE", "SUCH", "WILL", "EACH", "THAN",
                      "AFTER", "OTHER", "INTO", "UPON", "ALSO", "MOST",
                      "VERY", "OVER", "PART", "CHAP"}:
            return hw

    # Capitalized name (person bio): "Name, ..." or "Name Name"
    m = re.match(r'([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)', text)
    if m:
        hw = m.group(1).strip().upper()
        if len(hw) >= 3:
            return hw

    # No plausible headword found
    return ""


def group_cross_edition(detections: list[dict]) -> list[dict]:
    """Group detections by parent article title + break headword across editions.

    Breaks that appear in multiple editions are much more likely to be real
    swallowed articles, since the same content was carried across editions.
    Detections with no extractable headword are left ungrouped (count=1).
    """
    # Extract a break headword for each detection
    for d in detections:
        d["break_headword"] = extract_break_headword(d["after_start"])

    # Group by (parent_title, break_headword) — only if headword is non-empty
    groups = defaultdict(list)
    for d in detections:
        bh = d["break_headword"]
        if bh:
            key = (d["title"].upper(), bh)
            groups[key].append(d)

    # Annotate each detection with cross-edition count
    for key, group in groups.items():
        editions = sorted(set(d["edition_year"] for d in group))
        for d in group:
            d["cross_edition_count"] = len(editions)
            d["cross_edition_years"] = editions

    # Detections with no headword get count=1
    for d in detections:
        if "cross_edition_count" not in d:
            d["cross_edition_count"] = 1
            d["cross_edition_years"] = [d["edition_year"]]

    return detections


def write_report(detections: list[dict]):
    """Write markdown report with cross-edition grouping."""
    # Separate multi-edition and single-edition detections
    multi = [d for d in detections if d.get("cross_edition_count", 1) >= 2]
    single = [d for d in detections if d.get("cross_edition_count", 1) == 1]

    # Group multi-edition by (parent, break_headword)
    multi_groups = defaultdict(list)
    for d in multi:
        key = (d["title"].upper(), d["break_headword"])
        multi_groups[key].append(d)

    by_class = defaultdict(list)
    for d in detections:
        by_class[d["classification"]].append(d)

    lines = [
        "# Swallowed Article Detections",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Method:** Consecutive paragraph embedding similarity (voyage-4-large)",
        f"**Thresholds:** per-category (mid_word/mid_sentence: all, new_headword: <0.35, "
        f"person_bio/topic_change: <0.20)",
        f"**Total detections:** {len(detections)} "
        f"({len(multi)} multi-edition, {len(single)} single-edition)",
        "",
        "## Summary by Classification",
        "",
        "| Type | Count | Threshold | Description |",
        "|------|-------|-----------|-------------|",
        f"| mid_word | {len(by_class['mid_word'])} | all | Starts mid-word — definitely swallowed |",
        f"| mid_sentence | {len(by_class['mid_sentence'])} | all | Starts mid-sentence — likely swallowed |",
        f"| new_headword | {len(by_class['new_headword'])} | <0.35 | ALLCAPS heading — missed headword |",
        f"| person_bio | {len(by_class['person_bio'])} | <0.20 | Person name — swallowed biography |",
        f"| topic_change | {len(by_class['topic_change'])} | <0.20 | Topic change — may be legitimate |",
        "",
        "## Cross-Edition Breaks (HIGH CONFIDENCE)",
        "",
        f"**{len(multi_groups)} unique breaks** appearing in 2+ editions "
        f"({len(multi)} total detections):",
        "",
    ]

    # Sort by number of editions (descending), then parent title
    for key in sorted(multi_groups, key=lambda k: (-len(set(d["edition_year"] for d in multi_groups[k])), k[0])):
        group = multi_groups[key]
        parent, bh = key
        editions = sorted(set(d["edition_year"] for d in group))
        best = min(group, key=lambda d: d["similarity"])
        lines.append(
            f"- **{parent}** → {bh} ({len(editions)} editions: {', '.join(str(y) for y in editions)}) "
            f"best sim={best['similarity']:.3f} [{best['classification']}]"
        )

    lines.append("")
    lines.append("## Single-Edition Breaks")
    lines.append("")
    lines.append(f"**{len(single)} detections** in only one edition (lower confidence):")
    lines.append("")

    for cls in ["mid_word", "mid_sentence", "new_headword", "person_bio", "topic_change"]:
        items = [d for d in single if d["classification"] == cls]
        if not items:
            continue
        lines.append(f"### {cls} ({len(items)})")
        lines.append("")
        for d in sorted(items, key=lambda x: x["similarity"])[:30]:
            before = d["before_end"][-60:].replace("\n", " ")
            after = d["after_start"][:80].replace("\n", " ")
            lines.append(
                f"- **{d['title']}** ({d['edition_year']}) "
                f"para {d['para_before']}→{d['para_after']} "
                f"sim={d['similarity']:.3f}"
            )
            lines.append(f"  - `...{before}`")
            lines.append(f"  - `{after}...`")
            lines.append("")
        if len(items) > 30:
            lines.append(f"  ... and {len(items) - 30} more")
            lines.append("")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Report: {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser(description="Detect swallowed articles")
    parser.add_argument("--edition-year", type=int)
    parser.add_argument("--drop-threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--abs-threshold", type=float, default=ABS_THRESHOLD)
    parser.add_argument("--no-filter", action="store_true",
                        help="Output all raw detections (skip per-category thresholds)")
    args = parser.parse_args()

    years = [args.edition_year] if args.edition_year else EDITION_YEARS

    all_detections = []
    for year in years:
        print(f"Loading {year}...", end=" ", flush=True)
        emb, meta = load_paragraph_embeddings(year)
        if len(emb) == 0:
            print("no embeddings found")
            continue
        print(f"{len(meta):,} paragraphs")

        detections = detect_breaks(emb, meta, args.drop_threshold, args.abs_threshold)
        print(f"  {len(detections)} raw breaks")
        all_detections.extend(detections)

    print(f"\nRaw total: {len(all_detections)} detections across {len(years)} editions")

    # Apply per-category thresholds
    if not args.no_filter:
        all_detections = apply_thresholds(all_detections)
        print(f"After thresholds: {len(all_detections)} detections")

    # Cross-edition grouping
    all_detections = group_cross_edition(all_detections)
    multi = sum(1 for d in all_detections if d.get("cross_edition_count", 1) >= 2)
    single = len(all_detections) - multi
    print(f"Cross-edition: {multi} multi-edition, {single} single-edition")

    # Write outputs
    with open(OUTPUT_JSONL, "w") as f:
        for d in all_detections:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"JSONL:  {OUTPUT_JSONL}")

    write_report(all_detections)

    # Quick summary
    by_class = defaultdict(int)
    for d in all_detections:
        by_class[d["classification"]] += 1
    print("\nBy classification:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    # Cross-edition summary
    groups = defaultdict(set)
    for d in all_detections:
        key = (d["title"].upper(), d["break_headword"])
        groups[key].add(d["edition_year"])
    by_count = defaultdict(int)
    for editions in groups.values():
        by_count[len(editions)] += 1
    print("\nCross-edition groups:")
    for n in sorted(by_count):
        label = f"{n} edition{'s' if n > 1 else ''}"
        print(f"  {label}: {by_count[n]} unique breaks")


if __name__ == "__main__":
    main()
