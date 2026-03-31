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


def write_report(detections: list[dict]):
    """Write markdown report."""
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["classification"]].append(d)

    lines = [
        "# Swallowed Article Detections",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Method:** Consecutive paragraph embedding similarity (voyage-4-large)",
        f"**Total detections:** {len(detections)}",
        "",
        "## Summary by Classification",
        "",
        "| Type | Count | Description |",
        "|------|-------|-------------|",
        f"| mid_word | {len(by_class['mid_word'])} | Starts mid-word — definitely swallowed |",
        f"| mid_sentence | {len(by_class['mid_sentence'])} | Starts mid-sentence — likely swallowed |",
        f"| new_headword | {len(by_class['new_headword'])} | ALLCAPS heading — missed headword boundary |",
        f"| person_bio | {len(by_class['person_bio'])} | Person name — swallowed biography |",
        f"| topic_change | {len(by_class['topic_change'])} | Generic topic change — may be legitimate |",
        "",
    ]

    for cls in ["mid_word", "mid_sentence", "new_headword", "person_bio", "topic_change"]:
        items = by_class.get(cls, [])
        if not items:
            continue
        lines.append(f"## {cls} ({len(items)} detections)")
        lines.append("")
        for d in items[:50]:  # cap at 50 per category for readability
            before = d["before_end"][-60:].replace("\n", " ")
            after = d["after_start"][:80].replace("\n", " ")
            lines.append(
                f"- **{d['title']}** ({d['edition_year']}) "
                f"para {d['para_before']}→{d['para_after']} "
                f"sim={d['similarity']:.3f} drop={d['drop']:.3f}"
            )
            lines.append(f"  - `...{before}`")
            lines.append(f"  - `{after}...`")
            lines.append("")
        if len(items) > 50:
            lines.append(f"  ... and {len(items) - 50} more")
            lines.append("")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Report: {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser(description="Detect swallowed articles")
    parser.add_argument("--edition-year", type=int)
    parser.add_argument("--drop-threshold", type=float, default=DROP_THRESHOLD)
    parser.add_argument("--abs-threshold", type=float, default=ABS_THRESHOLD)
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
        print(f"  {len(detections)} breaks detected")
        all_detections.extend(detections)

    print(f"\nTotal: {len(all_detections)} detections across {len(years)} editions")

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


if __name__ == "__main__":
    main()
