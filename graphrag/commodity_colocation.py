#!/usr/bin/env python3
"""
Windowed co-occurrence analysis: find toponyms near commodity mentions.

Instead of article-level co-occurrence (where a 50-page article on CHEMISTRY
counts as linking "sugar" to every place mentioned anywhere in it), this finds
toponyms within a word window around each commodity mention in the raw text.

Usage:
    python graphrag/commodity_colocation.py --commodity sugar
    python graphrag/commodity_colocation.py --commodity sugar --window 100
    python graphrag/commodity_colocation.py --commodity cotton --window 200 --min-count 5
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
NER_DIR = REPO_ROOT / "data" / "ner"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# Skip known false-positive toponyms
TOPONYM_SKIP = {
    "W. Long.", "W. Long", "N. Lat.", "N. Lat", "E. Long.", "E. Long",
    "CANTARO", "river", "east", "west", "north", "south",
}


def find_all_occurrences(text_lower: str, term: str) -> list[int]:
    """Find all character positions of term in text (case-insensitive)."""
    positions = []
    start = 0
    while True:
        idx = text_lower.find(term, start)
        if idx == -1:
            break
        # Check word boundaries to avoid matching "sugarcane" for "sugar"
        # (but do match "sugar-" "sugar," etc.)
        before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
        after_end = idx + len(term)
        after_ok = after_end >= len(text_lower) or not text_lower[after_end].isalpha()
        if before_ok and after_ok:
            positions.append(idx)
        start = idx + 1
    return positions


def char_to_word_index(text: str) -> list[int]:
    """Build mapping from character position to word index.

    Returns a list where result[char_pos] = word_index.
    """
    mapping = [0] * len(text)
    word_idx = 0
    in_word = False
    for i, ch in enumerate(text):
        if ch.isspace():
            in_word = False
        else:
            if not in_word:
                word_idx += 1
                in_word = True
        mapping[i] = word_idx
    return mapping


def find_toponyms_near_commodity(text: str, commodity: str,
                                  toponyms: list[str],
                                  window: int) -> list[tuple[str, int]]:
    """Find toponyms within `window` words of each commodity mention.

    Returns list of (toponym, distance_in_words) pairs.
    """
    text_lower = text.lower()
    commodity_lower = commodity.lower()

    # Find commodity positions
    commodity_positions = find_all_occurrences(text_lower, commodity_lower)
    if not commodity_positions:
        return []

    # Build char->word mapping
    word_map = char_to_word_index(text)

    # Get word indices of commodity mentions
    commodity_word_indices = [word_map[pos] for pos in commodity_positions]

    # Find each toponym and check proximity
    results = []
    for toponym in toponyms:
        if toponym in TOPONYM_SKIP:
            continue
        toponym_lower = toponym.lower()
        toponym_positions = find_all_occurrences(text_lower, toponym_lower)
        for tpos in toponym_positions:
            t_word = word_map[tpos]
            for c_word in commodity_word_indices:
                dist = abs(t_word - c_word)
                if dist <= window and dist > 0:  # >0 to skip self
                    results.append((toponym, dist))
                    break  # Count each toponym occurrence once per commodity match

    return results


def run_analysis(commodity: str, window: int = 150, min_count: int = 3,
                 top_n: int = 20):
    """Run windowed co-occurrence across all editions."""
    print(f"Commodity: '{commodity}', window: {window} words, min mentions: {min_count}\n")

    for year in EDITION_YEARS:
        # Load NER results to get toponym list per article
        ner_files = list(NER_DIR.glob(f"eb_*_{year}.entities.jsonl"))
        ner_file = [f for f in ner_files if '_v' not in f.stem]
        if not ner_file:
            print(f"  {year}: no NER file, skipping")
            continue
        ner_file = ner_file[0]

        # Build article_id -> toponyms mapping
        article_toponyms = {}
        article_has_commodity = set()
        with open(ner_file) as f:
            for line in f:
                rec = json.loads(line)
                aid = rec["article_id"]
                toponyms = [e["text"] for e in rec["entities"]
                           if e["type"] == "TOPONYM"]
                commodities = [e["text"].lower() for e in rec["entities"]
                              if e["type"] == "COMMODITY"]
                article_toponyms[aid] = toponyms
                if commodity.lower() in commodities:
                    article_has_commodity.add(aid)

        if not article_has_commodity:
            print(f"  {year}: no articles with '{commodity}'")
            continue

        # Load text for relevant articles
        export_files = list(EXPORT_DIR.glob(f"eb_*_{year}.jsonl"))
        if not export_files:
            continue

        nearby_toponyms = Counter()
        avg_distances = defaultdict(list)
        n_articles = 0

        with open(export_files[0]) as f:
            for line in f:
                art = json.loads(line)
                aid = art["article_id"]
                if aid not in article_has_commodity:
                    continue

                text = art.get("text", "")
                if not text:
                    continue

                toponyms = article_toponyms.get(aid, [])
                if not toponyms:
                    continue

                pairs = find_toponyms_near_commodity(
                    text, commodity, toponyms, window
                )
                if pairs:
                    n_articles += 1
                    for toponym, dist in pairs:
                        nearby_toponyms[toponym] += 1
                        avg_distances[toponym].append(dist)

        # Compute edition label
        ed_label = ner_file.stem.split('_')[1]

        # Print results
        top = [(t, c, sum(avg_distances[t]) / len(avg_distances[t]))
               for t, c in nearby_toponyms.most_common(50)
               if c >= min_count][:top_n]

        print(f"=== {ed_label} edition ({year}) — {n_articles} articles with nearby toponyms ===")
        if top:
            print(f"  {'Place':25s} {'Mentions':>8s} {'Avg dist':>10s}")
            print(f"  {'─' * 25} {'─' * 8} {'─' * 10}")
            for place, count, avg_dist in top:
                print(f"  {place:25s} {count:8d} {avg_dist:8.0f}w")
        else:
            print(f"  No toponyms with >= {min_count} co-occurrences")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Windowed commodity-toponym co-occurrence analysis"
    )
    parser.add_argument(
        "--commodity", type=str, required=True,
        help="Commodity to search for (e.g., sugar, cotton, slaves)",
    )
    parser.add_argument(
        "--window", type=int, default=150,
        help="Window size in words around commodity mention (default: 150)",
    )
    parser.add_argument(
        "--min-count", type=int, default=3,
        help="Minimum co-occurrences to report (default: 3)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top locations to show per edition (default: 20)",
    )
    args = parser.parse_args()

    run_analysis(
        commodity=args.commodity,
        window=args.window,
        min_count=args.min_count,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
