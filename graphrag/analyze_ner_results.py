#!/usr/bin/env python3
"""
Analyze NER results from EarlyModernNER extraction.

Prints summary stats, top entities by type, coverage metrics,
cross-references with concept_index.json, and sample articles for review.

Usage:
    python graphrag/analyze_ner_results.py data/ner/eb_1st_1771.entities.jsonl
    python graphrag/analyze_ner_results.py data/ner/eb_1st_1771.entities.jsonl --top 30 --samples 20
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONCEPT_INDEX = REPO_ROOT / "graphrag" / "concept_index.json"


def load_entities(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def analyze(records: list[dict], top_n: int = 20, n_samples: int = 10):
    total = len(records)
    if not total:
        print("No records found.")
        return

    edition = records[0].get("edition", "?")
    year = records[0].get("edition_year", "?")

    # Aggregate entity stats
    type_counts = Counter()
    entity_freq = {}  # {type: Counter(text -> count)}
    articles_with_ents = 0

    for r in records:
        has_any = False
        for ent in r.get("entities", []):
            et = ent["type"]
            type_counts[et] += 1
            entity_freq.setdefault(et, Counter())[ent["text"]] += 1
            has_any = True
        if has_any:
            articles_with_ents += 1

    # --- Summary ---
    print(f"NER Results: {edition} edition ({year})")
    print(f"{'=' * 60}")
    print(f"  Articles:             {total:,}")
    print(f"  With entities:        {articles_with_ents:,} "
          f"({100 * articles_with_ents / total:.1f}%)")
    print(f"  Total entities:       {sum(type_counts.values()):,}")
    print()

    # Per-type breakdown
    print(f"  {'Type':15s} {'Count':>8s} {'Unique':>8s} {'Avg/art':>8s}")
    print(f"  {'-' * 41}")
    for et in ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]:
        c = type_counts.get(et, 0)
        u = len(entity_freq.get(et, {}))
        avg = c / total
        print(f"  {et:15s} {c:>8,} {u:>8,} {avg:>8.2f}")

    # --- Top entities per type ---
    for et in ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]:
        freq = entity_freq.get(et)
        if not freq:
            continue
        print(f"\n  Top {top_n} {et}:")
        for text, count in freq.most_common(top_n):
            print(f"    {count:>5}  {text}")

    # --- Entity density by volume ---
    vol_stats = {}  # vol -> {articles, entities}
    for r in records:
        vol = r.get("volume")
        if vol is None:
            continue
        vs = vol_stats.setdefault(vol, {"articles": 0, "entities": 0})
        vs["articles"] += 1
        vs["entities"] += len(r.get("entities", []))

    if vol_stats:
        print(f"\n  Entity density by volume:")
        print(f"  {'Vol':>4s} {'Articles':>9s} {'Entities':>9s} {'Avg':>6s}")
        print(f"  {'-' * 30}")
        for vol in sorted(vol_stats):
            vs = vol_stats[vol]
            avg = vs["entities"] / vs["articles"] if vs["articles"] else 0
            print(f"  {vol:>4} {vs['articles']:>9,} {vs['entities']:>9,} {avg:>6.1f}")

    # --- Cross-reference with concept index ---
    if CONCEPT_INDEX.exists():
        print(f"\n{'=' * 60}")
        print("  Cross-reference with concept_index.json")
        print(f"{'=' * 60}")

        with open(CONCEPT_INDEX) as f:
            concepts = json.load(f)

        # Collect all unique entity texts, uppercased for matching
        all_ent_texts = set()
        ent_text_freq = Counter()
        for r in records:
            for ent in r.get("entities", []):
                upper = ent["text"].upper()
                all_ent_texts.add(upper)
                ent_text_freq[upper] += 1

        concept_keys = set(concepts.keys())
        # Remove non-concept keys (like metadata)
        concept_keys -= {"total_concepts", "core_concepts"}

        matches = all_ent_texts & concept_keys
        non_matches = all_ent_texts - concept_keys

        print(f"  Unique entity strings:     {len(all_ent_texts):,}")
        print(f"  Match EB headwords:        {len(matches):,} "
              f"({100 * len(matches) / len(all_ent_texts):.1f}%)")
        print(f"  New (not in headwords):    {len(non_matches):,}")

        # Sample matches
        print(f"\n  Sample matches (NER entity = existing EB article):")
        match_by_freq = sorted(matches, key=lambda x: -ent_text_freq[x])
        for m in match_by_freq[:12]:
            c = concepts.get(m, {})
            eds = list(c.get("editions", {}).keys()) if isinstance(c, dict) else []
            print(f"    {ent_text_freq[m]:>4}x  {m} "
                  f"({'in ' + ', '.join(eds[:4]) + ' ...' if len(eds) > 4 else 'in ' + ', '.join(eds)})")

        # Top non-matches (potential new concepts)
        print(f"\n  Top non-matches (potential new graph nodes):")
        non_match_by_freq = sorted(non_matches, key=lambda x: -ent_text_freq[x])
        for m in non_match_by_freq[:15]:
            # Identify which entity type(s) tagged this
            types_for = set()
            for r in records:
                for ent in r.get("entities", []):
                    if ent["text"].upper() == m:
                        types_for.add(ent["type"])
            print(f"    {ent_text_freq[m]:>4}x  {m}  [{', '.join(sorted(types_for))}]")

    # --- Sample articles for manual review ---
    print(f"\n{'=' * 60}")
    print(f"  Sample articles for manual review")
    print(f"{'=' * 60}")

    random.seed(42)
    by_count = sorted(records, key=lambda r: len(r.get("entities", [])), reverse=True)

    samples = []
    # Entity-rich
    samples.extend(by_count[:n_samples // 3])
    # Mid-range
    mid = len(by_count) // 2
    mid_pool = by_count[max(0, mid - 50):mid + 50]
    samples.extend(random.sample(mid_pool, min(n_samples // 3, len(mid_pool))))
    # Low
    low_pool = [r for r in by_count if 1 <= len(r.get("entities", [])) <= 3]
    remaining_n = n_samples - len(samples)
    if low_pool and remaining_n > 0:
        samples.extend(random.sample(low_pool, min(remaining_n, len(low_pool))))

    for r in samples[:n_samples]:
        ents = r.get("entities", [])
        print(f"\n  [{r['article_id']}] {r['title']}  (vol {r.get('volume', '?')})")
        print(f"  {len(ents)} entities:")
        by_type = {}
        for e in ents:
            by_type.setdefault(e["type"], []).append(e["text"])
        for et in ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]:
            texts = by_type.get(et, [])
            if texts:
                display = ", ".join(texts[:8])
                if len(texts) > 8:
                    display += f" ... (+{len(texts) - 8})"
                print(f"    {et}: {display}")


def main():
    parser = argparse.ArgumentParser(description="Analyze NER extraction results")
    parser.add_argument("input", type=Path, help="Entity JSONL file")
    parser.add_argument("--top", type=int, default=20, help="Top N entities per type")
    parser.add_argument("--samples", type=int, default=10, help="Sample articles for review")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input}")

    records = load_entities(args.input)
    analyze(records, top_n=args.top, n_samples=args.samples)


if __name__ == "__main__":
    main()
