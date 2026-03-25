#!/usr/bin/env python3
"""
Helper to prepare and apply Sonnet agent judgments for person disambiguation.

Usage:
  # Show batch N (0-indexed) of candidates for Sonnet judging:
  python judge_person_batch.py --show-batch 0 --batch-size 50

  # Apply judgments from a JSONL file:
  python judge_person_batch.py --apply judgments.jsonl

  # Show stats on what's been judged vs remaining:
  python judge_person_batch.py --status
"""

import json
import argparse
from pathlib import Path

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"
CANDIDATES_PATH = NER_DIR / "person_candidates.jsonl"
MATCHES_PATH = NER_DIR / "person_matches.jsonl"


def load_candidates():
    candidates = []
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # skip malformed lines (e.g. partial writes)
    return candidates


def load_matched_ids():
    matched = set()
    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    matched.add(json.loads(line)["cluster_id"])
    return matched


def get_unjudged(candidates, matched_ids):
    """Return candidates that have search results and aren't already matched."""
    return [c for c in candidates
            if c.get("candidates") and c["cluster_id"] not in matched_ids]


def status():
    candidates = load_candidates()
    matched_ids = load_matched_ids()
    unjudged = get_unjudged(candidates, matched_ids)

    total_cands = len(candidates)
    with_results = sum(1 for c in candidates if c.get("candidates"))
    no_results = sum(1 for c in candidates if not c.get("candidates"))

    print(f"Candidates file: {total_cands:,} entries")
    print(f"  With Wikidata results: {with_results:,}")
    print(f"  No results (empty): {no_results:,}")
    print(f"Already matched/judged: {len(matched_ids):,}")
    print(f"Remaining to judge: {len(unjudged):,}")

    if unjudged:
        # Show mention distribution
        tiers = [(100, "100+"), (50, "50-99"), (20, "20-49"),
                 (10, "10-19"), (5, "5-9")]
        for threshold, label in tiers:
            n = sum(1 for c in unjudged if c["total_mentions"] >= threshold)
            print(f"  >= {threshold} mentions: {n:,}")

        batch_size = 50
        n_batches = (len(unjudged) + batch_size - 1) // batch_size
        print(f"\nAt batch_size={batch_size}: {n_batches} batches needed")


def show_batch(batch_num, batch_size=50):
    candidates = load_candidates()
    matched_ids = load_matched_ids()
    unjudged = get_unjudged(candidates, matched_ids)

    # Sort by mentions descending
    unjudged.sort(key=lambda c: c["total_mentions"], reverse=True)

    start = batch_num * batch_size
    end = start + batch_size
    batch = unjudged[start:end]

    if not batch:
        print(f"Batch {batch_num} is empty (only {len(unjudged)} unjudged candidates)")
        return

    print(f"=== BATCH {batch_num} ({len(batch)} clusters, "
          f"mentions {batch[0]['total_mentions']}-{batch[-1]['total_mentions']}) ===\n")

    for c in batch:
        cand_lines = []
        for i, cand in enumerate(c["candidates"][:5]):
            cand_lines.append(
                f"    {i+1}. {cand['qid']}: {cand['label']} — {cand['description']}"
            )

        print(f"CLUSTER: \"{c['search_label']}\" "
              f"(cluster_id=\"{c['cluster_id']}\", {c['total_mentions']} mentions, "
              f"{c.get('edition_count', '?')} editions, "
              f"concept_headword={c.get('is_concept_headword', False)})")
        print(f"  Sample articles: {c.get('sample_articles', [])}")
        print(f"  Candidates:")
        print("\n".join(cand_lines))
        print()


def apply_judgments(judgments_file):
    """Apply judgments from a JSONL file to matches."""
    judgments = []
    with open(judgments_file) as f:
        for line in f:
            line = line.strip()
            if line:
                judgments.append(json.loads(line))

    matched_ids = load_matched_ids()
    new_count = 0

    with open(MATCHES_PATH, "a") as f:
        for j in judgments:
            cid = j["cluster_id"]
            if cid in matched_ids:
                continue

            match_qid = j.get("match")
            if match_qid and match_qid not in ("none", "false_positive"):
                match_rec = {
                    "cluster_id": cid,
                    "wikidata_qid": match_qid,
                    "wikidata_label": j.get("label", ""),
                    "wikidata_description": j.get("description", ""),
                    "match_type": "wikidata",
                    "confidence": j.get("confidence", 0.5),
                    "reason": j.get("reason", ""),
                }
            elif match_qid == "false_positive":
                match_rec = {
                    "cluster_id": cid,
                    "match_type": "false_positive",
                    "reason": j.get("reason", ""),
                }
            else:
                match_rec = {
                    "cluster_id": cid,
                    "match_type": "none",
                    "reason": j.get("reason", ""),
                }

            json.dump(match_rec, f, ensure_ascii=False)
            f.write("\n")
            new_count += 1

    print(f"Applied {new_count} new judgments to {MATCHES_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Person candidate batch helper")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--show-batch", type=int, default=None,
                        help="Show batch N for Sonnet judging")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size (default: 50)")
    parser.add_argument("--apply", type=str, default=None,
                        help="Apply judgments from JSONL file")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.show_batch is not None:
        show_batch(args.show_batch, args.batch_size)
    elif args.apply:
        apply_judgments(args.apply)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
