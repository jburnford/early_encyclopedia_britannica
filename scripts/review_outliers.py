#!/usr/bin/env python3
"""
LLM-assisted review of alphabetic outliers.

For each outlier, the LLM determines:
- MERGE: outlier should merge into another article (specify target)
- RENAME: headword is OCR error (specify corrected headword)
- KEEP: valid standalone article (rare - requires justification)
- OCR_REVIEW: complex case that needs raw OCR review

Usage:
    python3 review_outliers.py --edition 1815 --batch 1
    python3 review_outliers.py --edition 1815 --all
    python3 review_outliers.py --status
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
BATCH_DIR = PROJECT_ROOT / "llm_corrections" / "outlier_batches"
DECISIONS_FILE = PROJECT_ROOT / "llm_corrections" / "outlier_decisions.json"

def load_decisions() -> dict:
    """Load existing decisions."""
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE) as f:
            return json.load(f)
    return {"decisions": [], "stats": {"merge": 0, "rename": 0, "keep": 0, "ocr_review": 0}}

def save_decisions(decisions: dict):
    """Save decisions."""
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_FILE, 'w') as f:
        json.dump(decisions, f, indent=2)

def format_outlier_for_review(item: dict) -> str:
    """Format an outlier for human/LLM review."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"OUTLIER: {item['headword']}")
    lines.append(f"Edition: {item['edition_year']} | Volume: {item['volume_num']} | Pages: {item['start_page']}-{item['end_page']}")
    lines.append(f"Words: {item['word_count']} | Letter: {item['first_letter']} (expected: {item['expected_range']})")
    lines.append(f"Reason: {item['reason']}")
    lines.append("-" * 70)
    lines.append("TEXT PREVIEW (first 800 chars):")
    lines.append(item['text_preview'][:800])

    if item.get('text_end') and len(item.get('text_preview', '')) > 800:
        lines.append("\n... [truncated] ...")
        lines.append("\nTEXT END (last 300 chars):")
        lines.append(item['text_end'][-300:])

    lines.append("-" * 70)
    lines.append("MERGE CANDIDATES:")

    for i, cand in enumerate(item.get('merge_candidates', []), 1):
        direction = cand.get('direction', 'unknown')
        lines.append(f"\n  [{i}] {cand['headword']} ({direction})")
        lines.append(f"      Pages: {cand['start_page']}-{cand['end_page']} | Words: {cand['word_count']}")
        if cand.get('text_end'):
            lines.append(f"      End of article: ...{cand['text_end'][-300:]}")

    lines.append("\n" + "-" * 70)
    lines.append("CONTEXT:")
    lines.append(f"  Previous articles: {[a['headword'] for a in item.get('prev_articles', [])]}")
    lines.append(f"  Next articles: {[a['headword'] for a in item.get('next_articles', [])]}")
    lines.append("=" * 70)

    return "\n".join(lines)

def get_decision_prompt() -> str:
    """Return the decision prompt."""
    return """
DECISION OPTIONS:
  MERGE <target_headword> - Merge this outlier into the specified article
  RENAME <correct_headword> - Fix OCR error in headword (article is valid)
  KEEP <reason> - Keep as standalone (RARE - only if genuinely valid article)
  OCR_REVIEW - Flag for manual review of raw OCR

Examples:
  MERGE AGRICULTURE
  MERGE ENTOMOLOGY
  RENAME BURNTISLAND
  RENAME AUNCEL-WEIGHT
  KEEP Valid biography of minor figure
  OCR_REVIEW

Your decision: """

def parse_decision(decision_str: str) -> tuple[str, str]:
    """Parse a decision string into (decision_type, detail)."""
    decision_str = decision_str.strip()

    if decision_str.upper().startswith("MERGE "):
        return ("merge", decision_str[6:].strip())
    elif decision_str.upper().startswith("RENAME "):
        return ("rename", decision_str[7:].strip())
    elif decision_str.upper().startswith("KEEP"):
        reason = decision_str[4:].strip() if len(decision_str) > 4 else "No reason given"
        return ("keep", reason)
    elif decision_str.upper().startswith("OCR_REVIEW") or decision_str.upper() == "OCR":
        return ("ocr_review", "")
    else:
        return (None, decision_str)

def review_batch(edition_year: int, batch_num: int, decisions_data: dict) -> int:
    """Review a single batch. Returns number of decisions made."""
    batch_file = BATCH_DIR / f"outlier_batch_{edition_year}_{batch_num:03d}.json"

    if not batch_file.exists():
        print(f"Batch file not found: {batch_file}")
        return 0

    with open(batch_file) as f:
        batch = json.load(f)

    # Track which articles already have decisions
    existing_ids = {d['article_id'] for d in decisions_data['decisions']}

    items_to_review = [
        item for item in batch['items']
        if item['article_id'] not in existing_ids
    ]

    if not items_to_review:
        print(f"All items in batch {edition_year}/{batch_num} already reviewed")
        return 0

    print(f"\nReviewing {len(items_to_review)} items from {edition_year} batch {batch_num}")
    print(f"(Batch {batch_num} of {batch['total_batches']})")
    print()

    decisions_made = 0

    for i, item in enumerate(items_to_review):
        print(f"\n[{i+1}/{len(items_to_review)}]")
        print(format_outlier_for_review(item))
        print(get_decision_prompt(), end="")

        try:
            decision_str = input()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping review (progress saved)")
            break

        if decision_str.lower() in ('q', 'quit', 'exit'):
            print("Stopping review (progress saved)")
            break

        if decision_str.lower() in ('s', 'skip'):
            print("Skipped")
            continue

        decision_type, detail = parse_decision(decision_str)

        if decision_type is None:
            print(f"Invalid decision: {decision_str}")
            print("Valid options: MERGE <target>, RENAME <headword>, KEEP <reason>, OCR_REVIEW")
            continue

        decision = {
            "article_id": item['article_id'],
            "headword": item['headword'],
            "edition_year": item['edition_year'],
            "volume_num": item['volume_num'],
            "start_page": item['start_page'],
            "decision": decision_type,
            "detail": detail,
            "timestamp": datetime.now().isoformat()
        }

        decisions_data['decisions'].append(decision)
        decisions_data['stats'][decision_type] = decisions_data['stats'].get(decision_type, 0) + 1
        decisions_made += 1

        # Save after each decision
        save_decisions(decisions_data)
        print(f"  -> {decision_type.upper()}: {detail}")

    return decisions_made

def show_status(decisions_data: dict):
    """Show current review status."""
    print("\nOutlier Review Status")
    print("=" * 50)

    # Load summary
    summary_file = BATCH_DIR / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        total_outliers = summary['total_outliers']
        total_decisions = len(decisions_data['decisions'])

        print(f"Total outliers: {total_outliers}")
        print(f"Decisions made: {total_decisions}")
        print(f"Remaining: {total_outliers - total_decisions}")
        print()
        print("Decision breakdown:")
        for dtype, count in decisions_data['stats'].items():
            print(f"  {dtype}: {count}")

        print()
        print("By edition:")
        for year, info in summary['editions'].items():
            year_decisions = sum(1 for d in decisions_data['decisions'] if d['edition_year'] == int(year))
            print(f"  {year}: {year_decisions}/{info['outliers']} reviewed")

def main():
    parser = argparse.ArgumentParser(description="Review alphabetic outliers")
    parser.add_argument("--edition", type=int, help="Edition year")
    parser.add_argument("--batch", type=int, help="Batch number")
    parser.add_argument("--all", action="store_true", help="Review all batches for edition")
    parser.add_argument("--status", action="store_true", help="Show review status")

    args = parser.parse_args()

    decisions_data = load_decisions()

    if args.status:
        show_status(decisions_data)
        return

    if not args.edition:
        print("Error: --edition required")
        parser.print_help()
        sys.exit(1)

    if args.all:
        # Review all batches for this edition
        batch_num = 1
        while True:
            batch_file = BATCH_DIR / f"outlier_batch_{args.edition}_{batch_num:03d}.json"
            if not batch_file.exists():
                break
            review_batch(args.edition, batch_num, decisions_data)
            batch_num += 1
    elif args.batch:
        review_batch(args.edition, args.batch, decisions_data)
    else:
        # Find first batch with unreviewed items
        existing_ids = {d['article_id'] for d in decisions_data['decisions']}

        batch_num = 1
        while True:
            batch_file = BATCH_DIR / f"outlier_batch_{args.edition}_{batch_num:03d}.json"
            if not batch_file.exists():
                print(f"No more batches for {args.edition}")
                break

            with open(batch_file) as f:
                batch = json.load(f)

            unreviewed = [i for i in batch['items'] if i['article_id'] not in existing_ids]
            if unreviewed:
                review_batch(args.edition, batch_num, decisions_data)
                break

            batch_num += 1

    print("\nFinal status:")
    show_status(decisions_data)

if __name__ == '__main__':
    main()
