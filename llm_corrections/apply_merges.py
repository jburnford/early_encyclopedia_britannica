#!/usr/bin/env python3
"""
Apply merge decisions to JSONL files.

Reads decisions from corrections/decisions.json and applies:
- MERGE: Appends flagged article text to parent with page marker
- DELETE: Removes article from corpus
- KEEP: No change (article stays as-is)

Usage:
    python3 apply_merges.py --preview          # Show what would happen
    python3 apply_merges.py --apply            # Apply changes
    python3 apply_merges.py --apply --edition 1771  # Apply to specific edition
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CORRECTIONS_DIR = SCRIPT_DIR / "corrections"
OUTPUT_DIR = PROJECT_ROOT / "output_v2"
BACKUP_DIR = PROJECT_ROOT / "output_v2" / "backup_before_merges"

ALL_EDITIONS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]


def load_decisions() -> list[dict]:
    """Load all recorded decisions."""
    decisions_file = CORRECTIONS_DIR / "decisions.json"
    if decisions_file.exists():
        with open(decisions_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def load_articles(filepath: Path) -> list[dict]:
    """Load all articles from a JSONL file."""
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return articles


def save_articles(articles: list[dict], filepath: Path):
    """Save articles to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


def apply_merge(parent_article: dict, child_article: dict) -> dict:
    """Merge child article into parent with page marker."""
    merged_text = parent_article.get('text', '')

    # Add page marker for traceability
    child_page = child_article.get('start_page', '?')
    merged_text += f"\n\n[Continued from p.{child_page}]\n\n"
    merged_text += child_article.get('text', '')

    parent_article['text'] = merged_text
    parent_article['end_page'] = child_article.get('end_page', parent_article.get('end_page'))
    parent_article['word_count'] = len(merged_text.split())

    # Track merge history
    if 'merged_from' not in parent_article:
        parent_article['merged_from'] = []
    parent_article['merged_from'].append({
        'article_id': child_article.get('article_id'),
        'headword': child_article.get('headword'),
        'page': child_page
    })

    # Clear the needs_review flag if it was set
    if 'needs_review' in parent_article:
        del parent_article['needs_review']
    if 'issues' in parent_article:
        del parent_article['issues']

    return parent_article


def find_parent_article(
    parent_headword: str,
    parent_id: Optional[str],
    articles: list[dict]
) -> Optional[dict]:
    """Find parent article by headword or ID."""
    # First try by ID if available
    if parent_id:
        for article in articles:
            if article.get('article_id') == parent_id:
                return article

    # Fall back to headword match
    for article in articles:
        if article.get('headword') == parent_headword:
            return article

    return None


def process_edition(
    edition_year: int,
    decisions: list[dict],
    preview: bool = True
) -> dict:
    """Process all decisions for an edition."""
    input_file = OUTPUT_DIR / f"articles_{edition_year}.jsonl"

    if not input_file.exists():
        return {'error': f"File not found: {input_file}"}

    # Filter decisions for this edition
    edition_decisions = [d for d in decisions if d.get('edition_year') == edition_year]

    if not edition_decisions:
        return {'skipped': True, 'reason': 'No decisions for this edition'}

    # Load articles
    articles = load_articles(input_file)
    articles_by_id = {a.get('article_id'): a for a in articles}
    articles_by_headword = {a.get('headword'): a for a in articles}

    # Track changes
    stats = {
        'edition': edition_year,
        'original_count': len(articles),
        'merges': 0,
        'deletes': 0,
        'keeps': 0,
        'errors': []
    }

    # Group merge decisions by parent
    merges_by_parent = defaultdict(list)
    to_delete_ids = set()
    to_delete_headwords = set()

    for decision in edition_decisions:
        decision_type = decision.get('decision', '').upper()
        article_id = decision.get('article_id')
        headword = decision.get('headword', '')

        if decision_type == 'MERGE':
            merge_into = decision.get('merge_into')
            if merge_into:
                # Store both ID and headword for child lookup
                merges_by_parent[merge_into].append({
                    'article_id': article_id,
                    'headword': headword
                })
                to_delete_ids.add(article_id)
                to_delete_headwords.add(headword)
            else:
                stats['errors'].append(f"MERGE without target: {article_id}")

        elif decision_type == 'DELETE':
            to_delete_ids.add(article_id)
            to_delete_headwords.add(headword)
            stats['deletes'] += 1

        elif decision_type == 'KEEP':
            stats['keeps'] += 1

    # Apply merges
    for parent_headword, child_info_list in merges_by_parent.items():
        # Find parent article
        parent = None
        for article in articles:
            if article.get('headword') == parent_headword:
                parent = article
                break

        if not parent:
            stats['errors'].append(f"Parent not found: {parent_headword}")
            # Don't delete children if parent not found
            for child_info in child_info_list:
                to_delete_ids.discard(child_info['article_id'])
                to_delete_headwords.discard(child_info['headword'])
            continue

        # Merge all children into parent (in page order)
        children = []
        for child_info in child_info_list:
            # Try by article_id first, then by headword
            child = articles_by_id.get(child_info['article_id'])
            if not child:
                child = articles_by_headword.get(child_info['headword'])

            if child:
                children.append(child)
            else:
                stats['errors'].append(f"Child not found: {child_info['headword']}")
                to_delete_ids.discard(child_info['article_id'])
                to_delete_headwords.discard(child_info['headword'])

        # Sort children by page
        children.sort(key=lambda x: x.get('start_page', 0))

        for child in children:
            parent = apply_merge(parent, child)
            stats['merges'] += 1

        if preview:
            print(f"  MERGE: {[c.get('headword') for c in children]} -> {parent_headword}")

    # Filter out deleted articles (check both ID and headword)
    final_articles = [
        a for a in articles
        if a.get('article_id') not in to_delete_ids
        and a.get('headword') not in to_delete_headwords
    ]

    stats['final_count'] = len(final_articles)
    stats['removed'] = len(articles) - len(final_articles)

    if not preview:
        # Backup original
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_file = BACKUP_DIR / f"articles_{edition_year}.jsonl"
        shutil.copy2(input_file, backup_file)

        # Save modified articles
        save_articles(final_articles, input_file)

    return stats


def preview_changes(editions: list[int] = None):
    """Preview what changes would be made."""
    decisions = load_decisions()

    if not decisions:
        print("No decisions recorded yet.")
        return

    print("\n" + "=" * 60)
    print("PREVIEW: Changes that would be applied")
    print("=" * 60)

    editions_to_check = editions or ALL_EDITIONS

    total_merges = 0
    total_deletes = 0

    for edition in editions_to_check:
        print(f"\n{edition} Edition:")
        stats = process_edition(edition, decisions, preview=True)

        if stats.get('skipped'):
            print(f"  Skipped: {stats.get('reason')}")
        elif stats.get('error'):
            print(f"  Error: {stats.get('error')}")
        else:
            print(f"  Original: {stats['original_count']} articles")
            print(f"  Merges: {stats['merges']}")
            print(f"  Deletes: {stats['deletes']}")
            print(f"  Keeps: {stats['keeps']}")
            print(f"  Final: {stats['final_count']} articles")
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")
                for err in stats['errors'][:5]:
                    print(f"    - {err}")

            total_merges += stats['merges']
            total_deletes += stats['deletes']

    print(f"\n{'=' * 60}")
    print(f"Total merges: {total_merges}")
    print(f"Total deletes: {total_deletes}")
    print(f"Net article reduction: {total_merges + total_deletes}")


def apply_changes(editions: list[int] = None):
    """Apply all changes to JSONL files."""
    decisions = load_decisions()

    if not decisions:
        print("No decisions recorded yet.")
        return

    print("\n" + "=" * 60)
    print("APPLYING CHANGES")
    print("=" * 60)

    editions_to_apply = editions or ALL_EDITIONS

    for edition in editions_to_apply:
        print(f"\nProcessing {edition} edition...")
        stats = process_edition(edition, decisions, preview=False)

        if stats.get('skipped'):
            print(f"  Skipped: {stats.get('reason')}")
        elif stats.get('error'):
            print(f"  Error: {stats.get('error')}")
        else:
            print(f"  Applied {stats['merges']} merges, {stats['deletes']} deletes")
            print(f"  {stats['original_count']} -> {stats['final_count']} articles")
            if stats['errors']:
                print(f"  Errors: {len(stats['errors'])}")

    print(f"\n{'=' * 60}")
    print(f"Backups saved to: {BACKUP_DIR}")
    print(f"To regenerate site: python3 generate_site_optimized.py")


def main():
    parser = argparse.ArgumentParser(
        description="Apply merge decisions to JSONL files"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview changes without applying"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to JSONL files"
    )
    parser.add_argument(
        "--edition",
        type=int,
        choices=ALL_EDITIONS,
        help="Process specific edition only"
    )

    args = parser.parse_args()

    editions = [args.edition] if args.edition else None

    if args.preview:
        preview_changes(editions)
    elif args.apply:
        apply_changes(editions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
