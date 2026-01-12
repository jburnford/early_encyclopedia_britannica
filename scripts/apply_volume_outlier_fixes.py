#!/usr/bin/env python3
"""
Apply volume outlier fixes to article JSONL files.

Supports automatic application of high-confidence decisions and
interactive review for medium/low confidence cases.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import shutil


def load_outliers(input_file: Path) -> list[dict]:
    """Load all outliers from detection results."""
    with open(input_file) as f:
        results = json.load(f)

    all_outliers = []
    for edition in results:
        for outlier in edition['outliers']:
            all_outliers.append(outlier)

    return all_outliers


def load_articles(jsonl_path: Path) -> tuple[list[dict], dict[str, int]]:
    """Load articles and build index."""
    articles = []
    id_to_idx = {}

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            art = json.loads(line)
            articles.append(art)
            id_to_idx[art.get('article_id', '')] = i

    return articles, id_to_idx


def backup_file(file_path: Path) -> Path:
    """Create timestamped backup."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    return backup_path


def apply_merge(articles: list[dict], id_to_idx: dict, outlier_id: str, target_headword: str) -> bool:
    """Merge outlier into target article."""
    if outlier_id not in id_to_idx:
        return False

    outlier_idx = id_to_idx[outlier_id]
    outlier = articles[outlier_idx]

    if outlier is None:
        return False  # Already deleted

    # Find target by headword (prefer previous articles, same volume)
    target_idx = None
    outlier_vol = outlier.get('volume_num', 0)

    for i, art in enumerate(articles):
        if art is None:
            continue
        if art.get('volume_num') == outlier_vol and art.get('headword', '').upper() == target_headword.upper():
            target_idx = i
            break

    if target_idx is None:
        # Try to find by page proximity
        outlier_page = outlier.get('start_page', 0)
        for i in range(outlier_idx - 1, -1, -1):
            if articles[i] is None:
                continue
            if articles[i].get('volume_num') == outlier_vol:
                target_idx = i
                break

    if target_idx is None:
        return False

    target = articles[target_idx]

    # Append outlier text to target
    target_text = target.get('text', '')
    outlier_text = outlier.get('text', '')
    target['text'] = target_text + '\n\n' + outlier_text

    # Update word count
    target['word_count'] = len(target['text'].split())

    # Mark outlier for deletion
    articles[outlier_idx] = None

    return True


def apply_delete(articles: list[dict], id_to_idx: dict, outlier_id: str) -> bool:
    """Delete outlier article."""
    if outlier_id not in id_to_idx:
        return False

    outlier_idx = id_to_idx[outlier_id]
    if articles[outlier_idx] is None:
        return False  # Already deleted
    articles[outlier_idx] = None
    return True


def apply_rename(articles: list[dict], id_to_idx: dict, outlier_id: str, new_headword: str) -> bool:
    """Rename outlier headword."""
    if outlier_id not in id_to_idx:
        return False

    outlier_idx = id_to_idx[outlier_id]
    articles[outlier_idx]['headword'] = new_headword
    return True


def save_articles(articles: list[dict], output_path: Path):
    """Save articles, removing deleted ones."""
    with open(output_path, 'w') as f:
        for art in articles:
            if art is not None:
                f.write(json.dumps(art) + '\n')


def apply_high_confidence_fixes(outliers: list[dict], dry_run: bool = True) -> dict:
    """Apply all high-confidence fixes (MERGE and DELETE)."""

    # Group by edition
    by_edition = defaultdict(list)
    for o in outliers:
        cls = o.get('classification', {})
        if cls.get('confidence') == 'high' and cls.get('decision') in ('MERGE', 'DELETE'):
            by_edition[o['edition_year']].append(o)

    stats = {
        'editions': {},
        'total_merged': 0,
        'total_deleted': 0,
        'total_failed': 0
    }

    for edition_year, edition_outliers in sorted(by_edition.items()):
        jsonl_path = Path(f'output_v2/articles_{edition_year}.jsonl')

        if not jsonl_path.exists():
            print(f"  Skipping {edition_year}: file not found")
            continue

        print(f"\n  Processing {edition_year}...")

        if not dry_run:
            backup_path = backup_file(jsonl_path)
            print(f"    Backed up to: {backup_path.name}")

        articles, id_to_idx = load_articles(jsonl_path)

        merged = 0
        deleted = 0
        failed = 0

        for o in edition_outliers:
            cls = o.get('classification', {})
            decision = cls.get('decision')
            article_id = o.get('article_id', '')

            if decision == 'MERGE':
                target = cls.get('merge_target', '')
                if not target:
                    # Use first previous article as target
                    prev_arts = o.get('prev_articles', [])
                    if prev_arts:
                        target = prev_arts[-1].get('headword', '')

                if target:
                    if dry_run:
                        merged += 1
                    else:
                        if apply_merge(articles, id_to_idx, article_id, target):
                            merged += 1
                        else:
                            failed += 1
                else:
                    failed += 1

            elif decision == 'DELETE':
                if dry_run:
                    deleted += 1
                else:
                    if apply_delete(articles, id_to_idx, article_id):
                        deleted += 1
                    else:
                        failed += 1

        if not dry_run:
            save_articles(articles, jsonl_path)

        stats['editions'][edition_year] = {
            'merged': merged,
            'deleted': deleted,
            'failed': failed
        }
        stats['total_merged'] += merged
        stats['total_deleted'] += deleted
        stats['total_failed'] += failed

        print(f"    Merged: {merged}, Deleted: {deleted}, Failed: {failed}")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Apply volume outlier fixes')
    parser.add_argument('--apply', action='store_true',
                       help='Actually apply fixes (default is dry run)')
    parser.add_argument('--confidence', choices=['high', 'medium', 'all'],
                       default='high', help='Confidence level to apply')
    args = parser.parse_args()

    input_file = Path('llm_corrections/outliers/volume_outliers.json')

    if not input_file.exists():
        print(f"Error: Run detect_volume_outliers.py first")
        return

    print("Loading outliers...")
    outliers = load_outliers(input_file)

    # Count by confidence
    high_conf = sum(1 for o in outliers
                    if o.get('classification', {}).get('confidence') == 'high'
                    and o.get('classification', {}).get('decision') in ('MERGE', 'DELETE'))

    print(f"Found {len(outliers)} outliers")
    print(f"High-confidence MERGE/DELETE: {high_conf}")

    if args.apply:
        print(f"\n{'='*60}")
        print("APPLYING HIGH-CONFIDENCE FIXES")
        print('='*60)
        stats = apply_high_confidence_fixes(outliers, dry_run=False)
    else:
        print(f"\n{'='*60}")
        print("DRY RUN - No changes will be made")
        print('='*60)
        stats = apply_high_confidence_fixes(outliers, dry_run=True)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total merged: {stats['total_merged']}")
    print(f"Total deleted: {stats['total_deleted']}")
    print(f"Total failed: {stats['total_failed']}")

    if not args.apply:
        print("\nTo apply these fixes, run:")
        print("  python3 scripts/apply_volume_outlier_fixes.py --apply")


if __name__ == '__main__':
    main()
