#!/usr/bin/env python3
"""
Apply outlier correction decisions to article JSONL files.

Reads decisions from llm_corrections/outlier_decisions.json and applies:
- MERGE: Appends outlier text to target article, deletes outlier
- RENAME: Updates headword and article_id
- SKIP: Removes article (ads, errata, non-articles)
- SPLIT: Separates bundled articles (complex)
- KEEP: No change (just marks as reviewed)

Usage:
    python3 apply_outlier_fixes.py --preview          # Show what would change
    python3 apply_outlier_fixes.py --preview --edition 1815  # Preview one edition
    python3 apply_outlier_fixes.py --apply            # Apply all changes
    python3 apply_outlier_fixes.py --apply --edition 1815    # Apply to one edition
"""

import argparse
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
DECISIONS_FILE = PROJECT_ROOT / "llm_corrections" / "outlier_decisions.json"
OUTPUT_DIR = PROJECT_ROOT / "output_v2"
BACKUP_DIR = PROJECT_ROOT / "llm_corrections" / "backups"

# Log of all operations
OPERATIONS_LOG = []


def normalize_headword(hw: str) -> str:
    """Normalize headword for matching: uppercase, remove apostrophes, normalize spaces."""
    if not hw:
        return ""
    # Uppercase
    result = hw.upper()
    # Remove all apostrophe variants (including fancy quotes)
    result = result.replace("'", "").replace("'", "").replace("'", "")
    result = result.replace('"', "").replace('"', "").replace('"', "")
    # Normalize whitespace
    result = " ".join(result.split())
    return result


def load_decisions() -> list[dict]:
    """Load correction decisions."""
    if not DECISIONS_FILE.exists():
        print(f"No decisions file found: {DECISIONS_FILE}")
        return []

    with open(DECISIONS_FILE) as f:
        data = json.load(f)
    return data.get('decisions', [])


def load_articles(edition_year: int, use_backup: bool = False) -> list[dict]:
    """Load articles from JSONL file.

    Args:
        edition_year: The edition year to load
        use_backup: If True, load from backup file if it exists
    """
    if use_backup:
        # Try backup file first
        backup_path = OUTPUT_DIR / f"articles_{edition_year}_backup.jsonl"
        if backup_path.exists():
            filepath = backup_path
        else:
            filepath = OUTPUT_DIR / f"articles_{edition_year}.jsonl"
    else:
        filepath = OUTPUT_DIR / f"articles_{edition_year}.jsonl"

    articles = []
    with open(filepath) as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def save_articles(articles: list[dict], edition_year: int, create_backup: bool = True):
    """Save articles to JSONL file."""
    filepath = OUTPUT_DIR / f"articles_{edition_year}.jsonl"

    if create_backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"articles_{edition_year}_{timestamp}.jsonl"
        if filepath.exists():
            shutil.copy(filepath, backup_path)
            print(f"  Backup: {backup_path.name}")

    with open(filepath, 'w') as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + '\n')


def normalize_id_part(s: str) -> str:
    """Normalize a part of an article ID for matching.

    Handles:
    - Apostrophe variants ('S, 'S)
    - Space-S in ID format (_S_, _S at end)
    - Section suffixes (_s2, _s10)
    """
    result = s.upper()
    # Remove trailing _sN section markers first
    result = re.sub(r'_S\d+$', '', result)
    # Normalize possessive patterns
    result = result.replace("'S", "S").replace("'S", "S")  # Apostrophe-S
    result = result.replace("_S_", "S_")  # Mid-word space-S
    # Handle trailing _S (possessive at end): ST_DAVID_S -> ST_DAVIDS
    if result.endswith("_S"):
        result = result[:-2] + "S"
    # Remove remaining apostrophes
    result = result.replace("'", "").replace("'", "").replace("'", "")
    return result


def find_article_by_id(articles: list[dict], article_id: str) -> tuple[dict, int]:
    """Find article by ID match. Returns (article, index) or (None, -1).

    Tries exact match first, then tries with different volume numbers
    and normalized headword parts.
    """
    # Try exact match
    for i, art in enumerate(articles):
        if art['article_id'] == article_id:
            return art, i

    # Try matching with different volume numbers and normalized headwords
    # ID format: {year}_{vol}_{headword} or {year}_{vol}_{headword}_s{n}
    parts = article_id.split('_')
    if len(parts) >= 3:
        year = parts[0]
        headword_parts = parts[2:]  # Everything after year_vol
        headword_suffix = '_'.join(headword_parts)
        norm_suffix = normalize_id_part(headword_suffix)

        for i, art in enumerate(articles):
            art_parts = art['article_id'].split('_')
            if len(art_parts) >= 3:
                art_year = art_parts[0]
                art_headword_suffix = '_'.join(art_parts[2:])
                art_norm_suffix = normalize_id_part(art_headword_suffix)
                # Match if same year and same normalized headword part
                if art_year == year and art_norm_suffix == norm_suffix:
                    return art, i

    return None, -1


def find_article_by_headword(articles: list[dict], headword: str, volume_num: int = None) -> tuple[dict, int]:
    """Find article by headword with fuzzy matching.

    Returns (article, index) or (None, -1).

    Matching strategies (in order):
    1. Exact normalized match
    2. Target contained in article headword
    3. Article headword contained in target
    """
    norm_target = normalize_headword(headword)

    # Strategy 1: Exact normalized match
    for i, art in enumerate(articles):
        if normalize_headword(art['headword']) == norm_target:
            if volume_num is None or art.get('volume_num') == volume_num:
                return art, i

    # Strategy 2: Target contained in article headword
    for i, art in enumerate(articles):
        art_norm = normalize_headword(art['headword'])
        if norm_target in art_norm:
            if volume_num is None or art.get('volume_num') == volume_num:
                return art, i

    # Strategy 3: Article headword contained in target
    for i, art in enumerate(articles):
        art_norm = normalize_headword(art['headword'])
        if art_norm in norm_target and len(art_norm) >= 4:  # Minimum match length
            if volume_num is None or art.get('volume_num') == volume_num:
                return art, i

    return None, -1


def find_source_article(articles: list[dict], decision: dict) -> tuple[dict, int]:
    """Find the source article for a decision.

    Tries article_id first, then falls back to headword matching.
    Returns (article, index) or (None, -1).
    """
    # Try ID match first
    art, idx = find_article_by_id(articles, decision['article_id'])
    if art:
        return art, idx

    # Fall back to headword match
    return find_article_by_headword(
        articles,
        decision['headword'],
        decision.get('volume_num')
    )


def apply_merge(articles: list[dict], decision: dict) -> tuple[bool, str]:
    """Apply a MERGE decision. Returns (success, message)."""
    target_headword = decision['detail']

    # Find source article
    source, source_idx = find_source_article(articles, decision)
    if source is None:
        return False, f"Source not found: {decision['headword']} ({decision['article_id']})"

    # Find target article (search in same volume first, then any volume)
    target, target_idx = find_article_by_headword(
        articles,
        target_headword,
        decision.get('volume_num')
    )

    if target is None:
        # Try without volume restriction
        target, target_idx = find_article_by_headword(articles, target_headword)

    if target is None:
        return False, f"Target not found: {target_headword} for {decision['headword']}"

    # Don't merge into self
    if source['article_id'] == target['article_id']:
        return False, f"Cannot merge article into itself: {decision['headword']}"

    # Merge: append source text to target
    source_pages = source.get('pages', [])
    if source_pages:
        start_page = source_pages[0].get('page_num', '?')
    else:
        start_page = source.get('start_page', '?')

    page_marker = f"\n\n[Merged from {source['headword']}, p.{start_page}]\n\n"
    target['text'] = target.get('text', '') + page_marker + source.get('text', '')

    # Update target metadata
    target['word_count'] = len(target['text'].split())

    # Update pages if source has pages
    if source_pages and 'pages' in target:
        # Add source pages to target
        target['pages'].extend(source_pages)
        target['pages'].sort(key=lambda p: p.get('page_num', 0))

    # Remove source article (adjust index if target was before source)
    if target_idx < source_idx:
        articles.pop(source_idx)
    else:
        articles.pop(source_idx)

    OPERATIONS_LOG.append({
        'type': 'merge',
        'source': source['headword'],
        'target': target['headword'],
        'edition': decision.get('edition_year')
    })

    return True, f"Merged {source['headword']} -> {target['headword']}"


def apply_rename(articles: list[dict], decision: dict) -> tuple[bool, str]:
    """Apply a RENAME decision. Returns (success, message)."""
    new_headword = decision['detail']

    # Find article
    source, idx = find_source_article(articles, decision)
    if source is None:
        return False, f"Article not found: {decision['headword']} ({decision['article_id']})"

    old_headword = source['headword']
    old_id = source['article_id']

    # Update headword
    source['headword'] = new_headword

    # Update article_id
    parts = old_id.split('_')
    if len(parts) >= 3:
        # Preserve year and volume, replace headword part
        new_id = f"{parts[0]}_{parts[1]}_{new_headword.replace(' ', '_').upper()}"
        source['article_id'] = new_id

    OPERATIONS_LOG.append({
        'type': 'rename',
        'old': old_headword,
        'new': new_headword,
        'edition': decision.get('edition_year')
    })

    return True, f"Renamed {old_headword} -> {new_headword}"


def apply_skip(articles: list[dict], decision: dict) -> tuple[bool, str]:
    """Apply a SKIP decision (delete article). Returns (success, message)."""
    # Find article
    source, idx = find_source_article(articles, decision)
    if source is None:
        return False, f"Article not found: {decision['headword']} ({decision['article_id']})"

    headword = source['headword']

    # Remove article
    articles.pop(idx)

    OPERATIONS_LOG.append({
        'type': 'skip',
        'deleted': headword,
        'reason': decision.get('reason', 'No reason provided'),
        'edition': decision.get('edition_year')
    })

    return True, f"Deleted {headword} ({decision.get('reason', 'skipped')})"


def apply_split(articles: list[dict], decision: dict) -> tuple[bool, str]:
    """Apply a SPLIT decision (separate bundled articles). Returns (success, message).

    This is complex and requires parsing the article text to find embedded headwords.
    For now, we log it for manual handling.
    """
    # Find article
    source, idx = find_source_article(articles, decision)
    if source is None:
        return False, f"Article not found: {decision['headword']} ({decision['article_id']})"

    # SPLIT is complex - log for manual handling
    OPERATIONS_LOG.append({
        'type': 'split_pending',
        'article': source['headword'],
        'word_count': source.get('word_count', 0),
        'edition': decision.get('edition_year'),
        'note': 'Requires manual splitting'
    })

    return True, f"SPLIT flagged for manual handling: {source['headword']} ({source.get('word_count', 0)} words)"


def preview_changes(decisions: list[dict], edition_filter: int = None):
    """Preview what changes would be made."""
    by_edition = defaultdict(list)
    for d in decisions:
        by_edition[d['edition_year']].append(d)

    editions = [edition_filter] if edition_filter else sorted(by_edition.keys())

    print("\nPREVIEW OF CHANGES")
    print("=" * 60)

    stats = defaultdict(int)

    for year in editions:
        edition_decisions = by_edition.get(year, [])
        if not edition_decisions:
            continue

        print(f"\n{year} Edition ({len(edition_decisions)} decisions):")

        for d in edition_decisions:
            decision_type = d['decision'].lower()
            stats[decision_type] += 1

            if decision_type == 'merge':
                print(f"  MERGE: {d['headword']} -> {d['detail']}")
            elif decision_type == 'rename':
                print(f"  RENAME: {d['headword']} -> {d['detail']}")
            elif decision_type == 'skip':
                print(f"  SKIP: {d['headword']} ({d.get('reason', '')})")
            elif decision_type == 'split':
                print(f"  SPLIT: {d['headword']} (manual handling required)")
            elif decision_type == 'keep':
                pass  # Don't show KEEP in preview
            else:
                print(f"  {decision_type.upper()}: {d['headword']}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  MERGE:  {stats['merge']:3d} (articles will be combined)")
    print(f"  RENAME: {stats['rename']:3d} (headwords will be fixed)")
    print(f"  SKIP:   {stats['skip']:3d} (articles will be deleted)")
    print(f"  SPLIT:  {stats['split']:3d} (require manual handling)")
    print(f"  KEEP:   {stats['keep']:3d} (no changes)")

    articles_removed = stats['merge'] + stats['skip']
    print(f"\nNet effect: ~{articles_removed} articles will be removed")


def apply_changes(decisions: list[dict], edition_filter: int = None, use_backup: bool = True):
    """Apply all changes."""
    global OPERATIONS_LOG
    OPERATIONS_LOG = []

    by_edition = defaultdict(list)
    for d in decisions:
        by_edition[d['edition_year']].append(d)

    editions = [edition_filter] if edition_filter else sorted(by_edition.keys())

    total_applied = 0
    total_errors = 0
    total_skipped = 0

    for year in editions:
        edition_decisions = by_edition.get(year, [])
        if not edition_decisions:
            continue

        # Filter to actionable decisions
        actionable = [d for d in edition_decisions
                      if d['decision'].lower() in ('merge', 'rename', 'skip', 'split')]

        if not actionable:
            print(f"{year}: No actionable decisions (only KEEP)")
            continue

        print(f"\n{year} Edition: Applying {len(actionable)} changes...")

        # Load articles (from backup if available)
        try:
            articles = load_articles(year, use_backup=use_backup)
        except FileNotFoundError:
            print(f"  ERROR: Article file not found for {year}")
            continue

        original_count = len(articles)

        applied = 0
        errors = []
        skipped = 0

        # Process in order: SKIP first, then MERGE, then RENAME
        # This ensures we don't try to merge articles that should be skipped
        for decision_type in ['skip', 'merge', 'rename', 'split']:
            type_decisions = [d for d in actionable if d['decision'].lower() == decision_type]

            for d in type_decisions:
                dtype = d['decision'].lower()

                if dtype == 'merge':
                    success, msg = apply_merge(articles, d)
                elif dtype == 'rename':
                    success, msg = apply_rename(articles, d)
                elif dtype == 'skip':
                    success, msg = apply_skip(articles, d)
                elif dtype == 'split':
                    success, msg = apply_split(articles, d)
                    skipped += 1  # SPLIT is flagged for manual
                else:
                    continue

                if success:
                    print(f"  OK: {msg}")
                    applied += 1
                else:
                    print(f"  ERROR: {msg}")
                    errors.append(msg)

        # Save articles
        save_articles(articles, year, create_backup=True)

        new_count = len(articles)
        print(f"  Articles: {original_count} -> {new_count} ({original_count - new_count:+d})")

        total_applied += applied
        total_errors += len(errors)
        total_skipped += skipped

    print("\n" + "=" * 60)
    print(f"COMPLETE: {total_applied} changes applied, {total_errors} errors, {total_skipped} manual pending")

    if total_errors > 0:
        print("\nERRORS (require attention):")
        for err in errors[:20]:  # Show first 20 errors
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Save operations log
    if OPERATIONS_LOG:
        log_path = BACKUP_DIR / f"operations_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(OPERATIONS_LOG, f, indent=2)
        print(f"\nOperations log saved: {log_path}")


def validate_decisions(decisions: list[dict], edition_filter: int = None):
    """Validate that all decisions can be applied."""
    by_edition = defaultdict(list)
    for d in decisions:
        by_edition[d['edition_year']].append(d)

    editions = [edition_filter] if edition_filter else sorted(by_edition.keys())

    print("\nVALIDATING DECISIONS")
    print("=" * 60)

    total_valid = 0
    total_invalid = 0

    for year in editions:
        edition_decisions = by_edition.get(year, [])
        if not edition_decisions:
            continue

        try:
            articles = load_articles(year, use_backup=True)
        except FileNotFoundError:
            print(f"{year}: Article file not found")
            continue

        valid = 0
        invalid = []

        for d in edition_decisions:
            dtype = d['decision'].lower()

            # Check source exists
            source, _ = find_source_article(articles, d)
            if source is None and dtype != 'keep':
                invalid.append(f"Source not found: {d['headword']}")
                continue

            # Check merge target exists
            if dtype == 'merge':
                target, _ = find_article_by_headword(articles, d['detail'])
                if target is None:
                    invalid.append(f"Merge target not found: {d['detail']} for {d['headword']}")
                    continue

            valid += 1

        print(f"\n{year}: {valid}/{len(edition_decisions)} valid")
        if invalid:
            for msg in invalid[:5]:
                print(f"  - {msg}")
            if len(invalid) > 5:
                print(f"  ... and {len(invalid) - 5} more")

        total_valid += valid
        total_invalid += len(invalid)

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_valid} valid, {total_invalid} invalid")

    return total_invalid == 0


def main():
    parser = argparse.ArgumentParser(description="Apply outlier correction decisions")
    parser.add_argument("--preview", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument("--validate", action="store_true", help="Validate all decisions can be applied")
    parser.add_argument("--edition", type=int, help="Only process specific edition")
    parser.add_argument("--from-current", action="store_true",
                        help="Load from current files instead of backups")

    args = parser.parse_args()

    if not args.preview and not args.apply and not args.validate:
        print("Error: Must specify --preview, --validate, or --apply")
        parser.print_help()
        return

    decisions = load_decisions()
    if not decisions:
        print("No decisions to apply.")
        return

    print(f"Loaded {len(decisions)} decisions from {DECISIONS_FILE}")

    if args.validate:
        validate_decisions(decisions, args.edition)
    elif args.preview:
        preview_changes(decisions, args.edition)
    elif args.apply:
        preview_changes(decisions, args.edition)
        print("\n" + "=" * 60)
        confirm = input("Apply these changes? (yes/no): ")
        if confirm.lower() in ('yes', 'y'):
            apply_changes(decisions, args.edition, use_backup=not args.from_current)
        else:
            print("Aborted.")


if __name__ == '__main__':
    main()
