#!/usr/bin/env python3
"""
Generate batches of flagged articles for LLM review.

For each flagged article, extracts:
- Article metadata and issues
- First 500 chars of flagged article text
- Parent candidate (article before it in page order)
- Last 500 chars of parent article text

Usage:
    python3 generate_batches.py --edition 1771       # Generate batches for specific edition
    python3 generate_batches.py --all                # Generate batches for all editions
    python3 generate_batches.py --stats              # Show stats without generating
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directories
STATE_DIR = Path(__file__).parent / "state"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# All edition years
ALL_EDITIONS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# Context length for boundary text
CONTEXT_CHARS = 500

# Batch size
BATCH_SIZE = 50


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


def get_flagged_articles(articles: list[dict]) -> list[dict]:
    """Get all articles flagged with needs_review."""
    return [a for a in articles if a.get('needs_review', False)]


def get_first_letter(headword: str) -> str:
    """Extract first alphabetic character from headword."""
    for char in headword:
        if char.isalpha():
            return char.upper()
    return ''


def find_page_adjacent_parent(
    flagged_article: dict,
    volume_articles: list[dict]
) -> Optional[dict]:
    """
    Find the page-adjacent parent (previous article by page order).

    Good for: Section headings in the middle of long treatises.
    """
    flagged_page = flagged_article.get('start_page', 0)
    flagged_letter = get_first_letter(flagged_article.get('headword', ''))

    if not flagged_letter or not flagged_page:
        return None

    # Sort volume articles by start_page
    sorted_articles = sorted(volume_articles, key=lambda x: x.get('start_page', 0))

    # Find flagged article's position
    flagged_idx = None
    for i, a in enumerate(sorted_articles):
        if a.get('article_id') == flagged_article.get('article_id'):
            flagged_idx = i
            break

    if flagged_idx is None or flagged_idx == 0:
        return None

    # Look at articles immediately before the flagged one
    for i in range(flagged_idx - 1, max(0, flagged_idx - 10), -1):
        candidate = sorted_articles[i]
        candidate_page = candidate.get('end_page', candidate.get('start_page', 0))

        if candidate_page and candidate_page >= flagged_page - 2:
            # Good candidate - on same or nearby page
            if not candidate.get('needs_review', False):
                return candidate
            return candidate

    return None


def find_semantic_parent(
    flagged_article: dict,
    volume_articles: list[dict]
) -> Optional[dict]:
    """
    Find a semantic parent based on headword containment.

    For entries like "BLACK CHALK", finds "CHALK" among articles on the
    same or nearby pages. Sub-entries typically appear on the same page
    as their parent entry.
    """
    flagged_headword = flagged_article.get('headword', '').upper()
    flagged_page = flagged_article.get('start_page', 0)
    flagged_id = flagged_article.get('article_id')

    if not flagged_headword or not flagged_page:
        return None

    # Split headword into words to find potential parent headwords
    # e.g., "BLACK CHALK" -> check for "CHALK"
    words = flagged_headword.split()
    if len(words) < 2:
        return None  # Single-word headwords can't have semantic parents

    # Potential parent headwords: last word, last two words, etc.
    potential_parents = []
    for i in range(1, len(words)):
        potential_parents.append(' '.join(words[i:]))

    # Search articles within ±1 page for semantic matches
    # This handles sub-entries that appear on same page as parent
    PAGE_RANGE = 1
    best_match = None
    best_score = 0

    for article in volume_articles:
        if article.get('article_id') == flagged_id:
            continue

        article_page = article.get('start_page', 0)
        article_headword = article.get('headword', '').upper()

        # Must be on same or adjacent page
        page_distance = abs(article_page - flagged_page)
        if page_distance > PAGE_RANGE:
            continue

        # Check if this article's headword is a potential parent
        for i, parent_hw in enumerate(potential_parents):
            if article_headword == parent_hw:
                # Score based on: same page (bonus), shorter suffix (bonus)
                score = 10
                if page_distance == 0:
                    score += 5  # Same page
                score -= i  # Prefer shorter suffixes (first in list)

                if score > best_score:
                    best_score = score
                    best_match = article

    return best_match


def find_parent_candidates(
    flagged_article: dict,
    all_articles: list[dict],
    volume_articles: list[dict]
) -> dict:
    """
    Find all potential parent candidates using multiple strategies.

    Returns dict with:
    - page_adjacent: Previous article by page order
    - semantic: Article with headword that flagged headword contains
    """
    return {
        'page_adjacent': find_page_adjacent_parent(flagged_article, volume_articles),
        'semantic': find_semantic_parent(flagged_article, volume_articles)
    }


def extract_boundary_context(article: dict, position: str = 'start') -> str:
    """Extract boundary text from article."""
    text = article.get('text', '')

    if position == 'start':
        return text[:CONTEXT_CHARS].strip()
    else:  # 'end'
        return text[-CONTEXT_CHARS:].strip()


def get_surrounding_letter(
    flagged_article: dict,
    volume_articles: list[dict]
) -> str:
    """Get the predominant letter of surrounding articles."""
    flagged_page = flagged_article.get('start_page', 0)
    flagged_id = flagged_article.get('article_id')

    sorted_articles = sorted(volume_articles, key=lambda x: x.get('start_page', 0))

    # Find flagged article's position
    flagged_idx = None
    for i, a in enumerate(sorted_articles):
        if a.get('article_id') == flagged_id:
            flagged_idx = i
            break

    if flagged_idx is None:
        return '?'

    # Get letters of surrounding articles
    surrounding_letters = []
    for i in range(max(0, flagged_idx - 3), min(len(sorted_articles), flagged_idx + 4)):
        if i != flagged_idx:
            letter = get_first_letter(sorted_articles[i].get('headword', ''))
            if letter:
                surrounding_letters.append(letter)

    if not surrounding_letters:
        return '?'

    # Return most common letter
    from collections import Counter
    return Counter(surrounding_letters).most_common(1)[0][0]


def get_primary_issue(article: dict) -> str:
    """Get the primary issue type from article's issues."""
    issues = article.get('issues', [])
    if not issues:
        return 'unknown'

    # Priority order for issue types
    priority = ['alphabetical_break', 'sentence_fragment', 'out_of_range', 'too_long', 'ocr_error']

    issue_types = [i.get('issue_type', '') for i in issues]
    for p in priority:
        if p in issue_types:
            return p

    return issue_types[0] if issue_types else 'unknown'


def make_parent_info(parent: Optional[dict]) -> Optional[dict]:
    """Create parent info dict from article."""
    if not parent:
        return None
    return {
        "article_id": parent.get('article_id'),
        "headword": parent.get('headword'),
        "start_page": parent.get('start_page'),
        "end_page": parent.get('end_page'),
        "word_count": parent.get('word_count', len(parent.get('text', '').split())),
        "text_end": extract_boundary_context(parent, 'end')
    }


def generate_batch_item(
    flagged_article: dict,
    all_articles: list[dict],
    volume_articles: list[dict]
) -> dict:
    """Generate a single batch item for LLM review."""
    parents = find_parent_candidates(flagged_article, all_articles, volume_articles)
    surrounding_letter = get_surrounding_letter(flagged_article, volume_articles)
    primary_issue = get_primary_issue(flagged_article)

    item = {
        "flagged": {
            "article_id": flagged_article.get('article_id'),
            "headword": flagged_article.get('headword'),
            "start_page": flagged_article.get('start_page'),
            "end_page": flagged_article.get('end_page'),
            "word_count": flagged_article.get('word_count', len(flagged_article.get('text', '').split())),
            "volume_num": flagged_article.get('volume_num'),
            "primary_issue": primary_issue,
            "issues": flagged_article.get('issues', []),
            "text_preview": extract_boundary_context(flagged_article, 'start'),
            "surrounding_letter": surrounding_letter
        },
        "parent_candidates": {
            "page_adjacent": make_parent_info(parents['page_adjacent']),
            "semantic": make_parent_info(parents['semantic'])
        }
    }

    return item


def generate_batches_for_edition(edition_year: int) -> list[dict]:
    """Generate all batches for a single edition."""
    input_file = PROJECT_ROOT / f"output_v2/articles_{edition_year}.jsonl"

    if not input_file.exists():
        print(f"  Warning: {input_file} not found")
        return []

    # Load all articles
    all_articles = load_articles(input_file)
    flagged_articles = get_flagged_articles(all_articles)

    if not flagged_articles:
        print(f"  No flagged articles in {edition_year}")
        return []

    print(f"  Found {len(flagged_articles)} flagged articles")

    # Group articles by volume for faster lookup
    by_volume = defaultdict(list)
    for article in all_articles:
        vol = article.get('volume_num', 0)
        by_volume[vol].append(article)

    # Generate batch items
    batches = []
    current_batch = []

    for flagged in flagged_articles:
        vol = flagged.get('volume_num', 0)
        volume_articles = by_volume.get(vol, [])

        item = generate_batch_item(flagged, all_articles, volume_articles)
        current_batch.append(item)

        if len(current_batch) >= BATCH_SIZE:
            batches.append(current_batch)
            current_batch = []

    # Add remaining items
    if current_batch:
        batches.append(current_batch)

    return batches


def save_batches(batches: list[list[dict]], edition_year: int):
    """Save batches to JSON files."""
    for i, batch in enumerate(batches, 1):
        batch_file = STATE_DIR / f"batch_{edition_year}_{i:03d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump({
                "edition_year": edition_year,
                "batch_num": i,
                "total_batches": len(batches),
                "articles": batch
            }, f, indent=2, ensure_ascii=False)
        print(f"    Saved {batch_file.name} ({len(batch)} articles)")


def generate_progress_file(edition_stats: dict):
    """Generate initial progress file."""
    progress_file = STATE_DIR / "progress.json"

    total_flagged = sum(stats['flagged'] for stats in edition_stats.values())
    total_batches = sum(stats['batches'] for stats in edition_stats.values())

    progress = {
        "total_flagged": total_flagged,
        "total_batches": total_batches,
        "processed": 0,
        "current_edition": None,
        "current_batch": 0,
        "by_edition": edition_stats,
        "decisions": {
            "merge": 0,
            "keep_separate": 0,
            "delete": 0
        }
    }

    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

    print(f"\nProgress file: {progress_file}")


def show_stats():
    """Show statistics about flagged articles."""
    print("\nFlagged Article Statistics\n" + "=" * 50)

    total_flagged = 0
    by_edition = {}

    for edition_year in ALL_EDITIONS:
        input_file = PROJECT_ROOT / f"output_v2/articles_{edition_year}.jsonl"

        if not input_file.exists():
            continue

        articles = load_articles(input_file)
        flagged = get_flagged_articles(articles)

        # Count by issue type
        issue_counts = defaultdict(int)
        for article in flagged:
            for issue in article.get('issues', []):
                issue_counts[issue.get('issue_type', 'unknown')] += 1

        batches = (len(flagged) + BATCH_SIZE - 1) // BATCH_SIZE

        by_edition[edition_year] = {
            'flagged': len(flagged),
            'batches': batches,
            'issues': dict(issue_counts)
        }

        total_flagged += len(flagged)

        print(f"\n{edition_year} Edition:")
        print(f"  Flagged: {len(flagged)}")
        print(f"  Batches: {batches}")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"    {issue_type}: {count}")

    print(f"\n{'=' * 50}")
    print(f"Total flagged: {total_flagged}")
    print(f"Total batches: {sum(e['batches'] for e in by_edition.values())}")

    return by_edition


def main():
    parser = argparse.ArgumentParser(
        description="Generate batches of flagged articles for LLM review"
    )
    parser.add_argument(
        "--edition",
        type=int,
        choices=ALL_EDITIONS,
        help="Generate batches for specific edition"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate batches for all editions"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics without generating batches"
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not args.edition and not args.all:
        print("Error: Must specify --edition YEAR or --all")
        parser.print_help()
        sys.exit(1)

    # Ensure state directory exists
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    editions_to_process = ALL_EDITIONS if args.all else [args.edition]
    edition_stats = {}

    for edition_year in editions_to_process:
        print(f"\nProcessing {edition_year} edition...")
        batches = generate_batches_for_edition(edition_year)

        if batches:
            save_batches(batches, edition_year)
            edition_stats[edition_year] = {
                'flagged': sum(len(b) for b in batches),
                'batches': len(batches)
            }

    # Generate progress file
    if edition_stats:
        generate_progress_file(edition_stats)

    print("\nDone!")


if __name__ == "__main__":
    main()
