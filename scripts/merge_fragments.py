#!/usr/bin/env python3
"""Merge running-header fragments in parsed article JSONL files.

OLMoCR sometimes preserves running page headers (e.g., "SHIP-BUILDING" at the
top of every page). The LIS parser picks these up as article boundaries, splitting
one long treatise into many fragments that break mid-sentence.

This script detects and merges those fragments by checking whether the boundary
between consecutive same-headword articles is mid-sentence.

Usage:
    # Dry run (report only, no changes)
    python scripts/merge_fragments.py --dry-run

    # Apply merges
    python scripts/merge_fragments.py

    # Apply merges and show detailed log
    python scripts/merge_fragments.py --verbose
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

ARTICLES_DIR = REPO_DIR / "data" / "articles"

# Terminal punctuation that indicates a sentence ending
TERMINAL_PUNCT = set('.;!?"\u201d)')

# Words that strongly indicate mid-sentence continuation
CONTINUATION_STARTERS = {
    'the', 'of', 'and', 'which', 'but', 'that', 'for', 'in', 'to', 'by',
    'from', 'or', 'with', 'as', 'on', 'at', 'is', 'are', 'was', 'were',
    'it', 'its', 'this', 'these', 'those', 'their', 'his', 'her', 'he',
    'she', 'they', 'we', 'who', 'whom', 'whose', 'where', 'when', 'than',
    'not', 'so', 'if', 'into', 'upon', 'between', 'through', 'under',
    'over', 'after', 'before', 'during', 'without', 'within', 'about',
    'being', 'having', 'also', 'however', 'therefore', 'thus', 'hence',
    'yet', 'nor', 'neither', 'either', 'both', 'each', 'every', 'all',
}

# Article-opening patterns that indicate a genuine new article
ARTICLE_OPENING_RE = re.compile(
    r'^[A-Z][A-Z\'\- ]+[,.]'  # HEADWORD, definition...
    r'|^[A-Z][a-z]'            # Capitalized word (normal sentence start)
    r'|^\([A-Z]'               # (Qualifier)
    r'|^In [a-z]'              # "In botany, ..."
    r'|^See '                  # Cross-reference
    r'|^A [a-z]'              # "A genus of..."
    r'|^The [a-z]'            # "The name of..."
)


TRAILING_NOISE_RE = re.compile(
    r'(?:\n|^)(?:Part\s+[IVXLC]+\.?|Chap\.?\s+[IVXLC]+\.?|Sect\.?\s+[IVXLC0-9]+\.?'
    r'|PLATE\s+[A-Z0-9]+\.?|[A-Z][A-Z\- ]+\.?)\s*$'
)

LEADING_NOISE_RE = re.compile(
    r'^(?:Part\s+[IVXLC]+\.?\s*\n|Chap\.?\s+[IVXLC]+\.?\s*\n|Sect\.?\s+[IVXLC0-9]+\.?\s*\n'
    r'|History\.?\s*\n|[A-Z][A-Z\- ]{2,}\.?\s*\n)'
)


def is_mid_sentence_boundary(prev_text: str, next_text: str) -> bool:
    """Check if the boundary between two text fragments is mid-sentence.

    Returns True if the evidence suggests the break is mid-sentence
    (running header split), False if it looks like a genuine article boundary.
    """
    prev_stripped = prev_text.rstrip()
    next_stripped = next_text.lstrip()

    if not prev_stripped or not next_stripped:
        return False

    # Strip trailing running-header noise (Part I., PLATE CCC., section titles)
    prev_clean = TRAILING_NOISE_RE.sub('', prev_stripped).rstrip()
    next_clean = LEADING_NOISE_RE.sub('', next_stripped).lstrip()

    if not prev_clean:
        prev_clean = prev_stripped
    if not next_clean:
        next_clean = next_stripped

    last_char = prev_clean[-1]
    first_word = next_clean.split()[0].lower() if next_clean.split() else ''

    # Strong mid-sentence signals
    if last_char not in TERMINAL_PUNCT:
        return True
    if first_word in CONTINUATION_STARTERS:
        return True
    if next_clean[0].islower():
        return True

    return False


def merge_fragment_group(fragments: list[dict]) -> dict:
    """Merge a group of same-headword fragments into one article."""
    merged = dict(fragments[0])  # Copy first fragment's metadata

    # Concatenate all text
    texts = [f['text'] for f in fragments]
    merged['text'] = '\n\n'.join(texts)

    # Update metadata
    merged['char_end'] = fragments[-1]['char_end']
    merged['word_count'] = len(merged['text'].split())
    merged['paragraph_count'] = merged['text'].count('\n\n') + 1

    # Mark as merged
    merged['heading_pattern'] = merged.get('heading_pattern', '') + '_merged'
    merged['lis_confidence'] = min(f.get('lis_confidence', 1.0) for f in fragments)

    return merged


def process_file(filepath: Path, dry_run: bool = False, verbose: bool = False) -> dict:
    """Process one article JSONL file. Returns merge statistics."""
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f if line.strip()]

    if not articles:
        return {'file': filepath.name, 'merges': 0, 'articles_removed': 0}

    # Group articles by title, preserving order
    # We need to handle non-consecutive fragments (interleaved with other articles)
    title_indices = defaultdict(list)  # title -> list of indices in articles[]
    for i, art in enumerate(articles):
        if art.get('type') == 'article':
            title_indices[art['title']].append(i)

    # Find titles with multiple occurrences
    merge_indices = set()  # indices to remove (merged into first occurrence)
    merge_groups = []  # for logging

    for title, indices in title_indices.items():
        if len(indices) < 2:
            continue

        # Check boundaries between consecutive fragments
        fragments = [articles[i] for i in indices]
        mid_sentence_count = 0
        total_boundaries = len(fragments) - 1

        for j in range(total_boundaries):
            if is_mid_sentence_boundary(fragments[j]['text'], fragments[j + 1]['text']):
                mid_sentence_count += 1

        # Char-span coverage test: if fragments tile their total span
        # with >80% coverage, they're running-header splits, not multi-sense.
        # Genuine multi-sense entries (BAKER-person1 vs BAKER-occupation) have
        # large gaps between them filled by other articles.
        total_span = fragments[-1]['char_end'] - fragments[0]['char_start']
        total_text_chars = sum(len(f['text']) for f in fragments)
        coverage = total_text_chars / total_span if total_span > 0 else 0

        # Merge if:
        # - majority of boundaries are mid-sentence, OR
        # - char-span coverage > 80% (fragments tile the range = one article)
        should_merge = (
            mid_sentence_count > total_boundaries / 2
            or coverage > 0.8
        )
        if should_merge:
            # Merge all fragments into the first one
            merged = merge_fragment_group(fragments)
            first_idx = indices[0]
            articles[first_idx] = merged

            # Mark subsequent fragments for removal
            for idx in indices[1:]:
                merge_indices.add(idx)

            merge_groups.append({
                'title': title,
                'fragment_count': len(fragments),
                'mid_sentence_boundaries': mid_sentence_count,
                'total_boundaries': total_boundaries,
                'merged_words': merged['word_count'],
            })

            if verbose:
                frag_wcs = [f['word_count'] for f in fragments]
                print(f"  MERGE: {title} ({len(fragments)} frags, "
                      f"{mid_sentence_count}/{total_boundaries} mid-sentence, "
                      f"{merged['word_count']:,}w total)")
                print(f"         Fragment sizes: {frag_wcs}")

    if not merge_groups:
        return {'file': filepath.name, 'merges': 0, 'articles_removed': 0}

    # Remove merged fragments
    new_articles = [art for i, art in enumerate(articles) if i not in merge_indices]

    stats = {
        'file': filepath.name,
        'merges': len(merge_groups),
        'articles_removed': len(merge_indices),
        'groups': merge_groups,
    }

    if not dry_run:
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            for art in new_articles:
                f.write(json.dumps(art, ensure_ascii=False) + '\n')

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge running-header fragments")
    parser.add_argument('--dry-run', action='store_true',
                        help="Report only, don't modify files")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show detailed merge log")
    parser.add_argument('--backup', action='store_true',
                        help="Create .bak files before modifying")
    args = parser.parse_args()

    files = sorted(ARTICLES_DIR.glob("*.articles.jsonl"))
    if not files:
        print(f"No article files found in {ARTICLES_DIR}")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} article files...")
    print()

    total_merges = 0
    total_removed = 0
    all_groups = []

    for filepath in files:
        if args.backup and not args.dry_run:
            shutil.copy2(filepath, filepath.with_suffix('.jsonl.bak'))

        stats = process_file(filepath, dry_run=args.dry_run, verbose=args.verbose)

        if stats['merges'] > 0:
            total_merges += stats['merges']
            total_removed += stats['articles_removed']
            all_groups.extend(stats.get('groups', []))

            if not args.verbose:
                print(f"  {stats['file']}: {stats['merges']} merges, "
                      f"{stats['articles_removed']} fragments removed")

    print()
    print(f"Total: {total_merges} headwords merged, "
          f"{total_removed} excess articles removed")

    if all_groups:
        # Summary of biggest merges
        biggest = sorted(all_groups, key=lambda g: -g['fragment_count'])[:15]
        print(f"\nTop merges by fragment count:")
        for g in biggest:
            print(f"  {g['title']:40s} {g['fragment_count']:>3} frags → "
                  f"{g['merged_words']:>8,}w  "
                  f"({g['mid_sentence_boundaries']}/{g['total_boundaries']} mid-sentence)")


if __name__ == "__main__":
    main()
