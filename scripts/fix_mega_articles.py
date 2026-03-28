#!/usr/bin/env python3
"""Fix specific mega-articles that swallowed neighboring entries.

Each fix is hand-specified based on manual analysis of the article text.
This is a one-time cleanup script, not a general-purpose tool.

Usage:
    python scripts/fix_mega_articles.py --dry-run    # preview changes
    python scripts/fix_mega_articles.py              # apply changes
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

ARTICLES_DIR = REPO_DIR / "data" / "articles"


def find_split_point(text, pattern, after_pct=0):
    """Find the character position where `pattern` starts a new article.

    Returns the char offset in `text` where the split should occur
    (everything before this point stays in the current article,
     everything from this point starts the new article).

    after_pct: only match occurrences after this percentage through the text.
    """
    min_pos = int(len(text) * after_pct / 100)
    for m in re.finditer(pattern, text):
        if m.start() >= min_pos:
            # Back up to start of the line (after previous \n\n)
            pos = text.rfind('\n\n', 0, m.start())
            if pos == -1:
                pos = 0
            else:
                pos += 2  # skip the \n\n
            return pos
    return None


def split_article(articles, idx, splits):
    """Split articles[idx] at the given split points.

    splits: list of (new_title, pattern, after_pct) tuples, in order.
    Returns list of new articles to replace articles[idx].
    """
    original = articles[idx]
    text = original['text']
    result = []

    # Find all split positions
    positions = []  # (char_pos, new_title)
    for new_title, pattern, after_pct in splits:
        pos = find_split_point(text, pattern, after_pct)
        if pos is not None:
            positions.append((pos, new_title))

    # Sort by position
    positions.sort(key=lambda x: x[0])

    # Create article slices
    prev_pos = 0
    prev_title = original['title']

    for pos, new_title in positions:
        if pos <= prev_pos:
            continue
        chunk_text = text[prev_pos:pos].strip()
        if chunk_text:
            art = dict(original)
            art['title'] = prev_title
            art['text'] = chunk_text
            art['word_count'] = len(chunk_text.split())
            art['char_start'] = original['char_start'] + prev_pos
            art['char_end'] = original['char_start'] + pos
            art['article_id'] = f"{original['article_id']}_{len(result)}" if result else original['article_id']
            art['heading_pattern'] = 'mega_split_manual'
            result.append(art)
        prev_pos = pos
        prev_title = new_title

    # Final chunk
    chunk_text = text[prev_pos:].strip()
    if chunk_text:
        art = dict(original)
        art['title'] = prev_title
        art['text'] = chunk_text
        art['word_count'] = len(chunk_text.split())
        art['char_start'] = original['char_start'] + prev_pos
        art['char_end'] = original['char_end']
        art['article_id'] = f"{original['article_id']}_{len(result)}"
        art['heading_pattern'] = 'mega_split_manual'
        result.append(art)

    return result


# ============================================================================
# Fix specifications: (year, title, file_pattern, splits)
# Each split is (new_title, regex_pattern, min_percent)
# ============================================================================

FIXES = [
    # BOSWORTH-MARKET swallowed BOTAL and BOTANY
    (1860, 'BOSWORTH-MARKET', 'eb_8th_1860_v05', [
        ('BOTAL', r'BOTAL,', 0),
        ('BOTANY', r'VEGETABLE ORGANOGRAPHY AND PHYSIOLOGY', 0),
    ]),

    # UNIVERSITY OF PARIS swallowed all university sub-articles (1860)
    (1860, 'UNIVERSITY OF PARIS', 'eb_8th_1860_v21', [
        ('UNIVERSITIES (English)', r'ENGLISH UNIVERSITIES', 10),
        ('UNIVERSITY OF OXFORD', r'UNIVERSITY OF OXFORD', 10),
        ('UNIVERSITY OF CAMBRIDGE', r'UNIVERSITY OF CAMBRIDGE', 20),
        ('UNIVERSITY OF LONDON', r'UNIVERSITY OF LONDON', 35),
        ('UNIVERSITIES (Scottish)', r'SCOTTISH UNIVERSITIES', 50),
        ('UNIVERSITY OF GLASGOW', r'UNIVERSITY OF GLASGOW', 58),
        ('UNIVERSITY OF ABERDEEN', r'UNIVERSITY OF ABERDEEN', 63),
        ('UNIVERSITY OF EDINBURGH', r'UNIVERSITY OF EDINBURGH', 68),
        ('UNIVERSITY OF DUBLIN', r'UNIVERSITY OF DUBLIN', 75),
        ('UNIVERSITIES (Colonial)', r'COLONIAL UNIVERSITIES', 88),
        ('UNIVERSITY OF FRANCE', r'UNIVERSITY OF FRANCE', 90),
    ]),

    # UNIVERSITY OF PARIS swallowed sub-articles (1842)
    (1842, 'UNIVERSITY OF PARIS', 'eb_7th_1842_v21', [
        ('UNIVERSITIES (English)', r'ENGLISH UNIVERSITIES', 10),
        ('UNIVERSITY OF LONDON', r'UNIVERSITY OF LONDON', 40),
        ('UNIVERSITIES (Scottish)', r'SCOTI.H UNIVERSITIES', 48),
        ('UNIVERSITY OF ABERDEEN', r'UNIVERSITY OF ABERDEEN', 60),
        ('UNIVERSITY OF EDINBURGH', r'UNIVERSITY OF EDINBURGH', 70),
        ('UNIVERSITY OF DUBLIN', r'UNIVERSITY OF DUBLIN', 78),
        ('UNIVERSITY OF FRANCE', r'ROYAL UNIVERSITY OF FRANCE', 88),
    ]),

    # MINERALOGY swallowed GEOLOGY (1842)
    (1842, 'MINERALOGY', 'eb_7th_1842_v15', [
        ('GEOLOGY', r'OBJECTS OF GEOLOGICAL SCIENCE', 60),
    ]),

    # SCOTLAND IS BY NO → should be SCOTLAND (1815 broken headword)
    (1815, 'SCOTLAND IS BY NO', 'eb_5th_1815_v18', [
        # Just rename, don't split — it's the real SCOTLAND article with a broken title
    ]),

    # ANTAGONISTS OF HOBBIESTS → part of DISSERTATIONS (1842 broken headword)
    # This is actually a fragment of the Dissertations prelim material
    # Just rename it
    (1842, 'ANTAGONISTS OF HOBBIESTS', 'eb_7th_1842_v01', []),

    # CLOCK AND WATCH WORK — broken headword, probably CLOCKS
    (1842, 'CLOCK AND WATCH WORK', 'eb_7th_1842_v06', []),

    # HYDRODYNAMICS swallowed INDEX and DIRECTIONS (1810, 1815, 1823)
    (1810, 'HYDRODYNAMICS', 'eb_4th_1810_v10', []),
    (1815, 'HYDRODYNAMICS', 'eb_5th_1815_v10', []),
    (1823, 'HYDRODYNAMICS', 'eb_6th_1823_v10', []),
]


def process_fix(year, title, file_pattern, splits, dry_run=False):
    """Apply a single fix."""
    # Find the file
    matches = list(ARTICLES_DIR.glob(f"{file_pattern}*.articles.jsonl"))
    if not matches:
        print(f"  WARNING: No file matching {file_pattern}")
        return 0

    for filepath in matches:
        if filepath.suffix == '.bak':
            continue
        with open(filepath, 'r') as f:
            articles = [json.loads(line) for line in f if line.strip()]

        # Find the article
        target_idx = None
        for i, a in enumerate(articles):
            if a['title'] == title and a['edition_year'] == year:
                target_idx = i
                break

        if target_idx is None:
            continue

        art = articles[target_idx]

        # Handle rename-only (no splits)
        if not splits:
            if title == 'SCOTLAND IS BY NO':
                print(f"  RENAME: '{title}' → 'SCOTLAND' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'SCOTLAND'
            elif title == 'ANTAGONISTS OF HOBBIESTS':
                print(f"  RENAME: '{title}' → 'DISSERTATIONS' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'DISSERTATIONS'
            elif title == 'CLOCK AND WATCH WORK':
                print(f"  RENAME: '{title}' → 'CLOCKS' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'CLOCKS'
            elif title == 'HYDRODYNAMICS':
                # Strip trailing INDEX and DIRECTIONS if present
                text = art['text']
                idx_match = re.search(r'\n\nINDEX[,.]', text)
                dir_match = re.search(r'\n\nDIRECTIONS FOR PLACING', text)
                cut_point = None
                if idx_match:
                    cut_point = idx_match.start()
                elif dir_match:
                    cut_point = dir_match.start()
                if cut_point:
                    old_wc = art['word_count']
                    art['text'] = text[:cut_point].strip()
                    art['word_count'] = len(art['text'].split())
                    print(f"  TRIM: '{title}' trailing matter removed ({old_wc:,}w → {art['word_count']:,}w)")
                    if not dry_run:
                        articles[target_idx] = art
                else:
                    print(f"  SKIP: '{title}' — no trailing INDEX found")
                    return 0
            else:
                print(f"  SKIP: '{title}' — no splits defined")
                return 0

            if not dry_run:
                with open(filepath, 'w') as f:
                    for a in articles:
                        f.write(json.dumps(a, ensure_ascii=False) + '\n')
            return 1

        # Apply splits
        new_articles = split_article(articles, target_idx, splits)

        if len(new_articles) <= 1:
            print(f"  WARNING: No splits found for {title}")
            return 0

        print(f"  SPLIT: '{title}' ({art['word_count']:,}w) → {len(new_articles)} articles:")
        for na in new_articles:
            print(f"    {na['title']:40s} {na['word_count']:>8,}w")

        if not dry_run:
            # Replace the original article with the split versions
            articles[target_idx:target_idx + 1] = new_articles
            with open(filepath, 'w') as f:
                for a in articles:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')

        return len(new_articles) - 1  # excess articles added

    print(f"  WARNING: Article '{title}' not found in {file_pattern}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fix mega-articles")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print(f"{'DRY RUN: ' if args.dry_run else ''}Fixing {len(FIXES)} mega-articles...\n")

    total_changes = 0
    for year, title, file_pattern, splits in FIXES:
        print(f"\n{year} {title}:")
        changes = process_fix(year, title, file_pattern, splits, dry_run=args.dry_run)
        total_changes += changes

    print(f"\nTotal changes: {total_changes}")


if __name__ == "__main__":
    main()
