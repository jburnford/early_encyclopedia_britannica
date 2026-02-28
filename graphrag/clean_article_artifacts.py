#!/usr/bin/env python3
"""Remove parser artifacts (publisher names, volume markers, ads) from article files.

Reads data/articles/*.articles.jsonl, removes/fixes artifacts, writes back in-place.
Then regenerates data/export/ consolidated files.

Artifacts removed:
  - "END OF THE * VOLUME" / "THE END OF" markers
  - "AND SONS" (publisher: Thomas Wilson and Sons, York)
  - "ADAM AND CHARLES BLACK" (publisher)
  - "AND ALL BOOKSELLERS" / "BLACK'S GUIDE *" / "BOOKS FOR CHRISTMAS" (publisher ads)
  - "JAMES DONALDSON" / "WILLIAM CAXTON" (publisher names)
  - "VOLUME" (stray headings)
  - "NOTICE" (front matter, small entries only)
  - Publisher-fragment "GALE" entries (detected by text content)

Special fixes:
  - 1860 v1 "END OF THE SECOND DISSERTATION" → rename to "DISSERTATION THIRD"
  - 1771 v1 "NOTICE" (4135 words) → remove (8th edition front matter in 1st edition file)
"""

import json
import glob
import os
import re
from pathlib import Path
from collections import defaultdict


def is_artifact(article: dict) -> tuple[bool, str]:
    """Check if an article is a parser artifact. Returns (is_artifact, reason)."""
    title = article.get('title', '')
    text = article.get('text', '')
    wc = article.get('word_count', 0)
    t_upper = title.upper()

    # End-of-volume markers
    if t_upper.startswith('END OF') or t_upper.startswith('THE END'):
        # Special case: 1860 v1 "END OF THE SECOND DISSERTATION" is actually
        # Dissertation Third (31K words) — fix title instead of removing
        if 'DISSERTATION' in t_upper and wc > 1000:
            return False, ''
        return True, 'end-of-volume marker'

    # Publisher fragments
    if t_upper == 'AND SONS':
        return True, 'publisher (Thomas Wilson and Sons)'
    if t_upper.startswith('ADAM AND CHARLES BLACK'):
        return True, 'publisher (Adam and Charles Black)'
    if t_upper == 'AND ALL BOOKSELLERS':
        return True, 'publisher ad'
    if t_upper.startswith("BLACK'S GUIDE"):
        return True, 'publisher ad'
    if t_upper == 'BOOKS FOR CHRISTMAS PRESENTS':
        return True, 'publisher ad'
    if t_upper == 'JAMES DONALDSON':
        return True, 'publisher name'
    if t_upper == 'WILLIAM CAXTON':
        return True, 'publisher ad'
    if t_upper == 'VOLUME':
        return True, 'stray volume heading'

    # GALE: real article about wind vs publisher fragment (Gale, Curtis, and Fenner)
    if t_upper == 'GALE':
        text_start = text[:200].upper()
        if 'CURTIS' in text_start or 'FENNER' in text_start or (
            'YORK' in text[:50].upper() and wc < 50):
            return True, 'publisher (Gale, Curtis, and Fenner)'

    # NOTICE: front matter (keep large real articles)
    if t_upper == 'NOTICE':
        year = article.get('edition_year', 0)
        if year == 1771:
            return True, '8th ed front matter in 1st ed file'
        if wc < 200:
            return True, 'front matter notice'

    return False, ''


def fix_title(article: dict) -> dict | None:
    """Fix misnamed articles. Returns modified article or None if no fix needed."""
    title = article.get('title', '')
    text = article.get('text', '')
    t_upper = title.upper()

    # 1860 "END OF THE SECOND DISSERTATION" → "DISSERTATION THIRD"
    if 'SECOND DISSERTATION' in t_upper and 'DISSERTATION THIRD' in text[:200].upper():
        article = dict(article)
        article['title'] = 'DISSERTATION THIRD'
        return article

    # 1842 "END OF THE SECOND DISSERTATION" (small, 33 words) — just remove
    return None


def main():
    base = Path(__file__).parent.parent
    articles_dir = base / 'data' / 'articles'
    export_dir = base / 'data' / 'export'

    print("Scanning for artifacts...")

    # Process each article file
    total_removed = 0
    total_fixed = 0
    total_articles = 0
    removed_log = []

    for fpath in sorted(articles_dir.glob('*.articles.jsonl')):
        articles = []
        removed = 0

        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                article = json.loads(line)
                total_articles += 1

                is_art, reason = is_artifact(article)
                if is_art:
                    removed += 1
                    removed_log.append(
                        f"  {article['edition_year']} v{article['volume']:>2} | "
                        f"{article['title'][:40]:<40} wc={article['word_count']:>5} | {reason}"
                    )
                    continue

                # Check for title fixes
                if 'DISSERTATION' in article.get('title', '').upper():
                    fixed = fix_title(article)
                    if fixed:
                        article = fixed
                        total_fixed += 1
                        print(f"  Fixed: {article['edition_year']} v{article['volume']} "
                              f"→ {article['title']}")

                articles.append(article)

        if removed > 0:
            # Write back cleaned file
            with open(fpath, 'w') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
            total_removed += removed
            print(f"  {fpath.name}: removed {removed} artifacts")

    print(f"\nRemoved {total_removed} artifacts from {total_articles} articles")
    print(f"Fixed {total_fixed} titles")
    print()
    for line in sorted(removed_log):
        print(line)

    # Regenerate export files
    print(f"\nRegenerating exports in {export_dir}...")
    edition_years = {
        '1st': 1771, '2nd': 1778, '3rd': 1797, '4th': 1810,
        '5th': 1815, '6th': 1823, '7th': 1842, '8th': 1860,
    }

    by_edition = defaultdict(list)
    for fpath in sorted(articles_dir.glob('*.articles.jsonl')):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                article = json.loads(line)
                by_edition[article['edition']].append(article)

    for edition, articles in sorted(by_edition.items()):
        year = edition_years[edition]
        articles.sort(key=lambda a: (a.get('volume', 0), a.get('char_start', 0)))
        for i, article in enumerate(articles, 1):
            article['article_id'] = f"eb_{edition}_{year}_{i:06d}"

        filename = f"eb_{edition}_{year}.jsonl"
        output_path = export_dir / filename
        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        print(f"  {filename}: {len(articles):,} articles")

    final_total = sum(len(a) for a in by_edition.values())
    print(f"\nDone. {final_total:,} articles remaining (was {total_articles:,})")


if __name__ == '__main__':
    main()
