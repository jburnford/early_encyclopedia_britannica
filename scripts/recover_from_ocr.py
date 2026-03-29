#!/usr/bin/env python3
"""Recover articles from OCR that the parser missed.

For each PARSING_OR_EDITORIAL gap where the headword exists in the OCR
with a strong pattern (\n\nHEADWORD, ...), extract the article text from
the OCR file and add it to the articles.

Usage:
    python scripts/recover_from_ocr.py --dry-run    # preview
    python scripts/recover_from_ocr.py              # apply
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

ARTICLES_DIR = REPO_DIR / "data" / "articles"
OCR_DIR = REPO_DIR / "data" / "ocr" / "organized"
GAP_CSV = REPO_DIR / "data" / "gap_classifications.csv"


def normalize(title):
    return re.sub(r'[^A-Z ]', '', title.upper()).strip()


def load_existing_articles():
    """Load existing article titles and max word counts per edition year."""
    by_year = defaultdict(dict)  # year -> {norm: max_wc}
    for fp in sorted(ARTICLES_DIR.glob("*.articles.jsonl")):
        if '.bak' in fp.name or '.junk' in fp.name:
            continue
        for line in open(fp):
            if not line.strip():
                continue
            a = json.loads(line)
            norm = normalize(a['title'])
            year = a['edition_year']
            wc = a.get('word_count', 0)
            if norm not in by_year[year] or wc > by_year[year][norm]:
                by_year[year][norm] = wc
    return by_year


def load_ocr_files():
    """Load OCR files indexed by year."""
    by_year = defaultdict(list)
    for fp in sorted(OCR_DIR.glob("*.jsonl")):
        m = re.search(r'eb_(\w+)_(\d{4})', fp.name)
        if not m:
            continue
        year = int(m.group(2))
        with open(fp) as f:
            ocr = json.load(f)
        text = ocr.get('text', '')
        if len(text) < 50000:
            continue
        edition = m.group(1)
        by_year[year].append({
            'filename': fp.name,
            'text': text,
            'edition': edition,
            'volume': ocr.get('volume', 0),
        })
    return by_year


def find_article_in_ocr(title, ocr_text):
    """Find an article in OCR text and extract it.
    
    Returns (start_pos, end_pos, extracted_text) or None.
    """
    escaped = re.escape(title)

    # Try patterns in order of confidence:
    # 1. ALLCAPS headword after \n\n: strongest signal
    # 2. ALLCAPS headword after \n: still strong for encyclopedias
    # 3. Mixed-case compound titles after \n\n with definition pattern
    # 4. Mixed-case single words after \n\n with topic validation
    candidates = []

    # Pattern 1: ALLCAPS after double newline
    for m in re.finditer(r'\n\n(' + escaped + r')\s*[,;.]\s', ocr_text):
        hw = m.group(1)
        if hw.isupper():
            candidates.append(m)
            break

    # Pattern 1b: ALLCAPS after single newline
    if not candidates:
        for m in re.finditer(r'\n(' + escaped + r')\s*[,;.]\s', ocr_text):
            hw = m.group(1)
            if hw.isupper():
                # Validate: topic must recur
                check_region = ocr_text[m.start():m.start()+3000]
                mentions = len(re.findall(escaped, check_region, re.IGNORECASE))
                if mentions >= 3:
                    candidates.append(m)
                    break

    # Pattern 2: mixed-case headword with topic validation
    # For single words, verify the extracted text is actually about that topic
    # by checking the headword recurs in the first 500 words.
    if not candidates:
        is_compound = len(title.split()) > 1 or '-' in title or "'" in title
        for m in re.finditer(r'\n\n(' + escaped + r')\s*[,;.]\s', ocr_text, re.IGNORECASE):
            hw = m.group(1)
            if not hw[0].isupper():
                continue
            after = ocr_text[m.end():m.end()+60].lstrip()
            if not re.match(r'(?:a |an |the |in |or |one |from |of |is |was |are |[a-z])', after):
                continue

            if not is_compound:
                # Single word: validate topic by checking headword recurs
                check_region = ocr_text[m.start():m.start()+3000]
                mentions = len(re.findall(escaped, check_region, re.IGNORECASE))
                if mentions < 3:
                    continue  # headword barely appears — probably false match

            candidates.append(m)
            break

    if not candidates:
        return None

    m = candidates[0]
    start = m.start() + 2  # skip the \n\n

    # Find the end: next \n\nHEADWORD, pattern (ALLCAPS word followed by comma)
    search_from = m.end() + 100  # skip past the definition start

    next_hw = re.search(r'\n\n([A-Z][A-Z\'\-]{2,}(?:\s+[A-Z]+)*)\s*[,;.]\s',
                        ocr_text[search_from:])

    if next_hw:
        end = search_from + next_hw.start()
    else:
        end = len(ocr_text)

    extracted = ocr_text[start:end].strip()
    return start, end, extracted


def main():
    parser = argparse.ArgumentParser(description="Recover articles from OCR")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help="Limit recoveries (0=all)")
    args = parser.parse_args()

    # Load gaps
    gaps = []
    with open(GAP_CSV) as f:
        for r in csv.DictReader(f):
            if r['classification'] == 'PARSING_OR_EDITORIAL':
                gaps.append(r)
    print(f"PARSING_OR_EDITORIAL gaps: {len(gaps)}")

    # Load existing articles
    existing = load_existing_articles()

    # Load OCR
    print("Loading OCR files...")
    ocr_by_year = load_ocr_files()
    print(f"Loaded {sum(len(v) for v in ocr_by_year.values())} OCR files")

    # For each gap, try to find and extract from OCR
    recovered = []  # (year, title, word_count, ocr_filename)
    by_file = defaultdict(list)  # ocr_filename -> [article_dicts]

    for gap in gaps:
        title = gap['title']
        year = int(gap['missing_year'])
        norm = normalize(title)

        # Skip if already exists with substantial content (>500w)
        existing_wc = existing[year].get(norm, 0)
        if existing_wc > 500:
            continue

        # Search OCR files for this year
        for ocr_info in ocr_by_year.get(year, []):
            result = find_article_in_ocr(title, ocr_info['text'])
            if result:
                start, end, text = result
                wc = len(text.split())
                
                # Skip if extracted text is tiny (< 50 words) — probably a false match
                if wc < 50:
                    continue

                # Skip if extracted text is a cross-reference (< 200w with "See" pattern)
                median_wc = int(gap['median_wc']) if gap['median_wc'] else 0
                if wc < 200 and median_wc > 1000:
                    continue

                # Skip if absurdly large (>5x expected) — probably overshot boundary
                if median_wc > 0 and wc > median_wc * 5:
                    continue

                # Build article dict
                edition_map = {
                    '1st': '1st', '2nd': '2nd', '3rd': '3rd', '4th': '4th',
                    '5th': '5th', '6th': '6th', '7th': '7th', '8th': '8th',
                }
                ed = ocr_info['edition']
                for k, v in edition_map.items():
                    if k in ed:
                        ed = v
                        break

                art = {
                    'article_id': f"recovered_{year}_{norm.replace(' ', '_')[:30]}",
                    'title': title,
                    'edition': ed,
                    'edition_year': year,
                    'volume': ocr_info['volume'],
                    'source_file': ocr_info['filename'],
                    'type': 'article',
                    'text': text,
                    'word_count': wc,
                    'paragraph_count': text.count('\n\n') + 1,
                    'heading_pattern': 'recovered_from_ocr',
                    'char_start': start,
                    'char_end': end,
                }

                art_fname = ocr_info['filename'].replace('.jsonl', '.articles.jsonl')
                by_file[art_fname].append(art)
                recovered.append((year, title, wc, ocr_info['filename']))
                existing[year][norm] = wc  # prevent duplicates
                break

        if args.limit and len(recovered) >= args.limit:
            break

    # Report
    print(f"\nRecovered: {len(recovered)} articles")
    total_wc = sum(wc for _, _, wc, _ in recovered)
    print(f"Total words: {total_wc:,}")

    by_year = defaultdict(int)
    for year, _, wc, _ in recovered:
        by_year[year] += 1
    print(f"\nBy year:")
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]}")

    # Top recoveries
    top = sorted(recovered, key=lambda x: -x[2])[:20]
    print(f"\nTop 20 by word count:")
    for year, title, wc, fname in top:
        print(f"  {year} {title:40s} {wc:>8,}w from {fname}")

    if args.dry_run:
        print("\nDRY RUN — no files written")
        return

    # Write recovered articles to their article files
    written = 0
    for art_fname, arts in sorted(by_file.items()):
        fp = ARTICLES_DIR / art_fname
        # Append to existing file, or create new
        mode = 'a' if fp.exists() else 'w'
        with open(fp, mode) as f:
            for art in arts:
                f.write(json.dumps(art, ensure_ascii=False) + '\n')
                written += 1

    print(f"\nWrote {written} articles to {len(by_file)} files")


if __name__ == "__main__":
    main()
