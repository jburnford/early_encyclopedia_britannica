#!/usr/bin/env python3
"""Classify cross-edition gaps as variant headwords, swallowed articles,
OCR gaps, editorial decisions, or unknown.

Usage:
    python scripts/classify_gaps.py [--verbose]
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

ARTICLES_DIR = REPO_DIR / "data" / "articles"
OCR_DIR = REPO_DIR / "data" / "ocr" / "organized"
INDEX_PATH = REPO_DIR / "data" / "cross_edition_index.jsonl"
OUTPUT_JSONL = REPO_DIR / "data" / "gap_classifications.jsonl"
OUTPUT_CSV = REPO_DIR / "data" / "gap_classifications.csv"

YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]


def normalize(title):
    return re.sub(r'[^A-Z ]', '', title.upper()).strip()


# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------

def load_edition_articles():
    """Load all articles grouped by edition year."""
    by_year = defaultdict(list)  # year -> list of {title, word_count, norm, volume}
    by_year_norm = defaultdict(dict)  # year -> {norm: {title, wc, volume}}

    for fp in sorted(ARTICLES_DIR.glob("*.articles.jsonl")):
        if fp.suffix != '.jsonl' or '.bak' in fp.name or '.junk' in fp.name:
            continue
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                a = json.loads(line)
                if a.get('type') == 'cross_reference':
                    continue
                year = a['edition_year']
                norm = normalize(a['title'])
                wc = a.get('word_count', 0)
                entry = {'title': a['title'], 'word_count': wc,
                         'norm': norm, 'volume': a.get('volume', 0)}
                by_year[year].append(entry)
                if norm not in by_year_norm[year] or wc > by_year_norm[year][norm]['word_count']:
                    by_year_norm[year][norm] = entry

    return by_year, by_year_norm


def load_mega_articles():
    """Load text of articles >10K words for swallowed detection."""
    mega = defaultdict(list)  # year -> [{title, text, word_count, volume}]
    for fp in sorted(ARTICLES_DIR.glob("*.articles.jsonl")):
        if '.bak' in fp.name or '.junk' in fp.name:
            continue
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                a = json.loads(line)
                if a.get('word_count', 0) < 10000:
                    continue
                if a.get('type') == 'cross_reference':
                    continue
                mega[a['edition_year']].append({
                    'title': a['title'],
                    'text': a['text'],
                    'word_count': a['word_count'],
                    'volume': a.get('volume', 0),
                })
    return mega


def load_ocr_ranges():
    """Build alphabetical coverage map per edition from OCR files."""
    ranges = defaultdict(list)  # year -> [(first_hw, last_hw, volume, filename)]
    for fp in sorted(OCR_DIR.glob("*.jsonl")):
        with open(fp) as f:
            ocr = json.load(f)
        text = ocr.get('text', '')
        if len(text) < 50000:
            continue  # skip truncated files

        # Determine edition year from filename
        m = re.search(r'eb_\w+_(\d{4})', fp.name)
        if not m:
            continue
        year = int(m.group(1))

        # Find first and last headword-like patterns
        hws = re.findall(r'\n\n([A-Z][A-Z\'\-]{2,}),\s+', text[:30000])
        last_hws = re.findall(r'\n\n([A-Z][A-Z\'\-]{2,}),\s+', text[-30000:])

        first = normalize(hws[0]) if hws else ''
        last = normalize(last_hws[-1]) if last_hws else ''
        if first and last:
            vol = ocr.get('volume', 0)
            ranges[year].append((first, last, vol, fp.name))

    return ranges


# -----------------------------------------------------------------------
# Signal 1: Variant headword matching
# -----------------------------------------------------------------------

def find_variant(missing_norm, missing_title, year, by_year_norm):
    """Check if article exists under a variant headword."""
    edition_norms = by_year_norm.get(year, {})

    # Exact match (shouldn't happen if it's truly missing, but check)
    if missing_norm in edition_norms:
        v = edition_norms[missing_norm]
        if v['word_count'] >= 50:
            return v['title'], v['word_count'], 'exact'

    # Plural/singular variants
    for suffix in ['S', 'ES', 'IES']:
        if missing_norm.endswith(suffix):
            stem = missing_norm[:-len(suffix)]
            if stem in edition_norms:
                v = edition_norms[stem]
                if v['word_count'] >= 200:
                    return v['title'], v['word_count'], 'singular'
        variant = missing_norm + suffix
        if variant in edition_norms:
            v = edition_norms[variant]
            if v['word_count'] >= 200:
                return v['title'], v['word_count'], 'plural'

    # Prefix match (first 6+ chars) with high similarity AND reasonable size
    prefix = missing_norm[:6] if len(missing_norm) >= 6 else missing_norm
    candidates = [(n, e) for n, e in edition_norms.items()
                  if n.startswith(prefix) and e['word_count'] >= 500
                  and n != missing_norm]
    if candidates:
        best_sim = 0
        best = None
        for n, e in candidates:
            sim = SequenceMatcher(None, missing_norm, n).ratio()
            if sim > best_sim:
                best_sim = sim
                best = (e['title'], e['word_count'], f'prefix_sim={sim:.2f}')
        if best_sim >= 0.85:
            return best

    # Substring match: only if one headword fully contains the other
    # AND the match is substantial (matched article >= 20% of expected size)
    if len(missing_norm) >= 8:
        for n, e in edition_norms.items():
            if e['word_count'] < 500:
                continue
            if n == missing_norm:
                continue
            # Only match if shorter is a prefix of longer (not arbitrary substring)
            if missing_norm.startswith(n) or n.startswith(missing_norm):
                if abs(len(n) - len(missing_norm)) <= 5:
                    return e['title'], e['word_count'], 'substring'

    return None


# -----------------------------------------------------------------------
# Signal 2: Swallowed article detection
# -----------------------------------------------------------------------

def find_swallowed(missing_title, year, mega_articles):
    """Check if article text is buried inside a mega-article.

    Only matches headword-like patterns (HEADWORD, followed by lowercase text
    that looks like a definition), not casual mentions.
    """
    megas = mega_articles.get(year, [])
    target = missing_title.upper()

    # Skip very short or common headwords that appear everywhere as mentions
    if len(target) <= 4:
        return None

    # Require: \n\nHEADWORD, lowercase_text (article opening pattern)
    # or \n\nHEADWORD. Capitalized (treatise pattern)
    pattern = re.compile(
        r'\n\n' + re.escape(target) + r',\s+[a-z]'
        r'|\n\n' + re.escape(target) + r'\.\s+[A-Z]',
        re.IGNORECASE
    )

    for m in megas:
        if m['title'].upper() == target:
            continue
        match = pattern.search(m['text'])
        if match:
            # Verify there's substantial text after the match (not just a passing mention)
            after = m['text'][match.end():match.end() + 500]
            # Count words in the next 500 chars — if there's a paragraph, it's likely an article
            if len(after.split()) >= 30:
                pos_pct = match.start() / len(m['text']) * 100
                return m['title'], m['word_count'], f'at {pos_pct:.0f}% in {m["title"]}'

    return None


# -----------------------------------------------------------------------
# Signal 3: OCR range coverage
# -----------------------------------------------------------------------

def check_ocr_coverage(missing_norm, year, ocr_ranges):
    """Check if headword falls within OCR coverage for this edition."""
    ranges = ocr_ranges.get(year, [])
    if not ranges:
        return None  # no range data

    for first, last, vol, fn in ranges:
        if first <= missing_norm <= last:
            return True  # covered

    return False  # not covered by any range


# -----------------------------------------------------------------------
# Signal 5: Editorial heuristics
# -----------------------------------------------------------------------

def check_editorial(record):
    """Check for editorial decision patterns."""
    editions = record.get('editions', {})
    present_years = sorted(int(y) for y in editions.keys())

    if len(present_years) < 2:
        return None

    wcs = [editions[str(y)]['word_count'] for y in present_years]

    # Check if article is shrinking over time (being phased out)
    if len(wcs) >= 3:
        early_avg = sum(wcs[:len(wcs)//2]) / (len(wcs)//2)
        late_avg = sum(wcs[len(wcs)//2:]) / (len(wcs) - len(wcs)//2)
        if late_avg < early_avg * 0.1:
            return 'shrinking_to_stub'

    # Check if it only exists in early editions (pre-1823)
    if max(present_years) <= 1823 and min(present_years) <= 1797:
        return 'early_editions_only'

    return None


# -----------------------------------------------------------------------
# Main classification
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("Loading data...")
    t0 = time.monotonic()

    by_year, by_year_norm = load_edition_articles()
    print(f"  Articles loaded: {sum(len(v) for v in by_year.values()):,}")

    print("  Loading mega-articles for swallowed detection...")
    mega = load_mega_articles()
    print(f"  Mega-articles: {sum(len(v) for v in mega.values())}")

    print("  Loading OCR ranges...")
    ocr_ranges = load_ocr_ranges()
    print(f"  OCR ranges: {sum(len(v) for v in ocr_ranges.values())}")

    # Load cross-edition index
    with open(INDEX_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Cross-edition records: {len(records):,}")

    elapsed = time.monotonic() - t0
    print(f"  Loaded in {elapsed:.1f}s\n")

    # Process gaps
    results = []
    counts = defaultdict(int)

    for record in records:
        gap_years = record.get('gap_years', [])
        if not gap_years:
            continue

        for missing_year in gap_years:
            title = record['canonical_title']
            norm = record['normalized']
            median_wc = record['max_word_count']

            classification = None
            confidence = 0.0
            evidence = ''
            variant_match = None
            swallowed_by = None

            # Signal 1: Variant headword
            variant = find_variant(norm, title, missing_year, by_year_norm)
            if variant:
                vt, vwc, vtype = variant
                classification = 'VARIANT'
                confidence = 0.85
                evidence = f'{vtype}: found as "{vt}" ({vwc:,}w)'
                variant_match = {'title': vt, 'word_count': vwc, 'match_type': vtype}

            # Signal 2: Swallowed (only if not already classified)
            if not classification:
                swallowed = find_swallowed(title, missing_year, mega)
                if swallowed:
                    st, swc, sdesc = swallowed
                    classification = 'SWALLOWED'
                    confidence = 0.90
                    evidence = f'Found inside "{st}" ({swc:,}w) — {sdesc}'
                    swallowed_by = {'title': st, 'word_count': swc}

            # Signal 3: OCR coverage
            if not classification:
                covered = check_ocr_coverage(norm, missing_year, ocr_ranges)
                if covered is False:
                    classification = 'OCR_GAP'
                    confidence = 0.80
                    evidence = f'Headword "{norm}" not in any OCR range for {missing_year}'

            # Signal 5: Editorial heuristics
            if not classification:
                editorial = check_editorial(record)
                if editorial:
                    classification = 'EDITORIAL'
                    confidence = 0.60
                    evidence = f'Pattern: {editorial}'

            # Default
            if not classification:
                # Check if it's in the covered OCR range — if so, it's likely
                # a parsing error or genuine editorial removal
                if covered is True:
                    classification = 'PARSING_OR_EDITORIAL'
                    confidence = 0.50
                    evidence = 'In OCR range but not parsed — possible parsing error or editorial removal'
                else:
                    classification = 'UNKNOWN'
                    confidence = 0.30
                    evidence = 'No signal matched'

            result = {
                'id': record['id'],
                'canonical_title': title,
                'missing_year': missing_year,
                'classification': classification,
                'confidence': confidence,
                'evidence': evidence,
                'variant_match': variant_match,
                'swallowed_by': swallowed_by,
                'median_wc': median_wc,
                'edition_count': record['edition_count'],
                'present_years': [int(y) for y in record['editions'].keys()],
            }
            results.append(result)
            counts[classification] += 1

            if args.verbose and classification in ('SWALLOWED', 'VARIANT'):
                print(f"  {missing_year} {title:30s} → {classification}: {evidence}")

    # Sort by classification, then by word count
    results.sort(key=lambda r: (r['classification'], -r['median_wc']))

    # Write JSONL
    with open(OUTPUT_JSONL, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'title', 'missing_year', 'classification', 'confidence',
                     'evidence', 'variant_title', 'swallowed_by', 'median_wc',
                     'edition_count', 'present_years'])
        for r in results:
            w.writerow([
                r['id'], r['canonical_title'], r['missing_year'],
                r['classification'], f"{r['confidence']:.2f}",
                r['evidence'],
                r['variant_match']['title'] if r['variant_match'] else '',
                r['swallowed_by']['title'] if r['swallowed_by'] else '',
                r['median_wc'], r['edition_count'],
                '|'.join(str(y) for y in r['present_years']),
            ])

    print(f"\nClassified {len(results)} gaps:")
    for cls in sorted(counts, key=lambda c: -counts[c]):
        print(f"  {cls:25s} {counts[cls]:>5}")
    print(f"\nOutput: {OUTPUT_JSONL}")
    print(f"        {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
