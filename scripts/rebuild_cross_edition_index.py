"""Rebuild cross_edition_index.jsonl from current article files."""
import json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '1815EncyclopediaBritannicaNLS' / 'scripts'))

REPO = Path('/home/jic823/1815EncyclopediaBritannicaNLS')
ARTICLES_DIR = REPO / 'data' / 'articles'
INDEX_PATH = REPO / 'data' / 'cross_edition_index.jsonl'

YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

def normalize(title):
    return re.sub(r'[^A-Z ]', '', title.upper()).strip()

# Load all articles
by_norm_year = defaultdict(dict)  # norm -> {year: {title, wc, volume, id}}
for fp in sorted(ARTICLES_DIR.glob('*.articles.jsonl')):
    if '.bak' in fp.name or '.junk' in fp.name:
        continue
    with open(fp) as f:
        for line in f:
            if not line.strip(): continue
            a = json.loads(line)
            if a.get('type') == 'cross_reference': continue
            year = a['edition_year']
            norm = normalize(a['title'])
            wc = a.get('word_count', 0)
            # Keep largest version if multiple
            if year not in by_norm_year[norm] or wc > by_norm_year[norm][year].get('word_count', 0):
                by_norm_year[norm][year] = {
                    'title': a['title'],
                    'word_count': wc,
                    'volume': a.get('volume', 0),
                    'article_id': a.get('article_id', ''),
                }

# Build index: substantive articles (>=1000w in at least one edition, present in >=2 editions)
records = []
for norm, years_data in sorted(by_norm_year.items()):
    max_wc = max(d['word_count'] for d in years_data.values())
    edition_count = len(years_data)
    if max_wc < 1000 or edition_count < 2:
        continue
    
    # Find canonical title (from the edition with max wc)
    best_year = max(years_data, key=lambda y: years_data[y]['word_count'])
    canonical = years_data[best_year]['title']
    
    # Find gap years
    present_years = set(years_data.keys())
    all_years = set(YEARS)
    first_year = min(present_years)
    last_year = max(present_years)
    # Only flag gaps between first and last appearance
    gap_years = sorted(y for y in all_years if first_year < y < last_year and y not in present_years)
    
    editions = {}
    for y in sorted(years_data):
        editions[str(y)] = years_data[y]
    
    records.append({
        'id': canonical.upper().replace(' ', '_'),
        'canonical_title': canonical,
        'normalized': norm,
        'max_word_count': max_wc,
        'edition_count': edition_count,
        'editions': editions,
        'present_years': sorted(present_years),
        'gap_years': gap_years,
    })

# Sort by max_word_count descending
records.sort(key=lambda r: -r['max_word_count'])

with open(INDEX_PATH, 'w') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# Also write CSV
CSV_PATH = REPO / 'data' / 'cross_edition_index.csv'
import csv
with open(CSV_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'canonical_title', 'max_word_count', 'edition_count',
                'present_years', 'gap_years'] + [str(y) for y in YEARS])
    for r in records:
        row = [r['id'], r['canonical_title'], r['max_word_count'], r['edition_count'],
               '|'.join(str(y) for y in r['present_years']),
               '|'.join(str(y) for y in r['gap_years'])]
        for y in YEARS:
            ed = r['editions'].get(str(y))
            row.append(ed['word_count'] if ed else '')
        w.writerow(row)

with_gaps = sum(1 for r in records if r['gap_years'])
print(f'Cross-edition index: {len(records)} substantive articles, {with_gaps} with gaps')
print(f'Total gaps: {sum(len(r["gap_years"]) for r in records)}')
