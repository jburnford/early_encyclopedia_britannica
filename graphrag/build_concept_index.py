#!/usr/bin/env python3
"""Build a cross-edition concept index from the cleaned headword dictionary
and LIS parser article output.

Reads:
  - data/headword_dictionary_clean.json
  - data/articles/*.articles.jsonl  (155 files, 126K articles from Plato LIS parser)

Produces:
  - graphrag/concept_index.json

Each article file has records like:
  {
    "article_id": "eb_1st_1771_v01_0001",
    "title": "AARSEO",
    "edition": "1st",
    "edition_year": 1771,
    "volume": 1,
    "source_file": "eb_1st_1771_v01_ADA-NOT.jsonl",
    "type": "article" | "cross_reference",
    "word_count": 11,
    "target": null | "OPTICS",
    ...
  }
"""

import json
import glob
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


EDITION_INFO = {
    1771: {'edition': '1st', 'name': 'First Edition', 'volumes': 3},
    1778: {'edition': '2nd', 'name': 'Second Edition', 'volumes': 10},
    1797: {'edition': '3rd', 'name': 'Third Edition', 'volumes': 18},
    1810: {'edition': '4th', 'name': 'Fourth Edition (Supplement)', 'volumes': 20},
    1815: {'edition': '5th', 'name': 'Fifth Edition', 'volumes': 20},
    1823: {'edition': '6th', 'name': 'Sixth Edition', 'volumes': 20},
    1842: {'edition': '7th', 'name': 'Seventh Edition', 'volumes': 21},
    1860: {'edition': '8th', 'name': 'Eighth Edition', 'volumes': 21},
}


def normalize_sort_key(headword: str) -> str:
    """Replicate the parser's sort key normalization."""
    key = headword.upper()
    key = key.replace('U', 'V').replace('I', 'J')
    key = unicodedata.normalize('NFKD', key)
    key = key.encode('ASCII', 'ignore').decode('ASCII')
    key = re.sub(r"['\-]", '', key)
    key = re.sub(r'\s+', ' ', key).strip()
    return key


def load_articles(articles_dir: str) -> list[dict]:
    """Load all articles from the LIS parser output."""
    articles = []
    files = sorted(glob.glob(os.path.join(articles_dir, '*.articles.jsonl')))
    print(f"  Found {len(files)} article files")

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                articles.append(obj)

    return articles


def build_concept_index(base_dir: str) -> dict:
    """Build the cross-edition concept index."""
    base = Path(base_dir)

    # Load cleaned dictionary
    dict_path = base / 'data' / 'headword_dictionary_clean.json'
    print(f"Loading cleaned dictionary from {dict_path}...")
    with open(dict_path) as f:
        hw_dict = json.load(f)

    # Build sort_key -> dict_entry lookup
    key_to_dict = {}
    for sort_key, entry in hw_dict.items():
        key_to_dict[sort_key] = entry
        # Also index by the display headword's sort key
        hw_key = normalize_sort_key(entry['headword'])
        if hw_key not in key_to_dict:
            key_to_dict[hw_key] = entry

    print(f"  {len(hw_dict)} dictionary entries, {len(key_to_dict)} lookup keys")

    # Load articles
    articles_dir = base / 'data' / 'articles'
    print(f"Loading articles from {articles_dir}...")
    articles = load_articles(str(articles_dir))
    print(f"  Loaded {len(articles)} articles")

    # Count by edition
    ed_counts = defaultdict(int)
    for a in articles:
        ed_counts[a['edition_year']] += 1
    for y in sorted(ed_counts):
        info = EDITION_INFO.get(y, {})
        print(f"    {y} ({info.get('edition', '?')}): {ed_counts[y]} articles")

    # Build concept index
    concepts = {}
    stats = defaultdict(int)

    for article in articles:
        title = article.get('title', '')
        if not title or not title.strip():
            continue

        sort_key = normalize_sort_key(title)
        if not sort_key:
            continue

        # Look up in dictionary
        dict_entry = key_to_dict.get(sort_key)

        # Get or create concept
        if sort_key not in concepts:
            if dict_entry:
                label = dict_entry['headword']
                aliases = dict_entry.get('aliases', [])
                stats['matched_dictionary'] += 1
            else:
                label = title
                aliases = []
                stats['not_in_dictionary'] += 1

            concepts[sort_key] = {
                'concept_id': f'eb_{sort_key}',
                'label': label,
                'aliases': aliases,
                'editions': {},
                'in_dictionary': dict_entry is not None,
            }

        concept = concepts[sort_key]
        year_str = str(article['edition_year'])
        word_count = article.get('word_count', 0)
        article_type = article.get('type', 'article')

        # Add or update edition entry — keep the largest article per edition
        if year_str not in concept['editions']:
            ed_entry = {
                'volume': article.get('volume', 0),
                'word_count': word_count,
                'type': article_type,
                'source_file': article.get('source_file', ''),
            }
            if article.get('target'):
                ed_entry['target'] = article['target']
            concept['editions'][year_str] = ed_entry
        else:
            existing = concept['editions'][year_str]
            if word_count > existing['word_count']:
                ed_entry = {
                    'volume': article.get('volume', 0),
                    'word_count': word_count,
                    'type': article_type,
                    'source_file': article.get('source_file', ''),
                }
                if article.get('target'):
                    ed_entry['target'] = article['target']
                concept['editions'][year_str] = ed_entry

    print(f"\n  Concepts matched to dictionary: {stats['matched_dictionary']}")
    print(f"  Concepts not in dictionary:     {stats['not_in_dictionary']}")
    print(f"  Total concepts:                 {len(concepts)}")

    # Compute metrics
    for sort_key, concept in concepts.items():
        eds = concept['editions']
        years = sorted(int(y) for y in eds.keys())
        concept['edition_count'] = len(years)
        concept['first_edition'] = years[0] if years else None
        concept['last_edition'] = years[-1] if years else None
        concept['total_word_count'] = sum(e['word_count'] for e in eds.values())

        if len(years) >= 2:
            first_wc = eds[str(years[0])]['word_count']
            last_wc = eds[str(years[-1])]['word_count']
            if first_wc > 0 and last_wc > 0:
                concept['growth_ratio'] = round(last_wc / first_wc, 2)
            else:
                concept['growth_ratio'] = None
        else:
            concept['growth_ratio'] = None

        if not concept['aliases']:
            del concept['aliases']

    return concepts


def main():
    base_dir = str(Path(__file__).parent.parent)
    concepts = build_concept_index(base_dir)

    output_path = os.path.join(base_dir, 'graphrag', 'concept_index.json')
    print(f"\nWriting {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(concepts, f, indent=1, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"CONCEPT INDEX SUMMARY")
    print(f"{'='*60}")
    print(f"Total concepts: {len(concepts)}")

    ed_counts = defaultdict(int)
    for c in concepts.values():
        for y in c['editions']:
            ed_counts[y] += 1
    print(f"\nConcepts per edition:")
    for y in sorted(ed_counts.keys()):
        info = EDITION_INFO.get(int(y), {})
        print(f"  {y} ({info.get('edition', '?')}): {ed_counts[y]:>6}")

    core = [c for c in concepts.values() if c['edition_count'] >= 6]
    print(f"\nCore concepts (6+ editions): {len(core)}")

    growing = [c for c in concepts.values()
               if c.get('growth_ratio') and c['growth_ratio'] > 10
               and c['edition_count'] >= 4]
    growing.sort(key=lambda c: c['growth_ratio'], reverse=True)
    print(f"\nFastest growing (>10x, 4+ editions): {len(growing)}")
    for c in growing[:15]:
        first_wc = c['editions'].get(str(c['first_edition']), {}).get('word_count', 0)
        last_wc = c['editions'].get(str(c['last_edition']), {}).get('word_count', 0)
        print(f"  {c['label']}: {c['growth_ratio']}x ({first_wc} → {last_wc} words, "
              f"{c['edition_count']} editions)")

    new_after_1800 = [c for c in concepts.values()
                      if c.get('first_edition') and c['first_edition'] > 1800
                      and c['edition_count'] >= 3]
    new_after_1800.sort(key=lambda c: c['first_edition'])
    print(f"\nIntroduced after 1800 (3+ editions): {len(new_after_1800)}")
    for c in new_after_1800[:10]:
        print(f"  {c['label']}: first={c['first_edition']}, editions={c['edition_count']}")

    print(f"\nWrote {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
