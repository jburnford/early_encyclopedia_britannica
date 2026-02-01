#!/usr/bin/env python3
"""
Find articles that appear across multiple editions of Encyclopedia Britannica.

This script loads all article data from docs/{year}/data/*.json files,
normalizes headwords, and identifies articles appearing in 2+ editions.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path


def normalize_headword(headword: str) -> str:
    """Normalize headword for comparison across editions."""
    if not headword:
        return ""
    # Lowercase
    h = headword.lower()
    # Remove leading/trailing whitespace
    h = h.strip()
    # Normalize whitespace
    h = re.sub(r'\s+', ' ', h)
    # Remove trailing punctuation
    h = re.sub(r'[.,;:]+$', '', h)
    return h


def load_edition(year: int, docs_path: Path) -> dict:
    """
    Load all articles from an edition.

    Returns dict mapping normalized headword to article info.
    """
    edition_path = docs_path / str(year) / "data"
    articles = {}

    if not edition_path.exists():
        print(f"  Warning: Path not found: {edition_path}")
        return articles

    # Load all vol*.json files (skip _original and _corrected variants for now)
    json_files = sorted(edition_path.glob("vol*.json"))
    # Filter to just the main files (not _original, _corrected)
    main_files = [f for f in json_files if '_' not in f.stem or f.stem.startswith('vol')]
    main_files = [f for f in json_files if not ('_original' in f.name or '_corrected' in f.name)]

    for json_file in main_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for article in data:
                headword = article.get('h', '')
                normalized = normalize_headword(headword)

                if normalized:
                    # Store article info
                    if normalized not in articles:
                        articles[normalized] = {
                            'original_headword': headword,
                            'text_length': len(article.get('t', '')),
                            'volume': json_file.stem,
                            'start_page': article.get('sp'),
                            'end_page': article.get('ep')
                        }
                    else:
                        # Same headword appears multiple times in edition
                        # Keep the longer article
                        if len(article.get('t', '')) > articles[normalized]['text_length']:
                            articles[normalized] = {
                                'original_headword': headword,
                                'text_length': len(article.get('t', '')),
                                'volume': json_file.stem,
                                'start_page': article.get('sp'),
                                'end_page': article.get('ep')
                            }

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    return articles


def find_cross_edition_articles(docs_path: Path, years: list[int]) -> dict:
    """
    Find articles appearing in multiple editions.

    Returns dict mapping normalized headword to edition info.
    """
    # Load all editions
    editions_data = {}
    for year in years:
        print(f"Loading {year} edition...")
        editions_data[year] = load_edition(year, docs_path)
        print(f"  Found {len(editions_data[year]):,} unique headwords")

    # Find all unique headwords
    all_headwords = set()
    for year_data in editions_data.values():
        all_headwords.update(year_data.keys())

    print(f"\nTotal unique headwords across all editions: {len(all_headwords):,}")

    # Find cross-edition articles
    cross_edition = {}
    for headword in all_headwords:
        editions_with_article = []
        for year in years:
            if headword in editions_data[year]:
                editions_with_article.append({
                    'year': year,
                    'original_headword': editions_data[year][headword]['original_headword'],
                    'text_length': editions_data[year][headword]['text_length'],
                    'volume': editions_data[year][headword]['volume']
                })

        if len(editions_with_article) >= 2:
            cross_edition[headword] = {
                'editions': [e['year'] for e in editions_with_article],
                'count': len(editions_with_article),
                'details': editions_with_article
            }

    return cross_edition, editions_data


def main():
    # Configuration
    docs_path = Path(__file__).parent.parent / "docs"

    # Find available editions
    available_years = []
    for year_dir in sorted(docs_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            data_dir = year_dir / "data"
            if data_dir.exists() and list(data_dir.glob("vol*.json")):
                available_years.append(int(year_dir.name))

    print(f"Available editions: {available_years}")
    print("=" * 60)

    # Find cross-edition articles
    cross_edition, editions_data = find_cross_edition_articles(docs_path, available_years)

    # Statistics
    print("\n" + "=" * 60)
    print("CROSS-EDITION ARTICLE STATISTICS")
    print("=" * 60)

    # Count by number of editions
    by_edition_count = defaultdict(list)
    for headword, info in cross_edition.items():
        by_edition_count[info['count']].append(headword)

    print(f"\nArticles appearing in multiple editions:")
    for count in sorted(by_edition_count.keys(), reverse=True):
        articles = by_edition_count[count]
        print(f"  {count} editions: {len(articles):,} articles")

    print(f"\nTotal cross-edition articles: {len(cross_edition):,}")

    # Show some examples of articles in all editions
    max_editions = max(by_edition_count.keys()) if by_edition_count else 0
    if max_editions > 0:
        print(f"\nSample articles appearing in all {max_editions} editions:")
        sample = sorted(by_edition_count[max_editions])[:20]
        for headword in sample:
            info = cross_edition[headword]
            lengths = [f"{d['year']}:{d['text_length']:,}c" for d in info['details']]
            print(f"  {headword}: {', '.join(lengths)}")

    # Show articles with biggest growth across editions
    print(f"\nArticles with significant growth across editions:")
    growth_articles = []
    for headword, info in cross_edition.items():
        if info['count'] >= 2:
            lengths = [d['text_length'] for d in info['details']]
            if min(lengths) > 0:
                growth_ratio = max(lengths) / min(lengths)
                if growth_ratio > 2:
                    growth_articles.append((headword, growth_ratio, info))

    growth_articles.sort(key=lambda x: -x[1])
    for headword, ratio, info in growth_articles[:15]:
        lengths = [f"{d['year']}:{d['text_length']:,}c" for d in info['details']]
        print(f"  {headword} ({ratio:.1f}x growth): {', '.join(lengths)}")

    # Save results
    output_path = Path(__file__).parent.parent / "cross_edition_articles.json"

    # Create a cleaner output format
    output = {
        'metadata': {
            'editions_analyzed': available_years,
            'total_cross_edition_articles': len(cross_edition),
            'articles_per_edition': {year: len(data) for year, data in editions_data.items()}
        },
        'by_edition_count': {
            str(count): len(articles) for count, articles in by_edition_count.items()
        },
        'articles': {
            headword: {
                'editions': info['editions'],
                'count': info['count'],
                'details': info['details']
            }
            for headword, info in sorted(cross_edition.items())
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
