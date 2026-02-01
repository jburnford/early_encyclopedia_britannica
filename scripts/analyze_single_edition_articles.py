#!/usr/bin/env python3
"""
Analyze articles that appear in only one edition of Encyclopedia Britannica.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


def normalize_headword(headword: str) -> str:
    """Normalize headword for comparison across editions."""
    if not headword:
        return ""
    h = headword.lower().strip()
    h = re.sub(r'\s+', ' ', h)
    h = re.sub(r'[.,;:]+$', '', h)
    return h


def load_edition(year: int, docs_path: Path) -> dict:
    """Load all articles from an edition."""
    edition_path = docs_path / str(year) / "data"
    articles = {}

    if not edition_path.exists():
        return articles

    json_files = sorted(edition_path.glob("vol*.json"))
    main_files = [f for f in json_files if not ('_original' in f.name or '_corrected' in f.name)]

    for json_file in main_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for article in data:
                headword = article.get('h', '')
                normalized = normalize_headword(headword)

                if normalized and normalized not in articles:
                    articles[normalized] = {
                        'original_headword': headword,
                        'text': article.get('t', ''),
                        'text_length': len(article.get('t', '')),
                        'volume': json_file.stem
                    }
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    return articles


def main():
    docs_path = Path(__file__).parent.parent / "docs"

    # Find available editions
    available_years = []
    for year_dir in sorted(docs_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            data_dir = year_dir / "data"
            if data_dir.exists() and list(data_dir.glob("vol*.json")):
                available_years.append(int(year_dir.name))

    print(f"Analyzing editions: {available_years}")
    print("=" * 70)

    # Load all editions
    editions_data = {}
    for year in available_years:
        print(f"Loading {year}...")
        editions_data[year] = load_edition(year, docs_path)
        print(f"  {len(editions_data[year]):,} articles")

    # Build index of which editions have each headword
    headword_editions = defaultdict(set)
    for year, articles in editions_data.items():
        for headword in articles.keys():
            headword_editions[headword].add(year)

    # Find single-edition articles
    single_edition = defaultdict(list)
    for headword, editions in headword_editions.items():
        if len(editions) == 1:
            year = list(editions)[0]
            single_edition[year].append(headword)

    # Statistics
    print("\n" + "=" * 70)
    print("SINGLE-EDITION ARTICLE ANALYSIS")
    print("=" * 70)

    print("\nArticles unique to each edition:")
    print("-" * 70)
    print(f"{'Edition':<10} {'Unique':<10} {'Total':<10} {'% Unique':<10}")
    print("-" * 70)

    for year in available_years:
        unique_count = len(single_edition[year])
        total_count = len(editions_data[year])
        pct = (unique_count / total_count * 100) if total_count > 0 else 0
        print(f"{year:<10} {unique_count:<10,} {total_count:<10,} {pct:.1f}%")

    total_single = sum(len(v) for v in single_edition.values())
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_single:<10,}")

    # Analyze patterns in single-edition articles
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Length distribution of single-edition articles
    print("\nText length distribution of single-edition articles:")
    for year in available_years:
        lengths = [editions_data[year][h]['text_length'] for h in single_edition[year]]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            very_short = sum(1 for l in lengths if l < 100)
            short = sum(1 for l in lengths if 100 <= l < 500)
            medium = sum(1 for l in lengths if 500 <= l < 2000)
            long_articles = sum(1 for l in lengths if l >= 2000)
            print(f"  {year}: avg={avg_len:,.0f}c | <100c:{very_short} | 100-500c:{short} | 500-2K:{medium} | >2K:{long_articles}")

    # Sample unique articles from each edition
    print("\n" + "=" * 70)
    print("SAMPLE UNIQUE ARTICLES BY EDITION")
    print("=" * 70)

    for year in available_years:
        articles = single_edition[year]
        if not articles:
            continue

        # Sort by text length to show variety
        sorted_articles = sorted(articles, key=lambda h: editions_data[year][h]['text_length'], reverse=True)

        print(f"\n{year} Edition - {len(articles):,} unique articles:")

        # Show 5 longest
        print("  Longest unique articles:")
        for h in sorted_articles[:5]:
            info = editions_data[year][h]
            preview = info['text'][:80].replace('\n', ' ')
            print(f"    {info['original_headword']}: {info['text_length']:,}c - \"{preview}...\"")

        # Show 5 shortest (potential parsing errors?)
        print("  Shortest unique articles:")
        for h in sorted_articles[-5:]:
            info = editions_data[year][h]
            preview = info['text'][:80].replace('\n', ' ')
            print(f"    {info['original_headword']}: {info['text_length']:,}c - \"{preview}\"")

    # Look for patterns that might indicate parsing errors
    print("\n" + "=" * 70)
    print("POTENTIAL PARSING ANOMALIES")
    print("=" * 70)

    anomalies = {
        'all_caps_long': [],  # ALL CAPS headwords that are long
        'contains_the': [],   # Headwords containing "THE" (often parsing errors)
        'very_short_text': [], # Very short text (< 20 chars)
        'numeric_start': [],  # Starts with numbers
    }

    for year in available_years:
        for h in single_edition[year]:
            info = editions_data[year][h]
            original = info['original_headword']

            # Check for anomalies
            if original.isupper() and len(original) > 20:
                anomalies['all_caps_long'].append((year, original, info['text_length']))
            if ' THE ' in original.upper() or original.upper().startswith('THE '):
                anomalies['contains_the'].append((year, original, info['text_length']))
            if info['text_length'] < 20:
                anomalies['very_short_text'].append((year, original, info['text']))
            if original and original[0].isdigit():
                anomalies['numeric_start'].append((year, original, info['text_length']))

    print(f"\nHeadwords containing 'THE' (often parsing errors): {len(anomalies['contains_the'])}")
    for year, hw, length in anomalies['contains_the'][:10]:
        print(f"  {year}: {hw} ({length:,}c)")

    print(f"\nVery short text (<20 chars): {len(anomalies['very_short_text'])}")
    for year, hw, text in anomalies['very_short_text'][:10]:
        print(f"  {year}: {hw} -> \"{text}\"")

    print(f"\nHeadwords starting with numbers: {len(anomalies['numeric_start'])}")
    for year, hw, length in anomalies['numeric_start'][:10]:
        print(f"  {year}: {hw} ({length:,}c)")

    # First letter distribution
    print("\n" + "=" * 70)
    print("FIRST LETTER DISTRIBUTION OF UNIQUE ARTICLES")
    print("=" * 70)

    for year in available_years:
        first_letters = Counter()
        for h in single_edition[year]:
            original = editions_data[year][h]['original_headword']
            if original:
                first_letters[original[0].upper()] += 1

        top_5 = first_letters.most_common(5)
        bottom_5 = first_letters.most_common()[-5:] if len(first_letters) > 5 else []
        print(f"  {year}: Most common: {top_5} | Least common: {bottom_5}")

    # Save detailed results
    output = {
        'summary': {
            'total_single_edition': total_single,
            'by_edition': {year: len(single_edition[year]) for year in available_years}
        },
        'single_edition_articles': {
            str(year): [
                {
                    'headword': h,
                    'original': editions_data[year][h]['original_headword'],
                    'text_length': editions_data[year][h]['text_length']
                }
                for h in sorted(single_edition[year])
            ]
            for year in available_years
        },
        'anomalies': {
            'contains_the': [(y, h, l) for y, h, l in anomalies['contains_the']],
            'very_short_text': [(y, h, t) for y, h, t in anomalies['very_short_text']]
        }
    }

    output_path = Path(__file__).parent.parent / "single_edition_articles.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
