#!/usr/bin/env python3
"""
Detect alphabetic outliers in Encyclopedia Britannica articles.

Within each volume, articles should follow alphabetical order by page position.
Any article whose first letter breaks the alphabetical progression is likely
a parsing error (mid-article heading extracted as separate article).

These outliers should be merged back into the surrounding article.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def load_articles(jsonl_path: Path) -> list[dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            articles.append(json.loads(line))
    return articles

def detect_outliers_in_volume(articles: list[dict], edition_year: int, volume_num: int) -> list[dict]:
    """
    Detect articles that break alphabetical progression within a volume.

    Returns list of outlier articles with context about where they should merge.
    """
    # Sort by page number
    sorted_arts = sorted(articles, key=lambda a: (a.get('start_page', 0), a.get('headword', '')))

    if len(sorted_arts) < 3:
        return []

    outliers = []

    # Track the "expected" letter range as we progress through pages
    # We allow the letter to stay same or advance, but not go backwards significantly

    for i, art in enumerate(sorted_arts):
        headword = art.get('headword', '')
        if not headword:
            continue

        first_letter = headword[0].upper()
        start_page = art.get('start_page', 0)

        # Get surrounding articles for context
        prev_arts = sorted_arts[max(0, i-3):i]
        next_arts = sorted_arts[i+1:i+4]

        # Determine the expected letter range from neighbors
        neighbor_letters = []
        for neighbor in prev_arts + next_arts:
            hw = neighbor.get('headword', '')
            if hw:
                neighbor_letters.append(hw[0].upper())

        if not neighbor_letters:
            continue

        # Find the dominant letter range
        min_neighbor = min(neighbor_letters)
        max_neighbor = max(neighbor_letters)

        # Check if this article's letter is an outlier
        # An outlier is when the letter is significantly outside the neighbor range
        is_outlier = False
        reason = ""

        if first_letter < min_neighbor and ord(min_neighbor) - ord(first_letter) > 1:
            # Letter is before the range (e.g., 'A' when neighbors are all 'E')
            is_outlier = True
            reason = f"Letter '{first_letter}' appears before expected range '{min_neighbor}-{max_neighbor}'"
        elif first_letter > max_neighbor and ord(first_letter) - ord(max_neighbor) > 1:
            # Letter is after the range (e.g., 'S' when neighbors are all 'E')
            is_outlier = True
            reason = f"Letter '{first_letter}' appears after expected range '{min_neighbor}-{max_neighbor}'"

        if is_outlier:
            # Find merge candidates from BOTH previous and next articles
            # The outlier could be a subsection (merge into main after it)
            # Or a misparsed heading (merge into article before it)
            merge_candidates = []

            # Previous articles (outlier merges into earlier article)
            for prev in reversed(prev_arts):
                prev_page = prev.get('start_page', 0)
                merge_candidates.append({
                    'article_id': prev.get('article_id'),
                    'headword': prev.get('headword'),
                    'start_page': prev_page,
                    'end_page': prev.get('end_page', prev_page),
                    'word_count': prev.get('word_count', 0),
                    'direction': 'previous'
                })

            # Next articles (outlier is prefix of a later article)
            for nxt in next_arts:
                nxt_page = nxt.get('start_page', 0)
                merge_candidates.append({
                    'article_id': nxt.get('article_id'),
                    'headword': nxt.get('headword'),
                    'start_page': nxt_page,
                    'end_page': nxt.get('end_page', nxt_page),
                    'word_count': nxt.get('word_count', 0),
                    'direction': 'next'
                })

            outliers.append({
                'article_id': art.get('article_id'),
                'headword': headword,
                'edition_year': edition_year,
                'volume_num': volume_num,
                'start_page': start_page,
                'end_page': art.get('end_page', start_page),
                'word_count': art.get('word_count', 0),
                'first_letter': first_letter,
                'expected_range': f"{min_neighbor}-{max_neighbor}",
                'reason': reason,
                'text_preview': art.get('text', '')[:200],
                'merge_candidates': merge_candidates[:6],  # Up to 3 prev + 3 next
                'prev_articles': [
                    {'headword': a.get('headword'), 'start_page': a.get('start_page'), 'end_page': a.get('end_page')}
                    for a in prev_arts
                ],
                'next_articles': [
                    {'headword': a.get('headword'), 'start_page': a.get('start_page')}
                    for a in next_arts
                ]
            })

    return outliers

def analyze_edition(jsonl_path: Path) -> dict:
    """Analyze a single edition for alphabetic outliers."""
    articles = load_articles(jsonl_path)

    # Extract edition year from filename
    edition_year = int(jsonl_path.stem.split('_')[1])

    # Group by volume
    by_volume = defaultdict(list)
    for art in articles:
        vol = art.get('volume_num', 0)
        by_volume[vol].append(art)

    all_outliers = []
    for vol_num in sorted(by_volume.keys()):
        vol_articles = by_volume[vol_num]
        outliers = detect_outliers_in_volume(vol_articles, edition_year, vol_num)
        all_outliers.extend(outliers)

    return {
        'edition_year': edition_year,
        'total_articles': len(articles),
        'volumes': len(by_volume),
        'outliers': all_outliers
    }

def main():
    output_dir = Path('output_v2')

    all_results = []

    for jsonl_file in sorted(output_dir.glob('articles_*.jsonl')):
        print(f"Analyzing {jsonl_file.name}...")
        result = analyze_edition(jsonl_file)
        all_results.append(result)

        print(f"  {result['edition_year']}: {len(result['outliers'])} outliers in {result['total_articles']} articles")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Alphabetic Outliers by Edition")
    print("="*60)

    total_outliers = 0
    for result in all_results:
        year = result['edition_year']
        count = len(result['outliers'])
        total_outliers += count
        print(f"{year}: {count:4d} outliers")

        # Show first few outliers as examples
        for outlier in result['outliers'][:5]:
            print(f"    p.{outlier['start_page']:4d} | {outlier['headword'][:30]:30s} | {outlier['reason']}")
        if len(result['outliers']) > 5:
            print(f"    ... and {len(result['outliers']) - 5} more")

    print(f"\nTotal: {total_outliers} outliers across all editions")

    # Save detailed results
    output_file = Path('llm_corrections/outliers/alphabetic_outliers.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()
