#!/usr/bin/env python3
"""
Find parsing errors by detecting pages where articles have conflicting
first letters - indicating some articles are misplaced.

The key insight: on any given page of an encyclopedia, articles should
have similar starting letters. If page 406 has BE- articles but also
"FREE BENCH", that's a parsing error.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


def analyze_volume(articles: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Find pages with conflicting article prefixes.
    Returns: (conflict_pages, reparse_ranges)
    """
    # Group articles by start page
    page_articles = defaultdict(list)
    for i, a in enumerate(articles):
        sp = a.get('sp')
        h = a.get('h', '')
        if sp and h:
            first_letter = h[0].upper() if h[0].isalpha() else None
            page_articles[sp].append({
                'position': i,
                'title': h,
                'first_letter': first_letter,
                'sp': sp,
                'ep': a.get('ep') or sp
            })

    # Find pages with conflicting first letters
    conflicts = []
    for page in sorted(page_articles.keys()):
        items = page_articles[page]
        letters = [item['first_letter'] for item in items if item['first_letter']]

        if not letters:
            continue

        # Count letters and find the dominant one
        letter_counts = Counter(letters)
        dominant_letter, dominant_count = letter_counts.most_common(1)[0]

        # Find minority articles (different first letter)
        minority = [item for item in items
                   if item['first_letter'] and item['first_letter'] != dominant_letter]

        if minority:
            conflicts.append({
                'page': page,
                'dominant_letter': dominant_letter,
                'dominant_count': dominant_count,
                'minority_articles': minority,
                'all_articles': items
            })

    # Also check for articles that break alphabetical flow
    # (even if they're the only article on a page)
    flow_breaks = find_flow_breaks(articles, page_articles)

    # Combine conflicts with flow breaks
    all_errors = []
    error_pages = set()

    for c in conflicts:
        for m in c['minority_articles']:
            all_errors.append({
                'type': 'conflict',
                'page': c['page'],
                'title': m['title'],
                'first_letter': m['first_letter'],
                'expected': c['dominant_letter'],
                'position': m['position'],
                'ep': m['ep']
            })
            error_pages.add(c['page'])

    for fb in flow_breaks:
        if fb['page'] not in error_pages:
            all_errors.append(fb)

    # Group into reparse ranges
    ranges = group_into_ranges(all_errors)

    return all_errors, ranges


def find_flow_breaks(articles: List[dict], page_articles: Dict) -> List[dict]:
    """
    Find articles that break the alphabetical flow even if alone on a page.
    """
    breaks = []

    # Build expected flow from page ranges
    sorted_pages = sorted(page_articles.keys())
    if len(sorted_pages) < 10:
        return breaks

    # For each page, determine expected letter based on neighbors
    page_expected = {}
    for i, page in enumerate(sorted_pages):
        items = page_articles[page]
        letters = [item['first_letter'] for item in items if item['first_letter']]
        if letters:
            page_expected[page] = Counter(letters).most_common(1)[0][0]

    # Now find pages that break the flow
    for i, page in enumerate(sorted_pages):
        if page not in page_expected:
            continue

        current = page_expected[page]

        # Get nearby expected letters
        nearby = []
        for j in range(max(0, i-3), min(len(sorted_pages), i+4)):
            if j != i:
                neighbor_page = sorted_pages[j]
                if neighbor_page in page_expected:
                    nearby.append(page_expected[neighbor_page])

        if not nearby:
            continue

        # If current letter is very different from neighbors, it's suspicious
        nearby_counter = Counter(nearby)
        expected, _ = nearby_counter.most_common(1)[0]

        # Check if current is far from expected (allowing for natural progression)
        if current != expected:
            # Allow A->B, B->C etc progression
            if abs(ord(current) - ord(expected)) > 2:
                # This page seems wrong
                for item in page_articles[page]:
                    if item['first_letter'] == current:
                        breaks.append({
                            'type': 'flow_break',
                            'page': page,
                            'title': item['title'],
                            'first_letter': current,
                            'expected': expected,
                            'position': item['position'],
                            'ep': item['ep']
                        })

    return breaks


def group_into_ranges(errors: List[dict], context: int = 5) -> List[dict]:
    """Group errors into page ranges for re-parsing."""
    if not errors:
        return []

    ranges = []
    for e in errors:
        start = max(1, e['page'] - context)
        end = e['ep'] + context
        ranges.append({
            'start': start,
            'end': end,
            'errors': [e]
        })

    # Sort and merge
    ranges.sort(key=lambda x: x['start'])
    merged = []

    for r in ranges:
        if merged and r['start'] <= merged[-1]['end'] + 3:
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
            merged[-1]['errors'].extend(r['errors'])
        else:
            merged.append(r)

    return merged


def main():
    import sys

    target_edition = sys.argv[1] if len(sys.argv) > 1 else None

    docs_dir = Path('docs')
    all_results = []

    editions = [
        ('1771', '1st Edition'),
        ('1778', '2nd Edition'),
        ('1797', '3rd Edition'),
        ('1810', '4th Edition'),
        ('1815', '5th Edition'),
        ('1823', '6th Edition'),
        ('1842', '7th Edition'),
        ('1853', '8th Edition'),
        ('1860', '8th Edition Alt'),
    ]

    for year, name in editions:
        if target_edition and year != target_edition:
            continue

        data_dir = docs_dir / year / 'data'
        if not data_dir.exists():
            continue

        print(f"\n{'=' * 70}")
        print(f"{name} ({year})")
        print('=' * 70)

        edition_errors = 0
        edition_ranges = []

        for json_file in sorted(data_dir.glob('vol*.json')):
            vol = json_file.stem

            # Skip vol0 for pre-1842
            if vol == 'vol0' and year < '1842':
                continue
            # Skip split files
            if '_main' in vol or '_supplement' in vol:
                continue

            with open(json_file, 'r') as f:
                articles = json.load(f)

            errors, ranges = analyze_volume(articles)

            if errors:
                print(f"\n{vol}: {len(errors)} errors -> {len(ranges)} reparse ranges")

                # Group by type
                conflicts = [e for e in errors if e['type'] == 'conflict']
                flow_breaks = [e for e in errors if e['type'] == 'flow_break']

                if conflicts:
                    print(f"  Conflicts (wrong letter on page):")
                    for e in conflicts[:5]:
                        print(f"    p.{e['page']}: \"{e['title'][:40]}\" ({e['first_letter']} != {e['expected']})")
                    if len(conflicts) > 5:
                        print(f"    ... and {len(conflicts) - 5} more")

                if flow_breaks:
                    print(f"  Flow breaks (breaks alphabetical sequence):")
                    for e in flow_breaks[:5]:
                        print(f"    p.{e['page']}: \"{e['title'][:40]}\" ({e['first_letter']} != {e['expected']})")
                    if len(flow_breaks) > 5:
                        print(f"    ... and {len(flow_breaks) - 5} more")

                print(f"\n  Reparse ranges:")
                for r in ranges[:5]:
                    titles = [e['title'][:25] for e in r['errors'][:2]]
                    print(f"    Pages {r['start']}-{r['end']}: {len(r['errors'])} errors")

                edition_errors += len(errors)
                for r in ranges:
                    edition_ranges.append({
                        'edition': year,
                        'volume': vol,
                        'start_page': r['start'],
                        'end_page': r['end'],
                        'error_count': len(r['errors']),
                        'sample_errors': [{'title': e['title'][:50], 'page': e['page']}
                                         for e in r['errors'][:3]]
                    })

        all_results.extend(edition_ranges)
        print(f"\n{name} Total: {edition_errors} errors, {len(edition_ranges)} ranges")

    # Save
    output = Path('reparse_ranges.json')
    with open(output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(all_results)} page ranges to re-parse")
    total_pages = sum(r['end_page'] - r['start_page'] + 1 for r in all_results)
    print(f"Total pages: ~{total_pages}")
    print(f"Saved to: {output}")


if __name__ == '__main__':
    main()
