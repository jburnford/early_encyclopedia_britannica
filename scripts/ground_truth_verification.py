#!/usr/bin/env python3
"""
Ground truth verification: Read random sample of "clean" single-edition articles.
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path


def normalize_headword(headword: str) -> str:
    if not headword:
        return ""
    h = headword.lower().strip()
    h = re.sub(r'\s+', ' ', h)
    h = re.sub(r'[.,;:]+$', '', h)
    return h


def has_parsing_error(headword: str, text_length: int) -> bool:
    """Quick check for obvious parsing errors."""
    if not headword:
        return True

    upper_hw = headword.upper()
    words = upper_hw.split()

    sentence_words = ['THE', 'THIS', 'THAT', 'WHEN', 'WHILE', 'WHICH', 'WHERE',
                      'HAVING', 'BEING', 'THESE', 'THOSE', 'WITH', 'FROM', 'INTO']

    # Starts with sentence word (except short "THE X" titles)
    if len(words) > 2 and words[0] in sentence_words:
        if not (words[0] == 'THE' and len(words) <= 4):
            return True

    # Too long headword
    if len(headword) > 50:
        return True

    # Ends with function word
    ending_words = ['THE', 'A', 'AN', 'OF', 'TO', 'BY', 'IN', 'ON', 'AT', 'FOR', 'WITH', 'FROM', 'AS', 'IS']
    if words and words[-1] in ending_words:
        return True

    # Huge article (likely merged)
    if text_length > 100000:
        return True

    # Volume marker
    if 'END OF' in upper_hw or 'VOLUME' in upper_hw:
        return True

    return False


def load_edition_with_text(year: int, docs_path: Path) -> dict:
    """Load all articles including full text."""
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
                text = article.get('t', '')

                if normalized and normalized not in articles:
                    articles[normalized] = {
                        'original_headword': headword,
                        'text': text,
                        'text_length': len(text),
                        'volume': json_file.stem,
                        'start_page': article.get('sp'),
                        'end_page': article.get('ep')
                    }
        except Exception as e:
            pass

    return articles


def main():
    random.seed(42)  # For reproducibility

    docs_path = Path(__file__).parent.parent / "docs"

    # Find available editions
    available_years = []
    for year_dir in sorted(docs_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            data_dir = year_dir / "data"
            if data_dir.exists() and list(data_dir.glob("vol*.json")):
                available_years.append(int(year_dir.name))

    print("GROUND TRUTH VERIFICATION: RANDOM SAMPLE OF 'CLEAN' SINGLE-EDITION ARTICLES")
    print("=" * 90)

    # Load all editions with full text
    print("\nLoading editions with full text...")
    editions_data = {}
    for year in available_years:
        editions_data[year] = load_edition_with_text(year, docs_path)
        print(f"  {year}: {len(editions_data[year]):,} articles")

    # Build headword index
    headword_editions = defaultdict(set)
    for year, articles in editions_data.items():
        for headword in articles.keys():
            headword_editions[headword].add(year)

    # Find clean single-edition articles
    clean_articles = []
    for headword, editions in headword_editions.items():
        if len(editions) == 1:
            year = list(editions)[0]
            info = editions_data[year][headword]

            if not has_parsing_error(info['original_headword'], info['text_length']):
                clean_articles.append((year, headword, info))

    print(f"\nTotal 'clean' single-edition articles: {len(clean_articles):,}")

    # Sample stratified by edition and length
    samples = []

    # Get samples from each edition
    for year in available_years:
        year_articles = [(y, h, i) for y, h, i in clean_articles if y == year]
        if not year_articles:
            continue

        # Sort by length and sample from different size ranges
        year_articles.sort(key=lambda x: x[2]['text_length'])

        n = len(year_articles)
        # Sample: shortest, 25th percentile, median, 75th percentile, longest
        indices = [0, n//4, n//2, 3*n//4, n-1]
        for idx in indices:
            if idx < len(year_articles):
                samples.append(year_articles[idx])

    # Also add some truly random samples
    random_samples = random.sample(clean_articles, min(15, len(clean_articles)))
    samples.extend(random_samples)

    # Remove duplicates
    seen = set()
    unique_samples = []
    for s in samples:
        key = (s[0], s[1])
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)

    # Shuffle for variety
    random.shuffle(unique_samples)

    # Display samples
    print(f"\nReviewing {len(unique_samples)} randomly selected articles:")
    print("=" * 90)

    results = []

    for i, (year, headword, info) in enumerate(unique_samples[:30], 1):
        text = info['text']
        original_hw = info['original_headword']
        length = info['text_length']
        volume = info['volume']

        print(f"\n{'─' * 90}")
        print(f"SAMPLE {i}: {original_hw}")
        print(f"Edition: {year} | Volume: {volume} | Length: {length:,} chars")
        print(f"{'─' * 90}")

        if length <= 500:
            # Show full article
            print(f"[FULL ARTICLE]")
            print(text)
        else:
            # Show first and last parts
            first_part = text[:400]
            last_part = text[-400:]

            print(f"[FIRST 400 CHARS]")
            print(first_part)
            print(f"\n[... {length - 800:,} chars omitted ...]")
            print(f"\n[LAST 400 CHARS]")
            print(last_part)

        # Quick assessment
        assessment = "LIKELY_VALID"
        issues = []

        # Check for signs this might not be a real article
        if text.startswith((' ', '\n', '\t')):
            issues.append("starts_with_whitespace")
        if not text.strip():
            issues.append("empty_text")
            assessment = "INVALID"
        if original_hw.lower() not in text.lower()[:500] and length > 100:
            # Headword not in first 500 chars - might be misaligned
            issues.append("headword_not_in_opening")
        if re.search(r'^[a-z]', text.strip()):
            issues.append("starts_lowercase")
        if text.count('\n\n') > 10 and length < 2000:
            issues.append("many_paragraph_breaks")

        # Check if text seems to be multiple articles
        # Look for patterns like "HEADWORD, ..." appearing mid-text
        potential_headers = re.findall(r'\n([A-Z]{3,}[A-Z\s,]+),?\s+(?:a|an|the|in|is|are|was|one)', text[200:])
        if len(potential_headers) > 2:
            issues.append(f"possible_merged_articles({len(potential_headers)} headers found)")
            assessment = "NEEDS_REVIEW"

        if issues:
            print(f"\n⚠️  ISSUES DETECTED: {', '.join(issues)}")
            assessment = "NEEDS_REVIEW" if assessment != "INVALID" else assessment

        print(f"\n📋 ASSESSMENT: {assessment}")

        results.append({
            'year': year,
            'headword': original_hw,
            'text_length': length,
            'assessment': assessment,
            'issues': issues
        })

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY OF GROUND TRUTH VERIFICATION")
    print("=" * 90)

    valid_count = sum(1 for r in results if r['assessment'] == 'LIKELY_VALID')
    review_count = sum(1 for r in results if r['assessment'] == 'NEEDS_REVIEW')
    invalid_count = sum(1 for r in results if r['assessment'] == 'INVALID')

    print(f"\n  LIKELY_VALID:  {valid_count:>3} ({100*valid_count/len(results):.1f}%)")
    print(f"  NEEDS_REVIEW:  {review_count:>3} ({100*review_count/len(results):.1f}%)")
    print(f"  INVALID:       {invalid_count:>3} ({100*invalid_count/len(results):.1f}%)")

    if review_count > 0:
        print(f"\nArticles needing review:")
        for r in results:
            if r['assessment'] == 'NEEDS_REVIEW':
                print(f"  {r['year']}: {r['headword']} - {', '.join(r['issues'])}")

    # Save results
    output = {
        'total_clean_articles': len(clean_articles),
        'samples_reviewed': len(results),
        'summary': {
            'likely_valid': valid_count,
            'needs_review': review_count,
            'invalid': invalid_count
        },
        'details': results
    }

    output_path = Path(__file__).parent.parent / "ground_truth_verification.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
