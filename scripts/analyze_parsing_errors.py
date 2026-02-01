#!/usr/bin/env python3
"""
Analyze single-edition article titles to identify parsing errors.
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path


def classify_headword(headword: str, text_length: int) -> list[str]:
    """
    Classify a headword and return list of error types detected.
    """
    errors = []

    if not headword:
        errors.append("empty_headword")
        return errors

    # 1. Sentence fragments - contains common sentence words
    sentence_words = ['THE', 'THIS', 'THAT', 'WHEN', 'WHILE', 'WHICH', 'WHERE',
                      'HAVING', 'BEING', 'THESE', 'THOSE', 'WITH', 'FROM', 'INTO',
                      'UPON', 'AFTER', 'BEFORE', 'DURING', 'BETWEEN', 'THROUGH',
                      'HOWEVER', 'THEREFORE', 'MOREOVER', 'FURTHERMORE', 'HENCE',
                      'THUS', 'ALSO', 'ONLY', 'VERY', 'MOST', 'SUCH', 'OTHER',
                      'BESIDES', 'INDEED', 'NOW', 'THEN']

    upper_hw = headword.upper()
    words = upper_hw.split()

    # Check if starts with sentence word (not "THE X" as a title)
    if len(words) > 2 and words[0] in sentence_words:
        if not (words[0] == 'THE' and len(words) <= 4):  # Allow "THE HAGUE" etc
            errors.append("starts_with_sentence_word")

    # Check for sentence words in middle of headword
    if len(words) > 3:
        middle_words = words[1:-1]
        sentence_in_middle = sum(1 for w in middle_words if w in sentence_words)
        if sentence_in_middle >= 2:
            errors.append("sentence_fragment")

    # 2. Too long headword (likely a sentence)
    if len(headword) > 50:
        errors.append("headword_too_long")

    if len(headword) > 100:
        errors.append("headword_extremely_long")

    # 3. Contains lowercase (unusual for encyclopedia headwords)
    if re.search(r'[a-z]{3,}', headword) and not headword[0].islower():
        # Has lowercase words but doesn't start lowercase
        # This might be a sentence like "WHEN the doctor..."
        if len(words) > 3:
            errors.append("mixed_case_long")

    # 4. Ends with preposition or article (sentence fragment)
    ending_words = ['THE', 'A', 'AN', 'OF', 'TO', 'BY', 'IN', 'ON', 'AT', 'FOR',
                    'WITH', 'FROM', 'AS', 'IS', 'WAS', 'ARE', 'WERE', 'BE', 'BEEN',
                    'THAT', 'WHICH', 'WHO', 'WHOM']
    if words and words[-1] in ending_words:
        errors.append("ends_with_function_word")

    # 5. Contains punctuation that suggests sentence
    if re.search(r'[,;:].*[,;:]', headword):  # Multiple punctuation marks
        errors.append("multiple_punctuation")

    # 6. Suspiciously large text for a single article
    if text_length > 100000:
        errors.append("text_over_100k")
    if text_length > 200000:
        errors.append("text_over_200k")
    if text_length > 500000:
        errors.append("text_over_500k")

    # 7. Looks like "END OF VOLUME" or similar
    if 'END OF' in upper_hw or 'VOLUME' in upper_hw or 'FINIS' in upper_hw:
        errors.append("volume_marker")

    # 8. Contains numbers in unusual way
    if re.search(r'\d{4,}', headword):  # 4+ digit number
        errors.append("contains_long_number")

    # 9. Starts with number
    if headword and headword[0].isdigit():
        errors.append("starts_with_number")

    # 10. All caps very long
    if headword.isupper() and len(headword) > 30:
        errors.append("all_caps_long")

    # 11. Contains "See" pattern suggesting cross-reference got parsed as headword
    if re.search(r'\bSEE\b', upper_hw) and len(words) > 2:
        errors.append("contains_see_reference")

    # 12. Genus/species pattern that's too long (GENUS SOMETHING SOMETHING...)
    if words and words[0] == 'GENUS' and len(words) > 3:
        errors.append("genus_fragment")

    return errors


def normalize_headword(headword: str) -> str:
    """Normalize headword for comparison."""
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
                        'text_length': len(article.get('t', '')),
                        'volume': json_file.stem
                    }
        except Exception as e:
            pass

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

    print("PARSING ERROR ANALYSIS FOR SINGLE-EDITION ARTICLES")
    print("=" * 80)

    # Load all editions
    editions_data = {}
    for year in available_years:
        editions_data[year] = load_edition(year, docs_path)

    # Build headword index
    headword_editions = defaultdict(set)
    for year, articles in editions_data.items():
        for headword in articles.keys():
            headword_editions[headword].add(year)

    # Find single-edition articles and classify them
    all_errors = defaultdict(list)  # error_type -> [(year, headword, text_length)]
    by_edition = defaultdict(lambda: {'total': 0, 'errors': defaultdict(int), 'clean': 0})
    error_examples = defaultdict(list)

    for headword, editions in headword_editions.items():
        if len(editions) == 1:
            year = list(editions)[0]
            info = editions_data[year][headword]
            original = info['original_headword']
            text_length = info['text_length']

            errors = classify_headword(original, text_length)

            by_edition[year]['total'] += 1

            if errors:
                for err in errors:
                    all_errors[err].append((year, original, text_length))
                    by_edition[year]['errors'][err] += 1
                    if len(error_examples[err]) < 5:
                        error_examples[err].append((year, original, text_length))
            else:
                by_edition[year]['clean'] += 1

    # Summary by error type
    print("\nERROR TYPE SUMMARY")
    print("-" * 80)
    print(f"{'Error Type':<35} {'Count':>8} {'Description'}")
    print("-" * 80)

    error_descriptions = {
        'starts_with_sentence_word': 'Starts with THE, WHEN, WHILE, etc.',
        'sentence_fragment': 'Contains multiple sentence words',
        'headword_too_long': 'Headword > 50 characters',
        'headword_extremely_long': 'Headword > 100 characters',
        'mixed_case_long': 'Mixed case with many words',
        'ends_with_function_word': 'Ends with OF, THE, BY, etc.',
        'multiple_punctuation': 'Contains multiple punctuation marks',
        'text_over_100k': 'Article text > 100K chars (likely merged)',
        'text_over_200k': 'Article text > 200K chars',
        'text_over_500k': 'Article text > 500K chars',
        'volume_marker': 'Contains END OF or VOLUME',
        'contains_long_number': 'Contains 4+ digit number',
        'starts_with_number': 'Starts with a number',
        'all_caps_long': 'All caps and > 30 characters',
        'contains_see_reference': 'Contains SEE (cross-ref as headword)',
        'genus_fragment': 'GENUS followed by long text',
        'empty_headword': 'Empty headword',
    }

    sorted_errors = sorted(all_errors.items(), key=lambda x: -len(x[1]))
    for err_type, instances in sorted_errors:
        desc = error_descriptions.get(err_type, '')
        print(f"{err_type:<35} {len(instances):>8} {desc}")

    # Calculate totals
    total_single = sum(by_edition[y]['total'] for y in available_years)

    # Count unique articles with ANY error
    articles_with_errors = set()
    for err_type, instances in all_errors.items():
        for year, hw, _ in instances:
            articles_with_errors.add((year, hw))

    total_with_errors = len(articles_with_errors)
    total_clean = total_single - total_with_errors

    print("-" * 80)
    print(f"\nOVERALL SUMMARY")
    print(f"  Total single-edition articles: {total_single:,}")
    print(f"  Articles with parsing errors:  {total_with_errors:,} ({100*total_with_errors/total_single:.1f}%)")
    print(f"  Clean articles:                {total_clean:,} ({100*total_clean/total_single:.1f}%)")

    # By edition breakdown
    print("\n" + "=" * 80)
    print("BREAKDOWN BY EDITION")
    print("-" * 80)
    print(f"{'Edition':<8} {'Total':>8} {'Errors':>8} {'Clean':>8} {'% Error':>10}")
    print("-" * 80)

    for year in available_years:
        total = by_edition[year]['total']
        # Count unique headwords with errors for this year
        year_errors = set()
        for err_type, instances in all_errors.items():
            for y, hw, _ in instances:
                if y == year:
                    year_errors.add(hw)
        error_count = len(year_errors)
        clean = total - error_count
        pct = 100 * error_count / total if total > 0 else 0
        print(f"{year:<8} {total:>8,} {error_count:>8,} {clean:>8,} {pct:>9.1f}%")

    # Examples of each error type
    print("\n" + "=" * 80)
    print("EXAMPLES OF EACH ERROR TYPE")
    print("=" * 80)

    for err_type, instances in sorted_errors[:12]:  # Top 12 error types
        print(f"\n{err_type} ({len(instances)} instances):")
        for year, hw, length in error_examples[err_type]:
            hw_display = hw[:70] + "..." if len(hw) > 70 else hw
            print(f"  {year}: {hw_display} ({length:,}c)")

    # Categorize errors into severity levels
    print("\n" + "=" * 80)
    print("ERROR SEVERITY CLASSIFICATION")
    print("=" * 80)

    definite_errors = set()  # Clearly parsing errors
    probable_errors = set()  # Likely parsing errors
    possible_errors = set()  # Might be errors

    definite_error_types = ['sentence_fragment', 'headword_extremely_long',
                           'text_over_200k', 'volume_marker', 'starts_with_sentence_word',
                           'ends_with_function_word']
    probable_error_types = ['headword_too_long', 'text_over_100k', 'all_caps_long',
                           'genus_fragment']
    possible_error_types = ['mixed_case_long', 'multiple_punctuation',
                           'contains_see_reference']

    for err_type, instances in all_errors.items():
        for year, hw, length in instances:
            key = (year, hw)
            if err_type in definite_error_types:
                definite_errors.add(key)
            elif err_type in probable_error_types and key not in definite_errors:
                probable_errors.add(key)
            elif err_type in possible_error_types and key not in definite_errors and key not in probable_errors:
                possible_errors.add(key)

    # Remove overlaps
    probable_errors -= definite_errors
    possible_errors -= definite_errors
    possible_errors -= probable_errors

    print(f"\n  DEFINITE parsing errors:  {len(definite_errors):>6,} (sentence fragments, huge articles, etc.)")
    print(f"  PROBABLE parsing errors:  {len(probable_errors):>6,} (long headwords, 100K+ articles)")
    print(f"  POSSIBLE parsing errors:  {len(possible_errors):>6,} (mixed case, punctuation issues)")
    print(f"  ---")
    total_flagged = len(definite_errors) + len(probable_errors) + len(possible_errors)
    print(f"  Total flagged:            {total_flagged:>6,}")
    print(f"  Likely clean:             {total_single - total_flagged:>6,}")

    # Save detailed results
    output = {
        'summary': {
            'total_single_edition': total_single,
            'total_with_any_error': total_with_errors,
            'total_clean': total_clean,
            'definite_errors': len(definite_errors),
            'probable_errors': len(probable_errors),
            'possible_errors': len(possible_errors),
        },
        'by_error_type': {
            err_type: {
                'count': len(instances),
                'examples': [(y, h, l) for y, h, l in instances[:20]]
            }
            for err_type, instances in sorted_errors
        },
        'definite_errors': [
            {'year': y, 'headword': h}
            for y, h in sorted(definite_errors, key=lambda x: (x[0], x[1]))
        ],
        'probable_errors': [
            {'year': y, 'headword': h}
            for y, h in sorted(probable_errors, key=lambda x: (x[0], x[1]))
        ]
    }

    output_path = Path(__file__).parent.parent / "parsing_error_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
