#!/usr/bin/env python3
"""
Audit script for 1860 (8th Edition) Encyclopedia Britannica articles
Identifies quality issues including:
- Articles outside alphabetical range
- Unusually short articles
- Unusually long articles
- Duplicate articles
- Large alphabetical jumps
- OCR/parsing errors
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/docs/1860/data")
REPORT_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/reports")

# Volume ranges from index.html
VOLUME_RANGES = {
    0: (None, None),  # Volume 0 - unclear range
    1: ("Dissertations", None),  # Dissertations - special volume
    2: ("A", "ANATOMY"),
    3: ("ANATOMY", "ASTRONOMY"),
    4: ("ASTRONOMY", "BOM"),
    5: ("BOMBAY", "BUR"),
    6: ("BURNING", None),  # Volume 6 range unclear from title
    7: ("CLI", "DIA"),
    8: ("DIAMOND", "ENTAIL"),
    9: ("ENTOMOLOGY", "FRA"),
    10: ("FRANCE", "GRA"),
    11: ("GRA", "HUM"),
    12: ("HUME", "JOM"),
    13: ("JONAH", "MAG"),
    14: ("MAGNETISM", "MIH"),
    15: ("MILAN", "NAV"),
    16: ("NAVIGATION", "ORNITHOLOGY"),
    17: ("ORO", "PLATO"),
    18: ("PLA", "REI"),
    19: ("REID", "SCYTHIA"),
    20: ("SEAMANSHIP", "SZO"),
    21: ("T", "ZWO"),
}

def load_all_articles():
    """Load all articles from all volume JSON files."""
    all_articles = {}
    for i in range(22):
        json_path = DATA_DIR / f"vol{i}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
                    all_articles[i] = articles
                    print(f"Loaded vol{i}.json: {len(articles)} articles")
            except Exception as e:
                print(f"Error loading vol{i}.json: {e}")
    return all_articles

def strip_html(text):
    """Remove HTML tags from text to get plain text length."""
    return re.sub(r'<[^>]+>', '', text)

def is_valid_headword(headword):
    """Check if headword looks like valid text (not OCR garbage)."""
    # Check for excessive special characters
    special_count = len(re.findall(r'[^A-Za-z0-9\s\-\',\(\)]', headword))
    if len(headword) > 0 and special_count / len(headword) > 0.3:
        return False
    # Check for likely OCR errors (strings of random characters)
    if re.search(r'[^aeiouAEIOU]{8,}', headword):  # 8+ consonants in a row
        return False
    return True

def get_first_letter(headword):
    """Get first alphabetic letter of headword for range checking."""
    for char in headword.upper():
        if char.isalpha():
            return char
    return None

def check_alphabetical_range(vol_num, headword, ranges):
    """Check if article belongs in volume's alphabetical range."""
    if vol_num in [0, 1]:  # Skip special volumes
        return True

    start, end = ranges.get(vol_num, (None, None))
    if not start:
        return True

    hw_upper = headword.upper()
    first_letter = get_first_letter(headword)

    if not first_letter:
        return True  # Can't determine

    start_letter = start[0] if start else None
    end_letter = end[0] if end else None

    # Basic check: first letter should be within range
    if start_letter and end_letter:
        if first_letter < start_letter or first_letter > end_letter:
            return False
    elif start_letter:
        if first_letter < start_letter:
            return False

    return True

def analyze_articles(all_articles):
    """Perform full analysis on all articles."""
    issues = {
        'outside_range': [],
        'too_short': [],
        'too_long': [],
        'duplicates': defaultdict(list),
        'alphabetical_jumps': [],
        'ocr_errors': [],
    }

    all_headwords = defaultdict(list)  # For duplicate detection

    for vol_num, articles in all_articles.items():
        prev_headword = None

        for idx, article in enumerate(articles):
            headword = article.get('h', '')
            text = article.get('t', '')
            start_page = article.get('sp', 0)
            end_page = article.get('ep', 0)

            article_id = f"vol{vol_num}:idx{idx}"

            # Strip HTML to get actual text length
            plain_text = strip_html(text)
            text_len = len(plain_text)

            # Track for duplicate detection
            all_headwords[headword.upper()].append({
                'vol': vol_num,
                'idx': idx,
                'headword': headword,
                'pages': f"{start_page}-{end_page}",
                'text_len': text_len
            })

            # Check 1: Outside alphabetical range
            if not check_alphabetical_range(vol_num, headword, VOLUME_RANGES):
                issues['outside_range'].append({
                    'id': article_id,
                    'headword': headword,
                    'volume': vol_num,
                    'expected_range': VOLUME_RANGES.get(vol_num, (None, None)),
                    'pages': f"{start_page}-{end_page}"
                })

            # Check 2: Too short (under 50 chars of actual content)
            if text_len < 50:
                issues['too_short'].append({
                    'id': article_id,
                    'headword': headword,
                    'volume': vol_num,
                    'text_len': text_len,
                    'text_preview': plain_text[:100],
                    'pages': f"{start_page}-{end_page}"
                })

            # Check 3: Too long (over 50000 chars - likely merged)
            if text_len > 50000:
                issues['too_long'].append({
                    'id': article_id,
                    'headword': headword,
                    'volume': vol_num,
                    'text_len': text_len,
                    'word_count': len(plain_text.split()),
                    'pages': f"{start_page}-{end_page}"
                })

            # Check 4: OCR errors in headwords
            if not is_valid_headword(headword):
                issues['ocr_errors'].append({
                    'id': article_id,
                    'headword': headword,
                    'volume': vol_num,
                    'pages': f"{start_page}-{end_page}"
                })

            # Check 5: Alphabetical jumps (within same volume, excluding vol 0 and 1)
            if vol_num not in [0, 1] and prev_headword and headword:
                prev_first = get_first_letter(prev_headword)
                curr_first = get_first_letter(headword)

                if prev_first and curr_first:
                    # Check if letters jump significantly
                    if ord(curr_first) - ord(prev_first) > 2:
                        issues['alphabetical_jumps'].append({
                            'volume': vol_num,
                            'prev_article': prev_headword,
                            'next_article': headword,
                            'prev_idx': idx - 1,
                            'next_idx': idx,
                            'letter_jump': f"{prev_first} -> {curr_first}"
                        })

            prev_headword = headword

    # Find duplicates (same headword appearing multiple times)
    for hw, occurrences in all_headwords.items():
        if len(occurrences) > 1:
            issues['duplicates'][hw] = occurrences

    return issues

def generate_report(issues, all_articles):
    """Generate markdown report of all issues found."""

    total_articles = sum(len(arts) for arts in all_articles.values())

    report = []
    report.append("# 1860 (8th Edition) Encyclopedia Britannica Quality Audit Report")
    report.append("")
    report.append(f"**Generated:** 2026-01-03")
    report.append(f"**Total Articles Analyzed:** {total_articles:,}")
    report.append(f"**Volumes Analyzed:** {len(all_articles)}")
    report.append("")
    report.append("---")
    report.append("")

    # Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("| Issue Category | Count | Severity |")
    report.append("|----------------|-------|----------|")
    report.append(f"| Articles Outside Alphabetical Range | {len(issues['outside_range'])} | HIGH |")
    report.append(f"| Unusually Short Articles (<50 chars) | {len(issues['too_short'])} | MEDIUM |")
    report.append(f"| Unusually Long Articles (>50K chars) | {len(issues['too_long'])} | MEDIUM |")
    report.append(f"| Duplicate Headwords | {len(issues['duplicates'])} | LOW |")
    report.append(f"| Large Alphabetical Jumps | {len(issues['alphabetical_jumps'])} | LOW |")
    report.append(f"| OCR/Parsing Errors in Headwords | {len(issues['ocr_errors'])} | HIGH |")
    report.append("")
    report.append("---")
    report.append("")

    # Section 1: Outside Range
    report.append("## 1. Articles Outside Alphabetical Range")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("These articles appear in volumes that do not cover their alphabetical range, suggesting misplacement or parsing errors.")
    report.append("")

    if issues['outside_range']:
        report.append("| Article ID | Headword | Volume | Expected Range | Pages |")
        report.append("|------------|----------|--------|----------------|-------|")
        for item in issues['outside_range'][:100]:  # Limit to 100
            expected = item['expected_range']
            range_str = f"{expected[0] or '?'} - {expected[1] or '?'}"
            report.append(f"| {item['id']} | {item['headword'][:40]} | {item['volume']} | {range_str} | {item['pages']} |")
        if len(issues['outside_range']) > 100:
            report.append(f"\n*...and {len(issues['outside_range']) - 100} more*")
    else:
        report.append("*No issues found in this category.*")
    report.append("")

    # Section 2: Short Articles
    report.append("## 2. Unusually Short Articles")
    report.append("")
    report.append("**Severity: MEDIUM**")
    report.append("")
    report.append("Articles with less than 50 characters of content may indicate parsing errors, incomplete OCR, or extraction issues.")
    report.append("")

    if issues['too_short']:
        report.append("| Article ID | Headword | Volume | Length | Preview |")
        report.append("|------------|----------|--------|--------|---------|")
        for item in issues['too_short'][:50]:
            preview = item['text_preview'][:50].replace('|', ' ').replace('\n', ' ')
            report.append(f"| {item['id']} | {item['headword'][:30]} | {item['volume']} | {item['text_len']} | {preview} |")
        if len(issues['too_short']) > 50:
            report.append(f"\n*...and {len(issues['too_short']) - 50} more*")
    else:
        report.append("*No issues found in this category.*")
    report.append("")

    # Section 3: Long Articles
    report.append("## 3. Unusually Long Articles")
    report.append("")
    report.append("**Severity: MEDIUM**")
    report.append("")
    report.append("Articles over 50,000 characters may contain multiple merged articles or incomplete parsing.")
    report.append("")

    if issues['too_long']:
        report.append("| Article ID | Headword | Volume | Characters | Word Count | Pages |")
        report.append("|------------|----------|--------|------------|------------|-------|")
        for item in sorted(issues['too_long'], key=lambda x: -x['text_len']):
            report.append(f"| {item['id']} | {item['headword'][:40]} | {item['volume']} | {item['text_len']:,} | {item['word_count']:,} | {item['pages']} |")
    else:
        report.append("*No issues found in this category.*")
    report.append("")

    # Section 4: Duplicates
    report.append("## 4. Duplicate Headwords")
    report.append("")
    report.append("**Severity: LOW**")
    report.append("")
    report.append("Same headword appearing multiple times. Some may be legitimate (same term in different contexts), others may be parsing errors.")
    report.append("")

    if issues['duplicates']:
        # Sort by number of occurrences
        sorted_dups = sorted(issues['duplicates'].items(), key=lambda x: -len(x[1]))
        report.append(f"Found {len(issues['duplicates'])} headwords with duplicates:")
        report.append("")

        for hw, occurrences in sorted_dups[:30]:
            report.append(f"### {hw} ({len(occurrences)} occurrences)")
            report.append("")
            for occ in occurrences:
                report.append(f"- Volume {occ['vol']}, Index {occ['idx']}: Pages {occ['pages']}, {occ['text_len']:,} chars")
            report.append("")

        if len(sorted_dups) > 30:
            report.append(f"*...and {len(sorted_dups) - 30} more duplicate headwords*")
    else:
        report.append("*No duplicates found.*")
    report.append("")

    # Section 5: Alphabetical Jumps
    report.append("## 5. Large Alphabetical Jumps")
    report.append("")
    report.append("**Severity: LOW**")
    report.append("")
    report.append("Significant gaps in alphabetical sequence may indicate missing articles or extraction issues.")
    report.append("")

    if issues['alphabetical_jumps']:
        report.append("| Volume | Previous Article | Next Article | Letter Jump |")
        report.append("|--------|------------------|--------------|-------------|")
        for item in issues['alphabetical_jumps'][:50]:
            report.append(f"| {item['volume']} | {item['prev_article'][:30]} | {item['next_article'][:30]} | {item['letter_jump']} |")
        if len(issues['alphabetical_jumps']) > 50:
            report.append(f"\n*...and {len(issues['alphabetical_jumps']) - 50} more*")
    else:
        report.append("*No significant alphabetical jumps found.*")
    report.append("")

    # Section 6: OCR Errors
    report.append("## 6. OCR/Parsing Errors in Headwords")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("Headwords containing unusual character patterns suggesting OCR errors or parsing problems.")
    report.append("")

    if issues['ocr_errors']:
        report.append("| Article ID | Headword | Volume | Pages |")
        report.append("|------------|----------|--------|-------|")
        for item in issues['ocr_errors'][:50]:
            hw_escaped = item['headword'].replace('|', ' ')[:40]
            report.append(f"| {item['id']} | {hw_escaped} | {item['volume']} | {item['pages']} |")
        if len(issues['ocr_errors']) > 50:
            report.append(f"\n*...and {len(issues['ocr_errors']) - 50} more*")
    else:
        report.append("*No OCR errors detected in headwords.*")
    report.append("")

    # Recommendations
    report.append("---")
    report.append("")
    report.append("## Recommendations")
    report.append("")
    report.append("1. **HIGH Priority**: Review articles flagged as outside alphabetical range - these may be misplaced or indicate volume boundary issues.")
    report.append("2. **HIGH Priority**: Investigate OCR error headwords - these likely need manual correction or re-processing.")
    report.append("3. **MEDIUM Priority**: Review short articles to determine if they are legitimate brief entries or extraction failures.")
    report.append("4. **MEDIUM Priority**: Check long articles for potential merged content that should be split.")
    report.append("5. **LOW Priority**: Review duplicate headwords to distinguish legitimate duplicates from parsing errors.")
    report.append("")

    return "\n".join(report)

def main():
    print("Loading all articles...")
    all_articles = load_all_articles()

    print("\nAnalyzing articles...")
    issues = analyze_articles(all_articles)

    print("\nGenerating report...")
    report = generate_report(issues, all_articles)

    # Ensure reports directory exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "audit_1860_8th_edition.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Articles outside range: {len(issues['outside_range'])}")
    print(f"Too short articles: {len(issues['too_short'])}")
    print(f"Too long articles: {len(issues['too_long'])}")
    print(f"Duplicate headwords: {len(issues['duplicates'])}")
    print(f"Alphabetical jumps: {len(issues['alphabetical_jumps'])}")
    print(f"OCR errors: {len(issues['ocr_errors'])}")

if __name__ == "__main__":
    main()
