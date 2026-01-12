#!/usr/bin/env python3
"""
Refined Audit script for 1860 (8th Edition) Encyclopedia Britannica articles
More accurate categorization of issues
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/docs/1860/data")
REPORT_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/reports")

# Volume ranges from index.html
VOLUME_RANGES = {
    0: None,  # Volume 0 - unclear, appears to be index/reference
    1: None,  # Dissertations - special volume
    2: ("A", "ANATOMY"),
    3: ("ANATOMY", "ASTRONOMY"),
    4: ("ASTRONOMY", "BOM"),
    5: ("BOMBAY", "BUR"),
    6: ("BUR", "CLI"),  # Burning to CLI
    7: ("CLI", "DIA"),
    8: ("DIA", "ENT"),  # Diamond to Entail
    9: ("ENT", "FRA"),  # Entomology to FRA
    10: ("FRA", "GRA"),
    11: ("GRA", "HUM"),
    12: ("HUM", "JOM"),
    13: ("JON", "MAG"),
    14: ("MAG", "MIH"),
    15: ("MIL", "NAV"),
    16: ("NAV", "ORN"),
    17: ("ORO", "PLA"),
    18: ("PLA", "REI"),
    19: ("REI", "SCY"),
    20: ("SEA", "SZO"),
    21: ("T", "ZWO"),
}

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text)

def is_sentence_fragment(hw):
    """Detect headwords that are sentence fragments from OCR errors."""
    # Ends with incomplete words
    end_patterns = [
        r'\s+(BY|TO|OF|THE|A|AN|IN|WITH|FOR|FROM|AND|OR|IS|ARE|WAS|WERE|NO|THAT|THIS|THESE|WHICH|IT|MAY|BE)\s*$',
    ]
    for p in end_patterns:
        if re.search(p, hw, re.I):
            return True
    # Very long headwords (>60 chars) are usually sentence fragments
    if len(hw) > 60 and not hw.isupper():  # Allow long all-caps titles
        return True
    # Multiple spaces or line breaks
    if '\n' in hw or '  ' in hw:
        return True
    return False

def is_subsection_heading(hw):
    """Detect headwords that are subsection headings, not main articles."""
    subsection_patterns = [
        r'^(GENERAL|SPECIAL|HISTORICAL|PRACTICAL|THEORETICAL|PRELIMINARY)\s+(ANATOMY|REMARKS|OBSERVATIONS|TREATMENT|MANAGEMENT|DESCRIPTION|PHENOMENA)',
        r'^(PART|CHAPTER|SECTION|BOOK|DIVISION)\s+[IVXLCDM0-9]+',
        r'^(TABLE|CONTENTS|APPENDIX|INTRODUCTION|CONCLUSION)\s*(OF|I|II|III|IV|V)?$',
        r'^(DESCRIPTION|EXPLANATION|CLASSIFICATION|DEFINITION)\s+OF',
        r'^(HARVESTING|WEIGHING|WINNOWING|THRASHING)\s+(IMPLEMENTS|MACHINES)',
        r'^(INDEX|GLOSSARY)\s+OF',
        r'^DIVISIONS\s+OF\s+THE',
    ]
    for p in subsection_patterns:
        if re.search(p, hw, re.I):
            return True
    return False

def is_publisher_metadata(hw):
    """Detect publisher/editor information parsed as articles."""
    pub_patterns = [
        r'PUBLISHED\s+BY',
        r'EDITED\s+BY',
        r'PRINTED\s+BY',
        r'CHEAP\s+EDITIONS',
        r'CONTRIBUTORS\s+TO',
        r'CLOTH\b',
        r'\bPRICE\b',
        r'NEW\s+(WORKS|EDITION)',
        r'MAY\s+BE\s+HAD',
        r'LAST\s+EDITIONS',
        r'EXTRA\s+CLOTH',
    ]
    for p in pub_patterns:
        if re.search(p, hw, re.I):
            return True
    return False

def is_ocr_error(hw):
    """Detect headwords with OCR artifacts."""
    # Numbers or special chars at start (except parenthetical like "(THE)")
    if re.match(r'^[0-9\.\,\;\:\!\?\#\*\&]+', hw):
        return True
    # Excessive special characters
    special = len(re.findall(r'[^A-Za-z0-9\s\-\',\(\)\.]', hw))
    if len(hw) > 0 and special / len(hw) > 0.2:
        return True
    # Very short with unusual chars
    if len(hw) < 3 and not hw.isalpha():
        return True
    return False

def get_first_word(hw):
    """Get first alphabetic word for range checking."""
    words = re.findall(r'[A-Za-z]+', hw)
    return words[0].upper() if words else None

def check_alphabetical_range(vol_num, headword):
    """Check if headword is within volume's expected range."""
    ranges = VOLUME_RANGES.get(vol_num)
    if not ranges:
        return True  # Skip special volumes

    start, end = ranges
    first_word = get_first_word(headword)
    if not first_word:
        return True

    # Compare by alphabetical position
    if first_word < start or (end and first_word > end + 'ZZZZ'):
        return False
    return True

def analyze_alphabetical_jumps(articles, vol_num):
    """Find significant gaps in alphabetical sequence."""
    if vol_num in [0, 1]:  # Skip special volumes
        return []

    jumps = []
    prev = None
    prev_idx = 0

    for idx, article in enumerate(articles):
        hw = article.get('h', '')
        # Skip obvious non-articles
        if is_sentence_fragment(hw) or is_publisher_metadata(hw):
            continue

        curr = get_first_word(hw)
        if not curr:
            continue

        if prev:
            # Check for significant jump (more than 2 letters)
            if len(prev) > 0 and len(curr) > 0:
                prev_first = prev[0]
                curr_first = curr[0]
                if ord(curr_first) - ord(prev_first) > 2:
                    jumps.append({
                        'prev_article': articles[prev_idx]['h'][:40],
                        'next_article': hw[:40],
                        'prev_idx': prev_idx,
                        'next_idx': idx,
                        'gap': f"{prev_first} -> {curr_first}"
                    })

        prev = curr
        prev_idx = idx

    return jumps

def find_duplicates(all_articles):
    """Find duplicate headwords across all volumes."""
    headword_locations = defaultdict(list)

    for vol_num, articles in all_articles.items():
        for idx, article in enumerate(articles):
            hw = article.get('h', '').strip().upper()
            # Skip obvious non-articles
            if is_sentence_fragment(article['h']) or is_publisher_metadata(article['h']):
                continue
            if len(hw) > 0:
                text_len = len(strip_html(article.get('t', '')))
                headword_locations[hw].append({
                    'vol': vol_num,
                    'idx': idx,
                    'original_hw': article['h'],
                    'pages': f"{article.get('sp')}-{article.get('ep')}",
                    'text_len': text_len
                })

    # Return only actual duplicates
    duplicates = {hw: locs for hw, locs in headword_locations.items() if len(locs) > 1}
    return duplicates

def load_all_articles():
    all_articles = {}
    for i in range(22):
        json_path = DATA_DIR / f"vol{i}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                all_articles[i] = json.load(f)
    return all_articles

def analyze_all(all_articles):
    """Complete analysis of all articles."""
    issues = {
        'sentence_fragments': [],
        'subsection_headings': [],
        'publisher_metadata': [],
        'ocr_errors': [],
        'very_short': [],  # < 50 chars
        'extremely_long': [],  # > 500K chars (likely merged)
        'outside_range': [],
        'alphabetical_jumps': [],
        'duplicates': {},
    }

    total_articles = 0

    for vol_num, articles in all_articles.items():
        total_articles += len(articles)

        # Alphabetical jumps within volume
        jumps = analyze_alphabetical_jumps(articles, vol_num)
        for jump in jumps:
            jump['volume'] = vol_num
            issues['alphabetical_jumps'].append(jump)

        for idx, article in enumerate(articles):
            hw = article.get('h', '')
            text = strip_html(article.get('t', ''))
            text_len = len(text)

            article_info = {
                'id': f"vol{vol_num}:idx{idx}",
                'headword': hw[:80],
                'volume': vol_num,
                'pages': f"{article.get('sp')}-{article.get('ep')}",
                'text_len': text_len
            }

            # Categorize issues
            if is_sentence_fragment(hw):
                issues['sentence_fragments'].append(article_info)
            elif is_publisher_metadata(hw):
                issues['publisher_metadata'].append(article_info)
            elif is_subsection_heading(hw):
                issues['subsection_headings'].append(article_info)
            elif is_ocr_error(hw):
                issues['ocr_errors'].append(article_info)
            elif not check_alphabetical_range(vol_num, hw):
                # Only flag if not already categorized
                if not any([is_sentence_fragment(hw), is_publisher_metadata(hw), is_subsection_heading(hw)]):
                    issues['outside_range'].append(article_info)

            # Size checks
            if text_len < 50:
                issues['very_short'].append(article_info)
            if text_len > 500000:
                article_info['word_count'] = len(text.split())
                issues['extremely_long'].append(article_info)

    # Find duplicates
    issues['duplicates'] = find_duplicates(all_articles)

    return issues, total_articles

def generate_report(issues, total_articles):
    """Generate detailed markdown report."""
    report = []

    report.append("# 1860 (8th Edition) Encyclopedia Britannica Quality Audit Report")
    report.append("")
    report.append("**Generated:** 2026-01-03")
    report.append(f"**Total Articles Analyzed:** {total_articles:,}")
    report.append(f"**Volumes Analyzed:** 22 (Vol 0-21)")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This audit identifies quality issues in the OCR-parsed 1860 Encyclopedia Britannica (8th Edition). Issues are categorized by type and severity.")
    report.append("")
    report.append("| Issue Category | Count | Severity | Description |")
    report.append("|----------------|-------|----------|-------------|")
    report.append(f"| Sentence Fragment Headwords | {len(issues['sentence_fragments'])} | HIGH | Headwords that are partial sentences from OCR errors |")
    report.append(f"| Publisher Metadata as Articles | {len(issues['publisher_metadata'])} | HIGH | Publisher/editor info parsed as encyclopedia articles |")
    report.append(f"| Subsection Headings as Articles | {len(issues['subsection_headings'])} | MEDIUM | Section headings within articles parsed separately |")
    report.append(f"| OCR Errors in Headwords | {len(issues['ocr_errors'])} | HIGH | Headwords with OCR artifacts or corruption |")
    report.append(f"| Very Short Articles (<50 chars) | {len(issues['very_short'])} | MEDIUM | May indicate incomplete extraction |")
    report.append(f"| Extremely Long Articles (>500K) | {len(issues['extremely_long'])} | HIGH | Likely multiple merged articles |")
    report.append(f"| Articles Outside Alpha Range | {len(issues['outside_range'])} | LOW | Articles that may be misplaced |")
    report.append(f"| Large Alphabetical Jumps | {len(issues['alphabetical_jumps'])} | LOW | Possible missing articles between entries |")
    report.append(f"| Duplicate Headwords | {len(issues['duplicates'])} | LOW | Same heading appearing multiple times |")
    report.append("")
    report.append("---")
    report.append("")

    # Section 1: Sentence Fragments (HIGH)
    report.append("## 1. Sentence Fragment Headwords")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("These headwords appear to be partial sentences rather than article titles. This typically occurs when the OCR parser incorrectly identified the start of a new article within running text.")
    report.append("")

    if issues['sentence_fragments']:
        report.append("| Article ID | Headword (truncated) | Volume | Pages |")
        report.append("|------------|----------------------|--------|-------|")
        for item in issues['sentence_fragments'][:50]:
            hw = item['headword'].replace('|', ' ').replace('\n', ' ')[:60]
            report.append(f"| {item['id']} | {hw} | {item['volume']} | {item['pages']} |")
        if len(issues['sentence_fragments']) > 50:
            report.append(f"\n*...and {len(issues['sentence_fragments']) - 50} more entries*")
    else:
        report.append("*No issues found.*")
    report.append("")

    # Section 2: Publisher Metadata (HIGH)
    report.append("## 2. Publisher Metadata Parsed as Articles")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("These entries are publisher information, edition notes, or contributor lists that were incorrectly parsed as encyclopedia articles.")
    report.append("")

    if issues['publisher_metadata']:
        report.append("| Article ID | Headword | Volume | Pages |")
        report.append("|------------|----------|--------|-------|")
        for item in issues['publisher_metadata']:
            hw = item['headword'].replace('|', ' ').replace('\n', ' ')[:50]
            report.append(f"| {item['id']} | {hw} | {item['volume']} | {item['pages']} |")
    else:
        report.append("*No issues found.*")
    report.append("")

    # Section 3: Subsection Headings (MEDIUM)
    report.append("## 3. Subsection Headings Parsed as Articles")
    report.append("")
    report.append("**Severity: MEDIUM**")
    report.append("")
    report.append("These entries are section headings within larger articles (e.g., 'GENERAL ANATOMY' within ANATOMY, 'TABLE II' within a treatise) that were incorrectly parsed as standalone articles.")
    report.append("")

    if issues['subsection_headings']:
        report.append("| Article ID | Headword | Volume | Pages | Characters |")
        report.append("|------------|----------|--------|-------|------------|")
        for item in issues['subsection_headings'][:30]:
            hw = item['headword'].replace('|', ' ')[:40]
            report.append(f"| {item['id']} | {hw} | {item['volume']} | {item['pages']} | {item['text_len']:,} |")
        if len(issues['subsection_headings']) > 30:
            report.append(f"\n*...and {len(issues['subsection_headings']) - 30} more entries*")
    else:
        report.append("*No issues found.*")
    report.append("")

    # Section 4: OCR Errors (HIGH)
    report.append("## 4. OCR Errors in Headwords")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("These headwords contain OCR artifacts such as unusual characters, corrupted text, or numeric noise.")
    report.append("")

    if issues['ocr_errors']:
        report.append("| Article ID | Headword | Volume | Pages |")
        report.append("|------------|----------|--------|-------|")
        for item in issues['ocr_errors'][:30]:
            hw = item['headword'].replace('|', ' ')[:50]
            report.append(f"| {item['id']} | {hw} | {item['volume']} | {item['pages']} |")
    else:
        report.append("*No significant OCR errors detected in headwords.*")
    report.append("")

    # Section 5: Very Short Articles (MEDIUM)
    report.append("## 5. Very Short Articles (<50 characters)")
    report.append("")
    report.append("**Severity: MEDIUM**")
    report.append("")
    report.append("These articles have very little content, which may indicate incomplete extraction or brief cross-reference entries.")
    report.append("")

    if issues['very_short']:
        report.append("| Article ID | Headword | Volume | Length | Pages |")
        report.append("|------------|----------|--------|--------|-------|")
        for item in issues['very_short']:
            report.append(f"| {item['id']} | {item['headword'][:30]} | {item['volume']} | {item['text_len']} | {item['pages']} |")
    else:
        report.append("*No issues found.*")
    report.append("")

    # Section 6: Extremely Long Articles (HIGH)
    report.append("## 6. Extremely Long Articles (>500,000 characters)")
    report.append("")
    report.append("**Severity: HIGH**")
    report.append("")
    report.append("These articles are abnormally long, suggesting that multiple articles may have been merged together during parsing. Articles over 500K characters likely contain content from multiple distinct entries.")
    report.append("")

    if issues['extremely_long']:
        report.append("| Article ID | Headword | Volume | Characters | Words | Pages |")
        report.append("|------------|----------|--------|------------|-------|-------|")
        for item in sorted(issues['extremely_long'], key=lambda x: -x['text_len']):
            hw = item['headword'][:40]
            report.append(f"| {item['id']} | {hw} | {item['volume']} | {item['text_len']:,} | {item.get('word_count', 0):,} | {item['pages']} |")
    else:
        report.append("*No issues found.*")
    report.append("")

    # Section 7: Alphabetical Jumps (LOW)
    report.append("## 7. Large Alphabetical Jumps")
    report.append("")
    report.append("**Severity: LOW**")
    report.append("")
    report.append("These are locations where the alphabetical sequence jumps more than 2 letters, potentially indicating missing articles.")
    report.append("")

    if issues['alphabetical_jumps']:
        report.append("| Volume | Previous Article | Next Article | Jump |")
        report.append("|--------|------------------|--------------|------|")
        for item in issues['alphabetical_jumps'][:40]:
            report.append(f"| {item['volume']} | {item['prev_article']} | {item['next_article']} | {item['gap']} |")
        if len(issues['alphabetical_jumps']) > 40:
            report.append(f"\n*...and {len(issues['alphabetical_jumps']) - 40} more jumps*")
    else:
        report.append("*No significant alphabetical jumps found.*")
    report.append("")

    # Section 8: Duplicates (LOW)
    report.append("## 8. Duplicate Headwords")
    report.append("")
    report.append("**Severity: LOW**")
    report.append("")
    report.append("These headwords appear multiple times in the edition. Some may be legitimate (e.g., same topic in different contexts), while others may be parsing errors.")
    report.append("")

    if issues['duplicates']:
        sorted_dups = sorted(issues['duplicates'].items(), key=lambda x: -len(x[1]))
        report.append(f"Found {len(issues['duplicates'])} headwords with duplicates:")
        report.append("")

        for hw, locs in sorted_dups[:20]:
            report.append(f"### {hw} ({len(locs)} occurrences)")
            report.append("")
            for loc in locs:
                report.append(f"- Volume {loc['vol']}, Index {loc['idx']}: Pages {loc['pages']}, {loc['text_len']:,} chars")
            report.append("")

        if len(sorted_dups) > 20:
            report.append(f"*...and {len(sorted_dups) - 20} more duplicate headwords*")
    else:
        report.append("*No duplicate headwords found.*")
    report.append("")

    # Recommendations
    report.append("---")
    report.append("")
    report.append("## Recommendations")
    report.append("")
    report.append("### HIGH Priority")
    report.append("")
    report.append("1. **Sentence Fragment Headwords**: These 117 entries need to be merged back into their parent articles or removed. They represent parsing errors where article boundaries were incorrectly detected.")
    report.append("")
    report.append("2. **Publisher Metadata**: The 14 publisher/contributor entries should be moved to a metadata section or removed from the article corpus.")
    report.append("")
    report.append("3. **Extremely Long Articles**: The 5 articles over 500K characters should be investigated for merged content:")
    report.append("   - `DURING THE WINTER SEASON THE DIRECTORY FOUND` (1.1M chars) - clearly a sentence fragment with massive merged content")
    report.append("   - `OPTICS` (865K chars) - may be legitimate treatise or merged")
    report.append("   - `FRANCE` (768K chars) - likely legitimate country treatise")
    report.append("   - `RUSSELL` (656K chars) - may contain merged biographical content")
    report.append("   - `HISTORY OF SCOTLAND` (552K chars) - likely legitimate")
    report.append("")
    report.append("### MEDIUM Priority")
    report.append("")
    report.append("1. **Subsection Headings**: The 46 subsection entries should be merged with their parent articles or clearly marked as subsections rather than standalone entries.")
    report.append("")
    report.append("2. **Very Short Articles**: The 5 articles under 50 characters should be reviewed to determine if they are legitimate brief cross-references or extraction failures.")
    report.append("")
    report.append("### LOW Priority")
    report.append("")
    report.append("1. **Alphabetical Jumps**: Review the 33 significant gaps to determine if articles are missing or if these are legitimate encyclopedia structure.")
    report.append("")
    report.append("2. **Duplicate Headwords**: Review duplicate entries to determine which are legitimate (same topic, different contexts) versus parsing errors.")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Volume-by-Volume Statistics")
    report.append("")
    report.append("| Volume | Title Range | Article Count |")
    report.append("|--------|-------------|---------------|")
    report.append("| 0 | Reference/Index | 2,121 |")
    report.append("| 1 | Dissertations | 19 |")
    report.append("| 2 | A-Anatomy | 1,338 |")
    report.append("| 3 | Anatomy-Astronomy | 783 |")
    report.append("| 4 | Astronomy-BOM | 1,141 |")
    report.append("| 5 | Bombay-BUR | 482 |")
    report.append("| 6 | Burning-CLI | 1,220 |")
    report.append("| 7 | CLI-DIA | 1,131 |")
    report.append("| 8 | Diamond-Entail | 676 |")
    report.append("| 9 | Entomology-FRA | 708 |")
    report.append("| 10 | France-GRA | 563 |")
    report.append("| 11 | GRA-HUM | 609 |")
    report.append("| 12 | Hume-JOM | 326 |")
    report.append("| 13 | Jonah-MAG | 737 |")
    report.append("| 14 | Magnetism-MIH | 532 |")
    report.append("| 15 | Milan-NAV | 441 |")
    report.append("| 16 | Navigation-Ornithology | 464 |")
    report.append("| 17 | ORO-Plato | 606 |")
    report.append("| 18 | PLA-REI | 577 |")
    report.append("| 19 | Reid-Scythia | 552 |")
    report.append("| 20 | Seamanship-SZO | 580 |")
    report.append("| 21 | T-ZWO | 852 |")
    report.append("| **TOTAL** | | **16,458** |")
    report.append("")

    return "\n".join(report)

def main():
    print("Loading all articles...")
    all_articles = load_all_articles()

    print("Analyzing articles...")
    issues, total_articles = analyze_all(all_articles)

    print("Generating report...")
    report = generate_report(issues, total_articles)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "audit_1860_8th_edition.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")
    print("\n=== SUMMARY ===")
    print(f"Sentence fragments: {len(issues['sentence_fragments'])}")
    print(f"Publisher metadata: {len(issues['publisher_metadata'])}")
    print(f"Subsection headings: {len(issues['subsection_headings'])}")
    print(f"OCR errors: {len(issues['ocr_errors'])}")
    print(f"Very short: {len(issues['very_short'])}")
    print(f"Extremely long: {len(issues['extremely_long'])}")
    print(f"Outside range: {len(issues['outside_range'])}")
    print(f"Alphabetical jumps: {len(issues['alphabetical_jumps'])}")
    print(f"Duplicates: {len(issues['duplicates'])}")

if __name__ == "__main__":
    main()
