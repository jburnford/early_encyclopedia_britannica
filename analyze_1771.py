#!/usr/bin/env python3
"""
Analyze 1771 (1st Edition) Encyclopedia Britannica articles for quality issues.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
import html

# Volume letter ranges from index.html
VOLUME_RANGES = {
    1: ('A', 'B'),  # Volume 1: A-B
    2: ('C', 'L'),  # Volume 2: C-L
    3: ('M', 'Z'),  # Volume 3: M-Z
}

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def strip_html(text):
    """Remove HTML tags from text."""
    clean = re.sub(r'<[^>]+>', '', text)
    return html.unescape(clean)

def get_first_letter(headword):
    """Get the first alphabetical letter from a headword."""
    # Strip any leading punctuation/numbers/spaces
    clean = re.sub(r'^[^a-zA-Z]*', '', headword)
    if clean:
        return clean[0].upper()
    return None

def is_valid_headword(headword):
    """Check if headword looks valid (starts with letters, reasonable format)."""
    if not headword:
        return False
    # Should start with letters (after stripping)
    clean = re.sub(r'^[^a-zA-Z]*', '', headword)
    if not clean:
        return False
    return True

def analyze_volume(articles, vol_num, letter_range):
    """Analyze a single volume for quality issues."""
    issues = {
        'out_of_range': [],
        'short_articles': [],
        'long_articles': [],
        'duplicates': [],
        'alpha_jumps': [],
        'ocr_errors': [],
    }

    start_letter, end_letter = letter_range
    headword_counts = defaultdict(list)
    prev_headword = None
    prev_first_letter = None

    for i, article in enumerate(articles):
        headword = article.get('h', '')
        text = article.get('t', '')
        text_plain = strip_html(text)

        article_id = f"vol{vol_num}_idx{i}"
        first_letter = get_first_letter(headword)

        # Track duplicates
        headword_upper = headword.upper()
        headword_counts[headword_upper].append((article_id, headword, i))

        # 1. Check if article is outside alphabetical range
        if first_letter:
            if first_letter < start_letter or first_letter > end_letter:
                issues['out_of_range'].append({
                    'id': article_id,
                    'headword': headword,
                    'first_letter': first_letter,
                    'expected_range': f"{start_letter}-{end_letter}",
                    'severity': 'HIGH'
                })

        # 2. Check for unusually short articles (under 50 chars)
        if len(text_plain) < 50:
            issues['short_articles'].append({
                'id': article_id,
                'headword': headword,
                'length': len(text_plain),
                'text': text_plain[:100],
                'severity': 'MEDIUM' if len(text_plain) < 20 else 'LOW'
            })

        # 3. Check for unusually long articles (over 50000 chars - potential merge)
        if len(text_plain) > 50000:
            issues['long_articles'].append({
                'id': article_id,
                'headword': headword,
                'length': len(text_plain),
                'text_preview': text_plain[:200] + '...',
                'severity': 'HIGH' if len(text_plain) > 100000 else 'MEDIUM'
            })

        # 5. Check for large alphabetical jumps
        if prev_first_letter and first_letter:
            if first_letter and prev_first_letter:
                letter_diff = ord(first_letter) - ord(prev_first_letter)
                # Jump of more than 1 letter (except at boundaries)
                if letter_diff > 1:
                    issues['alpha_jumps'].append({
                        'id': article_id,
                        'prev_headword': prev_headword,
                        'current_headword': headword,
                        'jump': f"{prev_first_letter} -> {first_letter}",
                        'severity': 'MEDIUM' if letter_diff <= 3 else 'HIGH'
                    })
                # Backward jump (not normal)
                elif letter_diff < 0:
                    issues['alpha_jumps'].append({
                        'id': article_id,
                        'prev_headword': prev_headword,
                        'current_headword': headword,
                        'jump': f"{prev_first_letter} -> {first_letter} (BACKWARD)",
                        'severity': 'HIGH'
                    })

        # 6. Check for OCR/parsing errors in headwords
        # Patterns that suggest errors:
        ocr_issues = []

        # Random punctuation or garbage characters
        if re.search(r'[^\w\s\-\',\(\)&\.:]', headword):
            bad_chars = re.findall(r'[^\w\s\-\',\(\)&\.:]', headword)
            ocr_issues.append(f"unusual characters: {bad_chars}")

        # Very long headwords (likely sentence fragments)
        if len(headword) > 60:
            ocr_issues.append(f"very long ({len(headword)} chars)")

        # Headword looks like a sentence
        if re.search(r'\b(the|is|are|was|were|of|in|to|for|and|or|a|an)\b', headword.lower()) and len(headword) > 30:
            ocr_issues.append("appears to be sentence fragment")

        # Starts with lowercase (unusual for encyclopedia)
        if headword and headword[0].islower():
            ocr_issues.append("starts with lowercase")

        # Contains multiple spaces
        if '  ' in headword:
            ocr_issues.append("contains multiple spaces")

        # Numeric-heavy headwords
        if re.search(r'\d{3,}', headword):
            ocr_issues.append("contains long number sequence")

        if ocr_issues:
            issues['ocr_errors'].append({
                'id': article_id,
                'headword': headword,
                'issues': ocr_issues,
                'severity': 'HIGH' if len(ocr_issues) > 1 else 'MEDIUM'
            })

        prev_headword = headword
        prev_first_letter = first_letter

    # 4. Find duplicates (after processing all articles)
    for headword, occurrences in headword_counts.items():
        if len(occurrences) > 1:
            issues['duplicates'].append({
                'headword': headword,
                'occurrences': occurrences,
                'count': len(occurrences),
                'severity': 'HIGH' if len(occurrences) > 2 else 'MEDIUM'
            })

    return issues

def generate_report(all_issues, output_path, vol0_stats=None):
    """Generate markdown report."""
    lines = []
    lines.append("# 1771 (1st Edition) Encyclopedia Britannica - Quality Audit Report")
    lines.append("")
    lines.append("**Generated**: 2026-01-03")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    # Count totals
    total_out_of_range = sum(len(v['out_of_range']) for v in all_issues.values())
    total_short = sum(len(v['short_articles']) for v in all_issues.values())
    total_long = sum(len(v['long_articles']) for v in all_issues.values())
    total_duplicates = sum(len(v['duplicates']) for v in all_issues.values())
    total_jumps = sum(len(v['alpha_jumps']) for v in all_issues.values())
    total_ocr = sum(len(v['ocr_errors']) for v in all_issues.values())

    lines.append("| Issue Type | Count | Severity Distribution |")
    lines.append("|------------|-------|----------------------|")

    # Calculate severity distributions
    def count_severity(issues_list):
        high = sum(1 for i in issues_list if i.get('severity') == 'HIGH')
        med = sum(1 for i in issues_list if i.get('severity') == 'MEDIUM')
        low = sum(1 for i in issues_list if i.get('severity') == 'LOW')
        return f"HIGH: {high}, MEDIUM: {med}, LOW: {low}"

    all_out_of_range = []
    all_short = []
    all_long = []
    all_duplicates = []
    all_jumps = []
    all_ocr = []

    for vol, issues in all_issues.items():
        all_out_of_range.extend(issues['out_of_range'])
        all_short.extend(issues['short_articles'])
        all_long.extend(issues['long_articles'])
        all_duplicates.extend(issues['duplicates'])
        all_jumps.extend(issues['alpha_jumps'])
        all_ocr.extend(issues['ocr_errors'])

    lines.append(f"| Articles Outside Range | {total_out_of_range} | {count_severity(all_out_of_range)} |")
    lines.append(f"| Unusually Short Articles | {total_short} | {count_severity(all_short)} |")
    lines.append(f"| Unusually Long Articles | {total_long} | {count_severity(all_long)} |")
    lines.append(f"| Duplicate Articles | {total_duplicates} | {count_severity(all_duplicates)} |")
    lines.append(f"| Large Alphabetical Jumps | {total_jumps} | {count_severity(all_jumps)} |")
    lines.append(f"| OCR/Parsing Errors | {total_ocr} | {count_severity(all_ocr)} |")
    lines.append("")

    # Categorize out-of-range articles
    if all_out_of_range:
        plate_explanations = [a for a in all_out_of_range if 'PLATE' in a['headword'].upper() or 'EXPLANATION' in a['headword'].upper()]
        anatomical = [a for a in all_out_of_range if any(term in a['headword'].upper() for term in ['MUSCLE', 'EXTENSOR', 'FLEXOR', 'DELTOID', 'LATISSIMUS', 'OBLIQ', 'RADIAL', 'ULNAR', 'RECT', 'DIAPHRAGM', 'MEDULLA', 'CEREBR', 'INTESTIN'])]
        end_markers = [a for a in all_out_of_range if 'END OF' in a['headword'].upper() or 'VOLUME' in a['headword'].upper()]
        propositions = [a for a in all_out_of_range if 'PROPOSITION' in a['headword'].upper()]
        sentence_frags = [a for a in all_out_of_range if len(a['headword']) > 40]

        lines.append("### Out-of-Range Article Categories")
        lines.append("")
        lines.append(f"The {total_out_of_range} out-of-range articles fall into these categories:")
        lines.append("")
        lines.append(f"- **Plate/Figure Explanations**: {len(plate_explanations)} (e.g., \"EXPLANATION OF PLATE XIII\")")
        lines.append(f"- **Anatomical Terms from ANATOMY treatise**: {len(anatomical)} (e.g., \"EXTENSOR DIGITORUM\", \"LATISSIMUS DORSI\")")
        lines.append(f"- **End-of-Volume Markers**: {len(end_markers)} (e.g., \"END OF THE FIRST VOLUME\")")
        lines.append(f"- **Proposition Headings from treatises**: {len(propositions)} (e.g., \"PROPOSITION IX\")")
        lines.append(f"- **Sentence Fragments (parsing errors)**: {len(sentence_frags)} (e.g., \"MANY ATTEMPTS HAVE BEEN MADE...\")")
        lines.append("")
        lines.append("**Analysis**: Most out-of-range articles are sub-entries from treatises (especially ANATOMY and GEOMETRY) that were extracted as separate articles. These represent structural features of the encyclopedia's treatise format rather than true alphabetical articles.")
        lines.append("")

    # Detailed sections for each volume
    for vol_num in sorted(all_issues.keys()):
        issues = all_issues[vol_num]
        letter_range = VOLUME_RANGES.get(vol_num, ('?', '?'))

        lines.append(f"---")
        lines.append(f"## Volume {vol_num} ({letter_range[0]}-{letter_range[1]})")
        lines.append("")

        # Out of range
        if issues['out_of_range']:
            lines.append(f"### 1. Articles Outside Alphabetical Range ({len(issues['out_of_range'])})")
            lines.append("")
            lines.append("These articles appear in a volume where their starting letter doesn't match the expected range.")
            lines.append("")
            for item in issues['out_of_range'][:25]:  # Limit to 25 examples
                lines.append(f"- **[{item['severity']}]** `{item['id']}`: \"{item['headword']}\" (starts with '{item['first_letter']}', expected {item['expected_range']})")
            if len(issues['out_of_range']) > 25:
                lines.append(f"- *... and {len(issues['out_of_range']) - 25} more*")
            lines.append("")

        # Short articles
        if issues['short_articles']:
            lines.append(f"### 2. Unusually Short Articles ({len(issues['short_articles'])})")
            lines.append("")
            lines.append("Articles under 50 characters may indicate parsing errors or incomplete extraction.")
            lines.append("")
            # Show HIGH severity first
            sorted_short = sorted(issues['short_articles'], key=lambda x: (0 if x['severity'] == 'HIGH' else 1, x['length']))
            for item in sorted_short[:30]:
                text_preview = item['text'].replace('\n', ' ')[:50]
                lines.append(f"- **[{item['severity']}]** `{item['id']}`: \"{item['headword']}\" ({item['length']} chars)")
                lines.append(f"  - Text: \"{text_preview}\"")
            if len(issues['short_articles']) > 30:
                lines.append(f"- *... and {len(issues['short_articles']) - 30} more*")
            lines.append("")

        # Long articles
        if issues['long_articles']:
            lines.append(f"### 3. Unusually Long Articles ({len(issues['long_articles'])})")
            lines.append("")
            lines.append("Articles over 50,000 characters may contain merged content from multiple articles.")
            lines.append("")
            for item in sorted(issues['long_articles'], key=lambda x: -x['length']):
                lines.append(f"- **[{item['severity']}]** `{item['id']}`: \"{item['headword']}\" ({item['length']:,} chars)")
                lines.append(f"  - Preview: \"{item['text_preview'][:100]}...\"")
            lines.append("")

        # Duplicates
        if issues['duplicates']:
            lines.append(f"### 4. Duplicate Articles ({len(issues['duplicates'])})")
            lines.append("")
            lines.append("Same headword appearing multiple times in the volume.")
            lines.append("")
            sorted_dups = sorted(issues['duplicates'], key=lambda x: -x['count'])
            for item in sorted_dups[:25]:
                occ_str = ", ".join([f"`{o[0]}`" for o in item['occurrences'][:5]])
                if len(item['occurrences']) > 5:
                    occ_str += f" (+{len(item['occurrences'])-5} more)"
                lines.append(f"- **[{item['severity']}]** \"{item['headword']}\" appears {item['count']} times: {occ_str}")
            if len(issues['duplicates']) > 25:
                lines.append(f"- *... and {len(issues['duplicates']) - 25} more*")
            lines.append("")

        # Alphabetical jumps
        if issues['alpha_jumps']:
            lines.append(f"### 5. Large Alphabetical Jumps ({len(issues['alpha_jumps'])})")
            lines.append("")
            lines.append("Gaps in alphabetical sequence suggesting missing articles or sorting issues.")
            lines.append("")
            for item in issues['alpha_jumps'][:25]:
                lines.append(f"- **[{item['severity']}]** `{item['id']}`: {item['jump']}")
                lines.append(f"  - From: \"{item['prev_headword']}\" -> To: \"{item['current_headword']}\"")
            if len(issues['alpha_jumps']) > 25:
                lines.append(f"- *... and {len(issues['alpha_jumps']) - 25} more*")
            lines.append("")

        # OCR errors
        if issues['ocr_errors']:
            lines.append(f"### 6. OCR/Parsing Errors ({len(issues['ocr_errors'])})")
            lines.append("")
            lines.append("Headwords with unusual characters, formatting issues, or apparent parsing problems.")
            lines.append("")
            sorted_ocr = sorted(issues['ocr_errors'], key=lambda x: (0 if x['severity'] == 'HIGH' else 1))
            for item in sorted_ocr[:30]:
                issues_str = ", ".join(item['issues'])
                lines.append(f"- **[{item['severity']}]** `{item['id']}`: \"{item['headword']}\"")
                lines.append(f"  - Issues: {issues_str}")
            if len(issues['ocr_errors']) > 30:
                lines.append(f"- *... and {len(issues['ocr_errors']) - 30} more*")
            lines.append("")

    # Recommendations
    lines.append("---")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### High Priority Fixes")
    lines.append("")
    lines.append("1. **Review out-of-range articles**: These may indicate volume assignment errors or cross-references parsed as articles.")
    lines.append("2. **Investigate very short articles**: Articles under 20 characters likely represent parsing failures.")
    lines.append("3. **Check very long articles**: Articles over 100,000 characters may contain multiple merged articles.")
    lines.append("4. **Review backward alphabetical jumps**: These indicate potential sorting or extraction errors.")
    lines.append("")
    lines.append("### Medium Priority Fixes")
    lines.append("")
    lines.append("1. **Review duplicate entries**: Some may be legitimate (e.g., different senses), others parsing errors.")
    lines.append("2. **Check OCR issues in headwords**: Focus on headwords with multiple issues flagged.")
    lines.append("3. **Investigate large forward jumps**: May indicate missing articles in the sequence.")
    lines.append("")
    lines.append("### Notes on 1st Edition")
    lines.append("")
    lines.append("The 1771 First Edition is notable for being the original Encyclopaedia Britannica. Some characteristics to consider:")
    lines.append("")
    lines.append("- Treatise-style organization means some letters have few standalone articles")
    lines.append("- Cross-references are extensively used")
    lines.append("- Original OCR from 18th century typography may have higher error rates")
    lines.append("- Three-volume structure limits article counts per letter")
    lines.append("")

    # Add vol0 analysis
    if vol0_stats:
        lines.append("---")
        lines.append("## Appendix: vol0.json Cross-Reference Data Analysis")
        lines.append("")
        lines.append("The `vol0.json` file contains cross-reference entries that link to articles in other editions or provide stubs for terms referenced but not fully defined in the main volumes.")
        lines.append("")
        lines.append("### Statistics")
        lines.append("")
        lines.append(f"- **Total entries**: {vol0_stats['total']}")
        lines.append(f"- **Entries without page numbers**: {vol0_stats['no_page']} ({vol0_stats['no_page']*100//vol0_stats['total']}%)")
        lines.append(f"- **Entries with cross-references**: {vol0_stats['with_xref']}")
        lines.append(f"- **Duplicate headwords**: {len(vol0_stats['duplicates'])}")
        lines.append("")

        # Letter distribution
        lines.append("### Letter Distribution")
        lines.append("")
        lines.append("| Letter | Count |")
        lines.append("|--------|-------|")
        for letter in sorted(vol0_stats['letter_dist'].keys()):
            lines.append(f"| {letter} | {vol0_stats['letter_dist'][letter]} |")
        lines.append("")

        if vol0_stats['duplicates']:
            lines.append("### Duplicate Headwords in vol0")
            lines.append("")
            sorted_dups = sorted(vol0_stats['duplicates'], key=lambda x: -x[1])[:20]
            for headword, count in sorted_dups:
                lines.append(f"- \"{headword}\" appears {count} times")
            if len(vol0_stats['duplicates']) > 20:
                lines.append(f"- *... and {len(vol0_stats['duplicates']) - 20} more*")
            lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path

def analyze_vol0(articles):
    """Analyze vol0.json which contains cross-edition reference data."""
    stats = {
        'total': len(articles),
        'no_page': 0,
        'with_xref': 0,
        'duplicates': [],
        'letter_dist': defaultdict(int)
    }

    headword_counts = defaultdict(int)

    for article in articles:
        headword = article.get('h', '')
        text = article.get('t', '')
        sp = article.get('sp')

        if sp is None:
            stats['no_page'] += 1

        if '<a class="xref"' in text:
            stats['with_xref'] += 1

        first_letter = get_first_letter(headword)
        if first_letter:
            stats['letter_dist'][first_letter] += 1

        headword_upper = headword.upper()
        headword_counts[headword_upper] += 1

    for headword, count in headword_counts.items():
        if count > 1:
            stats['duplicates'].append((headword, count))

    return stats

def main():
    base_path = Path('/home/jic823/1815EncyclopediaBritannicaNLS/docs/1771/data')
    output_path = Path('/home/jic823/1815EncyclopediaBritannicaNLS/reports/audit_1771_1st_edition.md')

    all_issues = {}
    vol0_stats = None

    for vol_num in [1, 2, 3]:
        json_path = base_path / f'vol{vol_num}.json'
        if json_path.exists():
            print(f"Analyzing Volume {vol_num}...")
            articles = load_json(json_path)
            print(f"  Loaded {len(articles)} articles")
            letter_range = VOLUME_RANGES.get(vol_num, ('A', 'Z'))
            issues = analyze_volume(articles, vol_num, letter_range)
            all_issues[vol_num] = issues

            # Print summary
            print(f"  Issues found:")
            print(f"    - Out of range: {len(issues['out_of_range'])}")
            print(f"    - Short articles: {len(issues['short_articles'])}")
            print(f"    - Long articles: {len(issues['long_articles'])}")
            print(f"    - Duplicates: {len(issues['duplicates'])}")
            print(f"    - Alpha jumps: {len(issues['alpha_jumps'])}")
            print(f"    - OCR errors: {len(issues['ocr_errors'])}")
        else:
            print(f"Warning: {json_path} not found")

    # Also analyze vol0.json (cross-reference data)
    vol0_path = base_path / 'vol0.json'
    if vol0_path.exists():
        print(f"Analyzing vol0.json (cross-reference data)...")
        vol0_articles = load_json(vol0_path)
        vol0_stats = analyze_vol0(vol0_articles)
        print(f"  Loaded {vol0_stats['total']} articles in vol0")
        print(f"  No page info: {vol0_stats['no_page']}")
        print(f"  With cross-refs: {vol0_stats['with_xref']}")
        print(f"  Duplicates: {len(vol0_stats['duplicates'])}")

    print(f"\nGenerating report...")
    report_path = generate_report(all_issues, output_path, vol0_stats)
    print(f"Report written to: {report_path}")

if __name__ == '__main__':
    main()
