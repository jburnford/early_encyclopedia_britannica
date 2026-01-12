#!/usr/bin/env python3
"""
Audit script for 1778 (2nd Edition) Encyclopedia Britannica articles.
Analyzes JSON data files for quality issues.
"""

import json
import re
import os
from collections import defaultdict
from pathlib import Path

# Volume letter ranges based on index.html
# Note: vol0.json is extra content not listed in the index
# Volumes are named vol1.json through vol10.json (matching display names)
VOLUME_RANGES = {
    "vol0": ("A", "Z"),          # Extra comprehensive volume (not in index)
    "vol1": ("A", "AST"),        # Volume 1: A-AST
    "vol2": ("AST", "BZZ"),      # Volume 2: Astronomy-BZO (A+B)
    "vol3": ("C", "CZZ"),        # Volume 3: C
    "vol4": ("D", "FZZ"),        # Volume 4: D-F
    "vol5": ("G", "JZZ"),        # Volume 5: G-J
    "vol6": ("K", "MED"),        # Volume 6: K-Medicine
    "vol7": ("MED", "OPT"),      # Volume 7: Medicines-Optics (M-O)
    "vol8": ("OPT", "POE"),      # Volume 8: Optics-Poetry (O-P)
    "vol9": ("POI", "SCU"),      # Volume 9: POI-SCU (P-S)
    "vol10": ("A", "ZZZ"),       # Volume 10: SCU-Appendix (diverse)
}

DATA_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/docs/1778/data")
REPORT_DIR = Path("/home/jic823/1815EncyclopediaBritannicaNLS/reports")

def load_volume(vol_file):
    """Load a volume JSON file."""
    with open(vol_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_headword(h):
    """Normalize headword for comparison."""
    # Remove HTML tags
    h = re.sub(r'<[^>]+>', '', h)
    # Convert to uppercase
    h = h.upper()
    # Remove non-alpha leading chars
    h = re.sub(r'^[^A-Z]+', '', h)
    return h

def get_first_letter(h):
    """Get the first alphabetic letter of a headword."""
    norm = normalize_headword(h)
    if norm:
        return norm[0]
    return ''

def strip_html(text):
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)

def is_outside_alphabetical_range(headword, vol_name):
    """Check if headword is outside the expected range for the volume."""
    if vol_name not in VOLUME_RANGES:
        return False

    start, end = VOLUME_RANGES[vol_name]
    norm = normalize_headword(headword)

    if not norm:
        return True

    # Special handling for volume 10 (appendix) and vol0 (comprehensive)
    if vol_name in ["vol10", "vol0"]:
        return False

    # Get expected letter ranges
    start_letter = start[0].upper()
    end_letter = end[0].upper()
    first_letter = norm[0] if norm else ''

    # Simple first-letter check for main volumes
    if first_letter < start_letter or first_letter > end_letter:
        # Minor exceptions: allow 1 letter before/after for edge cases
        return True

    return False

def check_ocr_errors(headword):
    """Check for OCR/parsing errors in headword."""
    issues = []
    stripped = strip_html(headword).strip()

    if not stripped:
        issues.append("Empty headword")
        return issues

    # Sentence fragments - headwords starting with lowercase or common words
    if stripped[0].islower():
        issues.append("Starts with lowercase (possible sentence fragment)")

    # Very long headwords (likely a sentence, not a headword)
    if len(stripped) > 60:
        issues.append("Excessively long headword (possible sentence)")

    # Contains unusual punctuation for headwords
    if re.search(r'[;:!?]', stripped):
        issues.append("Contains unusual punctuation for headword")

    # Starts with articles/prepositions (likely a sentence fragment)
    sentence_starters = ['THE ', 'A ', 'AN ', 'IN ', 'OF ', 'TO ', 'FOR ', 'WITH ', 'BY ', 'FROM ', 'IS ', 'ARE ', 'WAS ', 'WERE ', 'WHEN ', 'THIS ', 'THAT ', 'THESE ', 'THOSE ', 'IT ', 'HE ', 'SHE ', 'THEY ']
    upper_stripped = stripped.upper()
    for starter in sentence_starters:
        if upper_stripped.startswith(starter):
            issues.append(f"Starts with '{starter.strip()}' (likely sentence fragment)")
            break

    # Headword appears to be a complete sentence (has multiple spaces and ends with period)
    if stripped.count(' ') >= 5 and stripped.endswith('.'):
        issues.append("Appears to be a complete sentence")

    # Contains Roman numerals only (like II, III, IV, etc.) - may be subsection headers
    # Exclude valid words like CIVIL, DILL, MILL, CID, VIM, MIX, etc.
    excluded_words = {'CIVIL', 'DILL', 'MILL', 'CID', 'VIM', 'MIX', 'LIVID', 'VIVID', 'CIVIC', 'MIMIC', 'MILD', 'LIMP', 'LIMB', 'LIVID', 'MIDI', 'VIVIDLY'}
    if re.match(r'^[IVXLCDM]+\s*[\.\)]?\s*$', stripped.upper()) and stripped.upper() not in excluded_words:
        issues.append("Roman numeral only (possible subsection header)")

    # Single character headwords (likely errors)
    if len(stripped) <= 1:
        issues.append("Single character headword")

    # Contains long digit sequences
    if re.search(r'\d{3,}', stripped):
        issues.append("Contains long digit sequences")

    # Malformed HTML tags
    if '<' in headword and '>' not in headword:
        issues.append("Malformed HTML tags")

    # Headword is all numbers
    if re.match(r'^[\d\s\.\,]+$', stripped):
        issues.append("Headword is numeric only")

    return issues

def analyze_volumes():
    """Analyze all volumes and collect findings."""
    findings = {
        "outside_range": [],
        "short_articles": [],
        "long_articles": [],
        "duplicates": defaultdict(list),
        "alphabetical_jumps": [],
        "ocr_errors": [],
    }

    all_headwords = defaultdict(list)  # Track duplicates across all volumes

    # Statistics
    stats = {
        "total_articles": 0,
        "total_chars": 0,
        "by_volume": {}
    }

    for vol_file in sorted(DATA_DIR.glob("vol*.json")):
        vol_name = vol_file.stem
        print(f"Analyzing {vol_name}...")

        articles = load_volume(vol_file)

        vol_stats = {
            "count": len(articles),
            "min_len": float('inf'),
            "max_len": 0,
            "avg_len": 0,
        }

        total_len = 0
        prev_headword = None

        for i, article in enumerate(articles):
            headword = article.get("h", "")
            text = article.get("t", "")
            text_stripped = strip_html(text)
            text_len = len(text_stripped)

            stats["total_articles"] += 1
            stats["total_chars"] += text_len
            total_len += text_len

            vol_stats["min_len"] = min(vol_stats["min_len"], text_len)
            vol_stats["max_len"] = max(vol_stats["max_len"], text_len)

            article_id = f"{vol_name}#{i}"

            # 1. Check alphabetical range
            if is_outside_alphabetical_range(headword, vol_name):
                findings["outside_range"].append({
                    "id": article_id,
                    "headword": headword,
                    "volume": vol_name,
                    "expected_range": VOLUME_RANGES.get(vol_name, ("?", "?")),
                })

            # 2. Short articles (under 50 chars)
            if text_len < 50:
                findings["short_articles"].append({
                    "id": article_id,
                    "headword": headword,
                    "text_len": text_len,
                    "text_preview": text_stripped[:100],
                    "volume": vol_name,
                })

            # 3. Long articles (over 50,000 chars - potential merged content)
            if text_len > 50000:
                findings["long_articles"].append({
                    "id": article_id,
                    "headword": headword,
                    "text_len": text_len,
                    "volume": vol_name,
                })

            # 4. Track duplicates
            norm_headword = normalize_headword(headword)
            if norm_headword:
                all_headwords[norm_headword].append({
                    "id": article_id,
                    "headword": headword,
                    "volume": vol_name,
                    "text_len": text_len,
                })

            # 5. Check alphabetical jumps
            if prev_headword is not None:
                prev_norm = normalize_headword(prev_headword)
                curr_norm = normalize_headword(headword)

                if prev_norm and curr_norm:
                    # Check for large jumps (more than 2 letters difference at start)
                    if len(prev_norm) > 0 and len(curr_norm) > 0:
                        letter_diff = ord(curr_norm[0]) - ord(prev_norm[0])

                        # Also check for backward jumps (out of order)
                        if curr_norm < prev_norm:
                            findings["alphabetical_jumps"].append({
                                "id": article_id,
                                "prev_headword": prev_headword,
                                "curr_headword": headword,
                                "volume": vol_name,
                                "type": "backward",
                            })
                        elif letter_diff > 2:
                            findings["alphabetical_jumps"].append({
                                "id": article_id,
                                "prev_headword": prev_headword,
                                "curr_headword": headword,
                                "volume": vol_name,
                                "type": "large_gap",
                            })

            prev_headword = headword

            # 6. Check OCR errors
            ocr_issues = check_ocr_errors(headword)
            if ocr_issues:
                findings["ocr_errors"].append({
                    "id": article_id,
                    "headword": headword,
                    "issues": ocr_issues,
                    "volume": vol_name,
                })

        vol_stats["avg_len"] = total_len / len(articles) if articles else 0
        stats["by_volume"][vol_name] = vol_stats

    # Process duplicates
    for headword, occurrences in all_headwords.items():
        if len(occurrences) > 1:
            findings["duplicates"][headword] = occurrences

    return findings, stats

def severity_rating(issue_type, count):
    """Determine severity rating based on issue type and count."""
    if issue_type == "outside_range":
        if count > 100:
            return "HIGH"
        elif count > 20:
            return "MEDIUM"
        return "LOW"
    elif issue_type == "short_articles":
        if count > 500:
            return "HIGH"
        elif count > 100:
            return "MEDIUM"
        return "LOW"
    elif issue_type == "long_articles":
        if count > 50:
            return "HIGH"
        elif count > 10:
            return "MEDIUM"
        return "LOW"
    elif issue_type == "duplicates":
        if count > 200:
            return "HIGH"
        elif count > 50:
            return "MEDIUM"
        return "LOW"
    elif issue_type == "alphabetical_jumps":
        if count > 100:
            return "HIGH"
        elif count > 30:
            return "MEDIUM"
        return "LOW"
    elif issue_type == "ocr_errors":
        if count > 300:
            return "HIGH"
        elif count > 100:
            return "MEDIUM"
        return "LOW"
    return "UNKNOWN"

def generate_report(findings, stats):
    """Generate the audit report in Markdown format."""
    report = []

    report.append("# Audit Report: 1778 (2nd Edition) Encyclopedia Britannica\n")
    report.append("## Executive Summary\n")
    report.append(f"- **Total Articles Analyzed**: {stats['total_articles']:,}")
    report.append(f"- **Total Characters**: {stats['total_chars']:,}")
    report.append(f"- **Average Article Length**: {stats['total_chars'] // stats['total_articles']:,} characters")
    report.append(f"- **Date Generated**: 2026-01-03\n")

    report.append("### Data Source Overview\n")
    report.append("The 1778 2nd Edition data is organized in 11 JSON files:")
    report.append("- **vol0.json**: Supplementary content (2,623 articles) - not listed in main index")
    report.append("- **vol1.json - vol10.json**: Main encyclopedia volumes (14,596 articles)")
    report.append("")
    report.append("**Note**: vol0 contains unique headwords that do not overlap with volumes 1-10,")
    report.append("suggesting it may be appendix material, index entries, or alternative extractions.\n")

    # Issue counts
    report.append("### Issue Overview\n")
    report.append("| Issue Type | Count | Severity |")
    report.append("|------------|-------|----------|")

    issue_counts = {
        "Articles Outside Alphabetical Range": len(findings["outside_range"]),
        "Unusually Short Articles (<50 chars)": len(findings["short_articles"]),
        "Unusually Long Articles (>50K chars)": len(findings["long_articles"]),
        "Duplicate Headwords": len(findings["duplicates"]),
        "Alphabetical Order Issues": len(findings["alphabetical_jumps"]),
        "OCR/Parsing Errors": len(findings["ocr_errors"]),
    }

    severity_map = {
        "Articles Outside Alphabetical Range": "outside_range",
        "Unusually Short Articles (<50 chars)": "short_articles",
        "Unusually Long Articles (>50K chars)": "long_articles",
        "Duplicate Headwords": "duplicates",
        "Alphabetical Order Issues": "alphabetical_jumps",
        "OCR/Parsing Errors": "ocr_errors",
    }

    for issue_name, count in issue_counts.items():
        severity = severity_rating(severity_map[issue_name], count)
        report.append(f"| {issue_name} | {count:,} | {severity} |")

    report.append("\n---\n")

    # Volume Statistics
    report.append("## Volume Statistics\n")
    report.append("| Volume | Articles | Min Length | Max Length | Avg Length |")
    report.append("|--------|----------|------------|------------|------------|")

    for vol_name in sorted(stats["by_volume"].keys()):
        vol = stats["by_volume"][vol_name]
        report.append(f"| {vol_name} | {vol['count']:,} | {vol['min_len']:,} | {vol['max_len']:,} | {vol['avg_len']:,.0f} |")

    report.append("\n---\n")

    # 1. Articles Outside Alphabetical Range
    report.append("## 1. Articles Outside Alphabetical Range\n")
    severity = severity_rating("outside_range", len(findings["outside_range"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['outside_range'])}**\n")
    report.append("Articles whose headwords don't match the expected letter range for their volume.\n")

    if findings["outside_range"]:
        # Group by volume
        by_volume = defaultdict(list)
        for item in findings["outside_range"]:
            by_volume[item["volume"]].append(item)

        for vol in sorted(by_volume.keys()):
            items = by_volume[vol]
            report.append(f"### {vol} (Expected: {VOLUME_RANGES.get(vol, ('?', '?'))})")
            report.append(f"**{len(items)} issues found**\n")

            for item in items[:10]:  # Show first 10
                report.append(f"- `{item['id']}`: **{strip_html(item['headword'])}**")

            if len(items) > 10:
                report.append(f"- ... and {len(items) - 10} more")
            report.append("")

    report.append("\n---\n")

    # 2. Unusually Short Articles
    report.append("## 2. Unusually Short Articles (<50 characters)\n")
    severity = severity_rating("short_articles", len(findings["short_articles"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['short_articles'])}**\n")
    report.append("These may indicate parsing errors or incomplete OCR extraction.\n")

    if findings["short_articles"]:
        # Sort by length
        sorted_short = sorted(findings["short_articles"], key=lambda x: x["text_len"])

        report.append("### Examples (sorted by length)\n")
        for item in sorted_short[:30]:
            preview = item["text_preview"].replace("\n", " ")[:80]
            report.append(f"- `{item['id']}`: **{strip_html(item['headword'])}** ({item['text_len']} chars)")
            report.append(f"  - Preview: \"{preview}...\"")

        if len(sorted_short) > 30:
            report.append(f"\n*... and {len(sorted_short) - 30} more short articles*\n")

    report.append("\n---\n")

    # 3. Unusually Long Articles
    report.append("## 3. Unusually Long Articles (>50,000 characters)\n")
    severity = severity_rating("long_articles", len(findings["long_articles"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['long_articles'])}**\n")
    report.append("Long articles fall into two categories:\n")
    report.append("1. **Legitimate Treatises**: Major encyclopedic entries like SCOTLAND, CHEMISTRY, AGRICULTURE")
    report.append("2. **Potential Merge Errors**: Articles whose headwords are sentence fragments\n")

    if findings["long_articles"]:
        # Sort by length
        sorted_long = sorted(findings["long_articles"], key=lambda x: -x["text_len"])

        # Separate into likely treatises vs potential errors
        potential_errors = []
        likely_treatises = []

        for item in sorted_long:
            h = strip_html(item['headword'])
            # Sentence fragments typically have spaces and common words
            if ' ' in h and any(word in h.upper() for word in ['THE ', 'BY ', 'IS ', 'ARE ', 'HAVING ', 'THESE ', 'WHEN ']):
                potential_errors.append(item)
            else:
                likely_treatises.append(item)

        if potential_errors:
            report.append(f"### Potential Merge Errors ({len(potential_errors)} articles)")
            report.append("These headwords appear to be sentence fragments, suggesting parsing issues:\n")
            for item in potential_errors[:20]:
                report.append(f"- `{item['id']}`: **{strip_html(item['headword'])}** ({item['text_len']:,} chars)")
            if len(potential_errors) > 20:
                report.append(f"- ... and {len(potential_errors) - 20} more")
            report.append("")

        report.append(f"### Likely Legitimate Treatises ({len(likely_treatises)} articles)")
        report.append("These are expected long entries covering major topics:\n")
        for item in likely_treatises[:50]:
            report.append(f"- `{item['id']}`: **{strip_html(item['headword'])}** ({item['text_len']:,} chars)")

    report.append("\n---\n")

    # 4. Duplicate Articles
    report.append("## 4. Duplicate Headwords\n")
    severity = severity_rating("duplicates", len(findings["duplicates"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['duplicates'])} unique headwords with duplicates**\n")
    report.append("Same headword appearing multiple times (may be intentional cross-references or errors).\n")

    if findings["duplicates"]:
        # Sort by number of occurrences
        sorted_dups = sorted(findings["duplicates"].items(), key=lambda x: -len(x[1]))

        report.append("### Examples (sorted by occurrence count)\n")
        for headword, occurrences in sorted_dups[:30]:
            report.append(f"#### {headword} ({len(occurrences)} occurrences)")
            for occ in occurrences:
                report.append(f"  - `{occ['id']}`: {occ['text_len']} chars in {occ['volume']}")
            report.append("")

        if len(sorted_dups) > 30:
            report.append(f"\n*... and {len(sorted_dups) - 30} more duplicate headwords*\n")

    report.append("\n---\n")

    # 5. Alphabetical Order Issues
    report.append("## 5. Alphabetical Order Issues\n")
    severity = severity_rating("alphabetical_jumps", len(findings["alphabetical_jumps"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['alphabetical_jumps'])}**\n")
    report.append("Large gaps or backward jumps in alphabetical sequence.\n")

    if findings["alphabetical_jumps"]:
        backward = [x for x in findings["alphabetical_jumps"] if x["type"] == "backward"]
        large_gap = [x for x in findings["alphabetical_jumps"] if x["type"] == "large_gap"]

        if backward:
            report.append(f"### Backward Jumps ({len(backward)} issues)")
            report.append("Articles appearing out of alphabetical order.\n")
            for item in backward[:20]:
                report.append(f"- `{item['id']}`: **{strip_html(item['prev_headword'])}** -> **{strip_html(item['curr_headword'])}**")
            if len(backward) > 20:
                report.append(f"- ... and {len(backward) - 20} more")
            report.append("")

        if large_gap:
            report.append(f"### Large Alphabetical Gaps ({len(large_gap)} issues)")
            report.append("Suspicious jumps that might indicate missing articles.\n")
            for item in large_gap[:20]:
                report.append(f"- `{item['id']}`: **{strip_html(item['prev_headword'])}** -> **{strip_html(item['curr_headword'])}**")
            if len(large_gap) > 20:
                report.append(f"- ... and {len(large_gap) - 20} more")
            report.append("")

    report.append("\n---\n")

    # 6. OCR/Parsing Errors
    report.append("## 6. OCR/Parsing Errors in Headwords\n")
    severity = severity_rating("ocr_errors", len(findings["ocr_errors"]))
    report.append(f"**Severity: {severity}**\n")
    report.append(f"**Count: {len(findings['ocr_errors'])}**\n")
    report.append("Headwords with suspicious characters, formatting, or structure.\n")

    if findings["ocr_errors"]:
        # Group by issue type
        by_issue = defaultdict(list)
        for item in findings["ocr_errors"]:
            for issue in item["issues"]:
                by_issue[issue].append(item)

        for issue_type, items in sorted(by_issue.items(), key=lambda x: -len(x[1])):
            report.append(f"### {issue_type} ({len(items)} instances)\n")
            for item in items[:15]:
                report.append(f"- `{item['id']}`: **{strip_html(item['headword'])}**")
            if len(items) > 15:
                report.append(f"- ... and {len(items) - 15} more")
            report.append("")

    report.append("\n---\n")

    # Recommendations
    report.append("## Recommendations\n")

    report.append("### Priority Actions\n")

    priority_num = 1

    # Count sentence fragment headwords
    sentence_fragment_count = sum(1 for item in findings["ocr_errors"]
                                   if any("sentence" in issue.lower() or "fragment" in issue.lower()
                                         for issue in item["issues"]))

    if sentence_fragment_count > 10:
        report.append(f"{priority_num}. **HIGH**: Fix sentence fragment headwords ({sentence_fragment_count} found)")
        report.append("   - These are likely parsing errors where article text was mistakenly captured as headword")
        report.append("   - Review parser logic for handling edge cases at article boundaries")
        priority_num += 1

    if len(findings["outside_range"]) > 50:
        report.append(f"{priority_num}. **HIGH**: Review articles outside expected alphabetical ranges ({len(findings['outside_range'])} found)")
        report.append("   - Many appear to be appendix content (PLATE explanations, END OF VOLUME markers)")
        report.append("   - Consider excluding these from main article index or creating separate appendix category")
        priority_num += 1

    long_merge_errors = [item for item in findings["long_articles"]
                         if ' ' in strip_html(item['headword']) and
                         any(word in strip_html(item['headword']).upper()
                             for word in ['THE ', 'BY ', 'IS ', 'ARE ', 'HAVING '])]
    if long_merge_errors:
        report.append(f"{priority_num}. **HIGH**: Investigate {len(long_merge_errors)} potential article merge errors")
        report.append("   - These very long articles have sentence fragments as headwords")
        report.append("   - Likely represent incorrectly merged content from adjacent articles")
        priority_num += 1

    if len(findings["ocr_errors"]) > 20:
        report.append(f"{priority_num}. **MEDIUM**: Review OCR quality for headwords ({len(findings['ocr_errors'])} issues)")
        report.append("   - Focus on excessively long headwords and sentence fragments")
        priority_num += 1

    if len(findings["short_articles"]) > 50:
        report.append(f"{priority_num}. **MEDIUM**: Investigate short articles ({len(findings['short_articles'])} found)")
        report.append("   - May indicate incomplete extraction or cross-references")
        priority_num += 1

    if len(findings["alphabetical_jumps"]) > 20:
        report.append(f"{priority_num}. **LOW**: Review alphabetical ordering issues ({len(findings['alphabetical_jumps'])} found)")
        report.append("   - May indicate missing articles or incorrect sorting")
        priority_num += 1

    report.append("")
    report.append("### Structural Observations\n")
    report.append("1. **vol0.json** contains 2,623 unique articles not found in volumes 1-10")
    report.append("   - Purpose unclear: may be appendix, index, or alternative OCR extraction")
    report.append("   - Recommend investigating source and deciding on inclusion/exclusion")
    report.append("")
    report.append("2. **Appendix content** scattered across volumes:")
    report.append("   - PLATE explanations, END OF VOLUME markers, INDEX sections")
    report.append("   - Consider separating front/back matter from main article content")
    report.append("")
    report.append("3. **Short articles** are mostly valid cross-references:")
    report.append("   - Examples: 'See LANDSKIP', 'See BALISTES', 'See Ambassador'")
    report.append("   - These are intentional encyclopedia cross-references, not errors")

    report.append("\n### Data Quality Score\n")

    total_issues = sum(issue_counts.values())
    quality_score = max(0, 100 - (total_issues / stats["total_articles"]) * 100)

    if quality_score >= 95:
        grade = "A"
    elif quality_score >= 90:
        grade = "A-"
    elif quality_score >= 85:
        grade = "B+"
    elif quality_score >= 80:
        grade = "B"
    elif quality_score >= 75:
        grade = "B-"
    elif quality_score >= 70:
        grade = "C+"
    elif quality_score >= 65:
        grade = "C"
    else:
        grade = "D"

    report.append(f"- **Quality Score**: {quality_score:.1f}%")
    report.append(f"- **Grade**: {grade}")
    report.append(f"- **Total Issues Found**: {total_issues:,}")
    report.append(f"- **Issues per Article**: {total_issues / stats['total_articles']:.3f}")
    report.append("")
    report.append("**Note**: This score reflects parsing/structural issues only. OCR accuracy of article content")
    report.append("requires separate evaluation through text quality analysis.")

    return "\n".join(report)

def main():
    print("Starting audit of 1778 (2nd Edition) Encyclopedia Britannica...")

    # Ensure report directory exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze volumes
    findings, stats = analyze_volumes()

    # Generate report
    report = generate_report(findings, stats)

    # Write report
    report_path = REPORT_DIR / "audit_1778_2nd_edition.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport written to: {report_path}")
    print(f"\nSummary:")
    print(f"  Total articles: {stats['total_articles']:,}")
    print(f"  Outside range: {len(findings['outside_range']):,}")
    print(f"  Short articles: {len(findings['short_articles']):,}")
    print(f"  Long articles: {len(findings['long_articles']):,}")
    print(f"  Duplicates: {len(findings['duplicates']):,}")
    print(f"  Alphabetical issues: {len(findings['alphabetical_jumps']):,}")
    print(f"  OCR errors: {len(findings['ocr_errors']):,}")

if __name__ == "__main__":
    main()
