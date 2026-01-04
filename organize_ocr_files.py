#!/usr/bin/env python3
"""
Organize all OCR files into a single directory with clear, descriptive filenames.

Output format: britannica_{edition}ed_{year}_vol{nn}_{range}.jsonl
Example: britannica_1st_1771_vol01_A-B.jsonl
"""

import json
import os
import re
import shutil
from pathlib import Path

OUTPUT_DIR = Path("ocr_organized")

# Edition info
EDITIONS = {
    1771: "1st",
    1778: "2nd",
    1797: "3rd",
    1810: "4th",
    1815: "5th",
    1823: "6th",
    1842: "7th",
    1860: "8th",
}


def extract_volume_info(text: str) -> tuple:
    """Extract volume number and letter range from text header."""
    # Look for volume patterns
    vol_patterns = [
        r'VOL(?:UME)?\.?\s*([IVXLCDM]+|\d+)',
        r'VOLUME\s+([IVXLCDM]+|\d+)',
    ]

    vol_num = None
    for pattern in vol_patterns:
        match = re.search(pattern, text[:2000], re.IGNORECASE)
        if match:
            vol_str = match.group(1)
            # Convert Roman numerals
            if vol_str.isdigit():
                vol_num = int(vol_str)
            else:
                roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
                try:
                    result = 0
                    prev = 0
                    for c in vol_str.upper()[::-1]:
                        curr = roman_map.get(c, 0)
                        if curr < prev:
                            result -= curr
                        else:
                            result += curr
                        prev = curr
                    vol_num = result
                except:
                    pass
            break

    # Look for letter range (e.g., "A-AME", "GOT-HYD")
    range_patterns = [
        r',\s*([A-Z](?:[A-Z]*)?)\s*[-–—]\s*([A-Z][A-Za-z]*)',
        r'([A-Z]{1,3})\s*[-–—]\s*([A-Z][A-Za-z]{2,})',
    ]

    letter_range = None
    for pattern in range_patterns:
        match = re.search(pattern, text[:500])
        if match:
            start = match.group(1)[:3]
            end = match.group(2)[:3]
            letter_range = f"{start}-{end}"
            break

    return vol_num, letter_range


def detect_edition(text: str) -> int:
    """Detect which edition based on text content."""
    patterns = [
        (r'FIRST\s+EDITION', 1771),
        (r'SECOND\s+EDITION', 1778),
        (r'THIRD\s+EDITION', 1797),
        (r'FOURTH\s+EDITION', 1810),
        (r'FIFTH\s+EDITION', 1815),
        (r'THE FIFTH EDITION', 1815),
        (r'SIXTH\s+EDITION', 1823),
        (r'THE SIXTH EDITION', 1823),
        (r'SEVENTH\s+EDITION', 1842),
        (r'EIGHTH\s+EDITION', 1860),
        # Year-based fallback
        (r'\b1771\b', 1771),
        (r'\b1778\b', 1778),
        (r'\b1797\b', 1797),
        (r'\b1810\b', 1810),
        (r'\b1815\b', 1815),
        (r'\b1823\b', 1823),
        (r'M\.?D\.?C\.?C\.?C\.?X\.?L\.?I\.?I', 1842),  # MDCCCXLII
        (r'\b1842\b', 1842),
        (r'MDCCCLX\b', 1860),
        (r'\b1860\b', 1860),
    ]

    header = text[:2000].upper()
    for pattern, year in patterns:
        if re.search(pattern, header, re.IGNORECASE):
            return year
    return None


def process_jsonl_file(filepath: Path, known_edition: int = None) -> list:
    """Process a JSONL file and return list of (edition, vol, range, text) tuples."""
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                text = entry.get('text', '')
                if len(text) < 500:
                    continue

                edition = known_edition or detect_edition(text)
                vol_num, letter_range = extract_volume_info(text)

                if edition:
                    results.append({
                        'edition': edition,
                        'vol': vol_num,
                        'range': letter_range,
                        'text': text,
                        'source': filepath.name,
                    })
            except json.JSONDecodeError:
                continue

    return results


def process_md_file(filepath: Path, known_edition: int = None) -> list:
    """Process a Markdown file and return list of entries."""
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    if len(text) < 500:
        return results

    edition = known_edition or detect_edition(text)
    vol_num, letter_range = extract_volume_info(text)

    # Try to get range from filename
    if not letter_range:
        match = re.search(r'Volume \d+,\s*([A-Z]+)-([A-Z]+)', filepath.name)
        if match:
            letter_range = f"{match.group(1)[:3]}-{match.group(2)[:3]}"

    if edition:
        results.append({
            'edition': edition,
            'vol': vol_num,
            'range': letter_range,
            'text': text,
            'source': filepath.name,
        })

    return results


def process_json_file(filepath: Path) -> list:
    """Process a JSON file (array format) and return list of entries."""
    results = []

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list) and data:
        text = data[0].get('text', '')
        if len(text) > 500:
            edition = detect_edition(text)
            vol_num, letter_range = extract_volume_info(text)

            # Get range from filename
            if not letter_range:
                match = re.search(r'Volume \d+,\s*([A-Z][A-Za-z]*)-([A-Z][A-Za-z]*)', filepath.name)
                if match:
                    letter_range = f"{match.group(1)[:3]}-{match.group(2)[:3]}"

            if edition:
                results.append({
                    'edition': edition,
                    'vol': vol_num,
                    'range': letter_range,
                    'text': text,
                    'source': filepath.name,
                })

    return results


def main():
    print("=" * 60)
    print("ORGANIZING OCR FILES")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_entries = []

    # Process each source directory
    sources = [
        ("ocr_results/1771_britannica_1st", "*.jsonl", 1771),
        ("ocr_results/1778_britannica_2nd", "*.jsonl", 1778),
        ("ocr_results/1797_britannica_3rd", "*.md", 1797),
        ("ocr_results/1797_britannica_3rd", "*.jsonl", 1797),
        ("ocr_results/1810_britannica_4th", "*.jsonl", 1810),
        ("json", "*.json", None),  # Mixed 5th/6th
        ("ocr_results/britannica_pipeline_batch", "*.jsonl", None),  # Mixed 7th/8th
    ]

    for src_dir, pattern, known_edition in sources:
        src_path = Path(src_dir)
        if not src_path.exists():
            continue

        print(f"\nProcessing {src_dir}...")

        for filepath in sorted(src_path.glob(pattern)):
            if pattern == "*.jsonl":
                entries = process_jsonl_file(filepath, known_edition)
            elif pattern == "*.md":
                entries = process_md_file(filepath, known_edition)
            elif pattern == "*.json":
                entries = process_json_file(filepath)
            else:
                continue

            all_entries.extend(entries)
            if entries:
                print(f"  {filepath.name}: {len(entries)} entries")

    # Group by edition
    by_edition = {}
    for entry in all_entries:
        ed = entry['edition']
        if ed not in by_edition:
            by_edition[ed] = []
        by_edition[ed].append(entry)

    # Write organized files
    print("\n" + "=" * 60)
    print("WRITING ORGANIZED FILES")
    print("=" * 60)

    for edition_year in sorted(by_edition.keys()):
        entries = by_edition[edition_year]
        ed_name = EDITIONS.get(edition_year, str(edition_year))

        # Sort by volume
        entries.sort(key=lambda x: (x['vol'] or 99, x['range'] or 'ZZZ'))

        # Write each entry to a separate file
        vol_counts = {}
        for entry in entries:
            vol = entry['vol'] or 0
            vol_counts[vol] = vol_counts.get(vol, 0) + 1
            part = vol_counts[vol]

            range_str = entry['range'] or "unknown"
            range_str = range_str.replace('/', '-').replace(' ', '')

            if part > 1:
                filename = f"britannica_{ed_name}_{edition_year}_vol{vol:02d}_part{part}_{range_str}.jsonl"
            else:
                filename = f"britannica_{ed_name}_{edition_year}_vol{vol:02d}_{range_str}.jsonl"

            output_path = OUTPUT_DIR / filename

            # Write as JSONL
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'edition': edition_year,
                    'edition_name': ed_name,
                    'volume': vol,
                    'range': entry['range'],
                    'source_file': entry['source'],
                    'text': entry['text'],
                }, f, ensure_ascii=False)
                f.write('\n')

        print(f"  {ed_name} ({edition_year}): {len(entries)} files")

    # Summary
    print("\n" + "=" * 60)
    total_files = len(list(OUTPUT_DIR.glob("*.jsonl")))
    print(f"COMPLETE: {total_files} organized files in {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
