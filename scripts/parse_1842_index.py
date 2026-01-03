#!/usr/bin/env python3
"""
Parse the 1842 Encyclopaedia Britannica General Index (Volume 22).

The index provides the canonical taxonomy for the encyclopedia, mapping
topics to specific volumes and pages across the 21 content volumes.

Index Entry Types:
- Main entries (CAPS): "ASTRONOMY, II. 449" = Article in Vol II, page 449
- Cross-references: "ABANGA. See ADY." = Redirect to another entry
- Sub-entries (lowercase): "Abaddé Arabs, II. 225" = Sub-topic
- Multi-references: "INK, XII. 277; VIII. 311" = Multiple locations

Output: JSONL with structured entries for Neo4j knowledge graph.
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# Patterns for parsing index entries
MAIN_ENTRY_PATTERN = re.compile(
    r'^([A-Z][A-Z\'\-\s]+(?:\([^)]+\))?),\s*'  # HEADWORD (possibly with parenthetical)
    r'(.+)$'  # Rest of line with references
)

# Volume reference pattern: "II. 449" or "XII. 277, 280"
VOL_REF_PATTERN = re.compile(
    r'([IVXLCDM]+)\.\s*(\d+(?:,\s*\d+)*)'
)

# Cross-reference pattern: "See ASTRONOMY" or "See Astronomy"
SEE_PATTERN = re.compile(
    r'See\s+([A-Za-z][A-Za-z\'\-\s]+)',
    re.IGNORECASE
)

# Page separator for index columns
PAGE_SEPARATOR = re.compile(r'^\d+[-–—]+$')

# Page header pattern (letter ranges like "ABA—ABB")
PAGE_HEADER = re.compile(r'^[A-Z]{2,3}[-–—][A-Z]{2,3}$')


@dataclass
class IndexEntry:
    """A single index entry."""
    term: str
    entry_type: str  # 'main', 'sub', 'cross_ref'
    references: list  # [{vol: int, pages: [int]}]
    see_also: list  # Cross-references
    raw_line: str


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev = 0
    for char in reversed(roman.upper()):
        curr = values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    return result


def parse_references(ref_text: str) -> list:
    """Parse volume/page references from text.

    Examples:
        "II. 449" -> [{"vol": 2, "pages": [449]}]
        "XII. 277, 280" -> [{"vol": 12, "pages": [277, 280]}]
        "II. 225; IX. 377" -> [{"vol": 2, "pages": [225]}, {"vol": 9, "pages": [377]}]
    """
    refs = []
    for match in VOL_REF_PATTERN.finditer(ref_text):
        vol_roman = match.group(1)
        pages_str = match.group(2)

        vol_num = roman_to_int(vol_roman)
        pages = [int(p.strip()) for p in pages_str.split(',') if p.strip().isdigit()]

        if pages:
            refs.append({"vol": vol_num, "pages": pages})

    return refs


def parse_see_also(text: str) -> list:
    """Extract cross-references from text."""
    see_refs = []
    for match in SEE_PATTERN.finditer(text):
        ref = match.group(1).strip().rstrip('.')
        if ref:
            see_refs.append(ref.upper())
    return see_refs


def parse_line(line: str) -> Optional[IndexEntry]:
    """Parse a single index line."""
    line = line.strip()

    # Skip empty lines, page separators, headers
    if not line:
        return None
    if PAGE_SEPARATOR.match(line):
        return None
    if PAGE_HEADER.match(line):
        return None
    if line.startswith('!['):  # Image captions
        return None
    if line.startswith('#'):  # Markdown headers
        return None

    # Detect entry type based on capitalization
    first_word = line.split(',')[0].split('.')[0].strip()

    # Main entry: starts with ALL CAPS word
    if first_word and first_word.isupper() and len(first_word) > 1:
        match = MAIN_ENTRY_PATTERN.match(line)
        if match:
            term = match.group(1).strip()
            rest = match.group(2)

            refs = parse_references(rest)
            see_also = parse_see_also(rest)

            return IndexEntry(
                term=term,
                entry_type='main',
                references=refs,
                see_also=see_also,
                raw_line=line
            )
        else:
            # Check for pure cross-reference: "ABANGA. See ADY."
            see_match = re.match(r'^([A-Z][A-Z\'\-\s]+)\.\s*See\s+(.+)', line, re.IGNORECASE)
            if see_match:
                term = see_match.group(1).strip()
                see_also = parse_see_also(line)
                return IndexEntry(
                    term=term,
                    entry_type='cross_ref',
                    references=[],
                    see_also=see_also,
                    raw_line=line
                )

    # Sub-entry: starts with lowercase or mixed case
    elif first_word and not first_word.isupper():
        # Extract the term (up to first comma or period)
        term_match = re.match(r'^([^,\.]+)', line)
        if term_match:
            term = term_match.group(1).strip()
            refs = parse_references(line)
            see_also = parse_see_also(line)

            if refs or see_also:
                return IndexEntry(
                    term=term,
                    entry_type='sub',
                    references=refs,
                    see_also=see_also,
                    raw_line=line
                )

    return None


def parse_index_file(filepath: Path) -> list:
    """Parse the entire index file."""
    entries = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = parse_line(line)
                if entry:
                    entries.append(entry)
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)

    return entries


def main():
    index_file = Path("ocr_results/1842_general_index/192632444.23.md")
    output_file = Path("output_v2/index_1842.jsonl")

    if not index_file.exists():
        print(f"Index file not found: {index_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {index_file}...")
    entries = parse_index_file(index_file)

    # Statistics
    main_entries = [e for e in entries if e.entry_type == 'main']
    sub_entries = [e for e in entries if e.entry_type == 'sub']
    cross_refs = [e for e in entries if e.entry_type == 'cross_ref']

    print(f"\nParsed {len(entries)} entries:")
    print(f"  Main entries: {len(main_entries)}")
    print(f"  Sub-entries: {len(sub_entries)}")
    print(f"  Cross-references: {len(cross_refs)}")

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + '\n')

    print(f"\nOutput written to {output_file}")

    # Show sample entries
    print("\nSample main entries:")
    for entry in main_entries[:5]:
        print(f"  {entry.term}: {entry.references}")

    print("\nSample cross-references:")
    for entry in cross_refs[:5]:
        print(f"  {entry.term} -> {entry.see_also}")


if __name__ == '__main__':
    main()
