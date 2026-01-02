#!/usr/bin/env python3
"""
Unified Britannica Corpus Extractor

Extracts Articles, Sections, and Chunks with FULL provenance from OCR output.
Handles multiple OCR output formats:
- britannica_pipeline_batch: .meta.json + output_*.jsonl pairs
- Per-edition dirs (1771, 1778, etc.): output_*.jsonl with embedded metadata
- 1797 dir: Named .md files

Output Structure:
- output_v2/volumes_{year}.jsonl     - Volume metadata
- output_v2/articles_{year}.jsonl    - Full article text with page numbers
- output_v2/sections_{year}.jsonl    - Full section text with page numbers
- output_v2/chunks_{year}.jsonl      - Chunks with full provenance chain

Usage:
    python3 extract_britannica_corpus.py --list              # Show available editions
    python3 extract_britannica_corpus.py --editions 1778     # Extract one edition
    python3 extract_britannica_corpus.py --all               # Extract all editions
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VolumeInfo:
    """Metadata for a single volume."""
    volume_id: str
    edition_year: int
    edition_name: str
    edition_number: int
    volume_num: int
    letter_range: str
    page_count: int
    source_collection: str
    source_file: str
    ocr_file: str


@dataclass
class Article:
    """A single encyclopedia article with full provenance."""
    article_id: str
    headword: str
    text: str
    article_type: str
    edition_year: int
    edition_name: str
    volume_id: str
    volume_num: int
    start_page: Optional[int]
    end_page: Optional[int]
    char_start: int
    char_end: int
    sense_number: int = 1
    word_count: int = 0
    is_cross_reference: bool = False

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class Section:
    """A section within an article with full text."""
    section_id: str
    title: str
    text: str
    level: int
    index: int
    article_id: str
    parent_headword: str
    edition_year: int
    extraction_method: str
    char_start: int
    char_end: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class Chunk:
    """A text chunk with full provenance chain."""
    chunk_id: str
    text: str
    index: int
    section_id: Optional[str]
    section_title: Optional[str]
    section_index: Optional[int]
    article_id: str
    parent_headword: str
    edition_year: int
    volume_id: str
    char_start: int
    char_end: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# =============================================================================
# Edition Configuration
# =============================================================================

EDITIONS = {
    1771: {"name": "Britannica 1st", "number": 1},
    1778: {"name": "Britannica 2nd", "number": 2},
    1797: {"name": "Britannica 3rd", "number": 3},
    1810: {"name": "Britannica 4th", "number": 4},
    1815: {"name": "Britannica 5th", "number": 5},
    1823: {"name": "Britannica 6th", "number": 6},
    1842: {"name": "Britannica 7th", "number": 7},
    1860: {"name": "Britannica 8th", "number": 8},
}

MAJOR_TREATISES = {
    'AGRICULTURE', 'ALGEBRA', 'ANATOMY', 'ARCHITECTURE', 'ARITHMETIC',
    'ASTRONOMY', 'BOOK-KEEPING', 'BOTANY', 'CHEMISTRY', 'COMMERCE',
    'ELECTRICITY', 'FARRIERY', 'GEOGRAPHY', 'GEOLOGY', 'GEOMETRY',
    'GRAMMAR', 'HERALDRY', 'HISTORY', 'LAW', 'LOGIC', 'MECHANICS',
    'MEDICINE', 'METAPHYSICS', 'MIDWIFERY', 'MINERALOGY', 'MUSIC',
    'NAVIGATION', 'OPTICS', 'PHARMACY', 'PHILOSOPHY', 'POETRY',
    'RELIGION', 'SCOTLAND', 'SURGERY', 'SURVEYING', 'THEOLOGY',
}


# =============================================================================
# Parsing Patterns
# =============================================================================

ARTICLE_PATTERN = re.compile(
    r'(?:^|\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?,\s+',
    re.MULTILINE
)

TREATISE_PATTERN = re.compile(
    r'(?:\n+|---\s*\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.\s*\n+',
    re.MULTILINE
)

CROSS_REF_PATTERN = re.compile(r'^See\s+[A-Z]', re.IGNORECASE)

SECTION_PATTERNS = [
    (re.compile(r'\n(PART\s+[IVX\d]+\.?[^\n]*)', re.IGNORECASE), 1),
    (re.compile(r'\n(CHAP(?:TER)?\.?\s+[IVX\d]+\.?[^\n]*)', re.IGNORECASE), 2),
    (re.compile(r'\n(SECT(?:ION)?\.?\s+[IVX\d]+\.?[^\n]*)', re.IGNORECASE), 2),
    (re.compile(r'\n(\u00a7\s*\d+\.?[^\n]*)', re.IGNORECASE), 2),
    (re.compile(r'\n(BOOK\s+[IVX\d]+\.?[^\n]*)', re.IGNORECASE), 1),
]

FRONT_MATTER = [
    r'^OR$', r'^DICTIONARY', r'^OF$', r'^ARTS', r'^SCIENCES',
    r'^PREFACE', r'^ENCYCLOP', r'^LONDON', r'^EDINBURGH', r'^PRINTED',
]

# OCR artifacts - figure labels, repeated letters, etc.
# These patterns match headwords that are almost certainly not real articles
REPEATED_LETTERS = re.compile(r'^([A-Z])\1+$')  # AA, BBB, CCCC, etc.
SHORT_LETTER_COMBOS = re.compile(r'^[A-Z]{1,4}$')  # A, AB, ABC, ABCD - potential figure labels
SEQUENTIAL_LETTERS = re.compile(r'^[A-Z]{2,4}$')  # Check if letters are sequential (ABC, BCD, etc.)

# Known non-article words that appear in encyclopedias
NON_ARTICLE_WORDS = {
    'FINIS', 'INDEX', 'ERRATA', 'ADDENDA', 'CORRIGENDA', 'APPENDIX',
    'CONTENTS', 'PREFACE', 'INTRODUCTION', 'ADVERTISEMENT', 'DIRECTIONS',
    'SUPPLEMENT', 'PLATES', 'FIGURES', 'TABLES', 'END', 'CONCLUSION',
}

# Word fragments (common suffixes that shouldn't be standalone headwords)
WORD_FRAGMENTS = re.compile(r'^(GRAPHY|OLOGY|ATION|MENT|NESS|ICAL|IOUS|EOUS|ABLE|IBLE|MENT|TURE|SION|TION)$')


def is_sequential_letters(s: str) -> bool:
    """Check if string is sequential letters like ABC, BCD, etc."""
    if len(s) < 2 or len(s) > 5:
        return False
    for i in range(len(s) - 1):
        if ord(s[i+1]) - ord(s[i]) != 1:
            return False
    return True


def is_likely_ocr_artifact(headword: str, text: str) -> bool:
    """Check if a headword is likely an OCR artifact (figure label, etc.)."""
    word_count = len(text.split())

    # Repeated letters are almost always figure labels (AA, BBB, etc.)
    if REPEATED_LETTERS.match(headword):
        return True

    # Sequential letters are figure labels (ABC, BCD, CDE, etc.)
    if is_sequential_letters(headword):
        return True

    # Mixed repeated/sequential patterns (AABC, ABBC, etc.)
    if len(headword) <= 5 and headword.isalpha():
        unique_chars = len(set(headword))
        if unique_chars <= 3 and len(headword) >= 3:
            # Few unique chars in short string = likely figure label
            if word_count < 100:
                return True

    # Known non-article words
    if headword in NON_ARTICLE_WORDS:
        return True

    # Word fragments that aren't real headwords
    if WORD_FRAGMENTS.match(headword):
        return True

    # Very short headwords (1-4 letters) with short text are likely artifacts
    if SHORT_LETTER_COMBOS.match(headword):
        # Short text with short headword = likely figure label
        if word_count < 50:
            return True
        # Check if text looks like figure explanation
        text_lower = text[:200].lower()
        if any(x in text_lower for x in ['fig.', 'figure', 'plate', 'the point', 'the line']):
            return True

    # Very short articles (< 10 words) are suspicious
    if word_count < 10:
        # Allow cross-references which are naturally short
        if not text.lower().startswith('see '):
            return True

    return False


# =============================================================================
# Source Discovery
# =============================================================================

def detect_edition_from_text(text: str) -> Optional[int]:
    """Detect edition year from text content."""
    patterns = [
        (r'First edition', 1771),
        (r'Second edition', 1778),
        (r'Third edition', 1797),
        (r'Fourth edition', 1810),
        (r'Fifth edition', 1815),
        (r'Sixth edition', 1823),
        (r'Seventh edition', 1842),
        (r'Eighth edition', 1860),
    ]
    for pattern, year in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return year
    return None


def extract_volume_info_from_text(text: str) -> Tuple[Optional[int], str]:
    """Extract volume number and letter range from text."""
    # Try "Volume X, A-B" pattern
    m = re.search(r'Volume\s+(\d+),?\s+([A-Z]+(?:-[A-Z]+)?)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), m.group(2)

    # Try just "Volume X"
    m = re.search(r'Volume\s+(\d+)', text, re.IGNORECASE)
    if m:
        return int(m.group(1)), ""

    return None, ""


def discover_sources(ocr_base: Path) -> dict:
    """Discover all OCR sources and group by edition."""
    sources = {}

    # Check for json/ directory with 1815 data (JSON arrays, not JSONL)
    json_dir = ocr_base.parent / "json"
    if json_dir.exists():
        for json_path in json_dir.glob("Volume *.json"):
            vol_num, letter_range = extract_volume_info_from_text(json_path.name)

            # These are 1815 (5th edition) files
            edition_year = 1815
            if edition_year not in sources:
                sources[edition_year] = []

            sources[edition_year].append({
                "type": "json_array",
                "json_path": str(json_path),
                "volume_id": f"1815_v{vol_num:02d}" if vol_num else json_path.stem,
                "volume_num": vol_num or 0,
                "letter_range": letter_range,
                "page_count": 0,
                "title": f"Fifth edition - {json_path.name}",
            })

    for subdir in sorted(ocr_base.iterdir()):
        if not subdir.is_dir():
            continue

        name = subdir.name

        # Skip non-Britannica sources
        if name.startswith('17') and 'britannica' not in name.lower():
            continue

        # Check for meta.json files (pipeline_batch format)
        meta_files = list(subdir.glob("*.meta.json"))
        if meta_files:
            for meta_path in meta_files:
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)

                    title = meta.get("title", "")
                    edition_year = detect_edition_from_text(title)
                    if not edition_year:
                        continue

                    vol_num, letter_range = extract_volume_info_from_text(title)

                    # Find matching JSONL
                    nls_id = meta.get("identifier", "")
                    jsonl_path = find_jsonl_for_nls_id(subdir, nls_id)

                    if edition_year not in sources:
                        sources[edition_year] = []

                    sources[edition_year].append({
                        "type": "meta_jsonl",
                        "meta_path": str(meta_path),
                        "jsonl_path": jsonl_path,
                        "volume_id": nls_id,
                        "volume_num": vol_num or meta.get("volume_num", 0),
                        "letter_range": letter_range,
                        "page_count": meta.get("page_count", 0),
                        "title": title,
                    })
                except Exception as e:
                    logger.warning(f"Error reading {meta_path}: {e}")
            continue

        # Check for JSONL-only directories (1771, 1778, 1810 format)
        jsonl_files = list(subdir.glob("output_*.jsonl"))
        if jsonl_files and not meta_files:
            for jsonl_path in jsonl_files:
                try:
                    with open(jsonl_path) as f:
                        entry = json.loads(f.readline())

                    source_file = entry.get("metadata", {}).get("Source-File", "")
                    edition_year = detect_edition_from_text(source_file)

                    # Fallback: use directory name
                    if not edition_year and 'britannica' in name:
                        dir_match = re.search(r'(\d{4})', name)
                        if dir_match:
                            edition_year = int(dir_match.group(1))

                    if not edition_year:
                        continue

                    vol_num, letter_range = extract_volume_info_from_text(source_file)
                    page_count = entry.get("metadata", {}).get("pdf-total-pages", 0)

                    if edition_year not in sources:
                        sources[edition_year] = []

                    sources[edition_year].append({
                        "type": "jsonl_only",
                        "jsonl_path": str(jsonl_path),
                        "volume_id": Path(source_file).stem if source_file else jsonl_path.stem,
                        "volume_num": vol_num or 0,
                        "letter_range": letter_range,
                        "page_count": page_count,
                        "title": source_file,
                    })
                except Exception as e:
                    logger.warning(f"Error reading {jsonl_path}: {e}")

        # Check for named MD files (1797 format) - only if no JSONL was found
        # Note: We prefer JSONL because it has page number data
        md_files = list(subdir.glob("*.md"))
        if md_files and not jsonl_files:
            for md_path in md_files:
                edition_year = detect_edition_from_text(md_path.name)
                if not edition_year:
                    continue

                vol_num, letter_range = extract_volume_info_from_text(md_path.name)

                if edition_year not in sources:
                    sources[edition_year] = []

                sources[edition_year].append({
                    "type": "md_only",
                    "md_path": str(md_path),
                    "volume_id": md_path.stem,
                    "volume_num": vol_num or 0,
                    "letter_range": letter_range,
                    "page_count": 0,
                    "title": md_path.name,
                })

    # Deduplicate sources by (edition_year, volume_num)
    # Keep the first source with the most page_count, or prefer JSONL over MD
    for edition_year in sources:
        seen = {}
        deduped = []
        for src in sources[edition_year]:
            key = src.get("volume_num", 0)
            if key not in seen:
                seen[key] = src
                deduped.append(src)
            else:
                # Prefer source with higher page_count or JSONL type
                existing = seen[key]
                existing_score = existing.get("page_count", 0) + (10 if "jsonl" in existing.get("type", "") else 0)
                new_score = src.get("page_count", 0) + (10 if "jsonl" in src.get("type", "") else 0)
                if new_score > existing_score:
                    # Replace with better source
                    deduped.remove(existing)
                    deduped.append(src)
                    seen[key] = src
        sources[edition_year] = deduped
        logger.info(f"  {edition_year}: {len(deduped)} unique volumes (deduplicated)")

    return sources


def find_jsonl_for_nls_id(directory: Path, nls_id: str) -> Optional[str]:
    """Find the JSONL file matching an NLS ID."""
    for jsonl_path in directory.glob("output_*.jsonl"):
        try:
            with open(jsonl_path) as f:
                entry = json.loads(f.readline())
            source = entry.get("metadata", {}).get("Source-File", "")
            if nls_id in source:
                return str(jsonl_path)
        except:
            pass
    return None


# =============================================================================
# OCR Text Loading
# =============================================================================

def load_ocr_content(source: dict) -> Tuple[str, List]:
    """Load OCR text and page numbers from source."""
    source_type = source.get("type")

    if source_type in ("meta_jsonl", "jsonl_only"):
        jsonl_path = source.get("jsonl_path")
        if not jsonl_path or not Path(jsonl_path).exists():
            return "", []

        with open(jsonl_path) as f:
            for line in f:
                entry = json.loads(line)
                text = entry.get("text", "")
                page_nums = entry.get("attributes", {}).get("pdf_page_numbers", [])
                return text, page_nums

    elif source_type == "json_array":
        # 1815 format: JSON array with single entry per file
        json_path = source.get("json_path")
        if not json_path or not Path(json_path).exists():
            return "", []

        with open(json_path) as f:
            data = json.load(f)

        if data and len(data) > 0:
            entry = data[0]
            text = entry.get("text", "")
            page_nums = entry.get("attributes", {}).get("pdf_page_numbers", [])
            return text, page_nums

    elif source_type == "md_only":
        md_path = source.get("md_path")
        if not md_path or not Path(md_path).exists():
            return "", []

        with open(md_path) as f:
            text = f.read()
        return text, []

    return "", []


def get_page_for_char(char_pos: int, page_numbers: list) -> Optional[int]:
    """Look up page number for character position."""
    for start, end, page_num in page_numbers:
        if start <= char_pos <= end:
            return page_num
    return None


# =============================================================================
# Article Extraction
# =============================================================================

def is_cross_reference_context(text: str, match_start: int) -> bool:
    """Check if match is preceded by 'See' indicating a cross-reference."""
    # Look at preceding context (up to 20 chars before the newline that precedes the match)
    context_start = max(0, match_start - 20)
    context = text[context_start:match_start].lower()
    # Check for "see" at the end of context (allowing whitespace/newlines between)
    return bool(re.search(r'see\s*$', context))


def extract_articles(text: str, volume: VolumeInfo, page_numbers: list) -> List[Article]:
    """Extract articles from OCR text."""

    # Find matches
    dict_matches = [(m.start(), m.end(), m.group(1).upper().strip(), False)
                    for m in ARTICLE_PATTERN.finditer(text) if len(m.group(1)) >= 2]

    # Filter out treatise matches that are cross-references (e.g., "See\nCHEMISTRY.\n\n")
    treatise_matches = []
    for m in TREATISE_PATTERN.finditer(text):
        if len(m.group(1)) < 3:
            continue
        # Check if this is a cross-reference (preceded by "See")
        if is_cross_reference_context(text, m.start()):
            continue
        treatise_matches.append((m.start(), m.end(), m.group(1).upper().strip(), True))

    # Combine and sort
    all_entries = dict_matches + treatise_matches
    all_entries.sort(key=lambda x: x[0])

    # Deduplicate nearby
    unique = []
    for e in all_entries:
        if not unique or e[0] > unique[-1][0] + 20:
            unique.append(e)
        elif e[3] and not unique[-1][3]:
            unique[-1] = e

    # Skip front matter
    first_valid = 0
    for i, (start, end, hw, is_t) in enumerate(unique):
        if any(re.match(p, hw) for p in FRONT_MATTER):
            continue
        if 1 <= len(hw) <= 6 and hw.replace("'", "").replace("-", "").isalpha():
            first_valid = i
            break

    unique = unique[first_valid:]

    # Build articles
    articles = []
    headword_counts = {}

    for i, (start, end, headword, is_treatise) in enumerate(unique):
        text_start = end
        text_end = unique[i + 1][0] if i + 1 < len(unique) else len(text)

        article_text = text[text_start:text_end].strip()
        if len(article_text) < 10:
            continue

        # Filter out OCR artifacts (figure labels, etc.)
        if is_likely_ocr_artifact(headword, article_text):
            continue

        headword_counts[headword] = headword_counts.get(headword, 0) + 1
        sense = headword_counts[headword]

        # Merge duplicate treatises
        if headword in MAJOR_TREATISES and sense > 1 and articles:
            if articles[-1].headword == headword:
                articles[-1].text += "\n\n" + article_text
                articles[-1].char_end = text_end
                articles[-1].end_page = get_page_for_char(text_end - 1, page_numbers)
                articles[-1].word_count = len(articles[-1].text.split())
                continue

        # Determine type
        is_crossref = bool(CROSS_REF_PATTERN.match(article_text))
        if is_crossref:
            article_type = "cross_reference"
        elif is_treatise or headword in MAJOR_TREATISES or len(article_text) > 10000:
            article_type = "treatise"
        else:
            article_type = "dictionary"

        article_id = f"{volume.edition_year}_v{volume.volume_num:02d}_{headword.replace(' ', '_')}"
        if sense > 1:
            article_id += f"_s{sense}"

        articles.append(Article(
            article_id=article_id,
            headword=headword,
            text=article_text,
            article_type=article_type,
            edition_year=volume.edition_year,
            edition_name=volume.edition_name,
            volume_id=volume.volume_id,
            volume_num=volume.volume_num,
            start_page=get_page_for_char(start, page_numbers),
            end_page=get_page_for_char(text_end - 1, page_numbers),
            char_start=start,
            char_end=text_end,
            sense_number=sense,
            is_cross_reference=is_crossref,
        ))

    return articles


# =============================================================================
# Section Extraction
# =============================================================================

def extract_sections(article: Article, page_numbers: list) -> List[Section]:
    """Extract sections from an article."""
    text = article.text
    sections = []

    # Find markers
    markers = []
    for pattern, level in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            markers.append((m.start(), m.end(), m.group(1).strip(), level))

    markers.sort(key=lambda x: x[0])

    # No markers - single section
    if not markers:
        sections.append(Section(
            section_id=f"{article.article_id}_s00",
            title="[Full Article]",
            text=text,
            level=1,
            index=0,
            article_id=article.article_id,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            extraction_method="fallback",
            char_start=0,
            char_end=len(text),
            start_page=article.start_page,
            end_page=article.end_page,
        ))
        return sections

    # Add intro section
    if markers[0][0] > 100:
        intro_text = text[:markers[0][0]].strip()
        if intro_text:
            sections.append(Section(
                section_id=f"{article.article_id}_s00",
                title="Introduction",
                text=intro_text,
                level=1,
                index=0,
                article_id=article.article_id,
                parent_headword=article.headword,
                edition_year=article.edition_year,
                extraction_method="explicit",
                char_start=0,
                char_end=markers[0][0],
                start_page=article.start_page,
                end_page=get_page_for_char(article.char_start + markers[0][0], page_numbers),
            ))

    # Create sections from markers
    for i, (start, end, title, level) in enumerate(markers):
        section_end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        section_text = text[end:section_end].strip()

        sections.append(Section(
            section_id=f"{article.article_id}_s{len(sections):02d}",
            title=title,
            text=section_text,
            level=level,
            index=len(sections),
            article_id=article.article_id,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            extraction_method="explicit",
            char_start=start,
            char_end=section_end,
            start_page=get_page_for_char(article.char_start + start, page_numbers),
            end_page=get_page_for_char(article.char_start + section_end, page_numbers),
        ))

    return sections


# =============================================================================
# Chunking
# =============================================================================

def chunk_text(text: str, max_size: int = 4000, min_size: int = 300, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """Split text into chunks at paragraph boundaries."""
    if len(text) <= max_size:
        return [(0, len(text), text)]

    chunks = []
    paragraphs = re.split(r'\n\n+', text)

    current = ""
    current_start = 0
    pos = 0

    for para in paragraphs:
        para_sep = para + "\n\n"

        if len(current) + len(para_sep) > max_size:
            if len(current) >= min_size:
                chunks.append((current_start, pos, current.strip()))
                overlap_text = current[-overlap:] if len(current) > overlap else ""
                current = overlap_text + para_sep
                current_start = pos - len(overlap_text)
            else:
                current += para_sep
        else:
            current += para_sep

        pos += len(para_sep)

    if current.strip():
        chunks.append((current_start, pos, current.strip()))

    return chunks


def create_chunks(article: Article, sections: List[Section], page_numbers: list) -> List[Chunk]:
    """Create chunks from sections."""
    all_chunks = []
    chunk_idx = 0

    for section in sections:
        text_chunks = chunk_text(section.text)

        for rel_start, rel_end, chunk_content in text_chunks:
            abs_start = article.char_start + section.char_start + rel_start
            abs_end = article.char_start + section.char_start + rel_end

            all_chunks.append(Chunk(
                chunk_id=f"{section.section_id}_c{chunk_idx:03d}",
                text=chunk_content,
                index=chunk_idx,
                section_id=section.section_id,
                section_title=section.title,
                section_index=section.index,
                article_id=article.article_id,
                parent_headword=article.headword,
                edition_year=article.edition_year,
                volume_id=article.volume_id,
                char_start=section.char_start + rel_start,
                char_end=section.char_start + rel_end,
                start_page=get_page_for_char(abs_start, page_numbers),
                end_page=get_page_for_char(abs_end, page_numbers),
            ))
            chunk_idx += 1

    return all_chunks


# =============================================================================
# Main Processing
# =============================================================================

def process_edition(edition_year: int, sources: list, output_dir: Path) -> dict:
    """Process all volumes for an edition."""
    config = EDITIONS.get(edition_year, {"name": f"Edition {edition_year}", "number": 0})

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {config['name']} ({edition_year}) - {len(sources)} volumes")
    logger.info(f"{'='*60}")

    all_volumes = []
    all_articles = []
    all_sections = []
    all_chunks = []

    for source in sorted(sources, key=lambda x: x.get("volume_num") or 0):
        text, page_nums = load_ocr_content(source)
        if not text:
            logger.warning(f"  No text for: {source.get('title', 'unknown')[:50]}")
            continue

        volume = VolumeInfo(
            volume_id=source.get("volume_id", ""),
            edition_year=edition_year,
            edition_name=config["name"],
            edition_number=config["number"],
            volume_num=source.get("volume_num") or 0,
            letter_range=source.get("letter_range", ""),
            page_count=source.get("page_count", 0),
            source_collection="britannica_nls",
            source_file=source.get("title", ""),
            ocr_file=source.get("jsonl_path") or source.get("md_path", ""),
        )

        logger.info(f"  Volume {volume.volume_num}: {volume.letter_range}")

        articles = extract_articles(text, volume, page_nums)
        logger.info(f"    {len(articles)} articles")

        treatises = [a for a in articles if a.article_type == "treatise"]
        for article in treatises:
            sections = extract_sections(article, page_nums)
            chunks = create_chunks(article, sections, page_nums)
            all_sections.extend(sections)
            all_chunks.extend(chunks)

        all_volumes.append(volume)
        all_articles.extend(articles)

    # Deduplicate articles within edition (OCR sometimes duplicates content)
    seen_articles = {}
    unique_articles = []
    for a in all_articles:
        # Create key from headword + volume + first 100 chars of text
        key = (a.headword.upper(), a.volume_num, a.text[:100] if a.text else "")
        if key not in seen_articles:
            seen_articles[key] = a
            unique_articles.append(a)

    if len(unique_articles) < len(all_articles):
        logger.info(f"  Removed {len(all_articles) - len(unique_articles)} duplicate articles (OCR artifacts)")
    all_articles = unique_articles

    # Save outputs
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / f"volumes_{edition_year}.jsonl", 'w') as f:
        for v in all_volumes:
            f.write(json.dumps(asdict(v), ensure_ascii=False) + '\n')

    with open(output_dir / f"articles_{edition_year}.jsonl", 'w') as f:
        for a in all_articles:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + '\n')

    with open(output_dir / f"sections_{edition_year}.jsonl", 'w') as f:
        for s in all_sections:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + '\n')

    with open(output_dir / f"chunks_{edition_year}.jsonl", 'w') as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + '\n')

    stats = {
        "edition_year": edition_year,
        "edition_name": config["name"],
        "volumes": len(all_volumes),
        "articles": len(all_articles),
        "treatises": sum(1 for a in all_articles if a.article_type == "treatise"),
        "sections": len(all_sections),
        "chunks": len(all_chunks),
    }

    logger.info(f"\n  Total: {stats['articles']} articles, {stats['treatises']} treatises, "
                f"{stats['sections']} sections, {stats['chunks']} chunks")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract Britannica corpus")
    parser.add_argument("--ocr-dir", default="ocr_results", help="OCR results directory")
    parser.add_argument("--output-dir", default="output_v2", help="Output directory")
    parser.add_argument("--editions", help="Comma-separated edition years")
    parser.add_argument("--all", action="store_true", help="Process all editions")
    parser.add_argument("--list", action="store_true", help="List available editions")

    args = parser.parse_args()

    ocr_dir = Path(__file__).parent / args.ocr_dir
    output_dir = Path(__file__).parent / args.output_dir

    logger.info("Discovering OCR sources...")
    sources = discover_sources(ocr_dir)

    if args.list:
        print("\nAvailable editions:")
        for year in sorted(sources.keys()):
            config = EDITIONS.get(year, {"name": f"Edition {year}"})
            print(f"  {year} ({config['name']}): {len(sources[year])} volumes")
        return

    if args.all:
        editions = sorted(sources.keys())
    elif args.editions:
        editions = [int(y.strip()) for y in args.editions.split(",")]
    else:
        parser.print_help()
        return

    all_stats = []
    for year in editions:
        if year not in sources:
            logger.warning(f"No sources found for {year}")
            continue
        stats = process_edition(year, sources[year], output_dir)
        all_stats.append(stats)

    # Save summary
    with open(output_dir / "extraction_stats.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "editions": all_stats}, f, indent=2)

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    for s in all_stats:
        print(f"  {s['edition_name']}: {s['articles']} articles, {s['chunks']} chunks")


if __name__ == "__main__":
    main()
