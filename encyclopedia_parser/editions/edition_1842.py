"""
Edition-specific configuration for Encyclopaedia Britannica 7th Edition (1842).

Key characteristics:
- 22 volumes (21 content + 1 General Index)
- Volume 22 is the General Index - the "Rosetta Stone" for taxonomy
- The index provides the canonical term hierarchy
- Clean typography (no long s)

Historical context:
- Published 1830-1842 in Edinburgh
- The General Index (Volume 22) is crucial for understanding
  the encyclopedia's internal structure and cross-references
- Often considered the most scholarly of the early editions
"""

import re
from typing import Optional
from ..models import ArticleType


# === EDITION METADATA ===
EDITION_YEAR = 1842
EDITION_NAME = "Britannica 7th"
EDITION_NUMBER = 7
VOLUME_COUNT = 22  # 21 content volumes + 1 index volume

# Index volume is Volume 22 (sometimes labeled as Volume 0 in OCR)
INDEX_VOLUME = 22

# === CONTENT RULES ===

# 7th edition has all article types
HAS_BIOGRAPHY = True
HAS_GEOGRAPHY = True

# Valid article types
ALLOWED_ARTICLE_TYPES = list(ArticleType)

# === MAJOR TREATISES ===
# The index provides the definitive list - this is a starter set
# TODO: Populate from parsed index
MAJOR_TREATISES = {
    # Will be populated from the General Index
    "AGRICULTURE", "ANATOMY", "ARCHITECTURE", "ASTRONOMY",
    "BIOGRAPHY", "BOTANY", "CHEMISTRY", "COMMERCE",
    "ELECTRICITY", "GEOGRAPHY", "GEOLOGY", "GEOMETRY",
    "GRAMMAR", "HERALDRY", "HISTORY", "LAW",
    "MECHANICS", "MEDICINE", "MINERALOGY", "MUSIC",
    "NAVIGATION", "OPTICS", "PAINTING", "PHILOSOPHY",
    "PHYSICS", "PHYSIOLOGY", "POETRY", "RELIGION",
    "SCULPTURE", "SURGERY", "THEOLOGY",
}

# === INDEX PARSING ===

# Patterns for parsing the General Index (Volume 22)
INDEX_PATTERNS = {
    # Main entry: "TERM, subtopic. Vol X:page"
    "main_entry": re.compile(
        r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*),\s*'
        r'([^.]+)\.\s*'
        r'(?:Vol\.?\s*)?(\d+):(\d+)',
        re.MULTILINE
    ),

    # Cross-reference: "TERM. See MAIN_HEADING"
    "cross_ref": re.compile(
        r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\.\s*'
        r'See\s+([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)',
        re.MULTILINE | re.IGNORECASE
    ),

    # Subentry (indented): "  subtopic, Vol X:page"
    "subentry": re.compile(
        r'^\s{2,}([^,]+),\s*'
        r'(?:Vol\.?\s*)?(\d+):(\d+)',
        re.MULTILINE
    ),
}

# === OCR ARTIFACT PATTERNS ===

OCR_ARTIFACTS = {
    "PLATE", "PLATES", "FIG", "FIGURE", "FIGURES",
    "CONTINUED", "CONCLUSION",
    "ERRATA", "ERRATUM", "ADDENDA", "CORRIGENDA",
    "GENERAL INDEX",  # The index title itself
    "INDEX",
}

OCR_ARTIFACT_PATTERNS = [
    re.compile(r'^PLATE\s+[IVXLCDM\d]+$'),
    re.compile(r'^FIG\.\s*\d+$'),
    re.compile(r'^PAGE\s+\d+$'),
    re.compile(r'^VOL\.?\s*[IVXLCDM\d]+$'),
]

# === VALIDATION RULES ===

def is_valid_headword(headword: str, volume: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a headword for the 1842 edition.

    Args:
        headword: The candidate headword
        volume: Optional volume number for range checking

    Returns:
        (is_valid, error_reason) tuple
    """
    headword = headword.upper().strip()

    # Check against known OCR artifacts
    if headword in OCR_ARTIFACTS:
        return False, f"Known OCR artifact: {headword}"

    # Check against artifact patterns
    for pattern in OCR_ARTIFACT_PATTERNS:
        if pattern.match(headword):
            return False, f"Matches OCR artifact pattern"

    # Special handling for index volume
    if volume == INDEX_VOLUME:
        # Index volume has different structure - skip normal validation
        return True, None

    return True, None


def is_valid_article_type(article_type: ArticleType) -> bool:
    """Check if an article type is valid for the 1842 edition."""
    return article_type in ALLOWED_ARTICLE_TYPES


def should_be_treatise(headword: str, text_length: int) -> bool:
    """
    Determine if an article should be classified as a treatise.

    Args:
        headword: The article headword
        text_length: Length of article text in characters

    Returns:
        True if this should be a treatise
    """
    headword = headword.upper().strip()

    # Known major treatises
    if headword in MAJOR_TREATISES:
        return True

    # Length-based detection
    if text_length >= 50000:  # ~38 pages
        return True

    return False


def is_index_volume(volume: int) -> bool:
    """Check if a volume is the General Index."""
    return volume == INDEX_VOLUME


# === INDEX PARSING FUNCTIONS ===

def parse_index_entry(line: str) -> Optional[dict]:
    """
    Parse a single index entry line.

    Args:
        line: A line from the index

    Returns:
        Dictionary with parsed entry or None if not a valid entry

    Entry types:
        - main: "TERM, subtopic. Vol X:page"
        - cross_ref: "TERM. See MAIN_HEADING"
        - subentry: "  subtopic, Vol X:page"
    """
    line = line.strip()

    # Try main entry pattern
    match = INDEX_PATTERNS["main_entry"].match(line)
    if match:
        return {
            "type": "main",
            "term": match.group(1).upper(),
            "subtopic": match.group(2).strip(),
            "volume": int(match.group(3)),
            "page": int(match.group(4)),
        }

    # Try cross-reference pattern
    match = INDEX_PATTERNS["cross_ref"].match(line)
    if match:
        return {
            "type": "cross_ref",
            "term": match.group(1).upper(),
            "see_also": match.group(2).upper(),
        }

    # Try subentry pattern
    match = INDEX_PATTERNS["subentry"].match(line)
    if match:
        return {
            "type": "subentry",
            "subtopic": match.group(1).strip(),
            "volume": int(match.group(2)),
            "page": int(match.group(3)),
        }

    return None
