"""
Edition-specific configuration for Encyclopaedia Britannica 1st Edition (1771).

Key characteristics:
- 3 volumes only: A-B, C-L, M-Z
- Arts & Sciences focus ONLY - no Biography or History of persons
- Treatises are the core content with shorter dictionary entries
- Smaller treatise length threshold (>15 pages vs >50 for later editions)
- Some unique OCR artifacts from the original typography

Historical context:
- Published in Edinburgh by Andrew Bell and Colin Macfarquhar
- First edition of what would become the iconic encyclopedia
- Explicitly excluded biography to focus on scientific knowledge
"""

import re
from typing import Optional
from ..models import ArticleType


# === EDITION METADATA ===
EDITION_YEAR = 1771
EDITION_NAME = "Britannica 1st"
EDITION_NUMBER = 1
VOLUME_COUNT = 3

# Volume letter ranges
VOLUME_RANGES = {
    1: ("A", "B"),
    2: ("C", "L"),
    3: ("M", "Z"),
}

# === CONTENT RULES ===

# This edition does NOT have biographical entries
HAS_BIOGRAPHY = False
HAS_GEOGRAPHY = True

# Valid article types for 1st edition
ALLOWED_ARTICLE_TYPES = [
    ArticleType.DICTIONARY,
    ArticleType.TREATISE,
    ArticleType.GEOGRAPHICAL,
    ArticleType.CROSS_REFERENCE,
]

# === MAJOR TREATISES ===
# These are the core content of the 1st edition
MAJOR_TREATISES = {
    "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
    "ASTRONOMY", "BOTANY", "BREWING", "CHEMISTRY", "CHRONOLOGY",
    "COMMERCE", "COSMOGRAPHY", "CRITICISM", "DISTILLING", "DYEING",
    "ELECTRICITY", "ETHICS", "FARRIERY", "FLUXIONS", "FORTIFICATION",
    "GEOGRAPHY", "GEOMETRY", "GRAMMAR", "HERALDRY", "HISTORY",
    "HYDRAULICS", "HYDROSTATICS", "LAW", "LOGIC", "MAGNETISM",
    "MECHANICS", "MEDICINE", "METAPHYSICS", "MIDWIFERY", "MUSIC",
    "NAVIGATION", "OPTICS", "PAINTING", "PERSPECTIVE", "PHARMACY",
    "PHILOSOPHY", "PHYSIOLOGY", "PNEUMATICS", "POETRY", "RHETORIC",
    "SCULPTURE", "SURGERY", "THEOLOGY", "TRIGONOMETRY", "WAR",
}

# Treatise length threshold for 1st edition (shorter than later editions)
# 1st edition treatises are typically >15 pages vs >50 for later editions
TREATISE_CHAR_THRESHOLD = 20000  # ~15 pages at ~1300 chars/page

# === OCR ARTIFACT PATTERNS ===
# Patterns specific to 1771 OCR output

# False positive headwords that appear in 1st edition OCR
OCR_ARTIFACTS = {
    # Plate references that get parsed as headwords
    "PLATE", "PLATES", "FIG", "FIGURE", "FIGURES",
    # Running headers
    "CONTINUED", "CONCLUSION",
    # Errata markers
    "ERRATA", "ERRATUM",
}

# Regex patterns for 1771-specific artifacts
OCR_ARTIFACT_PATTERNS = [
    re.compile(r'^PLATE\s+[IVXLCDM]+$'),  # PLATE I, PLATE II, etc.
    re.compile(r'^FIG\.\s*\d+$'),  # FIG. 1, FIG. 2
    re.compile(r'^[IVXLCDM]+\.$'),  # Roman numerals with period
]

# === VALIDATION RULES ===

def is_valid_headword(headword: str, volume: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a headword for the 1771 edition.

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

    # Check volume letter range
    if volume and volume in VOLUME_RANGES:
        start_letter, end_letter = VOLUME_RANGES[volume]
        first_letter = headword[0] if headword else ""
        if not (start_letter <= first_letter <= end_letter):
            return False, f"Headword {headword} outside volume {volume} range ({start_letter}-{end_letter})"

    return True, None


def is_valid_article_type(article_type: ArticleType) -> bool:
    """Check if an article type is valid for the 1771 edition."""
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

    # Length-based detection (lower threshold for 1st edition)
    if text_length >= TREATISE_CHAR_THRESHOLD:
        return True

    return False


# === BIOGRAPHICAL DETECTION OVERRIDE ===

def detect_biographical(text: str) -> bool:
    """
    Override biographical detection for 1st edition.

    The 1st edition explicitly excluded biography, so we should
    NOT classify any articles as biographical.

    Returns:
        Always False for 1771 edition
    """
    return False
