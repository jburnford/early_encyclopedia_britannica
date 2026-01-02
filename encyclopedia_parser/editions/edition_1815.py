"""
Edition-specific configuration for Encyclopaedia Britannica 5th Edition (1815).

Key characteristics:
- 20 volumes (largest article count: ~18,000 articles)
- Full range of article types: dictionary, treatise, biographical, geographical
- Most comprehensive treatise list
- Detailed volume letter ranges available
- Supplement published alongside

Historical context:
- Sometimes called the "Napier Edition" after editor Macvey Napier
- Published during the Scottish Enlightenment period
- Contains many articles by prominent scholars of the era
"""

import re
from typing import Optional
from ..models import ArticleType


# === EDITION METADATA ===
EDITION_YEAR = 1815
EDITION_NAME = "Britannica 5th"
EDITION_NUMBER = 5
VOLUME_COUNT = 20

# Volume letter ranges (detailed mapping)
VOLUME_RANGES = {
    1: ("A", "ANA"),
    2: ("ANA", "BAC"),
    3: ("BAC", "BUR"),
    4: ("BUR", "CHI"),
    5: ("CHI", "CRI"),
    6: ("CRI", "ECO"),
    7: ("ECO", "FEU"),
    8: ("FEU", "GOR"),
    9: ("GOR", "HYD"),
    10: ("HYD", "LEP"),
    11: ("LEP", "MED"),
    12: ("MED", "MUS"),
    13: ("MUS", "PEN"),
    14: ("PEN", "PRI"),
    15: ("PRI", "SAL"),
    16: ("SAL", "SHI"),
    17: ("SHI", "STE"),
    18: ("STE", "TUR"),
    19: ("TUR", "WAT"),
    20: ("WAT", "ZYM"),
}

# === CONTENT RULES ===

# 5th edition has all article types
HAS_BIOGRAPHY = True
HAS_GEOGRAPHY = True

# Valid article types for 5th edition
ALLOWED_ARTICLE_TYPES = list(ArticleType)

# === MAJOR TREATISES ===
# Comprehensive list for the 5th edition
MAJOR_TREATISES = {
    # Core Sciences
    "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
    "ASTRONOMY", "BOTANY", "CHEMISTRY", "CHRONOLOGY", "CRYSTALLIZATION",
    "ELECTRICITY", "FLUXIONS", "GEOGRAPHY", "GEOLOGY", "GEOMETRY",
    "HYDRAULICS", "HYDROSTATICS", "MAGNETISM", "MECHANICS", "MEDICINE",
    "METAPHYSICS", "MINERALOGY", "MUSIC", "NAVIGATION", "OPTICS",
    "PHARMACY", "PHILOSOPHY", "PHYSIOLOGY", "PNEUMATICS", "SURGERY",
    "THEOLOGY", "TRIGONOMETRY",

    # Applied Sciences & Trades
    "BREWING", "COMMERCE", "DISTILLING", "DYEING", "FARRIERY",
    "FORTIFICATION", "HERALDRY", "MIDWIFERY",

    # Arts & Humanities
    "CRITICISM", "ETHICS", "GRAMMAR", "HISTORY", "LAW", "LOGIC",
    "PAINTING", "PERSPECTIVE", "POETRY", "RHETORIC", "SCULPTURE",

    # Military
    "WAR",

    # Major geographical/historical entries
    "SCOTLAND", "ENGLAND", "IRELAND", "FRANCE", "AMERICA",
    "AFRICA", "ASIA", "EUROPE",
}

# Treatise length threshold (larger for 5th edition)
TREATISE_CHAR_THRESHOLD = 50000  # ~38 pages at ~1300 chars/page

# === OCR ARTIFACT PATTERNS ===
# Patterns specific to 1815 OCR output

OCR_ARTIFACTS = {
    # Common false positives
    "PLATE", "PLATES", "FIG", "FIGURE", "FIGURES",
    "CONTINUED", "CONCLUSION",
    "ERRATA", "ERRATUM", "ADDENDA", "CORRIGENDA",
    # Running headers
    "ENCYCLOPAEDIA", "BRITANNICA",
    # Page markers
    "INDEX",
}

# Regex patterns for 1815-specific artifacts
OCR_ARTIFACT_PATTERNS = [
    re.compile(r'^PLATE\s+[IVXLCDM\d]+$'),
    re.compile(r'^FIG\.\s*\d+$'),
    re.compile(r'^[IVXLCDM]+\.$'),
    re.compile(r'^PAGE\s+\d+$'),
]

# === VALIDATION RULES ===

def is_valid_headword(headword: str, volume: Optional[int] = None) -> tuple[bool, Optional[str]]:
    """
    Validate a headword for the 1815 edition.

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

    # Check volume letter range (more complex for 1815 with detailed ranges)
    if volume and volume in VOLUME_RANGES:
        start_range, end_range = VOLUME_RANGES[volume]
        first_letter = headword[0] if headword else ""

        # Simple check: first letter should be in the right range
        if len(start_range) == 1 and len(end_range) == 1:
            if not (start_range <= first_letter <= end_range):
                return False, f"Headword {headword} outside volume {volume} range"

    return True, None


def is_valid_article_type(article_type: ArticleType) -> bool:
    """Check if an article type is valid for the 1815 edition."""
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
    if text_length >= TREATISE_CHAR_THRESHOLD:
        return True

    return False


# === KNOWN CONTRIBUTORS ===
# Notable authors who contributed signed articles to the 5th edition
KNOWN_CONTRIBUTORS = {
    "Thomas Young": ["EGYPT", "CHROMATICS", "TIDES"],
    "James Mill": ["GOVERNMENT", "JURISPRUDENCE"],
    "David Ricardo": ["FUNDING SYSTEM"],
    "Walter Scott": ["CHIVALRY", "ROMANCE"],
    "Dugald Stewart": ["PHILOSOPHY"],
}
