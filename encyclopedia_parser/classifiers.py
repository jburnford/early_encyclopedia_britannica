"""
Article type classification and metadata extraction.

Classifies articles as: dictionary, treatise, biographical, geographical, cross_reference

Now with edition-aware classification:
- 1771 (1st edition): No biographical entries allowed
- 1778+ editions: All article types allowed
- Edition-specific treatise lists and thresholds
"""

import re
from typing import Optional
import logging

from .models import Article, ArticleType, EditionConfig, get_edition_config
from .patterns import (
    FULL_CROSS_REF,
    extract_coordinates,
    extract_dates,
    extract_cross_references,
)

logger = logging.getLogger(__name__)


def _get_edition_module(year: int):
    """
    Dynamically load edition-specific module if available.

    Args:
        year: Edition year

    Returns:
        Edition module or None if not available
    """
    try:
        if year == 1771:
            from .editions import edition_1771 as edition_module
            return edition_module
        elif year == 1815:
            from .editions import edition_1815 as edition_module
            return edition_module
        elif year == 1842:
            from .editions import edition_1842 as edition_module
            return edition_module
    except ImportError:
        pass
    return None


class ArticleClassifier:
    """
    Classifies encyclopedia articles by type and extracts relevant metadata.

    Now edition-aware: respects edition-specific rules for article types.
    For example, 1771 (1st edition) does not allow biographical entries.
    """

    # Default thresholds for classification
    TREATISE_LENGTH_THRESHOLD = 5000  # Characters
    SHORT_CROSS_REF_THRESHOLD = 100   # Max length for "See X" only entries

    def __init__(self, edition_config: Optional[EditionConfig] = None, edition_year: Optional[int] = None):
        """
        Initialize the classifier.

        Args:
            edition_config: Optional edition configuration for treatise list
            edition_year: Optional edition year for loading edition-specific rules
        """
        # Determine edition year
        if edition_config:
            self.edition_year = edition_config.year
            self.edition_config = edition_config
        elif edition_year:
            self.edition_year = edition_year
            self.edition_config = get_edition_config(edition_year)
        else:
            self.edition_year = None
            self.edition_config = None

        # Load edition-specific module if available
        self.edition_module = _get_edition_module(self.edition_year) if self.edition_year else None

        # Get major treatises from config or module
        if self.edition_module and hasattr(self.edition_module, 'MAJOR_TREATISES'):
            self.major_treatises = self.edition_module.MAJOR_TREATISES
        elif self.edition_config:
            self.major_treatises = set(self.edition_config.major_treatises)
        else:
            self.major_treatises = set()

        # Get treatise threshold from edition module or use default
        if self.edition_module and hasattr(self.edition_module, 'TREATISE_CHAR_THRESHOLD'):
            self.treatise_threshold = self.edition_module.TREATISE_CHAR_THRESHOLD
        else:
            self.treatise_threshold = self.TREATISE_LENGTH_THRESHOLD

    def classify(self, article: Article) -> Article:
        """
        Classify an article and extract relevant metadata.

        Modifies the article in place and returns it.

        Args:
            article: The article to classify

        Returns:
            The article with article_type and metadata populated
        """
        text = article.text
        headword = article.headword

        # Extract metadata first (needed for classification)
        article.cross_references = [ref for ref, _ in extract_cross_references(text)]
        article.coordinates = extract_coordinates(text)
        article.person_dates = extract_dates(text)

        # Classify in priority order
        if self._is_cross_reference(article):
            article.article_type = ArticleType.CROSS_REFERENCE
            article.is_cross_reference = True

        elif self._is_geographical(article):
            article.article_type = ArticleType.GEOGRAPHICAL

        elif self._is_biographical(article):
            article.article_type = ArticleType.BIOGRAPHICAL

        elif self._is_treatise(article):
            article.article_type = ArticleType.TREATISE

        else:
            article.article_type = ArticleType.DICTIONARY

        return article

    def _is_cross_reference(self, article: Article) -> bool:
        """
        Check if article is a cross-reference entry.

        Cross-references are short entries that just redirect to another article.
        Examples:
            "See ASTRONOMY"
            "**ABRIDGEMENT.** See Abstract."
        """
        text = article.text.strip()

        # Check if entire text is a "See X" reference
        if FULL_CROSS_REF.match(text):
            return True

        # Short text with "See" is likely a cross-reference
        if len(text) < self.SHORT_CROSS_REF_THRESHOLD:
            if re.search(r'\bSee\s+[A-Z]', text, re.IGNORECASE):
                return True

        return False

    def _is_geographical(self, article: Article) -> bool:
        """
        Check if article is a geographical entry.

        Geographical entries describe places and typically contain coordinates.
        Examples:
            "MONTREAL, a city in Canada... E. Long. 73.35. N. Lat. 45.30."
        """
        # Has coordinates
        if article.coordinates is not None:
            return True

        # Contains coordinate patterns but extraction failed
        text = article.text
        if re.search(r'[EW]\.\s*Long\.', text, re.IGNORECASE):
            return True
        if re.search(r'[NS]\.\s*Lat\.', text, re.IGNORECASE):
            return True

        # Common geographical indicators
        geo_indicators = [
            r'\b(?:city|town|village|river|mountain|island|county|district)\s+(?:of|in)\b',
            r'\b(?:lies|situated|bounded|borders)\b',
            r'\bmiles?\s+(?:north|south|east|west|from)\b',
            r'\bpopulation\s+(?:of\s+)?\d',
        ]

        text_lower = text.lower()
        indicator_count = sum(
            1 for pattern in geo_indicators
            if re.search(pattern, text_lower)
        )

        # Multiple indicators suggest geographical entry
        return indicator_count >= 2

    def _is_biographical(self, article: Article) -> bool:
        """
        Check if article is a biographical entry.

        Biographical entries describe people with birth/death dates.
        Examples:
            "NEWTON (Sir Isaac), born 1642, died 1727..."

        Note: Edition-aware - 1771 (1st edition) does not have biographical entries.
        """
        # Check if edition allows biographical entries
        if self.edition_config and not self.edition_config.has_biography:
            return False

        # Use edition-specific detection if available
        if self.edition_module and hasattr(self.edition_module, 'detect_biographical'):
            return self.edition_module.detect_biographical(article.text)

        # Has extracted dates
        if article.person_dates is not None:
            return True

        text = article.text

        # Common biographical patterns
        bio_patterns = [
            r'\bborn\s+(?:in\s+)?(?:about\s+)?\d{4}',
            r'\bdied\s+(?:in\s+)?(?:about\s+)?\d{4}',
            r'\(\d{4}\s*[-–—]\s*\d{4}\)',  # (1642-1727)
            r'\bflourished\s+(?:about\s+)?\d{4}',
            r'\b(?:he|she)\s+was\s+(?:a\s+)?(?:celebrated|famous|eminent|distinguished)',
            r'\b(?:son|daughter)\s+of\b',
        ]

        for pattern in bio_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _is_treatise(self, article: Article) -> bool:
        """
        Check if article is a treatise (long scholarly article).

        Treatises are major articles covering entire subjects like ASTRONOMY,
        CHEMISTRY, MEDICINE, etc.

        Uses edition-specific thresholds and treatise lists.
        """
        headword = article.headword.upper()

        # Use edition-specific detection if available
        if self.edition_module and hasattr(self.edition_module, 'should_be_treatise'):
            return self.edition_module.should_be_treatise(headword, len(article.text))

        # Known major treatise
        if headword in self.major_treatises:
            return True

        # Long articles are likely treatises (use edition-specific threshold)
        if len(article.text) > self.treatise_threshold:
            return True

        # Contains section structure
        section_patterns = [
            r'\bPART\s+[IVXLCDM]+\b',
            r'\bSECT(?:ION)?\.?\s+[IVXLCDM\d]+\b',
            r'\bCHAPTER\s+[IVXLCDM\d]+\b',
        ]

        for pattern in section_patterns:
            if re.search(pattern, article.text[:2000]):  # Check beginning
                return True

        return False


def classify_article(
    article: Article,
    edition_config: Optional[EditionConfig] = None,
    edition_year: Optional[int] = None,
) -> Article:
    """
    Convenience function to classify a single article.

    Args:
        article: The article to classify
        edition_config: Optional edition configuration
        edition_year: Optional edition year (used if edition_config not provided)

    Returns:
        The classified article
    """
    # Use article's edition year if not specified
    if edition_year is None and edition_config is None:
        edition_year = article.edition_year

    classifier = ArticleClassifier(edition_config, edition_year)
    return classifier.classify(article)


def classify_articles(
    articles: list[Article],
    edition_config: Optional[EditionConfig] = None,
    edition_year: Optional[int] = None,
) -> list[Article]:
    """
    Classify a list of articles.

    Args:
        articles: List of articles to classify
        edition_config: Optional edition configuration
        edition_year: Optional edition year (used if edition_config not provided)

    Returns:
        List of classified articles
    """
    # Use first article's edition year if not specified
    if edition_year is None and edition_config is None and articles:
        edition_year = articles[0].edition_year

    classifier = ArticleClassifier(edition_config, edition_year)
    return [classifier.classify(article) for article in articles]


def get_classification_stats(articles: list[Article]) -> dict:
    """
    Get classification statistics for a list of articles.

    Args:
        articles: List of classified articles

    Returns:
        Dictionary with counts by article type
    """
    stats = {
        "total": len(articles),
        "by_type": {},
        "with_cross_references": 0,
        "with_coordinates": 0,
        "with_dates": 0,
        "average_length": 0,
        "treatise_count": 0,
    }

    type_counts = {}
    total_length = 0

    for article in articles:
        # Count by type
        article_type = article.article_type
        if isinstance(article_type, ArticleType):
            article_type = article_type.value
        type_counts[article_type] = type_counts.get(article_type, 0) + 1

        # Count metadata
        if article.cross_references:
            stats["with_cross_references"] += 1
        if article.coordinates:
            stats["with_coordinates"] += 1
        if article.person_dates:
            stats["with_dates"] += 1

        total_length += len(article.text)

    stats["by_type"] = type_counts
    stats["average_length"] = total_length // len(articles) if articles else 0
    stats["treatise_count"] = type_counts.get("treatise", 0)

    return stats
