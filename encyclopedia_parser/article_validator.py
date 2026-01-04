"""
Article validator for Encyclopedia Britannica corpus cleanup.

Identifies and flags problematic articles based on:
- Sentence fragment headwords
- Structural markers (END_OF_VOLUME, PLATE explanations)
- Articles outside alphabetical range
- Unusually short/long articles
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class IssueType(str, Enum):
    """Types of article quality issues."""
    SENTENCE_FRAGMENT = "sentence_fragment"
    STRUCTURAL_MARKER = "structural_marker"
    OUT_OF_RANGE = "out_of_range"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    OCR_ERROR = "ocr_error"
    DUPLICATE = "duplicate"


class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    HIGH = "high"      # Should be removed
    MEDIUM = "medium"  # Should be flagged for review
    LOW = "low"        # Minor issue, keep but note


@dataclass
class ValidationIssue:
    """A detected issue with an article."""
    issue_type: IssueType
    severity: IssueSeverity
    reason: str
    confidence: float = 1.0  # How confident we are this is an issue

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type.value,
            "severity": self.severity.value,
            "reason": self.reason,
            "confidence": self.confidence
        }


@dataclass
class ValidationResult:
    """Result of validating an article."""
    is_valid: bool
    action: str  # "keep", "remove", "flag"
    issues: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "action": self.action,
            "issues": [i.to_dict() for i in self.issues]
        }


# Words that commonly end sentence fragments
FRAGMENT_ENDINGS = {
    'BY', 'TO', 'OF', 'THE', 'A', 'AN', 'IN', 'WITH', 'FOR', 'FROM',
    'AND', 'OR', 'IS', 'ARE', 'WAS', 'WERE', 'NO', 'THAT', 'THIS',
    'THESE', 'WHICH', 'IT', 'MAY', 'BE', 'AS', 'AT', 'ON', 'UPON',
    'INTO', 'BUT', 'SO', 'IF', 'WHEN', 'THEN', 'THAN', 'NOT', 'BEING',
    'HAVING', 'HIS', 'HER', 'THEIR', 'ITS', 'WHO', 'WHOM', 'WHOSE'
}

# Patterns that indicate structural markers (not real articles)
# Note: Some words like ADVERTISEMENT, PREFACE can be real articles about those topics
# These are only structural if they have short/no content
STRUCTURAL_PATTERNS = [
    r'^END_OF_THE_.*_VOLUME$',
    r'^END_OF_VOLUME',
    r'^EXPLANATION_OF_PLATE',
    r'^EXPLANATION_OF_THE_PLATE',
    r'^DIRECTIONS_FOR_PLACING',
    r'^CLASS_[IVX]+$',
    r'^ARTICLE_[IVX]+$',
    r'^PART_[IVX]+$',
    r'^SECT(?:ION)?_[IVX]+$',
    r'^PLATE_[IVXLCDM]+$',
    r'^PLATE_\d+$',
    r'^TABLE_[IVX]+$',
    r'^PROPOSITION_[IVX]+$',
    r'^CHAPTER_[IVX]+$',
    r'^BOOK_[IVX]+$',
]

# Patterns that are only structural markers if article is SHORT (< 500 chars)
# These can be legitimate articles about the topic if they have substantial content
CONDITIONAL_STRUCTURAL_PATTERNS = [
    r'^ADDENDUM$',
    r'^ERRATA$',
    r'^INDEX$',
    r'^CONTENTS$',
    r'^PREFACE$',
    r'^ADVERTISEMENT$',
]

# Compile patterns for efficiency
STRUCTURAL_REGEX = [re.compile(p, re.IGNORECASE) for p in STRUCTURAL_PATTERNS]
CONDITIONAL_STRUCTURAL_REGEX = [re.compile(p, re.IGNORECASE) for p in CONDITIONAL_STRUCTURAL_PATTERNS]

# Minimum text length for conditional structural patterns to be considered real articles
CONDITIONAL_MIN_LENGTH = 500


class ArticleValidator:
    """
    Validates encyclopedia articles and identifies issues.

    Usage:
        validator = ArticleValidator()
        result = validator.validate(article_dict)
        if result.action == "remove":
            # Skip this article
        elif result.action == "flag":
            article_dict["needs_review"] = True
            article_dict["issues"] = result.to_dict()["issues"]
    """

    def __init__(
        self,
        min_text_length: int = 20,
        max_headword_length: int = 60,
        max_text_length: int = 500_000,
        removal_confidence_threshold: float = 0.95,
        flag_confidence_threshold: float = 0.70
    ):
        """
        Initialize the validator.

        Args:
            min_text_length: Articles shorter than this are flagged
            max_headword_length: Headwords longer than this are likely fragments
            max_text_length: Articles longer than this are flagged for review
            removal_confidence_threshold: Issues above this are removed
            flag_confidence_threshold: Issues above this are flagged
        """
        self.min_text_length = min_text_length
        self.max_headword_length = max_headword_length
        self.max_text_length = max_text_length
        self.removal_threshold = removal_confidence_threshold
        self.flag_threshold = flag_confidence_threshold

        # Volume ranges loaded on demand
        self._volume_ranges: Optional[dict] = None

    def validate(self, article: dict) -> ValidationResult:
        """
        Validate an article and return the result.

        Args:
            article: Dictionary with article data (headword, text, etc.)

        Returns:
            ValidationResult with issues and recommended action
        """
        issues = []

        headword = article.get('headword', '')
        text = article.get('text', '')
        text_length = len(text)

        # Check for sentence fragment headwords
        fragment_issue = self._check_sentence_fragment(headword)
        if fragment_issue:
            issues.append(fragment_issue)

        # Check for structural markers
        structural_issue = self._check_structural_marker(headword, text_length)
        if structural_issue:
            issues.append(structural_issue)

        # Check for OCR errors in headword
        ocr_issue = self._check_ocr_errors(headword)
        if ocr_issue:
            issues.append(ocr_issue)

        # Check text length
        length_issue = self._check_text_length(text_length, article.get('is_cross_reference', False))
        if length_issue:
            issues.append(length_issue)

        # Check alphabetical range (if volume info available)
        if 'volume_num' in article and 'edition_year' in article:
            range_issue = self._check_alphabetical_range(
                headword,
                article['volume_num'],
                article['edition_year']
            )
            if range_issue:
                issues.append(range_issue)

        # Determine action based on issues
        action = self._determine_action(issues)
        is_valid = action == "keep"

        return ValidationResult(
            is_valid=is_valid,
            action=action,
            issues=issues
        )

    def _check_sentence_fragment(self, headword: str) -> Optional[ValidationIssue]:
        """Check if headword appears to be a sentence fragment."""
        if not headword:
            return None

        # Normalize for checking
        hw_upper = headword.upper().replace('_', ' ')
        words = hw_upper.split()

        # Check 1: Ends with common preposition/conjunction
        if len(words) > 1 and words[-1] in FRAGMENT_ENDINGS:
            return ValidationIssue(
                issue_type=IssueType.SENTENCE_FRAGMENT,
                severity=IssueSeverity.HIGH,
                reason=f"Headword ends with '{words[-1]}' - likely sentence fragment",
                confidence=0.98
            )

        # Check 2: Very long headword (> max_headword_length chars)
        if len(headword) > self.max_headword_length:
            return ValidationIssue(
                issue_type=IssueType.SENTENCE_FRAGMENT,
                severity=IssueSeverity.HIGH,
                reason=f"Headword too long ({len(headword)} chars) - likely sentence fragment",
                confidence=0.95
            )

        # Check 3: Contains multiple words (> 6) without being a known pattern
        if len(words) > 6:
            # Some legitimate multi-word entries exist, but 7+ words is suspicious
            return ValidationIssue(
                issue_type=IssueType.SENTENCE_FRAGMENT,
                severity=IssueSeverity.MEDIUM,
                reason=f"Headword has {len(words)} words - possible sentence fragment",
                confidence=0.80
            )

        # Check 4: Contains newlines or multiple consecutive spaces
        if '\n' in headword or '  ' in headword:
            return ValidationIssue(
                issue_type=IssueType.SENTENCE_FRAGMENT,
                severity=IssueSeverity.HIGH,
                reason="Headword contains newlines or multiple spaces",
                confidence=0.95
            )

        return None

    def _check_structural_marker(self, headword: str, text_length: int = 0) -> Optional[ValidationIssue]:
        """Check if headword is a structural marker (not a real article)."""
        if not headword:
            return None

        hw_normalized = headword.upper().replace(' ', '_')

        # Check unconditional structural patterns (always structural)
        for pattern in STRUCTURAL_REGEX:
            if pattern.match(hw_normalized):
                return ValidationIssue(
                    issue_type=IssueType.STRUCTURAL_MARKER,
                    severity=IssueSeverity.HIGH,
                    reason=f"Matches structural pattern: {pattern.pattern}",
                    confidence=0.99
                )

        # Check conditional patterns (only structural if text is short)
        # Words like ADVERTISEMENT, PREFACE can be real articles if they have content
        for pattern in CONDITIONAL_STRUCTURAL_REGEX:
            if pattern.match(hw_normalized):
                if text_length < CONDITIONAL_MIN_LENGTH:
                    return ValidationIssue(
                        issue_type=IssueType.STRUCTURAL_MARKER,
                        severity=IssueSeverity.HIGH,
                        reason=f"Matches structural pattern '{pattern.pattern}' with short text ({text_length} chars)",
                        confidence=0.95
                    )
                # If text is long enough, it's likely a real article about this topic
                # Don't flag it

        return None

    def _check_ocr_errors(self, headword: str) -> Optional[ValidationIssue]:
        """Check for obvious OCR errors in headword."""
        if not headword:
            return None

        # Very short headwords (1-2 chars) that aren't common entries
        if len(headword) <= 2:
            # Some 2-letter entries are valid (AA, AB, etc.)
            if not headword.isalpha():
                return ValidationIssue(
                    issue_type=IssueType.OCR_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    reason=f"Very short non-alphabetic headword: '{headword}'",
                    confidence=0.85
                )

        # Contains non-printable or unusual characters
        if any(ord(c) > 127 or ord(c) < 32 for c in headword if c not in ' \t'):
            return ValidationIssue(
                issue_type=IssueType.OCR_ERROR,
                severity=IssueSeverity.MEDIUM,
                reason="Headword contains unusual characters",
                confidence=0.80
            )

        # All digits (likely page number or similar)
        if headword.replace('_', '').isdigit():
            return ValidationIssue(
                issue_type=IssueType.OCR_ERROR,
                severity=IssueSeverity.HIGH,
                reason="Headword is purely numeric",
                confidence=0.95
            )

        return None

    def _check_text_length(self, text_length: int, is_cross_ref: bool) -> Optional[ValidationIssue]:
        """Check for unusually short or long article text."""

        # Very short articles (unless cross-reference)
        if text_length < self.min_text_length and not is_cross_ref:
            return ValidationIssue(
                issue_type=IssueType.TOO_SHORT,
                severity=IssueSeverity.MEDIUM,
                reason=f"Article text very short ({text_length} chars)",
                confidence=0.75
            )

        # Very long articles (flag for review)
        if text_length > self.max_text_length:
            return ValidationIssue(
                issue_type=IssueType.TOO_LONG,
                severity=IssueSeverity.LOW,  # Just flag, don't remove
                reason=f"Article very long ({text_length:,} chars) - may contain merged content",
                confidence=0.60
            )

        return None

    def _check_alphabetical_range(
        self,
        headword: str,
        volume_num: int,
        edition_year: int
    ) -> Optional[ValidationIssue]:
        """Check if headword falls within expected volume range."""
        # Load volume ranges if not already loaded
        if self._volume_ranges is None:
            self._load_volume_ranges()

        if edition_year not in self._volume_ranges:
            return None

        edition_ranges = self._volume_ranges[edition_year]
        if volume_num not in edition_ranges:
            return None

        start_letter, end_letter = edition_ranges[volume_num]

        # Get first letter of headword
        first_char = headword[0].upper() if headword else ''
        if not first_char.isalpha():
            # Skip non-alphabetic headwords for range check
            return None

        # Simple range check
        if first_char < start_letter or first_char > end_letter:
            return ValidationIssue(
                issue_type=IssueType.OUT_OF_RANGE,
                severity=IssueSeverity.MEDIUM,
                reason=f"'{headword[:20]}...' starts with '{first_char}' but volume {volume_num} covers {start_letter}-{end_letter}",
                confidence=0.85
            )

        return None

    def _load_volume_ranges(self):
        """Load volume ranges from configuration."""
        # Import here to avoid circular imports
        try:
            from .volume_ranges import VOLUME_RANGES
            self._volume_ranges = VOLUME_RANGES
        except ImportError:
            # Fallback to basic ranges from models.py
            from .models import EDITION_CONFIGS
            self._volume_ranges = {}
            for year, config in EDITION_CONFIGS.items():
                if config.volume_ranges:
                    self._volume_ranges[year] = config.volume_ranges

    def _determine_action(self, issues: list[ValidationIssue]) -> str:
        """Determine what action to take based on issues found."""
        if not issues:
            return "keep"

        # Check if any issue warrants removal
        for issue in issues:
            if issue.severity == IssueSeverity.HIGH and issue.confidence >= self.removal_threshold:
                return "remove"

        # Check if any issue warrants flagging
        for issue in issues:
            if issue.confidence >= self.flag_threshold:
                return "flag"

        return "keep"

    def get_removal_stats(self, articles: list[dict]) -> dict:
        """
        Get statistics on how many articles would be affected.

        Args:
            articles: List of article dictionaries

        Returns:
            Dictionary with counts per action and issue type
        """
        stats = {
            "total": len(articles),
            "keep": 0,
            "remove": 0,
            "flag": 0,
            "by_issue_type": {t.value: 0 for t in IssueType}
        }

        for article in articles:
            result = self.validate(article)
            stats[result.action] += 1
            for issue in result.issues:
                stats["by_issue_type"][issue.issue_type.value] += 1

        return stats
