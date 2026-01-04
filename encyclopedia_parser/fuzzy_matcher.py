"""
Fuzzy Matcher - Phase 2 of Smart Parser

This module finds expected headwords in OCR text using multiple strategies:
1. Exact matching (HEADWORD,)
2. Case-insensitive matching (Headword, or HeAdWoRd,)
3. Fuzzy matching for OCR errors (AALSMEEB -> AALSMEER)
4. Known variation matching (learned from previous runs)

Uses rapidfuzz for efficient fuzzy string matching.

Usage:
    from encyclopedia_parser.fuzzy_matcher import FuzzyMatcher
    from encyclopedia_parser.expected_articles import ExpectedArticleRegistry

    registry = ExpectedArticleRegistry()
    registry.load_from_index("output_v2/index_1842.jsonl", 1842)

    expected = registry.get_expected_for_edition(1842)
    matcher = FuzzyMatcher(expected)

    matches = matcher.find_in_text(ocr_text)
    for m in matches:
        print(f"{m.headword} at {m.position} ({m.match_type}, {m.confidence:.0%})")
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import re
import json
from collections import defaultdict

# rapidfuzz for efficient fuzzy matching
try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False
    print("Warning: rapidfuzz not installed. Fuzzy matching disabled.")
    print("Install with: pip install rapidfuzz")


@dataclass
class FuzzyMatch:
    """A fuzzy match result."""
    headword: str  # The canonical headword (normalized uppercase)
    matched_text: str  # What was actually found in OCR
    position: int  # Character position in text
    confidence: float  # Match confidence (0-1)
    match_type: str  # "exact", "case_insensitive", "fuzzy", "variation"
    line_number: int = 0  # Approximate line number
    context: str = ""  # Surrounding text for debugging

    def to_dict(self) -> dict:
        return {
            "headword": self.headword,
            "matched_text": self.matched_text,
            "position": self.position,
            "confidence": self.confidence,
            "match_type": self.match_type,
            "line_number": self.line_number,
            "context": self.context
        }


class FuzzyMatcher:
    """Find expected headwords in OCR text with fuzzy matching."""

    # Common OCR substitution patterns
    OCR_SUBSTITUTIONS = {
        'E': ['F', 'B', 'L'],  # E often misread
        'I': ['J', 'L', '1', '|'],
        'O': ['0', 'Q', 'D'],
        'S': ['5', '$'],
        'B': ['8', 'R', 'E'],
        'G': ['6', 'C'],
        'Z': ['2'],
        'A': ['4'],
        'T': ['I', '7'],
        'M': ['N', 'W'],
        'N': ['M', 'H'],
        'R': ['P', 'B'],
        'U': ['V', 'O'],
        'V': ['U', 'Y'],
        'W': ['M', 'VV'],
        'C': ['G', 'O', '('],
        'L': ['I', '1', '|'],
        'H': ['N', 'R'],
        'D': ['O', 'P'],
        'P': ['R', 'D'],
        'F': ['E', 'T'],
        'K': ['X', 'R'],
        'Y': ['V', 'T'],
    }

    def __init__(
        self,
        expected_headwords: Set[str],
        min_headword_length: int = 2,
        fuzzy_threshold: int = 80
    ):
        """
        Initialize the fuzzy matcher.

        Args:
            expected_headwords: Set of normalized uppercase headwords to find
            min_headword_length: Minimum length for headwords to match
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.headwords = {
            hw for hw in expected_headwords
            if len(hw) >= min_headword_length
        }
        self.min_length = min_headword_length
        self.fuzzy_threshold = fuzzy_threshold
        self.variation_map: Dict[str, str] = {}  # variant -> canonical
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for headword detection."""
        # Pattern for ALL CAPS article starts: HEADWORD, followed by text
        # Captures the headword (allowing hyphens, spaces, apostrophes)
        self.allcaps_pattern = re.compile(
            r'(?:^|\n\n?)([A-Z][A-Z\s\-\'\.]+?),\s+[a-z]',
            re.MULTILINE
        )

        # Pattern for Title Case article starts: Headword, followed by text
        self.titlecase_pattern = re.compile(
            r'(?:^|\n\n?)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+[a-z]',
            re.MULTILINE
        )

        # Combined pattern for any case
        self.any_article_pattern = re.compile(
            r'(?:^|\n\n?)([A-Za-z][A-Za-z\s\-\'\.]+?),\s+[a-z]',
            re.MULTILINE
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.upper().strip().replace('  ', ' ')

    def _get_line_number(self, text: str, position: int) -> int:
        """Get approximate line number for a position."""
        return text[:position].count('\n') + 1

    def _get_context(self, text: str, position: int, chars: int = 50) -> str:
        """Get context around a position."""
        start = max(0, position - chars)
        end = min(len(text), position + chars)
        context = text[start:end].replace('\n', ' ')
        return f"...{context}..."

    def find_exact_matches(self, text: str) -> List[FuzzyMatch]:
        """Find exact matches for expected headwords."""
        matches = []
        found_positions = set()

        for pattern in [self.allcaps_pattern, self.titlecase_pattern]:
            for match in pattern.finditer(text):
                if match.start() in found_positions:
                    continue

                candidate = match.group(1).strip()
                normalized = self._normalize(candidate)

                # Remove trailing periods for matching
                normalized_clean = normalized.rstrip('.')

                if normalized_clean in self.headwords:
                    matches.append(FuzzyMatch(
                        headword=normalized_clean,
                        matched_text=candidate,
                        position=match.start(),
                        confidence=1.0,
                        match_type="exact",
                        line_number=self._get_line_number(text, match.start()),
                        context=self._get_context(text, match.start())
                    ))
                    found_positions.add(match.start())

        return matches

    def find_variation_matches(self, text: str) -> List[FuzzyMatch]:
        """Find matches using known OCR variations."""
        matches = []

        for variant, canonical in self.variation_map.items():
            # Build pattern for this variant
            pattern = re.compile(
                rf'(?:^|\n\n?)({re.escape(variant)}),\s+[a-z]',
                re.MULTILINE | re.IGNORECASE
            )

            for match in pattern.finditer(text):
                matches.append(FuzzyMatch(
                    headword=canonical,
                    matched_text=match.group(1),
                    position=match.start(),
                    confidence=0.95,
                    match_type="variation",
                    line_number=self._get_line_number(text, match.start()),
                    context=self._get_context(text, match.start())
                ))

        return matches

    def find_fuzzy_matches(
        self,
        text: str,
        already_found: Set[str],
        max_candidates: int = 5000
    ) -> List[FuzzyMatch]:
        """
        Find fuzzy matches for missing headwords.

        Args:
            text: OCR text to search
            already_found: Headwords already found by exact matching
            max_candidates: Maximum number of text candidates to check

        Returns:
            List of fuzzy matches
        """
        if not HAVE_RAPIDFUZZ:
            return []

        matches = []
        missing = self.headwords - already_found

        if not missing:
            return []

        # Find all potential headword positions in text
        candidates = []
        for match in self.any_article_pattern.finditer(text):
            candidate = match.group(1).strip()
            normalized = self._normalize(candidate)
            if len(normalized) >= self.min_length:
                candidates.append((normalized, candidate, match.start()))

            if len(candidates) >= max_candidates:
                break

        if not candidates:
            return []

        # For each candidate, check if it fuzzy-matches any missing headword
        for normalized, original, position in candidates:
            # Skip if already matched
            if normalized in already_found or normalized in self.headwords:
                continue

            # Find best match among missing headwords
            result = process.extractOne(
                normalized,
                missing,
                scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold
            )

            if result:
                best_match, score, _ = result
                matches.append(FuzzyMatch(
                    headword=best_match,
                    matched_text=original,
                    position=position,
                    confidence=score / 100.0,
                    match_type="fuzzy",
                    line_number=self._get_line_number(text, position),
                    context=self._get_context(text, position)
                ))

        return matches

    def find_in_text(
        self,
        text: str,
        include_fuzzy: bool = True,
        fuzzy_score_cutoff: int = None
    ) -> List[FuzzyMatch]:
        """
        Find expected headwords in text using all strategies.

        Args:
            text: OCR text to search
            include_fuzzy: Whether to include fuzzy matching (slower)
            fuzzy_score_cutoff: Override default fuzzy threshold

        Returns:
            List of matches sorted by position
        """
        if fuzzy_score_cutoff:
            self.fuzzy_threshold = fuzzy_score_cutoff

        all_matches = []
        found_positions = set()

        # Strategy 1: Exact matches
        exact = self.find_exact_matches(text)
        for m in exact:
            if m.position not in found_positions:
                all_matches.append(m)
                found_positions.add(m.position)

        # Strategy 2: Known variations
        variations = self.find_variation_matches(text)
        for m in variations:
            if m.position not in found_positions:
                all_matches.append(m)
                found_positions.add(m.position)

        # Strategy 3: Fuzzy matching for remaining
        if include_fuzzy and HAVE_RAPIDFUZZ:
            found_headwords = {m.headword for m in all_matches}
            fuzzy = self.find_fuzzy_matches(text, found_headwords)
            for m in fuzzy:
                if m.position not in found_positions:
                    all_matches.append(m)
                    found_positions.add(m.position)

        return sorted(all_matches, key=lambda m: m.position)

    def learn_variations(
        self,
        parsed_headwords: List[str],
        expected_headwords: Set[str],
        score_cutoff: int = 80
    ) -> Dict[str, str]:
        """
        Learn OCR variations by comparing parsed headwords to expected.

        When the parser finds a headword that isn't in the expected set,
        but is very similar to an expected one, it's likely an OCR error.

        Args:
            parsed_headwords: Headwords found by the regex parser
            expected_headwords: Expected headwords from index
            score_cutoff: Minimum similarity score to consider a variation

        Returns:
            Dict of {variant: canonical} mappings learned
        """
        if not HAVE_RAPIDFUZZ:
            return {}

        learned = {}
        expected_set = {self._normalize(hw) for hw in expected_headwords}

        for parsed in parsed_headwords:
            normalized = self._normalize(parsed)

            # Skip if it's already a known headword
            if normalized in expected_set:
                continue

            # Skip if too short
            if len(normalized) < self.min_length:
                continue

            # Find closest expected headword
            result = process.extractOne(
                normalized,
                expected_set,
                scorer=fuzz.ratio,
                score_cutoff=score_cutoff
            )

            if result:
                canonical, score, _ = result
                if score < 100:  # Not exact match
                    learned[normalized] = canonical
                    self.variation_map[normalized] = canonical

        return learned

    def get_ocr_error_analysis(
        self,
        variations: Dict[str, str]
    ) -> Dict[str, int]:
        """
        Analyze OCR error patterns from learned variations.

        Args:
            variations: Dict of {variant: canonical} from learn_variations

        Returns:
            Dict of {error_pattern: count} showing common OCR errors
        """
        error_counts = defaultdict(int)

        for variant, canonical in variations.items():
            # Compare character by character
            if len(variant) == len(canonical):
                for v_char, c_char in zip(variant, canonical):
                    if v_char != c_char:
                        error_counts[f"{c_char}->{v_char}"] += 1

        return dict(sorted(error_counts.items(), key=lambda x: -x[1]))

    def add_variation(self, variant: str, canonical: str):
        """Manually add a known variation."""
        self.variation_map[self._normalize(variant)] = self._normalize(canonical)

    def save_variations(self, filepath: str):
        """Save learned variations to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.variation_map, f, indent=2)

    def load_variations(self, filepath: str):
        """Load variations from JSON file."""
        with open(filepath, 'r') as f:
            self.variation_map = json.load(f)


def analyze_missing_articles(
    registry,
    edition_year: int = 1842,
    sample_size: int = 100
) -> Dict:
    """
    Analyze patterns in missing articles to understand why they're missed.

    Args:
        registry: ExpectedArticleRegistry instance
        edition_year: Edition year to analyze
        sample_size: Number of missing articles to analyze

    Returns:
        Dict with analysis results
    """
    missing = registry.get_missing_for_edition(edition_year)
    sample = sorted(list(missing))[:sample_size]

    analysis = {
        "total_missing": len(missing),
        "patterns": {
            "short_words": 0,  # 2-3 chars
            "geographic": 0,  # Likely place names (AA, AAR, etc.)
            "with_parentheses": 0,  # AJAX (Greater)
            "with_numbers": 0,  # HENRY VII
            "all_caps": 0,
            "mixed_case": 0,
        },
        "length_distribution": defaultdict(int),
        "first_letter_distribution": defaultdict(int),
        "sample": sample[:20]
    }

    for hw in sample:
        # Length analysis
        analysis["length_distribution"][len(hw)] += 1
        analysis["first_letter_distribution"][hw[0]] += 1

        # Pattern detection
        if len(hw) <= 3:
            analysis["patterns"]["short_words"] += 1
        if len(hw) <= 5 and hw.isalpha():
            analysis["patterns"]["geographic"] += 1
        if '(' in hw:
            analysis["patterns"]["with_parentheses"] += 1
        if any(c.isdigit() for c in hw):
            analysis["patterns"]["with_numbers"] += 1

    return analysis


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from encyclopedia_parser.expected_articles import ExpectedArticleRegistry

    print("Testing Fuzzy Matcher...")
    print("=" * 60)

    # Build registry
    registry = ExpectedArticleRegistry()
    registry.load_from_index("output_v2/index_1842.jsonl", 1842)
    registry.load_from_parsed_articles("output_v2/articles_1842.jsonl", 1842)

    # Get expected and found
    expected = registry.get_expected_for_edition(1842)
    found = registry.get_found_for_edition(1842)
    missing = expected - found

    print(f"Expected: {len(expected):,}")
    print(f"Found: {len(found):,}")
    print(f"Missing: {len(missing):,}")

    # Analyze missing articles
    print("\n" + "=" * 60)
    print("ANALYZING MISSING ARTICLES")
    print("=" * 60)

    analysis = analyze_missing_articles(registry, 1842)
    print(f"Total missing: {analysis['total_missing']:,}")
    print("\nPatterns detected:")
    for pattern, count in analysis['patterns'].items():
        print(f"  {pattern}: {count}")

    print("\nLength distribution (first 10):")
    for length, count in sorted(analysis['length_distribution'].items())[:10]:
        print(f"  {length} chars: {count}")

    # Test fuzzy matching on sample text
    print("\n" + "=" * 60)
    print("TESTING FUZZY MATCHING")
    print("=" * 60)

    # Create matcher with missing headwords
    matcher = FuzzyMatcher(missing, fuzzy_threshold=85)

    # Sample OCR text with intentional errors
    sample_text = """
AALEN, a town in the kingdom of Wurtemberg, in the circle of Jaxt,
situated on the Kocher, contains about 4000 inhabitants.

AALSMEEB, a town in Holland, in the province of North Holland,
about 8 miles south-west of Amsterdam.

AARAU, a town in Switzerland, capital of the canton of Aargau,
situated on the right bank of the Aar.

AARON, the first high priest of the Jews, was the son of Amram
and Jochebed, and the elder brother of Moses.
"""

    matches = matcher.find_in_text(sample_text)

    print(f"Found {len(matches)} matches in sample text:")
    for m in matches:
        print(f"  {m.headword} <- '{m.matched_text}' ({m.match_type}, {m.confidence:.0%})")

    # Learn variations from parsed vs expected
    print("\n" + "=" * 60)
    print("LEARNING OCR VARIATIONS")
    print("=" * 60)

    # Get all parsed headwords from 1842
    import json
    parsed_headwords = []
    with open("output_v2/articles_1842.jsonl") as f:
        for line in f:
            article = json.loads(line)
            parsed_headwords.append(article.get('headword', ''))

    # Create new matcher with expected headwords
    matcher2 = FuzzyMatcher(expected, fuzzy_threshold=80)
    variations = matcher2.learn_variations(parsed_headwords, expected)

    print(f"Learned {len(variations)} OCR variations:")
    for variant, canonical in list(variations.items())[:20]:
        print(f"  {variant} -> {canonical}")

    if variations:
        error_analysis = matcher2.get_ocr_error_analysis(variations)
        print("\nCommon OCR errors:")
        for error, count in list(error_analysis.items())[:10]:
            print(f"  {error}: {count}")
