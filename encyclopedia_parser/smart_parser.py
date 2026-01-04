"""
Smart Parser - Phase 4 Integration

This module combines all three strategies for robust article extraction:
1. Expected Article Registry - knows what articles SHOULD exist
2. Fuzzy Matcher - finds OCR variations of expected headwords
3. LLM Extractor - classifies ambiguous boundaries

The smart parser first uses existing regex-parsed articles as a baseline,
then attempts to recover missing articles using fuzzy matching and LLM
classification.

Usage:
    from encyclopedia_parser.smart_parser import SmartBritannicaParser

    parser = SmartBritannicaParser(edition_year=1842)
    parser.load_registry("output_v2/index_1842.jsonl", "output_v2")

    results = parser.parse(ocr_text)
    stats = parser.get_coverage_stats()
    print(f"Recovered {len(results)} additional articles")
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import os

from .expected_articles import ExpectedArticleRegistry, ExpectedArticle
from .fuzzy_matcher import FuzzyMatcher, FuzzyMatch
from .llm_extractor import LLMArticleExtractor, ExtractionResult


@dataclass
class SmartParseResult:
    """Result of smart parsing - a recovered article."""
    headword: str
    start_position: int
    end_position: int
    text: str
    confidence: float
    match_type: str  # "regex", "exact", "fuzzy", "llm", "variation"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    ocr_text_matched: str = ""  # What was actually in the OCR

    def to_dict(self) -> dict:
        return {
            "headword": self.headword,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "text_length": len(self.text),
            "confidence": self.confidence,
            "match_type": self.match_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "ocr_text_matched": self.ocr_text_matched
        }


@dataclass
class SmartParserStats:
    """Statistics from smart parsing."""
    edition_year: int
    expected_count: int = 0
    already_parsed: int = 0
    recovered_exact: int = 0
    recovered_fuzzy: int = 0
    recovered_llm: int = 0
    still_missing: int = 0
    total_recovered: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0

    def to_dict(self) -> dict:
        return {
            "edition_year": self.edition_year,
            "expected_count": self.expected_count,
            "already_parsed": self.already_parsed,
            "recovered_exact": self.recovered_exact,
            "recovered_fuzzy": self.recovered_fuzzy,
            "recovered_llm": self.recovered_llm,
            "total_recovered": self.total_recovered,
            "still_missing": self.still_missing,
            "coverage_before": f"{self.coverage_before:.1f}%",
            "coverage_after": f"{self.coverage_after:.1f}%"
        }


class SmartBritannicaParser:
    """Smart parser combining regex, fuzzy matching, and LLM extraction."""

    def __init__(
        self,
        edition_year: int,
        registry: Optional[ExpectedArticleRegistry] = None,
        use_llm: bool = False,
        fuzzy_threshold: int = 85,
        llm_model: str = "claude-3-haiku-20240307"
    ):
        """
        Initialize the smart parser.

        Args:
            edition_year: The edition year to parse (e.g., 1842)
            registry: Pre-built ExpectedArticleRegistry (created if None)
            use_llm: Whether to use LLM for ambiguous cases
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            llm_model: Claude model for LLM classification
        """
        self.edition_year = edition_year
        self.registry = registry or ExpectedArticleRegistry()
        self.use_llm = use_llm
        self.fuzzy_threshold = fuzzy_threshold

        # Initialize components
        self.fuzzy_matcher: Optional[FuzzyMatcher] = None
        self.llm_extractor: Optional[LLMArticleExtractor] = None

        if use_llm:
            self.llm_extractor = LLMArticleExtractor(model=llm_model)

        self._stats = SmartParserStats(edition_year=edition_year)
        self._already_parsed: Set[str] = set()

    def load_registry(
        self,
        index_path: str = "output_v2/index_1842.jsonl",
        output_dir: str = "output_v2"
    ):
        """
        Load the expected article registry from index and parsed articles.

        Args:
            index_path: Path to 1842 index JSONL
            output_dir: Directory containing articles_*.jsonl files
        """
        # Load index
        if os.path.exists(index_path):
            self.registry.load_from_index(index_path, 1842)

        # Load parsed articles for this edition
        articles_path = os.path.join(output_dir, f"articles_{self.edition_year}.jsonl")
        if os.path.exists(articles_path):
            self.registry.load_from_parsed_articles(articles_path, self.edition_year)

            # Track already-parsed headwords
            with open(articles_path, 'r') as f:
                for line in f:
                    try:
                        article = json.loads(line)
                        hw = article.get('headword', '').upper().strip()
                        if hw:
                            self._already_parsed.add(hw)
                    except json.JSONDecodeError:
                        continue

        # Initialize fuzzy matcher with expected headwords
        expected = self.registry.get_expected_for_edition(1842)
        self.fuzzy_matcher = FuzzyMatcher(expected, fuzzy_threshold=self.fuzzy_threshold)

        # Update stats
        self._stats.expected_count = len(expected)
        self._stats.already_parsed = len(self._already_parsed & expected)
        self._stats.coverage_before = (
            100 * self._stats.already_parsed / self._stats.expected_count
            if self._stats.expected_count > 0 else 0
        )

    def _find_article_end(
        self,
        text: str,
        start: int,
        max_length: int = 100000
    ) -> int:
        """
        Find the end of an article (next article start or max length).

        Args:
            text: Full OCR text
            start: Start position of article
            max_length: Maximum article length

        Returns:
            End position of article
        """
        # Look for next article start pattern
        # Skip first 50 chars to avoid matching current article
        search_start = start + 50
        search_text = text[search_start:start + max_length]

        # Pattern for article start: newline(s), then HEADWORD, comma, lowercase
        pattern = re.compile(r'\n\n?([A-Z][A-Z\s\-\']+),\s+[a-z]')
        match = pattern.search(search_text)

        if match:
            return search_start + match.start()
        return min(start + max_length, len(text))

    def _get_page_at_position(
        self,
        position: int,
        text: str,
        page_markers: Optional[List[Tuple[int, int]]] = None
    ) -> Optional[int]:
        """
        Get the page number at a given character position.

        Args:
            position: Character position in text
            text: Full OCR text
            page_markers: Optional list of (position, page_number) tuples

        Returns:
            Page number or None if cannot determine
        """
        if not page_markers:
            return None

        # Find the page marker just before this position
        for pos, page in reversed(page_markers):
            if pos <= position:
                return page
        return None

    def parse(
        self,
        text: str,
        page_markers: Optional[List[Tuple[int, int]]] = None,
        min_confidence: float = 0.8,
        max_llm_calls: int = 100
    ) -> List[SmartParseResult]:
        """
        Parse text to recover missing articles.

        Args:
            text: OCR text to parse
            page_markers: Optional list of (position, page_number) for page tracking
            min_confidence: Minimum confidence to accept a match
            max_llm_calls: Maximum LLM API calls for this parse

        Returns:
            List of SmartParseResult for recovered articles
        """
        if not self.fuzzy_matcher:
            raise RuntimeError("Registry not loaded. Call load_registry() first.")

        results = []
        found_positions: Set[int] = set()

        # Get missing headwords (expected but not parsed)
        expected = self.registry.get_expected_for_edition(1842)
        missing = expected - self._already_parsed

        if not missing:
            return []

        # Step 1: Find matches using fuzzy matcher
        self.fuzzy_matcher.headwords = missing  # Only search for missing
        matches = self.fuzzy_matcher.find_in_text(text, include_fuzzy=True)

        # Step 2: Process matches
        llm_calls = 0

        for match in matches:
            if match.position in found_positions:
                continue

            # Skip low confidence matches
            if match.confidence < min_confidence:
                # Try LLM for borderline cases
                if self.use_llm and self.llm_extractor and llm_calls < max_llm_calls:
                    if match.confidence >= 0.7:  # Worth checking with LLM
                        llm_result = self.llm_extractor.classify_candidate(
                            text, match.headword, match.position
                        )
                        llm_calls += 1

                        if llm_result.is_article_start and llm_result.confidence >= min_confidence:
                            end_pos = self._find_article_end(text, match.position)
                            results.append(SmartParseResult(
                                headword=match.headword,
                                start_position=match.position,
                                end_position=end_pos,
                                text=text[match.position:end_pos],
                                confidence=llm_result.confidence,
                                match_type="llm",
                                page_start=self._get_page_at_position(match.position, text, page_markers),
                                page_end=self._get_page_at_position(end_pos, text, page_markers),
                                ocr_text_matched=match.matched_text
                            ))
                            found_positions.add(match.position)
                            self._stats.recovered_llm += 1
                continue

            # Accept high-confidence match
            end_pos = self._find_article_end(text, match.position)
            results.append(SmartParseResult(
                headword=match.headword,
                start_position=match.position,
                end_position=end_pos,
                text=text[match.position:end_pos],
                confidence=match.confidence,
                match_type=match.match_type,
                page_start=self._get_page_at_position(match.position, text, page_markers),
                page_end=self._get_page_at_position(end_pos, text, page_markers),
                ocr_text_matched=match.matched_text
            ))
            found_positions.add(match.position)

            # Update stats
            if match.match_type == "exact":
                self._stats.recovered_exact += 1
            elif match.match_type == "fuzzy":
                self._stats.recovered_fuzzy += 1

        # Step 3: Try LLM for remaining high-value missing headwords
        if self.use_llm and self.llm_extractor and llm_calls < max_llm_calls:
            recovered_headwords = {r.headword for r in results}
            still_missing = missing - recovered_headwords - self._already_parsed

            # Prioritize shorter headwords (more likely to be real articles)
            priority_missing = sorted(still_missing, key=len)[:max_llm_calls - llm_calls]

            for headword in priority_missing:
                # Search for this specific headword
                pattern = re.compile(
                    rf'(?:^|\n)({re.escape(headword)})\s*[\(,]',
                    re.MULTILINE | re.IGNORECASE
                )

                for match in pattern.finditer(text):
                    if match.start() in found_positions:
                        continue

                    llm_result = self.llm_extractor.classify_candidate(
                        text, headword, match.start()
                    )
                    llm_calls += 1

                    if llm_result.is_article_start and llm_result.confidence >= min_confidence:
                        end_pos = self._find_article_end(text, match.start())
                        results.append(SmartParseResult(
                            headword=headword,
                            start_position=match.start(),
                            end_position=end_pos,
                            text=text[match.start():end_pos],
                            confidence=llm_result.confidence,
                            match_type="llm",
                            page_start=self._get_page_at_position(match.start(), text, page_markers),
                            page_end=self._get_page_at_position(end_pos, text, page_markers),
                            ocr_text_matched=match.group(1)
                        ))
                        found_positions.add(match.start())
                        self._stats.recovered_llm += 1
                        break  # Found this headword, move to next

                    if llm_calls >= max_llm_calls:
                        break

                if llm_calls >= max_llm_calls:
                    break

        # Update final stats
        self._stats.total_recovered = len(results)
        recovered_headwords = {r.headword for r in results}
        all_found = self._already_parsed | recovered_headwords
        self._stats.still_missing = len(expected - all_found)
        self._stats.coverage_after = (
            100 * len(all_found & expected) / self._stats.expected_count
            if self._stats.expected_count > 0 else 0
        )

        return sorted(results, key=lambda r: r.start_position)

    def get_stats(self) -> SmartParserStats:
        """Get parsing statistics."""
        return self._stats

    def get_coverage_report(self) -> str:
        """Generate a human-readable coverage report."""
        s = self._stats
        lines = [
            f"Smart Parser Coverage Report - {s.edition_year} Edition",
            "=" * 50,
            f"Expected articles (from 1842 index): {s.expected_count:,}",
            f"",
            f"Already parsed by regex: {s.already_parsed:,}",
            f"Recovered by exact match: {s.recovered_exact:,}",
            f"Recovered by fuzzy match: {s.recovered_fuzzy:,}",
            f"Recovered by LLM: {s.recovered_llm:,}",
            f"Total recovered: {s.total_recovered:,}",
            f"",
            f"Still missing: {s.still_missing:,}",
            f"",
            f"Coverage before: {s.coverage_before:.1f}%",
            f"Coverage after: {s.coverage_after:.1f}%",
            f"Improvement: +{s.coverage_after - s.coverage_before:.1f}%",
        ]
        return "\n".join(lines)

    def save_results(self, results: List[SmartParseResult], filepath: str):
        """Save parsed results to JSONL file."""
        with open(filepath, 'w') as f:
            for result in results:
                # Create article-like structure
                article = {
                    "article_id": f"{self.edition_year}_smart_{result.headword}",
                    "headword": result.headword,
                    "text": result.text,
                    "match_type": result.match_type,
                    "confidence": result.confidence,
                    "ocr_matched": result.ocr_text_matched,
                    "start_position": result.start_position,
                    "page_start": result.page_start
                }
                f.write(json.dumps(article) + "\n")


def recover_missing_articles(
    edition_year: int = 1842,
    index_path: str = "output_v2/index_1842.jsonl",
    output_dir: str = "output_v2",
    ocr_dir: str = "ocr_results",
    use_llm: bool = False,
    fuzzy_threshold: int = 85,
    output_file: Optional[str] = None
) -> Tuple[List[SmartParseResult], SmartParserStats]:
    """
    Convenience function to recover missing articles for an edition.

    Args:
        edition_year: Edition year to process
        index_path: Path to 1842 index
        output_dir: Directory with parsed articles
        ocr_dir: Directory with OCR text files
        use_llm: Whether to use LLM classification
        fuzzy_threshold: Minimum fuzzy match score
        output_file: Optional path to save results

    Returns:
        Tuple of (results list, stats)
    """
    # Initialize parser
    parser = SmartBritannicaParser(
        edition_year=edition_year,
        use_llm=use_llm,
        fuzzy_threshold=fuzzy_threshold
    )
    parser.load_registry(index_path, output_dir)

    # Find OCR files for this edition
    all_results = []

    # Map edition years to OCR directories
    ocr_paths = {
        1771: "1771_britannica_1st",
        1778: "1778_britannica_2nd",
        1797: "1797_britannica_3rd",
        1810: "1810_britannica_4th",
        # 5th and 6th editions are in json/ directory
    }

    edition_dir = ocr_paths.get(edition_year)
    if edition_dir:
        ocr_path = os.path.join(ocr_dir, edition_dir)
        if os.path.exists(ocr_path):
            for filename in sorted(os.listdir(ocr_path)):
                if filename.endswith('.md'):
                    filepath = os.path.join(ocr_path, filename)
                    with open(filepath, 'r') as f:
                        text = f.read()

                    results = parser.parse(text)
                    all_results.extend(results)

    # Save results if requested
    if output_file and all_results:
        parser.save_results(all_results, output_file)

    return all_results, parser.get_stats()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from encyclopedia_parser.expected_articles import ExpectedArticleRegistry
    from encyclopedia_parser.fuzzy_matcher import FuzzyMatcher
    from encyclopedia_parser.llm_extractor import LLMArticleExtractor

    print("Testing Smart Parser...")
    print("=" * 60)

    # Initialize parser
    parser = SmartBritannicaParser(edition_year=1797, use_llm=False)
    parser.load_registry("output_v2/index_1842.jsonl", "output_v2")

    print(f"Loaded registry with {parser._stats.expected_count:,} expected articles")
    print(f"Already parsed: {parser._stats.already_parsed:,}")
    print(f"Coverage before: {parser._stats.coverage_before:.1f}%")

    # Load a sample OCR file
    ocr_file = "ocr_results/1797_britannica_3rd/Third edition - Encyclopaedia Britannica Volume 1, A-ANG.md"

    if os.path.exists(ocr_file):
        with open(ocr_file, 'r') as f:
            text = f.read()

        print(f"\nLoaded OCR text: {len(text):,} chars")

        # Parse to recover missing articles
        results = parser.parse(text, min_confidence=0.85)

        print(f"\nRecovered {len(results)} articles:")
        for r in results[:20]:
            print(f"  {r.headword:20} ({r.match_type}, {r.confidence:.0%})")
            if r.ocr_text_matched != r.headword:
                print(f"    OCR: {r.ocr_text_matched}")

        print("\n" + parser.get_coverage_report())
    else:
        print(f"OCR file not found: {ocr_file}")

        # Test with sample text
        sample_text = """
AACHEN, or AIX-LA-CHAPELLE, a city of Prussia, is situated
in a valley on the small river Worm. It was the favorite
residence of Charlemagne, who built a palace here.

AAHUS, a seaport of Denmark, in Jutland. Population 5000.

AALBORG, a city of Denmark, capital of the diocese of that
name, situated on the south side of the Lymfiord.

AALEN, a town of Wurtemberg, on the river Kocher, contains
about 4000 inhabitants and manufactures woollen cloth.
"""
        print("\nTesting with sample text...")
        results = parser.parse(sample_text, min_confidence=0.8)

        print(f"Recovered {len(results)} articles:")
        for r in results:
            print(f"  {r.headword:20} ({r.match_type}, {r.confidence:.0%})")
