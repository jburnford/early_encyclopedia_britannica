"""
LLM Extractor - Phase 3 of Smart Parser

This module uses Claude to classify ambiguous article boundaries when
regex and fuzzy matching are uncertain. It determines whether a headword
at a given position is:
1. The START of a new encyclopedia article
2. A mere MENTION within another article
3. UNCERTAIN - requires human review

Usage:
    from encyclopedia_parser.llm_extractor import LLMArticleExtractor

    extractor = LLMArticleExtractor()
    result = extractor.classify_candidate(text, "AARON", position=1234)
    if result.is_article_start:
        print(f"Found article: {result.headword}")
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import os
import json
import re

# Use anthropic SDK
try:
    import anthropic
    HAVE_ANTHROPIC = True
except ImportError:
    HAVE_ANTHROPIC = False


@dataclass
class ExtractionResult:
    """Result of LLM-assisted extraction."""
    headword: str
    is_article_start: bool
    confidence: float  # 0-1
    start_position: int
    end_position: Optional[int] = None
    reasoning: str = ""
    classification: str = ""  # "ARTICLE_START", "MENTION", "UNCERTAIN"

    def to_dict(self) -> dict:
        return {
            "headword": self.headword,
            "is_article_start": self.is_article_start,
            "confidence": self.confidence,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "reasoning": self.reasoning,
            "classification": self.classification
        }


@dataclass
class BatchExtractionStats:
    """Statistics from batch extraction."""
    total_candidates: int = 0
    article_starts: int = 0
    mentions: int = 0
    uncertain: int = 0
    heuristic_fallbacks: int = 0
    api_calls: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "total_candidates": self.total_candidates,
            "article_starts": self.article_starts,
            "mentions": self.mentions,
            "uncertain": self.uncertain,
            "heuristic_fallbacks": self.heuristic_fallbacks,
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens
        }


class LLMArticleExtractor:
    """Use LLM to classify ambiguous article boundaries."""

    SYSTEM_PROMPT = """You are an expert at analyzing historical encyclopedia text from the Encyclopedia Britannica (18th-19th century editions). Your task is to determine whether a capitalized term at a specific position is the START of a new encyclopedia article entry, or just a mention/reference within another article.

Encyclopedia articles in this period typically follow these patterns:
- Article starts: "HEADWORD, a definition or description beginning with lowercase..."
- Article starts: "HEADWORD (disambiguation), explanation..."
- Cross-references: "See HEADWORD" or "HEADWORD. See OTHER_TERM"
- Mentions: Terms referenced mid-sentence or in lists

Be precise and consistent in your classifications."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 300,
        cache_results: bool = True
    ):
        """
        Initialize the LLM extractor.

        Args:
            model: Claude model to use (haiku recommended for cost)
            max_tokens: Maximum response tokens
            cache_results: Whether to cache results to avoid duplicate API calls
        """
        self.model = model
        self.max_tokens = max_tokens
        self.cache_results = cache_results
        self._cache: Dict[str, ExtractionResult] = {}
        self._stats = BatchExtractionStats()

        if HAVE_ANTHROPIC:
            self.client = anthropic.Anthropic()
        else:
            self.client = None

    def _get_cache_key(self, headword: str, context: str) -> str:
        """Generate cache key from headword and context."""
        # Use first 200 chars of context for cache key
        return f"{headword}:{hash(context[:200])}"

    def classify_candidate(
        self,
        text: str,
        headword: str,
        position: int,
        context_chars: int = 400
    ) -> ExtractionResult:
        """
        Determine if a headword at position is an article start.

        Args:
            text: Full OCR text
            headword: The headword to classify
            position: Character position in text
            context_chars: Characters of context to include

        Returns:
            ExtractionResult with classification
        """
        self._stats.total_candidates += 1

        # Extract context around position
        start = max(0, position - context_chars)
        end = min(len(text), position + context_chars)
        context = text[start:end]
        relative_pos = position - start

        # Check cache
        if self.cache_results:
            cache_key = self._get_cache_key(headword, context)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Try LLM classification
        if self.client:
            result = self._llm_classify(headword, context, relative_pos, position)
        else:
            result = self._heuristic_classify(text, headword, position)
            self._stats.heuristic_fallbacks += 1

        # Update stats
        if result.classification == "ARTICLE_START":
            self._stats.article_starts += 1
        elif result.classification == "MENTION":
            self._stats.mentions += 1
        else:
            self._stats.uncertain += 1

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = result

        return result

    def _llm_classify(
        self,
        headword: str,
        context: str,
        relative_pos: int,
        absolute_pos: int
    ) -> ExtractionResult:
        """Use Claude to classify the candidate."""
        # Mark the position in context
        marked_context = (
            context[:relative_pos] +
            ">>>" +
            context[relative_pos:]
        )

        prompt = f"""Analyze this encyclopedia text to determine if "{headword}" at the marked position (>>>) is the START of a new article entry, or just a mention within another article.

Context (position marked with >>>):
{marked_context}

Signs of ARTICLE_START:
- Headword appears at the beginning of a line or after a blank line
- Followed by comma and lowercase definition text (e.g., "TERM, a description...")
- Preceded by the end of a previous article (period, then newline)
- May include disambiguation in parentheses: "TERM (Place),"

Signs of MENTION:
- Appears mid-sentence within another article's text
- Part of "See TERM" cross-reference
- Within a list or enumeration
- Referenced as related topic

Respond with EXACTLY one of these on the first line:
ARTICLE_START
MENTION
UNCERTAIN

Then briefly explain your reasoning (1-2 sentences)."""

        try:
            self._stats.api_calls += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()
            lines = result_text.split('\n')
            classification = lines[0].strip().upper()

            # Parse classification
            if classification not in ["ARTICLE_START", "MENTION", "UNCERTAIN"]:
                # Try to extract from first line
                if "ARTICLE_START" in classification:
                    classification = "ARTICLE_START"
                elif "MENTION" in classification:
                    classification = "MENTION"
                else:
                    classification = "UNCERTAIN"

            is_start = classification == "ARTICLE_START"
            confidence = 0.9 if classification in ["ARTICLE_START", "MENTION"] else 0.5
            reasoning = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""

            # Track tokens
            if hasattr(response, 'usage'):
                self._stats.total_tokens += response.usage.input_tokens + response.usage.output_tokens

            return ExtractionResult(
                headword=headword,
                is_article_start=is_start,
                confidence=confidence,
                start_position=absolute_pos,
                reasoning=reasoning,
                classification=classification
            )

        except Exception as e:
            # Fallback to heuristics on error
            self._stats.heuristic_fallbacks += 1
            return self._heuristic_classify_from_context(
                headword, context, relative_pos, absolute_pos, str(e)
            )

    def _heuristic_classify(
        self,
        text: str,
        headword: str,
        position: int
    ) -> ExtractionResult:
        """Fallback heuristic classification when LLM unavailable."""
        # Check if at line start (handle multiple newlines)
        line_start = text.rfind('\n', 0, position)
        chars_before = text[line_start+1:position].strip() if line_start >= 0 else text[:position].strip()

        # Check what follows the headword
        after_pos = position + len(headword)
        chars_after = text[after_pos:after_pos+50] if after_pos < len(text) else ""

        # Heuristics
        at_line_start = len(chars_before) == 0

        # Check for various article start patterns:
        # 1. "TERM, definition..." - direct comma then lowercase
        # 2. "TERM (Disambiguation), definition..." - parenthetical then comma
        # 3. "TERM. See OTHER" - cross-reference article
        followed_by_comma = chars_after.lstrip().startswith(',')
        followed_by_lowercase = bool(re.match(r',\s+[a-z]', chars_after))

        # Handle parenthetical disambiguators: "TERM (Name)," or "TERM (Place),"
        paren_pattern = re.match(r'\s*\([^)]+\),\s+[a-z]', chars_after)
        has_paren_then_lowercase = paren_pattern is not None

        # Handle "TERM. See OTHER" pattern (cross-reference definition)
        see_def_pattern = re.match(r'\.\s+See\s+', chars_after, re.IGNORECASE)
        is_see_definition = see_def_pattern is not None

        # Check for "See TERM" pattern before the headword (mention, not article)
        before_text = text[max(0, position-20):position].lower()
        is_see_reference = 'see ' in before_text or before_text.strip().endswith('see')

        # Determine classification
        if is_see_reference:
            is_start = False
            confidence = 0.85
            classification = "MENTION"
            reasoning = "Appears to be a 'See TERM' cross-reference"
        elif at_line_start and (followed_by_lowercase or has_paren_then_lowercase):
            is_start = True
            confidence = 0.9
            classification = "ARTICLE_START"
            reasoning = f"At line start with definition pattern: {chars_after[:40]}"
        elif at_line_start and is_see_definition:
            is_start = True
            confidence = 0.85
            classification = "ARTICLE_START"
            reasoning = f"Cross-reference article: {chars_after[:40]}"
        elif at_line_start and followed_by_comma:
            is_start = True
            confidence = 0.75
            classification = "ARTICLE_START"
            reasoning = f"At line start with comma: {chars_after[:40]}"
        elif at_line_start:
            # At line start but no clear pattern - might still be article
            is_start = False
            confidence = 0.5
            classification = "UNCERTAIN"
            reasoning = f"At line start but unclear pattern: {chars_after[:40]}"
        else:
            is_start = False
            confidence = 0.7
            classification = "MENTION"
            reasoning = f"Mid-text position, before='{chars_before[-20:]}'"

        return ExtractionResult(
            headword=headword,
            is_article_start=is_start,
            confidence=confidence,
            start_position=position,
            reasoning=f"Heuristic: {reasoning}",
            classification=classification
        )

    def _heuristic_classify_from_context(
        self,
        headword: str,
        context: str,
        relative_pos: int,
        absolute_pos: int,
        error: str
    ) -> ExtractionResult:
        """Heuristic classification using pre-extracted context."""
        before = context[:relative_pos]
        after = context[relative_pos + len(headword):]

        # Check line position
        last_newline = before.rfind('\n')
        chars_before_on_line = before[last_newline+1:].strip() if last_newline >= 0 else before.strip()

        at_line_start = len(chars_before_on_line) == 0
        followed_by_comma = after.lstrip().startswith(',')
        followed_by_lowercase = bool(re.match(r',\s+[a-z]', after.lstrip()))

        if at_line_start and followed_by_lowercase:
            return ExtractionResult(
                headword=headword,
                is_article_start=True,
                confidence=0.75,
                start_position=absolute_pos,
                reasoning=f"Heuristic fallback (LLM error: {error}): at line start with definition pattern",
                classification="ARTICLE_START"
            )
        else:
            return ExtractionResult(
                headword=headword,
                is_article_start=False,
                confidence=0.6,
                start_position=absolute_pos,
                reasoning=f"Heuristic fallback (LLM error: {error}): not at line start or no definition pattern",
                classification="MENTION"
            )

    def classify_batch(
        self,
        text: str,
        candidates: List[Tuple[str, int]],
        confidence_threshold: float = 0.8,
        max_llm_calls: int = 100
    ) -> List[ExtractionResult]:
        """
        Classify multiple candidates, using LLM only for uncertain cases.

        Args:
            text: Full OCR text
            candidates: List of (headword, position) tuples
            confidence_threshold: Use LLM only if heuristic confidence below this
            max_llm_calls: Maximum number of LLM API calls

        Returns:
            List of ExtractionResults
        """
        results = []
        llm_calls = 0

        for headword, position in candidates:
            # First try heuristics
            heuristic_result = self._heuristic_classify(text, headword, position)

            if heuristic_result.confidence >= confidence_threshold:
                # Heuristic is confident enough
                results.append(heuristic_result)
            elif self.client and llm_calls < max_llm_calls:
                # Use LLM for uncertain cases
                llm_result = self.classify_candidate(text, headword, position)
                results.append(llm_result)
                llm_calls += 1
            else:
                # Fall back to heuristic
                results.append(heuristic_result)

        return results

    def get_stats(self) -> BatchExtractionStats:
        """Get extraction statistics."""
        return self._stats

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = BatchExtractionStats()

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()


def find_article_boundaries_with_llm(
    text: str,
    expected_headwords: set,
    extractor: Optional[LLMArticleExtractor] = None,
    max_candidates: int = 500
) -> List[ExtractionResult]:
    """
    Find article boundaries for expected headwords using LLM assistance.

    This function searches for expected headwords in text and uses the
    LLM extractor to classify whether each occurrence is an article start.

    Args:
        text: OCR text to search
        expected_headwords: Set of headwords to find
        extractor: LLMArticleExtractor instance (created if None)
        max_candidates: Maximum candidates to check

    Returns:
        List of ExtractionResults for confirmed article starts
    """
    if extractor is None:
        extractor = LLMArticleExtractor()

    results = []
    candidates_checked = 0

    for headword in expected_headwords:
        if candidates_checked >= max_candidates:
            break

        # Search for headword in text (case-insensitive)
        pattern = re.compile(
            rf'(?:^|\n)({re.escape(headword)}),?\s',
            re.MULTILINE | re.IGNORECASE
        )

        for match in pattern.finditer(text):
            if candidates_checked >= max_candidates:
                break

            result = extractor.classify_candidate(
                text, headword, match.start()
            )

            if result.is_article_start and result.confidence >= 0.7:
                results.append(result)

            candidates_checked += 1

    return results


if __name__ == "__main__":
    print("Testing LLM Article Extractor...")
    print("=" * 60)

    # Check if anthropic is available
    if HAVE_ANTHROPIC:
        print("Anthropic SDK available")
    else:
        print("Anthropic SDK not installed - using heuristics only")

    # Test with sample text
    sample_text = """
the ancient city of Aachen was known for its thermal springs.

AARON, the first high priest of the Jews, was the son of Amram
and Jochebed. He was the elder brother of Moses, and three years
older. The account of his life is to be found in the books of
Exodus, Leviticus, Numbers, and Deuteronomy. When Moses was
commanded to go to Pharaoh, Aaron was appointed to be his
spokesman. See MOSES for more details about this period.

AARSSENS (Francis), lord of Sommelsdyk, a Dutch statesman,
was born at the Hague in 1572. His father Peter was registrar
to the states-general. Francis was a man of great abilities.
"""

    extractor = LLMArticleExtractor()

    # Test candidates
    candidates = [
        ("AARON", sample_text.find("AARON,")),
        ("AARSSENS", sample_text.find("AARSSENS")),
        ("MOSES", sample_text.find("MOSES")),  # This is a mention, not article start
    ]

    print("\nClassifying candidates:")
    for headword, pos in candidates:
        if pos >= 0:
            result = extractor.classify_candidate(sample_text, headword, pos)
            print(f"\n{headword} at position {pos}:")
            print(f"  Classification: {result.classification}")
            print(f"  Is article start: {result.is_article_start}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Reasoning: {result.reasoning[:100]}...")

    # Show stats
    stats = extractor.get_stats()
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Total candidates: {stats.total_candidates}")
    print(f"  Article starts: {stats.article_starts}")
    print(f"  Mentions: {stats.mentions}")
    print(f"  Heuristic fallbacks: {stats.heuristic_fallbacks}")
    print(f"  API calls: {stats.api_calls}")
