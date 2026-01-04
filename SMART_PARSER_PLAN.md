# Smart Parser Implementation Plan

## Project Context

This project contains OCR'd text from 8 editions of the Encyclopedia Britannica (1771-1860) from the National Library of Scotland. We have:

- **8 editions**: 1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860
- **1842 Index**: 18,215 main entries with volume/page references
- **Current parser**: `parse_britannica_articles.py` - regex-based, misses ~30-40% of articles
- **Website**: Generated via `generate_site_optimized.py` with 4.2M hyperlinks

### Key Files
- `parse_britannica_articles.py` - Main article parser (has Title Case support added)
- `generate_site_optimized.py` - Aho-Corasick based site generator
- `output_v2/articles_*.jsonl` - Parsed articles per edition
- `output_v2/index_1842.jsonl` - Parsed 1842 index with 36,377 entries
- `encyclopedia_parser/` - Existing parser package with models, chunkers, classifiers

### Problem Statement

The current regex-based parser misses ~30-40% of articles due to:
1. **OCR case variations**: "Aalen," vs "AALEN,"
2. **OCR spelling errors**: "Aalsmeer" vs "AALSMEEB"
3. **Formatting variations**: Different headword patterns across editions
4. **No cross-validation**: Parser doesn't use knowledge of what SHOULD exist

### Recent Progress
- Added Title Case pattern detection (validated against 1842 index)
- Improved 1842 vol 2 coverage from 72% to 83%
- Remaining gap (~17%) is due to OCR spelling errors

---

## Implementation Plan

### Phase 1: Expected Article Registry

Create `encyclopedia_parser/expected_articles.py`:

```python
from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional
import json
import os

@dataclass
class ExpectedArticle:
    """An article we expect to find in one or more editions."""
    headword: str  # Normalized uppercase
    editions_expected: Set[int] = field(default_factory=set)
    confidence: float = 1.0  # How certain we are
    source: str = "unknown"  # "index", "cross_edition", "cross_ref"
    known_variations: List[str] = field(default_factory=list)

class ExpectedArticleRegistry:
    """Registry of expected articles built from index + cross-edition data."""

    def __init__(self):
        self.articles: Dict[str, ExpectedArticle] = {}

    def load_from_index(self, index_path: str, edition_year: int):
        """Load headwords from 1842 index JSONL."""
        with open(index_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if entry.get('entry_type') == 'main' and entry.get('references'):
                    term = entry['term'].upper().strip()
                    if len(term) > 1:
                        if term not in self.articles:
                            self.articles[term] = ExpectedArticle(
                                headword=term,
                                editions_expected={edition_year},
                                confidence=1.0,
                                source="index"
                            )
                        else:
                            self.articles[term].editions_expected.add(edition_year)

    def load_from_parsed_articles(self, articles_path: str, edition_year: int):
        """Load headwords from already-parsed articles."""
        with open(articles_path, 'r') as f:
            for line in f:
                article = json.loads(line)
                headword = article.get('headword', '').upper().strip()
                if len(headword) > 1:
                    if headword not in self.articles:
                        self.articles[headword] = ExpectedArticle(
                            headword=headword,
                            editions_expected={edition_year},
                            confidence=0.9,
                            source="parsed"
                        )
                    else:
                        self.articles[headword].editions_expected.add(edition_year)

    def infer_cross_edition(self):
        """Infer expected articles based on cross-edition patterns."""
        # If article in 1815 AND 1823, likely in 1810 too
        edition_families = {
            1810: {1815, 1823},  # 1810 likely has what 1815 and 1823 share
            1797: {1810, 1815},
        }
        for target_year, source_years in edition_families.items():
            for headword, article in self.articles.items():
                if source_years.issubset(article.editions_expected):
                    if target_year not in article.editions_expected:
                        article.editions_expected.add(target_year)
                        article.confidence = min(article.confidence, 0.7)

    def get_expected_for_edition(self, year: int) -> Set[str]:
        """Get all headwords expected for a given edition."""
        return {hw for hw, a in self.articles.items() if year in a.editions_expected}

    def add_variation(self, canonical: str, variant: str):
        """Add a known OCR variation."""
        if canonical in self.articles:
            self.articles[canonical].known_variations.append(variant)
```

### Phase 2: Fuzzy Matcher

Create `encyclopedia_parser/fuzzy_matcher.py`:

```python
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import re

# Optional: pip install rapidfuzz for faster fuzzy matching
try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False

@dataclass
class FuzzyMatch:
    """A fuzzy match result."""
    headword: str  # The canonical headword
    matched_text: str  # What was found in OCR
    position: int  # Character position in text
    confidence: float  # Match confidence (0-1)
    match_type: str  # "exact", "case_insensitive", "fuzzy", "variation"

class FuzzyMatcher:
    """Find expected headwords in OCR text with fuzzy matching."""

    def __init__(self, expected_headwords: Set[str]):
        self.headwords = expected_headwords
        self.variation_map: Dict[str, str] = {}  # variant -> canonical
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for headword detection."""
        # Pattern for article starts: HEADWORD, followed by definition
        self.article_pattern = re.compile(
            r'(?:^|\n)([A-Z][A-Z\s\-\']+),\s+[a-z]',
            re.MULTILINE
        )

    def find_in_text(self, text: str, max_distance: int = 2) -> List[FuzzyMatch]:
        """Find expected headwords in text using multiple strategies."""
        matches = []
        found_positions = set()

        # Strategy 1: Exact matches (ALL CAPS)
        for match in self.article_pattern.finditer(text):
            headword = match.group(1).strip()
            normalized = headword.upper().replace('  ', ' ')
            if normalized in self.headwords:
                if match.start() not in found_positions:
                    matches.append(FuzzyMatch(
                        headword=normalized,
                        matched_text=headword,
                        position=match.start(),
                        confidence=1.0,
                        match_type="exact"
                    ))
                    found_positions.add(match.start())

        # Strategy 2: Known variations
        for variant, canonical in self.variation_map.items():
            pattern = re.compile(rf'(?:^|\n)({re.escape(variant)}),\s+[a-z]', re.MULTILINE | re.IGNORECASE)
            for match in pattern.finditer(text):
                if match.start() not in found_positions:
                    matches.append(FuzzyMatch(
                        headword=canonical,
                        matched_text=match.group(1),
                        position=match.start(),
                        confidence=0.95,
                        match_type="variation"
                    ))
                    found_positions.add(match.start())

        # Strategy 3: Fuzzy matching for missing headwords
        if HAVE_RAPIDFUZZ:
            found_headwords = {m.headword for m in matches}
            missing = self.headwords - found_headwords

            # Find candidate positions in text
            candidate_pattern = re.compile(r'(?:^|\n)([A-Z][A-Za-z\s\-\']+),', re.MULTILINE)
            for match in candidate_pattern.finditer(text):
                candidate = match.group(1).strip().upper()
                if match.start() in found_positions:
                    continue

                # Check fuzzy match against missing headwords
                result = process.extractOne(
                    candidate,
                    missing,
                    scorer=fuzz.ratio,
                    score_cutoff=85
                )
                if result:
                    best_match, score, _ = result
                    matches.append(FuzzyMatch(
                        headword=best_match,
                        matched_text=match.group(1),
                        position=match.start(),
                        confidence=score / 100.0,
                        match_type="fuzzy"
                    ))
                    found_positions.add(match.start())

        return sorted(matches, key=lambda m: m.position)

    def learn_variations(self, ocr_headwords: List[str], canonical_headwords: Set[str]):
        """Learn OCR variations by comparing parsed headwords to expected."""
        if not HAVE_RAPIDFUZZ:
            return

        for ocr_hw in ocr_headwords:
            normalized = ocr_hw.upper().strip()
            if normalized not in canonical_headwords:
                # Find closest canonical match
                result = process.extractOne(
                    normalized,
                    canonical_headwords,
                    scorer=fuzz.ratio,
                    score_cutoff=80
                )
                if result:
                    canonical, score, _ = result
                    if score >= 80 and score < 100:
                        self.variation_map[normalized] = canonical
```

### Phase 3: LLM-Assisted Extractor

Create `encyclopedia_parser/llm_extractor.py`:

```python
from typing import Optional, Tuple, List
from dataclasses import dataclass
import os

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
    confidence: float
    start_position: int
    end_position: Optional[int] = None
    reasoning: str = ""

class LLMArticleExtractor:
    """Use LLM to classify ambiguous article boundaries."""

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        if HAVE_ANTHROPIC:
            self.client = anthropic.Anthropic()
        else:
            self.client = None

    def classify_candidate(
        self,
        text: str,
        headword: str,
        position: int,
        context_chars: int = 500
    ) -> ExtractionResult:
        """Determine if a headword at position is an article start."""

        if not self.client:
            # Fallback: use heuristics
            return self._heuristic_classify(text, headword, position)

        # Extract context around position
        start = max(0, position - context_chars)
        end = min(len(text), position + context_chars)
        context = text[start:end]
        relative_pos = position - start

        prompt = f"""Analyze this encyclopedia text to determine if "{headword}" at the marked position is the START of a new article entry, or just a mention within another article.

Context (position {relative_pos} marked with >>>):
{context[:relative_pos]}>>>{context[relative_pos:]}

Signs of article start:
- Headword appears at beginning of line or after blank line
- Followed by comma and lowercase definition text
- Preceded by end of previous article (period, blank line)
- May have cross-references like "See ASTRONOMY"

Signs of mere mention:
- Appears mid-sentence
- Part of cross-reference within another article
- Within a list or enumeration

Respond with exactly one of:
ARTICLE_START - This is the beginning of a new encyclopedia article
MENTION - This is just a mention within another article
UNCERTAIN - Cannot determine with confidence

Then briefly explain your reasoning."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()
            first_line = result_text.split('\n')[0].strip()

            is_start = first_line == "ARTICLE_START"
            confidence = 0.9 if first_line in ["ARTICLE_START", "MENTION"] else 0.5

            return ExtractionResult(
                headword=headword,
                is_article_start=is_start,
                confidence=confidence,
                start_position=position,
                reasoning=result_text
            )

        except Exception as e:
            return self._heuristic_classify(text, headword, position)

    def _heuristic_classify(
        self,
        text: str,
        headword: str,
        position: int
    ) -> ExtractionResult:
        """Fallback heuristic classification."""
        # Check if at line start
        line_start = text.rfind('\n', 0, position)
        chars_before = text[line_start+1:position].strip() if line_start >= 0 else text[:position].strip()

        # Check what follows
        after_pos = position + len(headword)
        chars_after = text[after_pos:after_pos+10] if after_pos < len(text) else ""

        # Heuristics
        at_line_start = len(chars_before) == 0
        followed_by_comma = chars_after.startswith(',')

        is_start = at_line_start and followed_by_comma
        confidence = 0.7 if is_start else 0.3

        return ExtractionResult(
            headword=headword,
            is_article_start=is_start,
            confidence=confidence,
            start_position=position,
            reasoning=f"Heuristic: line_start={at_line_start}, comma_after={followed_by_comma}"
        )
```

### Phase 4: Smart Parser Integration

Create `encyclopedia_parser/smart_parser.py`:

```python
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import json
import os

from .expected_articles import ExpectedArticleRegistry, ExpectedArticle
from .fuzzy_matcher import FuzzyMatcher, FuzzyMatch
from .llm_extractor import LLMArticleExtractor, ExtractionResult

@dataclass
class SmartParseResult:
    """Result of smart parsing."""
    headword: str
    start_position: int
    end_position: int
    text: str
    confidence: float
    match_type: str  # "regex", "fuzzy", "llm"
    page_start: int
    page_end: int

class SmartBritannicaParser:
    """Smart parser combining regex, fuzzy matching, and LLM extraction."""

    def __init__(
        self,
        edition_year: int,
        registry: Optional[ExpectedArticleRegistry] = None,
        use_llm: bool = False
    ):
        self.edition_year = edition_year
        self.registry = registry or ExpectedArticleRegistry()
        self.use_llm = use_llm

        # Initialize components
        expected = self.registry.get_expected_for_edition(edition_year)
        self.fuzzy_matcher = FuzzyMatcher(expected)
        self.llm_extractor = LLMArticleExtractor() if use_llm else None

    def parse(
        self,
        text: str,
        page_numbers: List[int],
        existing_articles: Optional[List[dict]] = None
    ) -> List[SmartParseResult]:
        """Parse text with smart multi-strategy approach."""

        results = []
        found_headwords: Set[str] = set()

        # Step 1: Use existing regex-parsed articles as baseline
        if existing_articles:
            for article in existing_articles:
                hw = article.get('headword', '').upper()
                found_headwords.add(hw)
                # These are already parsed, skip adding to results

        # Step 2: Find missing articles with fuzzy matching
        expected = self.registry.get_expected_for_edition(self.edition_year)
        missing = expected - found_headwords

        if missing:
            # Use fuzzy matcher to find missing headwords
            matches = self.fuzzy_matcher.find_in_text(text)

            for match in matches:
                if match.headword in missing and match.confidence >= 0.85:
                    # Found a missing article via fuzzy matching
                    end_pos = self._find_article_end(text, match.position)
                    page = self._get_page_at_position(match.position, text, page_numbers)

                    results.append(SmartParseResult(
                        headword=match.headword,
                        start_position=match.position,
                        end_position=end_pos,
                        text=text[match.position:end_pos],
                        confidence=match.confidence,
                        match_type=match.match_type,
                        page_start=page,
                        page_end=self._get_page_at_position(end_pos, text, page_numbers)
                    ))
                    found_headwords.add(match.headword)

        # Step 3: Use LLM for remaining uncertain cases
        if self.use_llm and self.llm_extractor:
            still_missing = expected - found_headwords
            for headword in list(still_missing)[:50]:  # Limit LLM calls
                # Search for headword mentions in text
                import re
                pattern = re.compile(rf'\b{re.escape(headword)}\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    result = self.llm_extractor.classify_candidate(
                        text, headword, match.start()
                    )
                    if result.is_article_start and result.confidence >= 0.8:
                        end_pos = self._find_article_end(text, match.start())
                        page = self._get_page_at_position(match.start(), text, page_numbers)

                        results.append(SmartParseResult(
                            headword=headword,
                            start_position=match.start(),
                            end_position=end_pos,
                            text=text[match.start():end_pos],
                            confidence=result.confidence,
                            match_type="llm",
                            page_start=page,
                            page_end=self._get_page_at_position(end_pos, text, page_numbers)
                        ))
                        found_headwords.add(headword)
                        break  # Found it, move to next headword

        return sorted(results, key=lambda r: r.start_position)

    def _find_article_end(self, text: str, start: int, max_length: int = 100000) -> int:
        """Find the end of an article (next article start or max length)."""
        import re
        pattern = re.compile(r'\n\n[A-Z][A-Z\s\-\']+,\s+[a-z]')
        search_text = text[start+100:start+max_length]
        match = pattern.search(search_text)
        if match:
            return start + 100 + match.start()
        return min(start + max_length, len(text))

    def _get_page_at_position(
        self,
        position: int,
        text: str,
        page_numbers: List[int]
    ) -> int:
        """Get the page number at a given character position."""
        # Simple heuristic: assume even distribution
        if not page_numbers:
            return 1
        chars_per_page = len(text) / len(page_numbers)
        page_index = min(int(position / chars_per_page), len(page_numbers) - 1)
        return page_numbers[page_index]

    def get_coverage_stats(
        self,
        found_headwords: Set[str]
    ) -> Dict:
        """Get coverage statistics for this edition."""
        expected = self.registry.get_expected_for_edition(self.edition_year)
        found = found_headwords & expected
        missing = expected - found_headwords

        return {
            "edition_year": self.edition_year,
            "expected_count": len(expected),
            "found_count": len(found),
            "missing_count": len(missing),
            "coverage_pct": len(found) / len(expected) * 100 if expected else 0,
            "missing_sample": list(missing)[:20]
        }
```

---

## Execution Order

1. **Install dependencies**: `pip install rapidfuzz anthropic`
2. **Create the 4 new files** in `encyclopedia_parser/`
3. **Update `encyclopedia_parser/__init__.py`** to export new classes
4. **Build registry** from 1842 index + existing parsed articles
5. **Test on 1842 vol 2** (known gap of ~300 articles)
6. **Measure coverage improvement**
7. **Expand to all editions**

---

## Session Prompt

Copy this prompt to start a new session:

```
I'm continuing work on the Encyclopedia Britannica smart parser. The plan is in SMART_PARSER_PLAN.md.

Current state:
- 8 editions (1771-1860) with OCR'd text
- Current regex parser misses ~30-40% of articles
- Title Case support added (improved 1842 vol 2 from 72% to 83%)
- 1842 index has 18,215 main entries we can use as ground truth
- Existing encyclopedia_parser package with models, chunkers, classifiers

The plan has 4 phases:
1. Expected Article Registry - build list of expected articles per edition
2. Fuzzy Matcher - find OCR spelling variations (using rapidfuzz)
3. LLM Extractor - classify ambiguous cases with Claude
4. Smart Parser - integrate all strategies

Please implement Phase 1 (Expected Article Registry) first by creating:
- encyclopedia_parser/expected_articles.py

Key data files:
- output_v2/index_1842.jsonl - 1842 index entries
- output_v2/articles_*.jsonl - already-parsed articles per edition

After implementing, test by loading the 1842 index and counting expected articles.
```

---

## Success Metrics

- **Coverage**: % of index entries found (target: 95%+)
- **Precision**: % of extracted articles that are real (target: 99%+)
- **New articles recovered**: Count of previously-missed articles
