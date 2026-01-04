"""
Expected Article Registry - Phase 1 of Smart Parser

This module builds a registry of expected articles from multiple sources:
1. The 1842 General Index (18,000+ entries with volume/page references)
2. Already-parsed articles from each edition
3. Cross-edition inference (articles in multiple editions likely in others)

Usage:
    from encyclopedia_parser.expected_articles import ExpectedArticleRegistry

    registry = ExpectedArticleRegistry()
    registry.load_from_index("output_v2/index_1842.jsonl", 1842)
    registry.load_all_parsed_articles("output_v2")

    # Get expected headwords for a specific edition
    expected = registry.get_expected_for_edition(1842)
    print(f"Expected {len(expected)} articles in 1842 edition")
"""

from dataclasses import dataclass, field
from typing import Set, Dict, List, Optional, Tuple
from pathlib import Path
import json
import os
import re


@dataclass
class ExpectedArticle:
    """An article we expect to find in one or more editions."""
    headword: str  # Normalized uppercase
    editions_expected: Set[int] = field(default_factory=set)
    editions_found: Set[int] = field(default_factory=set)
    confidence: float = 1.0  # How certain we are this is a real article
    source: str = "unknown"  # "index", "parsed", "cross_edition", "cross_ref"
    known_variations: List[str] = field(default_factory=list)
    volume_refs: Dict[int, List[int]] = field(default_factory=dict)  # edition -> [vol references]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "headword": self.headword,
            "editions_expected": sorted(self.editions_expected),
            "editions_found": sorted(self.editions_found),
            "confidence": self.confidence,
            "source": self.source,
            "known_variations": self.known_variations,
            "volume_refs": {str(k): v for k, v in self.volume_refs.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpectedArticle":
        """Create from dictionary."""
        return cls(
            headword=data["headword"],
            editions_expected=set(data.get("editions_expected", [])),
            editions_found=set(data.get("editions_found", [])),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            known_variations=data.get("known_variations", []),
            volume_refs={int(k): v for k, v in data.get("volume_refs", {}).items()}
        )


class ExpectedArticleRegistry:
    """Registry of expected articles built from index + cross-edition data."""

    # Known edition years
    EDITIONS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

    # Edition families for cross-edition inference
    # Later editions often contain articles from earlier related editions
    EDITION_FAMILIES = {
        # If article in 1815 AND 1823, likely in 1810 too (same "family")
        1810: {1815, 1823},
        1797: {1810, 1815},
        1778: {1797, 1810},
        # 7th edition (1842) may share with Supplement-enhanced editions
        1842: {1823, 1860},
    }

    def __init__(self):
        self.articles: Dict[str, ExpectedArticle] = {}
        self._stats = {
            "index_entries": 0,
            "parsed_articles": {},
            "inferred_articles": 0
        }

    @staticmethod
    def normalize_headword(headword: str) -> str:
        """Normalize a headword for consistent matching."""
        if not headword:
            return ""
        # Uppercase, strip whitespace, normalize multiple spaces
        normalized = headword.upper().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove trailing punctuation except for abbreviations
        normalized = re.sub(r'[,;:]+$', '', normalized)
        return normalized

    def load_from_index(self, index_path: str, edition_year: int = 1842) -> int:
        """
        Load headwords from 1842 index JSONL.

        The 1842 index contains entries with:
        - entry_type: "main" (articles), "sub" (sub-entries), "cross_ref" (see also)
        - references: [{vol: int, pages: [int]}] - volume and page references

        Returns count of main entries loaded.
        """
        count = 0
        with open(index_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get('entry_type', '')
                term = entry.get('term', '').strip()
                references = entry.get('references', [])

                # Only load main entries that have page references
                # (these are actual articles, not cross-references or sub-entries)
                if entry_type != 'main':
                    continue

                if not term or len(term) <= 1:
                    continue

                normalized = self.normalize_headword(term)
                if not normalized:
                    continue

                # Extract volume references
                vol_refs = []
                for ref in references:
                    if ref.get('vol'):
                        vol_refs.append(ref['vol'])

                if normalized not in self.articles:
                    self.articles[normalized] = ExpectedArticle(
                        headword=normalized,
                        editions_expected={edition_year},
                        confidence=1.0,
                        source="index"
                    )
                    if vol_refs:
                        self.articles[normalized].volume_refs[edition_year] = list(set(vol_refs))
                else:
                    self.articles[normalized].editions_expected.add(edition_year)
                    if vol_refs:
                        existing = self.articles[normalized].volume_refs.get(edition_year, [])
                        self.articles[normalized].volume_refs[edition_year] = list(set(existing + vol_refs))

                count += 1

        self._stats["index_entries"] = count
        return count

    def load_from_parsed_articles(
        self,
        articles_path: str,
        edition_year: int,
        add_to_expected: bool = False
    ) -> int:
        """
        Load headwords from already-parsed articles JSONL.

        These are articles that the current regex parser successfully extracted.

        Args:
            articles_path: Path to articles_*.jsonl file
            edition_year: The edition year
            add_to_expected: If True, also add to editions_expected (use for
                             editions without an index). Default False.

        Returns count of articles loaded.
        """
        if not os.path.exists(articles_path):
            return 0

        count = 0
        with open(articles_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    continue

                headword = article.get('headword', '').strip()
                if not headword or len(headword) <= 1:
                    continue

                normalized = self.normalize_headword(headword)
                if not normalized:
                    continue

                if normalized not in self.articles:
                    self.articles[normalized] = ExpectedArticle(
                        headword=normalized,
                        editions_expected={edition_year} if add_to_expected else set(),
                        editions_found={edition_year},
                        confidence=0.95,  # High but not 100% - could be parser error
                        source="parsed"
                    )
                else:
                    if add_to_expected:
                        self.articles[normalized].editions_expected.add(edition_year)
                    self.articles[normalized].editions_found.add(edition_year)
                    # If from multiple sources, boost confidence
                    if self.articles[normalized].source == "index":
                        self.articles[normalized].confidence = 1.0

                count += 1

        self._stats["parsed_articles"][edition_year] = count
        return count

    def load_all_parsed_articles(self, output_dir: str) -> Dict[int, int]:
        """
        Load parsed articles from all edition files in output directory.

        Expects files named: articles_{year}.jsonl

        Returns dict of {year: count} for each edition loaded.
        """
        results = {}
        output_path = Path(output_dir)

        for edition_year in self.EDITIONS:
            articles_file = output_path / f"articles_{edition_year}.jsonl"
            if articles_file.exists():
                count = self.load_from_parsed_articles(str(articles_file), edition_year)
                results[edition_year] = count

        return results

    def infer_cross_edition(self) -> int:
        """
        Infer expected articles based on cross-edition patterns.

        If an article appears in multiple related editions, it's likely
        to also exist in other editions in the same "family".

        Returns count of inferred additions.
        """
        inferred = 0

        for target_year, source_years in self.EDITION_FAMILIES.items():
            for headword, article in self.articles.items():
                # Check if article found in all source editions
                if source_years.issubset(article.editions_found):
                    if target_year not in article.editions_expected:
                        article.editions_expected.add(target_year)
                        # Lower confidence for inferred articles
                        if article.source != "index":
                            article.confidence = min(article.confidence, 0.7)
                        article.source = f"{article.source}+inferred"
                        inferred += 1

        self._stats["inferred_articles"] = inferred
        return inferred

    def get_expected_for_edition(self, year: int) -> Set[str]:
        """Get all headwords expected for a given edition."""
        return {hw for hw, a in self.articles.items() if year in a.editions_expected}

    def get_found_for_edition(self, year: int) -> Set[str]:
        """Get all headwords found (parsed) for a given edition."""
        return {hw for hw, a in self.articles.items() if year in a.editions_found}

    def get_missing_for_edition(self, year: int) -> Set[str]:
        """Get headwords expected but not found for a given edition."""
        expected = self.get_expected_for_edition(year)
        found = self.get_found_for_edition(year)
        return expected - found

    def add_variation(self, canonical: str, variant: str):
        """Add a known OCR variation for a headword."""
        normalized_canonical = self.normalize_headword(canonical)
        normalized_variant = self.normalize_headword(variant)

        if normalized_canonical in self.articles:
            if normalized_variant not in self.articles[normalized_canonical].known_variations:
                self.articles[normalized_canonical].known_variations.append(normalized_variant)

    def get_article(self, headword: str) -> Optional[ExpectedArticle]:
        """Get article info by headword (normalized)."""
        normalized = self.normalize_headword(headword)
        return self.articles.get(normalized)

    def get_coverage_stats(self, year: int) -> Dict:
        """
        Get coverage statistics for a specific edition.

        Returns dict with expected, found, missing counts and percentages.
        """
        expected = self.get_expected_for_edition(year)
        found = self.get_found_for_edition(year)
        missing = expected - found
        extra = found - expected  # Found but not expected (possible false positives)

        return {
            "edition_year": year,
            "expected_count": len(expected),
            "found_count": len(found),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "coverage_pct": round(len(found & expected) / len(expected) * 100, 2) if expected else 0,
            "missing_sample": sorted(list(missing))[:25],
            "extra_sample": sorted(list(extra))[:10]
        }

    def get_all_stats(self) -> Dict:
        """Get summary statistics for all editions."""
        stats = {
            "total_unique_headwords": len(self.articles),
            "index_entries_loaded": self._stats["index_entries"],
            "parsed_by_edition": self._stats["parsed_articles"],
            "inferred_count": self._stats["inferred_articles"],
            "editions": {}
        }

        for year in self.EDITIONS:
            expected = self.get_expected_for_edition(year)
            found = self.get_found_for_edition(year)
            if expected or found:
                stats["editions"][year] = {
                    "expected": len(expected),
                    "found": len(found),
                    "coverage_pct": round(len(found & expected) / len(expected) * 100, 2) if expected else 0
                }

        return stats

    def save(self, filepath: str):
        """Save registry to JSON file."""
        data = {
            "version": "1.0",
            "stats": self._stats,
            "articles": {hw: a.to_dict() for hw, a in self.articles.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load registry from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._stats = data.get("stats", {})
        self.articles = {
            hw: ExpectedArticle.from_dict(a)
            for hw, a in data.get("articles", {}).items()
        }

    def __len__(self) -> int:
        return len(self.articles)

    def __contains__(self, headword: str) -> bool:
        return self.normalize_headword(headword) in self.articles


def build_registry_from_defaults(
    index_path: str = "output_v2/index_1842.jsonl",
    output_dir: str = "output_v2",
    infer_cross_edition: bool = True
) -> ExpectedArticleRegistry:
    """
    Convenience function to build a fully populated registry.

    Args:
        index_path: Path to 1842 index JSONL
        output_dir: Directory containing articles_*.jsonl files
        infer_cross_edition: Whether to infer cross-edition expectations

    Returns:
        Fully populated ExpectedArticleRegistry
    """
    registry = ExpectedArticleRegistry()

    # Load 1842 index (ground truth)
    if os.path.exists(index_path):
        registry.load_from_index(index_path, 1842)

    # Load all parsed articles
    registry.load_all_parsed_articles(output_dir)

    # Infer cross-edition expectations
    if infer_cross_edition:
        registry.infer_cross_edition()

    return registry


if __name__ == "__main__":
    # Test the registry
    import sys

    # Use provided paths or defaults
    index_path = sys.argv[1] if len(sys.argv) > 1 else "output_v2/index_1842.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_v2"

    print("Building Expected Article Registry...")
    print("=" * 60)

    registry = ExpectedArticleRegistry()

    # Load 1842 index
    if os.path.exists(index_path):
        count = registry.load_from_index(index_path, 1842)
        print(f"Loaded {count:,} main entries from 1842 index")
        print(f"  -> {len(registry):,} unique normalized headwords")
    else:
        print(f"Index file not found: {index_path}")

    # Load parsed articles
    print()
    parsed_counts = registry.load_all_parsed_articles(output_dir)
    for year, count in sorted(parsed_counts.items()):
        print(f"Loaded {count:,} parsed articles from {year} edition")

    print()
    print(f"Total unique headwords in registry: {len(registry):,}")

    # Show 1842 index-only coverage (the key metric)
    print("\n" + "=" * 60)
    print("1842 INDEX COVERAGE (before cross-edition inference)")
    print("=" * 60)

    expected_1842 = registry.get_expected_for_edition(1842)
    found_1842 = registry.get_found_for_edition(1842)
    overlap_1842 = expected_1842 & found_1842
    missing_1842 = expected_1842 - found_1842
    extra_1842 = found_1842 - expected_1842

    print(f"Index headwords: {len(expected_1842):,}")
    print(f"Parsed headwords: {len(found_1842):,}")
    print(f"Successfully matched: {len(overlap_1842):,} ({100*len(overlap_1842)/len(expected_1842):.1f}%)")
    print(f"Missing (parser gap): {len(missing_1842):,} ({100*len(missing_1842)/len(expected_1842):.1f}%)")
    print(f"Extra (not in index): {len(extra_1842):,}")

    print("\nSample MISSING articles (targets for smart parser):")
    for hw in sorted(list(missing_1842))[:15]:
        print(f"  - {hw}")

    # Infer cross-edition
    print("\n" + "=" * 60)
    print("CROSS-EDITION INFERENCE")
    print("=" * 60)
    inferred = registry.infer_cross_edition()
    print(f"Inferred {inferred:,} additional cross-edition expectations")

    # Summary after inference
    print("\n" + "=" * 60)
    print("FINAL COVERAGE BY EDITION")
    print("=" * 60)
    print(f"{'Edition':<8} {'Expected':>10} {'Found':>10} {'Coverage':>10}")
    print("-" * 40)

    for year in registry.EDITIONS:
        expected = registry.get_expected_for_edition(year)
        found = registry.get_found_for_edition(year)
        overlap = expected & found
        if expected:
            pct = 100 * len(overlap) / len(expected)
            print(f"{year:<8} {len(expected):>10,} {len(found):>10,} {pct:>9.1f}%")
        elif found:
            print(f"{year:<8} {'n/a':>10} {len(found):>10,} {'n/a':>10}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"The smart parser needs to recover ~{len(missing_1842):,} missing articles")
    print(f"from the 1842 index. These are likely:")
    print(f"  - Short geographic entries (AA, AALDEN, etc.)")
    print(f"  - OCR variations (AALSMEEB = AALSMEER?)")
    print(f"  - Title Case entries missed by ALL CAPS pattern")
