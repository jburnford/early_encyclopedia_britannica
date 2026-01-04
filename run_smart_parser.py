#!/usr/bin/env python3
"""
Run Smart Parser on ALL 8 editions and merge recovered articles.

This script:
1. Loads the expected article registry from the 1842 index
2. Runs the smart parser on all available OCR files (JSONL, MD, JSON)
3. Merges recovered articles with existing parsed articles
4. Outputs updated article files ready for site generation
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from encyclopedia_parser import (
    ExpectedArticleRegistry,
    SmartBritannicaParser,
    FuzzyMatcher,
)


# Configuration - ALL 8 editions
OCR_SOURCES = {
    1771: {
        "type": "jsonl_dir",
        "path": "ocr_results/1771_britannica_1st",
    },
    1778: {
        "type": "jsonl_dir",
        "path": "ocr_results/1778_britannica_2nd",
    },
    1797: {
        "type": "mixed",  # Has both .md and .jsonl files
        "path": "ocr_results/1797_britannica_3rd",
    },
    1810: {
        "type": "jsonl_dir",
        "path": "ocr_results/1810_britannica_4th",
    },
    1815: {
        "type": "json_dir",
        "path": "json",
        "filter": "FIFTH",  # Filter by edition marker in text
    },
    1823: {
        "type": "json_dir",
        "path": "json",
        "filter": "SIXTH",
    },
    1842: {
        "type": "batch_jsonl",  # Batch pipeline output with edition markers
        "path": "ocr_results/britannica_pipeline_batch",
        "filter": "SEVENTH",
    },
    1860: {
        "type": "batch_jsonl",  # Batch pipeline output with edition markers
        "path": "ocr_results/britannica_pipeline_batch",
        "filter": "EIGHTH",
    },
}

OUTPUT_DIR = Path("output_v2")
INDEX_PATH = "output_v2/index_1842.jsonl"


def load_jsonl_texts(jsonl_dir: str) -> list:
    """Load all texts from JSONL files in a directory."""
    texts = []
    jsonl_path = Path(jsonl_dir)

    if not jsonl_path.exists():
        return texts

    for jsonl_file in sorted(jsonl_path.glob("*.jsonl")):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    if len(text) > 1000:  # Skip tiny entries
                        source = entry.get('metadata', {}).get('Source-File', jsonl_file.name)
                        texts.append((text, source))
        except Exception as e:
            print(f"    Warning: Could not read {jsonl_file}: {e}")

    return texts


def load_md_texts(md_dir: str) -> list:
    """Load all texts from MD files in a directory."""
    texts = []
    md_path = Path(md_dir)

    if not md_path.exists():
        return texts

    for md_file in sorted(md_path.glob("*.md")):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                text = f.read()
                if len(text) > 1000:
                    texts.append((text, md_file.name))
        except Exception as e:
            print(f"    Warning: Could not read {md_file}: {e}")

    return texts


def load_json_texts(json_dir: str, edition_filter: str = None) -> list:
    """Load texts from JSON files (single-entry format), optionally filtered by edition."""
    texts = []
    json_path = Path(json_dir)

    if not json_path.exists():
        return texts

    for json_file in sorted(json_path.glob("*.json")):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list) and data:
                text = data[0].get('text', '')

                # Filter by edition if specified
                if edition_filter:
                    header = text[:1000].upper()
                    if edition_filter not in header:
                        continue

                if len(text) > 1000:
                    texts.append((text, json_file.name))
        except Exception as e:
            print(f"    Warning: Could not read {json_file}: {e}")

    return texts


def load_batch_jsonl_texts(jsonl_dir: str, edition_filter: str = None) -> list:
    """Load texts from batch pipeline JSONL files, filtered by edition marker."""
    texts = []
    jsonl_path = Path(jsonl_dir)

    if not jsonl_path.exists():
        return texts

    for jsonl_file in sorted(jsonl_path.glob("*.jsonl")):
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    text = entry.get('text', '')

                    # Filter by edition if specified
                    if edition_filter:
                        header = text[:1500].upper()
                        if edition_filter not in header:
                            continue

                    if len(text) > 1000:
                        texts.append((text, jsonl_file.name))
        except Exception as e:
            print(f"    Warning: Could not read {jsonl_file}: {e}")

    return texts


def process_edition(edition_year: int, config: dict, registry: ExpectedArticleRegistry) -> tuple:
    """Process a single edition and return recovered articles."""
    print(f"\n  [{edition_year}] Loading OCR texts...")

    texts = []
    source_type = config["type"]
    path = config["path"]

    if source_type == "jsonl_dir":
        texts = load_jsonl_texts(path)
    elif source_type == "json_dir":
        texts = load_json_texts(path, config.get("filter"))
    elif source_type == "mixed":
        texts = load_md_texts(path) + load_jsonl_texts(path)
    elif source_type == "batch_jsonl":
        texts = load_batch_jsonl_texts(path, config.get("filter"))

    if not texts:
        print(f"    No texts found for {edition_year}")
        return [], None

    print(f"    Found {len(texts)} text sources")

    # Initialize parser for this edition
    parser = SmartBritannicaParser(
        edition_year=edition_year,
        use_llm=False,
        fuzzy_threshold=88
    )
    parser.load_registry(INDEX_PATH, str(OUTPUT_DIR))

    # Process all texts
    all_results = []
    for text, source in texts:
        results = parser.parse(text, min_confidence=0.88)
        if results:
            print(f"    {Path(source).stem[:40]}: +{len(results)} articles")
            all_results.extend(results)

    return all_results, parser.get_stats()


def merge_articles(existing_path: str, recovered: list, edition_year: int) -> dict:
    """Merge recovered articles with existing ones."""
    existing = {}

    if os.path.exists(existing_path):
        with open(existing_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    article = json.loads(line)
                    hw = article.get('headword', '').upper().strip()
                    if hw:
                        existing[hw] = article
                except json.JSONDecodeError:
                    continue

    # Add recovered articles (only if not already present)
    added = 0
    for result in recovered:
        hw = result.headword.upper().strip()
        if hw and hw not in existing:
            article = {
                "article_id": f"{edition_year}_smart_{hw.replace(' ', '_')}",
                "headword": result.headword,
                "text": result.text,
                "source": "smart_parser",
                "match_type": result.match_type,
                "confidence": result.confidence,
                "ocr_matched": result.ocr_text_matched,
            }
            existing[hw] = article
            added += 1

    return existing, added


def save_articles(articles: dict, output_path: str):
    """Save articles to JSONL file."""
    sorted_articles = sorted(articles.values(), key=lambda x: x.get('headword', '').upper())

    with open(output_path, 'w', encoding='utf-8') as f:
        for article in sorted_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


def main():
    print("=" * 70)
    print("SMART PARSER - ALL 8 EDITIONS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Build registry
    print("1. LOADING EXPECTED ARTICLE REGISTRY")
    print("-" * 50)
    registry = ExpectedArticleRegistry()

    if os.path.exists(INDEX_PATH):
        count = registry.load_from_index(INDEX_PATH, 1842)
        print(f"   Loaded {count:,} entries from 1842 index")

    # Load all existing parsed articles
    for year in [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]:
        articles_path = OUTPUT_DIR / f"articles_{year}.jsonl"
        if articles_path.exists():
            registry.load_from_parsed_articles(str(articles_path), year)

    expected = registry.get_expected_for_edition(1842)
    print(f"   Total expected from index: {len(expected):,}")
    print()

    # Process each edition
    print("2. PROCESSING ALL EDITIONS")
    print("-" * 50)

    all_stats = {}
    editions_updated = []

    for edition_year, config in sorted(OCR_SOURCES.items()):
        recovered, stats = process_edition(edition_year, config, registry)

        if recovered:
            # Merge with existing articles
            articles_path = str(OUTPUT_DIR / f"articles_{edition_year}.jsonl")
            merged, added = merge_articles(articles_path, recovered, edition_year)

            if added > 0:
                # Save to temp file first
                temp_path = str(OUTPUT_DIR / f"articles_{edition_year}_smart.jsonl")
                save_articles(merged, temp_path)
                print(f"    TOTAL: {len(merged):,} articles (+{added:,} new)")
                editions_updated.append(edition_year)

        if stats:
            all_stats[edition_year] = stats

    # Summary
    print("\n" + "=" * 70)
    print("3. SUMMARY")
    print("=" * 70)

    total_new = 0
    for year in sorted(all_stats.keys()):
        stats = all_stats[year]
        print(f"  {year}: +{stats.total_recovered:,} recovered")
        total_new += stats.total_recovered

    print(f"\n  TOTAL NEW ARTICLES: {total_new:,}")
    print()

    # Update files
    if editions_updated:
        print("4. UPDATING ARTICLE FILES")
        print("-" * 50)

        for year in editions_updated:
            smart_path = OUTPUT_DIR / f"articles_{year}_smart.jsonl"
            orig_path = OUTPUT_DIR / f"articles_{year}.jsonl"
            backup_path = OUTPUT_DIR / f"articles_{year}_backup.jsonl"

            if smart_path.exists():
                if orig_path.exists():
                    os.rename(orig_path, backup_path)
                    print(f"   Backed up {orig_path.name}")
                os.rename(smart_path, orig_path)
                print(f"   Updated {orig_path.name}")

    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nNext: Run 'python3 generate_site_optimized.py' to regenerate the website")


if __name__ == "__main__":
    main()
