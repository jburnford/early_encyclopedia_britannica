#!/usr/bin/env python3
"""
Build a source-linked article index for the Encyclopedia Britannica knowledge graph.

Creates article_index.jsonl with full provenance for each article:
- Deterministic article_id based on edition/volume/index
- Source file path and array index
- Page boundaries (start_page, end_page)
- Quality classification (green/yellow/red)
- Text length and headword

This index becomes the foundation for all downstream processing.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional


def normalize_headword(headword: str) -> str:
    """Normalize headword for comparison."""
    if not headword:
        return ""
    h = headword.lower().strip()
    h = re.sub(r'\s+', ' ', h)
    h = re.sub(r'[.,;:]+$', '', h)
    return h


def classify_headword_quality(headword: str, text_length: int) -> tuple[str, list[str]]:
    """
    Classify article quality based on headword and text patterns.

    Returns: (quality_flag, list_of_issues)
    - green: High confidence, ready for KG
    - yellow: Minor issues, usable with caveats
    - red: Definite parsing error, skip for now
    """
    issues = []

    if not headword:
        return "red", ["empty_headword"]

    upper_hw = headword.upper()
    words = upper_hw.split()

    # RED flags - definite parsing errors

    # Mid-sentence headwords (sentence fragments)
    sentence_starters = ['THE', 'THIS', 'THAT', 'WHEN', 'WHILE', 'WHICH', 'WHERE',
                        'HAVING', 'BEING', 'THESE', 'THOSE', 'WITH', 'FROM', 'INTO',
                        'ALTHOUGH', 'HOWEVER', 'ANOTHER', 'BOTH', 'EITHER', 'NEITHER',
                        'SUCH', 'SOME', 'ALL', 'ANY', 'EVERY', 'EACH', 'MOST']

    if len(words) > 2 and words[0] in sentence_starters:
        # Exception: "THE [NOUN]" titles are valid (e.g., "THE HAGUE")
        if not (words[0] == 'THE' and len(words) <= 4):
            issues.append("mid_sentence_headword")
            return "red", issues

    # Ends with preposition/article (mid-sentence break)
    ending_words = ['THE', 'A', 'AN', 'OF', 'TO', 'BY', 'IN', 'ON', 'AT',
                   'FOR', 'WITH', 'FROM', 'AS', 'IS', 'ARE', 'WAS', 'WERE']
    if words and words[-1] in ending_words:
        issues.append("ends_with_function_word")
        return "red", issues

    # Very long headword (likely sentence fragment)
    if len(headword) > 60:
        issues.append("headword_too_long")
        return "red", issues

    # Volume/section markers
    if 'END OF' in upper_hw or 'VOLUME' in upper_hw or 'FINIS' in upper_hw:
        issues.append("volume_marker")
        return "red", issues

    # YELLOW flags - potential issues but usable

    # Moderately long headword
    if len(headword) > 40:
        issues.append("long_headword")

    # Very short text (might be cross-reference only)
    if text_length < 50:
        issues.append("very_short_text")

    # Very long text (might be merged articles)
    if text_length > 100000:
        issues.append("possibly_merged")

    # Headword starts with lowercase
    if headword and headword[0].islower():
        issues.append("lowercase_start")

    # Contains unusual characters
    if re.search(r'[<>{}|\\]', headword):
        issues.append("unusual_characters")

    # Determine final classification
    if issues:
        # Check if any issue is severe enough for yellow
        yellow_issues = ['long_headword', 'very_short_text', 'possibly_merged',
                        'lowercase_start', 'unusual_characters']
        if any(i in yellow_issues for i in issues):
            return "yellow", issues

    return "green", issues


def extract_volume_number(filename: str) -> int:
    """Extract volume number from filename like 'vol5.json'."""
    match = re.search(r'vol(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def build_article_id(edition: int, volume: str, array_index: int, headword: str) -> str:
    """
    Build deterministic article ID.

    Format: enc_{edition}_{volume}_idx{index}_{normalized_headword_prefix}
    """
    # Normalize headword for ID (alphanumeric only, truncated)
    hw_clean = re.sub(r'[^a-zA-Z0-9]', '_', headword)[:30].strip('_').upper()
    if not hw_clean:
        hw_clean = "UNKNOWN"

    return f"enc_{edition}_{volume}_idx{array_index:04d}_{hw_clean}"


def load_edition(docs_path: Path, edition: int) -> list[dict]:
    """
    Load all articles from an edition with full source tracking.

    Character offsets are computed per-volume in page order to enable
    gap/overlap detection in the text stream.
    """
    edition_path = docs_path / str(edition) / "data"
    articles = []

    if not edition_path.exists():
        print(f"  Warning: Edition path not found: {edition_path}")
        return articles

    # Get main vol*.json files (exclude _original, _corrected)
    json_files = sorted(edition_path.glob("vol*.json"))
    main_files = [f for f in json_files
                  if '_original' not in f.name and '_corrected' not in f.name]

    for json_file in main_files:
        volume = json_file.stem  # e.g., "vol5"
        relative_path = f"docs/{edition}/data/{json_file.name}"

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # First pass: collect all articles with their data
            volume_articles = []
            for array_index, article in enumerate(data):
                headword = article.get('h', '')
                text = article.get('t', '')
                text_length = len(text)
                start_page = article.get('sp')
                end_page = article.get('ep')

                # Classify quality
                quality_flag, issues = classify_headword_quality(headword, text_length)

                volume_articles.append({
                    "array_index": array_index,
                    "headword": headword,
                    "text_length": text_length,
                    "start_page": start_page,
                    "end_page": end_page,
                    "quality_flag": quality_flag,
                    "issues": issues
                })

            # Sort by (start_page, array_index) to get page order
            # array_index as tiebreaker for articles on same page
            sorted_by_page = sorted(
                volume_articles,
                key=lambda x: (x["start_page"] or 0, x["array_index"])
            )

            # Compute cumulative character offsets in page order
            char_offset = 0
            page_order_map = {}  # array_index -> (char_start, char_end, page_order)
            for page_order, art in enumerate(sorted_by_page):
                idx = art["array_index"]
                char_start = char_offset
                char_end = char_offset + art["text_length"]
                page_order_map[idx] = (char_start, char_end, page_order)
                char_offset = char_end

            volume_total_chars = char_offset

            # Second pass: build article records with character offsets
            for art in volume_articles:
                idx = art["array_index"]
                char_start, char_end, page_order = page_order_map[idx]

                article_record = {
                    "article_id": build_article_id(edition, volume, idx, art["headword"]),
                    "headword": art["headword"],
                    "headword_normalized": normalize_headword(art["headword"]),
                    "quality_flag": art["quality_flag"],
                    "issues": art["issues"],
                    "source": {
                        "file": relative_path,
                        "edition": edition,
                        "volume": volume,
                        "volume_number": extract_volume_number(volume),
                        "array_index": idx,
                        "page_order": page_order,  # Position when sorted by page
                        "volume_total_chars": volume_total_chars
                    },
                    "boundaries": {
                        "start_page": art["start_page"],
                        "end_page": art["end_page"],
                        "text_length": art["text_length"],
                        "char_start": char_start,  # Cumulative offset in page order
                        "char_end": char_end
                    }
                }

                articles.append(article_record)

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    return articles


def main():
    docs_path = Path(__file__).parent.parent / "docs"
    output_path = Path(__file__).parent.parent / "article_index.jsonl"

    # Find available editions
    available_years = []
    for year_dir in sorted(docs_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            data_dir = year_dir / "data"
            if data_dir.exists() and list(data_dir.glob("vol*.json")):
                available_years.append(int(year_dir.name))

    print(f"Building article index for editions: {available_years}")
    print("=" * 70)

    # Collect all articles
    all_articles = []
    stats = {
        "by_edition": {},
        "by_quality": defaultdict(int),
        "by_issue": defaultdict(int)
    }

    for year in available_years:
        print(f"\nLoading {year} edition...")
        articles = load_edition(docs_path, year)

        # Collect stats
        edition_stats = {"total": len(articles), "green": 0, "yellow": 0, "red": 0}
        for a in articles:
            edition_stats[a["quality_flag"]] += 1
            stats["by_quality"][a["quality_flag"]] += 1
            for issue in a["issues"]:
                stats["by_issue"][issue] += 1

        stats["by_edition"][year] = edition_stats
        all_articles.extend(articles)

        print(f"  {len(articles):,} articles: "
              f"{edition_stats['green']:,} green, "
              f"{edition_stats['yellow']:,} yellow, "
              f"{edition_stats['red']:,} red")

    # Write JSONL output
    print(f"\nWriting {len(all_articles):,} articles to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in all_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    # Write summary stats
    stats_path = Path(__file__).parent.parent / "article_index_stats.json"
    summary = {
        "total_articles": len(all_articles),
        "editions": available_years,
        "by_edition": stats["by_edition"],
        "by_quality": dict(stats["by_quality"]),
        "by_issue": dict(sorted(stats["by_issue"].items(), key=lambda x: -x[1]))
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("ARTICLE INDEX SUMMARY")
    print("=" * 70)
    print(f"\nTotal articles: {len(all_articles):,}")
    print(f"\nBy quality flag:")
    for flag in ["green", "yellow", "red"]:
        count = stats["by_quality"][flag]
        pct = 100 * count / len(all_articles) if all_articles else 0
        print(f"  {flag:8s}: {count:>7,} ({pct:5.1f}%)")

    print(f"\nTop issues detected:")
    for issue, count in sorted(stats["by_issue"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {issue}: {count:,}")

    print(f"\nOutput files:")
    print(f"  {output_path}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
