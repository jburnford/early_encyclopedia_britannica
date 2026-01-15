#!/usr/bin/env python3
"""
Rebuild article JSON using Gemini's corrected article boundaries.

Takes Gemini output (article boundaries) and raw OCR to produce corrected
article JSON with proper content extraction.

Usage:
    python3 rebuild_from_gemini.py --edition 1771 --volume vol2 --dry-run
    python3 rebuild_from_gemini.py --edition 1771 --all
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from copy import deepcopy


class RawOCRLoader:
    """Load raw OCR text with page number mappings."""

    OCR_FILE_MAP = {
        ('1771', 1): 'ocr_results/1771_britannica_1st/output_114aa9270857798ee869db8d06996ca185bd7105.jsonl',
        ('1771', 2): 'ocr_results/1771_britannica_1st/output_1718f3a66eab1422763966bf5470b3b3906faa08.jsonl',
        ('1771', 3): 'ocr_results/1771_britannica_1st/output_1455a43ee23170019c3b9c5052be80106e7aaf8e.jsonl',
    }

    def __init__(self):
        self.cache = {}

    def load_volume(self, edition: str, volume: int) -> Dict:
        """Load OCR data for a volume."""
        cache_key = (edition, volume)
        if cache_key in self.cache:
            return self.cache[cache_key]

        filepath = self.OCR_FILE_MAP.get(cache_key)
        if not filepath:
            raise FileNotFoundError(f"No OCR mapping for {edition} vol{volume}")

        with open(filepath) as f:
            content = f.read()

        if content.strip().startswith('['):
            data = json.loads(content)
            entry = data[0] if isinstance(data, list) else data
        else:
            entry = json.loads(content.split('\n')[0])

        text = entry.get('text', '')
        page_numbers = entry.get('attributes', {}).get('pdf_page_numbers', [])
        page_map = [(p[0], p[1], p[2]) for p in page_numbers]

        result = {
            'text': text,
            'page_map': page_map,
        }

        self.cache[cache_key] = result
        return result

    def get_char_range_for_pages(self, edition: str, volume: int,
                                  start_page: int, end_page: int) -> Tuple[int, int]:
        """Get character range for a page range."""
        data = self.load_volume(edition, volume)
        page_map = data['page_map']

        start_char = None
        end_char = None

        for char_start, char_end, page_num in page_map:
            if page_num == start_page:
                start_char = char_start
            if page_num == end_page:
                end_char = char_end
            if start_page <= page_num <= end_page:
                if start_char is None:
                    start_char = char_start
                end_char = char_end

        return start_char or 0, end_char or 0

    def get_text_for_pages(self, edition: str, volume: int,
                           start_page: int, end_page: int) -> str:
        """Get raw OCR text for a page range."""
        data = self.load_volume(edition, volume)
        start_char, end_char = self.get_char_range_for_pages(
            edition, volume, start_page, end_page
        )
        return data['text'][start_char:end_char]


def normalize_for_search(title: str) -> str:
    """Normalize title for fuzzy OCR matching."""
    # Replace special characters with ASCII equivalents
    replacements = {
        'Æ': 'AE', 'æ': 'ae', 'Œ': 'OE', 'œ': 'oe',
        'Ä': 'A', 'Ö': 'O', 'Ü': 'U', 'ä': 'a', 'ö': 'o', 'ü': 'u',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'é': 'e', 'è': 'e', 'ê': 'e',
        'À': 'A', 'Â': 'A', 'à': 'a', 'â': 'a',
        'Ç': 'C', 'ç': 'c', 'Ñ': 'N', 'ñ': 'n',
    }
    result = title
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def extract_article_content(raw_text: str, title: str,
                           next_title: Optional[str] = None) -> str:
    """
    Extract article content from raw OCR text.

    Finds the title in the text and extracts until the next title or end.
    """
    # Normalize title for search
    search_title = normalize_for_search(title)
    title_pattern = re.escape(search_title)

    # Try exact match first
    match = re.search(rf'\b{title_pattern}\b', raw_text, re.IGNORECASE)

    if not match:
        # Try with common OCR error patterns
        fuzzy_pattern = title_pattern
        fuzzy_pattern = fuzzy_pattern.replace('I', '[IL1]')
        fuzzy_pattern = fuzzy_pattern.replace('O', '[O0Q]')
        fuzzy_pattern = fuzzy_pattern.replace('S', '[S5]')
        fuzzy_pattern = fuzzy_pattern.replace('G', '[GC]')
        match = re.search(rf'\b{fuzzy_pattern}\b', raw_text, re.IGNORECASE)

    if not match:
        # Try prefix match (first 6+ chars)
        if len(search_title) >= 6:
            prefix = re.escape(search_title[:6])
            match = re.search(rf'\b{prefix}[A-Z]*\b', raw_text, re.IGNORECASE)

    if not match:
        return ""

    start_pos = match.start()

    # Find end position (next title or end of text)
    if next_title:
        next_pattern = re.escape(next_title)
        next_match = re.search(rf'\b{next_pattern}\b', raw_text[start_pos + len(title):], re.IGNORECASE)
        if next_match:
            end_pos = start_pos + len(title) + next_match.start()
        else:
            end_pos = len(raw_text)
    else:
        end_pos = len(raw_text)

    content = raw_text[start_pos:end_pos].strip()
    return content


def rebuild_volume(edition: str, volume: str,
                   gemini_file: str, dry_run: bool = False) -> Dict:
    """
    Rebuild a volume's article JSON using Gemini corrections.

    Returns statistics about the rebuild.
    """
    vol_num = int(volume.replace('vol', ''))

    # Load original articles
    original_path = Path('docs') / edition / 'data' / f'{volume}.json'
    with open(original_path) as f:
        original_articles = json.load(f)

    # Load Gemini corrections
    with open(gemini_file) as f:
        gemini_data = json.load(f)

    # Load OCR
    ocr_loader = RawOCRLoader()

    stats = {
        'original_count': len(original_articles),
        'ranges_processed': 0,
        'articles_removed': 0,
        'articles_added': 0,
        'articles_unchanged': 0,
    }

    # Build a map of which pages have been corrected
    corrected_pages = set()
    corrections_by_range = {}

    for item in gemini_data:
        range_info = item['range']
        start_page = range_info['start_page']
        end_page = range_info['end_page']

        for p in range(start_page, end_page + 1):
            corrected_pages.add(p)

        corrections_by_range[(start_page, end_page)] = item['result'].get('articles', [])
        stats['ranges_processed'] += 1

    # Build new article list
    new_articles = []

    # Keep articles that don't overlap with corrected ranges
    for art in original_articles:
        sp = art.get('sp')
        ep = art.get('ep') or sp

        if sp is None:
            new_articles.append(art)
            stats['articles_unchanged'] += 1
            continue

        # Check if this article overlaps with any corrected range
        overlaps = False
        for p in range(sp, ep + 1):
            if p in corrected_pages:
                overlaps = True
                break

        if not overlaps:
            new_articles.append(art)
            stats['articles_unchanged'] += 1
        else:
            stats['articles_removed'] += 1

    # Add corrected articles
    for (start_page, end_page), gemini_articles in corrections_by_range.items():
        # Get raw OCR for this range
        raw_text = ocr_loader.get_text_for_pages(edition, vol_num, start_page, end_page)

        # Sort articles by page
        sorted_articles = sorted(gemini_articles, key=lambda x: (x.get('sp', 0), x.get('title', '')))

        for i,gart in enumerate(sorted_articles):
            title = gart.get('title', '')
            sp = gart.get('sp')
            ep = gart.get('ep') or sp

            # Get next title for content extraction boundary
            next_title = sorted_articles[i + 1]['title'] if i + 1 < len(sorted_articles) else None

            # Extract content
            content = extract_article_content(raw_text, title, next_title)

            if content:
                new_articles.append({
                    'h': title,
                    't': content,
                    'sp': sp,
                    'ep': ep,
                })
                stats['articles_added'] += 1

    # Sort by title (alphabetically) to match original format
    new_articles.sort(key=lambda x: x.get('h', '').upper())

    stats['new_count'] = len(new_articles)

    if not dry_run:
        # Write output
        output_path = Path('docs') / edition / 'data' / f'{volume}_corrected.json'
        with open(output_path, 'w') as f:
            json.dump(new_articles, f, indent=2)
        stats['output_file'] = str(output_path)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rebuild articles from Gemini corrections")
    parser.add_argument("--edition", required=True, help="Edition year")
    parser.add_argument("--volume", help="Specific volume (e.g., vol2)")
    parser.add_argument("--all", action="store_true", help="Process all volumes")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    args = parser.parse_args()

    volumes = []
    if args.volume:
        volumes = [args.volume]
    elif args.all:
        # Find all gemini output files for this edition
        for f in Path('.').glob(f'gemini_{args.edition}_vol*.json'):
            vol = f.stem.replace(f'gemini_{args.edition}_', '')
            volumes.append(vol)

    if not volumes:
        print("No volumes specified. Use --volume or --all")
        return

    print(f"Rebuilding {args.edition} edition: {', '.join(volumes)}")
    print(f"Dry run: {args.dry_run}\n")

    for vol in sorted(volumes):
        gemini_file = f'gemini_{args.edition}_{vol}.json'
        if not Path(gemini_file).exists():
            print(f"  {vol}: No Gemini file found ({gemini_file})")
            continue

        print(f"  Processing {vol}...")
        stats = rebuild_volume(args.edition, vol, gemini_file, args.dry_run)

        print(f"    Original: {stats['original_count']} articles")
        print(f"    Ranges processed: {stats['ranges_processed']}")
        print(f"    Articles removed: {stats['articles_removed']}")
        print(f"    Articles added: {stats['articles_added']}")
        print(f"    Articles unchanged: {stats['articles_unchanged']}")
        print(f"    New total: {stats['new_count']} articles")
        if 'output_file' in stats:
            print(f"    Output: {stats['output_file']}")
        print()


if __name__ == "__main__":
    main()
