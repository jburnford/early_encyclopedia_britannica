#!/usr/bin/env python3
"""
Gemini Flash API integration for re-parsing encyclopedia page ranges.
Version 2: Uses RAW OCR text, not pre-parsed articles.

Sends raw OCR text for flagged page ranges to Gemini 3 Flash,
which returns correct article boundaries.

Usage:
    python3 gemini_reparse_v2.py --edition 1771 --volume vol2 --dry-run
    python3 gemini_reparse_v2.py --edition 1771 --volume vol2 --range 0
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from glob import glob

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: google-generativeai package not installed.")
    print("Install with: pip install google-generativeai")
    sys.exit(1)


# Configuration
GEMINI_MODEL = "gemini-3-flash-preview"
MAX_RETRIES = 3
RETRY_DELAY = 5


def configure_gemini():
    """Configure Gemini API with credentials."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        key_file = Path.home() / ".config" / "gemini_api_key.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()

    if not api_key:
        print("ERROR: No Gemini API key found.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


class RawOCRLoader:
    """Load raw OCR text with page number mappings from OCR result files."""

    def __init__(self, mapping_file: str = "ocr_edition_mapping.json"):
        self.cache = {}  # (edition, volume) -> {text, page_map}

        # Load comprehensive mapping from file
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with open(mapping_path) as f:
                self.edition_mapping = json.load(f)
        else:
            # Fallback to minimal hardcoded mapping
            self.edition_mapping = {
                '1771': {
                    '1': 'ocr_results/1771_britannica_1st/output_114aa9270857798ee869db8d06996ca185bd7105.jsonl',
                    '2': 'ocr_results/1771_britannica_1st/output_1718f3a66eab1422763966bf5470b3b3906faa08.jsonl',
                    '3': 'ocr_results/1771_britannica_1st/output_1455a43ee23170019c3b9c5052be80106e7aaf8e.jsonl',
                }
            }

    def _find_ocr_file(self, edition: str, volume: int) -> Optional[Path]:
        """Find the OCR result file for a given edition/volume."""
        if edition not in self.edition_mapping:
            return None

        vol_mapping = self.edition_mapping[edition]
        vol_key = str(volume)

        if vol_key in vol_mapping:
            return Path(vol_mapping[vol_key])

        # Try integer key (JSON may have converted)
        if volume in vol_mapping:
            return Path(vol_mapping[volume])

        return None

    def _load_ocr_file(self, filepath: Path) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Load OCR file and extract text + page mappings."""
        with open(filepath) as f:
            content = f.read()

        # Handle both JSON array and JSONL formats
        if content.strip().startswith('['):
            data = json.loads(content)
            if isinstance(data, list):
                data = data[0]
        else:
            data = json.loads(content.split('\n')[0])

        text = data.get('text', '')

        # Parse pdf_page_numbers: [[start_char, end_char, page_num], ...]
        page_numbers = data.get('attributes', {}).get('pdf_page_numbers', [])
        page_map = [(p[0], p[1], p[2]) for p in page_numbers]

        return text, page_map

    def load_volume(self, edition: str, volume: int) -> Dict:
        """Load OCR data for a volume."""
        cache_key = (edition, volume)
        if cache_key in self.cache:
            return self.cache[cache_key]

        ocr_file = self._find_ocr_file(edition, volume)
        if not ocr_file:
            raise FileNotFoundError(f"No OCR file found for {edition} vol{volume}")

        text, page_map = self._load_ocr_file(ocr_file)

        result = {
            'text': text,
            'page_map': page_map,
            'file': str(ocr_file)
        }

        self.cache[cache_key] = result
        return result

    def get_page_range_text(self, edition: str, volume: int,
                            start_page: int, end_page: int) -> str:
        """Extract raw OCR text for a specific page range."""
        data = self.load_volume(edition, volume)
        text = data['text']
        page_map = data['page_map']

        # Find character range for pages
        start_char = None
        end_char = None

        for char_start, char_end, page_num in page_map:
            if page_num == start_page:
                start_char = char_start
            if page_num == end_page:
                end_char = char_end
            # Also capture if we're within the range
            if start_page <= page_num <= end_page:
                if start_char is None:
                    start_char = char_start
                end_char = char_end

        if start_char is None or end_char is None:
            return ""

        return text[start_char:end_char]


def convert_range_to_sample_errors(r: Dict) -> List[dict]:
    """
    Convert a range dict to sample_errors format for prompt building.
    Handles both reparse_ranges.json and repair_manifest.json formats.
    """
    # If already has sample_errors, use it directly
    if 'sample_errors' in r and r['sample_errors']:
        return r['sample_errors']

    # Convert from reparse_ranges.json format (sample_titles + reasons)
    sample_errors = []
    sample_titles = r.get('sample_titles', [])
    reasons = r.get('reasons', [])
    start_page = r.get('start_page', '?')

    for i, title in enumerate(sample_titles):
        error = {
            'title': title,
            'page': start_page,  # Approximate - we don't have exact page in this format
            'reason': reasons[i] if i < len(reasons) else 'unknown'
        }
        sample_errors.append(error)

    return sample_errors


def build_reparse_prompt(edition: str, volume: str,
                         start_page: int, end_page: int,
                         raw_text: str, sample_errors: List[dict]) -> str:
    """Build prompt for Gemini to identify article boundaries."""

    prompt = f"""You are analyzing raw OCR text from the {edition} Encyclopaedia Britannica, {volume}, pages {start_page}-{end_page}.

TASK: Identify all article boundaries in this text. Return the title and page range for each article.

CONTEXT:
- This is an alphabetically-organized encyclopedia
- Article titles appear in ALL CAPITALS followed by a comma or period (e.g., "CHEMISTRY," or "DICTIONARY.")
- Long articles (treatises) may span many pages and contain sub-sections
- Sub-sections like "PROPOSITION IX", "PROBLEM VII", "EXPLANATION OF PLATE" are NOT separate articles - they belong to the parent article
- Sentence fragments starting with "THIS", "THESE", "MANY", "WHEN" etc. are NOT article titles
- Word examples within a dictionary article (like "MOE", "NARROW", "WIDE") are NOT separate articles

KNOWN PARSING ERRORS IN THIS RANGE (these were incorrectly identified as articles):
"""

    for err in sample_errors[:5]:
        prompt += f"  - \"{err.get('title', 'unknown')[:50]}\" on page {err.get('page', '?')}\n"

    prompt += f"""
RAW OCR TEXT (pages {start_page}-{end_page}):
---
{raw_text}
---

INSTRUCTIONS:
Return a JSON array of articles found in this page range. For each article:
- "title": The article headword in capitals (e.g., "DICTIONARY")
- "sp": Start page number
- "ep": End page number

Example output:
{{
  "articles": [
    {{"title": "DIALECT", "sp": 371, "ep": 371}},
    {{"title": "DICTIONARY", "sp": 376, "ep": 382}},
    {{"title": "DIDACTIC", "sp": 382, "ep": 382}}
  ]
}}

IMPORTANT: Only include REAL article headwords. Do NOT include:
- Sub-section headers (PROPOSITION, PROBLEM, SCHOLIUM, etc.)
- Sentence fragments
- Word examples from within articles
- Cross-references embedded in text

Respond with ONLY valid JSON, no additional text.
"""
    return prompt


def call_gemini(model, prompt: str) -> Optional[dict]:
    """Call Gemini API and parse response."""
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Clean markdown code blocks
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]

            return json.loads(text.strip())

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return None


def merge_overlapping_ranges(ranges: List[Dict]) -> List[Dict]:
    """
    Merge overlapping page ranges within the same volume to avoid redundant processing.
    Returns a new list with merged ranges.
    """
    if not ranges:
        return []

    # Group by volume
    by_volume = {}
    for r in ranges:
        vol = r['volume']
        if vol not in by_volume:
            by_volume[vol] = []
        by_volume[vol].append(r)

    merged_ranges = []

    for vol in sorted(by_volume.keys()):
        vol_ranges = by_volume[vol]
        # Sort by start page
        vol_ranges.sort(key=lambda x: x['start_page'])

        merged = []
        for r in vol_ranges:
            if merged and r['start_page'] <= merged[-1]['end_page'] + 5:  # Allow small gaps
                # Merge with previous range
                merged[-1]['end_page'] = max(merged[-1]['end_page'], r['end_page'])
                # Combine sample_errors if present
                if 'sample_errors' in r and 'sample_errors' in merged[-1]:
                    merged[-1]['sample_errors'].extend(r.get('sample_errors', []))
            else:
                # Start new range (copy to avoid modifying original)
                merged.append({
                    'edition': r['edition'],
                    'volume': r['volume'],
                    'start_page': r['start_page'],
                    'end_page': r['end_page'],
                    'sample_errors': r.get('sample_errors', [])
                })
        merged_ranges.extend(merged)

    return merged_ranges


def split_large_ranges(ranges: List[Dict], max_pages: int = 20, overlap: int = 2) -> List[Dict]:
    """
    Split large page ranges into smaller chunks for API processing.

    Args:
        ranges: List of page ranges
        max_pages: Maximum pages per chunk (default 20)
        overlap: Pages to overlap between chunks to catch boundary articles (default 2)

    Returns:
        List of smaller page ranges
    """
    split_ranges = []

    for r in ranges:
        start = r['start_page']
        end = r['end_page']
        pages = end - start + 1

        if pages <= max_pages:
            # Range is small enough, keep as-is
            split_ranges.append(r)
        else:
            # Split into chunks
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + max_pages - 1, end)
                split_ranges.append({
                    'edition': r['edition'],
                    'volume': r['volume'],
                    'start_page': chunk_start,
                    'end_page': chunk_end,
                    'sample_errors': r.get('sample_errors', [])
                })
                # Move to next chunk with overlap
                chunk_start = chunk_end - overlap + 1
                # Avoid infinite loop if overlap >= max_pages
                if chunk_start <= split_ranges[-1]['start_page']:
                    break

    return split_ranges


def main():
    parser = argparse.ArgumentParser(description="Re-parse OCR with Gemini (v2 - raw OCR)")
    parser.add_argument("--edition", required=True, help="Edition year (e.g., 1771)")
    parser.add_argument("--volume", help="Volume (e.g., vol2) - if omitted, processes all volumes")
    parser.add_argument("--range", type=int, help="Specific range index to process (after merging/splitting)")
    parser.add_argument("--limit", type=int, help="Max ranges to process (default: all)")
    parser.add_argument("--chunk-size", type=int, default=20, help="Max pages per API call (default: 20)")
    parser.add_argument("--save-every", type=int, default=25, help="Save results every N chunks (default: 25)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt without API call")
    parser.add_argument("--output", help="Output file (default: gemini_{edition}_{volume}.json)")
    parser.add_argument("--manifest", default="reparse_ranges.json",
                        help="Manifest file with page ranges (default: reparse_ranges.json)")
    args = parser.parse_args()

    # Load reparse ranges from manifest file
    # Supports both reparse_ranges.json (list format) and repair_manifest.json (dict with 'pages' key)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest_data = json.load(f)

    # Handle both formats
    if isinstance(manifest_data, list):
        # reparse_ranges.json format: direct list of ranges
        all_ranges = manifest_data
    elif isinstance(manifest_data, dict):
        # repair_manifest.json format: dict with 'pages' key
        all_ranges = manifest_data.get('pages', [])
    else:
        print(f"ERROR: Unknown manifest format in {args.manifest}")
        sys.exit(1)

    print(f"Loaded {len(all_ranges)} ranges from {args.manifest}")

    # Filter to requested edition/volume
    ranges = [r for r in all_ranges if r['edition'] == args.edition]

    if args.volume:
        ranges = [r for r in ranges if r['volume'] == args.volume]

    # Merge overlapping ranges to avoid redundant processing
    original_count = len(ranges)
    ranges = merge_overlapping_ranges(ranges)
    if original_count != len(ranges):
        print(f"Merged {original_count} overlapping ranges into {len(ranges)} merged ranges")

    # Split large ranges into manageable chunks
    merged_count = len(ranges)
    ranges = split_large_ranges(ranges, max_pages=args.chunk_size, overlap=2)
    if merged_count != len(ranges):
        print(f"Split into {len(ranges)} chunks of max {args.chunk_size} pages each")

    if args.range is not None:
        ranges = [ranges[args.range]] if args.range < len(ranges) else []
    elif args.limit:
        ranges = ranges[:args.limit]

    if not ranges:
        vol_str = args.volume if args.volume else "all volumes"
        print(f"No ranges found for {args.edition} {vol_str}")
        sys.exit(1)

    vol_str = args.volume if args.volume else "all volumes"
    print(f"Processing {len(ranges)} ranges for {args.edition} {vol_str}")

    # Initialize
    ocr_loader = RawOCRLoader()
    model = None if args.dry_run else configure_gemini()

    results = []

    for idx, r in enumerate(ranges):
        start_page = r['start_page']
        end_page = r['end_page']
        volume = r['volume']
        vol_num = int(volume.replace('vol', ''))

        print(f"\n  [{idx+1}/{len(ranges)}] {volume} pages {start_page}-{end_page}...", flush=True)

        # Get raw OCR text for this page range
        try:
            raw_text = ocr_loader.get_page_range_text(
                args.edition, vol_num, start_page, end_page
            )
        except FileNotFoundError as e:
            print(f"    Error: {e}")
            continue

        if not raw_text:
            print(f"    No text found for pages {start_page}-{end_page}", flush=True)
            continue

        print(f"    Raw text: {len(raw_text)} chars", flush=True)

        # Build prompt
        sample_errors = convert_range_to_sample_errors(r)
        prompt = build_reparse_prompt(
            args.edition, volume, start_page, end_page,
            raw_text, sample_errors
        )

        if args.dry_run:
            print(f"    [DRY RUN] Prompt length: {len(prompt)} chars")
            print(f"    First 500 chars of raw OCR:\n{raw_text[:500]}")
            results.append({
                'range': r,
                'raw_text_length': len(raw_text),
                'prompt_length': len(prompt)
            })
        else:
            result = call_gemini(model, prompt)
            if result:
                n_articles = len(result.get('articles', []))
                print(f"    Found {n_articles} articles")
                results.append({
                    'range': r,
                    'result': result
                })
            else:
                print(f"    Failed to get response")

        time.sleep(1)  # Rate limiting

        # Incremental save
        if not args.dry_run and args.save_every and (idx + 1) % args.save_every == 0:
            if args.output:
                output_file = args.output
            else:
                vol_suffix = args.volume if args.volume else "all"
                output_file = f"gemini_{args.edition}_{vol_suffix}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"    [Saved {len(results)} results to {output_file}]")

    # Final save
    if args.output:
        output_file = args.output
    else:
        vol_suffix = args.volume if args.volume else "all"
        output_file = f"gemini_{args.edition}_{vol_suffix}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
