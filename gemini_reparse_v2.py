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
from typing import List, Dict, Optional, Tuple
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
    """Load raw OCR text with page number mappings from original OCR results."""

    # Map (edition, volume) to OCR result file
    # These are in ocr_results/{edition}_britannica_{nth}/output_*.jsonl
    OCR_FILE_MAP = {
        # 1771 1st Edition
        ('1771', 1): 'ocr_results/1771_britannica_1st/output_114aa9270857798ee869db8d06996ca185bd7105.jsonl',
        ('1771', 2): 'ocr_results/1771_britannica_1st/output_1718f3a66eab1422763966bf5470b3b3906faa08.jsonl',
        ('1771', 3): 'ocr_results/1771_britannica_1st/output_1455a43ee23170019c3b9c5052be80106e7aaf8e.jsonl',
        # Add more mappings as needed
    }

    def __init__(self, ocr_dir: str = "ocr_results"):
        self.ocr_dir = Path(ocr_dir)
        self.cache = {}  # (edition, volume) -> {text, page_map}

    def _find_ocr_file(self, edition: str, volume: int) -> Optional[Path]:
        """Find the OCR result file for a given edition/volume."""
        # First check explicit mapping
        key = (edition, volume)
        if key in self.OCR_FILE_MAP:
            return Path(self.OCR_FILE_MAP[key])

        # Otherwise try to find by scanning directory
        edition_dirs = {
            '1771': '1771_britannica_1st',
            '1778': '1778_britannica_2nd',
            '1797': '1797_britannica_3rd',
            '1810': '1810_britannica_4th',
        }

        ed_dir = edition_dirs.get(edition)
        if not ed_dir:
            return None

        # Scan files and check Source-File metadata for volume number
        dir_path = self.ocr_dir / ed_dir
        if not dir_path.exists():
            return None

        for f in dir_path.glob('output_*.jsonl'):
            try:
                with open(f) as fh:
                    content = fh.read()
                if content.strip().startswith('['):
                    data = json.loads(content)
                    entry = data[0] if isinstance(data, list) else data
                else:
                    entry = json.loads(content.split('\n')[0])

                source = entry.get('metadata', {}).get('Source-File', '')
                if f'Volume {volume}' in source or f'Volume{volume}' in source:
                    return f
            except:
                continue

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
{raw_text[:12000]}
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


def main():
    parser = argparse.ArgumentParser(description="Re-parse OCR with Gemini (v2 - raw OCR)")
    parser.add_argument("--edition", required=True, help="Edition year (e.g., 1771)")
    parser.add_argument("--volume", required=True, help="Volume (e.g., vol2)")
    parser.add_argument("--range", type=int, help="Specific range index to process")
    parser.add_argument("--limit", type=int, default=5, help="Max ranges to process")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt without API call")
    parser.add_argument("--output", default="gemini_corrections_v2.json", help="Output file")
    args = parser.parse_args()

    # Load reparse ranges
    with open('reparse_ranges.json') as f:
        all_ranges = json.load(f)

    # Filter to requested edition/volume
    vol_num = int(args.volume.replace('vol', ''))
    ranges = [r for r in all_ranges
              if r['edition'] == args.edition and r['volume'] == args.volume]

    if args.range is not None:
        ranges = [ranges[args.range]] if args.range < len(ranges) else []
    else:
        ranges = ranges[:args.limit]

    if not ranges:
        print(f"No ranges found for {args.edition} {args.volume}")
        sys.exit(1)

    print(f"Processing {len(ranges)} ranges for {args.edition} {args.volume}")

    # Initialize
    ocr_loader = RawOCRLoader()
    model = None if args.dry_run else configure_gemini()

    results = []

    for idx, r in enumerate(ranges):
        start_page = r['start_page']
        end_page = r['end_page']

        print(f"\n  [{idx+1}/{len(ranges)}] Pages {start_page}-{end_page}...", flush=True)

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
        prompt = build_reparse_prompt(
            args.edition, args.volume, start_page, end_page,
            raw_text, r.get('sample_errors', [])
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

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
