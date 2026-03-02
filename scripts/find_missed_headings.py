#!/usr/bin/env python3
"""
Find ALL CAPS headings on single newlines that the parser missed.

Step 1: Extract candidates from OCR text (single \n before ALL CAPS + delimiter)
Step 2: Filter out known article titles
Step 3: Send to Gemini in batches for classification
Step 4: Output list of probable missed article headings

The parser requires \n\n before ALL CAPS headings, but some legitimate article
headings have only \n (e.g., UNITED STATES OF NORTH AMERICA after a crossref).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from config import ARTICLES_DIR

OCR_DIR = Path("/home/jic823/plato/ocr_organized")
OUTPUT_FILE = Path(__file__).parent.parent.parent / "1815EncyclopediaBritannicaNLS" / "data" / "missed_headings_candidates.jsonl"

# Pattern: \n followed by ALL CAPS text ending with delimiter
# Negative lookbehind ensures it's a single \n (not \n\n which parser already handles)
SINGLE_NL_CAPS = re.compile(
    r'(?<!\n)\n([A-Z][A-Z][A-Z\s\-\'\.&;:,]*?)(?:\.|,|;)\s',
)

# Also catch titlecase on single newlines (like "China, a country...")
SINGLE_NL_TITLE = re.compile(
    r'(?<!\n)\n([A-Z][a-z]+(?:[\s\-][A-Za-z]+)*),\s+(?:a|an|the|in|one|or)\s',
)

# Skip patterns
SKIP_RE = re.compile(
    r'^(PLATE|PLATES|FIG|FIGURE|TABLE|TABLES|CHAP|CHAPTER|SECT|SECTION|'
    r'PART\s|VOL|VOLUME|BOOK|NOTE|NOTES|SEE\s|END\sOF|INDEX|CONTENTS|'
    r'ERRATA|FINIS|ADVERTISEMENT|APPENDIX|PREFACE|INTRODUCTION|'
    r'PRINTED|MDCC|ENCYCLOP|BRITANNICA|MEMOIR|DISSERTAT|SUPPLEMENT|'
    r'[IVXLCDM]+\s*$)',
    re.IGNORECASE,
)


def extract_candidates():
    """Extract all single-newline ALL CAPS candidates from OCR files."""
    # Load existing article titles per edition
    found_titles = defaultdict(set)
    for p in sorted(ARTICLES_DIR.glob("*.articles.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                ey = art.get('edition_year')
                if ey:
                    found_titles[ey].add(art['title'].upper().strip())

    candidates = []

    for p in sorted(OCR_DIR.glob("*.jsonl")):
        fname = p.name
        parts = fname.split('_')
        try:
            ey = int(parts[2])
            vol = parts[3]  # e.g., "v21"
        except (IndexError, ValueError):
            continue

        with open(p) as f:
            for jline in f:
                rec = json.loads(jline)
                text = rec['text']
                text_len = len(text)

                for m in SINGLE_NL_CAPS.finditer(text):
                    candidate = m.group(1).strip().rstrip('.,;: ')

                    if len(candidate) < 3 or len(candidate) > 80:
                        continue
                    if candidate in found_titles[ey]:
                        continue
                    if SKIP_RE.match(candidate):
                        continue
                    if re.match(r'^[IVXLCDM\s\.]+$', candidate):
                        continue

                    pos = m.start()
                    # Context: 60 chars before and after
                    ctx_before = text[max(0, pos-60):pos].replace('\n', '\\n')
                    full_match = m.group()
                    ctx_after = text[pos+len(full_match):pos+len(full_match)+60].replace('\n', '\\n')

                    candidates.append({
                        'file': fname,
                        'edition': ey,
                        'vol': vol,
                        'candidate': candidate,
                        'position': pos,
                        'pct': round(100 * pos / text_len, 1),  # position as % of file
                        'before': ctx_before[-50:],
                        'match': full_match.replace('\n', '\\n').strip(),
                        'after': ctx_after[:50],
                    })

    return candidates


def batch_classify_with_gemini(candidates, batch_size=200):
    """Send candidates to Gemini in batches for classification."""
    import google.generativeai as genai

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    all_results = []

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(candidates) + batch_size - 1) // batch_size

        # Build the batch listing
        lines = []
        for i, c in enumerate(batch):
            idx = batch_start + i
            lines.append(
                f"{idx}|{c['edition']}|{c['vol']}|{c['candidate']}|"
                f"{c['before']}|||{c['match']}|||{c['after']}"
            )
        listing = '\n'.join(lines)

        prompt = f"""You are classifying text extracted from OCR of the Encyclopaedia Britannica (editions 1771-1860).

Below is a list of ALL CAPS text fragments found at line boundaries in the OCR. Each line has:
  INDEX|EDITION_YEAR|VOLUME|CANDIDATE_TEXT|...context_before...|||...the_match...|||...context_after...

Your task: classify each candidate as either:
  "article" — a genuine encyclopedia article heading (e.g., "UNITED STATES OF NORTH AMERICA", "GANGES", "BRESCIA")
  "not" — NOT an article heading (e.g., running headers, section labels, table headers, figure captions, cross-references within text, common words that happen to be capitalized)

Clues for "article":
- Followed by a definition pattern like ", a town of...", ", in music, the...", ", or...", ". See..."
- Geographic entries: ", a city/town/river/island/province of..."
- Biographical entries: ", a celebrated/famous..."
- Topic entries: ", in botany/music/law/surgery, ..."
- The heading introduces new content about a distinct subject

Clues for "not":
- Running page headers (repeated text like "SOCIETIES, BOTH AT HOME AND ABROAD")
- Section headings within an article (CHAP, SECT, PART, ORDER, CLASS)
- Table or figure labels
- Words that are ALL CAPS mid-sentence for emphasis
- Cross-reference mentions ("see ARTICLE" within text)

Return ONLY a JSON array of objects, one per candidate:
[{{"i": <index>, "v": "article"}}, {{"i": <index>, "v": "not"}}, ...]

Only include candidates you classify as "article". Omit "not" entries to keep the response small.

CANDIDATES:
{listing}"""

        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} candidates...", end=' ', flush=True)

        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Parse JSON
            json_str = raw
            if '```' in json_str:
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_str, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()

            results = json.loads(json_str)
            article_indices = {r['i'] for r in results if r.get('v') == 'article'}
            print(f"{len(article_indices)} articles found")

            for r in results:
                if r.get('v') == 'article':
                    idx = r['i']
                    if 0 <= idx < len(candidates):
                        all_results.append(candidates[idx])

        except Exception as e:
            print(f"ERROR: {e}")
            # On error, include all candidates from this batch as uncertain
            for c in batch:
                c['uncertain'] = True
                all_results.append(c)

        time.sleep(1)  # rate limit

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Find missed headings in OCR text')
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract candidates, skip Gemini classification')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Candidates per Gemini batch (default: 200)')
    args = parser.parse_args()

    print("Extracting single-newline ALL CAPS candidates...")
    candidates = extract_candidates()

    # Deduplicate by (edition, candidate, position)
    seen = set()
    deduped = []
    for c in candidates:
        key = (c['edition'], c['candidate'], c['position'])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    candidates = deduped

    print(f"  {len(candidates)} unique candidates")
    by_ed = defaultdict(int)
    for c in candidates:
        by_ed[c['edition']] += 1
    for ey in sorted(by_ed):
        print(f"    {ey}: {by_ed[ey]}")

    if args.extract_only:
        # Write raw candidates
        out = OUTPUT_FILE.with_suffix('.raw.jsonl')
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for c in candidates:
                f.write(json.dumps(c) + '\n')
        print(f"\nWrote {len(candidates)} candidates to {out}")
        return

    print(f"\nClassifying with Gemini (batch_size={args.batch_size})...")
    articles = batch_classify_with_gemini(candidates, args.batch_size)

    # Sort by edition, file, position
    articles.sort(key=lambda a: (a['edition'], a['file'], a['position']))

    # Write results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for a in articles:
            f.write(json.dumps(a) + '\n')

    print(f"\n{'='*70}")
    print(f"Gemini classified {len(articles)} as probable article headings")
    print(f"Written to: {OUTPUT_FILE}")
    print(f"{'='*70}")
    by_ed2 = defaultdict(int)
    for a in articles:
        by_ed2[a['edition']] += 1
    for ey in sorted(by_ed2):
        print(f"  {ey}: {by_ed2[ey]}")

    # Print the articles
    print(f"\n{'='*70}")
    print("PROBABLE MISSED ARTICLE HEADINGS:")
    print(f"{'='*70}")
    for a in articles:
        print(f"  {a['edition']} {a['vol']} {a['pct']:5.1f}% | {a['candidate']}")


if __name__ == '__main__':
    main()
