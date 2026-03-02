#!/usr/bin/env python3
"""
Gemini-based mega-article splitter.

Identifies articles that have absorbed adjacent entries (due to missing
headings or crossref overflow) and uses Gemini to find the exact split
boundary.  Python validates the response with an exact string match,
so LLM hallucination is harmless — we simply skip unfound splits.

Workflow:
  1. SCAN:  Find large articles with missing dictionary headwords in their text
  2. SPLIT: Ask Gemini "where does the article on X begin?"
  3. VALIDATE: text.find(gemini_response) — skip if not found
  4. APPLY: Rewrite the .articles.jsonl output files with proper splits

Usage:
    python gemini_mega_splitter.py                # scan + split + apply
    python gemini_mega_splitter.py --scan-only    # just show candidates
    python gemini_mega_splitter.py --dry-run      # scan + call Gemini but don't rewrite files
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from config import ARTICLES_DIR, HEADWORD_DICT_PATH
from lis_parser import normalize_sort_key

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WORD_THRESHOLD = 30_000       # minimum words to consider as mega-article
MIN_HEADWORD_SOURCES = 2      # dictionary entry must have 2+ sources
MIN_HEADWORD_LEN = 4          # skip very short headwords (ABE, FAS, etc.)
MAX_HEADWORD_LEN = 50
GEMINI_MODEL = 'gemini-3-flash-preview'
SPLITS_MANIFEST = ARTICLES_DIR / 'mega_splits.json'


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_headword_dict() -> dict[str, dict]:
    with open(HEADWORD_DICT_PATH) as f:
        return json.load(f)


def load_all_articles() -> dict[str, list[dict]]:
    """Load all output articles, keyed by source filename."""
    articles_by_file: dict[str, list[dict]] = {}
    for p in sorted(ARTICLES_DIR.glob('*.articles.jsonl')):
        articles = []
        with open(p) as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
        articles_by_file[p.name] = articles
    return articles_by_file


# ---------------------------------------------------------------------------
# 2. Find candidates
# ---------------------------------------------------------------------------

def find_candidates(
    articles_by_file: dict[str, list[dict]],
    headword_dict: dict[str, dict],
) -> list[dict]:
    """Find mega-articles that likely absorbed adjacent entries.

    Strategy: For each large article, find dictionary headwords that are:
      1. Expected in this edition but not found as separate articles
      2. Alphabetically in the range between this article and its neighbors
         (covers both forward absorption AND crossref-overflow absorption)
      3. Actually appear as a substring in the article text

    This alphabetical range constraint eliminates the false-positive problem
    of common English words matching (e.g., "ANIMALS" in an AGRICULTURE article).
    """
    # Build set of found headwords per edition
    found_by_edition: dict[int, set[str]] = defaultdict(set)
    for fname, articles in articles_by_file.items():
        for art in articles:
            ey = art.get('edition_year')
            if ey:
                found_by_edition[ey].add(normalize_sort_key(art['title']))

    # Build expected headwords per edition from dictionary, indexed by sort key
    expected_by_edition: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for norm_key, entry in headword_dict.items():
        if entry.get('source_count', 0) < MIN_HEADWORD_SOURCES:
            continue
        hw = entry['headword']
        if len(hw) < MIN_HEADWORD_LEN or len(hw) > MAX_HEADWORD_LEN:
            continue
        for ed_str in entry.get('editions', []):
            ey = int(ed_str)
            expected_by_edition[ey].append((norm_key, hw))

    # Find missing headwords per edition, sorted by sort key for range queries
    missing_by_edition: dict[int, list[tuple[str, str]]] = {}
    for ey, expected in expected_by_edition.items():
        found = found_by_edition.get(ey, set())
        missing = sorted([(nk, hw) for nk, hw in expected if nk not in found])
        if missing:
            missing_by_edition[ey] = missing

    log.info(f"Missing headwords by edition: "
             + ", ".join(f"{ey}: {len(m)}" for ey, m in sorted(missing_by_edition.items())))

    # For each mega-article, find missing headwords in its alphabetical range
    candidates = []
    mega_count = 0
    for fname, articles in articles_by_file.items():
        for i, art in enumerate(articles):
            if art['word_count'] < WORD_THRESHOLD:
                continue
            if art['type'] != 'article':
                continue

            ey = art.get('edition_year')
            if not ey or ey not in missing_by_edition:
                continue

            mega_count += 1
            art_key = normalize_sort_key(art['title'])

            # Determine alphabetical range: from PREVIOUS article to NEXT article
            # This catches both forward absorption (art swallowed stuff after it)
            # and overflow absorption (art received overflow from a crossref before it)
            prev_key = ''
            if i > 0:
                prev_key = normalize_sort_key(articles[i - 1]['title'])
            next_key = '\xff'  # sorts after everything
            if i + 1 < len(articles):
                next_key = normalize_sort_key(articles[i + 1]['title'])

            # Use the wider range: prev_key to next_key
            # (covers the full gap this article could have absorbed)
            range_lo = prev_key
            range_hi = next_key

            # Find missing headwords in this alphabetical range
            missing = missing_by_edition[ey]
            in_range = [
                (nk, hw) for nk, hw in missing
                if range_lo < nk < range_hi and nk != art_key
            ]

            if not in_range:
                continue

            # Now check which of these appear at a structural boundary in the text
            # (line start, paragraph break) — not just inline mentions
            text = art['text']
            text_upper = text.upper()
            art_title_upper = art['title'].upper()

            for norm_key, hw in in_range:
                hw_upper = hw.upper()

                # Skip if headword is a substring of the article's own title
                if hw_upper in art_title_upper:
                    continue

                # Look for the headword at a structural boundary:
                # \nHEADWORD or \n\nHeadword — indicating a section/article heading
                # This filters out inline mentions like "the united states" mid-sentence
                found_structural = False
                for pattern in [
                    f'\n{hw_upper}',           # uppercase at line start
                    f'\n{hw}',                 # original case at line start
                    f'\n\n{hw}',               # after paragraph break
                ]:
                    search_text = text_upper if pattern[1:].isupper() else text
                    if pattern.upper() in text_upper:
                        # Verify it's at line start (not mid-word)
                        idx = text_upper.find(pattern.upper())
                        if idx >= 0:
                            # Check character after headword is a delimiter
                            end_pos = idx + len(pattern)
                            if end_pos < len(text):
                                next_char = text[end_pos]
                                if next_char in ',. ;\n\t:—-(':
                                    found_structural = True
                                    break
                            else:
                                found_structural = True
                                break

                if not found_structural:
                    continue

                candidates.append({
                    'source_file': fname,
                    'article_index': i,
                    'article_id': art['article_id'],
                    'article_title': art['title'],
                    'article_words': art['word_count'],
                    'edition_year': ey,
                    'target_headword': hw,
                    'target_norm_key': norm_key,
                })

    log.info(f"Checked {mega_count} mega-articles (>{WORD_THRESHOLD} words)")

    # Deduplicate: one article might match many missing headwords.
    # Group by article, keep only the most significant missing headwords
    # (those with highest source_count in the dictionary).
    by_article: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_article[c['article_id']].append(c)

    # For each article, keep top N headwords by importance
    deduped = []
    for art_id, cands in by_article.items():
        # Sort by source_count descending, then by headword length descending
        cands.sort(key=lambda c: (
            headword_dict.get(c['target_norm_key'], {}).get('source_count', 0),
            len(c['target_headword']),
        ), reverse=True)
        # Keep top 10 per article
        deduped.extend(cands[:10])

    log.info(f"Found {len(deduped)} candidate splits across "
             f"{len(by_article)} mega-articles")
    return deduped


# ---------------------------------------------------------------------------
# 3. Gemini API
# ---------------------------------------------------------------------------

def get_gemini_model():
    """Initialize Gemini model."""
    import google.generativeai as genai

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        log.error("GEMINI_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def ask_gemini_for_split(
    model,
    article_text: str,
    target_headword: str,
    edition_year: int,
) -> dict:
    """Ask Gemini to find where the target article begins.

    Returns: {found: bool, quote: str, raw_response: str}
    """
    prompt = f"""You are analyzing text from an {edition_year} edition of the Encyclopaedia Britannica. Due to OCR errors during digitization, multiple encyclopedia articles have been merged into a single block of text.

Somewhere in the text below, a new article about **{target_headword}** begins. The start of the article is typically marked by the headword (possibly in ALL CAPS, Title Case, or with OCR errors) followed by a comma, period, or description.

Find where the article about {target_headword} begins and return ONLY a JSON object:

{{"found": true, "quote": "<exact first 60 characters starting from the headword, copied verbatim from the text including any OCR errors or unusual spacing>"}}

If you cannot confidently identify where the {target_headword} article begins, return:
{{"found": false, "quote": ""}}

IMPORTANT:
- Copy the text EXACTLY as it appears — do not fix OCR errors or normalize spacing
- The quote must be findable via exact string match in the original text
- Start from the headword itself, not from text before it
- Include about 60 characters to ensure uniqueness

TEXT:
{article_text}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Parse JSON from response (handle markdown code blocks)
        json_str = raw
        if '```' in json_str:
            # Extract content between code fences
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        result = json.loads(json_str)
        return {
            'found': result.get('found', False),
            'quote': result.get('quote', ''),
            'raw_response': raw,
        }
    except json.JSONDecodeError:
        log.warning(f"  Failed to parse Gemini JSON: {raw[:200]}")
        return {'found': False, 'quote': '', 'raw_response': raw}
    except Exception as e:
        log.error(f"  Gemini API error: {e}")
        return {'found': False, 'quote': '', 'raw_response': str(e)}


# ---------------------------------------------------------------------------
# 4. Validate and build splits
# ---------------------------------------------------------------------------

def validate_split(article_text: str, quote: str) -> int:
    """Find the exact position of the Gemini quote in the article text.

    Returns the character position, or -1 if not found.
    Tries exact match first, then progressively shorter prefixes.
    """
    if not quote or len(quote) < 5:
        return -1

    # Try exact match
    pos = article_text.find(quote)
    if pos >= 0:
        return pos

    # Try progressively shorter prefixes (Gemini might have gotten
    # the tail end slightly wrong)
    for trim in range(5, len(quote) // 2, 5):
        shorter = quote[:len(quote) - trim]
        if len(shorter) < 10:
            break
        pos = article_text.find(shorter)
        if pos >= 0:
            # Verify it's reasonably unique (not found again nearby)
            second = article_text.find(shorter, pos + 1)
            if second == -1 or second - pos > 10000:
                return pos

    return -1


def process_candidates(
    candidates: list[dict],
    articles_by_file: dict[str, list[dict]],
    model,
) -> list[dict]:
    """Send candidates to Gemini and validate responses."""
    splits = []

    for i, cand in enumerate(candidates):
        fname = cand['source_file']
        art_idx = cand['article_index']
        article = articles_by_file[fname][art_idx]
        hw = cand['target_headword']
        ey = cand['edition_year']

        log.info(f"[{i+1}/{len(candidates)}] {fname} — "
                 f"{cand['article_title']} ({cand['article_words']:,} words) "
                 f"→ looking for {hw}")

        # Call Gemini
        result = ask_gemini_for_split(model, article['text'], hw, ey)

        if not result['found']:
            log.info(f"  Gemini: not found")
            continue

        quote = result['quote']
        log.info(f"  Gemini quote: {quote[:80]!r}")

        # Validate: find in text
        pos = validate_split(article['text'], quote)
        if pos < 0:
            log.warning(f"  VALIDATION FAILED — quote not found in text")
            continue

        # Sanity check: split point should not be in first 100 or last 100 chars
        if pos < 100 or pos > len(article['text']) - 100:
            log.warning(f"  Split at pos {pos} too close to boundary, skipping")
            continue

        log.info(f"  VALIDATED at position {pos:,}")
        splits.append({
            'source_file': fname,
            'article_index': art_idx,
            'article_id': cand['article_id'],
            'article_title': cand['article_title'],
            'target_headword': hw,
            'split_position': pos,
            'quote': quote,
            'gemini_response': result['raw_response'],
        })

        # Rate limiting
        time.sleep(0.5)

    return splits


# ---------------------------------------------------------------------------
# 5. Apply splits
# ---------------------------------------------------------------------------

def apply_splits(
    splits: list[dict],
    articles_by_file: dict[str, list[dict]],
) -> int:
    """Apply validated splits to the output article files.

    For each split, the mega-article is shortened and a new article is
    inserted after it.  Returns the number of splits applied.
    """
    # Group splits by file, then by article index (descending to avoid
    # index shifts when inserting)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for s in splits:
        by_file[s['source_file']].append(s)

    applied = 0
    for fname, file_splits in by_file.items():
        articles = articles_by_file[fname]
        # Sort by article_index descending, then split_position descending
        # so insertions don't shift earlier indices
        file_splits.sort(key=lambda s: (s['article_index'], s['split_position']),
                         reverse=True)

        for s in file_splits:
            idx = s['article_index']
            art = articles[idx]
            pos = s['split_position']
            hw = s['target_headword']

            # Split the text
            before_text = art['text'][:pos].strip()
            after_text = art['text'][pos:].strip()

            if not before_text or not after_text:
                log.warning(f"  Empty split result for {hw} in {art['title']}, skipping")
                continue

            # Update the original article (trimmed)
            art['text'] = before_text
            art['word_count'] = len(before_text.split())
            art['paragraph_count'] = before_text.count('\n\n') + 1

            # Create new article for the split-off content
            new_art = dict(art)
            new_art['article_id'] = f"{art['article_id']}_gsplit"
            new_art['title'] = hw
            new_art['text'] = after_text
            new_art['word_count'] = len(after_text.split())
            new_art['paragraph_count'] = after_text.count('\n\n') + 1
            new_art['heading_pattern'] = 'gemini_split'
            new_art['lis_confidence'] = 0.7

            # Insert after the original
            articles.insert(idx + 1, new_art)
            applied += 1

            log.info(f"  Applied: {art['title']} ({art['word_count']:,}w) + "
                     f"{hw} ({new_art['word_count']:,}w)")

        # Rewrite the file
        output_path = ARTICLES_DIR / fname
        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        log.info(f"  Rewrote {fname} ({len(articles)} articles)")

    return applied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global WORD_THRESHOLD, GEMINI_MODEL

    parser = argparse.ArgumentParser(description='Gemini mega-article splitter')
    parser.add_argument('--scan-only', action='store_true',
                        help='Only scan for candidates, do not call Gemini')
    parser.add_argument('--dry-run', action='store_true',
                        help='Call Gemini but do not rewrite output files')
    parser.add_argument('--threshold', type=int, default=WORD_THRESHOLD,
                        help=f'Minimum word count for mega-articles (default: {WORD_THRESHOLD})')
    parser.add_argument('--model', default=GEMINI_MODEL,
                        help=f'Gemini model to use (default: {GEMINI_MODEL})')
    args = parser.parse_args()

    WORD_THRESHOLD = args.threshold
    GEMINI_MODEL = args.model

    log.info("Loading headword dictionary...")
    headword_dict = load_headword_dict()
    log.info(f"  {len(headword_dict)} entries")

    log.info("Loading output articles...")
    articles_by_file = load_all_articles()
    total = sum(len(a) for a in articles_by_file.values())
    log.info(f"  {total} articles across {len(articles_by_file)} files")

    log.info("Scanning for candidates...")
    candidates = find_candidates(articles_by_file, headword_dict)

    if not candidates:
        log.info("No candidates found.")
        return

    # Show candidates
    log.info(f"\nCandidates ({len(candidates)}):")
    for c in candidates:
        log.info(f"  {c['edition_year']} {c['article_title']:30s} "
                 f"({c['article_words']:>7,}w) → {c['target_headword']}")

    if args.scan_only:
        return

    # Call Gemini
    log.info(f"\nCalling Gemini ({GEMINI_MODEL})...")
    model = get_gemini_model()
    splits = process_candidates(candidates, articles_by_file, model)

    log.info(f"\nValidated splits: {len(splits)} / {len(candidates)}")

    # Save manifest
    with open(SPLITS_MANIFEST, 'w') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    log.info(f"Manifest saved to {SPLITS_MANIFEST}")

    if not splits:
        log.info("No valid splits found.")
        return

    if args.dry_run:
        log.info("Dry run — not rewriting files.")
        for s in splits:
            log.info(f"  Would split: {s['article_title']} → {s['target_headword']} "
                     f"at pos {s['split_position']:,}")
        return

    # Apply splits
    log.info("\nApplying splits...")
    applied = apply_splits(splits, articles_by_file)
    log.info(f"Done. Applied {applied} splits.")


if __name__ == '__main__':
    main()
