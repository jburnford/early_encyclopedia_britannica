#!/usr/bin/env python3
"""Quick test: send 2-3 candidates to Gemini to validate the split approach.

Two types of mega-article problems:
  TYPE A: Heading present but parser missed it (e.g., UPTON-ON-SEVERN inside UNIVERSE)
          → Ask: "Find where the article about X begins"
  TYPE B: Heading stripped by overflow (e.g., UNITED STATES absorbed into UNIVERSAL)
          → Ask: "This article is labeled X but its content is mostly about something else.
                  Find where the actual X definition begins."
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

sys.path.insert(0, str(Path(__file__).parent))
from config import ARTICLES_DIR
from gemini_mega_splitter import validate_split, get_gemini_model


def load_article(edition_year: int, title: str) -> dict | None:
    for p in sorted(ARTICLES_DIR.glob('*.articles.jsonl')):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                if art['title'] == title and art.get('edition_year') == edition_year:
                    return art
    return None


def ask_gemini(model, prompt: str) -> dict:
    """Send prompt, parse JSON response."""
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        json_str = raw
        if '```' in json_str:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
        result = json.loads(json_str)
        return {'result': result, 'raw': raw}
    except Exception as e:
        return {'result': {}, 'raw': str(e)}


def test_type_b_universal():
    """TYPE B: UNIVERSAL article is 213K chars of UNITED STATES + 28 words of actual UNIVERSAL.

    Ask Gemini to find where UNIVERSAL's own definition begins (= split point).
    Send just the last 2000 chars so it can see the transition.
    """
    print("\n" + "="*70)
    print("TEST TYPE B: 1842 UNIVERSAL → find where actual UNIVERSAL definition begins")
    print("="*70)

    art = load_article(1842, 'UNIVERSAL')
    if not art:
        print("  Article not found!")
        return

    text = art['text']
    print(f"  Article: {art['word_count']:,} words, {len(text):,} chars")
    print(f"  Last 300 chars: {text[-300:]!r}")

    model = get_gemini_model()

    # Send last 2000 chars
    tail = text[-2000:]
    prompt = f"""You are analyzing text from an 1842 edition of the Encyclopaedia Britannica.

This text is the TAIL END of an article labeled "UNIVERSAL". However, most of the article's content is actually about the UNITED STATES — it was incorrectly merged during OCR processing. The actual UNIVERSAL definition is a short passage near the very end.

Find where the actual definition of UNIVERSAL begins (it should be a brief dictionary-style definition about what "universal" means) and return ONLY a JSON object:

{{"found": true, "quote": "<exact first 60 characters of the UNIVERSAL definition, copied verbatim>"}}

If you cannot find it, return:
{{"found": false, "quote": ""}}

TEXT (last 2000 chars of the article):
{tail}"""

    print(f"  Sending {len(tail):,} chars to Gemini...")
    resp = ask_gemini(model, prompt)
    print(f"  Raw response: {resp['raw'][:400]}")

    result = resp['result']
    if result.get('found'):
        quote = result['quote']
        print(f"  Gemini quote: {quote!r}")
        pos = validate_split(text, quote)
        if pos >= 0:
            print(f"  VALIDATED at pos {pos}")
            before_words = len(text[:pos].split())
            after_words = len(text[pos:].split())
            print(f"  Split: UNITED STATES = {before_words:,}w, UNIVERSAL = {after_words:,}w")
        else:
            print(f"  VALIDATION FAILED")
    else:
        print(f"  Gemini: not found")


def test_type_a_upton():
    """TYPE A: UNIVERSE article contains UPTON-ON-SEVERN with clear heading."""
    print("\n" + "="*70)
    print("TEST TYPE A: 1842 UNIVERSE → find UPTON-ON-SEVERN heading")
    print("="*70)

    art = load_article(1842, 'UNIVERSE')
    if not art:
        print("  Article not found!")
        return

    text = art['text']
    print(f"  Article: {art['word_count']:,} words, {len(text):,} chars")

    model = get_gemini_model()

    # Find structural match position
    idx = text.upper().find('\nUPTON-ON-SEVERN')
    if idx < 0:
        print("  Structural match not found!")
        return
    print(f"  \\nUPTON-ON-SEVERN at pos {idx}")

    # Send 4000-char window
    start = max(0, idx - 1000)
    end = min(len(text), idx + 3000)
    window = text[start:end]

    from gemini_mega_splitter import ask_gemini_for_split
    result = ask_gemini_for_split(model, window, 'UPTON-ON-SEVERN', 1842)
    print(f"  Raw: {result['raw_response'][:300]}")

    if result['found']:
        quote = result['quote']
        print(f"  Gemini quote: {quote!r}")
        pos = validate_split(text, quote)
        if pos >= 0:
            print(f"  VALIDATED at pos {pos}")
            before_words = len(text[:pos].split())
            after_words = len(text[pos:].split())
            print(f"  Split: UNIVERSE = {before_words:,}w, UPTON-ON-SEVERN = {after_words:,}w")
        else:
            print(f"  VALIDATION FAILED")
    else:
        print(f"  Gemini: not found")


if __name__ == '__main__':
    # Only run what's specified, or both
    if len(sys.argv) > 1:
        if 'b' in sys.argv[1].lower():
            test_type_b_universal()
        if 'a' in sys.argv[1].lower():
            test_type_a_upton()
    else:
        test_type_b_universal()
        # test_type_a_upton()  # already validated, skip to save API calls
