#!/usr/bin/env python3
"""
Filter missed heading candidates using alphabetical ordering.

For each candidate, find the surrounding parsed articles (by position in OCR text)
and check whether the candidate heading falls alphabetically between them.

If a candidate is way out of alphabetical order (e.g., ULNARIS EXTERNUS inside
the A section), it's clearly a sub-heading within a long article, not a standalone
article.

Also filters out known false-positive patterns:
  - Running header stubs (2-4 letter fragments)
  - PLATE labels
  - Coptic glossary entries
  - Epitaph inscriptions
  - Roman numeral entries
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CANDIDATES_FILE = Path(__file__).parent.parent / "data" / "missed_headings_candidates.jsonl"
ARTICLES_DIR = Path("/home/jic823/plato/britannica_output/articles")
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "missed_headings_filtered.jsonl"
REPORT_FILE = Path(__file__).parent.parent / "data" / "missed_headings_filtered_report.txt"


def normalize_for_sort(title: str) -> str:
    """Normalize a title for alphabetical comparison.

    Removes spaces, punctuation, and leading articles so that multi-word titles
    like 'NOVA SCOTIA' sort correctly against 'NOVARA' (dictionary order ignores
    spaces: NOVASCOTIA vs NOVARA).
    """
    t = title.upper().strip()
    # Remove leading articles
    for prefix in ('THE ', 'A ', 'AN '):
        if t.startswith(prefix):
            t = t[len(prefix):]
    # Remove ALL non-alpha characters (including spaces) for dictionary sort
    t = re.sub(r'[^A-Z]', '', t)
    return t


def is_alpha_between(candidate: str, prev_title: str, next_title: str) -> bool:
    """
    Check if candidate falls alphabetically between prev and next titles.

    Uses a tolerant comparison: truncate all three to the length of the shortest
    common prefix between prev and next + 2 chars. This handles minor ordering
    variations in encyclopedia entries (e.g., BRESCIA between BRENTFORD and BRESCHIANO).
    """
    c = normalize_for_sort(candidate)
    p = normalize_for_sort(prev_title)
    n = normalize_for_sort(next_title)

    if not c or not p or not n:
        return True  # can't determine, let it pass

    # First try exact comparison
    if p <= c <= n:
        return True

    # Tolerant: find common prefix length between prev and next, then compare
    # at that granularity + 2 chars
    common = 0
    for i in range(min(len(p), len(n))):
        if p[i] == n[i]:
            common += 1
        else:
            break
    # Compare at common_prefix + 2 chars (enough to distinguish nearby entries)
    trunc = common + 2
    if trunc < 3:
        trunc = 3  # minimum 3 chars

    pt = p[:trunc]
    ct = c[:trunc]
    nt = n[:trunc]
    return pt <= ct <= nt


def is_running_header_stub(candidate: str, after_context: str) -> bool:
    """Detect running header stubs: 2-4 letter abbreviations."""
    c = candidate.strip()
    # 2-3 letter ALL CAPS that aren't known short articles
    KNOWN_SHORT = {'AIR', 'ALE', 'APE', 'ARM', 'ART', 'ASH', 'AXE', 'BAR',
                   'BAT', 'BED', 'BOW', 'BOX', 'BUD', 'BUS', 'CAB', 'CAM',
                   'CAP', 'CAR', 'CAT', 'COD', 'COW', 'CUP', 'DAM', 'DEN',
                   'DEW', 'DIN', 'DIP', 'DOG', 'DYE', 'EAR', 'EEL', 'EGG',
                   'ELK', 'ELM', 'EMU', 'ERA', 'EVE', 'EWE', 'EYE', 'FAN',
                   'FAT', 'FEN', 'FIG', 'FIN', 'FIR', 'FLY', 'FOG', 'FOX',
                   'FUR', 'GAP', 'GAS', 'GEM', 'GIN', 'GNU', 'GOD', 'GUM',
                   'GUN', 'GUT', 'HAM', 'HAT', 'HAY', 'HEN', 'HIP', 'HOG',
                   'HOP', 'ICE', 'INN', 'INK', 'IRE', 'IVY', 'JAM', 'JAR',
                   'JAW', 'JET', 'JOY', 'KEY', 'KID', 'LAC', 'LAW', 'LEA',
                   'LEG', 'LID', 'OAK', 'OAR', 'OAT', 'OIL', 'ORE', 'OWL',
                   'PEA', 'PEN', 'PIE', 'PIG', 'PIN', 'PIT', 'PLY', 'POD',
                   'POT', 'PUN', 'PUS', 'RAM', 'RAT', 'RAY', 'RIB', 'RIM',
                   'ROD', 'ROE', 'RUG', 'RUM', 'RUN', 'RYE', 'SAP', 'SAW',
                   'SEA', 'SKI', 'SKY', 'SOD', 'SOW', 'SPA', 'SPY', 'SUM',
                   'SUN', 'TAR', 'TAX', 'TEA', 'TIN', 'TON', 'TOY', 'TUB',
                   'TUN', 'URN', 'USE', 'VAN', 'VAT', 'VIA', 'VOW', 'WAR',
                   'WAX', 'WAY', 'WEB', 'WIG', 'WIT', 'WOK', 'YAM', 'YEW',
                   'ZOO'}
    if len(c) <= 3 and c not in KNOWN_SHORT:
        return True
    return False


def is_false_positive_pattern(candidate: str, before: str, after: str) -> str | None:
    """
    Check for known false positive patterns. Returns reason string if FP, None if OK.
    """
    c = candidate.strip()
    cu = c.upper()

    # PLATE labels
    if re.match(r'^PLATE\s+[DXIVLCM\d]+', cu):
        return "plate_label"

    # Coptic glossary entries (very short, from 1860 v11)
    if len(c) <= 5 and re.match(r'^[A-Z]{2,5}$', c):
        # Check context for Coptic markers
        if 'Copt' in after or 'ⲁ' in after or 'ⲉ' in after:
            return "coptic_glossary"

    # Epitaph inscriptions
    epitaph_markers = ['epitaph', 'monument', 'hic depositum', 'to the memory',
                       'here lie', 'mortal remains', 'S.T.P.', 'S.T.D.']
    combined = (before + ' ' + after).lower()
    for marker in epitaph_markers:
        if marker in combined:
            return "epitaph"

    # Roman numeral only entries
    if re.match(r'^[IVXLCDM\s\.]+$', cu):
        return "roman_numeral"

    # GENUS labels (taxonomy section headers)
    if re.match(r'^GENUS\s+[IVXLCDM\d]', cu):
        return "genus_label"

    # CLASS labels
    if re.match(r'^CLASS\s+[IVXLCDM\d]', cu):
        return "class_label"

    # SECT/SECTION/CHAPTER/PART labels
    if re.match(r'^(SECT|SECTION|CHAPTER|CHAP|PART)\s', cu):
        return "section_label"

    # ORDER labels
    if re.match(r'^ORDER\s+[IVXLCDM\d]', cu):
        return "order_label"

    return None


def load_articles_by_file():
    """Load all parsed articles, indexed by source OCR filename.

    Also builds an (edition, volume) -> [articles] index for fallback matching
    when the candidate's OCR file is a duplicate (_dup2, _alt2, etc.) that
    doesn't have its own articles file.
    """
    by_file = defaultdict(list)
    by_ed_vol = defaultdict(list)

    for p in sorted(ARTICLES_DIR.glob("*.articles.jsonl")):
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                src = art.get('source_file', '')
                entry = {
                    'title': art['title'],
                    'char_start': art['char_start'],
                    'char_end': art['char_end'],
                    'edition_year': art['edition_year'],
                }
                by_file[src].append(entry)
                ey = art['edition_year']
                vol = art.get('volume', 0)
                by_ed_vol[(ey, vol)].append(entry)

    # Sort each file's articles by position
    for src in by_file:
        by_file[src].sort(key=lambda a: a['char_start'])
    for key in by_ed_vol:
        by_ed_vol[key].sort(key=lambda a: a['char_start'])

    return by_file, by_ed_vol


def get_vol_number(vol_str: str) -> int:
    """Extract volume number from 'v02' format."""
    m = re.match(r'v?(\d+)', vol_str)
    return int(m.group(1)) if m else 0


def find_surrounding_articles(articles_list, position):
    """
    Find the article before and after a given position.
    Returns (prev_article, next_article) or (None, None).
    """
    prev_art = None
    next_art = None

    for art in articles_list:
        if art['char_start'] <= position:
            prev_art = art
        elif art['char_start'] > position and next_art is None:
            next_art = art
            break

    return prev_art, next_art


def main():
    print("Loading parsed articles...")
    articles_by_file, articles_by_ed_vol = load_articles_by_file()
    total_articles = sum(len(v) for v in articles_by_file.values())
    print(f"  {total_articles:,} articles across {len(articles_by_file)} files")

    print("Loading candidates...")
    candidates = []
    with open(CANDIDATES_FILE) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    print(f"  {len(candidates)} candidates")

    # Process each candidate
    kept = []
    rejected = defaultdict(list)  # reason -> list of candidates

    for c in candidates:
        ocr_file = c['file']
        position = c['position']
        candidate_title = c['candidate']
        before_ctx = c.get('before', '')
        after_ctx = c.get('after', '')

        # 1. Check pattern-based false positives
        fp_reason = is_false_positive_pattern(candidate_title, before_ctx, after_ctx)
        if fp_reason:
            rejected[fp_reason].append(c)
            continue

        # 2. Check running header stubs
        if is_running_header_stub(candidate_title, after_ctx):
            rejected['running_header_stub'].append(c)
            continue

        # 3. Check alphabetical ordering against POSITIONALLY adjacent articles
        #
        # Strategy: Find the parsed articles immediately before and after this
        # candidate's position in the OCR text. A genuine missed article should
        # fit alphabetically between these positional neighbors.
        #
        # EXCEPTION: If the candidate is inside a mega-article (>30K chars) and
        # it's alphabetically far from the host (different first letter), it's
        # likely a genuine article the parser absorbed — keep it.

        # Get articles for this file, with ed_vol fallback
        arts = articles_by_file.get(ocr_file, [])
        if not arts:
            vol_num = get_vol_number(c['vol'])
            arts = articles_by_ed_vol.get((c['edition'], vol_num), [])

        if not arts:
            c['alpha_check'] = 'no_articles_found'
            kept.append(c)
            continue

        c_norm = normalize_for_sort(candidate_title)

        # Find positional neighbors
        pos_prev, pos_next = find_surrounding_articles(arts, position)
        containing_article = pos_prev  # Article whose body contains this position

        # Check: does the candidate fit alphabetically between its positional neighbors?
        in_positional_order = True
        if pos_prev and pos_next:
            in_positional_order = is_alpha_between(
                candidate_title, pos_prev['title'], pos_next['title'])
        elif pos_prev:
            p_norm = normalize_for_sort(pos_prev['title'])
            in_positional_order = c_norm >= p_norm
        elif pos_next:
            n_norm = normalize_for_sort(pos_next['title'])
            in_positional_order = c_norm <= n_norm

        # Exception: large-article absorption detection
        # If the candidate is inside a large article AND the candidate topic is
        # unrelated to the host article, it's likely a genuine article the parser
        # swallowed. But if the candidate shares words with the host title, it's
        # probably a sub-heading (e.g., "PRACTICE OF NAVIGATION" inside NAVIGATION).
        inside_large = False
        if containing_article and not in_positional_order:
            host_size = containing_article.get('char_end', 0) - containing_article.get('char_start', 0)
            if host_size > 10000:
                # Check if candidate shares significant words with host
                host_words = set(normalize_for_sort(containing_article['title']))
                # Actually compare word-level overlap
                host_title_words = set(containing_article['title'].upper().split())
                cand_title_words = set(candidate_title.upper().split())
                # Remove common small words
                stop_words = {'OF', 'THE', 'AND', 'IN', 'ON', 'AT', 'TO', 'A', 'AN',
                              'OR', 'BY', 'FOR', 'WITH', 'FROM', 'IS', 'ARE', 'WAS'}
                host_sig = host_title_words - stop_words
                cand_sig = cand_title_words - stop_words
                shared = host_sig & cand_sig

                if not shared:
                    # No word overlap with host — likely genuine absorbed article
                    inside_large = True
                # else: shares words with host — likely a sub-heading, stay rejected

        if in_positional_order or inside_large:
            c['alpha_check'] = 'in_order' if in_positional_order else 'inside_large'
            if pos_prev:
                c['prev_article'] = pos_prev['title']
            if pos_next:
                c['next_article'] = pos_next['title']
            if containing_article:
                c['containing_article'] = containing_article['title']
            kept.append(c)
        else:
            if pos_prev:
                c['prev_article'] = pos_prev['title']
            if pos_next:
                c['next_article'] = pos_next['title']
            if containing_article:
                c['containing_article'] = containing_article['title']
            rejected['out_of_alpha_order'].append(c)

    # Deduplicate kept candidates by (edition, candidate title)
    # Keep the one with the most context
    seen = {}
    for c in kept:
        key = (c['edition'], c['candidate'])
        if key not in seen or len(c.get('after', '')) > len(seen[key].get('after', '')):
            seen[key] = c
    deduped = sorted(seen.values(), key=lambda c: (c['edition'], c['candidate']))

    # Write filtered results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for c in deduped:
            f.write(json.dumps(c) + '\n')

    # Separate into high-confidence (in_order) and medium-confidence (inside_large)
    high_conf = [c for c in deduped if c.get('alpha_check') == 'in_order']
    med_conf = [c for c in deduped if c.get('alpha_check') == 'inside_large']

    # Write report
    with open(REPORT_FILE, 'w') as f:
        f.write("MISSED HEADINGS — FILTERED RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Input:  {len(candidates)} candidates\n")
        total_rejected = sum(len(v) for v in rejected.values())
        f.write(f"Rejected: {total_rejected}\n")
        f.write(f"Kept (deduped): {len(deduped)} unique (edition, title) pairs\n")
        f.write(f"  HIGH confidence (alphabetically in order):  {len(high_conf)}\n")
        f.write(f"  MEDIUM confidence (inside large article):   {len(med_conf)}\n\n")

        f.write("REJECTION REASONS:\n")
        for reason in sorted(rejected.keys()):
            items = rejected[reason]
            f.write(f"  {reason}: {len(items)} rejected\n")
            for c in items[:5]:
                prev = c.get('prev_article', '?')
                nxt = c.get('next_article', '?')
                f.write(f"    {c['edition']} {c['vol']} | {c['candidate']}")
                if reason == 'out_of_alpha_order':
                    f.write(f"  (between {prev} and {nxt})")
                f.write("\n")
            if len(items) > 5:
                f.write(f"    ... and {len(items) - 5} more\n")

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("HIGH CONFIDENCE — Alphabetically in order with neighbors\n")
        f.write("These candidates fit between the preceding and following parsed\n")
        f.write("articles alphabetically. Very likely genuine missed articles.\n")
        f.write("=" * 70 + "\n\n")

        by_edition = defaultdict(list)
        for c in high_conf:
            by_edition[c['edition']].append(c)

        for ey in sorted(by_edition.keys()):
            items = by_edition[ey]
            f.write(f"\n--- {ey} ({len(items)} candidates) ---\n")
            for c in sorted(items, key=lambda x: normalize_for_sort(x['candidate'])):
                prev = c.get('prev_article', '?')
                nxt = c.get('next_article', '?')
                f.write(f"  {c['vol']:>3s} {c['pct']:5.1f}% | {c['candidate']:<45s}")
                f.write(f" between({prev}..{nxt})")
                f.write("\n")
                f.write(f"           | ...{c.get('before', '')[-30:]}|||{c.get('after', '')[:50]}\n")

        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("MEDIUM CONFIDENCE — Inside large articles, no word overlap\n")
        f.write("These candidates are inside large parsed articles (>10K chars)\n")
        f.write("and are alphabetically out of order with neighbors. They don't\n")
        f.write("share words with the host article title, suggesting they may be\n")
        f.write("genuine articles absorbed by the parser. Verify before using.\n")
        f.write("=" * 70 + "\n\n")

        by_edition2 = defaultdict(list)
        for c in med_conf:
            by_edition2[c['edition']].append(c)

        for ey in sorted(by_edition2.keys()):
            items = by_edition2[ey]
            f.write(f"\n--- {ey} ({len(items)} candidates) ---\n")
            for c in sorted(items, key=lambda x: normalize_for_sort(x['candidate'])):
                host = c.get('containing_article', '?')
                f.write(f"  {c['vol']:>3s} {c['pct']:5.1f}% | {c['candidate']:<45s}")
                f.write(f" inside({host})")
                f.write("\n")
                f.write(f"           | ...{c.get('before', '')[-30:]}|||{c.get('after', '')[:50]}\n")

    print(f"\n{'=' * 70}")
    print(f"Results:")
    print(f"  Input:     {len(candidates)} candidates")
    total_rejected = sum(len(v) for v in rejected.values())
    print(f"  Rejected:  {total_rejected}")
    for reason in sorted(rejected.keys()):
        print(f"    {reason}: {len(rejected[reason])}")
    print(f"  Kept:      {len(kept)} (before dedup)")
    print(f"  Deduped:   {len(deduped)} unique (edition, title)")
    print(f"\nWritten to: {OUTPUT_FILE}")
    print(f"Report:     {REPORT_FILE}")


if __name__ == '__main__':
    main()
