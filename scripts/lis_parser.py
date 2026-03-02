"""
LIS-based Britannica article extraction.

Extracts articles from OCR text by treating headword detection as a
Longest Increasing Subsequence (LIS) problem: encyclopedia articles appear
in strict alphabetical order on ascending positions, so the true headwords
form the longest alphabetically-increasing subsequence of all candidates.

Replaces the LLM-based Phases 1-3.5 with a single fast pass (~250ms/volume).
"""

import bisect
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    INPUT_DIR, ARTICLES_DIR, DEDUP_MANIFEST, OCR_MANIFEST,
    HEADWORD_DICT_PATH, DOCS_OLD_DIR, FULL_OCR_EDITIONS,
    SUPPLEMENTARY_HEADINGS_PATH,
    ensure_dirs,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

def normalize_sort_key(headword: str) -> str:
    """Create a sort key handling 18th-century alphabet conventions.

    - Uppercase everything
    - U -> V, I -> J (18th-century alphabet treats these as same letter)
    - Strip accents, hyphens, apostrophes
    - Collapse whitespace
    """
    key = headword.upper()
    key = key.replace('U', 'V').replace('I', 'J')
    key = unicodedata.normalize('NFKD', key)
    key = key.encode('ASCII', 'ignore').decode('ASCII')
    key = re.sub(r"['\-]", '', key)
    key = re.sub(r'\s+', ' ', key).strip()
    return key


# ---------------------------------------------------------------------------
# 2. Candidate generation
# ---------------------------------------------------------------------------

@dataclass
class HeadingCandidate:
    headword: str                       # Original text (uppercased)
    sort_key: str                       # Normalized for alphabetical comparison
    char_start: int                     # Position in source text
    char_end: int                       # End of heading match
    pattern: str                        # 'article', 'treatise', 'crossref', 'titlecase'
    crossref_target: Optional[str] = None
    confidence: float = 1.0


# --- Regex patterns ---

ARTICLE_PATTERN = re.compile(
    r'(?:^|\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?,\s+',
    re.MULTILINE,
)

TREATISE_PATTERN = re.compile(
    r'(?:\n\n|---\s*\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.\s*\n',
    re.MULTILINE,
)

TREATISE_COMMA_NL = re.compile(
    r'(?:\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?,\s*\n(?=[A-Z\n])',
    re.MULTILINE,
)

CROSSREF_PATTERN = re.compile(
    r'(?:\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.?\s+[Ss]ee\s+([A-Z][A-Za-z\-\' ]+?)\.?\s*(?:\n|$)',
    re.MULTILINE,
)

TITLECASE_PATTERN = re.compile(
    r'(?:\n\n)(?:\d{4}\.?\s*\n)?([A-Z][a-z]+(?:[\s\-][A-Za-z]+)*),\s+(?:a|an|the|in|one|or)\s',
    re.MULTILINE,
)

# Gap A: ALL-CAPS + period + inline text (not treatise-style period+newline)
# Catches: INDIA. The general name..., CHEMISTRY. This science...
ARTICLE_PERIOD_PATTERN = re.compile(
    r'(?:^|\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.[ \t]+(?=[A-Za-z])',
    re.MULTILINE,
)

# Gap B: ALL-CAPS + open parenthesis (qualifier instead of comma)
# Catches: AFRICA (according to Bochart..., ALPS (a word signifying...
ARTICLE_PAREN_PATTERN = re.compile(
    r'(?:^|\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\s+(?=\()',
    re.MULTILINE,
)

# --- V2: Single-newline variants (confidence 0.7) ---
# OCR often produces only \n where the original has \n\n.
# These patterns catch headwords after a single newline that the strict
# double-newline patterns miss, at lower confidence. The LIS algorithm
# filters out noise automatically because false positives break the
# alphabetical sequence.

ARTICLE_PATTERN_SINGLE_NL = re.compile(
    r'(?<!\n)\n(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?,\s+[a-z]',
    re.MULTILINE,
)

TREATISE_PATTERN_SINGLE_NL = re.compile(
    r'(?<!\n)\n(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.\s*\n',
    re.MULTILINE,
)

CROSSREF_PATTERN_SINGLE_NL = re.compile(
    r'(?<!\n)\n(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.?\s+[Ss]ee\s+([A-Z][A-Za-z\-\' ]+?)\.?\s*(?:\n|$)',
    re.MULTILINE,
)

ARTICLE_PERIOD_PATTERN_SINGLE_NL = re.compile(
    r'(?<!\n)\n(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.[ \t]+(?=[A-Za-z])',
    re.MULTILINE,
)

ARTICLE_PAREN_PATTERN_SINGLE_NL = re.compile(
    r'(?<!\n)\n(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\s+(?=\()',
    re.MULTILINE,
)

# --- Blacklists ---

BLACKLIST = {
    'CHAP', 'CHAPTER', 'SECT', 'SECTION', 'PART', 'ORDER', 'THEOREM',
    'CLASS', 'BOOK', 'PLATE', 'PLATES', 'FIG', 'FIGURE', 'FINIS',
    'DIRECTIONS', 'ILLUSTRATED', 'VOLUMES', 'TABLE', 'APPENDIX',
    'INDEX', 'CONTENTS', 'ERRATA', 'PREFACE', 'ADVERTISEMENT',
    'INTRODUCTION', 'SUBSCRIBERS', 'PRINTED',
    'SUPPLEMENT', 'VOL', 'VOLUME', 'DEFINITION', 'DEFINITIONS',
    'PROPOSITION', 'PROPOSITIONS', 'COROLLARY', 'LEMMA', 'AXIOM',
    'PROBLEM', 'SCHOLIUM', 'POSTULATE',
}

FALSE_POSITIVE_WORDS = {
    'THUS', 'MORE', 'MOST', 'MUCH', 'MANY', 'VERY', 'ALSO', 'ONLY', 'OTHER',
    'SAME', 'UPON', 'INTO', 'OVER', 'AFTER', 'BEFORE', 'ABOUT', 'UNDER',
    'BETWEEN', 'THROUGH', 'DURING', 'WITHOUT', 'HOWEVER', 'BEING', 'EVERY',
    'EACH', 'EITHER', 'NEITHER', 'BOTH', 'SHOULD', 'COULD', 'MIGHT', 'SHALL',
    'STILL', 'WHILE', 'SINCE', 'UNTIL', 'ALTHOUGH', 'BECAUSE', 'THEREFORE',
    'WHETHER', 'ANOTHER', 'CALLED', 'GIVEN', 'GREAT', 'HAVING',
    'LIKE', 'LONG', 'MADE', 'MAKE', 'MUST', 'NAME', 'NEAR', 'NEVER',
    'NEXT', 'OFTEN', 'ONCE', 'PART', 'PLACE', 'RATHER', 'SEVERAL', 'SMALL',
    'SOME', 'SOMETIMES', 'TAKEN', 'THEN', 'THOUGH', 'TRUE', 'TURN', 'USED',
    'WHEN', 'WHERE', 'WHICH', 'WITH', 'WITHIN', 'YOUR',
}

OVERSIZED_FALSE_POSITIVES = {
    'IF', 'AS', 'BE', 'ON', 'OR', 'SO', 'AN', 'AT', 'BY', 'DO', 'GO',
    'HE', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'TO', 'UP', 'US',
    'WE', 'HP', 'FE', 'ORA', 'PAKS',
}

ROMAN_NUMERAL = re.compile(r'^[IVXLCDM]+\.?$')

FRONT_MATTER_PATTERNS = re.compile(
    r'^(ENCYCLOP[AÆ]?DIA|BRITANNICA|OR|DICTIONARY|OF\b|ARTS\b|SCIENCES\b|'
    r'AND MISCELLANEOUS|CONSTRUCTED|COMPREHENDING|INCLUDING|TOGETHER|'
    r'COMPILED|ILLUSTRATED|VOL\.?|VOLUME|PRINTED|MDCC|'
    r'INDOCTI|ENTERED|THE THIRD|THE FOURTH|THE FIFTH|THE SIXTH|THE SEVENTH|THE EIGHTH|'
    r'THE FIRST|THE SECOND)',
    re.IGNORECASE,
)


def _is_blacklisted(headword: str) -> bool:
    """Check if a headword is blacklisted (section headings, front matter, etc.)."""
    hw_upper = headword.upper().strip()
    first_word = hw_upper.split()[0] if ' ' in hw_upper else hw_upper
    if first_word in BLACKLIST or first_word in FALSE_POSITIVE_WORDS:
        return True
    if ROMAN_NUMERAL.match(hw_upper) and len(hw_upper) <= 6:
        return True
    if len(headword.strip()) < 2:
        return True
    if FRONT_MATTER_PATTERNS.match(hw_upper):
        return True
    return False


def generate_candidates(
    text: str,
    edition_year: int,
    index_headwords: set[str] | None = None,
) -> list[HeadingCandidate]:
    """Extract all heading candidates from volume text using multiple regex patterns.

    Returns candidates sorted by char_start (positional order).
    """
    candidates: list[HeadingCandidate] = []
    seen_positions: set[int] = set()

    def add_candidate(headword, start, end, pattern, crossref_target=None, confidence=1.0):
        # Dedup: skip if within 20 chars of existing candidate
        for s in seen_positions:
            if abs(start - s) < 20:
                return
        if _is_blacklisted(headword):
            return

        seen_positions.add(start)
        hw_upper = re.sub(r'\s+', ' ', headword.upper()).strip()
        candidates.append(HeadingCandidate(
            headword=hw_upper,
            sort_key=normalize_sort_key(hw_upper),
            char_start=start,
            char_end=end,
            pattern=pattern,
            crossref_target=crossref_target,
            confidence=confidence,
        ))

    # Pattern 1: ALL-CAPS + comma (most common)
    for m in ARTICLE_PATTERN.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article')

    # Pattern 2: ALL-CAPS + period + newline (treatises)
    for m in TREATISE_PATTERN.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'treatise')

    # Pattern 3: ALL-CAPS + comma + newline (treatise hybrid)
    for m in TREATISE_COMMA_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'treatise')

    # Pattern 4: Cross-references ("HEADWORD. See TARGET")
    for m in CROSSREF_PATTERN.finditer(text):
        add_candidate(
            m.group(1), m.start(), m.end(), 'crossref',
            crossref_target=m.group(2).strip(),
        )

    # Pattern 5: Title Case (validated against headword dictionary/index)
    # Gap C: OCR sometimes renders headwords in titlecase instead of ALL-CAPS.
    # Safe for all editions when validated against the known headword set.
    if index_headwords:
        for m in TITLECASE_PATTERN.finditer(text):
            hw_upper = m.group(1).upper()
            if hw_upper in index_headwords:
                add_candidate(
                    m.group(1), m.start(), m.end(), 'titlecase',
                    confidence=0.8,
                )

    # Pattern 6 (Gap A): ALL-CAPS + period + inline text
    for m in ARTICLE_PERIOD_PATTERN.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article_period')

    # Pattern 7 (Gap B): ALL-CAPS + open parenthesis
    for m in ARTICLE_PAREN_PATTERN.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article_paren')

    # V2 Patterns: Single-newline variants (lower confidence)
    for m in ARTICLE_PATTERN_SINGLE_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article_snl', confidence=0.7)

    for m in TREATISE_PATTERN_SINGLE_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'treatise_snl', confidence=0.7)

    for m in CROSSREF_PATTERN_SINGLE_NL.finditer(text):
        add_candidate(
            m.group(1), m.start(), m.end(), 'crossref_snl',
            crossref_target=m.group(2).strip(),
            confidence=0.7,
        )

    for m in ARTICLE_PERIOD_PATTERN_SINGLE_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article_period_snl', confidence=0.7)

    for m in ARTICLE_PAREN_PATTERN_SINGLE_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'article_paren_snl', confidence=0.7)

    candidates.sort(key=lambda c: c.char_start)
    return candidates


# ---------------------------------------------------------------------------
# 3. Front matter stripping
# ---------------------------------------------------------------------------

def strip_front_matter(
    candidates: list[HeadingCandidate],
    text: str,
) -> list[HeadingCandidate]:
    """Remove candidates in front matter region.

    Strategy: find the first short (2-8 char) alphabetic headword with
    substantial following text.  Everything before it is front matter.
    """
    if not candidates:
        return candidates

    article_start_pos = 0
    for c in candidates:
        hw = c.headword.replace("'", "").replace("-", "")
        if hw.isalpha() and 2 <= len(hw) <= 8:
            text_after = text[c.char_end:c.char_end + 200].strip()
            if len(text_after) > 30:
                article_start_pos = c.char_start
                break

    return [c for c in candidates if c.char_start >= article_start_pos]


# ---------------------------------------------------------------------------
# 4. Back matter stripping
# ---------------------------------------------------------------------------

BACK_MATTER_SIGNALS = re.compile(
    r'\n\n(?:FINIS\b(?![\w\-])|DIRECTIONS\s+TO\s+THE\s+BINDER|END\s+OF\s+VOL)',
    re.IGNORECASE,
)


def strip_back_matter(
    candidates: list[HeadingCandidate],
    text: str,
) -> list[HeadingCandidate]:
    """Remove candidates that fall after back-matter markers."""
    m = BACK_MATTER_SIGNALS.search(text)
    if m:
        cutoff = m.start()
        return [c for c in candidates if c.char_start < cutoff]
    return candidates


# ---------------------------------------------------------------------------
# 5. LIS algorithm — patience sorting, O(n log n)
# ---------------------------------------------------------------------------

def longest_increasing_subsequence(
    candidates: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    """Find the longest increasing subsequence by (sort_key, char_start).

    Uses patience sorting for O(n log n) performance.
    Equal sort_keys are allowed when char_start is increasing (multi-sense entries).
    """
    if not candidates:
        return []

    n = len(candidates)
    keys = [(c.sort_key, c.char_start) for c in candidates]

    tail_keys: list[tuple[str, int]] = []
    tail_indices: list[int] = []
    parent = [-1] * n

    for i in range(n):
        pos = bisect.bisect_right(tail_keys, keys[i])

        if pos == len(tail_keys):
            tail_keys.append(keys[i])
            tail_indices.append(i)
        else:
            tail_keys[pos] = keys[i]
            tail_indices[pos] = i

        if pos > 0:
            parent[i] = tail_indices[pos - 1]

    # Reconstruct
    result_indices: list[int] = []
    idx = tail_indices[-1]
    while idx != -1:
        result_indices.append(idx)
        idx = parent[idx]
    result_indices.reverse()

    return [candidates[i] for i in result_indices]


# ---------------------------------------------------------------------------
# 6. Recovery pass
# ---------------------------------------------------------------------------

def recovery_pass(
    accepted: list[HeadingCandidate],
    rejected: list[HeadingCandidate],
    index_headwords: set[str] | None = None,
) -> list[HeadingCandidate]:
    """Re-insert rejected candidates that were only barely out of sequence.

    A candidate is recoverable if:
    1. It's in the cross-edition/index headword set, OR
    2. It fits alphabetically between its positional neighbors.
    """
    if not rejected:
        return accepted

    accepted_positions = [a.char_start for a in accepted]

    recoverable: list[HeadingCandidate] = []
    for r in rejected:
        # Check 1: In the headword index?
        if index_headwords and r.headword in index_headwords:
            # Still must fit between neighbors
            insert_pos = bisect.bisect_left(accepted_positions, r.char_start)
            if 0 < insert_pos < len(accepted):
                prev_key = accepted[insert_pos - 1].sort_key
                next_key = accepted[insert_pos].sort_key
                if prev_key <= r.sort_key <= next_key:
                    recoverable.append(r)
            elif insert_pos == 0 and len(accepted) > 0:
                if r.sort_key <= accepted[0].sort_key:
                    recoverable.append(r)
            elif insert_pos == len(accepted) and len(accepted) > 0:
                if r.sort_key >= accepted[-1].sort_key:
                    recoverable.append(r)
            continue

        # Check 2: Fits between positional neighbors alphabetically?
        insert_pos = bisect.bisect_left(accepted_positions, r.char_start)
        if 0 < insert_pos < len(accepted):
            prev_key = accepted[insert_pos - 1].sort_key
            next_key = accepted[insert_pos].sort_key
            if prev_key <= r.sort_key <= next_key:
                recoverable.append(r)

    if not recoverable:
        return accepted

    log.info(f"    Recovered {len(recoverable)} candidates")
    combined = accepted + recoverable
    combined.sort(key=lambda c: c.char_start)
    return combined


# ---------------------------------------------------------------------------
# 7. Range validation (post-LIS sanity check)
# ---------------------------------------------------------------------------

def validate_range(
    accepted: list[HeadingCandidate],
    volume_range: str | None,
) -> tuple[list[HeadingCandidate], str]:
    """Post-LIS: validate headwords fall within the volume's stated range.

    Does not remove candidates — just logs warnings and returns the effective range.
    """
    if not accepted:
        return accepted, 'EMPTY'

    first_hw = accepted[0].headword
    last_hw = accepted[-1].headword
    inferred = f"{first_hw[:3]}-{last_hw[:3]}"

    if not volume_range or volume_range.lower() in ('none', 'unknown'):
        return accepted, inferred

    parts = volume_range.split('-')
    if len(parts) != 2:
        return accepted, inferred

    range_start = normalize_sort_key(parts[0])
    range_end = normalize_sort_key(parts[1]) + 'ZZZZ'

    violations = [c for c in accepted if c.sort_key < range_start or c.sort_key > range_end]
    if violations:
        log.warning(f"    {len(violations)} headwords outside stated range {volume_range}")
        for v in violations[:5]:
            log.warning(f"      {v.headword} (sort_key={v.sort_key})")

    return accepted, volume_range


# ---------------------------------------------------------------------------
# 7b. Running header detection (V2)
# ---------------------------------------------------------------------------

def detect_running_headers(
    accepted: list[HeadingCandidate],
    all_single_nl: list[HeadingCandidate],
    text: str,
) -> list[HeadingCandidate]:
    """Detect running headers and missed boundaries inside accepted articles.

    For each single-newline candidate that falls inside an accepted article's
    text span:
    - If its headword matches the article's own title → running header, skip.
    - If it's a different headword that fits alphabetically between the
      article's neighbors → missed boundary, insert it.

    Returns the updated accepted list with new boundaries inserted.
    """
    if not all_single_nl or not accepted:
        return accepted

    # Build a set of accepted positions for quick lookup
    accepted_starts = {c.char_start for c in accepted}

    # Build position-sorted list for neighbor checks
    new_boundaries: list[HeadingCandidate] = []

    for i, article in enumerate(accepted):
        art_start = article.char_end  # text starts after heading
        art_end = accepted[i + 1].char_start if i + 1 < len(accepted) else len(text)

        # Skip short articles — unlikely to contain missed headwords
        if art_end - art_start < 500:
            continue

        article_key = article.sort_key
        next_key = accepted[i + 1].sort_key if i + 1 < len(accepted) else 'ZZZZZZ'

        for snl in all_single_nl:
            # Only consider candidates inside this article's text span
            if snl.char_start <= art_start or snl.char_start >= art_end:
                continue
            # Skip if already accepted
            if snl.char_start in accepted_starts:
                continue
            # Running header: same headword as article title → skip
            if snl.sort_key == article_key:
                continue
            # Missed boundary: different headword, fits alphabetically
            if article_key <= snl.sort_key <= next_key:
                new_boundaries.append(snl)

    if not new_boundaries:
        return accepted

    # Deduplicate by position (keep first occurrence)
    seen_pos = {c.char_start for c in accepted}
    unique_new = []
    for nb in new_boundaries:
        if nb.char_start not in seen_pos:
            unique_new.append(nb)
            seen_pos.add(nb.char_start)

    log.info(f"    Running header scan: {len(unique_new)} missed boundaries recovered")
    combined = accepted + unique_new
    combined.sort(key=lambda c: c.char_start)
    return combined


# ---------------------------------------------------------------------------
# 7c. Dictionary-guided candidate injection (V2)
# ---------------------------------------------------------------------------

def dictionary_guided_injection(
    accepted: list[HeadingCandidate],
    text: str,
    edition_year: int,
    headword_dict: dict[str, dict] | None = None,
) -> list[HeadingCandidate]:
    """Search OCR text for known headwords that regex missed.

    For each headword in the dictionary that should exist in this edition
    but wasn't found by regex patterns, search the OCR text for the headword
    string.  If found at a position that fits alphabetically between neighbors,
    add it as a candidate.

    Only considers headwords that:
    - Appear in 2+ independent sources (LLM, Gemini, docs_old)
    - Are expected in this edition_year
    - Alphabetically within the volume's range (first to last accepted headword)
    - Are not already in the accepted list
    """
    if not headword_dict or not accepted or len(accepted) < 2:
        return accepted

    edition_str = str(edition_year)
    accepted_norms = {c.sort_key for c in accepted}

    # Pre-filter: only look for headwords within this volume's alphabetical range
    range_start = accepted[0].sort_key
    range_end = accepted[-1].sort_key

    # Build list of headwords we should look for (filtered by range)
    targets = []
    for norm_key, entry in headword_dict.items():
        if norm_key in accepted_norms:
            continue
        if not (range_start <= norm_key <= range_end):
            continue
        if edition_str not in entry.get('editions', []):
            continue
        if entry.get('source_count', 0) < 2:
            continue
        hw = entry['headword']
        if len(hw) > 40 or len(hw) < 2:
            continue
        targets.append((norm_key, hw))

    if not targets:
        return accepted

    # Build position index for quick neighbor lookup
    accepted_positions = [a.char_start for a in accepted]
    accepted_keys = [a.sort_key for a in accepted]
    accepted_pos_set = set(accepted_positions)

    new_candidates = []
    # Position-preserving uppercase: leave chars whose .upper() changes length
    # (e.g. Greek ῆ → Η͂ = 2 chars) so len(text_upper) == len(text)
    text_upper = ''.join(
        c.upper() if len(c.upper()) == 1 else c
        for c in text
    )

    # Fast string search: use text.find() instead of regex per headword
    for norm_key, hw in targets:
        # Search for "\nHEADWORD" in uppercase text (fast string find)
        search_str = '\n' + hw
        start = 0
        found = False
        while not found:
            idx = text_upper.find(search_str, start)
            if idx == -1:
                break
            pos = idx + 1  # skip the \n

            # Verify next char is a delimiter (, . space or newline)
            end_pos = pos + len(hw)
            if end_pos < len(text) and text[end_pos] not in ',. \t\n;:':
                start = idx + 1
                continue

            # Check if this position fits alphabetically
            insert_idx = bisect.bisect_left(accepted_positions, pos)
            if insert_idx <= 0 or insert_idx >= len(accepted):
                start = idx + 1
                continue

            prev_key = accepted_keys[insert_idx - 1]
            next_key = accepted_keys[insert_idx]
            if prev_key <= norm_key <= next_key:
                # Check it's not too close to an existing candidate
                too_close = False
                for p in accepted_positions[max(0, insert_idx-1):insert_idx+2]:
                    if abs(pos - p) < 20:
                        too_close = True
                        break
                if not too_close:
                    new_candidates.append(HeadingCandidate(
                        headword=hw,
                        sort_key=norm_key,
                        char_start=pos,
                        char_end=pos + len(hw) + 2,
                        pattern='dict_guided',
                        confidence=0.6,
                    ))
                    found = True

            start = idx + 1

    if not new_candidates:
        return accepted

    log.info(f"    Dictionary-guided injection: {len(new_candidates)} headwords recovered "
             f"(searched {len(targets)} targets)")
    combined = accepted + new_candidates
    combined.sort(key=lambda c: c.char_start)
    return combined


# ---------------------------------------------------------------------------
# 7d. Supplementary heading injection (pre-validated missed headings)
# ---------------------------------------------------------------------------

def supplementary_injection(
    accepted: list[HeadingCandidate],
    source_file: str,
    supplementary_path: Path | None = None,
) -> list[HeadingCandidate]:
    """Inject pre-validated missed headings from Gemini classification + alpha filtering.

    These are single-newline ALL CAPS headings that the parser missed because it
    requires \\n\\n.  They were classified by Gemini and filtered by alphabetical
    order analysis, so no further validation is needed here — just dedup against
    existing candidates.
    """
    if supplementary_path is None:
        supplementary_path = SUPPLEMENTARY_HEADINGS_PATH

    if not supplementary_path.exists():
        return accepted

    # Load entries for this source file
    entries = []
    with open(supplementary_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('file') == source_file:
                entries.append(entry)

    if not entries:
        return accepted

    # Build position set for dedup (skip if within 20 chars of existing)
    existing_positions = [c.char_start for c in accepted]

    new_candidates = []
    for entry in entries:
        pos = entry['position']
        candidate = entry['candidate']

        # Skip if too close to an existing candidate
        too_close = False
        for ep in existing_positions:
            if abs(pos - ep) < 20:
                too_close = True
                break
        if too_close:
            continue

        new_candidates.append(HeadingCandidate(
            headword=candidate,
            sort_key=normalize_sort_key(candidate),
            char_start=pos,
            char_end=pos + len(candidate) + 2,
            pattern='supplementary',
            confidence=0.8,
        ))

    if not new_candidates:
        return accepted

    log.info(f"    Supplementary injection: {len(new_candidates)} headwords from "
             f"{len(entries)} candidates for {source_file}")
    combined = accepted + new_candidates
    combined.sort(key=lambda c: c.char_start)
    return combined


# ---------------------------------------------------------------------------
# 7e. Headword dictionary loader
# ---------------------------------------------------------------------------

def load_headword_dictionary(path: Path | None = None) -> dict[str, dict] | None:
    """Load the consolidated headword dictionary.

    Returns: {normalized_key: {headword, sources, editions, source_count, edition_count}}
    """
    if path is None:
        path = HEADWORD_DICT_PATH
    if not path.exists():
        log.warning(f"Headword dictionary not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)
    log.info(f"Loaded headword dictionary: {len(data)} entries")
    return data


def headword_dict_to_index(
    headword_dict: dict[str, dict],
    edition_year: int | None = None,
    min_sources: int = 1,
) -> set[str]:
    """Convert headword dictionary to an index_headwords set.

    If edition_year is given, only include headwords known to exist in that edition.
    min_sources filters to headwords confirmed by N+ independent sources.
    """
    result = set()
    edition_str = str(edition_year) if edition_year else None

    for norm_key, entry in headword_dict.items():
        if entry.get('source_count', 0) < min_sources:
            continue
        if edition_str and edition_str not in entry.get('editions', []):
            continue
        result.add(entry['headword'])
    return result


# ---------------------------------------------------------------------------
# 7e. docs_old fallback for volumes without OCR
# ---------------------------------------------------------------------------

def load_docs_old_articles(
    edition_year: int,
    headword_dict: dict[str, dict] | None = None,
) -> list[dict]:
    """Load articles from docs_old for volumes not covered by OCR files.

    Returns articles in the same format as extract_articles() output.
    Filters out garbled headwords (>40 chars) and known false positives.
    """
    import os

    # Map edition_year to docs_old directory name and edition_name
    edition_map = {
        1771: ('1771', '1st'), 1778: ('1778', '2nd'), 1797: ('1797', '3rd'),
        1810: ('1810', '4th'), 1815: ('1815', '5th'), 1823: ('1823', '6th'),
        1842: ('1842', '7th'), 1860: ('1860', '8th'),
    }
    if edition_year not in edition_map:
        return []
    dir_name, edition_name = edition_map[edition_year]

    docs_dir = DOCS_OLD_DIR / dir_name / "data"
    if not docs_dir.exists():
        return []

    articles = []
    for f in sorted(os.listdir(docs_dir)):
        if not f.endswith('.json') or '_corrected' in f or '_original' in f:
            continue
        # Skip vol0 (master/combined list)
        vol_match = re.match(r'vol(\d+)', f.replace('.json', ''))
        if not vol_match or vol_match.group(1) == '0':
            continue
        vol_num = int(vol_match.group(1))

        data = json.load(open(docs_dir / f))
        for i, a in enumerate(data):
            hw = a.get('h', '').strip()
            if not hw or len(hw) < 2 or len(hw) > 40:
                continue
            text = a.get('t', '').strip()
            if not text:
                continue

            article_id = f"eb_{edition_name}_{edition_year}_v{vol_num:02d}_do{i+1:04d}"
            articles.append({
                'article_id': article_id,
                'title': hw.upper(),
                'edition': edition_name,
                'edition_year': edition_year,
                'volume': vol_num,
                'source_file': f'docs_old/{dir_name}/data/{f}',
                'type': 'article',
                'char_start': 0,
                'char_end': len(text),
                'text': text,
                'word_count': len(text.split()),
                'paragraph_count': text.count('\n\n') + 1,
                'keywords': None,
                'author_attribution': None,
                'target': None,
                'subsections': [],
                'lis_confidence': 0.5,
                'heading_pattern': 'docs_old_fallback',
                'source': 'docs_old',
            })

    return articles


# ---------------------------------------------------------------------------
# 8. Text extraction — build articles from accepted headwords
# ---------------------------------------------------------------------------

CROSSREF_TEXT_LIMIT = 200

def extract_articles(
    accepted: list[HeadingCandidate],
    text: str,
    edition_name: str,
    edition_year: int,
    volume: int,
    source_file: str,
) -> list[dict]:
    """Slice text between accepted headword positions to produce articles."""
    articles: list[dict] = []
    overflow_text = ''

    for i, candidate in enumerate(accepted):
        raw_start = candidate.char_end
        raw_end = accepted[i + 1].char_start if i + 1 < len(accepted) else len(text)
        is_crossref = candidate.pattern in ('crossref', 'crossref_snl')

        if is_crossref:
            article_type = 'cross_reference'
            capped_end = min(raw_start + CROSSREF_TEXT_LIMIT, raw_end)
            article_text = text[raw_start:capped_end].strip()
            overflow_text = text[capped_end:raw_end] if capped_end < raw_end else ''
        else:
            article_type = 'article'
            base_text = text[raw_start:raw_end]
            if overflow_text:
                article_text = (overflow_text + base_text).strip()
                overflow_text = ''
            else:
                article_text = base_text.strip()

        article_id = f"eb_{edition_name}_{edition_year}_v{volume:02d}_{i + 1:04d}"

        articles.append({
            'article_id': article_id,
            'title': candidate.headword,
            'edition': edition_name,
            'edition_year': edition_year,
            'volume': volume,
            'source_file': source_file,
            'type': article_type,
            'char_start': candidate.char_start,
            'char_end': raw_end,
            'text': article_text,
            'word_count': len(article_text.split()),
            'paragraph_count': article_text.count('\n\n') + 1,
            'keywords': None,
            'author_attribution': None,
            'target': candidate.crossref_target,
            'subsections': [],
            'lis_confidence': candidate.confidence,
            'heading_pattern': candidate.pattern,
        })

    return articles


# ---------------------------------------------------------------------------
# 8a. Mega-article splitter
# ---------------------------------------------------------------------------

def split_mega_articles(
    articles: list[dict],
    text: str,
    headword_dict: dict[str, dict] | None = None,
    threshold: int = 50000,
) -> list[dict]:
    """Split mega-articles that have swallowed subsequent entries.

    For each article with > threshold words, search its body text for known
    dictionary headwords that alphabetically belong in the gap between this
    article and the next.  If found at line boundaries, split the article.

    Handles cases like SPAHIS swallowing SPAIN (1842), TZULIM swallowing
    UNITED STATES (1842/1860) where OCR corrupted or omitted the heading.
    """
    if not headword_dict or not articles:
        return articles

    result = []
    for i, art in enumerate(articles):
        if art['word_count'] < threshold or art['type'] != 'article':
            result.append(art)
            continue

        # Get alphabetical range: this article's sort_key to next article's
        art_key = normalize_sort_key(art['title'])
        if i + 1 < len(articles):
            next_key = normalize_sort_key(articles[i + 1]['title'])
        else:
            next_key = 'ZZZZZZ'

        # Find dictionary headwords expected in this gap
        edition_str = str(art['edition_year'])
        targets = []
        for norm_key, entry in headword_dict.items():
            if not (art_key < norm_key < next_key):
                continue
            if edition_str not in entry.get('editions', []):
                continue
            if entry.get('source_count', 0) < 2:
                continue
            hw = entry['headword']
            if len(hw) < 3 or len(hw) > 40:
                continue
            targets.append((norm_key, hw))

        if not targets:
            result.append(art)
            continue

        # Sort targets alphabetically so splits are in order
        targets.sort(key=lambda t: t[0])

        # Search the article body for these headwords at line boundaries
        body = art['text']
        body_upper = ''.join(
            c.upper() if len(c.upper()) == 1 else c
            for c in body
        )

        splits = []  # (position_in_body, headword, norm_key)
        for norm_key, hw in targets:
            # Search for \nHEADWORD followed by delimiter
            search_str = '\n' + hw
            idx = body_upper.find(search_str)
            while idx != -1:
                pos = idx + 1  # skip \n
                end_pos = pos + len(hw)
                if end_pos < len(body) and body[end_pos] in ',. \t\n;:(':
                    # Verify it's at the start of a line (after \n)
                    splits.append((pos, hw, norm_key))
                    break
                idx = body_upper.find(search_str, idx + 1)

        if not splits:
            result.append(art)
            continue

        # Sort by position
        splits.sort(key=lambda s: s[0])

        # Verify splits are in alphabetical order and after the main article
        valid_splits = []
        prev_key = art_key
        for pos, hw, norm_key in splits:
            if norm_key >= prev_key and pos > 100:  # at least 100 chars in
                valid_splits.append((pos, hw, norm_key))
                prev_key = norm_key

        if not valid_splits:
            result.append(art)
            continue

        log.info(f"    Splitting mega-article {art['title']} ({art['word_count']:,} words) "
                 f"into {len(valid_splits) + 1} parts: "
                 f"{', '.join(hw for _, hw, _ in valid_splits)}")

        # Create split articles
        prev_end = 0
        base_id = art['article_id']
        for j, (pos, hw, norm_key) in enumerate(valid_splits):
            # Truncate the previous article at this split point
            chunk_text = body[prev_end:pos].strip()
            if j == 0:
                # First chunk: the original article, trimmed
                art_copy = dict(art)
                art_copy['text'] = chunk_text
                art_copy['word_count'] = len(chunk_text.split())
                art_copy['paragraph_count'] = chunk_text.count('\n\n') + 1
                art_copy['char_end'] = art['char_start'] + pos
                result.append(art_copy)
            else:
                # Middle chunks from previous split
                prev_hw = valid_splits[j - 1][1]
                prev_art = result[-1]
                prev_art['text'] = chunk_text
                prev_art['word_count'] = len(chunk_text.split())
                prev_art['paragraph_count'] = chunk_text.count('\n\n') + 1

            # New article from this split point
            remaining = body[pos:]
            new_art = dict(art)
            new_art['article_id'] = f"{base_id}_split{j + 1}"
            new_art['title'] = hw
            new_art['text'] = remaining  # will be trimmed by next split
            new_art['word_count'] = len(remaining.split())
            new_art['char_start'] = art['char_start'] + pos
            new_art['heading_pattern'] = 'mega_split'
            new_art['lis_confidence'] = 0.5
            result.append(new_art)
            prev_end = pos

        # Fix last split article's text (trim to actual end)
        if len(valid_splits) > 1:
            # The last appended article already has remaining text to end, which is correct
            pass
        # Fix the last split's text — it's already body[pos:] which is correct

    return result


# ---------------------------------------------------------------------------
# 8b. Post-extraction cleanup
# ---------------------------------------------------------------------------

SUPPLEMENTARY_VOLUMES = {
    # (edition_name, volume): type
    ('7th', 1): 'front_matter',   # Dissertations, 14 articles / 3k words
    ('7th', 2): 'front_matter',   # Dissertations continued, 24 articles / 5k words
    ('8th', 1): 'front_matter',   # Dissertations, 21 articles / 2k words
}

CROSSREF_RE = re.compile(r'[Ss]ee\s+[A-Z]')


def consolidate_fragments(articles: list[dict], fragment_threshold: int = 200) -> list[dict]:
    """Merge consecutive articles with the same title, only when they're fragments.

    Skip merging when both the previous and current article are substantial
    (>= fragment_threshold words) — these are genuinely distinct articles
    sharing a headword (e.g. PARIS the person, PARIS the city, PARIS the herb).
    """
    if not articles:
        return articles
    merged = [articles[0]]
    for art in articles[1:]:
        prev = merged[-1]
        if (art['title'] == prev['title']
                and art['type'] == 'article' and prev['type'] == 'article'
                and (prev['word_count'] < fragment_threshold
                     or art['word_count'] < fragment_threshold)):
            # Merge: at least one is a fragment
            prev['text'] = prev['text'] + '\n\n' + art['text']
            prev['char_end'] = art['char_end']
            prev['word_count'] = len(prev['text'].split())
            prev['paragraph_count'] = prev['text'].count('\n\n') + 1
        else:
            merged.append(art)
    return merged


def filter_oversized_short_headwords(articles: list[dict]) -> list[dict]:
    """Remove articles whose headword is a known false-positive short word."""
    result = []
    for art in articles:
        if (art['type'] == 'article'
            and art['title'].strip() in OVERSIZED_FALSE_POSITIVES):
            log.info(f"  Filtered false headword: {art['title']} ({art['word_count']:,} words)")
            if result and result[-1]['type'] == 'article':
                result[-1]['text'] += '\n\n' + art['text']
                result[-1]['char_end'] = art['char_end']
                result[-1]['word_count'] = len(result[-1]['text'].split())
            continue
        result.append(art)
    return result


def reclassify_tiny_crossrefs(articles: list[dict], max_words: int = 25) -> list[dict]:
    """Reclassify tiny articles that are actually cross-references."""
    reclassified = 0
    for art in articles:
        if (art['type'] == 'article'
            and art['word_count'] <= max_words
            and CROSSREF_RE.search(art['text'])):
            art['type'] = 'cross_reference'
            # Extract target
            match = re.search(r'[Ss]ee\s+([A-Z][A-Za-z\s]+)', art['text'])
            if match:
                art['target'] = match.group(1).strip().rstrip('.')
            reclassified += 1
    if reclassified:
        log.info(f"  Reclassified {reclassified} tiny articles as cross-references")
    return articles


def reclassify_fat_crossrefs(articles: list[dict], min_words: int = 100) -> list[dict]:
    """Reclassify cross-references with >min_words back to articles."""
    reclassified = 0
    for art in articles:
        if art['type'] == 'cross_reference' and art['word_count'] > min_words:
            art['type'] = 'article'
            reclassified += 1
    if reclassified:
        log.info(f"  Reclassified {reclassified} fat cross-references back to articles")
    return articles


def validate_article_sizes(
    articles: list[dict],
    headword_dict: dict[str, dict] | None = None,
) -> None:
    """Log warnings for suspiciously small articles of well-known headwords.

    Gap C: Titlecase sub-entries or wrong-sense matches (e.g. PARIS = herb
    instead of city) produce ghost articles with far fewer words than expected.
    This flags them for manual review without modifying the article list.
    """
    if not headword_dict:
        return

    flagged = 0
    for art in articles:
        if art['type'] != 'article':
            continue
        norm_key = normalize_sort_key(art['title'])
        entry = headword_dict.get(norm_key)
        if not entry:
            continue
        edition_count = entry.get('edition_count', 0)
        source_count = entry.get('source_count', 0)
        if edition_count >= 7 and art['word_count'] < 50:
            log.debug(
                f"  Suspiciously small: {art['title']} = {art['word_count']} words "
                f"(appears in {edition_count} editions, {source_count} sources)"
            )
            flagged += 1
    if flagged:
        log.info(f"  {flagged} suspiciously small articles flagged for review")


# ---------------------------------------------------------------------------
# 9. Index headwords loader
# ---------------------------------------------------------------------------

def load_index_headwords(index_path: Path) -> set[str]:
    """Load headwords from an index JSONL file (e.g. 1842 index)."""
    headwords: set[str] = set()
    if not index_path.exists():
        log.warning(f"Index file not found: {index_path}")
        return headwords

    with open(index_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('entry_type') == 'main' and entry.get('references'):
                term = entry.get('term', '').upper().strip()
                if len(term) > 1:
                    headwords.add(term)

    return headwords


# ---------------------------------------------------------------------------
# 10. Single-volume pipeline
# ---------------------------------------------------------------------------

def parse_volume(
    input_path: Path,
    index_headwords: set[str] | None = None,
    headword_dict: dict[str, dict] | None = None,
) -> list[dict]:
    """Full LIS pipeline for one volume file."""
    with open(input_path) as f:
        meta = json.loads(f.readline())

    text = meta['text']
    edition_year = meta['edition']
    edition_name = meta['edition_name']
    volume = meta['volume']
    volume_range = meta.get('range')
    source_file = input_path.name

    log.info(f"  {source_file} (ed={edition_name}, vol={volume}, range={volume_range})")

    # Phase 1: Generate candidates
    candidates = generate_candidates(text, edition_year, index_headwords)
    log.info(f"    Candidates: {len(candidates)}")

    # Phase 2: Strip front matter
    candidates = strip_front_matter(candidates, text)
    log.info(f"    After front matter strip: {len(candidates)}")

    # Phase 3: Strip back matter
    candidates = strip_back_matter(candidates, text)
    log.info(f"    After back matter strip: {len(candidates)}")

    # Phase 4: LIS filtering
    accepted = longest_increasing_subsequence(candidates)
    accepted_set = {id(c) for c in accepted}
    rejected = [c for c in candidates if id(c) not in accepted_set]
    log.info(f"    After LIS: {len(accepted)} accepted, {len(rejected)} rejected")

    # Phase 5: Recovery pass
    accepted = recovery_pass(accepted, rejected, index_headwords)
    log.info(f"    After recovery: {len(accepted)}")

    # Phase 6: Range validation
    accepted, effective_range = validate_range(accepted, volume_range)

    # Phase 7 (V2): Running header detection — find missed boundaries
    # Collect all single-newline candidates for header scanning
    single_nl_candidates = [c for c in candidates if c.pattern.endswith('_snl')]
    accepted = detect_running_headers(accepted, single_nl_candidates, text)
    log.info(f"    After running header scan: {len(accepted)}")

    # Phase 7c (V2): Dictionary-guided injection — find known headwords
    # that regex missed entirely
    accepted = dictionary_guided_injection(
        accepted, text, edition_year, headword_dict,
    )

    # Phase 7d: Supplementary injection — pre-validated missed headings
    accepted = supplementary_injection(accepted, source_file)

    # Phase 8: Extract articles
    articles = extract_articles(
        accepted, text, edition_name, edition_year, volume, source_file,
    )

    # Phase 8a: Split mega-articles that swallowed subsequent entries
    articles = split_mega_articles(articles, text, headword_dict)

    # Phase 9: Consolidate same-headword fragments
    articles = consolidate_fragments(articles)
    log.info(f"    After consolidation: {len(articles)} articles")

    # Phase 9b: Remove empty articles
    articles = [a for a in articles if a['word_count'] > 0]

    # Phase 10: Filter false headwords (denylist)
    articles = filter_oversized_short_headwords(articles)

    # Phase 11: Tag supplementary volumes
    vol_key = (edition_name, volume)
    if vol_key in SUPPLEMENTARY_VOLUMES:
        vol_type = SUPPLEMENTARY_VOLUMES[vol_key]
        for art in articles:
            art['volume_type'] = vol_type
        log.info(f"  Tagged {len(articles)} articles as {vol_type}")

    # Phase 12: Reclassify tiny cross-references
    articles = reclassify_tiny_crossrefs(articles)

    # Phase 13: Reclassify fat cross-references back to articles
    articles = reclassify_fat_crossrefs(articles)

    # Phase 14: Validate article sizes (Gap C — warnings only)
    validate_article_sizes(articles, headword_dict)

    return articles


# ---------------------------------------------------------------------------
# 11. Main entry point
# ---------------------------------------------------------------------------

def get_canonical_files() -> list[str]:
    """Get canonical filenames from OCR manifest, dedup manifest, or all files."""
    # Prefer new OCR manifest (has correct volume assignments)
    if OCR_MANIFEST.exists():
        with open(OCR_MANIFEST) as f:
            manifest = json.load(f)
        canonical = sorted([
            e['filename'] for e in manifest.get('files', [])
            if e.get('is_canonical', True)
        ])
        if canonical:
            log.info(f"Using OCR manifest: {len(canonical)} canonical files")
            return canonical

    # Fall back to legacy dedup manifest
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        canonical = manifest.get('canonical', [])
        if canonical:
            log.info(f"Using dedup manifest: {len(canonical)} canonical files")
            return canonical

    log.warning("No manifest found — using all input files")
    return [p.name for p in sorted(INPUT_DIR.glob('*.jsonl'))]


def run(
    files: list[Path] | None = None,
    index_path: Path | None = None,
):
    """Run the LIS parser on all or specified files."""
    ensure_dirs()

    # Load index headwords if available
    index_headwords: set[str] | None = None
    if index_path and index_path.exists():
        index_headwords = load_index_headwords(index_path)
        log.info(f"Loaded {len(index_headwords)} index headwords")

    # Load headword dictionary (consolidated from LLM, Gemini, docs_old)
    headword_dict = load_headword_dictionary()
    if headword_dict:
        # Merge dictionary headwords into index_headwords for recovery
        dict_index = headword_dict_to_index(headword_dict, min_sources=2)
        if index_headwords:
            log.info(f"Merging {len(dict_index)} dictionary headwords into "
                     f"{len(index_headwords)} index headwords")
            index_headwords = index_headwords | dict_index
        else:
            index_headwords = dict_index
        log.info(f"Combined index_headwords: {len(index_headwords)}")

    # Resolve files
    if files is None:
        canonical = get_canonical_files()
        files = [INPUT_DIR / f for f in canonical if (INPUT_DIR / f).exists()]

    total_articles = 0
    total_crossrefs = 0
    total_words = 0

    # Track which edition+volume combos we process (for docs_old fallback)
    processed_volumes: dict[int, set[int]] = {}  # edition_year -> {vol_nums}

    for input_path in files:
        articles = parse_volume(input_path, index_headwords, headword_dict)

        # Track processed volumes
        if articles:
            ey = articles[0].get('edition_year')
            vol = articles[0].get('volume')
            if ey:
                processed_volumes.setdefault(ey, set())
                if vol is not None:
                    processed_volumes[ey].add(vol)

        # Write output
        output_path = ARTICLES_DIR / f"{input_path.stem}.articles.jsonl"
        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        n_articles = sum(1 for a in articles if a['type'] == 'article')
        n_crossrefs = sum(1 for a in articles if a['type'] == 'cross_reference')
        n_words = sum(a['word_count'] for a in articles)
        total_articles += n_articles
        total_crossrefs += n_crossrefs
        total_words += n_words

        log.info(f"    => {n_articles} articles, {n_crossrefs} cross-refs, {n_words:,} words")

    # docs_old fallback: for volumes without OCR files, load from docs_old
    # Skip editions with full OCR coverage (FULL_OCR_EDITIONS in config)
    if DOCS_OLD_DIR.exists():
        fallback_articles = 0
        fallback_words = 0
        for edition_year in [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]:
            if edition_year in FULL_OCR_EDITIONS:
                log.info(f"  Skipping docs_old for {edition_year} (full OCR coverage)")
                continue
            docs_articles = load_docs_old_articles(edition_year, headword_dict)
            if not docs_articles:
                continue

            # Group by volume
            by_vol: dict[int, list[dict]] = {}
            for a in docs_articles:
                by_vol.setdefault(a['volume'], []).append(a)

            processed = processed_volumes.get(edition_year, set())
            for vol_num, vol_articles in sorted(by_vol.items()):
                if vol_num in processed:
                    continue  # Already have OCR for this volume

                # Write fallback articles
                ed_name = vol_articles[0]['edition']
                output_stem = f"britannica_{ed_name}_{edition_year}_vol{vol_num:02d}_docs_old"
                output_path = ARTICLES_DIR / f"{output_stem}.articles.jsonl"
                with open(output_path, 'w') as f:
                    for article in vol_articles:
                        f.write(json.dumps(article, ensure_ascii=False) + '\n')

                n = len(vol_articles)
                w = sum(a['word_count'] for a in vol_articles)
                fallback_articles += n
                fallback_words += w
                log.info(f"  docs_old fallback: {ed_name} vol{vol_num:02d} => {n} articles, {w:,} words")

        if fallback_articles:
            total_articles += fallback_articles
            total_words += fallback_words
            log.info(f"  docs_old fallback total: {fallback_articles} articles, {fallback_words:,} words")

    log.info(f"LIS parser complete: {len(files)} files")
    log.info(f"  Total articles: {total_articles}")
    log.info(f"  Total cross-refs: {total_crossrefs}")
    log.info(f"  Total words: {total_words:,}")
