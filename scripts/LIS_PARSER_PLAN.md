Now I have a complete picture. Let me compile the full implementation plan.

---

Here is the comprehensive implementation plan for the LIS-based Britannica article extraction pipeline.

## Architecture Overview

The new pipeline replaces Phases 1-3.5 of the existing LLM pipeline (preprocess, classify, assemble, merge) with a single new module `lis_parser.py`, while reusing the existing `config.py`, `models.py`, `export.py`, `generate_site.py`, `dedup.py`, `verify.py`, and `compare.py` modules largely unchanged. The core idea: generate candidate headings via regex, then run Longest Increasing Subsequence on their alphabetical sort keys to filter noise.

## Answers to Design Questions

**1. New script or refactor?** New script. Create `lis_parser.py` as the main extraction engine. Modify `parse_britannica.py` to wire up the new phases. This avoids touching the LLM-based classify.py/assemble.py/merge.py files (keep them for comparison). The new script replaces Phases 1 through 3.5 with a single pass.

**2. Combining regex patterns?** Use a tiered candidate generator. The prototype's simple `\n\n([A-Z]{3,}...)` becomes the "coarse" filter. The old parser's `ARTICLE_PATTERN`, `TREATISE_PATTERN`, and cross-reference patterns become the "fine" filter. All patterns feed candidates into a unified list; duplicates are resolved by position. Each candidate gets a `pattern_source` tag (article, treatise, crossref, titlecase) that informs downstream confidence scoring.

**3. Title Case for 1842/1860?** Only accept Title Case candidates that match the 1842 index whitelist. The exploration above showed that naive Title Case regex (`[A-Z][a-z]+, a/an/the`) produces 2,553 matches in a single volume of which the vast majority are sentence starts, not headwords. The existing old parser's approach of requiring index validation is correct. For 1860 (no index), bootstrap from the 1842 index plus any headwords found via ALL-CAPS patterns in the 1860 edition itself.

**4. Cross-edition validation: bootstrap or ground truth?** Hybrid. Start with the 1842 index (36,377 terms) as a seed. Then build the union index from our own LIS parser output across all 8 editions. The 1842 index is not ground truth for all editions (vocabulary differs across eras), but it provides excellent coverage for 7th/8th editions and good coverage for common terms in earlier editions.

**5. LIS variant?** Use **patience sorting** for O(n log n) LIS, producing the single longest increasing subsequence. For ties, use a secondary sort key of character position (natural order). No need for weighted LIS -- the alphabetical constraint is strong enough. However, after LIS, apply a "recovery pass" that re-inserts candidates that were just barely out of sequence (edit distance 1-2 from where they should sort) -- these are likely OCR misspellings, not noise.

---

## Detailed Implementation Plan

### File-by-File Changes

---

### A. New File: `/home/jic823/plato/britannica_parser/lis_parser.py`

This is the core new module -- approximately 500-600 lines. It contains:

#### 1. Normalization Functions

```python
def normalize_sort_key(headword: str) -> str:
    """
    Create a sort key that handles 18th-century alphabet conventions.
    
    - Uppercase everything
    - Replace U -> V, I -> J for sort comparison only
    - Strip accents (ENCYCLOPAEDIA -> ENCYCLOPAEDIA)  
    - Strip hyphens and apostrophes for sort (BOOK-KEEPING -> BOOKKEEPING)
    - Collapse whitespace
    
    The original headword is preserved separately.
    """
    key = headword.upper()
    key = key.replace('U', 'V').replace('I', 'J')  # 18th-century alphabet
    key = unicodedata.normalize('NFKD', key)
    key = key.encode('ASCII', 'ignore').decode('ASCII')
    key = re.sub(r"['\-]", '', key)
    key = re.sub(r'\s+', ' ', key).strip()
    return key
```

#### 2. Candidate Generation

```python
@dataclass
class HeadingCandidate:
    headword: str           # Original text
    sort_key: str           # Normalized for alphabetical comparison
    char_start: int         # Position in original text where match begins
    char_end: int           # Position where match ends (after "HEADWORD, ")
    pattern: str            # 'article', 'treatise', 'crossref', 'titlecase'
    crossref_target: str | None = None  # For "See X" entries
    confidence: float = 1.0  # Reduced for titlecase, OCR-suspect, etc.


# Pattern constants
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n\n)([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*),\s+',
    re.MULTILINE
)
TREATISE_PATTERN = re.compile(
    r'(?:\n\n)([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\.[\s]*\n',
    re.MULTILINE
)
TREATISE_COMMA_NL = re.compile(
    r'(?:\n\n)([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*),\s*\n(?=[A-Z\n])',
    re.MULTILINE
)
CROSSREF_PATTERN = re.compile(
    r'(?:\n\n)([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\.?\s+See\s+([A-Z][A-Za-z\-\' ]+?)\.?\s*(?:\n|$)',
    re.MULTILINE
)
TITLECASE_PATTERN = re.compile(
    r'(?:\n\n)([A-Z][a-z]+(?:[\s\-][A-Za-z]+)*),\s+(?:a|an|the|in|one|or)\s',
    re.MULTILINE
)

# Blacklist: these ALL-CAPS tokens are never real headwords
BLACKLIST = {
    'CHAP', 'CHAPTER', 'SECT', 'SECTION', 'PART', 'ORDER', 'THEOREM',
    'CLASS', 'BOOK', 'PLATE', 'PLATES', 'FIG', 'FIGURE', 'FINIS',
    'DIRECTIONS', 'ILLUSTRATED', 'VOLUMES', 'TABLE', 'APPENDIX',
    'INDEX', 'CONTENTS', 'ERRATA', 'PREFACE', 'ADVERTISEMENT',
    'INTRODUCTION', 'SUBSCRIBERS', 'PRINTED', 'EDINBURGH', 'LONDON',
    'SUPPLEMENT', 'VOL', 'VOLUME',
}

# Roman numeral detector
ROMAN_NUMERAL = re.compile(r'^[IVXLCDM]+\.?$')

FRONT_MATTER_PATTERNS = re.compile(
    r'^(ENCYCLOP[AÆ]?DIA|BRITANNICA|OR|DICTIONARY|OF\b|ARTS\b|SCIENCES\b|'
    r'AND MISCELLANEOUS|CONSTRUCTED|COMPREHENDING|INCLUDING|TOGETHER|'
    r'COMPILED|ILLUSTRATED|VOL\.?|VOLUME|EDINBURGH|LONDON|PRINTED|MDCC|'
    r'INDOCTI|ENTERED|THE THIRD|THE FOURTH|THE FIFTH|THE SIXTH|THE SEVENTH|THE EIGHTH)',
    re.IGNORECASE
)


def generate_candidates(
    text: str,
    edition_year: int,
    index_headwords: set[str] | None = None,
) -> list[HeadingCandidate]:
    """
    Extract all heading candidates from volume text using multiple regex patterns.
    
    Returns candidates sorted by char_start (positional order).
    """
    candidates = []
    seen_positions = set()  # Dedup by start position (within 20 chars)
    
    def add_candidate(headword, start, end, pattern, crossref_target=None, confidence=1.0):
        # Dedup: skip if within 20 chars of existing candidate
        for s in seen_positions:
            if abs(start - s) < 20:
                return
        # Blacklist check
        hw_upper = headword.upper()
        first_word = hw_upper.split()[0] if ' ' in hw_upper else hw_upper
        if first_word in BLACKLIST:
            return
        if ROMAN_NUMERAL.match(hw_upper) and len(hw_upper) <= 4:
            return
        if len(headword) < 2:
            return
        
        seen_positions.add(start)
        candidates.append(HeadingCandidate(
            headword=hw_upper,  # Normalize to uppercase
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
    
    # Pattern 3: ALL-CAPS + comma + newline (treatise hybrid like SURGERY,\n)
    for m in TREATISE_COMMA_NL.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'treatise')
    
    # Pattern 4: Cross-references (HEADWORD. See TARGET)
    for m in CROSSREF_PATTERN.finditer(text):
        add_candidate(m.group(1), m.start(), m.end(), 'crossref',
                      crossref_target=m.group(2).strip())
    
    # Pattern 5: Title Case (1842/1860 only, with index validation)
    if edition_year in (1842, 1860) and index_headwords:
        for m in TITLECASE_PATTERN.finditer(text):
            hw_upper = m.group(1).upper()
            if hw_upper in index_headwords:
                add_candidate(m.group(1), m.start(), m.end(), 'titlecase',
                              confidence=0.8)
    
    # Sort by position
    candidates.sort(key=lambda c: c.char_start)
    return candidates
```

#### 3. Front Matter Stripping

```python
def strip_front_matter(candidates: list[HeadingCandidate], text: str) -> list[HeadingCandidate]:
    """
    Remove candidates that fall within front matter (title pages, prefaces).
    
    Strategy: Find the position where "real" articles begin by looking for
    the first short (1-6 char) alphabetic headword that has substantial 
    following text. Everything before that position is front matter.
    Also explicitly remove any candidate matching FRONT_MATTER_PATTERNS.
    """
    # Phase 1: Remove explicit front matter patterns
    filtered = []
    for c in candidates:
        if FRONT_MATTER_PATTERNS.match(c.headword):
            continue
        filtered.append(c)
    
    # Phase 2: Find article start position
    # The first real short headword (typical dictionary entry) marks the boundary
    article_start_pos = 0
    for c in filtered:
        hw = c.headword.replace("'", "").replace("-", "")
        if hw.isalpha() and 2 <= len(hw) <= 8:
            # Verify it has substantial text after it (not just another heading)
            text_after = text[c.char_end:c.char_end + 100].strip()
            if len(text_after) > 20:
                article_start_pos = c.char_start
                break
    
    # Keep only candidates at or after article start
    return [c for c in filtered if c.char_start >= article_start_pos]
```

#### 4. LIS Algorithm (Patience Sorting)

```python
def longest_increasing_subsequence(
    candidates: list[HeadingCandidate],
) -> list[HeadingCandidate]:
    """
    Find the longest increasing subsequence of candidates by sort_key.
    
    Uses patience sorting for O(n log n) performance.
    Ties in sort_key are broken by char_start (positional order).
    
    Returns the LIS subset of candidates, preserving original order.
    """
    if not candidates:
        return []
    
    import bisect
    
    n = len(candidates)
    # Each element is (sort_key, char_start) for comparison
    keys = [(c.sort_key, c.char_start) for c in candidates]
    
    # tails[i] = index in candidates of the smallest tail element of 
    #            all increasing subsequences of length i+1
    tails = []
    # parent[i] = index of predecessor of candidates[i] in the LIS
    parent = [-1] * n
    # indices[i] = candidate index stored at tails position i
    indices = []
    
    for i in range(n):
        # Binary search for the position where keys[i] would be inserted
        pos = bisect.bisect_left([keys[indices[j]] for j in range(len(tails))], keys[i])
        # Alternative: maintain a separate sorted keys array for efficiency
        
        if pos == len(tails):
            tails.append(i)
            indices.append(i)
        else:
            tails[pos] = i
            indices[pos] = i
        
        if pos > 0:
            parent[i] = indices[pos - 1]
    
    # Reconstruct LIS by backtracking through parent pointers
    lis_indices = []
    idx = indices[-1]
    while idx != -1:
        lis_indices.append(idx)
        idx = parent[idx]
    lis_indices.reverse()
    
    return [candidates[i] for i in lis_indices]
```

**Note on implementation:** The naive bisect approach above has a bug (recomputing the sorted keys list each iteration). The actual implementation should maintain a separate `tail_keys` array:

```python
def longest_increasing_subsequence(candidates):
    if not candidates:
        return []
    
    import bisect
    
    n = len(candidates)
    keys = [(c.sort_key, c.char_start) for c in candidates]
    
    tail_keys = []     # Smallest tail key for each LIS length
    tail_indices = []  # Corresponding candidate index
    parent = [-1] * n
    
    for i in range(n):
        pos = bisect.bisect_left(tail_keys, keys[i])
        
        if pos == len(tail_keys):
            tail_keys.append(keys[i])
            tail_indices.append(i)
        else:
            tail_keys[pos] = keys[i]
            tail_indices[pos] = i
        
        if pos > 0:
            parent[i] = tail_indices[pos - 1]
    
    # Reconstruct
    result_indices = []
    idx = tail_indices[-1]
    while idx != -1:
        result_indices.append(idx)
        idx = parent[idx]
    result_indices.reverse()
    
    return [candidates[i] for i in result_indices]
```

#### 5. Volume Range Constraint

```python
def apply_range_constraint(
    candidates: list[HeadingCandidate],
    volume_range: str | None,
) -> list[HeadingCandidate]:
    """
    Apply volume range constraint if available.
    
    Range format: "STR-ZYM" means headwords should start between STR and ZYM.
    Only filters the first and last few candidates to trim spillover.
    """
    if not volume_range or volume_range.lower() in ('none', 'unknown'):
        return candidates
    
    parts = volume_range.split('-')
    if len(parts) != 2:
        return candidates
    
    range_start = normalize_sort_key(parts[0])
    range_end = normalize_sort_key(parts[1])
    
    # Only filter candidates whose sort key is clearly outside range
    # Be generous -- allow 1-2 chars of slop for OCR
    return [
        c for c in candidates
        if c.sort_key[:len(range_start)] >= range_start[:len(range_start)]
        or c.sort_key[:3] >= range_start[:3]
    ]
    # Note: This needs refinement. The simplest safe approach is to only  
    # use the range as a post-LIS sanity check, not a pre-filter.
```

Actually, the range constraint should be applied **after** LIS as a light sanity check, not as a pre-filter (pre-filtering could remove valid candidates that LIS needs). Better design:

```python
def validate_range(
    accepted: list[HeadingCandidate],
    volume_range: str | None,
) -> tuple[list[HeadingCandidate], str]:
    """
    Post-LIS: validate that accepted headwords fall within volume range.
    If range is None, infer it from first/last accepted headwords.
    Returns (accepted_candidates, effective_range_string).
    """
    if not accepted:
        return accepted, 'EMPTY'
    
    first_hw = accepted[0].headword
    last_hw = accepted[-1].headword
    inferred_range = f"{first_hw[:3]}-{last_hw[:3]}"
    
    if not volume_range or volume_range.lower() in ('none', 'unknown'):
        return accepted, inferred_range
    
    # Just log warnings for out-of-range headwords; don't remove them
    # (LIS already enforced ordering; range violations are likely OCR artifacts)
    parts = volume_range.split('-')
    range_start = normalize_sort_key(parts[0])
    range_end = normalize_sort_key(parts[1]) + 'ZZZZ'  # Generous end
    
    violations = [c for c in accepted if c.sort_key < range_start or c.sort_key > range_end]
    
    return accepted, volume_range
```

#### 6. Recovery Pass

```python
def recovery_pass(
    accepted: list[HeadingCandidate],
    rejected: list[HeadingCandidate],
    index_headwords: set[str] | None = None,
) -> list[HeadingCandidate]:
    """
    Re-insert rejected candidates that were only barely out of sequence.
    
    A candidate is recoverable if:
    1. It's in the union headword index (cross-edition confirmation), OR
    2. It fits between two accepted neighbors alphabetically with at most
       1-position displacement (a swap, not random noise).
    
    This catches OCR-mangled headwords like "TANNIG" for "TANNING".
    """
    if not rejected:
        return accepted
    
    # Build position map of accepted candidates
    accepted_set = {c.char_start for c in accepted}
    
    recoverable = []
    for r in rejected:
        # Check 1: Is it in the cross-edition index?
        if index_headwords and r.headword in index_headwords:
            recoverable.append(r)
            continue
        
        # Check 2: Does it fit between its positional neighbors?
        # Find where it would be inserted by position
        insert_pos = bisect.bisect_left(
            [a.char_start for a in accepted], r.char_start
        )
        if 0 < insert_pos < len(accepted):
            prev_key = accepted[insert_pos - 1].sort_key
            next_key = accepted[insert_pos].sort_key
            if prev_key <= r.sort_key <= next_key:
                recoverable.append(r)
    
    if not recoverable:
        return accepted
    
    # Merge recovered candidates back into accepted list by position
    combined = accepted + recoverable
    combined.sort(key=lambda c: c.char_start)
    return combined
```

#### 7. Text Extraction and Article Construction

```python
def extract_articles(
    accepted: list[HeadingCandidate],
    text: str,
    edition_name: str,
    edition_year: int,
    volume: int,
    source_file: str,
) -> list[dict]:
    """
    Given accepted headword candidates in positional order, slice the text
    into articles. Each article's text runs from its heading to the next heading.
    """
    articles = []
    
    for i, candidate in enumerate(accepted):
        # Text starts at char_end of this heading's regex match
        text_start = candidate.char_end
        
        # Text ends at char_start of the next heading (or end of volume)
        if i + 1 < len(accepted):
            text_end = accepted[i + 1].char_start
        else:
            text_end = len(text)
        
        article_text = text[text_start:text_end].strip()
        
        # Determine article type
        if candidate.pattern == 'crossref':
            article_type = 'cross_reference'
        else:
            article_type = 'article'
        
        article_id = f"eb_{edition_name}_{edition_year}_v{volume:02d}_{i+1:04d}"
        
        article = {
            'article_id': article_id,
            'title': candidate.headword,
            'edition': edition_name,
            'edition_year': edition_year,
            'volume': volume,
            'source_file': source_file,
            'type': article_type,
            'char_start': candidate.char_start,
            'char_end': text_end,
            'text': article_text,
            'word_count': len(article_text.split()),
            'paragraph_count': article_text.count('\n\n') + 1,
            'keywords': None,
            'author_attribution': None,
            'target': candidate.crossref_target,
            'subsections': [],
            'lis_confidence': candidate.confidence,
            'heading_pattern': candidate.pattern,
        }
        articles.append(article)
    
    return articles
```

#### 8. Main Pipeline Function

```python
def parse_volume(
    input_path: Path,
    index_headwords: set[str] | None = None,
) -> list[dict]:
    """
    Full LIS pipeline for one volume file.
    
    Returns list of article dicts compatible with the export format.
    """
    with open(input_path) as f:
        meta = json.loads(f.readline())
    
    text = meta['text']
    edition_year = meta['edition']
    edition_name = meta['edition_name']
    volume = meta['volume']
    volume_range = meta.get('range')
    source_file = input_path.name
    
    # Phase 1: Generate candidates
    candidates = generate_candidates(text, edition_year, index_headwords)
    log.info(f"  Candidates: {len(candidates)}")
    
    # Phase 2: Strip front matter
    candidates = strip_front_matter(candidates, text)
    log.info(f"  After front matter strip: {len(candidates)}")
    
    # Phase 3: LIS filtering
    accepted = longest_increasing_subsequence(candidates)
    rejected = [c for c in candidates if c not in set_of_accepted]
    log.info(f"  After LIS: {len(accepted)} accepted, {len(rejected)} rejected")
    
    # Phase 4: Recovery pass
    accepted = recovery_pass(accepted, rejected, index_headwords)
    log.info(f"  After recovery: {len(accepted)}")
    
    # Phase 5: Range validation
    accepted, effective_range = validate_range(accepted, volume_range)
    
    # Phase 6: Extract articles
    articles = extract_articles(
        accepted, text, edition_name, edition_year, volume, source_file
    )
    
    return articles


def run(
    files: list[Path] | None = None,
    index_path: Path | None = None,
):
    """
    Run the LIS parser on all or specified files.
    """
    ensure_dirs()
    
    if files is None:
        files = sorted(INPUT_DIR.glob('*.jsonl'))
    
    # Load index headwords if available
    index_headwords = None
    if index_path and index_path.exists():
        index_headwords = load_index_headwords(index_path)
        log.info(f"Loaded {len(index_headwords)} index headwords")
    
    total_articles = 0
    for input_path in files:
        stem = input_path.stem
        articles = parse_volume(input_path, index_headwords)
        
        # Write output
        output_path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        total_articles += len(articles)
        log.info(f"{stem}: {len(articles)} articles")
    
    log.info(f"LIS parser complete: {len(files)} files, {total_articles} articles")
```

---

### B. Modified File: `/home/jic823/plato/britannica_parser/config.py`

Changes:
- Remove LLM-related settings (API_BASE, API_URL, MODEL, BATCH_SIZE, OVERLAP, etc.)
- Add LIS-specific config
- Keep all path constants and EDITIONS dict

```python
# Add these new constants:
INDEX_1842_PATH = Path("/home/jic823/1815EncyclopediaBritannicaNLS/output_v2/index_1842.jsonl")
UNION_INDEX_PATH = OUTPUT_DIR / "union_index.jsonl"

# Remove or comment out:
# API_BASE, API_URL, MODEL, BATCH_SIZE, OVERLAP, STEP_SIZE, PREVIEW_LENGTH
# MAX_CONCURRENT, REQUEST_TIMEOUT, MAX_RETRIES, LLM_TEMPERATURE, LLM_MAX_TOKENS
```

---

### C. Modified File: `/home/jic823/plato/britannica_parser/models.py`

Add a `HeadingCandidate` dataclass (or import from lis_parser) and extend Article:

```python
@dataclass
class HeadingCandidate:
    headword: str
    sort_key: str
    char_start: int
    char_end: int
    pattern: str  # 'article', 'treatise', 'crossref', 'titlecase'
    crossref_target: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Article:
    # ... existing fields ...
    lis_confidence: float = 1.0         # NEW: confidence from LIS pipeline
    heading_pattern: str = 'article'     # NEW: which regex matched
```

---

### D. Modified File: `/home/jic823/plato/britannica_parser/parse_britannica.py`

Replace Phase 1-3.5 with a single `lis` phase. Keep all other phases.

New phase choices: `"lis"`, `"verify"`, `"dedup"`, `"compare"`, `"audit"`, `"export"`, `"site"`, `"cross-edition"`, `"all"`

```python
def run_lis(files: list[Path], index_path: Path | None = None):
    """Run LIS-based article extraction (replaces Phases 1-3.5)."""
    import lis_parser
    lis_parser.run(files, index_path)


def run_cross_edition(files: list[Path]):
    """Build cross-edition union index and validate."""
    import cross_edition
    cross_edition.run(files)
```

Update argparse choices and the main dispatch logic accordingly.

---

### E. New File: `/home/jic823/plato/britannica_parser/cross_edition.py`

Approximately 150-200 lines. Cross-edition validation module (Phase 4).

```python
def build_union_index(articles_dir: Path, canonical_files: list[str]) -> dict[str, dict]:
    """
    Build a union headword index from all editions' articles.
    
    Returns: {normalized_headword: {
        'headword': str,  # canonical form
        'editions': {edition_name: [volume_numbers]},
        'count': int,     # how many editions have it
    }}
    """
    union = defaultdict(lambda: {'editions': defaultdict(list), 'count': 0})
    
    for filename in canonical_files:
        stem = filename.replace('.jsonl', '')
        path = articles_dir / f"{stem}.articles.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                article = json.loads(line)
                if article['type'] not in ('article', 'cross_reference'):
                    continue
                norm = normalize_sort_key(article['title'])
                entry = union[norm]
                if not entry.get('headword'):
                    entry['headword'] = article['title']
                edition = article['edition']
                vol = article['volume']
                if vol not in entry['editions'][edition]:
                    entry['editions'][edition].append(vol)
    
    # Count unique editions per headword
    for norm, entry in union.items():
        entry['count'] = len(entry['editions'])
    
    return dict(union)


def flag_anomalies(union_index: dict, threshold: int = 5) -> dict:
    """
    Flag headwords that appear in most editions but are missing from some.
    
    Returns: {edition_name: [missing_headwords_that_appear_in_threshold+_other_editions]}
    """
    all_editions = set()
    for entry in union_index.values():
        all_editions.update(entry['editions'].keys())
    
    missing_by_edition = defaultdict(list)
    for norm, entry in union_index.items():
        if entry['count'] >= threshold:
            present = set(entry['editions'].keys())
            missing = all_editions - present
            for edition in missing:
                missing_by_edition[edition].append(entry['headword'])
    
    return dict(missing_by_edition)


def run(files: list[Path] | None = None):
    """Build union index, flag anomalies, write report."""
    ...
```

---

### F. Modified File: `/home/jic823/plato/britannica_parser/verify.py`

Minimal changes. The existing verification checks work on `.articles.jsonl` files regardless of how they were generated. Add one new check:

```python
def check_lis_stats(articles_path: Path) -> dict:
    """Check LIS-specific metrics: pattern distribution, confidence."""
    articles = load_articles(articles_path)
    patterns = defaultdict(int)
    low_confidence = 0
    for a in articles:
        patterns[a.get('heading_pattern', 'unknown')] += 1
        if a.get('lis_confidence', 1.0) < 0.9:
            low_confidence += 1
    return {
        'patterns': dict(patterns),
        'low_confidence': low_confidence,
    }
```

---

### G. Files Reused Unchanged

- `/home/jic823/plato/britannica_parser/export.py` -- reads `.articles.jsonl`, no changes needed
- `/home/jic823/plato/britannica_parser/generate_site.py` -- reads from export dir, no changes needed
- `/home/jic823/plato/britannica_parser/dedup.py` -- operates on source JSONL files, no changes needed
- `/home/jic823/plato/britannica_parser/compare.py` -- reads `.articles.jsonl`, no changes needed

---

## Algorithm Pseudocode Summary

```
FOR each volume file:
    1. Load text and metadata
    2. CANDIDATES = apply all regex patterns to text
       - ARTICLE_PATTERN (ALL-CAPS + comma)
       - TREATISE_PATTERN (ALL-CAPS + period + newline)
       - TREATISE_COMMA_NL (ALL-CAPS + comma + newline)
       - CROSSREF_PATTERN (HEADWORD. See TARGET)
       - TITLECASE_PATTERN (for 1842/1860 with index validation)
    3. Remove blacklisted tokens (CHAP, SECT, PART, Fig., etc.)
    4. Remove front matter (ENCYCLOPAEDIA BRITANNICA, VOL, EDINBURGH, etc.)
    5. Sort candidates by char_start (position in text)
    6. Compute sort_key for each (U->V, I->J normalization)
    7. Run LIS on sort_keys to find longest alphabetically increasing subsequence
       - O(n log n) via patience sorting
       - Secondary sort on char_start for ties
    8. RECOVERY PASS: re-insert rejected candidates that:
       - Exist in cross-edition index, AND
       - Fit between their positional neighbors alphabetically
    9. Slice text between accepted headwords to extract article bodies
    10. Detect cross-references (pattern='crossref') and tag them
    11. Write .articles.jsonl in existing format

THEN (cross-edition phase):
    12. Build union headword index across all editions
    13. Flag headwords present in 5+ editions but missing from specific ones
    14. Generate anomaly report for human review
```

---

## Performance Estimate

- Candidate generation: ~200ms per volume (regex over 1-6MB text)
- LIS: ~1ms per volume (n=1000-2000 candidates, O(n log n))
- Text slicing: ~10ms per volume
- Total per volume: ~250ms
- Total for 166 canonical files: **~40 seconds** (vs. 5 GPU-days for LLM approach)

---

## Potential Challenges and Mitigations

1. **Treatise headwords that don't match any regex pattern**: The STRENGTH entry in vol18 begins with "Strength of materials in mechanics..." with no ALL-CAPS headword line. This is the hardest case -- it appears as a continuation of front matter. Mitigation: For volumes with a known range (e.g., "STR-ZYM"), if the first accepted headword doesn't start with the range prefix, insert a synthetic candidate for the range start based on the text immediately after front matter.

2. **Multi-sense entries (VALENCIA appearing twice)**: LIS handles this naturally -- both VALENCIA entries will have the same sort_key, and since we use `bisect_left` (not `bisect_right`), equal keys are allowed in sequence. The `(sort_key, char_start)` composite key ensures positional ordering within ties.

3. **Very long treatises (SURGERY: 500+ pages)**: These are single articles in the LIS output. The absence of Phase 3.5 merge logic is actually a simplification -- since we never fragment them in the first place, there is nothing to merge. Internal CHAP/SECT headings are blacklisted and never become candidates.

4. **1842/1860 Title Case OCR inconsistency**: The "ANCaster" example (mixed case in OCR) shows that some headwords in 1842 are rendered as Title Case by OCR even though they were originally ALL-CAPS. These will only be caught by the TITLECASE_PATTERN with index validation. For entries missing from both the ALL-CAPS and index-validated Title Case patterns, the cross-edition validation phase will flag them as missing.

5. **Edge case: empty LIS**: If a file has no valid candidates after front matter stripping (e.g., an index-only volume like `britannica_7th_1842_vol00_unknown.jsonl` which is 5,865 chars of index text), the parser should produce an empty articles file and log a warning.

---

## Testing Strategy

1. **Unit test the LIS on the proven prototype volume**: `britannica_3rd_1797_vol18_STR-ZYM.jsonl` should produce ~1,143 accepted candidates (matching the prototype result).

2. **Compare against LLM pipeline output**: For each edition, compare article counts and title lists. The LIS output should closely match the post-merge LLM output (146,585 articles across all editions).

3. **Compare against the old regex parser**: For 1st edition, compare against the web-published hybrid parser output (~7,764 articles) using the existing `compare.py`.

4. **Spot-check U/V normalization**: Verify that both UBIQUITARIANS and VACCINIUM appear in the accepted list for vol18, properly interleaved.

5. **Spot-check cross-references**: Verify that "STURGEON. See ACCIPENSER" is detected as a cross-reference with target "ACCIPENSER".

---

### Critical Files for Implementation
- `/home/jic823/plato/britannica_parser/lis_parser.py` - New core module: candidate generation, LIS algorithm, text extraction (to be created)
- `/home/jic823/plato/britannica_parser/parse_britannica.py` - CLI orchestrator: rewire phase dispatch to use LIS instead of LLM phases
- `/home/jic823/plato/britannica_parser/config.py` - Configuration: remove LLM settings, add index paths and LIS parameters
- `/home/jic823/plato/britannica_parser/models.py` - Data models: add HeadingCandidate dataclass, extend Article with LIS fields
- `/home/jic823/1815EncyclopediaBritannicaNLS/parse_britannica_articles.py` - Reference: regex patterns, blacklists, front matter filtering, Title Case handling to port into the new pipeline
agentId: adabbb7b09f07546d (for resuming to continue this agent's work if needed)
<usage>total_tokens: 119108
tool_uses: 44
duration_ms: 386664</usage>
