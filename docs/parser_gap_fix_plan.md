# Plan: Fix Parser Gaps for Major Geographic Articles

## Context

We discovered that the PARIS city article (~5,000 words on the Louvre, Notre Dame, Bastille) was absorbed into PARIETES in the 1797 edition. The OCR has `Paris, the capital of the kingdom of France` in titlecase, not ALL-CAPS, so the parser missed it. Meanwhile, the ALL-CAPS `PARIS` entry further down is the Herb Paris plant (243 words).

Investigation of 42 major geographic places across 8 editions reveals **48 completely missing** and **17 ghost entries** (<100 words). This is a systematic issue — the parser misses articles whose headwords are in titlecase or use `HEADWORD. Text...` / `HEADWORD (paren...` format instead of the standard `HEADWORD, text...`.

### The Three Parser Gaps

| Gap | Pattern | Example | Scale |
|-----|---------|---------|-------|
| A | `HEADWORD. Text...` (period + text, no newline) | `INDIA. The general name...` | ~14K candidates |
| B | `HEADWORD (paren...` (open-paren instead of comma) | `AFRICA (according to Bochart...` | ~10K candidates |
| C | Titlecase sub-entry (OCR didn't preserve caps) | `Paris, the capital of...` | Unknown |

### Major Places Coverage Matrix (current state)

```
Place                1771  1778  1797  1810  1815  1823  1842  1860
AFRICA                ---   ---   ---  24K   24K   12K   ---   25K
AMERICA               ---   23K   62K   ---  110K  149K   ---   ---
BRAZIL                ---   ---   ---   ---   ---   ---   28K   31K
CHINA                 191   16K   49K   51K   ---   51K   51K   54K
EDINBURGH             ---   15K   24K   18K   *6    30K   22K   *8
ENGLAND               *88   82K   94K   95K   93K   95K   54K   *8
FRANCE                124  *175   64K  175K  175K   22K  271K   2K
INDIA                 152   ---   ---   24K   ---   61K   ---   ---
PARIS                 106    3K  *243    5K    4K    4K   12K   12K
SCOTLAND              181  208K   86K   ---  220K   46K   75K   ---
SPAIN                 *85   31K   17K   66K   67K   61K   ---   ---

--- = missing entirely    * = suspiciously small / wrong article
```

---

## Approach: Parser Enhancement (not post-hoc patching)

Rather than write a one-off script to fix known cases, we should fix the parser's regex patterns so that re-running it produces correct output for ALL articles, not just the 42 places we checked.

### Step 1: Add `HEADWORD. Text...` pattern (Gap A)

**File:** `/home/jic823/plato/britannica_parser/lis_parser.py`

Add a new regex pattern for headwords followed by period + space + text (not newline):

```python
# Current TREATISE_PATTERN requires: HEADWORD.\n
# New pattern: HEADWORD. Lowercase text follows on same line
ARTICLE_PERIOD_PATTERN = re.compile(
    r'(?:^|\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.[ \t]+(?=[A-Za-z])',
    re.MULTILINE,
)
```

This catches `INDIA. The general name...`, `CHEMISTRY. This science...`, etc.

**Confidence:** 0.9 (high — double newline + ALL-CAPS + period + text is a strong signal)

### Step 2: Add `HEADWORD (paren...` pattern (Gap B)

Add a pattern for headwords followed by open-parenthesis:

```python
# HEADWORD (qualifier) text...
ARTICLE_PAREN_PATTERN = re.compile(
    r'(?:^|\n\n)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\s+(?=\()',
    re.MULTILINE,
)
```

This catches `AFRICA (according to Bochart, from a Punic word...) one of...`

**Confidence:** 0.8 (slightly lower — parenthetical can sometimes be a citation or footnote)

### Step 3: Handle titlecase sub-entries (Gap C) — limited scope

Gap C (titlecase headwords) is the hardest to fix systematically because lowercase text is everywhere. However, we can handle the specific case that caused the PARIS miss:

**When a known headword from the dictionary has a much smaller article than expected** (e.g., PARIS = 243 words but appears in 5+ other editions at 3,000+ words), flag it for review. The `dict_guided` injection already knows about these headwords — the issue is that it finds the wrong one.

For this, add a **post-processing validation step** that checks article word counts against the cross-edition median:

```python
def validate_article_sizes(articles, headword_dict):
    """Flag articles that are suspiciously small compared to other editions."""
    for article in articles:
        norm_key = normalize_sort_key(article['title'])
        entry = headword_dict.get(norm_key)
        if entry and entry.get('edition_count', 0) >= 4 and article['word_count'] < 200:
            log.warning(f"Suspiciously small: {article['title']} = {article['word_count']} words "
                        f"(appears in {entry['edition_count']} editions)")
```

This won't auto-fix the issue (that requires OCR-level investigation), but it will flag the ~17 ghost/wrong entries for manual review.

---

## Implementation Plan

### What changes and where

| File | Change |
|------|--------|
| `/home/jic823/plato/britannica_parser/lis_parser.py` | Add 2 new regex patterns (Gap A, B) + size validation (Gap C) |
| No other files change | The parser re-runs on existing OCR to produce corrected output |

### Detailed changes to `lis_parser.py`

1. **Add `ARTICLE_PERIOD_PATTERN`** alongside existing `ARTICLE_PATTERN`, `TREATISE_PATTERN` etc. in the candidate detection section
2. **Add `ARTICLE_PAREN_PATTERN`** similarly
3. Both patterns feed into the same LIS pipeline — the LIS algorithm will naturally pick the best candidates based on alphabetical ordering
4. **Add `validate_article_sizes()`** as a post-processing warning step

### Re-running the parser

After modifying `lis_parser.py`:
1. User pushes changes to GitLab
2. User pulls on Plato cluster
3. User re-runs parser on all 155 OCR files
4. User downloads new article files locally
5. We regenerate exports, concept index, and site

### Risk assessment

- **Gap A fix (period pattern):** Low risk. `\n\n` + ALL-CAPS + period + text is very specific. May find ~5,000-10,000 additional genuine articles across all editions.
- **Gap B fix (paren pattern):** Medium risk. Need to be careful not to match footnotes or parenthetical asides within existing articles. The `\n\n` prefix and ALL-CAPS requirement make this fairly safe.
- **Gap C (titlecase):** Not auto-fixing. Just flagging for manual review. Zero risk.
- **LIS algorithm:** The new candidates join the existing pool. The LIS optimization will filter out false positives that don't fit the alphabetical sequence.

---

## Verification

After parser re-run, check the same 42 major places matrix:

```bash
# Quick check
python3 -c "
import json, glob
places = ['PARIS', 'FRANCE', 'INDIA', 'AFRICA', 'SCOTLAND', 'SPAIN', 'BRAZIL', 'AMERICA', 'EDINBURGH', 'ENGLAND']
for f in sorted(glob.glob('data/export/eb_*.jsonl')):
    ...  # same matrix check as above
"
```

**Success criteria:**
- PARIS 1797: > 4,000 words (city article, not herb)
- FRANCE 1778: > 10,000 words (country, not Isle de France)
- INDIA appears in all 8 editions
- AFRICA appears in all 8 editions
- Ghost entries (EDINBURGH 1815 = 6 words) eliminated or corrected
- No regression: total article count should increase (not decrease)
- Manual spot-check of 10 newly-found articles confirms they're correct

---

## After Parser Fix: Resume Geocoding Plan

Once the parser is fixed and we have cleaner article data, we proceed with geographic enrichment:

1. **Extract coordinates from article text** (~5,000-6,000 concepts have embedded `E. Long. / N. Lat.`)
2. **Match remaining concepts against GeoNames** (6.2M Place nodes in Neo4j, or local SQLite from `cities500.txt`)
3. **Enrich concept_index.json** with `geo: {lat, lon, geonames_id, feature_class, source, confidence}`
4. **Find articles mentioning places** — search article text for geocoded place names

This geocoding step depends on having correct article data, so the parser fix comes first.
