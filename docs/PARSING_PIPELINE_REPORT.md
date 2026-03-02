# Parsing Pipeline Report

## Encyclopaedia Britannica Historical Corpus (1771--1860)

**Final corpus**: 126,088 articles across 8 editions, 120 million words, deployed at [jburnford.github.io/early_encyclopedia_britannica](https://jburnford.github.io/early_encyclopedia_britannica/)

---

## 1. The Problem

The goal was to extract individual encyclopedia articles from raw OCR text of 210 digitised volumes covering eight editions of the Encyclopaedia Britannica (1st through 8th, 1771--1860). The source PDFs come from the National Library of Scotland and the Internet Archive; the raw OCR was produced by OLMoCR (Allen Institute for AI) running on H100 GPUs.

The core challenge: encyclopedia volumes contain thousands of articles that flow continuously with minimal structural markup. Article headwords may be ALL-CAPS, Title Case, or garbled by OCR errors. Running headers repeat the current headword at the top of every page and look identical to article starts. Treatises spanning dozens of pages contain internal section headings that resemble new articles. Different editions use different typographic conventions. The 4th edition (1810) is a supplement where volumes are not alphabetically ordered. The 1842 and 1860 editions increasingly use Title Case headwords rather than ALL-CAPS.

---

## 2. Approach 1: Regex Parser (December 2025)

### Method

A straightforward single-pass regex parser (`parse_britannica_articles.py`) that scanned OCR text for ALL-CAPS headwords followed by a comma: `^[A-Z][A-Z'-]+, `.

### Results

- **18,172 articles** extracted from the 1815 5th edition alone
- Fast execution (seconds)

### Why It Failed

- **Missed 30--40% of articles**: The single regex pattern could not handle the variety of headword formats across editions.
- No handling of OCR case errors ("Aalen," vs "AALEN,"), Title Case entries (common in 1842/1860), cross-references ("See OPTICS"), or treatise-style entries ("ASTRONOMY.\n\n" with period-newline formatting).
- No cross-validation: the parser had no knowledge of what articles *should* exist.
- Running headers were not filtered, producing thousands of false positives.

---

## 3. Approach 2: Smart Parser + Gemini Corrections (January 2026)

### Method

A modular Python package (`encyclopedia_parser/`) that layered several strategies on top of the basic regex approach:

1. **Expected Article Registry**: Loaded 18,215 entries from the 1842 General Index as ground truth to know what articles should exist.
2. **Fuzzy Matcher**: Used `rapidfuzz` to detect OCR spelling variations. Learned 4,488 OCR error patterns (common: B->R, E->I, E->R).
3. **LLM-Assisted Extraction**: Claude Haiku classified ambiguous article boundaries.
4. **Post-hoc Correction Pipeline**: Sent problematic page ranges to the Gemini Flash API, which returned corrected article boundaries. Applied corrections to 2,169 chunks across all 8 editions.
5. **Manual Outlier Review**: An interactive LLM tool reviewed 4,404 flagged articles, making KEEP/MERGE/DELETE decisions.

### Results

| Stage | Articles |
|-------|----------|
| Initial regex parse | 121,161 |
| After LLM corrections | 135,117 |
| After outlier review | 134,456 |
| After Gemini re-parsing | **153,438** |

This was the version first deployed to the GitHub Pages website.

### Why It Was Insufficient

The fundamental extraction was still regex-based. Each correction layer was a band-aid:

- Title Case coverage only improved from 72% to 83% for the 1842 2nd volume.
- The 1842 and 1860 editions required the most Gemini correction chunks (698 and 699 respectively), indicating the parser was weakest precisely where it mattered most.
- The corrections were not reproducible without API access and cost ~$100--200 in Gemini API calls.
- Systematic issues persisted: sentence fragments parsed as articles, plate/figure explanations treated as entries, and botanical classification headers infiltrating the article list.
- The quality audit gave editions grades of B to A- with an estimated 95--97% accuracy -- good but not at the level needed for a research-quality corpus.

---

## 4. Approach 3: Full LLM Classification Pipeline (February 2026)

### Method

A complete pipeline rewrite using DeepSeek-R1-Distill-Llama-70B-AWQ (a 70B parameter LLM with guided JSON output) running on Plato HPC A100 GPUs. The idea was to classify every paragraph in all 210 OCR files rather than relying on regex patterns at all.

**Four-phase architecture:**

```
Phase 1: Paragraph splitting (preprocess.py)     ~28 seconds
Phase 2: LLM classification  (classify.py)        ~5 GPU-days
Phase 3: Article assembly     (assemble.py)        ~1 minute
Phase 3.5: Fragment merging   (merge.py)           ~1 minute
```

**Phase 1 -- Paragraph Splitting** (`preprocess.py`): Split 210 OCR files on `\n\n` boundaries, producing 1,650,866 paragraphs with character offset tracking.

**Phase 2 -- LLM Classification** (`classify.py`): Sent sliding windows of 20 paragraph previews (with 2-paragraph overlap) to DeepSeek-R1 via vLLM. Each paragraph was classified as `article_start`, `running_header`, `cross_reference`, `front_matter`, `back_matter`, `subsection_start`, `author_attribution`, `footnote_sep`, or `body_text`. Used `guided_json` schema to force structured output (essential -- without it, DeepSeek-R1's `<think>` reasoning tokens consumed the entire output budget).

**Phase 3 -- Assembly** (`assemble.py`): A state machine walked classified paragraphs sequentially. `article_start` triggered a new article; `running_header` and `footnote_sep` were skipped; `body_text` accumulated into the current article; `back_matter` was only accepted in the final 10% of the volume to prevent mid-volume misclassification.

**Phase 3.5 -- Fragment Merging** (`merge.py`): The LLM classifier frequently promoted internal section headings (Chapter IV, Of the Liver, etc.) to `article_start`, shattering long treatises into dozens of fragments. A cascading heuristic system attempted to re-merge them:
- `UNTITLED` or garbage titles (>80 chars) -> always merge
- Consecutive identical titles -> merge (handles "HEAT" repeated 80 times from running headers)
- Chapter/section patterns -> merge
- ALL-CAPS title >2 chars -> hard boundary (unless a snowball absorption pattern is detected)
- Mixed-case title after a large predecessor -> merge as subsection

This reduced 220,056 raw articles to 146,585 (33.4% fragment absorption).

### Infrastructure Challenges

- DeepSeek-R1's `<think>` tokens consumed the entire 2048-token output budget, producing empty classification arrays until `guided_json` was enabled.
- vLLM's default AWQ quantization kernels were **17x slower** than `awq_marlin`. Switching to the marlin backend cut per-batch latency from ~50s to ~3s.
- SSH disconnects killed long-running `nohup` processes. Moved to `tmux` sessions.
- Python version mismatches on Plato (3.7 in the default environment vs 3.11 needed for modern type hints).
- Checkpoint/resume system was essential given the 5-day processing time -- saved state every 50 windows.

### Results

| Metric | Value |
|--------|-------|
| Raw articles (pre-merge) | 220,056 |
| Post-merge articles | 146,585 |
| Cross-references | 31,928 |
| Total words | 155,714,863 |
| Running headers removed | 142,836 |
| GPU time | ~5 days on 2x A100 80GB |

### Why It Failed

Despite the massive compute investment, the LLM pipeline produced **worse results** than the corrected regex parser for the later editions:

| Edition | LLM Pipeline | Regex+Corrections |
|---------|-------------|-------------------|
| 1842 7th | **3,672** | 19,669 |
| 1860 8th | **4,253** | 16,458 |

The LLM catastrophically undercounted articles in the 1842 and 1860 editions -- precisely the editions where Title Case headwords are most common. The model was biased toward classifying Title Case text as `body_text` rather than `article_start`.

Additional problems:
- 17,994 alphabetical order flags remaining
- 47 duplicate titles per volume on average
- OCR garbage titles that passed through classification
- The merger heuristics were complex and fragile -- tuning them for one edition's treatises broke fragment detection in another
- **5 GPU-days of compute** made iterative improvement impractical

The classification approach treated each paragraph independently, ignoring the global constraint that encyclopedia articles are alphabetically ordered. This is like classifying each pixel in an image without considering the image as a whole.

---

## 5. The Final Pipeline: LIS Parser (Late February 2026)

### The Key Insight

Encyclopedia articles are alphabetically ordered. After generating heading candidates via regex, the true article headwords form the **Longest Increasing Subsequence (LIS)** of the candidates when sorted by alphabetical order at ascending text positions. Running headers, section headings, OCR noise, and other false positives break this alphabetical ordering and are automatically excluded by the LIS algorithm.

This transforms article boundary detection from a classification problem (requiring an LLM) into a **sequence optimization problem** (solvable in O(n log n) with patience sorting).

### Architecture

```
Input: 166 canonical OCR files (after deduplication)

  1. Candidate generation    (14 regex patterns)
  2. Blacklist filtering     (structural headings, false positives)
  3. Front/back matter strip (title pages, FINIS markers)
  4. LIS filtering           (patience sorting, O(n log n))
  5. Recovery pass           (re-insert near-misses from cross-edition index)
  6. Running header detection (inside accepted articles)
  7. Dictionary-guided injection (known headwords missed by regex)
  8. Supplementary injection (Gemini-classified single-newline headings)
  9. Article extraction      (text slicing between boundaries)
 10. Mega-article splitting  (>50K word articles searched for embedded headwords)
 11. Post-extraction cleanup (fragment consolidation, reclassification)

Output: Per-volume .articles.jsonl files
        -> cross_edition.py (confidence scoring)
        -> export.py (JSONL + search index)
        -> generate_site.py (static website)
```

**Processing time**: ~40 seconds for all 166 files (~250ms per volume).

### Stage-by-Stage Detail

#### 5.1 Candidate Generation

Fourteen regex patterns (7 double-newline variants at confidence 1.0, plus 7 single-newline variants at confidence 0.7):

| Pattern | Example | Description |
|---------|---------|-------------|
| `ARTICLE_PATTERN` | `ABACUS, an instrument...` | ALL-CAPS + comma (most common) |
| `TREATISE_PATTERN` | `ASTRONOMY.\n\n` | ALL-CAPS + period + newline |
| `TREATISE_COMMA_NL` | `ANATOMY,\nthe art of...` | ALL-CAPS + comma + newline |
| `CROSSREF_PATTERN` | `COLOUR. See OPTICS.` | Cross-reference format |
| `TITLECASE_PATTERN` | `Edinburgh, a city...` | Title Case validated against headword dictionary |
| `ARTICLE_PERIOD` | `INDIA. The general name...` | ALL-CAPS + period + inline text |
| `ARTICLE_PAREN` | `ALPS (a word signifying...` | ALL-CAPS + open parenthesis |

Each pattern also has a `_SINGLE_NL` variant matching after a single newline (for articles that OCR failed to separate with a blank line).

Candidates are deduplicated by position (within 20 characters) and filtered against extensive blacklists: structural headings (`CHAPTER`, `SECTION`, `PART`, `PLATE`), common false-positive words (`THUS`, `MUCH`, `HENCE`), and two-letter words (`IF`, `AS`, `OF`).

#### 5.2 18th-Century Alphabet Normalization

A critical detail: historical encyclopedias treat I/J and U/V as the same letter. The sort key normalization function converts U->V, I->J, strips accents, and removes hyphens and apostrophes. Without this, the LIS algorithm would reject legitimate headwords like "JURISPRUDENCE" (which sorts between I and K in the 18th-century alphabet).

#### 5.3 Front and Back Matter Stripping

**Front matter**: Scans for the first short (2--8 character) alphabetic headword with substantial following text. Everything before it is discarded as title pages, prefaces, and tables of contents.

**Back matter**: Removes candidates after markers like `FINIS`, `DIRECTIONS TO THE BINDER`, or `END OF VOL`.

#### 5.4 LIS Filtering (The Core Algorithm)

Uses patience sorting for O(n log n) performance on `(sort_key, char_start)` tuples:

1. Maintain a list of "piles" where each pile's top element is the smallest sort_key that can end a subsequence of that length.
2. For each candidate, use `bisect_right` to find which pile it extends.
3. Track parent pointers for backtracking.
4. The longest pile count is the LIS length; backtrack from the last pile to reconstruct the sequence.

Equal sort_keys at different positions are allowed (multi-sense entries like MERCURY the planet and MERCURY the element). The algorithm naturally eliminates:
- **Running headers**: They repeat at every page top but are out of alphabetical position relative to the surrounding articles.
- **Section headings**: "Of the Liver" inside a SURGERY article breaks the alphabetical sequence.
- **OCR noise**: Random ALL-CAPS words that don't form part of the encyclopedia's alphabetical sequence.

#### 5.5 Recovery Pass

Some legitimate articles are rejected by the LIS because they are very slightly out of alphabetical order (common with multi-part treatises or OCR errors in the first letter). The recovery pass re-inserts rejected candidates if:

- They appear in the cross-edition headword index (confirmed across multiple editions), AND
- They fit alphabetically between their positional neighbors in the accepted list.

Uses `bisect_left` for efficient neighbor lookup.

#### 5.6 Running Header Detection

For each accepted article's text span, examines single-newline candidates falling inside it:
- Same headword as the article title -> running header (skip)
- Different headword that fits alphabetically between neighbors -> missed boundary (insert, splitting the article)

#### 5.7 Dictionary-Guided Injection

Searches the OCR text for known headwords from a consolidated dictionary (49,812 entries from prior parses, Gemini corrections, and the 1842 General Index) that no regex pattern detected. Uses fast string search (`text.find()`) on a case-normalized copy. Only injects headwords confirmed by 2+ independent sources that fit alphabetically between neighbors. Confidence: 0.6.

#### 5.8 Supplementary Injection

Injects pre-validated missed headings from a JSONL file produced by a separate gap-detection pipeline (see Section 6). These are single-newline ALL-CAPS headings that the strict double-newline patterns missed.

#### 5.9 Mega-Article Splitting

Articles exceeding 50,000 words are searched for known dictionary headwords expected in the alphabetical gap between this article and the next. If found at a line boundary, the mega-article is split. This recovers articles that were absorbed when their headword fell on a single newline rather than a double newline.

#### 5.10 Post-Extraction Cleanup

A chain of cleanup passes:
- **Fragment consolidation**: Merges consecutive same-title articles when at least one is a fragment (<200 words).
- **False positive filtering**: Removes articles for known false-positive short words (IF, AS, etc.), merging their text into the predecessor.
- **Cross-reference reclassification**: Converts tiny articles (<25 words) containing "See" to cross-references; converts cross-references with >100 words back to articles.

### Cross-Edition Confidence Scoring

After the LIS parser produces articles, a separate cross-edition validation pass (`cross_edition.py`) exploits the fact that the same headwords appear across multiple editions:

**Pass 1 -- Union Index**: Scans all parsed articles across all 8 editions to build a union headword index of ~50,000 unique headwords.

**Pass 2 -- Scoring**: Assigns confidence based on cross-edition presence:
- 6+ editions: 0.95
- 3--5 editions: 0.80
- 2 editions: 0.70
- 1 edition only: 0.50

Combined confidence = `lis_confidence * cross_edition_confidence`.

**Pass 3 -- Recovery**: For headwords present in 5+ editions but missing from one, searches that edition's mega-articles for the missing headword and re-slices if found.

### Deduplication

Many NLS volumes were scanned multiple times, producing near-identical OCR files. The deduplication module (`dedup.py`) identifies duplicate scans using character-level similarity (>90% threshold) and Union-Find clustering, then selects a canonical file for each volume. This reduced 210 input files to 166 canonical files across 31 duplicate groups.

---

## 6. Supplementary Improvement: Missed Headings Pipeline

A semi-automated feedback loop to recover articles the LIS parser missed:

1. **Gap Detection** (`find_missed_headings.py`): Regex search for ALL-CAPS headings on single newlines (which the parser's double-newline requirement misses). Found 3,729 raw candidates across all editions.

2. **Gemini Classification**: Sent candidates in batches of 200 to Gemini Flash with surrounding context. Gemini classified each as "article heading" or "not an article heading" (running header, emphasis, section title, etc.). Produced 1,248 classified candidates.

3. **Alphabetical Filtering** (`filter_missed_headings.py`): For each candidate, found the parsed articles immediately before and after it in the OCR text by character position. If the candidate falls alphabetically between these neighbors, it is likely genuine. Also filtered pattern-based false positives (PLATE labels, Coptic glossary entries, epitaph inscriptions, Roman numerals, running header stubs).

4. **Injection**: 306 validated headings were injected into the supplementary headings file and fed back into the parser.

**Yield by edition**:

| Edition | Candidates | Injected |
|---------|-----------|----------|
| 1771 1st | 29 | -- |
| 1778 2nd | 60 | -- |
| 1797 3rd | 54 | -- |
| 1810 4th | 37 | -- |
| 1815 5th | 42 | -- |
| 1823 6th | 96 | -- |
| 1842 7th | **274** | -- |
| 1860 8th | 106 | -- |
| **Total** | **698** | **306** |

The 1842 7th edition produced 39% of all candidates, consistent with its complex typography and frequent single-newline headword formatting.

Notable recoveries included major articles: NAVIGATION, MAGNETISM, GARDENING, PRINTING, POLARISATION OF LIGHT, BIBLIOGRAPHY, UNITED STATES OF NORTH AMERICA, PORTUGAL, PRUSSIA, SCULPTURE, ROMAN HISTORY, and SURVEYING.

---

## 7. Missing Articles Investigation

After implementing parser gap fixes (period pattern, parenthesis pattern, titlecase extension), a systematic investigation checked whether major geographic articles were genuinely absent from the OCR or missed by the parser:

| Article | Status | Resolution |
|---------|--------|------------|
| INDIA 1797 | Recovered by gap fix | 45K words |
| ENGLAND 1860 | Recovered by gap fix | 75K words |
| EDINBURGH 1815 | Recovered by gap fix | 24K words |
| SPAIN 1842 | In OCR, titlecase format | Parser needs titlecase debug |
| CHINA 1815 | In OCR, titlecase format | Parser needs titlecase debug |
| EDINBURGH 1842 | In OCR, period format | Parser needs pattern debug |
| AFRICA 1842 | Not in OCR | Volume has gaps |
| SPAIN 1860 | Not in OCR | Genuinely missing |
| SCOTLAND 1860 | Not in OCR | Genuinely missing |
| AMERICA 1810 | Not in OCR | Supplement edition |

---

## 8. VLM Article Extraction (Experimental)

A parallel experimental approach used Qwen3-VL-235B (a vision-language model) to extract articles directly from PDF page images, bypassing the OCR -> text -> parser chain entirely.

**Method**: Convert PDF pages to 1200 DPI images (critical -- lower DPI causes VLM hallucination on these small-format NLS PDFs), send 3-page sliding windows to Qwen3-VL via OpenRouter, extract articles with page boundaries.

**Results on 1842 Vol 13** (790 pages):
- 1,022 articles extracted
- 0.6% genuine miss rate, 0% false positives
- **30% more articles** found than the OCR+parser pipeline
- Cost: $3.07 (~$0.004/page), ~22 hours sequential

The VLM approach consistently produced tighter, more accurate article boundaries. The main pipeline's systematic failure mode -- mega-articles absorbing adjacent entries -- was absent because the VLM reads each page directly rather than parsing continuous text streams.

A model shootout found Qwen2.5-VL-72B achieved 0.999 median similarity to the 235B model while fitting on a single GPU at INT4 quantization, making local deployment viable on a DGX Spark.

This approach remains experimental and has not been applied to the full corpus.

---

## 9. Final Corpus Statistics

| Metric | Value |
|--------|-------|
| **Editions** | 8 (1771--1860) |
| **Canonical OCR files** | 166 (from 210 total, 44 duplicates) |
| **Total exported articles** | 126,088 |
| **Total words** | 120,039,792 |
| **Volumes** | 134 |

**Per-edition breakdown**:

| Edition | Year | Volumes | Articles | Cross-refs | Words |
|---------|------|---------|----------|------------|-------|
| 1st | 1771 | 3 | 9,171 | 164 | 1.8M |
| 2nd | 1778 | 10 | 14,115 | 296 | 9.4M |
| 3rd | 1797 | 18 | 16,836 | 473 | 16.0M |
| 4th | 1810 | 20 | 18,606 | 538 | 17.7M |
| 5th | 1815 | 20 | 18,666 | 522 | 17.4M |
| 6th | 1823 | 20 | 16,636 | 472 | 17.5M |
| 7th | 1842 | 21 | 15,964 | 193 | 19.2M |
| 8th | 1860 | 21 | 12,799 | 637 | 21.3M |

---

## 10. Pipeline Files Reference

All scripts are in `/scripts/`:

| File | Role |
|------|------|
| `config.py` | Paths, editions, LLM settings |
| `parse_britannica.py` | CLI orchestrator (`--phase lis`, `--phase export`, etc.) |
| `lis_parser.py` | **Core parser**: candidate generation, LIS filtering, extraction |
| `cross_edition.py` | Cross-edition confidence scoring and false-negative recovery |
| `export.py` | Final JSONL + SQLite + statistics export |
| `generate_site.py` | Static GitHub Pages site generator |
| `find_missed_headings.py` | Gap detection (Gemini-classified single-newline headings) |
| `filter_missed_headings.py` | Alphabetical filtering of missed heading candidates |
| `dedup.py` | Source OCR file deduplication |
| `verify.py` | Quality assurance checks |
| `preprocess.py` | Legacy Phase 1: paragraph splitting |
| `classify.py` | Legacy Phase 2: LLM paragraph classification |
| `assemble.py` | Legacy Phase 3: article assembly |
| `merge.py` | Legacy Phase 3.5: fragment merging |
| `compare.py` | Comparison between parser versions |
| `gemini_mega_splitter.py` | Targeted mega-article splitting via Gemini |
| `models.py` | Data models |
| `audit_order.py` | Alphabetical order auditing |
| `test_gemini_split.py` | Tests for mega-splitter |

---

## 11. Lessons Learned

1. **Exploit domain structure, not just statistical classification.** The breakthrough came from recognizing that encyclopedia articles are alphabetically ordered -- a hard structural constraint that no amount of LLM classification can match. The LIS algorithm enforces this constraint globally, whereas the LLM classifies each paragraph independently.

2. **LLMs are expensive and unreliable for structural parsing.** Five GPU-days of DeepSeek-R1 classification produced worse results than a 40-second algorithmic parser. The LLM lacked the global view needed to distinguish running headers from article starts -- both look like ALL-CAPS headwords in isolation.

3. **Iterative correction is a trap.** The regex + Gemini correction pipeline kept adding layers of post-hoc fixes without addressing the fundamental extraction quality. Each layer added complexity and reduced reproducibility.

4. **Cross-edition validation is powerful.** The same ~50,000 headwords recur across 8 editions spanning 90 years. This provides a strong external signal for confirming or rejecting candidate headwords, independent of the OCR quality of any single edition.

5. **OCR quality is the binding constraint.** Several major articles (AFRICA 1842, SCOTLAND 1860, SPAIN 1860) are genuinely absent from the OCR because the source PDFs have gaps. No parser can recover what the OCR never captured.

6. **The hardest cases are format variations.** The biggest remaining gaps are articles in Title Case format (SPAIN 1842, CHINA 1815) or unusual punctuation patterns (EDINBURGH 1842). These require edition-specific pattern extensions rather than generic improvements.
