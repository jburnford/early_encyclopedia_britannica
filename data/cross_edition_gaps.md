# Cross-Edition Gap Analysis — 27 March 2026

Using the 1815 (5th edition) as reference, all substantive articles (>=1000 words) were checked for presence across all 8 editions. "Suspicious gaps" are articles present in editions both before AND after the missing edition.

## Summary

| Missing from | Gaps (6+ ed present) | Gaps (5+ ed) | Notable |
|-------------|---------------------|--------------|---------|
| 1778 (2nd) | 13 | 19 | DYEING, FIRE, LONGITUDE |
| 1797 (3rd) | 26 | 81 | PYROTECHNY, MALTA, CONIC SECTIONS |
| 1810 (4th) | 34 | 132 | AMERICA, ORNITHOLOGY, PERSIA |
| 1815 (5th) | 11 | — | NAVIGATION (known OCR gaps in 9 vols) |
| 1823 (6th) | 29 | 173 | ROME, MECHANICS, SILK |
| 1842 (7th) | 68 | 193 | **AGRICULTURE**, MEDICINE, PHILOLOGY |
| 1860 (8th) | — | — | (reference end, gaps are editorial removal) |

## Critical Findings

### 1842 Missing OCR Volume (ACE-ANA range)

**68 articles** present in 6+ other editions are missing from the 1842 (7th) edition. Investigation shows:

- 1842 vol 1-2 contain preliminary Dissertations
- 1842 vol 3 starts at ANATOMY
- The entire ACE-ANA range (including AGRICULTURE, AFRICA, ACADEMY, ACOUSTICS, etc.) is missing
- An OCR file `eb_7th_1842_v04_ADA-EXT.jsonl` exists but is **misattributed** — its text begins with "ENCYCLOPÆDIA BRITANNICA, EIGHTH EDITION" and has `edition_year: None`
- This file is likely 8th edition (1860) vol 4 misfiled under 1842

**Root cause identified**: The OCR file for 1842 vol 2 (`eb_7th_1842_v02_SEV-ADA.jsonl`) contains only 5,865 chars of front matter (title page and editorial notes). The actual articles — from AARDVARK through the A's, including AGRICULTURE starting at page 251 (image 260) — were never OCR'd. The NLS PDF exists and has the content; OLMoCR only processed the first few pages.

**Action needed**: Re-process the 1842 vol 2 NLS PDF through OLMoCR to recover the full A-section articles. Additionally, the file `eb_7th_1842_v04_ADA-EXT.jsonl` is misattributed 8th edition content and should be removed or relabelled.

### 1810 Missing Articles (132 gaps)

The 4th edition (1810) has the most gaps of any edition with full OCR coverage. Many large articles are missing: AMERICA (110K), ORNITHOLOGY (109K), ORDER (64K), PERSIA (25K), ROOF (23K), BOOK-KEEPING (20K). This edition has 20 OCR volumes but the parser reports 19/20 usable.

**Possible cause**: One or more OCR volumes may have range-label errors causing the parser to skip articles in certain alphabetical ranges.

### 1823 Missing ROME (155K words)

ROME (154K words) is present in editions 2-5 and 7-8 but missing from the 6th (1823). This is the largest single missing article by word count. Given ROME's size and importance, this is very likely a parsing or OCR gap.

### 1815 Missing NAVIGATION (49K-94K words)

NAVIGATION is present in all other editions (40K-94K words) but missing from 1815. This edition has 9 known missing OCR volumes (vols 3,7,9,10,12,13,15,19,20). NAVIGATION would fall in vol 14-15 range — vol 15 is one of the missing volumes.

## Top Missing Articles by Word Count

Articles present in 6+ editions but missing from one middle edition.

**NOTE**: Some "gaps" are actually headword changes between editions, not parsing errors. These are marked below. Only unmarked entries are suspected parsing/OCR gaps.

| Article | Missing | Typical size | Explanation |
|---------|---------|-------------|-------------|
| ROME | 1823 | 154K | **RENAMED**: appears as ROMANS (113K) + ROMANO (42K) in 1823 |
| AGRICULTURE | 1842 | 144K | **OCR GAP**: missing ACE-ANA volume (see below) |
| AMERICA | 1810 | 110K | Suspected OCR/parsing gap |
| ORNITHOLOGY | 1810 | 106K | Suspected OCR/parsing gap |
| MECHANICS | 1823 | 103K | Genuinely missing — suspected OCR/parsing gap |
| PHILOLOGY | 1842 | 88K | **RENAMED**: appears as LANGUAGE (38K) in 1842 |
| MEDICINE | 1842 | 27K-321K | **REORGANIZED**: split into PHYSIC (83K), PATHOLOGY (21K), MEDICAL JURISPRUDENCE (49K) |
| ABYSSINIA | 1842 | 66K | **RENAMED**: appears as ETHIOPIAN NATIONS (8K) — also partly an OCR gap |
| RIVER | 1823 | 59K | **RENAMED**: appears as RIVERS (57K) in 1823 |
| DYEING | 1778 | 54K | Suspected OCR/parsing gap |
| NAVIGATION | 1815 | 49K | **KNOWN OCR GAP**: vol 15 missing from NLS scans |
| SILK | 1823 | 11K | Genuinely missing — suspected OCR/parsing gap |
| AFRICA | 1842 | 24K | **OCR GAP**: same missing ACE-ANA volume |
| BLIND | 1842 | 26K | Suspected OCR/parsing gap (may be in missing volume) |
| SCULPTURE | 1842 | 13K | Suspected OCR/parsing gap |

## Likely Causes

1. **Missing OCR volumes**: 1815 has 9 known missing vols; 1842 appears to have a missing ACE-ANA volume
2. **Misattributed OCR files**: The ADA-EXT file contains 8th edition text filed under 7th edition
3. **Range label errors**: OCR volume range metadata may be wrong, causing parser to assign articles to wrong volumes or skip ranges
4. **Mega-article swallowing**: Some missing articles may be buried inside neighboring mega-articles (already partially addressed by `fix_mega_articles.py`)
5. **Genuine editorial changes**: Some gaps are real — articles were added, reorganized, or renamed between editions. This is more common for the 1842/1860 transition where the Britannica underwent major restructuring.

## Methodology

- Reference edition: 1815 (5th), all articles >= 1000 words (1,919 articles)
- Matching: normalized headwords (uppercase, alphanumeric only)
- Threshold: article must be >= 100 words in an edition to count as "present"
- A "suspicious gap" requires the article to be present in at least one earlier AND one later edition

## Next Steps

1. Investigate 1842 vol 3 range: is there an NLS PDF covering ACE-ANA that wasn't OCR'd?
2. Check the misattributed `eb_7th_1842_v04_ADA-EXT.jsonl` — confirm it's 8th edition
3. Investigate 1810 gaps: check OCR volume ranges for missing alphabetical coverage
4. Cross-reference 1823 gaps against OCR volume ranges
5. Consider running this analysis from each edition as reference (not just 1815) to catch articles that were added after 1815
