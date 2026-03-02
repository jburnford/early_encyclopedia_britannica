# Missing Major Articles Investigation

## Context

After implementing the parser gap fixes (Gap A: period pattern, Gap B: paren pattern, Gap C: titlecase extension, plus consolidation fix), we ran the full parser on all 190 canonical files. Results:

- **Total articles: 170,737**
- **Total cross-refs: 18,668**
- **Total words: 151,850,198**

The gap fixes recovered many major articles (INDIA 1797: 45K words, AFRICA 1797: 3K, PARIS 1797 city: 3K, EDINBURGH 1815: 24K, ENGLAND 1860: 75K, etc.).

However, several major geographic articles are still missing from the output. We investigated the raw OCR to determine if they're genuinely absent or if the parser is failing to detect them.

## Coverage Matrix (post gap-fix)

```
Place               1771    1778    1797    1810    1815    1823    1842    1860
PARIS             106/2  2K/4  3K/6  2K/2  2K/3  2K/2 10K/2 10K/2
FRANCE              *124 30K/2153K/4  175K174K/2153K/5219K/3    1K
INDIA               *152    ---   45K   60K   63K   60K   572   557
AFRICA              464  2K/3  3K/2   24K   24K 16K/3    ---   25K
SCOTLAND          181/2  207K 86K/2  218K  220K 44K/2 42K/3    ---
SPAIN              85/2 31K/3   17K   66K   67K 60K/2    ---    ---
AMERICA             275 22K/3 90K/2    ---  110K148K/2    ---   *6
BRAZIL               ---    ---   *39     ---    ---    ---   28K 18K/2
CHINA               *158 15K/3   48K   50K    --- 51K/2   51K   54K
EDINBURGH            ---   14K   24K   17K 24K/2 30K/2    --- 26K/2
ENGLAND             *88  83K/2   94K   95K   93K 94K/2 47K/2 75K/5
```

## Investigation Results

### 1842 Missing Volumes Note

The OCR manifest says 1842 (7th ed) is missing vols 5, 8, 9, 11, 12, 18. BUT all 22 volume files exist on disk and the parser processed all of them. The "missing" in the manifest means volume number couldn't be auto-detected, not that files are absent.

---

### SPAIN 1842 — IN OCR, PARSER MISSED (titlecase)

- **File**: `eb_7th_1842_v20_SEV-SUG.jsonl`
- **Format**: Titlecase `Spain` (not ALL-CAPS)
- **Size**: ~197,610 characters (very substantial)
- **Opening**: "Spain may be said to have been divided into two unequal parts..."
- **Why missed**: Gap C (titlecase). The titlecase pattern extension should have caught this IF "SPAIN" is in the headword dictionary. Need to verify:
  1. Is SPAIN in the headword dictionary?
  2. Does the titlecase pattern match this specific text format?
  3. Is the v20 file being processed by the parser? (YES - confirmed in parser logs)

### SPAIN 1860 — PROBABLY NOT IN OCR

- **File**: `eb_8th_1860_v21_ADA-ZWI.jsonl` has fragments
- **What's there**: Tartessus scholarly article mentioning Spain, Weights & Measures for Spain — NOT the main geography article
- **Conclusion**: The comprehensive Spain article does not appear to be in the 1860 OCR files

### AFRICA 1842 — NOT IN OCR

- **Searched**: All 1842 volume files
- **Result**: Only contextual mentions of "Africa" in other articles (ATLANTIC OCEAN, zoology articles, etc.)
- **Volume 4 (ADA-EXT)**: Should contain AF* articles but has mostly front matter and advertisements, jumps from A* to AT*
- **Conclusion**: Genuinely missing from OCR coverage. The volume that should contain it has gaps.

### AMERICA 1810 — NOT IN OCR (only cross-refs)

- **Searched**: All 41 files for 1810 (4th edition)
- **Result**: Only cross-references like "See AMERICA" found
- **Note**: 1810 4th ed is a supplement, volumes are NOT alphabetically ordered
- **Conclusion**: Article genuinely missing from 4th edition OCR

### AMERICA 1842 — IN OCR AS "UNITED STATES OF NORTH AMERICA"

- **File**: `eb_7th_1842_v21_SEV-ZYG.jsonl`
- **Format**: "UNITED STATES OF NORTH AMERICA" (different headword!)
- **Size**: ~213,909 characters / ~213KB
- **Opening**: "No single event in modern history has been of so much importance to mankind as the discovery of America..."
- **Why missed**: The headword is "UNITED STATES OF NORTH AMERICA" not "AMERICA". The parser may have found it under that title instead. Need to check if UNITED STATES appears in the parser output for 1842.

### AMERICA 1860 — CROSS-REF ONLY

- **File**: `eb_8th_1860_v02_ADA-GEN.jsonl`
- **Content**: "AMERICA, UNITED STATES OF. See UNITED STATES."
- **Conclusion**: Intentional cross-reference, not a missing article. Full article is under UNITED STATES.

### CHINA 1815 — IN OCR, PARSER MISSED (titlecase)

- **File**: `eb_5th_1815_v06_ENL-CRY.jsonl`
- **Format**: Titlecase `China,` followed by descriptive text
- **Opening**: "China, a country of Asia, situated on the most easterly part of that continent..."
- **Content**: Detailed geographic, administrative, and institutional information about China's provinces
- **Why missed**: Gap C (titlecase). Same issue as SPAIN 1842. Need to check if "CHINA" is in the headword dictionary and why the titlecase pattern didn't match.

### EDINBURGH 1842 — IN OCR, PARSER MISSED

- **File**: `eb_7th_1842_v08_DIA-VII.jsonl`
- **Format**: `EDINBURGH.` (ALL-CAPS heading, period, then descriptive text)
- **Opening**: "Edinburgh, a city, the capital of Scotland, and chief town of Mid-Lothian..."
- **Why missed**: This is `HEADWORD.` followed by text — exactly Gap A pattern (article_period). The parser HAS this pattern now. Need to check:
  1. Did the parser process v08? (YES - confirmed)
  2. Was EDINBURGH detected as a candidate?
  3. Was it rejected by LIS, consolidation, or some other filter?
  4. Is the text format `EDINBURGH.\n\nEdinburgh, a city...` (double newline) or `EDINBURGH.\nEdinburgh...` (single)?

### SCOTLAND 1860 — NOT IN OCR

- **File**: Searched all 21 volumes of 1860 8th edition
- **Result**: Only a cross-reference `(See SCOTLAND.)` in marketing/advertisement content
- **Expected location**: v19_ADA-SCY should contain SC* articles
- **Conclusion**: Genuinely missing from the 1860 OCR collection

---

## Summary

| Article | Status | Action Needed |
|---------|--------|---------------|
| SPAIN 1842 | **In OCR, titlecase** | Debug titlecase pattern for this case |
| SPAIN 1860 | Not in OCR | None (genuinely missing) |
| AFRICA 1842 | Not in OCR | None (volume has gaps) |
| AMERICA 1810 | Not in OCR | None (supplement edition) |
| AMERICA 1842 | **In OCR as "UNITED STATES"** | Check if found under different headword |
| AMERICA 1860 | Cross-ref to UNITED STATES | Check if UNITED STATES article exists |
| CHINA 1815 | **In OCR, titlecase** | Debug titlecase pattern for this case |
| EDINBURGH 1842 | **In OCR, ALLCAPS+period** | Debug article_period pattern for this case |
| SCOTLAND 1860 | Not in OCR | None (genuinely missing) |

## Next Steps

1. **Debug titlecase pattern** for SPAIN 1842 and CHINA 1815:
   - Verify these headwords are in the headword dictionary (`headword_dictionary.json`)
   - Check exact text format in OCR (what comes before the titlecase word?)
   - Run parser with debug logging on the specific volume files

2. **Debug article_period pattern** for EDINBURGH 1842:
   - Check exact OCR text format around `EDINBURGH.`
   - Check if candidate was generated but filtered by LIS or another stage

3. **Check UNITED STATES** in parser output for 1842 and 1860:
   - `grep "UNITED STATES" /home/jic823/plato/britannica_output/articles/eb_7th_1842_*.jsonl`
   - `grep "UNITED STATES" /home/jic823/plato/britannica_output/articles/eb_8th_1860_*.jsonl`

4. **Genuinely missing** (no action): AFRICA 1842, SPAIN 1860, AMERICA 1810, SCOTLAND 1860

## Files Modified in Gap Fix

- `/home/jic823/plato/britannica_parser/lis_parser.py` — all pattern changes
- `/home/jic823/plato/britannica_parser/config.py` — fixed local paths
- `/home/jic823/1815EncyclopediaBritannicaNLS/docs/parser_gap_fix_plan.md` — original plan
