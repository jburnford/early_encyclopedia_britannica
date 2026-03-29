# Next Session: Continuing Article Recovery and Parser Fixes

## Session Summary (Mar 28-29, 2026)

Reduced gaps from 2,201 → 1,788 (-413, 19% reduction):
- **76 swallowed articles split** from hosts via `fix_mega_articles.py` (variant→swallow + PARSING_OR_EDITORIAL→swallow)
- **323 articles extracted** from raw OCR via new `scripts/recover_from_ocr.py`
- **6 Preliminary Dissertations** cleaned up as standalone articles (784K words)
- **1.3M words of Dissertations junk** removed from edition files (DISS, WAY, STATICS, etc.)

### Current Gap Status
| Classification | Count | Notes |
|----------------|-------|-------|
| PARSING_OR_EDITORIAL | 972 | Biggest bucket — see analysis below |
| OCR_GAP | 328 | In alphabetical ranges with no OCR coverage |
| VARIANT | 246 | Headword renames (RIVER/RIVERS etc.) |
| EDITORIAL | 215 | Articles genuinely removed between editions |
| SWALLOWED | 27 | Detected by classifier's own logic |

## What Needs Doing

### 1. Split the Remaining Mega-Article Swallowers (High Value)

From `data/ambiguous_headwords.md`, 31 mega-articles have massive word-count outliers — they swallowed subsequent articles. The pattern is a short definition (WEEK, 37w) suddenly ballooning to 88K in one edition because the parser failed to detect the next headword.

**Key example**: ORDER (60-79K in 1778-1823) is the second half of a long article on ORATORY. The parser broke at "Order" mid-text and created a fake article. The real ORATORY article lost its tail, and ORDER gained 60K+ words of rhetoric/oratory content. Similar pattern for many others.

The biggest:

| Headword | Swallowed Edition | Size | Likely Content |
|----------|-------------------|------|----------------|
| ZYGOMATICUS | 1778 | 146K | Swallowed tail of volume |
| WEEK | 1810 | 88K | Swallowed ~88K |
| STRAIN | 1842 | 85K | Swallowed ~85K |
| PHYSIC | 1842 | 83K | Cross-ref (32w) swallowed ~83K |
| PART | 1810 | 81K | Swallowed ~81K |
| ORDER | 1778-1823 | 60-79K each | Tail of ORATORY article |
| GENUS | 1823 | 75K | Swallowed ~75K |
| FLUX | 1860 | 74K | Swallowed ~74K |
| SCOT | 1860 | 63K | Swallowed ~63K |
| WHITE | 1842 | 62K | Swallowed ~62K |
| BOND | 1771 | 44K | Swallowed ~44K |
| ROMANO | 1823 | 42K | Swallowed ~42K |
| AAA | 1823 | 87K | Contributor key + Dissertations — delete |

**Approach**: For each, read the text to find where the real article ends and the swallowed content begins. Many contain identifiable headword boundaries in mixed case. Add splits to `fix_mega_articles.py`. Some (like ORDER) need to be merged back into their parent article (ORATORY) rather than split.

### 2. Handle the 253 Articles Swallowed in Mega-Articles (Medium Value)

Analysis found 253 articles exist as mentions inside parsed mega-articles. The biggest swallowers:
- AGRICULTURE (144K 1810, 296K 1823) → swallowed 14 and 10 articles
- BRITAIN (326K 1810, 324K 1823) → swallowed 9 and 7 articles
- FRANCE (175K 1810) → swallowed 8 articles

**Challenge**: These are genuinely massive articles. Some "swallowed" content is legitimate sub-sections. Need to read text around each candidate to verify.

### 3. Recover More Mixed-Case OCR Headwords

`recover_from_ocr.py` currently handles ALLCAPS and topic-validated mixed-case. ~400 single-word articles in mixed case still fail. Could try:
- Embedding-based matching (compare to article text from another edition)
- Looser topic validation
- Reading first 1000-2000 words and finding topic shifts (as we did for LOMBARDY)

### 4. Partial-OCR Edition Gaps (Blocked)

401 gaps in editions with missing volumes:
- **1815 5th**: Missing 9 of 20 volumes
- **1842 7th**: Missing 6 of 21 volumes
- **1860 8th**: Missing 2 of 21 volumes

Unrecoverable without new OCR from NLS PDFs.

### 5. Clean Up Cross-Edition Index

Remove non-article entries: GENUS IX, CLASSIS IV, LOGARITHMS OF NUMBERS, SWEDEN IS BY NO, AMERICA IS BY NO, THUS, AAA.

### 6. Move Index Rebuild Script

The index rebuild script is currently in `/tmp/rebuild_index.py` — move to `scripts/rebuild_cross_edition_index.py`.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/fix_mega_articles.py` | Manual article splits — **97 fixes total** |
| `scripts/recover_from_ocr.py` | Extract articles from raw OCR by headword search |
| `scripts/classify_gaps.py` | Gap classification pipeline |
| `data/gap_classifications.csv` | Current gap triage — 1,788 gaps |
| `data/cross_edition_index.jsonl` | 4,369 substantive articles tracked across editions |
| `data/ambiguous_headwords.md` | 95 problem headwords with per-edition analysis |
| `data/articles/dissertations.articles.jsonl` | 6 clean Preliminary Dissertations |

## Pipeline After Fixes

```bash
python scripts/fix_mega_articles.py
python scripts/merge_fragments.py
python scripts/parse_britannica.py --phase export
python scripts/rebuild_cross_edition_index.py  # TODO: create from /tmp/rebuild_index.py
python scripts/classify_gaps.py
```
