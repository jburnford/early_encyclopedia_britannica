# Next Session: Continuing Article Recovery and Parser Fixes

## Session Summary (Mar 29, 2026 — Session 2)

Reduced gaps from 1,788 → 1,777 (-11) and fixed major content attribution issues:

### Splits (11 new)
- **SCOT (1860)** → SCOT bio (129w) + SCOTLAND history (63K)
- **ROMANO (1823)** → ROMANO bio (246w) + ROME history (42K)
- **BOND (1771)** → BOND (1.6K) + BOOK-KEEPING (42K)
- **MATERA (1810, 1815)** → MATERA (32w) + MATERIA MEDICA (46K, 29K)
- **ENGRAILED (1842, 1860)** → ENGRAILED + ENGRAVING (14K, 19K)
- **PERSHORE (1842, 1860)** → PERSHORE + PERSIA (20K, 15K)
- **NET (1842)** → NET (408w) + NETHERLANDS (31K)
- **ZYGOMATICUS (1778)** → ZYGOMATICUS (22w) + APPENDIX (146K)
- **PERSONIFYING (1797)** → PERSONIFYING (1.2K) + PERSPECTIVE (19K)
- **BURNING (1810)** → BURNING (5.8K) + BURNS (7.2K)

### Merges (6)
- **ORDER (1778, 1797, 1815, 1823)** → merged back into ORATORY (~113K each)
- **PART (1810)** → merged into ORATORY (81K → 86K total)
- **INDIAN (1810)** → merged into INDIA (52K → 60K total)

### Deletes (12)
- **WEEK (1810, 88K)** — misattributed MEDICINE fragment
- **STRAIN (1842, 85K)** — misattributed ORNITHOLOGY fragment
- **WHITE (1842, 62K)** — misattributed MAGNETISM fragment
- **AAA (1823, 87K)** — contributor key, not article
- **VOCAL (1842, 37K)** — misattributed English history fragment
- **THUS (1797×2, 1810)** — false headword fragments (62K + 7K + 21K)
- **GENUS IX (1797, 1823)** — medical subsection, not standalone (104K + 19K)
- **LOGARITHMS OF NUMBERS (1810-1823)** — numerical tables, not articles (60K total)

### Relabels (6)
- **SLAUGHTER (1810, 1815, 1823)** → SLAVERY (14K each)
- **SWEDEN IS BY NO (1815, 1823)** → SWEDEN (23K, 24K)
- **AMERICA IS BY NO (1778)** → AMERICA (22.5K)

### Current Gap Status
| Classification | Count | Change |
|----------------|-------|--------|
| PARSING_OR_EDITORIAL | 967 | -5 |
| OCR_GAP | 326 | -2 |
| VARIANT | 245 | -1 |
| EDITORIAL | 211 | -4 |
| SWALLOWED | 28 | +1 |

**Total: 1,777 gaps** (was 1,788). Index: 4,354 substantive articles.

## What Still Needs Doing

### 1. Remaining Mega-Article Swallowers (Medium Value)

From the original list, these were analyzed and determined to be **legitimate long articles** (not swallowed):
- STONE-MASONRY (1860, 26K) — genuine masonry article
- CENTER (1810-1823, ~15K each) — legitimate geometry/engineering article
- FLUX (1860, 74K) — legitimate FLUXIONS (calculus) article
- PHYSIC (1842, 83K) — legitimate "Practice of Physic" article
- HOUND (1860, 13K) — legitimate article about dogs/hunting
- LOGARITHMIC CURVE (1797, 28K) — mostly legitimate, slight tail from LOGIC

These may need Wikidata matching but no parser fixes.

### 2. Handle the 253 Articles Swallowed in Mega-Articles (Medium Value)

Analysis found 253 articles exist as mentions inside parsed mega-articles. The biggest swallowers:
- AGRICULTURE (144K 1810, 296K 1823) → swallowed 14 and 10 articles
- BRITAIN (326K 1810, 324K 1823) → swallowed 9 and 7 articles
- FRANCE (175K 1810) → swallowed 8 articles

**Challenge**: These are genuinely massive articles. Some "swallowed" content is legitimate sub-sections.

### 3. Recover More Mixed-Case OCR Headwords

`recover_from_ocr.py` currently handles ALLCAPS and topic-validated mixed-case. ~400 single-word articles in mixed case still fail.

### 4. Partial-OCR Edition Gaps (Blocked)

401 gaps in editions with missing volumes (1815, 1842, 1860). Unrecoverable without new OCR.

### 5. Cross-Edition Index Quality

Remaining non-article entries to review:
- Small THUS entries (67-94w in multiple editions) — legitimate cross-references, harmless
- ORDER still appears in index (135w 1771, 8K 1815 religious orders, 782w 1842)
- CLASSIS IV — check if still in article files

## Key Files

| File | Purpose |
|------|---------|
| `scripts/fix_mega_articles.py` | Manual article fixes — **108 split specs, 6 merges, 12 deletes, 6 relabels** |
| `scripts/recover_from_ocr.py` | Extract articles from raw OCR by headword search |
| `scripts/classify_gaps.py` | Gap classification pipeline |
| `scripts/rebuild_cross_edition_index.py` | Rebuild cross-edition index from article files |
| `data/gap_classifications.csv` | Current gap triage — 1,777 gaps |
| `data/cross_edition_index.jsonl` | 4,354 substantive articles tracked across editions |
| `data/ambiguous_headwords.md` | 95 problem headwords with per-edition analysis |

## Pipeline After Fixes

```bash
python scripts/fix_mega_articles.py
python scripts/merge_fragments.py
python scripts/parse_britannica.py --phase export
python scripts/rebuild_cross_edition_index.py
python scripts/classify_gaps.py
```
