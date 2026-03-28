# Parsing & Article Status — 27 March 2026

## Corpus Overview

| Edition | Year | Articles | Cross-refs | Words | Volumes |
|---------|------|----------|------------|-------|---------|
| 1st | 1771 | 8,269 | 1,752 | 1,762,486 | 3 |
| 2nd | 1778 | 15,013 | 1,395 | 9,408,528 | 10 |
| 3rd | 1797 | 18,850 | 1,561 | 15,980,363 | 18 |
| 4th | 1810 | 18,473 | 2,903 | 17,674,398 | 20 |
| 5th | 1815 | 18,073 | 2,784 | 17,404,123 | 20 |
| 6th | 1823 | 16,509 | 2,612 | 17,784,301 | 21 |
| 7th | 1842 | 16,817 | 401 | 19,181,150 | 21 |
| 8th | 1860 | 14,194 | 858 | 21,448,560 | 21 |
| **Total** | | **126,198** | **14,266** | **120,643,909** | **134** |

**Site**: 136,478 searchable entries (articles + cross-references) at 117.6M words.

## Changes Since Last Report

### Running-Header Fragment Merge (27 Mar 2026)

OLMoCR sometimes preserves running page headers (e.g., "SHIP-BUILDING" at the top of every page in a treatise). The LIS parser's alphabetical-sequence filter correctly allows equal sort keys with increasing positions, which means these running headers survived as article boundaries, splitting one long treatise into many fragments that break mid-sentence.

**New script: `scripts/merge_fragments.py`**

Two-pass detection:
1. **Mid-sentence boundary test**: If the previous fragment ends without terminal punctuation or the next starts with a lowercase/continuation word, it's a running-header split.
2. **Char-span coverage test**: If same-headword fragments tile >80% of their combined character range, they're fragments of one article regardless of sentence boundaries.

**Result**: 1,705 excess articles merged across 1,092 headwords.

Before/after for worst cases:

| Article | Edition | Before | After |
|---------|---------|--------|-------|
| SHIP-BUILDING | 1810 | 40 fragments | 1 article (67,794w) |
| HYDRODYNAMICS | 1815 | 39 fragments | 1 article (92,902w) |
| SHIP-BUILDING | 1823 | 37 fragments | 1 article (67,432w) |
| CHEMISTRY | 1810 | 36 fragments | 1 article (358,547w) |
| AGRICULTURE | 1815 | 28 fragments | 1 article (291,092w) |
| MEDICINE | 1778 | 26 fragments | 1 article (314,709w) |
| ASTRONOMY | 1815 | 49 fragments | 1 article (204,739w) |

### Mega-Article Fixes (27 Mar 2026)

**New script: `scripts/fix_mega_articles.py`**

Hand-specified fixes for articles that swallowed neighbors or had broken headwords:

| Fix | Edition | Detail |
|-----|---------|--------|
| BOSWORTH-MARKET split | 1860 | Recovered BOTAL (555w) and BOTANY (178,022w) |
| UNIVERSITY OF PARIS split | 1860 | Recovered 11 sub-articles (Oxford, Cambridge, London, Glasgow, Aberdeen, Edinburgh, Dublin, Colonial, France) |
| UNIVERSITY OF PARIS split | 1842 | Recovered 8 sub-articles |
| "SCOTLAND IS BY NO" rename | 1815 | → SCOTLAND (217,873w) |
| "ANTAGONISTS OF HOBBIESTS" rename | 1842 | → DISSERTATIONS (51,003w) |
| "CLOCK AND WATCH WORK" rename | 1842 | → CLOCKS (50,891w) |
| HYDRODYNAMICS trim | 1810/1815/1823 | Removed trailing INDEX/DIRECTIONS matter |

### Timber Search & Reports (26-27 Mar 2026)

**New script: `scripts/search_timber.py`**

Searched all 136K articles for content related to timber, wood rot, naval shipbuilding, oak/teak/pine, fungi, and preservation. Two output categories:
- **331 topical articles** (directly about timber/rot/shipbuilding)
- **1,475 chunk extractions** (relevant passages from other articles)

Generated three reports:
- `data/timber_analysis.md` — 10-section analytical report tracing timber discourse across editions
- `data/timber_science_comparison.md` — modern mycological assessment of the Britannica's claims
- `data/timber_report.md` — auto-generated article listings with links

## Remaining Known Issues

### Duplicate Headwords (34 remaining)

Down from ~4,600+ to 34 excess articles across 33 headwords. Almost all are genuine multi-sense entries:

| Edition | Headword | Count | Notes |
|---------|----------|-------|-------|
| 1860 | PITT | 3x | William Pitt the Elder, the Younger, and the place |
| Various | BOLOGNESE, BOII, etc. | 2x each | Genuine person/place homonyms |

These are correct — the parser should produce multiple articles for distinct entities sharing a name.

### Genuine Mega-Articles (115 articles >50K words)

The Britannica published book-length treatises on major subjects. These are genuine, not parsing errors:

| Article | Editions | Words (range) | Notes |
|---------|----------|---------------|-------|
| BRITAIN | 1810-1860 | 324K-381K | Full history, geography, institutions |
| CHEMISTRY | 1797-1860 | 282K-359K | Complete chemical textbook |
| MEDICINE | 1778, 1815 | 315K-321K | Full medical encyclopedia |
| ASTRONOMY | 1797-1860 | 203K-217K | Comprehensive astronomical treatise |
| SCOTLAND | 1778-1815 | 208K-218K | Full history and geography |
| AGRICULTURE | 1815 | 291K | Complete farming manual |
| DISSERTATIONS | 1842, 1860 | 215K-415K | Preliminary dissertations on philosophy |
| FRANCE | 1842 | 271K | Full history |
| ENTOMOLOGY | 1842, 1860 | 254K-269K | Complete insect taxonomy |
| OPTICS | 1860 | 203K | Full optics textbook |

25 articles exceed 200K words. 115 exceed 50K. These represent the Britannica's distinctive contribution to knowledge dissemination — multi-volume treatises by leading scholars, embedded within the alphabetical reference work.

### MEDICINE and BOTANY Internal Structure

MEDICINE (1778: 315K, 1815: 321K) contains numbered aphorisms, disease classifications, and sub-sections that look like headwords but are internal structure (PHRENSY, VOMICA, MEASLES, CHICKENPOX etc.). Similarly, BOTANY (1810: 181K, 1823: 194K) contains Linnaean class headings (CLASSIS I, ORDO I, genus names). These are correctly treated as single articles with internal structure, not as swallowed neighbors.

### OCR Coverage Gaps

From the OCR manifest, not all volumes have NLS PDF scans:

| Edition | Volumes Available | Missing |
|---------|-------------------|---------|
| 1771 1st | 3/3 | — |
| 1778 2nd | 10/10 | — |
| 1797 3rd | 18/18 | — |
| 1810 4th | 19/20 | vol 20 |
| 1815 5th | 11/20 | vols 3,7,9,10,12,13,15,19,20 |
| 1823 6th | 20/20 | — |
| 1842 7th | 21/21 | — |
| 1860 8th | 21/21 | — |

Missing volumes for 1810 (5th supplement vol) and 1815 (9 volumes) are filled from `docs_old` fallback data where available.

## Pipeline Summary

```
OCR (OLMoCR on NLS PDFs)
  → lis_parser.py (LIS headword detection + article extraction)
    → merge_fragments.py (running-header fragment merge)
      → fix_mega_articles.py (manual mega-article splits/renames)
        → export.py (consolidate to per-edition JSONL + SQLite)
          → generate_site.py (static HTML site with search)
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/lis_parser.py` | Main article extraction (LIS algorithm) |
| `scripts/merge_fragments.py` | Merge running-header fragments |
| `scripts/fix_mega_articles.py` | Manual mega-article splits/renames |
| `scripts/export.py` | Consolidate to JSONL + SQLite |
| `scripts/generate_site.py` | Generate static HTML site |
| `scripts/search_timber.py` | Thematic search with chunk extraction |

### NER Pipeline (complete, run on Narval cluster)

- 133 per-volume jobs across 8 editions
- 119,851 articles → 1,157,244 entities (TOPONYM, PERSON, ORG, COMMODITY)
- Toponym disambiguation: 94.3% grounded to Wikidata
- Person disambiguation: 1,194 matches to Wikidata QIDs
- Output: `data/ner/` (not tracked in git, 333MB)
