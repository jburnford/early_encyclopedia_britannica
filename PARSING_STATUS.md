# Encyclopedia Britannica Parsing Status Report

**Date:** January 23, 2026
**Repository:** [github.com/jburnford/early_encyclopedia_britannica](https://github.com/jburnford/early_encyclopedia_britannica)
**Website:** [jburnford.github.io/early_encyclopedia_britannica](https://jburnford.github.io/early_encyclopedia_britannica)

---

## Overview

This project provides OCR-parsed text from eight historical editions of the Encyclopaedia Britannica (1771-1860), hosted as a static GitHub Pages website with searchable articles.

### Current Statistics

| Edition | Year | Volumes | Articles | Gemini Corrections | Status |
|---------|------|---------|----------|-------------------|--------|
| 1st | 1771 | 4 | 11,609 | 25 chunks | ✅ Complete |
| 2nd | 1778 | 13 | 30,562 | 196 chunks | ✅ Complete |
| 3rd | 1797 | 19 | 36,352 | 126 chunks | ✅ Complete |
| 4th | 1810 | 20 | 9,165 | 78 chunks | ⚠️ Supplement edition |
| 5th | 1815 | 20 | 18,584 | 179 chunks | ✅ Complete |
| 6th | 1823 | 21 | 16,457 | 168 chunks | ✅ Complete |
| 7th | 1842 | 22 | 18,692 | 698 chunks | ✅ Complete |
| 8th | 1860 | 22 | 12,017 | 699 chunks | ✅ Complete |

**Total: 141 volumes, 153,438 articles**

---

## Data Pipeline

### Source Data
- **OCR Engine:** OLMoCR (Optical Language Model OCR)
- **Source PDFs:** National Library of Scotland digitization
- **Raw OCR Location:** `ocr_results/` directory (not in git)

### Processing Steps

1. **Initial Parsing** (`output_v2/articles_*.jsonl`)
   - Extracted article boundaries from raw OCR
   - Identified headwords, page numbers, volume assignments

2. **Gemini Correction** (`gemini_*_all.json`)
   - Used Gemini Flash API to re-parse problematic page ranges
   - Corrected article boundary detection errors
   - Merged fragmented articles, removed false positives

3. **Rebuild** (`rebuild_from_gemini.py`)
   - Applied Gemini corrections to article JSON
   - Created `*_corrected.json` files in `docs/*/data/`

4. **HTML Generation** (`regenerate_html.py`)
   - Generated browsable HTML for each volume
   - Created search index and cross-reference links

---

## Known Issues

### Parsing Issues

| Issue | Severity | Editions Affected | Description |
|-------|----------|-------------------|-------------|
| Missing articles | Medium | All | Some articles not extracted from OCR due to unusual formatting |
| False positives | Low | All | Some non-article text parsed as articles (mostly corrected by Gemini) |
| Fragmented articles | Low | 1771, 1842 | Long treatises sometimes split incorrectly |
| Page number errors | Low | All | OCR occasionally misreads page numbers |

### Edition-Specific Issues

**1771 (1st Edition)**
- Original 3-volume structure with generated vol0 index
- Some articles like TANNING required manual extraction from raw OCR
- Treatises (CHEMISTRY, GEOGRAPHY, etc.) may have parsing issues

**1778 (2nd Edition)**
- Includes main volumes + supplement (vol10_main, vol10_supplement)
- Large file warnings on GitHub (vol0.json > 55MB)

**1797 (3rd Edition)**
- Largest edition by article count
- Vol0 index file exceeds 98MB (GitHub warning)

**1810 (4th Edition - Supplement)**
- Not a complete encyclopedia - only contains new/updated articles
- Non-alphabetical volume organization
- Missing articles like TANNING are expected (not in supplement)

**1842 (7th Edition)**
- Complex multi-part articles
- Most chunks processed by Gemini (698)

**1860 (8th Edition)**
- Some articles are cross-references (e.g., "TANNING. See Leather.")
- May need to follow cross-references for complete content

### Technical Debt

- Large JSON files trigger GitHub warnings (50MB+ recommended limit)
- Cross-reference hyperlinks not yet implemented in corrected articles
- Search index may not include recently corrected articles

---

## File Structure

```
docs/
├── index.html              # Main landing page
├── search.html             # Search interface
├── about.html              # Project information
├── api/index.json          # Search index
├── {year}/                 # Edition directories
│   ├── index.html          # Edition landing page
│   ├── vol{N}.html         # Volume HTML pages
│   └── data/
│       ├── vol{N}.json           # Article data (current)
│       ├── vol{N}_corrected.json # Gemini-corrected version
│       └── vol{N}_original.json  # Pre-correction backup
```

---

## Recent Updates

### January 23, 2026
- Applied Gemini corrections to all 8 editions
- Extracted missing TANNING article from 1771 raw OCR
- Created research files: `tanning_articles.md`, `tanning_articles.json`
- Pushed updated site to GitHub

### January 22, 2026
- Completed Gemini reparsing for 1860 edition (699 chunks)
- Rebuilt HTML for all editions

### Earlier
- Implemented Gemini Flash API integration for article boundary correction
- Added volume outlier detection system
- Created precision reparsing pipeline

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `gemini_reparse_v2.py` | Send OCR to Gemini for article boundary detection |
| `rebuild_from_gemini.py` | Apply Gemini corrections to article JSON |
| `regenerate_html.py` | Generate HTML from article JSON |
| `generate_site.py` | Full site generation with cross-references |
| `find_page_mismatches.py` | Detect parsing errors via page conflicts |

---

## Next Steps

### High Priority
1. [ ] Add cross-reference hyperlinks to corrected articles
2. [ ] Investigate remaining parsing issues in 1771 treatises
3. [ ] Reduce large file sizes (consider splitting vol0 indexes)

### Medium Priority
4. [ ] Improve search index to include all corrected content
5. [ ] Add API documentation for programmatic access
6. [ ] Create volume-level statistics page

### Low Priority
7. [ ] Implement Git LFS for large data files
8. [ ] Add article comparison view across editions
9. [ ] Generate citation formats for articles

---

## Contributing

The raw OCR data and processing scripts are maintained locally. To report parsing errors or request specific article extractions, please open an issue on the GitHub repository.

---

*Generated: January 23, 2026*
