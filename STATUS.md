# Encyclopaedia Britannica Historical Corpus - Project Status

## Overview
Extracting and publishing OCR'd text from 7 editions of the Encyclopaedia Britannica (1771-1860) as a browsable website and structured data.

## Repository
- **GitHub**: `github.com:jburnford/early_encyclopedia_britannica.git`
- **GitHub Pages**: Enabled from `/docs` folder

## Key Files

| File | Purpose |
|------|---------|
| `extract_britannica_corpus.py` | Main extraction script - parses OCR output into articles |
| `generate_site.py` | Generates static HTML site from extracted articles |
| `docs/` | Generated website (committed to git) |
| `output_v2/` | Extracted JSONL files (not in git - too large) |
| `ocr_results/` | Source OCR files from OLMoCR (not in git) |
| `json/` | 1815 edition OCR in JSON format (not in git) |

## Current Status: 102,036 Articles

| Edition | Year | Volumes | Articles | Treatises |
|---------|------|---------|----------|-----------|
| 1st | 1771 | 3 | 11,351 | 360 |
| 2nd | 1778 | 10 | 13,948 | 1,003 |
| 4th | 1810 | 20 | 9,822 | 1,003 |
| 5th | 1815 | 19 | 18,178 | 1,625 |
| 6th | 1823 | 20 | 15,748 | 1,820 |
| 7th | 1842 | 22 | 18,528 | 2,137 |
| 8th | 1860 | 22 | 14,461 | 2,369 |

## Output Format (JSONL)

```json
{
  "article_id": "1771_v01_ASTRONOMY",
  "headword": "ASTRONOMY",
  "text": "full article text...",
  "article_type": "treatise|dictionary|cross_reference",
  "edition_year": 1771,
  "edition_name": "Britannica 1st",
  "volume_id": "...",
  "volume_num": 1,
  "start_page": 449,
  "end_page": 495,
  "char_start": 123456,
  "char_end": 234567,
  "word_count": 15234,
  "is_cross_reference": false
}
```

## Completed Work

1. **Unified extraction script** - Handles multiple OCR formats (JSONL, JSON arrays, MD files)
2. **Source deduplication** - Filters duplicate source files per volume
3. **Cross-reference filtering** - "See CHEMISTRY" no longer creates false articles
4. **OCR artifact filtering** - Removes figure labels (AA, BB, ABC, BCD), fragments (GRAPHY), end-matter (FINIS)
5. **Article deduplication** - Removes duplicate articles from OCR errors
6. **Page number provenance** - Each article has start/end page from OCR
7. **Static website** - Browsable by edition/volume with search and .md download
8. **Validation** - Zero page-order violations (articles in correct sequence)

## Known Issues / TODO

### 3rd Edition (1797) - Missing Volumes
Volumes 1, 10, 11, 12, 13 are missing from OCR. Need to re-OCR these from source PDFs.
- Currently excluded from website
- 14 volumes available, but incomplete

### Potential Article Quality Issues
- Some very short articles may still be artifacts
- Long "EXPLANATION OF PLATE" entries may need review
- Section extraction only works for treatises with explicit markers (PART I, CHAPTER I, etc.)

## Commands

```bash
# Extract specific editions
python3 extract_britannica_corpus.py --editions 1771,1778

# Extract all editions
python3 extract_britannica_corpus.py --all

# List available editions
python3 extract_britannica_corpus.py --list

# Regenerate website
python3 generate_site.py

# Commit and push
git add docs/ extract_britannica_corpus.py
git commit -m "Description"
git push origin main
```

## Data Sources
- **National Library of Scotland**: https://data.nls.uk/data/digitised-collections/encyclopaedia-britannica/
- **Internet Archive**: https://archive.org/
- **OCR**: OLMoCR (Allen Institute for AI)
