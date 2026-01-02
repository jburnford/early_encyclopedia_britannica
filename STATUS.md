# Encyclopaedia Britannica Historical Corpus - Project Status

## Overview
Extracting and publishing OCR'd text from 7 editions of the Encyclopaedia Britannica (1771-1860) as a browsable website and structured data, with edition-aware article classification.

## Repository
- **GitHub**: `github.com:jburnford/early_encyclopedia_britannica.git`
- **Live Website**: https://jburnford.github.io/early_encyclopedia_britannica/
- **GitHub Pages**: Enabled from `/docs` folder

## Key Files

| File | Purpose |
|------|---------|
| `extract_britannica_corpus.py` | Main extraction script - parses OCR output into articles |
| `generate_site.py` | Generates static HTML site from extracted articles |
| `encyclopedia_parser/` | Edition-aware parsing module with classification |
| `docs/` | Generated website (committed to git) |
| `output_v2/` | Extracted JSONL files (not in git - too large) |
| `ocr_results/` | Source OCR files from OLMoCR (not in git) |

## Current Status: 121,161 Articles

| Edition | Year | Volumes | Articles | Treatises | Biographical | Geographical |
|---------|------|---------|----------|-----------|--------------|--------------|
| 1st | 1771 | 3 | 11,447 | 385 | **0** | 2,074 |
| 2nd | 1778 | 10 | 14,136 | 1,067 | 637 | 2,238 |
| 3rd | 1797 | 18 | 17,139 | 1,820 | 694 | 2,655 |
| 4th | 1810 | 20 | 10,024 | 1,074 | 379 | 1,438 |
| 5th | 1815 | 19 | 18,531 | 1,776 | 720 | 2,789 |
| 6th | 1823 | 20 | 16,081 | 1,963 | 596 | 2,291 |
| 7th | 1842 | 22 | 19,003 | 2,312 | 583 | 4,088 |
| 8th | 1860 | 22 | 14,800 | 2,545 | 1,070 | 2,228 |

**Notes**:
- 1771 (1st edition) correctly shows 0 biographical entries - this edition explicitly excluded biography to focus on Arts & Sciences.
- 1797 (3rd edition) now fully OCR'd with all 18 volumes.

## Output Format (JSONL)

```json
{
  "article_id": "1771_v01_ASTRONOMY",
  "headword": "ASTRONOMY",
  "text": "full article text...",
  "article_type": "treatise|dictionary|biographical|geographical|cross_reference",
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

## Article Types

| Type | Description | Detection |
|------|-------------|-----------|
| `treatise` | Major scholarly articles (ASTRONOMY, CHEMISTRY, etc.) | Known treatise list + length > 10K chars |
| `dictionary` | Short definition entries | Default type |
| `biographical` | Person entries with dates | Birth/death patterns, "born 1642" |
| `geographical` | Place entries with coordinates | Lat/Long patterns, "city of", "miles from" |
| `cross_reference` | Redirect entries | "See CHEMISTRY" pattern |

## Edition-Aware Classification System

Each edition of the Encyclopaedia Britannica has different characteristics that require edition-specific handling.

### Module Structure

```
encyclopedia_parser/
├── models.py           # EditionConfig with edition-specific settings
├── classifiers.py      # Edition-aware article classification
├── editions/           # Edition-specific configuration files
│   ├── __init__.py     # EditionRegistry for loading configs
│   ├── edition_1771.py # 1st edition (Arts & Sciences only)
│   ├── edition_1815.py # 5th edition (largest article count)
│   └── edition_1842.py # 7th edition (has General Index)
```

### Edition-Specific Differences

| Edition | Year | Key Characteristics |
|---------|------|---------------------|
| 1st | 1771 | **No biography** - Arts & Sciences focus only |
| 2nd | 1778 | Adds biography & history, Vol 10 Appendix |
| 4th | 1810 | Reprint of 3rd with updates |
| 5th | 1815 | Largest article count (18,178) |
| 6th | 1823 | Cleanest OCR (no long s confusion) |
| 7th | 1842 | Has General Index in Volume 22 |
| 8th | 1860 | Final edition in corpus |

### EditionConfig Fields

```python
has_biography: bool      # Whether edition includes biographical entries
has_geography: bool      # Whether edition includes geographical entries
index_volume: int        # Volume number of index (if any)
major_treatises: set     # Edition-specific major treatise headwords
ocr_artifacts: list      # Edition-specific false positive patterns
```

## Website Features

- **Color-coded badges**: Treatise (brown), Biography (green), Place (blue)
- **Article type filters**: Filter by treatise, biographical, geographical
- **Statistics per volume**: Article count, treatise count, page range
- **Markdown export**: Download articles as .md files
- **Full-text search**: Search across all editions

## Completed Work

1. **Unified extraction script** - Handles multiple OCR formats (JSONL, JSON arrays, MD files)
2. **Source deduplication** - Filters duplicate source files per volume
3. **Cross-reference filtering** - "See CHEMISTRY" no longer creates false articles
4. **OCR artifact filtering** - Removes figure labels (AA, BB, ABC, BCD), fragments (GRAPHY), end-matter (FINIS)
5. **Article deduplication** - Removes duplicate articles from OCR errors
6. **Page number provenance** - Each article has start/end page from OCR
7. **Static website** - Browsable by edition/volume with search and .md download
8. **Validation** - Zero page-order violations (articles in correct sequence)
9. **Edition-aware classification** - 1771 correctly excludes biography; editions use specific rules
10. **Enhanced website** - Color-coded badges, filters, statistics by article type

## Next Phase: Neo4j Knowledge Graph

A detailed plan exists for building a Neo4j GraphRAG knowledge graph from the corpus.

**Plan file**: `.claude/plans/smooth-giggling-sifakis.md`

### Roadmap
1. **OCR 1842 General Index** (in progress) - Volume 22 is the "Rosetta Stone" for taxonomy
2. **Parse Index** - Extract terms, cross-references, volume/page references
3. **Build Neo4j Schema** - Concepts, Articles, Sections, Chunks with relationships
4. **Load Data** - Starting with 1842 edition (has index)
5. **Cross-Reference Extraction** - Link articles via "See CHEMISTRY" patterns
6. **Cross-Edition Linking** - Track how concepts evolved 1771-1860
7. **Vector Search** - Add Voyage-3 embeddings for RAG

### Neo4j Node Types (Planned)
- `Concept` - Abstract topics from index
- `Edition` - Encyclopedia editions
- `Volume` - Physical volumes
- `Article` - Dictionary/treatise entries
- `Section` - Parts of treatises
- `Chunk` - Semantic text chunks with embeddings

## Known Issues / TODO

### 1842 General Index - OCR In Progress
Volume 22 of the 1842 edition contains the General Index, which provides the canonical taxonomy for the encyclopedia. Currently only the preface has been extracted; full index OCR is in progress.

### 3rd Edition (1797) - COMPLETE
All 18 volumes now OCR'd and extracted. Previously missing volumes have been processed.

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
