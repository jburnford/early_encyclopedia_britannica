# Encyclopedia Britannica Historical Corpus Project

## Overview

This project processes OCR'd text from the first 8 editions of the Encyclopaedia Britannica (1771-1860) into a searchable, hyperlinked static website hosted on GitHub Pages.

**Final Statistics:**
- **136,848 articles** across 8 editions
- **4.4 million cross-reference hyperlinks** to places, people, and major topics
- **8 editions** spanning 89 years of publication

## Edition Summary

| Edition | Year | Articles | Volumes | Notes |
|---------|------|----------|---------|-------|
| 1st | 1771 | 12,624 | 3 | First edition, dictionary-style |
| 2nd | 1778 | 17,219 | 10 | Expanded with more treatises |
| 3rd | 1797 | 21,180 | 18 | Major expansion |
| 4th | 1810 | 15,070 | 20 | Supplement added |
| 5th | 1815 | 18,617 | 20 | Updated 4th edition |
| 6th | 1823 | 16,011 | 20 | Final Constable edition |
| 7th | 1842 | 19,669 | 21+Index | Most scholarly edition |
| 8th | 1860 | 16,458 | 22 | Last edition before 9th |

## Directory Structure

```
1815EncyclopediaBritannicaNLS/
├── docs/                          # GitHub Pages output (static website)
│   ├── index.html                 # Main landing page
│   ├── search.html                # Global search page
│   ├── about.html                 # About page
│   └── {year}/                    # Edition directories
│       ├── index.html             # Edition index
│       ├── vol{N}.html            # Volume pages
│       └── data/vol{N}.json       # Article data (JSON)
│
├── output_v2/                     # Parsed article data
│   ├── articles_{year}.jsonl      # Articles per edition (main data)
│   ├── volumes_{year}.jsonl       # Volume metadata
│   └── index_1842.jsonl           # 1842 General Index (18,215 entries)
│
├── ocr_results/                   # Raw OCR output
│   ├── 1771_britannica_1st/       # 1st edition JSONL
│   ├── 1778_britannica_2nd/       # 2nd edition JSONL
│   ├── 1797_britannica_3rd/       # 3rd edition MD + JSONL
│   ├── 1810_britannica_4th/       # 4th edition JSONL
│   ├── 1842_general_index/        # 7th edition General Index
│   └── britannica_pipeline_batch/ # 5th-8th edition JSONL (mixed)
│
├── ocr_organized/                 # Organized OCR files (210 files)
│   └── britannica_{ed}_{year}_vol{N}_{range}.jsonl
│
├── json/                          # Legacy JSON format (5th/6th editions)
│
└── encyclopedia_parser/           # Python parsing library
    ├── __init__.py                # Module exports
    ├── models.py                  # Data models (Article, TextChunk, etc.)
    ├── extractors/                # Text extraction (MD, JSONL)
    ├── classifiers.py             # Article type classification
    ├── chunkers.py                # Text chunking for RAG
    ├── sections.py                # Section extraction
    ├── patterns.py                # Regex patterns
    ├── expected_articles.py       # Expected article registry (Smart Parser)
    ├── fuzzy_matcher.py           # OCR variation detection (Smart Parser)
    ├── llm_extractor.py           # LLM-based extraction (Smart Parser)
    └── smart_parser.py            # Smart Parser integration
```

## Key Scripts

### Site Generation

**`generate_site_optimized.py`** - Main site generator
- Generates static HTML pages for GitHub Pages
- Injects cross-reference hyperlinks using Aho-Corasick algorithm
- Only links meaningful articles (geographical, biographical, treatise)
- Output: `docs/` directory

```bash
python3 generate_site_optimized.py
```

### Article Parsing

**`parse_britannica_articles.py`** - Core article parser
- Extracts articles from OCR text (MD, JSONL, JSON formats)
- Classifies articles by type (dictionary, geographical, biographical, treatise)
- Output: `output_v2/articles_{year}.jsonl`

```bash
python3 parse_britannica_articles.py
```

### Smart Parser

**`run_smart_parser.py`** - Enhanced article recovery
- Uses 1842 General Index as ground truth for expected articles
- Applies fuzzy matching to find OCR variations
- Recovers articles missed by the main parser
- Recovered 15,000+ additional articles across all editions

```bash
python3 run_smart_parser.py
```

### OCR Organization

**`organize_ocr_files.py`** - OCR file organizer
- Reorganizes OCR files with descriptive names
- Output: `ocr_organized/` directory

```bash
python3 organize_ocr_files.py
```

## Data Formats

### Article JSONL (`output_v2/articles_{year}.jsonl`)

```json
{
  "article_id": "1842_v01_BACON",
  "headword": "BACON",
  "text": "Full article text...",
  "article_type": "biographical",
  "edition_year": 1842,
  "volume_num": 1,
  "start_page": 1,
  "end_page": 15,
  "word_count": 7500,
  "is_cross_reference": false
}
```

### Article Types
- **dictionary** - Short definitions of common words (not hyperlinked)
- **geographical** - Places, countries, cities (hyperlinked)
- **biographical** - People, historical figures (hyperlinked)
- **treatise** - Major topic articles (hyperlinked)
- **cross_reference** - "See X" redirects

### Volume Metadata (`output_v2/volumes_{year}.jsonl`)

```json
{
  "volume_num": 1,
  "edition_year": 1842,
  "title": "Volume 1, A-ANA",
  "article_count": 850
}
```

### 1842 Index (`output_v2/index_1842.jsonl`)

```json
{
  "headword": "ABERDEEN",
  "volume_refs": [1, 2],
  "page_refs": {"1": [45], "2": [102, 103]},
  "has_see_also": false
}
```

## Smart Parser Components

The Smart Parser was developed to recover articles missed by the main parser:

### Phase 1: Expected Article Registry (`encyclopedia_parser/expected_articles.py`)
- Loads 18,215 entries from 1842 General Index
- Tracks which articles exist vs. expected

### Phase 2: Fuzzy Matcher (`encyclopedia_parser/fuzzy_matcher.py`)
- Uses rapidfuzz library for OCR variation detection
- Learned 4,488 OCR variations from data
- Common errors: B→R, E→I, E→R

### Phase 3: LLM Extractor (`encyclopedia_parser/llm_extractor.py`)
- Heuristic-based article boundary detection
- Handles patterns like "TERM, definition" and "TERM (Name), definition"

### Phase 4: Integration (`encyclopedia_parser/smart_parser.py`)
- Combines fuzzy matching + heuristic extraction
- Confidence scoring for recovered articles

## Hyperlink System

The site includes intelligent cross-reference hyperlinks:

### What Gets Linked
- **Geographical articles** (places, countries, cities)
- **Biographical articles** (people, historical figures)
- **Treatise articles** (major topics with substantial content)
- **Unknown type articles** with 2000+ characters

### What Doesn't Get Linked
- Dictionary definitions (common words like "time", "work", "power")
- Cross-references
- Short articles under the length threshold

### Implementation
- Uses Aho-Corasick algorithm for O(n+m) pattern matching
- ~8,600 linkable terms from 1842 index
- First occurrence of each term is linked

## GitHub Pages

The site is hosted at: https://jburnford.github.io/early_encyclopedia_britannica/

### Deployment
1. Run `python3 generate_site_optimized.py`
2. Commit changes to `docs/` directory
3. Push to `main` branch
4. GitHub Pages automatically deploys from `docs/`

## Development Notes

### Requirements
- Python 3.10+
- `pyahocorasick` - Fast multi-pattern matching
- `rapidfuzz` - Fuzzy string matching

```bash
pip install pyahocorasick rapidfuzz
```

### Memory Optimization
- Streaming article loading (~20MB vs ~600MB)
- Per-edition processing with garbage collection
- Pre-compiled pattern automaton

### Performance
- Site generation: ~5 minutes for all 8 editions
- Smart parser: ~7 minutes for full recovery run
- Hyperlink injection: 4.4M links in ~3 minutes

## Acknowledgments

- OCR provided by National Library of Scotland digitization project
- Original texts from various Britannica editions (1771-1860)
