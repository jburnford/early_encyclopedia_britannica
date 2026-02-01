# Encyclopedia Britannica Knowledge Graph - Status Report

**Date**: February 1, 2026
**Branch**: `claude/read-knowledge-graph-plan-XjvAz`

---

## Executive Summary

We have completed the foundational data analysis phase for building a knowledge graph from 8 editions of Encyclopedia Britannica (1771-1860). The article index is built with full source provenance, quality classification, and character-level tracking to support iterative refinement.

**Key Finding**: 98.5% of articles (152,833) are classified as GREEN (ready for knowledge graph), with only 426 RED articles requiring parsing fixes before inclusion.

---

## Data Overview

### Editions Analyzed

| Edition | Articles | Characters | Volumes |
|---------|----------|------------|---------|
| 1771 (1st) | 15,528 | 76.8M | 4 |
| 1778 (2nd) | 18,960 | 71.2M | 11 |
| 1797 (3rd) | 21,021 | 105.9M | 19 |
| 1810 (4th/5th) | 14,781 | 90.7M | 21 |
| 1815 (5th) | 18,452 | 92.5M | 20 |
| 1823 (6th) | 15,833 | 91.6M | 21 |
| 1842 (7th) | 35,236 | 200.5M | 22 |
| 1860 (8th) | 15,365 | 108.1M | 22 |
| **TOTAL** | **155,176** | **837.2M** | **140** |

### Cross-Edition Analysis

From prior work on this branch:

- **25,322** articles appear in 2+ editions
- **717** articles appear in all 8 editions
- **18,103** articles are unique to a single edition

Distribution by edition count:
```
8 editions:    717 articles
7 editions:  2,666 articles
6 editions:  4,248 articles
5 editions:  4,668 articles
4 editions:  4,359 articles
3 editions:  3,679 articles
2 editions:  4,985 articles
```

---

## Article Index

### Source Link Schema

Each article in `article_index.jsonl` includes full provenance:

```json
{
  "article_id": "enc_1815_vol5_idx0142_ELECTRICITY",
  "headword": "ELECTRICITY",
  "headword_normalized": "electricity",
  "quality_flag": "green",
  "issues": [],
  "source": {
    "file": "docs/1815/data/vol5.json",
    "edition": 1815,
    "volume": "vol5",
    "volume_number": 5,
    "array_index": 142,
    "page_order": 89,
    "volume_total_chars": 3948704
  },
  "boundaries": {
    "start_page": 234,
    "end_page": 267,
    "text_length": 45000,
    "char_start": 156789,
    "char_end": 201789
  }
}
```

### Key Fields

| Field | Purpose |
|-------|---------|
| `article_id` | Deterministic ID: `enc_{year}_{vol}_idx{N}_{HEADWORD}` |
| `source.file` | Relative path to source JSON |
| `source.array_index` | Position in JSON array |
| `source.page_order` | Position when sorted by page number |
| `boundaries.char_start/end` | Cumulative character offset in page order |
| `quality_flag` | Classification: green/yellow/red |

---

## Quality Classification

### Results

| Flag | Count | Percentage | Status |
|------|-------|------------|--------|
| **GREEN** | 152,833 | 98.5% | Ready for knowledge graph |
| **YELLOW** | 1,917 | 1.2% | Usable with caveats |
| **RED** | 426 | 0.3% | Parsing errors, deferred |

### Classification Criteria

**RED (Definite Parsing Errors)**:
- Mid-sentence headwords: `"THIS IS INDEED THE ONLY RATIONAL END..."`
- Ends with function word: `"DECOMPOSED BY"`, `"PURIFIED BY THE"`
- Headword > 60 characters
- Volume markers: `"END OF VOLUME"`, `"FINIS"`

**YELLOW (Minor Issues)**:
- Very short text (< 50 chars) - may be cross-reference only
- Very long text (> 100K chars) - possibly merged articles
- Headword > 40 characters
- Lowercase start

**GREEN (High Confidence)**:
- Passes all checks
- Ready for immediate inclusion in knowledge graph

### Issues Detected

| Issue | Count | Severity |
|-------|-------|----------|
| `possibly_merged` | 1,138 | Yellow |
| `very_short_text` | 716 | Yellow |
| `ends_with_function_word` | 266 | Red |
| `long_headword` | 71 | Yellow |
| `mid_sentence_headword` | 70 | Red |
| `volume_marker` | 45 | Red |
| `headword_too_long` | 45 | Red |

---

## Coverage Analysis

### Per-Edition Summary

| Edition | Missing Pages | Alpha Breaks | Red Articles |
|---------|---------------|--------------|--------------|
| 1771 | 762 | 206 | 4 |
| 1778 | 1,915 | 240 | 42 |
| 1797 | 3,173 | 285 | 43 |
| 1810 | 1,670 | 171 | 57 |
| 1815 | 4,422 | 276 | 49 |
| 1823 | 5,566 | 271 | 54 |
| 1842 | 2,918 | 960 | 106 |
| 1860 | 3,812 | 263 | 71 |

**Notes on "Missing Pages"**:
- Multi-column layouts mean multiple articles share pages (expected)
- Illustration plates have no text articles
- Long treatises span many pages without separate entries
- Not all "missing" pages indicate actual gaps

**Alphabetical Breaks**:
- Places where article sequence is out of alphabetical order
- Often legitimate (cross-references, variant spellings)
- Some indicate parsing boundaries that need review

---

## File Outputs

```
1815EncyclopediaBritannicaNLS/
├── article_index.jsonl          # 155,176 articles with source links
├── article_index_stats.json     # Summary statistics
├── coverage_report.json         # Per-volume coverage analysis
├── cross_edition_articles.json  # Articles appearing in 2+ editions
├── single_edition_articles.json # Articles unique to one edition
├── parsing_error_analysis.json  # Detailed error categorization
└── scripts/
    ├── build_article_index.py       # Creates article_index.jsonl
    ├── analyze_coverage.py          # Coverage analysis
    ├── find_cross_edition_articles.py
    ├── analyze_single_edition_articles.py
    ├── analyze_parsing_errors.py
    └── ground_truth_verification.py
```

---

## Architecture Decision: Graph First, NER/Embeddings Later

### Rationale

1. **Parsing fixes invalidate downstream work**: If a "merged" article gets split, any NER or embeddings done on it must be redone.

2. **Graph structure aids quality analysis**: Cross-references to non-existent headwords reveal parsing issues.

3. **Nodes and edges are lightweight**: Easy to rebuild as data quality improves.

4. **Iterative refinement**: The source links (`char_start`, `char_end`, `array_index`) enable precise tracking of what needs reprocessing.

### Deferred Components

- Named Entity Recognition (earlymodernner)
- Embedding generation (bge-large-en)
- Entity linking (Wikidata)

These will be added after the graph structure is stable and parsing issues are resolved.

---

## Next Steps

### Immediate (Graph Structure)

1. **Neo4j Loader** (`scripts/load_neo4j.py`)
   - Create `Edition` nodes (8)
   - Create `Article` nodes (152,833 GREEN)
   - Create `IN_EDITION` relationships
   - Create `EVOLVED_TO` relationships (same headword across editions)

2. **Cross-Reference Extraction** (`scripts/extract_crossrefs.py`)
   - Pattern match: "See ARTICLE", "under ARTICLE"
   - Validate targets exist in same edition
   - Create `REFERENCES` relationships

3. **Quality Analysis Queries**
   - Find broken cross-references (targets don't exist)
   - Find headword gaps in alphabetical sequence
   - Identify suspiciously large articles for splitting

### Future (After Graph Stable)

4. **Parsing Fixes**
   - Merge RED articles back into their parent articles
   - Split YELLOW "possibly_merged" articles
   - Re-run article index for affected volumes

5. **NER Pipeline** (Nibi cluster)
   - Process GREEN articles through earlymodernner
   - Extract: PERSON, TOPONYM, ORGANIZATION, COMMODITY

6. **Embeddings** (Nibi cluster)
   - Generate chunk embeddings for semantic search
   - Load into Neo4j vector index

---

## Neo4j Schema (Proposed)

```cypher
// Nodes
(:Edition {
  year: 1815,
  article_count: 18452,
  total_chars: 92469477
})

(:Article {
  article_id: "enc_1815_vol5_idx0142_ELECTRICITY",
  headword: "ELECTRICITY",
  headword_normalized: "electricity",
  quality_flag: "green",
  text_length: 45000,
  source_file: "docs/1815/data/vol5.json",
  array_index: 142,
  start_page: 234,
  end_page: 267,
  char_start: 156789,
  char_end: 201789
})

// Relationships
(:Article)-[:IN_EDITION]->(:Edition)
(:Article)-[:EVOLVED_TO {delta_chars: 5000}]->(:Article)
(:Article)-[:REFERENCES {context: "See OPTICS"}]->(:Article)
```

---

## Sample Queries (Future)

```cypher
// Find articles that grew significantly across editions
MATCH (a1:Article)-[e:EVOLVED_TO]->(a2:Article)
WHERE a2.text_length > a1.text_length * 2
RETURN a1.headword, a1.text_length, a2.text_length,
       a2.text_length / a1.text_length as growth_ratio
ORDER BY growth_ratio DESC
LIMIT 20

// Find broken cross-references (parsing clues)
MATCH (a:Article)-[r:REFERENCES]->(target)
WHERE target IS NULL
RETURN a.headword, r.target_headword, a.source_file

// Article evolution timeline
MATCH path = (a:Article {headword_normalized: "electricity"})-[:EVOLVED_TO*]->()
RETURN path
```

---

## Conclusion

The foundational data layer is complete:
- **155,176 articles** indexed with full source provenance
- **98.5% classified GREEN** and ready for knowledge graph
- **Character-level tracking** enables iterative refinement
- **Quality flags** separate clean data from parsing errors

The next phase focuses on building the graph structure (nodes + edges) before adding NER and embeddings. This approach ensures downstream processing isn't wasted on articles that may be re-parsed.

---

*Report generated: February 1, 2026*
