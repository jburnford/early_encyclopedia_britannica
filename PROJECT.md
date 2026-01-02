# Encyclopedia Evolution Knowledge Graph

A knowledge graph tracking the evolution of encyclopedic knowledge from 1704-1911, enabling research into how human understanding changed across the "long eighteenth and nineteenth centuries."

**Created:** December 21, 2025

---

## Project Goal

Track the evolution of knowledge across English encyclopedias by:
1. Extracting articles from OCR'd historical encyclopedias
2. Linking articles across editions to trace how topics evolved
3. Building a GraphRAG system for semantic search and research queries
4. Integrating with the existing Banks Archive for cross-project research

### Research Questions Enabled

- How did understanding of a topic (e.g., ELECTRICITY, CHEMISTRY) change over 140 years?
- When did new topics first appear in encyclopedias?
- Which articles expanded, contracted, or disappeared between editions?
- How did cross-references evolve?
- What connections exist between Banks' correspondence and encyclopedia content?

---

## Data Sources

### Encyclopedias (1704-1911)

| Work | Year | Volumes | PDFs | Status | Notes |
|------|------|---------|------|--------|-------|
| Lexicon Technicum | 1704 | 1 | 1 | OCR Complete | First English technical encyclopedia (John Harris) |
| Chambers Cyclopaedia | 1728 | 2 | 2 | OCR Complete | First general English encyclopedia (Ephraim Chambers) |
| Coetlogon Universal History | 1745 | 2 | 2 | OCR Complete | Bridges Chambers to Britannica |
| Britannica 1st Edition | 1771 | 3 | 6 | OCR Complete | First Britannica, smaller scope |
| Britannica 2nd Edition | 1778 | 10 | 20 | OCR Complete | Significantly expanded |
| Britannica 3rd Edition | 1797 | 18 | 18 | OCR In Progress | 13/18 complete, 5 pending |
| Britannica 4th Edition | 1810 | 20 | 40 | OCR Complete | Major revision |
| Britannica 5th Edition | 1815 | 20 | 20 | Parsed | Reprint of 4th with updates |
| Britannica 6th Edition | 1823 | 20 | - | Not Started | |
| Britannica 7th Edition | 1842 | 21 | - | Not Started | |
| Britannica 8th Edition | 1860 | 21 | - | Not Started | |
| Britannica 9th Edition | 1889 | 25 | - | Not Started | "Scholar's Edition" |
| Britannica 11th Edition | 1911 | 29 | - | Not Started | Most famous edition |

### Source Locations

- **PDFs on Nibi:** `~/projects/def-jic823/olmocr/encyclopedias/`
- **1810 PDFs:** `~/projects/def-jic823/olmocr/1810EncyclopediaBritannicaNLS/`
- **1815 Local JSONs:** `/home/jic823/1815EncyclopediaBritannicaNLS/json/`

---

## OCR Pipeline

### Infrastructure

- **Cluster:** Nibi (University of Saskatchewan HPC)
- **GPU:** NVIDIA H100
- **Container:** OLMoCR Apptainer (`~/projects/def-jic823/olmocr/olmocr.sif`)
- **Script:** `archive-olm-pipeline/olmocr/smart_process_pdf_chunks.slurm`

### Job Submission

```bash
# Example: Submit encyclopedia batch
PDF_DIR=~/projects/def-jic823/olmocr/encyclopedias/1778_britannica_2nd
BATCH_DIR=~/projects/def-jic823/encyclopedia_batches/1778_brit_2nd
OLMOCR_SCRIPT=~/projects/def-jic823/archive-olm-pipeline/olmocr/smart_process_pdf_chunks.slurm

sbatch --job-name="1778_brit" \
       --time=02:00:00 \
       --array=0-3 \
       --export=PDF_DIR="$PDF_DIR",BATCH_DIR="$BATCH_DIR" \
       "$OLMOCR_SCRIPT"
```

### Output Format

OLMoCR produces JSONL files with structure:
```json
{
  "text": "Full OCR text of the PDF...",
  "attributes": {
    "pdf_page_numbers": [[start_char, end_char, page_num], ...]
  }
}
```

### Results Locations

| Edition | Path |
|---------|------|
| 1704 Lexicon | `~/projects/def-jic823/encyclopedia_batches/1704_lexicon/results/results/` |
| 1728 Chambers | `~/projects/def-jic823/encyclopedia_batches/1728_chambers/results/results/` |
| 1745 Coetlogon | `~/projects/def-jic823/encyclopedia_batches/1745_coetlogon/results/results/` |
| 1771 1st Ed | `~/projects/def-jic823/encyclopedia_batches/1771_first_ed/results/results/` |
| 1778 2nd Ed | `~/projects/def-jic823/encyclopedia_batches/1778_brit_2nd/results/results/` |
| 1797 3rd Ed | `~/projects/def-jic823/encyclopedia_batches/1797_brit_3rd/results/results/` |
| 1810 4th Ed | `~/projects/def-jic823/1810_britannica_batch/results/results/` |

### Local Downloaded Results

```
/home/jic823/1815EncyclopediaBritannicaNLS/ocr_results/
├── 1704_lexicon/        (1 file)
├── 1728_chambers/       (2 files)
├── 1745_coetlogon/      (2 files)
├── 1771_britannica_1st/ (6 files)
├── 1778_britannica_2nd/ (20 files)
├── 1797_britannica_3rd/ (13 files, 5 pending)
└── 1810_britannica_4th/ (40 files)
```

---

## Article Parser

### Script

`/home/jic823/1815EncyclopediaBritannicaNLS/parse_britannica_articles.py`

### Usage

```bash
# Parse an edition
python3 parse_britannica_articles.py json/ -y 1815 -o articles_1815.jsonl -v

# List known editions
python3 parse_britannica_articles.py --list-editions

# Output only potential errors
python3 parse_britannica_articles.py json/ -y 1815 --errors-only
```

### Features

- Extracts articles by detecting ALL-CAPS headwords followed by comma
- Filters front matter (title pages, preface)
- Handles multiple senses per headword
- Maps articles to source page numbers
- Detects potential OCR errors (running headers, etc.)
- Supports all editions from 1704-1911

### Output Format

```json
{
  "headword": "ASTRONOMY",
  "sense": 1,
  "edition_year": 1815,
  "edition_name": "Britannica 5th",
  "volume": 2,
  "start_page": 234,
  "end_page": 289,
  "is_cross_reference": false,
  "potential_error": false,
  "error_reason": null,
  "text": "Full article text..."
}
```

### Current Parsing Results

**1815 5th Edition:**
- Total articles: 18,172
- Unique headwords: 16,536
- Cross-references: 20
- Multi-sense entries: 1,601
- Volumes present: 1-18, 20 (Volume 19 missing locally)

---

## Neo4j Schema

### Database Connection

Same database as Banks Archive:
```
URI:      neo4j://127.0.0.1:7687
User:     neo4j
Password: york2005
```

### Integration Strategy

1. **Distinct Labels:** Encyclopedia nodes use `Enc_` prefix (e.g., `Enc_Article`)
2. **Source Property:** All nodes have `source` property (e.g., `"britannica_5th_1815"`)
3. **Shared Entities:** Person, Place, Institution linked via Wikidata QIDs

### Node Types

| Label | Description |
|-------|-------------|
| `Enc_Edition` | Encyclopedia edition metadata |
| `Enc_Article` | Individual encyclopedia entry |
| `Enc_Topic` | Normalized topic for cross-edition linking |
| `Enc_TextChunk` | Text segments with embeddings for RAG |
| `Person` | Shared with Banks - historical figures |
| `Place` | Shared with Banks - geographic locations |
| `Institution` | Shared with Banks - organizations |
| `Taxon` | Shared with Banks - species/taxa |

### Key Relationships

| Relationship | Pattern | Description |
|--------------|---------|-------------|
| `IN_EDITION` | Article → Edition | Article belongs to edition |
| `ABOUT` | Article → Topic | Article is about topic |
| `EVOLVED_TO` | Article → Article | Same headword, next edition |
| `CROSS_REFERENCES` | Article → Article | "See ELECTRICITY" links |
| `MENTIONS` | Article → Person/Place | Named entity mentions |
| `HAS_CHUNK` | Article → TextChunk | For vector search |

### Cross-Project Queries

```cypher
// People in both Banks letters and encyclopedia
MATCH (p:Person)<-[:MENTIONS]-(a:Enc_Article)
MATCH (p)<-[:MENTIONED_IN|AUTHORED|RECEIVED]-(d:Document)
WHERE d.source = 'dawson_1958'
RETURN p.name, count(DISTINCT a) as articles, count(DISTINCT d) as letters
```

Full schema documentation: `ENCYCLOPEDIA_NEO4J_SCHEMA.md`

---

## File Inventory

### This Directory

| File | Purpose |
|------|---------|
| `PROJECT.md` | This documentation |
| `ENCYCLOPEDIA_NEO4J_SCHEMA.md` | Detailed Neo4j schema |
| `schema_diagram.md` | Visual schema diagrams |
| `britannica_kg_schema.md` | Original planning document |
| `parse_britannica_articles.py` | Article extraction parser |
| `articles_1815.jsonl` | Raw parsed 1815 articles |
| `articles_1815_clean.jsonl` | Cleaned 1815 articles (18,172) |
| `json/` | 1815 OCR JSON files |
| `ocr_results/` | Downloaded OCR results from Nibi |

### Related Directories

| Path | Purpose |
|------|---------|
| `/home/jic823/cluster/` | SLURM job scripts |
| `/home/jic823/archive-olm-pipeline/` | OLMoCR pipeline infrastructure |
| `/home/jic823/BanksArchivedizws/pdfs/` | Banks Archive project |

---

## Current Status (December 21, 2025)

### Completed

- [x] Built multi-edition article parser
- [x] Parsed 1815 5th edition (18,172 articles)
- [x] Designed Neo4j schema with Banks integration
- [x] OCR'd 1704 Lexicon Technicum
- [x] OCR'd 1728 Chambers Cyclopaedia
- [x] OCR'd 1745 Coetlogon Universal History
- [x] OCR'd 1771 Britannica 1st edition
- [x] OCR'd 1778 Britannica 2nd edition
- [x] OCR'd 1810 Britannica 4th edition (40 PDFs)
- [x] Downloaded results locally

### In Progress

- [ ] 1797 Britannica 3rd edition (5 PDFs remaining - Job 6113339)

### Next Steps

1. Download remaining 1797 results when complete
2. Parse all downloaded OCR results into articles
3. Load articles into Neo4j
4. Extract named entities (Person, Place, Institution)
5. Create cross-edition EVOLVED_TO relationships
6. Generate embeddings for GraphRAG
7. Acquire and process remaining editions (6th-11th)

---

## Commands Reference

### Check Job Status
```bash
ssh nibi "squeue -u jic823"
```

### Download Results
```bash
rsync -avz nibi:~/projects/def-jic823/encyclopedia_batches/*/results/results/*.jsonl ./ocr_results/
```

### Parse an Edition
```bash
python3 parse_britannica_articles.py ocr_results/1810_britannica_4th/ -y 1810 -o articles_1810.jsonl -v
```

### Count Articles
```bash
wc -l articles_*.jsonl
```

---

## Notes

- 1810 (4th) and 1815 (5th) editions have nearly identical content - 5th is a reprint with minor updates
- Volume 19 (Scripture-SLE) missing from local 1815 data but included in 1810 OCR
- Parser handles filenames with spaces correctly
- Front matter filtering removes title pages, prefaces, publisher info
- Multi-sense entries (same headword, different subjects) are normal and tracked with `sense` field

---

*Last updated: December 21, 2025*
