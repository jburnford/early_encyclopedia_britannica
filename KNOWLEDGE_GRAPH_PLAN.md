# Encyclopedia Britannica Knowledge Graph - Implementation Plan

## Project Summary

**Goal**: Build a knowledge graph extraction pipeline on Nibi cluster, transforming 155K encyclopedia articles into structured data with entities, chunks, and embeddings.

**Current State**:
- 8 editions: 1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860
- Data: `docs/{year}/data/vol*.json` (~812 MB)
- **Data quality issues**: Some parsing anomalies (fake headwords like "WITHOUT THE")

**Pipeline Target**: Nibi cluster with:
- earlymodernner (Qwen3-4B + LoRA) for NER
- Qwen for supplementary extraction
- Local embedding generation

---

## Data Analysis

### Article Length Distribution
```
Year     Count    <500     500-2K   2K-10K   10K-50K  >50K     Max
1771     15,528   12,746   1,684    424      212      462      695K
1778     16,986   8,762    4,566    2,498    921      239      608K
1815     18,452   9,293    4,850    2,771    1,127    411      1.27M
1842     35,236   15,715   9,879    5,348    3,591    703      792K
1860     15,365   5,500    5,028    3,025    1,349    463      1.9M
```

**Key Findings**:
- ~2,200 articles over 50K chars across all editions (need sectioning)
- Some "articles" are parsing errors: "WITHOUT THE: 580K chars"
- Longest legit articles: AGRICULTURE (1.27M), CHEMISTRY (806K)

---

## Phase 1: Cross-Edition Foundation

**Strategy**: Start with articles appearing in 2+ editions as the high-confidence foundation.

### 1.1 Find Cross-Edition Articles
**Script**: `scripts/find_cross_edition_articles.py`

```python
# Build headword index across all editions
headwords_by_edition = {
    1771: set(normalize(a['h']) for a in load_edition(1771)),
    1778: set(...),
    ...
}

# Find articles in 2+ editions
cross_edition = {}
for headword in all_headwords:
    editions = [y for y in YEARS if headword in headwords_by_edition[y]]
    if len(editions) >= 2:
        cross_edition[headword] = editions
```

**Output**: `cross_edition_articles.json`
```json
{
  "astronomy": {"editions": [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860], "count": 8},
  "electricity": {"editions": [1771, 1778, 1797, 1815, 1842, 1860], "count": 6},
  ...
}
```

**Expected**: ~30-50K headwords appearing in multiple editions

### 1.2 LLM Quality Verification
**Script**: `scripts/verify_article_quality.py`

For each cross-edition article, use LLM (Qwen on Nibi) to verify:

```python
PROMPT = """
Analyze this encyclopedia article for parsing quality.

HEADWORD: {headword}
TEXT (first 2000 chars): {text_preview}
TEXT (last 500 chars): {text_end}

Check:
1. Does the headword match the article subject?
2. Does the text appear complete (proper ending)?
3. Are there signs of merged articles (multiple unrelated subjects)?
4. Is there OCR noise or formatting issues?

Return JSON:
{
  "headword_match": true/false,
  "appears_complete": true/false,
  "possibly_merged": true/false,
  "ocr_quality": "good"/"fair"/"poor",
  "confidence": "green"/"yellow"/"red",
  "notes": "..."
}
"""
```

### 1.3 Quality Flags
- **GREEN**: High confidence - headword matches, complete, not merged, good OCR
- **YELLOW**: Minor issues - may need review but usable
- **RED**: Significant issues - skip for now, queue for manual review

### 1.4 Output
```
quality_assessment/
├── green_articles.jsonl    # Ready for KG pipeline
├── yellow_articles.jsonl   # Usable with caveats
├── red_articles.jsonl      # Needs parsing fixes
└── assessment_summary.json # Statistics
```

**Pipeline Rule**: Only GREEN articles enter the knowledge graph initially.

---

## Phase 2: Section-Aware Chunking

### 2.1 Chunking Strategy by Article Size

| Size | Strategy | Target Chunk |
|------|----------|--------------|
| <1,000 chars | Keep whole | 1 chunk |
| 1K-5K chars | Paragraph split | 500-1000 chars |
| 5K-50K chars | Section detection | 1000-2000 chars |
| >50K chars | Hierarchical sections | 1500-2500 chars |

### 2.2 Section Detection Patterns
For large treatises, detect structure:
```python
SECTION_PATTERNS = [
    r'^PART\s+[IVX]+[.,]',           # PART I.
    r'^SECT(?:ION)?\.?\s+[IVX\d]+',  # SECT. 1, SECTION II
    r'^CHAP(?:TER)?\.?\s+[IVX\d]+',  # CHAP. III
    r'^BOOK\s+[IVX]+',               # BOOK II
    r'^\d+\.\s+[A-Z]',               # 1. Definition
    r'^[A-Z][a-z]+\s+[IVX]+[.,]',    # Article I.
]
```

### 2.3 Chunking Script
**Script**: `nibi/chunk_articles.py`

```python
# Output format per chunk
{
    "chunk_id": "britannica_1815_ASTRONOMY_sec2_chunk3",
    "text": "...",
    "article_headword": "ASTRONOMY",
    "edition_year": 1815,
    "volume": 2,
    "section_path": ["PART I", "SECT. 2"],
    "chunk_index": 3,
    "char_start": 15000,
    "char_end": 17500
}
```

### 2.4 Output Files
```
chunks/
├── 1771_chunks.jsonl    # ~50K chunks
├── 1778_chunks.jsonl    # ~60K chunks
├── ...
└── 1860_chunks.jsonl    # ~80K chunks
```

**Estimated Total**: 400-600K chunks

---

## Phase 3: Named Entity Extraction (Nibi)

### 3.1 Tool: earlymodernner
- **Repo**: https://github.com/polayj/earlymodernner
- **Base**: Qwen3-4B-Instruct + 4 LoRA adapters
- **Entities**: PERSON, TOPONYM, ORGANIZATION, COMMODITY

### 3.2 Pipeline on Nibi

**SLURM Script**: `nibi/run_ner.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=enc_ner
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Process chunks through earlymodernner
python -m earlymodernner \
    --input chunks/${EDITION}_chunks.jsonl \
    --output entities/${EDITION}_entities.jsonl \
    --format jsonl
```

### 3.3 Output Format
```json
{
    "chunk_id": "britannica_1815_ASTRONOMY_sec2_chunk3",
    "entities": [
        {"text": "Sir Isaac Newton", "type": "PERSON", "start": 45, "end": 61},
        {"text": "Royal Society", "type": "ORGANIZATION", "start": 120, "end": 133},
        {"text": "London", "type": "TOPONYM", "start": 145, "end": 151}
    ]
}
```

### 3.4 Entity Aggregation
**Script**: `nibi/aggregate_entities.py`

Consolidate entities across chunks:
- Deduplicate by normalized name
- Count occurrences per article
- Prepare for Wikidata linking

---

## Phase 4: Entity Linking (Qwen on Nibi)

### 4.1 Wikidata Candidate Lookup
For top entities (by frequency), fetch Wikidata candidates:
```python
# Query Wikidata for "Isaac Newton"
# Returns: [Q935, Q15985003, ...]
```

### 4.2 LLM-Assisted Disambiguation
**Script**: `nibi/link_entities_qwen.py`

Use Qwen to select correct QID given context:
```
Entity: "Newton"
Context: "Newton discovered the laws of motion..."
Candidates: Q935 (Isaac Newton), Q712346 (John Newton)
→ Select: Q935
```

### 4.3 Output
```json
{
    "name_normalized": "isaac newton",
    "canonical_name": "Sir Isaac Newton",
    "wikidata_qid": "Q935",
    "entity_type": "PERSON",
    "mention_count": 847,
    "editions": [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]
}
```

---

## Phase 5: Embedding Generation (Nibi)

### 5.1 Model Options on Nibi

| Model | Dims | Speed | Notes |
|-------|------|-------|-------|
| `bge-large-en-v1.5` | 1024 | Fast | Good general purpose |
| `gte-large-en-v1.5` | 1024 | Fast | Slightly better retrieval |
| `Qwen2-embed` | 1536 | Medium | May handle historical text better |

### 5.2 Batch Embedding Script
**Script**: `nibi/embed_chunks.py`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Process in batches of 128
for batch in chunks_batched:
    embeddings = model.encode(batch, normalize_embeddings=True)
    save_to_parquet(batch_ids, embeddings)
```

### 5.3 Output Format
Parquet files for efficiency:
```
embeddings/
├── 1771_embeddings.parquet  # chunk_id, embedding (1024 floats)
├── 1778_embeddings.parquet
└── ...
```

### 5.4 SLURM Job
```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
# ~600K chunks at 500 chunks/sec = ~20 min per edition
```

---

## Phase 6: Cross-Reference Extraction

### 6.1 Pattern Matching
**Script**: `scripts/extract_crossrefs.py`

Extract from article text:
```python
CROSSREF_PATTERNS = [
    r'[Ss]ee\s+([A-Z][A-Z\s]+)',
    r'[Ss]ee the article\s+([A-Z][A-Z\s]+)',
    r'[Uu]nder\s+([A-Z][A-Z\s]+)',
]
```

### 6.2 Validation
Only keep references where target headword exists in same edition.

### 6.3 Output
```json
{
    "source_article": "ASTRONOMY",
    "target_article": "OPTICS",
    "reference_type": "see",
    "context": "...for the principles of refraction, see OPTICS..."
}
```

---

## Output Artifacts (GitHub Repo)

```
encyclopedia_kg_data/
├── articles/
│   ├── 1771_articles.jsonl       # Cleaned articles
│   └── ...
├── chunks/
│   ├── 1771_chunks.jsonl         # Sectioned chunks
│   └── ...
├── entities/
│   ├── 1771_entities.jsonl       # NER output per chunk
│   ├── entities_linked.jsonl     # Aggregated + Wikidata linked
│   └── ...
├── embeddings/
│   ├── 1771_embeddings.parquet   # Vector embeddings
│   └── ...
├── relationships/
│   ├── crossrefs.jsonl           # Cross-references
│   └── evolution.jsonl           # Same headword across editions
└── manifests/
    ├── editions.json             # Edition metadata
    └── statistics.json           # Counts, coverage
```

---

## Nibi Scripts to Create

| Script | Purpose | Runtime |
|--------|---------|---------|
| `nibi/audit_articles.py` | Find parsing anomalies | 5 min |
| `nibi/chunk_articles.py` | Section-aware chunking | 30 min |
| `nibi/run_ner.slurm` | NER with earlymodernner | 2-4 hrs/edition |
| `nibi/aggregate_entities.py` | Dedupe and count entities | 10 min |
| `nibi/link_entities_qwen.py` | Wikidata disambiguation | 1-2 hrs |
| `nibi/embed_chunks.py` | Generate embeddings | 20 min/edition |

---

## Execution Order

### Iteration 1: Foundation (GREEN articles only)
1. **Phase 1**: Find cross-edition articles, LLM quality verification
2. **Phase 2**: Chunk GREEN articles only
3. **Phase 3**: NER on GREEN chunks (Nibi)
4. **Phase 4**: Entity linking (Nibi)
5. **Phase 5**: Embeddings (Nibi)
6. **Phase 6**: Cross-references

### Iteration 2+: Expand Coverage
1. Review YELLOW articles, fix issues → promote to GREEN
2. Improve parsing for RED articles
3. Re-run pipeline for newly GREEN articles
4. Merge into existing KG data

### Parallelization
- Phases 3-5 can run in parallel on Nibi
- Quality verification (Phase 1.2) can run alongside parsing fixes

---

## Neo4j Loading (Deferred)

Once extraction is complete, create loader script:
- Read from `encyclopedia_kg_data/` artifacts
- Create nodes: `Enc_Edition`, `Enc_Article`, `Enc_Chunk`, `Person`, `Place`, etc.
- Create relationships: `IN_EDITION`, `EVOLVED_TO`, `MENTIONS`, `CROSS_REFERENCES`
- Load embeddings into vector index

This can use the existing `scripts/embed_and_load_neo4j.py` as a starting point.

---

## Handling Updates (Iterative Refinement)

### When Parsing Improves:
1. Re-run quality verification on affected articles
2. Re-chunk if article boundaries changed
3. Re-run NER on new chunks
4. Update entity aggregation (may discover new/different entities)
5. Re-embed affected chunks

### Data Structure for Updates:
```json
{
  "article_id": "britannica_1815_ELECTRICITY",
  "version": 3,
  "last_updated": "2026-02-15",
  "quality_flag": "green",
  "parsing_hash": "abc123..."  // Hash of source text for change detection
}
```

### Incremental Processing:
- Track `parsing_hash` for each article
- On reprocessing, skip articles with unchanged hash
- Only regenerate chunks/entities/embeddings for changed articles

---

## Immediate First Steps

### Step 1: Find Cross-Edition Articles (Local, ~10 min)
Create `scripts/find_cross_edition_articles.py`:
- Load all `docs/{year}/data/*.json` files
- Normalize headwords (lowercase, strip whitespace)
- Find headwords appearing in 2+ editions
- Output `cross_edition_articles.json` with counts

### Step 2: Sample Quality Check (Local, ~30 min)
Pick 100 random cross-edition articles and manually review:
- Do headwords match content?
- Are boundaries correct?
- What percentage would you call "green"?

This informs whether LLM verification is worth the effort or if data is already high-quality.

### Step 3: Set Up earlymodernner on Nibi
```bash
# Clone and test
git clone https://github.com/polayj/earlymodernner
pip install -e earlymodernner

# Test on a few articles
python -m earlymodernner --input test_articles.jsonl --output test_entities.jsonl
```

### Step 4: First Batch Pipeline
Run Phase 1-6 on a single well-parsed edition (e.g., 1771 with 15K articles) as proof of concept.

---

## Verification Queries

### After chunking:
```python
# Verify chunk distribution
import json
with open('chunks/1815_chunks.jsonl') as f:
    chunks = [json.loads(l) for l in f]
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c['text']) for c in chunks) / len(chunks):.0f} chars")
```

### After NER:
```python
# Top entities by frequency
from collections import Counter
entities = Counter()
with open('entities/1815_entities.jsonl') as f:
    for line in f:
        for e in json.loads(line)['entities']:
            entities[(e['text'], e['type'])] += 1
print(entities.most_common(20))
```

### After embeddings:
```python
# Test similarity search
import numpy as np
import pyarrow.parquet as pq

df = pq.read_table('embeddings/1815_embeddings.parquet').to_pandas()
query_embedding = model.encode("electricity and magnetism")
similarities = df['embedding'].apply(lambda x: np.dot(x, query_embedding))
top_5 = df.iloc[similarities.nlargest(5).index]
```

---

*Plan created: February 1, 2026*
