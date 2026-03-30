# Plan: Incremental Embedding Pipeline for Topic Shift Detection and GraphRAG

## Context

The cross-edition index (4,353 entries, 24,025 article-edition pairs) assumes each headword tracks the *same* topic across 8 editions. Manual inspection found 12 entries where this is wrong (e.g., BLACK = color in 1771 vs Joseph Black in 1810). With 60% of entries showing >10x word-count variance, there are likely hundreds more. Automated embedding-based detection is needed.

The user also continues to improve parsing (splits, merges, relabels) and needs incremental re-processing — not a full re-embed of 144K articles after every fix. The end goal is GraphRAG: structured Neo4j graph + vector retrieval over article text.

## Phased Implementation

### Phase 1: Content Fingerprinting (local, no GPU)

**New script: `graphrag/build_article_manifest.py`**

- Reads all 8 `data/export/eb_*_{year}.jsonl` files
- Computes SHA-256 of `article_id + text` for each of 143,954 articles
- Writes `data/article_manifest.jsonl`: `{article_id, title, edition_year, word_count, content_hash}`
- Diffs against previous manifest → `data/article_manifest.diff.json`: `{added: [...], changed: [...], deleted: [...]}`
- Append to the fix pipeline: `... → classify_gaps.py → build_article_manifest.py`

This is the foundation for all incremental work. Typical fix cycles touch 5-50 articles; the diff tells downstream scripts exactly what to re-process.

### Phase 2: Topic Shift Detection (HPC GPU, ~10 min)

**New script: `graphrag/embed_topic_shifts.py`**

**Embedding model: `nomic-ai/nomic-embed-text-v1.5`**
- 8192-token context (vs 512 for bge/mpnet — important for longer openings)
- Matryoshka dimensions (768d for detection, 256d for storage)
- Apache 2.0, runs fully offline on HPC
- `search_document:` / `search_query:` prefixes built in for GraphRAG

**Algorithm:**
1. For each of 4,353 cross-edition entries, extract **first 500 words** of each edition's article (~24K texts total)
2. Embed all with nomic-embed-text-v1.5 (batch, GPU, ~5 minutes)
3. Compute pairwise cosine similarity within each entry
4. Flag entries where any edition-pair similarity < threshold
5. Agglomerative clustering (single-linkage) to group editions into topic clusters
6. Validate against the 12 known topic shifts in `data/topic_shift_report.md` — tune threshold

**Output:**
- `data/embeddings/topic_shift_embeddings.npz` — numpy arrays (compact, ~60MB)
- `data/topic_shift_analysis.jsonl` — per-entry: `{id, min_sim, needs_split, clusters, pairs}`
- `data/topic_shift_detections.md` — human-readable report sorted by severity

**Why first-500-words works:** Topic shifts manifest in the definitional opening. "BLACK, a well known colour" vs "BLACK, Dr Joseph, distinguished for his discoveries in chemistry" — the first sentence is diagnostic. This is validated by the 12 manually identified cases.

**Incremental:** Re-run only for entries whose constituent articles appear in `article_manifest.diff.json`. For typical fix cycles, this means re-embedding 0-10 openings, not 24K.

### Phase 3: Full-Corpus Chunking and Embedding (HPC GPU, ~2-4 hours)

**New script: `graphrag/embed_articles.py`**

- Chunks articles using 1500-word windows, 200-word overlap (consistent with NER chunking pattern in `graphrag/run_ner.py`)
- Embeds each chunk with nomic-embed-text-v1.5 (768d)
- Estimated ~103K chunks for 155M words
- Checkpoint-based resume (same pattern as `run_ner.py`)
- `--edition-year` flag for per-edition processing (parallelizable across GPUs)
- `--incremental` flag reads `article_manifest.diff.json`, only re-embeds changed articles

**Output:** Per-edition files `data/embeddings/eb_{ed}_{year}.chunks.jsonl`:
```json
{"chunk_id": "...", "article_id": "...", "title": "...", "edition_year": 1771,
 "chunk_index": 0, "char_start": 0, "char_end": 8500, "word_count": 1500,
 "text": "...", "embedding": [...]}
```

**Why fixed-window, not semantic chunking:** The old semantic chunker (`old/scripts/encyclopedia_parser/chunkers.py`) requires OpenAI API calls — impossible on HPC compute nodes (no internet). Fixed-window is deterministic, fast, and the 200-word overlap preserves sentence/entity continuity. At 1500 words per chunk, each chunk is well within nomic-embed's 8192-token window.

**HPC job script: `graphrag/slurm/embed.sh`** — SLURM wrapper matching existing patterns in `graphrag/plato/`.

### Phase 4: Vector Storage (local)

**Approach: JSONL + numpy for now, SQLite later if needed**

At 103K vectors × 768d × 4 bytes = ~300MB — small enough for brute-force numpy cosine similarity (<100ms per query). No need for ChromaDB/FAISS/Qdrant at this scale.

**New script: `graphrag/load_embeddings.py`**
- Reads per-edition chunk JSONL files
- Builds a single numpy matrix + metadata index
- Saves as `data/embeddings/chunk_index.npz` (vectors) + `data/embeddings/chunk_metadata.jsonl` (text + article refs)
- Supports incremental update: load existing index, replace chunks for changed article_ids, save

### Phase 5: Neo4j Graph for GraphRAG (local)

**New script: `graphrag/load_neo4j_graphrag.py`**

Extends existing Neo4j at `bolt://206.12.90.118:7687` with:

**Nodes:**
- `(:EB_Article {article_id, title, edition_year, word_count})` — 144K articles
- `(:EB_Entry {id, canonical_title, edition_count, needs_split})` — 4,353 cross-edition entries
- `(:EB_Entity {text, type, qid})` — disambiguated entities from NER
- `(:WikidataItem {qid, label, description})` — Wikidata groundings

**Relationships:**
- `(:EB_Article)-[:IN_ENTRY]->(:EB_Entry)` — edition membership
- `(:EB_Article)-[:MENTIONS {count}]->(:EB_Entity)` — NER links
- `(:EB_Entity)-[:SAME_AS]->(:WikidataItem)` — entity grounding
- `(:EB_Entry)-[:GROUNDED_TO]->(:WikidataItem)` — headword grounding

Uses `MERGE` on primary keys for idempotent loads. Prefix `EB_` avoids collision with existing Early Atlantic World nodes.

### Phase 6: GraphRAG Query Interface (local)

**New script: `graphrag/query.py`**

Query flow:
1. Embed user question with nomic-embed (`search_query:` prefix)
2. Brute-force cosine search over chunk index → top-K chunks
3. For each chunk's article_id, pull graph context from Neo4j:
   - Cross-edition entry + other editions covering same topic
   - Named entities mentioned + their Wikidata links
   - Topic shift warnings if applicable
4. Assemble context (chunk text + graph triples) → send to LLM

This is the final destination. Implementation after phases 1-5 are validated.

## Implementation Order

```
Phase 1: build_article_manifest.py     ← start here (local, CPU, 2 min)
Phase 2: embed_topic_shifts.py         ← immediate value (HPC, 10 min)
         → topic shift report          ← manual review + threshold tuning
         → update cross-edition index  ← split confirmed topic shifts
Phase 3: embed_articles.py             ← heavy lift (HPC, 2-4 hrs)
Phase 4: load_embeddings.py            ← local consolidation
Phase 5: load_neo4j_graphrag.py        ← graph assembly
Phase 6: query.py                      ← GraphRAG queries
```

Phases 1-2 are the priority — they deliver topic shift detection and unblock continued parsing fixes. Phase 3+ can run in parallel with ongoing manual work.

## Key Files to Reuse

| Existing file | Reuse for |
|---|---|
| `graphrag/run_ner.py` | Chunking logic, checkpoint pattern, CLI args, per-edition processing |
| `scripts/rebuild_cross_edition_index.py` | Cross-edition iteration pattern, manifest format |
| `scripts/config.py` | REPO_DIR, EDITIONS, path conventions |
| `data/topic_shift_report.md` | Ground truth for threshold tuning (12 known shifts) |
| `data/cross_edition_index.jsonl` | Input for topic shift detection |
| `data/export/eb_*_{year}.jsonl` | Article text source |

## Verification

1. **Phase 1:** Run `build_article_manifest.py` twice with no changes between → diff should be empty. Make one article fix → diff should show exactly that article.
2. **Phase 2:** Run `embed_topic_shifts.py` → check that all 12 known topic shifts from `topic_shift_report.md` appear in detections. Tune threshold until precision/recall are acceptable.
3. **Phase 3:** Run `embed_articles.py --edition-year 1771 --max-articles 100` as a smoke test. Verify chunk boundaries, embedding dimensions, checkpoint resume.
4. **Phase 5:** Load Neo4j, run sample Cypher queries: `MATCH (e:EB_Entry)-[:GROUNDED_TO]->(w:WikidataItem) RETURN e.id, w.label LIMIT 10`
5. **Phase 6:** Test query with known question: "What does the encyclopedia say about Joseph Black's chemical discoveries?" → should retrieve BLACK entries from 1810-1860 editions.
