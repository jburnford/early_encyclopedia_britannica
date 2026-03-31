# Next Session: GraphRAG Build + Ongoing Data Improvement

## Session Summary (Mar 30, 2026)

### Swallowed Article Fixes
- 26 split specs added to `fix_mega_articles.py` (ABELARD, STOVE, MEDICINE, MECHANICS, etc.)
- Gaps: 1,777 → 1,749 (-28), SWALLOWED: 28 → 2 (false positives)

### ASTRONOMY 1778 Recovery
- Recovered 59,869 words of ASTRONOMY treatise opening from raw OCR
- Merged with SATURN section (15,703w) + continuation (50,241w) = 125,813w total
- The parser had skipped from vol 2 title page to a section heading, missing everything between

### GraphRAG Pipeline (Phases 1-2 complete)
- Phase 1: Content fingerprinting manifest (143,954 articles, SHA-256 hashes)
- Phase 2: Topic shift detection (nomic-embed on Plato A100, 4 min 17 sec)
  - 342 real topic shifts, 387 missing-edition artifacts, 435 short-expansion noise
  - 26 confirmed mid-word fragments (tails of predecessor articles)

### Plato Cluster Setup
- Repo cloned at `~/projects/def-jic823/1815EncyclopediaBritannicaNLS`
- Venv at `~/projects/def-jic823/embed_venv` (sentence-transformers, einops, scipy)
- SSH key configured for GitHub, export data rsynced
- SLURM: `--gpus-per-node=a100:1` (NOT --partition or --gres)

---

## Parallel Work Tracks

The key architectural insight: the **manifest diff system** (Phase 1) means we can keep fixing articles without re-embedding the whole corpus. After each fix cycle, only changed articles get re-embedded. This enables parallel progress on all fronts.

### Track A: Full-Corpus Embedding + GraphRAG Assembly

**Goal**: Get the GraphRAG working end-to-end, even with imperfect data.

#### A1. Full-corpus embedding (Phase 3)
- Write `graphrag/embed_articles.py` — 1500-word chunks, 200-word overlap
- Run on Plato A100 (~2-4 hours for 143K articles)
- Output: `data/embeddings/eb_{ed}_{year}.chunks.jsonl`
- Incremental mode reads `article_manifest.diff.json`

#### A2. Vector storage (Phase 4)
- JSONL + numpy brute-force (103K vectors × 768d = ~300MB, <100ms search)
- `graphrag/load_embeddings.py` — consolidate per-edition files into single index

#### A3. Neo4j graph assembly (Phase 5)
- `graphrag/load_neo4j_graphrag.py` — load to existing Neo4j at bolt://206.12.90.118:7687
- Nodes: EB_Article (144K), EB_Entry (4,353), EB_Entity (from NER), WikidataItem
- Relationships: IN_ENTRY, MENTIONS, SAME_AS, GROUNDED_TO
- Use MERGE for idempotent loads — safe to re-run after fixes

#### A4. Query interface (Phase 6)
- `graphrag/query.py` — embed question → vector search → graph context → LLM synthesis
- Test with known questions: "What does the encyclopedia say about Joseph Black?"

### Track B: Parser Error Fixes (Ongoing)

**Goal**: Keep improving article quality. Each fix cycle runs the pipeline then `build_article_manifest.py` to generate diffs for incremental re-embedding.

#### B1. Mid-word fragment merges (26 confirmed)
These articles start mid-word and are tails of the previous article:
- PORTRAIT→PORTO (1815,1823), TENCE→VALUE (1810), NUSANCE→NURSING (1797-1823)
- GAUL→GAUGING (1810-1823), CERES→BROWN (1810,1823), CAPRA→CAPPARIS (1810-1823)
- IMPOTENCE→MALACIA (1810), ORISSA→ORION (1842), PALAMEDEA→PALACE-COURT (1823)
- MOUNTAINS→MOUNTAIN (1778,1797), DIFFERENT→DIDACTIC (1823), VERDEN→VEGETABLES (1797)
Full list in `data/embeddings/topic_shift_analysis.jsonl` (shift_type="topic_shift", check for mid-word starts)

#### B2. Cross-volume treatise recovery (like ASTRONOMY 1778)
- Large treatises that span volume boundaries get broken by the parser
- Site review surfaces these: articles starting mid-sentence at beginning of a volume
- Fix: recover from raw OCR + merge, as done for ASTRONOMY

#### B3. Remaining swallowed articles
- 2 false positives still classified as SWALLOWED (FOUNDERY, STARCH — plate labels)
- Topic shift analysis may reveal more via the single-outlier category

#### B4. 1810 4th edition audit
- The 1810 supplement has non-alphabetical volumes → parser fails more often
- 56 single-edition outliers are from 1810
- Systematic check of all 1810 entries for swallowed content

### Track C: Topic Shift Index Splits (89 entries)

**Goal**: Split cross-edition index entries where the encyclopedia editors changed what a headword covers.

#### C1. Build topic split mapping file
- `data/topic_splits.jsonl` — maps headword → {cluster_name, editions, wikidata_qid}
- Example: FALCONER → {FALCONER_PROFESSION: [1771-1823], FALCONER_WILLIAM: [1842-1860]}
- Start with the 12 known shifts from `data/topic_shift_report.md`

#### C2. Update `rebuild_cross_edition_index.py`
- Read topic split mapping before building index
- Override edition grouping for mapped headwords
- Generate separate index entries with distinct IDs

#### C3. Extend to 89 detected shifts
- Review `data/embeddings/topic_shift_detections.md` Section 1
- Major cases: LAWRENCE (river→painter), BASIL (botany→saint→city), ROSA (botany→painter), LIBERIA (festival→republic), MUSA (plantain→village→person)

### Track D: Wikidata Grounding (Ongoing)

**Goal**: Link articles, entities, and topics to the knowledge graph.

#### D1. Headword disambiguation (844/4,353 = 19%)
- Continue with `/headword-disambig` skill
- Topic splits from Track C will need new QIDs for split entries
- Priority: ground the 342 topic-shift entries (they need correct QIDs per cluster)

#### D2. Person disambiguation (1,458 matches)
- Continue with `/person-disambig` skill
- 1,157,244 NER entities across 8 editions

#### D3. Toponym disambiguation (94.3% grounded)
- Mostly complete, continue with `/place-disambig` for remaining 5.7%

### Track E: Internal Cross-References

**Goal**: Build cross-edition article links and "See also" relationships.

#### E1. Parse "See X" cross-references
- Many articles contain "See ASTRONOMY", "See MECHANICS", etc.
- Extract these as relationships: (:EB_Article)-[:SEE_ALSO]->(:EB_Entry)
- The article files already have `type: "cross_reference"` and `target` field for pure cross-refs

#### E2. Cross-edition evolution links
- For non-split entries: link editions as (:EB_Article)-[:NEXT_EDITION]->(:EB_Article)
- For split entries: link within clusters only
- Enables queries like "How did the ASTRONOMY article change from 1771 to 1860?"

---

## Fix Pipeline (run after each batch of fixes)

```bash
# 1. Apply fixes
python scripts/fix_mega_articles.py
python scripts/merge_fragments.py

# 2. Re-export
python scripts/parse_britannica.py --phase export

# 3. Rebuild index + classify gaps
python scripts/rebuild_cross_edition_index.py
python scripts/classify_gaps.py

# 4. Update manifest (detects what changed)
python graphrag/build_article_manifest.py

# 5. Regenerate site
python scripts/generate_site.py

# 6. Incremental re-embed (on Plato, after rsync)
python graphrag/embed_topic_shifts.py --incremental
python graphrag/embed_articles.py --incremental  # Phase 3, once built
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/fix_mega_articles.py` | Manual fixes — splits, merges, deletes, relabels |
| `graphrag/build_article_manifest.py` | Content fingerprinting for incremental processing |
| `graphrag/embed_topic_shifts.py` | Topic shift detection (Phase 2, complete) |
| `graphrag/embed_articles.py` | Full-corpus embedding (Phase 3, TODO) |
| `graphrag/load_neo4j_graphrag.py` | Neo4j graph assembly (Phase 5, TODO) |
| `graphrag/query.py` | GraphRAG query interface (Phase 6, TODO) |
| `data/embeddings/topic_shift_analysis.jsonl` | 4,311 entries with similarity scores |
| `data/embeddings/topic_shift_detections.md` | Categorized report (342 + 387 + 435) |
| `data/graphrag_pipeline_plan_2026-03-30.md` | Original 6-phase plan |
| `data/article_manifest.jsonl` | 143,954 article fingerprints |
