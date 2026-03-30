# Next Session: Topic Shift Fixes + GraphRAG Pipeline

## Session Summary (Mar 30, 2026)

### Swallowed Article Fixes (28→2 SWALLOWED gaps)
- Added 26 split specs to `fix_mega_articles.py` for all confirmed SWALLOWED gaps
- STORK→STOVE splits added for 1797, 1810, 1815 (user-reported)
- 2 false positives identified (FOUNDERY, STARCH are plate labels)
- **Gaps: 1,777 → 1,749** (-28)

### GraphRAG Pipeline (Phases 1-2 complete)
- **Phase 1**: `graphrag/build_article_manifest.py` — SHA-256 fingerprinting of 143,954 articles
- **Phase 2**: `graphrag/embed_topic_shifts.py` — nomic-embed-text-v1.5 on Plato A100 (4 min 17 sec)
  - 23,096 article-edition openings embedded (first 500 words each)
  - Results at `data/embeddings/topic_shift_analysis.jsonl`

### Topic Shift Analysis Results (threshold=0.75)
| Category | Count | Description |
|----------|-------|-------------|
| **Topic shifts** | 342 | Multiple editions cover genuinely different topics |
| **Single-edition outliers** | 387 | One edition empty/missing in export (gap, not error) |
| **Short expansions** | 435 | 1771 short definitions expanded later (noise) |

The 387 single-edition outliers all have 0-word content — they're just missing editions, not parser errors.

### Mid-Word Fragment Analysis (26 confirmed)
Of the 342 topic shifts, 251 are clean 2-way splits. Within those, **26 entries start mid-word** (clearly the tail of the previous article). These are directly fixable as merges:

| Fragment | Years | Predecessor | Fragment text |
|----------|-------|-------------|---------------|
| PORTRAIT | 1815,1823 | PORTO | "d safe, that Columbus..." (PORTO-BELLO tail) |
| TENCE | 1810 | VALUE | "o death a single highway robber" |
| NUSANCE | 1797-1823 | NURSING OF CHILDREN | "per shape. The child should be laid" |
| GAUL | 1810-1823 | GAUGING | "oot long and about three-eighths" |
| VERDEN | 1797 | VEGETABLES | "r, in stating the following plan" |
| CERES | 1810,1823 | BROWN | "nce saw, and blest the happy swain" |
| CAPRA | 1810-1823 | CAPPARIS | "eful as detergents and aperients" |
| IMPOTENCE | 1810 | MALACIA | "o Venery. Anaphrodisia..." |
| ORISSA | 1842 | ORION | "ade his illustrious guest drunk" |
| PALAMEDEA | 1823 | PALACE-COURT | "s of a weaver; but attending" |
| MOUNTAINS | 1778,1797 | MOUNTAIN | "ith flat summits" / "ppear to many" |
| DIFFERENT | 1823 | DIDACTIC | "heir bulk, their distance" |

Other fragments that look broken but are actually valid articles: BEATING ("or Pulsation"), CONI ("a strong town"), COPPER, SPAIN, THEOLOGY, VARIATION — these are topic changes or legitimate articles starting with common words.

### What Needs Doing Next

#### 1. Fix 26 Mid-Word Fragments (High Priority)
Add merge specs to `fix_mega_articles.py` MERGES section. Each fragment should be merged into its predecessor article. Use char_start/char_end to verify adjacency.

**Caution**: Some entries appear in multiple volume files (1810 4th ed is a supplement with duplicates). CERES appears twice in 1810, THEOLOGY 3× — need to handle each file occurrence.

#### 2. Topic Change Index Splits (~89 entries, Medium Priority)
These are legitimate editorial decisions where the encyclopedia changed what a headword covers:
- FALCONER: falconry profession (1771-1823) → William Falconer poet (1842-1860)
- LAWRENCE: St. Lawrence River (most eds) → Sir Thomas Lawrence painter (1842)
- LIBERIA: Roman festival (1771-1823) → African republic (1842-1860)
- BASIL: botany/joinery (1771,1778,1842,1860) → St. Basil/Basel city (1797-1823)
- ROSA: botany (1771-1810) → Salvator Rosa painter (1815-1860)

Need a mapping file + logic in `rebuild_cross_edition_index.py` to split these into separate entries (e.g., FALCONER_PROFESSION and FALCONER_WILLIAM).

#### 3. Full Corpus Embedding (Phase 3, Lower Priority)
- `graphrag/embed_articles.py` — 1500-word chunks with 200-word overlap
- Run on Plato A100, ~2-4 hours total
- Incremental via `article_manifest.diff.json`

### Plato Setup
- **Repo**: `~/projects/def-jic823/1815EncyclopediaBritannicaNLS` (cloned from GitHub)
- **Venv**: `~/projects/def-jic823/embed_venv` (sentence-transformers, einops, scipy)
- **Export data**: rsynced to `data/export/` (gitignored)
- **SLURM**: use `--gpus-per-node=a100:1` (NOT --partition or --gres)
- **SSH key**: configured for GitHub access

### Pipeline After Fixes
```bash
python scripts/fix_mega_articles.py
python scripts/merge_fragments.py
python scripts/parse_britannica.py --phase export
python scripts/rebuild_cross_edition_index.py
python scripts/classify_gaps.py
python graphrag/build_article_manifest.py
# Then rsync exports to Plato and run embedding
```

## Key Files
| File | Purpose |
|------|---------|
| `scripts/fix_mega_articles.py` | Manual article fixes — splits, merges, deletes, relabels |
| `graphrag/build_article_manifest.py` | Content fingerprinting for incremental processing |
| `graphrag/embed_topic_shifts.py` | Topic shift detection via embeddings |
| `graphrag/slurm/embed_topic_shifts.sh` | SLURM wrapper for Plato A100 |
| `data/embeddings/topic_shift_analysis.jsonl` | Per-entry similarity scores + clusters |
| `data/embeddings/topic_shift_detections.md` | Human-readable report (342 shifts + 387 outliers + 435 expansions) |
| `data/graphrag_pipeline_plan_2026-03-30.md` | Full 6-phase GraphRAG plan |
