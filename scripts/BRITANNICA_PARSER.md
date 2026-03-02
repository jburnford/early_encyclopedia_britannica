# Britannica OCR Article Parser

Extracts structured encyclopedia articles from raw OLMoCR output of 8 editions of the Encyclopedia Britannica (1771-1860). Uses a pure LLM classification approach — DeepSeek-R1-Distill-Llama-70B-AWQ running on Plato HPC A100 GPUs — to identify article boundaries, running headers, cross-references, and other structural elements.

## Data

- **Source**: 210 JSONL files in `/home/jic823/plato/ocr_organized/` (~1GB total)
- **Format**: One JSON line per file with keys: `edition`, `edition_name`, `volume`, `range`, `source_file`, `text`
- **Text sizes**: 91KB to 6MB per volume
- **Editions**: 1st (1771), 2nd (1778), 3rd (1797), 4th (1810), 5th (1817), 6th (1823), 7th (1842), 8th (1860)

### Known source data issues

Some volumes have duplicate OCR scans (e.g., `vol02_part2` and `vol02_part4` for the 1st edition are near-identical ~5.45M char files). These will produce duplicate article sets that need deduplication downstream.

## Architecture

Four-phase pipeline, each phase idempotent and independently runnable:

```
Phase 1 (preprocess)  →  Phase 2 (classify)  →  Phase 3 (assemble)  →  Phase 3.5 (merge)
   no LLM, ~30s            LLM, ~5 days           no LLM, ~1 min         no LLM, ~1 min
```

### Phase 1 — Paragraph Splitting (`preprocess.py`)

Splits each volume's `text` field on `\n\n` boundaries. Records character offsets for later text extraction.

**Input**: `ocr_organized/*.jsonl`
**Output**: `britannica_output/paragraphs/*.paragraphs.jsonl`

Each paragraph record:
```json
{"index": 0, "char_start": 0, "char_end": 142, "text": "full text...", "preview": "first 300 chars..."}
```

**Stats**: 1,650,866 total paragraphs across 210 files. Runs in ~28 seconds.

### Phase 2 — LLM Classification (`classify.py`)

Sends batches of 20 consecutive paragraph previews to the LLM. The model classifies non-body paragraphs; unreported paragraphs default to `body_text`.

**Input**: Paragraph files + source metadata
**Output**: `britannica_output/classifications/*.classifications.jsonl`

Classification types:

| Type | Description | Extra Fields |
|------|-------------|--------------|
| `article_start` | New encyclopedia article | `title`, `keywords` |
| `subsection_start` | Section within a long article | `title` |
| `running_header` | OCR page header artifact (discarded) | — |
| `cross_reference` | "See X" redirect | `title`, `target` |
| `front_matter` | Title page, preface, dedication | — |
| `back_matter` | End-of-volume material | — |
| `author_attribution` | Author initials at end of article | — |
| `footnote_sep` | `---` separator line | — |
| `body_text` | Regular article body (default) | Not reported by LLM |

**Batching**: Sliding window of 20 paragraphs, 2-paragraph overlap, step size 18. Yields ~555 windows per volume, ~116K total.

**Concurrency**: 10 simultaneous async HTTP requests via aiohttp. vLLM's continuous batching handles GPU scheduling.

**Structured output**: Uses vLLM's `guided_json` parameter to force the model to produce valid JSON arrays directly, bypassing DeepSeek-R1's `<think>` reasoning tokens entirely. This was critical — without it, the model spends all output tokens on thinking and never produces JSON.

**Checkpointing**: Saves progress every 50 windows to `{stem}.checkpoint.json`. Fully resumable on job timeout or crash.

**Throughput**: ~0.1-0.2 calls/sec per GPU, ~80-100 minutes per volume.

### Phase 3 — Assembly (`assemble.py`)

Walks paragraphs in order, using classifications to build structured articles:

1. Each `article_start` opens a new article
2. Body text paragraphs accumulate until the next boundary
3. `subsection_start` creates nested sections within the current article
4. `running_header` and `footnote_sep` paragraphs are discarded
5. `cross_reference` entries become single-paragraph redirect articles
6. `front_matter` only collected before the first article; mid-volume misclassifications treated as body text
7. `back_matter` only triggers in the last 10% of the file to prevent mid-volume misclassification from swallowing entire volumes
8. `author_attribution` recorded as metadata but excluded from article text
9. Orphan body text (after articles start but with no current article) attaches to the previous article

**Input**: Paragraph files + classification files + source JSONL (for original text)
**Output**: `britannica_output/articles/*.articles.jsonl`

### Phase 3.5 — Fragment Merger (`merge.py`)

The LLM sometimes promotes internal section headings within long treatises (ANATOMY, AGRICULTURE, ALGEBRA) to `article_start`, fragmenting them into dozens of small articles. The merger detects and re-absorbs these fragments.

**Merge heuristics** (applied in order):

1. `UNTITLED` or garbage titles (>80 chars, digit+letter gibberish) — always merge
2. Consecutive identical titles (e.g., "HEAT" x80) — always merge
3. Chapter/section patterns (`Chap.`, `Part II`, `Of the...`, `PLATE`) — always merge
4. ALL-CAPS title >2 chars without chapter pattern — hard boundary (real article), except: merge small (<500w) ALL-CAPS articles into very large predecessors (>5000w, 3+ prior merges) since these are likely mislabeled running headers
5. Mixed-case title + contiguous (char gap <2000) + predecessor is large (>1000w) or already absorbed fragments — merge as subsection
6. Mixed-case title after short predecessor — keep separate (legitimate dictionary entry sequence like Aabam, Aacch, Aade)

**Results on test data**: Absorbed 3,756 fragments across 6 files (22.3% reduction). Treatises correctly reconstructed: AGRICULTURE (38,474 words, 34 subsections), ALGEBRA (16,535 words, 71 subsections).

## Article Output Format

One JSONL per source file in `britannica_output/articles/`:

```json
{
  "article_id": "eb_3rd_1797_v01_0042",
  "title": "ABACUS",
  "edition": "3rd",
  "edition_year": 1797,
  "volume": 1,
  "source_file": "britannica_3rd_1797_vol01_A-ANG.jsonl",
  "type": "article",
  "char_start": 4200,
  "char_end": 5800,
  "text": "ABACUS, a table strewed over with dust...",
  "word_count": 420,
  "paragraph_count": 3,
  "keywords": ["mathematics", "counting", "Roman"],
  "author_attribution": null,
  "subsections": [
    {"title": "History", "paragraph_start": 0, "paragraph_end": 5}
  ]
}
```

Cross-references include a `target` field:
```json
{
  "type": "cross_reference",
  "title": "ABANGA",
  "target": "ADY"
}
```

## Files

```
britannica_parser/
    parse_britannica.py   # CLI orchestrator
    preprocess.py         # Phase 1: paragraph splitting
    classify.py           # Phase 2: async LLM classification
    assemble.py           # Phase 3: article assembly
    merge.py              # Phase 3.5: fragment merging
    models.py             # Dataclasses
    config.py             # Paths, API URL, batch size, concurrency
    verify.py             # Quality checks + statistics
```

## Configuration (`config.py`)

```python
INPUT_DIR  = Path("/home/jic823/plato/ocr_organized")   # local
OUTPUT_DIR = Path("/home/jic823/plato/britannica_output")

API_BASE   = "http://platogpu001:8000/v1"
MODEL      = "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ"

BATCH_SIZE      = 20    # paragraphs per LLM call
OVERLAP         = 2     # paragraph overlap between windows
MAX_CONCURRENT  = 10    # simultaneous API requests
REQUEST_TIMEOUT = 120   # seconds per API call
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 2048
```

On Plato, paths are adjusted: `INPUT_DIR=/home/jic823/ocr_organized`, `OUTPUT_DIR=/home/jic823/britannica_output`.

## Usage

```bash
# On Plato, all commands run from ~/britannica_parser with:
cd ~/britannica_parser
export PYTHONPATH=/home/jic823/britannica_parser/libs

# Run individual phases
python3 parse_britannica.py --phase 1
python3 parse_britannica.py --phase 2
python3 parse_britannica.py --phase 3
python3 parse_britannica.py --phase merge
python3 parse_britannica.py --phase verify

# Run everything
python3 parse_britannica.py --phase all

# Process a single file
python3 parse_britannica.py --phase 2 --file britannica_3rd_1797_vol01_A-ANG.jsonl

# Process one edition
python3 parse_britannica.py --phase 2 --edition 3rd

# Override API endpoint (for second GPU)
python3 parse_britannica.py --phase 2 --api-base http://platogpu002:8000/v1

# Adjust concurrency
python3 parse_britannica.py --phase 2 --concurrency 20
```

## Infrastructure

### vLLM Model Server

DeepSeek-R1-Distill-Llama-70B-AWQ served via vLLM in an Apptainer container on Plato HPC.

```bash
sbatch ~/serve_deepseek_70b.sh
```

**Critical settings**:
- `--quantization awq_marlin` (NOT `awq`) — 17x faster. The default `awq` uses unoptimized kernels.
- `--dtype float16` — required for AWQ models
- `--gpu-memory-utilization 0.90` — 37GB weights, ~43GB free for KV cache
- `--max-model-len 8192`
- Container: `~/containers/vllm-openai-v0.10.2.sif`
- SSL fix: bind `/etc/pki` and `/etc/ssl`, set `REQUESTS_CA_BUNDLE`
- Apptainer temp dir: `APPTAINER_TMPDIR=$HOME/.apptainer/tmp`

### aiohttp on Plato

Compute Canada's patched aiohttp 3.6.2 is incompatible with Python 3.11 (`asyncio.coroutines._DEBUG` removed). A working aiohttp 3.13.3 is installed at `~/britannica_parser/libs/`. Set `PYTHONPATH=/home/jic823/britannica_parser/libs` before running.

### Multi-GPU Parallel Processing

To run Phase 2 on two GPUs simultaneously:

1. Submit a second vLLM server targeting a specific node:
   ```bash
   # serve_deepseek_70b_gpu2.sh has: #SBATCH --nodelist=platogpu002
   sbatch ~/serve_deepseek_70b_gpu2.sh
   ```

2. Split the file list and run a second process:
   ```bash
   # gpu2_files.txt contains the second half of input filenames
   # run_gpu2.py reads that list and routes to platogpu002
   nohup python3 run_gpu2.py > ~/phase2_gpu2.log 2>&1 &
   ```

3. Monitor both:
   ```bash
   tail -f ~/phase2_full.log     # gpu001
   tail -f ~/phase2_gpu2.log     # gpu002
   ```

The classify.py skip logic prevents duplicate work — any file already classified (with no checkpoint) is skipped automatically.

## Quality Assessment

Compared against an earlier hybrid regex+LLM parser that produced 7,764 articles for the 1st edition:

### Classification consistency

Type distributions are remarkably stable across files:
- `body_text`: 63-65%
- `article_start`: 16-20%
- `running_header`: 6-12%
- `cross_reference`: 1-5%
- `subsection_start`: 3-4%

### After merging (1st edition, A-B volume)

| Metric | Before Merge | After Merge | Old Parser |
|--------|-------------|-------------|------------|
| Articles | 3,656 | 2,901 | 2,376 |
| Cross-refs | 657 | 657 | — |

The remaining ~22% excess over the old parser is a mix of legitimate short entries the old parser missed and some residual fragmentation.

### What works well

- Alphabetical ordering of articles is correct
- Short dictionary entries (Aabam, Aam, Aar) correctly preserved as separate articles
- Long treatises reconstructed with subsections after merging
- Cross-references properly identified with targets
- Running headers removed (~7-12% of paragraphs)
- Front/back matter separated

### Known remaining issues

- 3rd edition front matter paragraphs classified as `article_start` ("ENCYCLOPAEDIA BRITANNICA" appears as multiple articles at volume start)
- ~47 duplicate titles per volume (mostly legitimate: ABATEMENT has separate heraldry, law, and commerce senses)
- Some OCR artifact titles ("D D", "Fig. 4")
- Duplicate source files produce duplicate article sets
- Later editions (4th-8th) not yet validated — quality may vary

## Verification (`verify.py`)

Runs these checks on assembled articles:

- **Article counts** per file and edition
- **Alphabetical order** — flags titles that break alphabetical sequence (tolerance for same first 2 chars)
- **Text coverage** — what percentage of original text is accounted for by articles
- **Running header stats** — headers removed vs suspicious short ALL-CAPS paragraphs remaining in articles
- **Cross-reference validity** — checks if target articles exist within the same edition

```bash
python3 parse_britannica.py --phase verify
# Saves report to britannica_output/articles/verification_report.json
```

## Lessons Learned

1. **`guided_json` is essential for DeepSeek-R1 models** — without structured output constraints, the model's `<think>` tokens consume the entire output budget, producing no usable JSON.

2. **`awq_marlin` vs `awq` quantization** — vLLM's default AWQ kernels are ~17x slower than the Marlin-optimized ones. Always check vLLM startup logs for the warning: `"Use quantization=awq_marlin for faster inference"`.

3. **LLM classification works well for paragraph-level structure** but struggles with long treatises that span hundreds of paragraphs across many sliding windows. The model correctly identifies _that_ a heading exists but sometimes misclassifies subsection headings as article boundaries. Post-processing (the merger) handles this reliably.

4. **Sparse output format** — having the LLM only report non-body paragraphs minimizes output tokens and speeds generation. An empty `[]` response for all-body-text windows is fast.

5. **Sliding windows with overlap** give the LLM context about surrounding paragraphs, improving classification accuracy at window boundaries. The merge-by-centrality strategy for overlap zones works well.

6. **Defensive assembly rules** — back_matter only in last 10% of file, front_matter only before first article — prevent single misclassifications from corrupting entire volumes.
