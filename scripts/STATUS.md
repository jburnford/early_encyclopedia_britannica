# Britannica Parser — Status Report

**Last updated**: 2026-02-25

## Pipeline Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 (paragraph splitting) | **Complete** | 1,650,866 paragraphs from 210 files |
| Phase 2 (LLM classification) | **Complete** | 210/210 files classified across 2 GPUs |
| Phase 3 (assembly) | **Complete** | 220,056 raw articles assembled |
| Phase 3.5 (merge) | **Complete** | 146,585 articles after merging (33% reduction) |
| Verification | **Complete** | Report generated |

## Final Results

### Output Summary

| Metric | Value |
|--------|-------|
| Total entries | **146,585** |
| Articles | 114,657 |
| Cross-references | 31,928 |
| Total words | **155,714,863** |
| Pre-merge articles | 220,056 |
| Fragments absorbed | 73,471 (33.4%) |

### By Edition

| Edition | Year | Files | Articles | Cross-refs | Words |
|---------|------|-------|----------|------------|-------|
| 1st | 1771 | 7 | 9,561 | 3,555 | 6,061,361 |
| 2nd | 1778 | 30 | 16,684 | 8,132 | 17,762,953 |
| 3rd | 1797 | 40 | 20,413 | 11,110 | 36,826,027 |
| 4th | 1810 | 20 | 7,408 | 3,028 | 21,268,989 |
| 5th | 1815 | 12 | 5,384 | 2,074 | 12,103,027 |
| 6th | 1823 | 35 | 10,754 | 3,972 | 22,791,735 |
| 7th | 1842 | 30 | 3,672 | 1,860 | 18,760,629 |
| 8th | 1860 | 36 | 4,253 | 2,143 | 20,140,142 |

### Verification Highlights

- **Running headers removed**: 142,836 (successfully stripped from article text)
- **Suspicious remaining headers**: 301 (short ALL-CAPS paragraphs still in articles)
- **Alphabetical order issues**: 17,994 (mostly legitimate — multi-sense entries, subsections)
- **Cross-reference resolution**: 22.5% (5th ed) to 59.7% (2nd ed) — expected given incomplete volume coverage within editions
- **Text coverage**: 122.7% average (>100% due to overlapping char ranges from merged articles)

## Phase 2 Classification — Completed Run

### Infrastructure

| Resource | Node | Job ID | Status |
|----------|------|--------|--------|
| vLLM server 1 | platogpu001 | 4941867 | Running (user has other work) |
| vLLM server 2 | platogpu002 | 4942855 | Running (user has other work) |
| Classifier 1 | login node (tmux `gpu1`) | — | Complete |
| Classifier 2 | login node (tmux `gpu2`) | — | Complete |

- **Model**: DeepSeek-R1-Distill-Llama-70B-AWQ via vLLM
- **Quantization**: `awq_marlin` (17x faster than default `awq`)
- **Structured output**: `guided_json` forces JSON, bypasses `<think>` tokens
- **Throughput**: ~0.1-0.2 calls/sec per GPU

### Process Management

Both classifiers ran in **tmux sessions** on the Plato login node:
```bash
tmux attach -t gpu1    # GPU1 classifier
tmux attach -t gpu2    # GPU2 classifier
```

Launcher scripts (needed because tmux defaults to Python 3.7):
- `/home/jic823/launch_gpu1.sh` — runs `parse_britannica.py --phase 2` with Python 3.11
- `/home/jic823/launch_gpu2.sh` — runs `run_gpu2.py` with Python 3.11

Log files:
- `/home/jic823/phase2_full_r2.log` (GPU1, final run)
- `/home/jic823/phase2_gpu2_r2.log` (GPU2, final run)
- `/home/jic823/phase2_full.log` (GPU1, original run — 67 completions)
- `/home/jic823/phase2_gpu2.log` (GPU2, original run — 71 completions)

### File Split

- **GPU1** (`parse_britannica.py --phase 2`): Processes all 210 files in sorted order, skips already-classified
- **GPU2** (`run_gpu2.py`): Processes second half of files (100 files from `gpu2_files.txt`)
- Skip logic in `classify.py` prevents duplicate work

### Completion by Edition

| Edition | Year | Volumes | Status |
|---------|------|---------|--------|
| 1st | 1771 | 7 | **Done** |
| 2nd | 1778 | 30 | **Done** |
| 3rd | 1797 | 40 | **Done** |
| 4th | 1810 | 20 | **Done** |
| 5th | 1815 | 12 | **Done** |
| 6th | 1823 | 35 | **Done** |
| 7th | 1842 | 30 | **Done** |
| 8th | 1860 | 36 | **Done** |

### Timeline

- **Started**: Feb 20, 2026 (~10:00)
- **GPU2 added**: Feb 20 (~20:40)
- **Processes crashed**: Feb 23 (~06:30) — SSH/login node killed nohup processes
- **Restarted in tmux**: Feb 23 (~06:40) — resumed from checkpoints
- **Timeout fix**: Feb 23 (~07:00) — reduced MAX_CONCURRENT 10→5, timeout 120→300s
- **Phase 2 complete**: Feb 25 — all 210 files classified
- **Assembly + merge + verify**: Feb 25 — full pipeline finished
- **Total elapsed**: ~5 days

### Incidents

1. **Feb 20 — DeepSeek-R1 thinking tokens**: Model consumed entire output budget on `<think>` tokens. Fixed with `guided_json` structured output.
2. **Feb 20 — AWQ slow kernels**: vLLM using unoptimized AWQ kernels (1.6 tok/sec). Fixed by switching to `awq_marlin` (27 tok/sec, 17x speedup).
3. **Feb 23 — nohup processes died**: Both classification processes killed when SSH session/login node cleaned up. Restarted in tmux sessions. No data lost — checkpoints preserved progress.
4. **Feb 23 — Python version in tmux**: tmux shells on Plato default to Python 3.7 (from `gentoo/2020` CVMFS). Fixed with explicit `/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/python3.11` in launcher scripts.
5. **Feb 23 — Timeout cascade**: After tmux restart, MAX_CONCURRENT=10 overwhelmed vLLM with guided_json requests. Reduced to 5 concurrent + 300s timeout.

### Checkpointing

Each file saves a checkpoint every 50 windows to `{stem}.checkpoint.json` in the classifications directory. On restart, the classifier resumes from the last checkpoint. Completed files are skipped entirely.

## Quality Assessment

### Merge Results (Full Dataset)

| Metric | Before Merge | After Merge |
|--------|-------------|-------------|
| Total articles | 220,056 | 146,585 |
| Fragments absorbed | — | 73,471 (33.4%) |

### Merge Results (6 Early Test Files)

| Metric | Before Merge | After Merge |
|--------|-------------|-------------|
| Total articles | 16,878 | 13,122 |
| Fragments absorbed | — | 3,756 (22.3%) |

### Comparison with Earlier Hybrid Parser (1st Edition)

| Volume | Old Parser | New (post-merge) | Delta |
|--------|-----------|------------------|-------|
| Vol 1 (A-B) | 2,376 | 2,901 | +22% |
| Vol 2 (C-L) | 2,913 | 3,248 | +11% |

Remaining excess is mostly legitimate short entries the old parser missed.

### Classification Distribution (stable across files)

| Type | Percentage |
|------|-----------|
| body_text | 63-65% |
| article_start | 16-20% |
| running_header | 6-12% |
| cross_reference | 1-5% |
| subsection_start | 3-4% |

### Known Issues

- 3rd edition front matter leaks as `article_start` ("ENCYCLOPAEDIA BRITANNICA" as articles)
- ~47 duplicate titles per volume (mostly legitimate multi-sense entries)
- Some OCR garbage titles ("D D", "Fig. 4")
- Duplicate source OCR files (e.g., vol02_part2 ≈ vol02_part4) produce duplicate articles
- Text coverage >100% due to overlapping char ranges from merged articles
- Later editions (4th-8th) not yet validated against external sources

## Next Steps

1. **Deduplicate** source files that are near-identical OCR scans
2. **Compare** full output against earlier hybrid parser across all editions
3. **Investigate** alphabetical order issues (17,994 flags) — likely benign but worth sampling
4. **Fix text coverage** metric (overlapping char ranges inflate coverage >100%)
5. **Export** final dataset
