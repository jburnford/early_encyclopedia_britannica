# Next Session: Post-Split Cleanup & Graph Assembly

## What Was Done (March 31 - April 2, 2026)

### OCR-Stable Paragraph IDs & Embeddings
- 1,044,777 paragraphs embedded with voyage-4-large (~$19)
- ID format: `{source_file_stem}__char_{ocr_absolute_position}` — stable across article re-parsing
- Checkpointing: resumes interrupted runs, detects old-format IDs

### Swallowed Article Detection Pipeline
1. **Paragraph similarity breaks**: 56,167 raw detections across 8 editions
2. **Per-category thresholds**: reduced to 12,990 (mid_word all, mid_sentence all, new_headword <0.35, person_bio <0.20, topic_change <0.20)
3. **Cross-edition grouping**: 1,322 unique breaks in 2+ editions
4. **Gap cross-referencing**: matched against 1,749 known gaps
5. **Noise filtering**: removed common words (THE, IN, VOL, CHAP, etc.)
6. **Alphabetical scoring** (0-5): scored distance between parent and break headword
   - Score 5: exact neighbor (shared 3+ prefix) — 2,209 fixes
   - Score 4: close neighbor (shared 2+, or backward with 4+ shared) — 186 fixes
   - Score 0: far away or wrong direction — 345 rejected

### Major Split Operation (2 passes)
- **2,485 total articles split** across all 8 editions (2,370 first pass + 115 second pass)
- Applied to **export files** (data/export/eb_*.jsonl) using exact paragraph character positions
- Paragraph article_ids remapped after first pass to enable second-pass detection
- Backup in `data/backup_20260401/` (3.3GB: articles + export + fix_mega_articles.py + gaps + cross-edition index)

| Edition | Original | Current | Gained |
|---------|----------|---------|--------|
| 1771 | 13,419 | 13,445 | +26 |
| 1778 | 12,751 | 12,896 | +145 |
| 1797 | 20,956 | 21,368 | +412 |
| 1810 | 21,435 | 21,989 | +554 |
| 1815 | 20,942 | 21,575 | +633 |
| 1823 | 20,738 | 21,257 | +519 |
| 1842 | 18,688 | 18,838 | +150 |
| 1860 | 15,040 | 15,086 | +46 |
| **TOTAL** | **143,969** | **146,454** | **+2,485** |

### Validation
- Spot-checks: all split boundaries are clean (different topics, alphabetically adjacent)
- **231 of 1,749 original gaps filled** (99 from auto_split, 132 pre-existing)
- 537 new unique article titles introduced
- 1,518 gaps remaining (mostly OCR_GAP and EDITORIAL)

## What Needs to Be Done Next

### 1. Reconcile Article Files with Export Files
- Export files have splits applied; article files (data/articles/) still have pre-split version
- Article files use per-volume IDs (eb_4th_1810_v17_0088) vs export uses sequential IDs (eb_4th_1810_014779)
- Options: rebuild articles from export, or treat export as canonical going forward
- NER pipeline reads exports; cross-edition index built from exports — export is effectively canonical

### 2. Third Detection Pass (diminishing returns)
- Second pass found 115 more splits; a third might find a few more
- Current detection shows 23K raw breaks — most are noise at this point
- Could be worth one more pass, or move on to other priorities

### 3. Lower-Confidence Fixes (score 1-3)
- Score-1 (near_back) had many real swallows: PERSEVERANCE→PERSEUS, PORTLANDIA→PORT-LOUIS
- ~95 fixes at score 1-3 worth manual review

### 4. Update Paragraph-Level RAG
- Paragraph embeddings have remapped article_ids (done)
- Build consolidated numpy array for fast search
- Update `graphrag/query.py` to search paragraphs instead of chunks

### 5. Neo4j Graph Assembly
- 146,454 articles across 8 editions — cleaner than ever
- Cross-edition linking via normalized headwords
- Entity data from NER (1,157,244 entities) ready to integrate

## Key Files
| File | Purpose |
|------|---------|
| `graphrag/embed_articles.py` | Paragraph embedding with OCR-stable IDs |
| `graphrag/detect_swallowed.py` | Break detection with thresholds + cross-edition grouping |
| `graphrag/generate_fixes.py` | Fix spec generation with alphabetical scoring |
| `scripts/apply_auto_splits.py` | Character-position-based splitting (operates on export files) |
| `scripts/fix_mega_articles.py` | Original manual fixes (158 entries, operates on article files) |
| `data/proposed_fixes.jsonl` | 2,814 fix specs (2,395 with alpha_score >= 4) |
| `data/swallowed_detections.jsonl` | 12,990 filtered detections |
| `data/backup_20260401/` | Pre-split backup (articles + export + gaps + index) |
| `data/embeddings/eb_*.paragraphs.jsonl` | 1,044,777 paragraph embeddings |

## Infrastructure
- **Voyage API**: voyage-4-large, 3M TPM / 2K RPM, ~$19 for full corpus
- **Neo4j**: bolt://206.12.90.118:7687
- Paragraph embeddings survive article re-parsing (OCR-stable IDs)
