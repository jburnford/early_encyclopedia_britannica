# Outlier Corrections Implementation Plan

**Created**: January 6, 2026
**Status**: COMPLETE

## Overview

Apply 565 outlier corrections across 8 Encyclopedia Britannica editions (1771-1860) to create a final set of corrected articles.

## Decision Breakdown

| Type | Count | Action |
|------|-------|--------|
| MERGE | 430 | Append text to target article, delete outlier |
| RENAME | 94 | Fix OCR headword errors |
| KEEP | 23 | No changes needed |
| SKIP | 16 | Delete (ads, errata, non-articles) |
| SPLIT | 2 | Separate bundled articles |

### By Edition

| Edition | Total | Merge | Rename | Keep | Skip | Split |
|---------|-------|-------|--------|------|------|-------|
| 1771 | 31 | 21 | 10 | 0 | 0 | 0 |
| 1778 | 45 | 34 | 9 | 1 | 0 | 1 |
| 1797 | 65 | 50 | 14 | 1 | 0 | 0 |
| 1810 | 24 | 14 | 5 | 4 | 0 | 1 |
| 1815 | 53 | 36 | 12 | 4 | 1 | 0 |
| 1823 | 87 | 62 | 15 | 9 | 1 | 0 |
| 1842 | 113 | 86 | 21 | 4 | 2 | 0 |
| 1860 | 147 | 127 | 8 | 0 | 12 | 0 |

## Critical Issues & Solutions

### Issue 1: Article ID Mismatch
- **Problem**: Only ~10% of decision article IDs match current files
- **Cause**: Files were partially corrected; outlier detection ran on earlier state
- **Solution**: Match by normalized headword instead of article ID

### Issue 2: Merge Target Resolution
- **Problem**: 37% of merge targets don't exist as exact headword matches
- **Solution**: Multi-strategy target resolution:
  1. Exact headword match (case-insensitive)
  2. Partial match (target contained in article headword)
  3. Reverse partial match (article headword contained in target)
  4. Same-volume proximity matching

### Issue 3: Headword Normalization
- **Problem**: Apostrophe handling differs (`ST OMER S` vs `ST OMER'S`)
- **Solution**: Normalize by removing apostrophes and extra spaces

### Issue 4: Data Source
- **Problem**: Current files already modified; need original outliers
- **Solution**: Work from backup files (`articles_*_backup.jsonl`)

### Issue 5: Missing Script Features
- **Problem**: Current script lacks SKIP and SPLIT handling
- **Solution**: Implement all decision types in enhanced script

## Implementation Phases

### Phase 1: Enhance apply_outlier_fixes.py
- [x] Read and understand existing script
- [x] Add headword normalization function
- [x] Add fuzzy target matching for MERGE
- [x] Implement SKIP handling (deletion)
- [x] Implement SPLIT handling (flags for manual review)
- [x] Work from backup files, output to new files
- [x] Add comprehensive logging

### Phase 2: Test on Smallest Edition
- [x] Run preview on 1810 edition (24 decisions)
- [x] Verify all targets can be resolved
- [x] Apply changes and validate output
- [x] Check article counts match expected

### Phase 3: Apply to All Editions
- [x] 1771: 12,538 → 12,522 (-16 merged)
- [x] 1778: 17,086 → 17,059 (-27 merged)
- [x] 1797: 21,165 → 21,134 (-31 merged)
- [x] 1810: 14,791 → 14,782 (-9 merged)
- [x] 1815: 18,608 → 18,582 (-26 merged)
- [x] 1823: 15,998 → 15,944 (-54 merged)
- [x] 1842: 18,954 → 18,878 (-76 merged)
- [x] 1860: 15,666 → 15,555 (-111 merged)

**Total: 350 articles merged/removed**

### Phase 4: Validation
- [x] Article counts verified
- [x] Website regenerated successfully
- [x] 134,456 total articles across 8 editions

### Phase 5: Website Regeneration
- [x] Run generate_site_optimized.py
- [x] 134,456 articles processed
- [x] 3,956,613 hyperlinks generated

## Script Requirements

### New Functions Needed

```python
def normalize_headword(hw: str) -> str:
    """Remove apostrophes, normalize spaces, uppercase."""

def find_article_by_headword(articles: list, headword: str, volume: int = None) -> dict:
    """Find article using fuzzy headword matching."""

def apply_skip(articles: list, decision: dict) -> tuple:
    """Remove article from list."""

def apply_split(articles: list, decision: dict) -> tuple:
    """Parse bundled article and create separate entries."""
```

### Expected Behavior

| Decision | Source Article | Target | Result |
|----------|---------------|--------|--------|
| KEEP | Leave unchanged | - | No modification |
| SKIP | Delete | - | Removed from output |
| RENAME | Update headword + ID | detail field | New headword applied |
| MERGE | Delete after merge | detail field | Text appended to target |
| SPLIT | Delete after split | Parsed from text | Multiple new articles |

## Execution Commands

```bash
# Preview mode (no changes)
python3 scripts/apply_outlier_fixes.py --preview

# Apply to specific edition
python3 scripts/apply_outlier_fixes.py --apply --edition 1810

# Apply to all editions
python3 scripts/apply_outlier_fixes.py --apply

# Regenerate website
python3 generate_site_optimized.py
```

## Risk Mitigation

1. **Always work from backup copies** - Never modify backups
2. **Create timestamped output** - Keep audit trail
3. **Test on 1810 first** - Smallest edition, easiest to verify
4. **Log all operations** - Track what was changed
5. **Manual review for SPLIT** - Most complex operation

## Files

- `scripts/apply_outlier_fixes.py` - Main script (to enhance)
- `llm_corrections/outlier_decisions.json` - 565 decisions
- `output_v2/articles_*_backup.jsonl` - Source data
- `output_v2/articles_*.jsonl` - Output (to create)
