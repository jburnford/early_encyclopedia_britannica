# Quick Reference: Outlier Review

## Commands

```bash
# Check progress
python3 scripts/review_outliers.py --status

# Start reviewing (auto-finds next unreviewed batch)
python3 scripts/review_outliers.py --edition 1815

# Review specific batch
python3 scripts/review_outliers.py --edition 1815 --batch 2
```

## Decision Syntax

| Decision | Syntax | When |
|----------|--------|------|
| **MERGE** | `MERGE AGRICULTURE` | Outlier belongs inside another article |
| **RENAME** | `RENAME BURNTISLAND` | OCR error in headword |
| **KEEP** | `KEEP Valid biography` | Actually valid (rare!) |
| **OCR_REVIEW** | `OCR_REVIEW` | Need to check raw OCR |
| Skip | `s` | Come back to this later |
| Quit | `q` | Stop and save progress |

## Quick Pattern Recognition

| If headword... | Then likely... |
|----------------|----------------|
| Starts with THEORY OF, PRACTICE OF, GENERAL | MERGE into main article |
| Starts with NEW, CAPE, ISLE, ST | MERGE into parent geography |
| Starts with THIS, THESE, WHEN, WHERE | MERGE into previous article |
| Starts with a person's first name | MERGE into BIOGRAPHY or parent treatise |
| Is Latin anatomical/botanical term | MERGE into ANATOMY/BOTANY |
| Looks like OCR error (wrong letter) | RENAME to correct spelling |
| Is garbled text (KXE, PM') | Usually MERGE or OCR_REVIEW |

## Batch Overview

| Edition | Batches | Start with |
|---------|---------|------------|
| 1810 | 1 | `--edition 1810` (smallest, good for practice) |
| 1771 | 2 | `--edition 1771` |
| 1778 | 2 | `--edition 1778` |
| 1797 | 3 | `--edition 1797` |
| 1815 | 3 | `--edition 1815` |
| 1823 | 4 | `--edition 1823` |
| 1842 | 5 | `--edition 1842` |
| 1860 | 6 | `--edition 1860` (largest) |

## After All Reviews

```bash
# Preview fixes
python3 scripts/apply_outlier_fixes.py --preview

# Apply fixes
python3 scripts/apply_outlier_fixes.py --apply

# Regenerate website
python3 generate_site_optimized.py
```
