# Cleanup Summary Report

Generated: 2026-01-04 09:13:40

## Overall Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Articles | 136,848 | 100% |
| Keep (clean) | 131,620 | 96.2% |
| Remove | 951 | 0.7% |
| Flag for Review | 4,277 | 3.1% |

## By Edition

| Edition | Total | Keep | Remove | Flag |
|---------|-------|------|--------|------|
| 1771 | 12,624 | 12,428 | 72 | 124 |
| 1778 | 17,219 | 16,765 | 91 | 363 |
| 1797 | 21,180 | 20,818 | 139 | 223 |
| 1810 | 15,070 | 12,993 | 89 | 1,988 |
| 1815 | 18,617 | 18,204 | 143 | 270 |
| 1823 | 16,011 | 15,590 | 156 | 265 |
| 1842 | 19,669 | 19,083 | 132 | 454 |
| 1860 | 16,458 | 15,739 | 129 | 590 |

## Issues by Type (All Editions)

| Issue Type | Count |
|------------|-------|
| out_of_range | 4,923 |
| sentence_fragment | 806 |
| structural_marker | 235 |
| ocr_error | 163 |
| too_long | 16 |

## Recommendations

1. **951 articles** will be removed as they are structural markers or parsing errors
2. **4,277 articles** are flagged for manual review (very long, short, or out-of-range)
3. Run with `--fix` to generate cleaned JSONL files
4. After fixing, regenerate the site with `python3 generate_site_optimized.py`
