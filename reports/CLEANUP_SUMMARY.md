# Cleanup Summary Report

Generated: 2026-01-04 09:55:09

## Overall Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Articles | 135,897 | 100% |
| Keep (clean) | 131,543 | 96.8% |
| Remove | 0 | 0.0% |
| Flag for Review | 4,354 | 3.2% |

## By Edition

| Edition | Total | Keep | Remove | Flag |
|---------|-------|------|--------|------|
| 1771 | 12,552 | 12,417 | 0 | 135 |
| 1778 | 17,128 | 16,751 | 0 | 377 |
| 1797 | 21,041 | 20,806 | 0 | 235 |
| 1810 | 14,981 | 12,987 | 0 | 1,994 |
| 1815 | 18,474 | 18,199 | 0 | 275 |
| 1823 | 15,855 | 15,574 | 0 | 281 |
| 1842 | 19,537 | 19,079 | 0 | 458 |
| 1860 | 16,329 | 15,730 | 0 | 599 |

## Issues by Type (All Editions)

| Issue Type | Count |
|------------|-------|
| out_of_range | 4,254 |
| alphabetical_break | 480 |
| sentence_fragment | 90 |
| too_long | 9 |
| ocr_error | 7 |

## Recommendations

1. **0 articles** will be removed as they are structural markers or parsing errors
2. **4,354 articles** are flagged for manual review (very long, short, or out-of-range)
3. Run with `--fix` to generate cleaned JSONL files
4. After fixing, regenerate the site with `python3 generate_site_optimized.py`
