# Encyclopedia Britannica LLM Correction Project - Continuation Prompt

## Project Overview
Reviewing flagged articles across 8 editions (1771-1860) of the Encyclopedia Britannica. Each article needs a decision: KEEP (valid standalone), MERGE (into parent article), or DELETE (errata/front matter).

## Current Progress
- **Total decisions**: 2,791
- **Editions complete**: 4/8

### Edition Status
| Edition | Articles | Status | KEEP | MERGE | DELETE |
|---------|----------|--------|------|-------|--------|
| 1771 | 135 | ✅ Complete | 35 | 95 | 5 |
| 1778 | 377 | ✅ Complete | 272 | 68 | 37 |
| 1797 | 235 | ✅ Complete | 103 | 124 | 8 |
| 1810 | 2,044 | ✅ Complete | 1,950 | 90 | 4 |
| 1815 | ? | 🔄 **START HERE** | - | - | - |
| 1823 | ? | ⏳ Pending | - | - | - |
| 1842 | ? | ⏳ Pending | - | - | - |
| 1860 | ? | ⏳ Pending | - | - | - |

## Working Directory
```
/home/jic823/1815EncyclopediaBritannicaNLS/llm_corrections
```

## Files
- `corrections/decisions.json` - All recorded decisions
- `state/batch_YYYY_NNN.json` - Batch data files

## Efficient Meta-Analysis Approach

For large editions (like 1810 with 2,044 articles), use this approach:

### Step 1: Analyze patterns
```python
# Check surrounding_letter vs headword first letter
# MATCH = headword[0] == surrounding_letter → likely valid (volume metadata issue)
# MISMATCH = headword[0] != surrounding_letter → needs manual review
```

### Step 2: Bulk-KEEP all MATCH articles
These are valid articles incorrectly flagged due to volume boundary metadata issues.

### Step 3: Manual review only MISMATCH articles
Examine text previews and make KEEP/MERGE/DELETE decisions.

## Key Decision Signals

### KEEP signals
- Headword letter matches surrounding_letter (valid article, volume metadata issue)
- Valid biography/geography/subject despite OCR errors
- Pharmacy/chemistry Latin terms (BALSAMUM, OLEUM, PIX, TEREBINTHINA, etc.)
- J/I boundary articles (historically grouped together)
- Cross-references with valid content

### MERGE signals
- Sentence fragment headwords (THIS/THESE/WHEN/WHILE/BEFORE/ACCORDING TO...)
- Section headers (PROBLEM, REMARK, EXAMPLE, COROLLARY, DEFINITIONS, AXIOM...)
- Roman numerals (VII, VIII, XII, XIII)
- Headword letter ≠ surrounding_letter + fragment content
- Linnaean classification sections (MONANDRIA, DECANDRIA, PENTANDRIA...)
- Content clearly continues adjacent article

### DELETE signals
- Title pages, dedication pages (ENLARGED AND IMPROVED, TO THE KING...)
- Errata/corrigenda sections
- Front matter, volume introductions
- Pure cross-references with no content

## Task: Process 1815 Edition

### 1. Run meta-analysis
```python
python3 << 'EOF'
import json
import os
from collections import Counter

articles = []
for i in range(1, 100):
    path = f'state/batch_1815_{i:03d}.json'
    if os.path.exists(path):
        with open(path) as f:
            batch = json.load(f)
            articles.extend(batch['articles'])

print(f"Total 1815 flagged articles: {len(articles)}")

# Analyze MATCH vs MISMATCH
matches = mismatches = 0
for a in articles:
    hw = a['flagged'].get('headword', '')
    surr = a['flagged'].get('surrounding_letter', '')
    if hw and surr:
        if hw[0].upper() == surr:
            matches += 1
        else:
            mismatches += 1

print(f"MATCH (bulk KEEP): {matches}")
print(f"MISMATCH (manual review): {mismatches}")
EOF
```

### 2. Bulk-KEEP matching articles
### 3. Manual review mismatches with LLM analysis
### 4. Record all decisions to corrections/decisions.json

## To Start
Read this file, then run the meta-analysis on 1815 edition and process using the efficient bulk approach demonstrated for 1810.
