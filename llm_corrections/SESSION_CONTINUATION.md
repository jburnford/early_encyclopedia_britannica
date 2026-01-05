# LLM Article Merge Correction - Session Continuation

## Project Overview

We're correcting mis-parsed articles in the Encyclopedia Britannica Historical Corpus (1771-1860). The parser sometimes created separate articles from section headings, resulting in 4,354 flagged articles across 8 editions that need human/LLM review.

### The Problem

Example: "BLACK CHALK" was parsed as a standalone article, but it's actually a sub-entry under the main "CHALK" article on the same page. Similarly, "GENERAL OBSERVATIONS" might be a section heading within a larger treatise like "LAW".

### The Solution

An LLM-assisted review system that:
1. Generates batches of flagged articles with context
2. Identifies two types of parent candidates:
   - **Semantic parent**: Headword-based match on same/nearby pages (e.g., "BLACK CHALK" → "CHALK")
   - **Page-adjacent parent**: Previous article by position (for section headings in treatises)
3. Presents both options in a prompt for decision-making
4. Records decisions (MERGE/KEEP/DELETE) to JSON
5. Applies merges to the JSONL corpus files

---

## Current Progress

| Edition | Flagged | Batches | Status |
|---------|---------|---------|--------|
| 1771 | 135 | 3 | Batch 1: 50 decisions recorded |
| 1778 | 377 | 8 | Not started |
| 1797 | 235 | 5 | Not started |
| 1810 | 1,994 | 40 | Not started |
| 1815 | 275 | 6 | Not started |
| 1823 | 281 | 6 | Not started |
| 1842 | 458 | 10 | Not started |
| 1860 | 599 | 12 | Not started |
| **TOTAL** | **4,354** | **90** | |

### Decisions Made (1771 Batch 1)

- **50 articles reviewed**
- 28 MERGE decisions (into ANATOMY, BOTANY, CHEMISTRY, etc.)
- 22 KEEP decisions (standalone entries)
- 0 DELETE decisions

---

## File Structure

```
llm_corrections/
├── generate_batches.py      # Creates review batches with parent candidates
├── llm_article_corrector.py # Generates prompts, records decisions
├── apply_merges.py          # Applies final corrections to JSONL
├── state/
│   ├── progress.json        # Overall progress tracking
│   └── batch_*.json         # Batch data with article contexts
├── prompts/
│   └── prompt_*.md          # Generated prompts for review
└── corrections/
    └── decisions.json       # All recorded decisions
```

---

## Key Commands

```bash
cd /home/jic823/1815EncyclopediaBritannicaNLS/llm_corrections

# Show current progress
python3 llm_article_corrector.py --status

# Generate prompt for next batch
python3 llm_article_corrector.py --prompt

# Generate prompt for specific batch
python3 llm_article_corrector.py --prompt --edition 1771 --batch 2

# Regenerate batches (if algorithm changes)
python3 generate_batches.py --edition 1771

# Preview merge changes (dry run)
python3 apply_merges.py --preview

# Apply merges to JSONL files
python3 apply_merges.py --apply
```

---

## Algorithm Details

### Semantic Parent Matching

For multi-word headwords like "BLACK CHALK":
1. Extract potential parent headwords: "CHALK"
2. Search all articles within ±1 page
3. Score matches by: same page (+5), shorter suffix (+1)
4. Return best match

### Page-Adjacent Parent

For section headings embedded in treatises:
1. Sort articles by start page
2. Find flagged article's position
3. Return previous non-flagged article within 2 pages

---

## Decision Criteria

**MERGE** when:
- Flagged headword is generic ("GENERAL OBSERVATIONS", "DISEASES OF THE FEET")
- Text continues the topic of parent article
- Very short article (<50 words) on same page as potential parent
- First letter doesn't match surrounding articles

**KEEP** when:
- Article is a complete, self-contained topic
- Has proper encyclopedia structure (definition, explanation)
- Could stand alone even if alphabetically misplaced

**DELETE** when:
- Structural marker (END_OF_VOLUME, PLATE_EXPLANATION)
- OCR garbage with no meaningful content

---

## Next Steps

1. **Continue 1771 edition**: Process batches 2 and 3 (85 remaining articles)
2. **Verify merges**: Run `apply_merges.py --preview` to check decisions
3. **Apply corrections**: Run `apply_merges.py --apply` after verification
4. **Regenerate site**: Rebuild HTML from corrected JSONL
5. **Process remaining editions**: 1778, 1797, 1810, 1815, 1823, 1842, 1860

---

## Session Continuation Prompt

Copy and paste the following to start a new session:

```
I'm continuing work on the Encyclopedia Britannica LLM correction project.

**Context**: We have 4,354 flagged articles across 8 editions (1771-1860) that need review to determine if they should MERGE into parent articles, KEEP as standalone, or DELETE.

**Current state**:
- 1771 batch 1 complete (50 decisions in corrections/decisions.json)
- Algorithm improved to detect both semantic parents (headword match) and page-adjacent parents
- Batches regenerated with dual-parent format

**Files**:
- Working directory: /home/jic823/1815EncyclopediaBritannicaNLS/llm_corrections
- Decisions: corrections/decisions.json
- Batches: state/batch_1771_*.json

**Next task**: Process 1771 batch 2 (50 articles). Run:
python3 llm_article_corrector.py --prompt --edition 1771 --batch 2

Then review the generated prompt and record decisions.
```

---

## Notes

- The `decisions.json` file uses `"into"` field for merge targets (headword, not article_id)
- Page numbers in prompts are for reference but position-based matching is more reliable
- Some headwords have OCR errors (e.g., "DICCTIONARY" instead of "DICTIONARY")
- Very long articles (>10K words) are often legitimate treatises, not merge candidates
