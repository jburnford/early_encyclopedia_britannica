# Alphabetic Outlier Correction Guide

## The Problem

The article parser sometimes extracts section headings, subsections, or mid-paragraph text as separate articles. These errors are detectable because **within each volume, articles should follow alphabetical order by page position**.

When an article's first letter doesn't match the expected alphabetical progression at that page, it's an **outlier** - almost certainly a parsing error.

### Example

In 1815 Edition Volume 8 (letters E-F), pages 1-848:
- Page 200: ENTOMOLOGY (correct - E)
- Page 201: SPHINX (outlier - S appears where E expected)
- Page 250: EPIDEMIC (correct - E)

SPHINX is the name of a hawk-moth genus discussed within the ENTOMOLOGY article. The parser incorrectly extracted it as a standalone 40,000-word article.

---

## Statistics

| Edition | Outliers | Batches |
|---------|----------|---------|
| 1771    | 31       | 2       |
| 1778    | 39       | 2       |
| 1797    | 64       | 3       |
| 1810    | 24       | 1       |
| 1815    | 53       | 3       |
| 1823    | 87       | 4       |
| 1842    | 113      | 5       |
| 1860    | 147      | 6       |
| **Total** | **558** | **26**  |

---

## Decision Types

For each outlier, choose ONE decision:

### MERGE (most common - ~90% of cases)

The outlier text should be appended to another article. Specify the target headword.

**When to use:**
- Section headers within treatises ("THEORY OF AGRICULTURE" → AGRICULTURE)
- Subsections ("HERESY OF ALMARIC" → ALMARIC or HERESY)
- Sentence fragments ("THIS IS THE BEST" → previous article)
- Geographic subdivisions ("NEW ALBION" → AMERICA)
- Technical terms within treatises ("VENTRICULUS" → ANATOMY)

**Format:** `MERGE <target_headword>`

**Examples:**
```
MERGE AGRICULTURE
MERGE ENTOMOLOGY
MERGE AMERICA
```

### RENAME (OCR errors - ~5% of cases)

The headword has an OCR error but the article is otherwise valid. Provide the corrected headword.

**When to use:**
- Single letter OCR errors ("RUNTISLAND" → "BURNTISLAND")
- Missing/extra letters ("IALOGISM" → "DIALOGISM")
- Letter confusion ("POLCMOTE" → "FOLKMOTE")

**Format:** `RENAME <corrected_headword>`

**Examples:**
```
RENAME BURNTISLAND
RENAME DIALOGISM
RENAME FOLKMOTE
```

### KEEP (rare - ~2% of cases)

The article is genuinely valid despite appearing out of alphabetical order. Requires justification.

**When to use:**
- Proper nouns that happen to have unusual letter placement
- Valid cross-references or see-also entries
- Genuinely misalphabetized by the original editors

**Format:** `KEEP <reason>`

**Examples:**
```
KEEP Valid biography - appears in biographical supplement section
KEEP Cross-reference entry for alternate spelling
```

### OCR_REVIEW (complex cases - ~3% of cases)

The outlier requires examination of the raw OCR to understand the page structure.

**When to use:**
- Unclear merge target (multiple candidates equally plausible)
- Possible page numbering errors in source
- Garbled text that needs source verification

**Format:** `OCR_REVIEW`

---

## How to Run a Review Session

### 1. Check Current Progress

```bash
cd /home/jic823/1815EncyclopediaBritannicaNLS
python3 scripts/review_outliers.py --status
```

This shows:
- Total outliers and how many reviewed
- Decision breakdown (merge/rename/keep/ocr_review)
- Progress by edition

### 2. Start Reviewing

**Review next unreviewed batch for an edition:**
```bash
python3 scripts/review_outliers.py --edition 1815
```

**Review a specific batch:**
```bash
python3 scripts/review_outliers.py --edition 1815 --batch 1
```

**Review all batches for an edition:**
```bash
python3 scripts/review_outliers.py --edition 1815 --all
```

### 3. During Review

For each outlier, you'll see:
- Headword and page numbers
- Why it's flagged (letter mismatch)
- Text preview (first 800 chars)
- Merge candidates (previous and next articles with their text)
- Context (surrounding article headwords)

Enter your decision:
- `MERGE AGRICULTURE` - merge into AGRICULTURE
- `RENAME BURNTISLAND` - fix OCR error
- `KEEP Valid biography` - keep with reason
- `OCR_REVIEW` - flag for later
- `s` or `skip` - skip this one for now
- `q` or `quit` - stop and save progress

### 4. Progress is Saved Automatically

Every decision is immediately saved to:
```
llm_corrections/outlier_decisions.json
```

You can stop at any time (`q` or Ctrl+C) and resume later.

---

## Common Patterns to Recognize

### Pattern 1: Section Headers in Treatises

**Clue:** Headword starts with "THEORY OF", "PRACTICE OF", "GENERAL OBSERVATIONS", "PART", "PROBLEM", etc.

**Action:** MERGE into the main article

| Outlier | Merge Into |
|---------|------------|
| THEORY OF AGRICULTURE | AGRICULTURE |
| PRACTICE OF NAVIGATION | NAVIGATION |
| GENERAL OBSERVATIONS ON THE SKELETON | ANATOMY |
| SEVENTH LAW | CHEMISTRY (check context) |

### Pattern 2: Geographic Subdivisions

**Clue:** Headword starts with NEW, CAPE, MOUNT, PORT, ISLE, ST

**Action:** Usually MERGE into parent geographic article

| Outlier | Merge Into |
|---------|------------|
| NEW ALBION | AMERICA |
| NEW ANDALUSIA | AMERICA |
| NEW HAMPSHIRE | AMERICA |
| CAPE-AUGUSTIN | Check context - possibly BRAZIL or AFRICA |
| ISLE OF WIGHT | Check - might be valid standalone |

### Pattern 3: Person Names as Section Headers

**Clue:** Starts with first name (ADAM, WILLIAM, THOMAS) or title (MASTER OF, ARCHBISHOP)

**Action:** MERGE into parent biographical/philosophical article

| Outlier | Merge Into |
|---------|------------|
| ADAM SMITH | POLITICAL ECONOMY or BIOGRAPHY |
| WILLIAM PALEY | MORAL PHILOSOPHY or BIOGRAPHY |
| MASTER OF TRINITY COLLEGE | BIOGRAPHY (check context) |

### Pattern 4: Sentence Fragments

**Clue:** Starts with THIS, THESE, WHEN, WHERE, HAVING, or lowercase letter

**Action:** MERGE into the immediately preceding article

| Outlier | Merge Into |
|---------|------------|
| THIS IS THE BEST | Previous article by page |
| HAVING FOUND | Previous article by page |
| WHERE RECOURSE CAN BE HAD | Previous article by page |

### Pattern 5: Latin/Technical Terms in Treatises

**Clue:** Latin anatomical, botanical, or legal terms appearing mid-volume

**Action:** MERGE into the relevant treatise

| Outlier | Likely Merge Into |
|---------|-------------------|
| VENTRICULUS | ANATOMY |
| PULMONES | ANATOMY |
| MONANDRIA | BOTANY |
| FREE BENCH | LAW or TENURE |

### Pattern 6: OCR Errors

**Clue:** Headword is close to a valid word with letter substitution

**Action:** RENAME to correct spelling

| Outlier | Rename To |
|---------|-----------|
| RUNTISLAND | BURNTISLAND |
| IALOGISM | DIALOGISM |
| POLCMOTE | FOLKMOTE |
| UNCHEL-WEIGHT | AUNCEL-WEIGHT |

---

## After All Reviews Complete

### 1. Verify Decisions

```bash
python3 scripts/review_outliers.py --status
```

Ensure all 558 outliers have decisions.

### 2. Apply Fixes

```bash
python3 scripts/apply_outlier_fixes.py --preview  # Preview changes
python3 scripts/apply_outlier_fixes.py --apply    # Apply changes
```

### 3. Regenerate Website

```bash
python3 generate_site_optimized.py
```

### 4. Commit Changes

```bash
git add output_v2/ docs/ llm_corrections/
git commit -m "Apply alphabetic outlier corrections (558 fixes)"
```

---

## File Locations

| File | Purpose |
|------|---------|
| `scripts/detect_alphabetic_outliers.py` | Detects outliers, saves to JSON |
| `scripts/generate_outlier_batches.py` | Creates review batches |
| `scripts/review_outliers.py` | Interactive review tool |
| `scripts/apply_outlier_fixes.py` | Applies decisions to articles |
| `llm_corrections/outliers/alphabetic_outliers.json` | Raw detection results |
| `llm_corrections/outlier_batches/*.json` | Review batches with context |
| `llm_corrections/outlier_decisions.json` | Your decisions (auto-saved) |

---

## Tips for Efficient Review

1. **Start with 1810** - smallest edition (24 outliers, 1 batch)
2. **Use context** - the prev/next articles often reveal the merge target
3. **Trust the page numbers** - if outlier is on page 323 and AGRICULTURE spans 316-566, it's likely part of AGRICULTURE
4. **When in doubt, use OCR_REVIEW** - better to flag than guess wrong
5. **Take breaks** - progress saves automatically, resume anytime

---

## Resuming Work

To pick up where you left off:

```bash
cd /home/jic823/1815EncyclopediaBritannicaNLS
python3 scripts/review_outliers.py --status   # See progress
python3 scripts/review_outliers.py --edition 1815  # Continue edition
```

The tool automatically skips already-reviewed items.
