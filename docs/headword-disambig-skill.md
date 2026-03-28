---
name: headword-disambig
description: Disambiguate Encyclopedia Britannica article headwords to Wikidata QIDs using MCP search. Use when grounding encyclopedia topics, places, and concepts to the modern knowledge graph.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, mcp__wikidata__search_items, mcp__wikidata__get_statements
argument-hint: [count-to-add | "status" | "audit"]
---

# Headword Disambiguation Workflow

You are grounding Encyclopedia Britannica headwords (article titles from 1771-1860, 8 editions) to Wikidata QIDs for GraphRAG.

## Key Files

| File | Description |
|------|-------------|
| `data/headword_matches.jsonl` | **Output**: headword_id -> QID matches |
| `data/cross_edition_index.jsonl` | 4,379 substantive headwords with per-edition word counts |
| `data/ner/person_matches.jsonl` | 1,458 person matches (for biographical headword overlap) |

## Match File Format

```json
{"headword_id": "CHEMISTRY", "canonical_title": "CHEMISTRY", "wikidata_qid": "Q2329", "wikidata_label": "chemistry", "wikidata_desc": "branch of physical science", "match_type": "mcp_verified", "total_word_count": 1843073, "edition_count": 8}
```

## Commands

If `$ARGUMENTS` is "status":
- Count matched headwords and show tier distribution
- Report remaining by word count tier

If `$ARGUMENTS` is "audit":
- Spot-check 20 random matches via MCP
- Check for duplicate headword_ids

Otherwise, treat `$ARGUMENTS` as target number of new matches to add (default: 50).

## Matching Procedure

### 1. Get unmatched headwords sorted by total word count

```python
python3 -c "
import json
matched = set()
try:
    with open('data/headword_matches.jsonl') as f:
        for line in f:
            matched.add(json.loads(line)['headword_id'])
except FileNotFoundError:
    pass
queue = []
with open('data/cross_edition_index.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj['id'] not in matched:
            twc = sum(e.get('word_count',0) for e in obj.get('editions',{}).values())
            queue.append((obj['id'], obj['canonical_title'], twc, obj.get('edition_count',0)))
queue.sort(key=lambda x: x[2], reverse=True)
for i, (hid, title, twc, ec) in enumerate(queue[:80]):
    print(f'{i+1:3d}. {title:<45s} {twc:>10,d} words  {ec} eds')
"
```

### 2. Search Wikidata via MCP

Use `mcp__wikidata__search_items` EXCLUSIVELY. **Never invent QIDs.**

Tips:
- Most headwords are unambiguous — CHEMISTRY, FRANCE, ANATOMY have one obvious Wikidata match
- Search the headword directly: "Chemistry", "France", "Anatomy"
- For multi-word: "Moral philosophy", "Strength of materials", "Steam engine"
- For historical terms: search the modern equivalent (FARRIERY → "farriery", PNEUMATICS → "pneumatics")
- Send 8-10 parallel queries per batch

### 3. Skip rules

- **DISS** and other known parsing artifacts (mega-articles from OCR errors)
- Headwords that are clearly sub-sections, not standalone concepts (e.g., "GENUS IX")
- Extremely generic terms where Wikidata has no clear single match

### 4. Write matches

```python
python3 -c "
import json
matches = [
    {'headword_id': 'CHEMISTRY', 'canonical_title': 'CHEMISTRY', 'wikidata_qid': 'Q2329', 'wikidata_label': 'chemistry', 'wikidata_desc': 'branch of physical science', 'match_type': 'mcp_verified', 'total_word_count': 1843073, 'edition_count': 8},
]
existing = set()
try:
    with open('data/headword_matches.jsonl') as f:
        for line in f:
            existing.add(json.loads(line)['headword_id'])
except FileNotFoundError:
    pass
added = 0
with open('data/headword_matches.jsonl', 'a') as f:
    for m in matches:
        if m['headword_id'] not in existing:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
            existing.add(m['headword_id'])
            added += 1
print(f'Added {added}, total: {len(existing)}')
"
```

### 5. Cross-reference existing data

Before MCP searching, check if a headword already has a match:
- **Biographical headwords**: check `data/ner/person_matches.jsonl` for QIDs
- **Geographic headwords**: many are country/city names already in toponym data

### 6. Commit periodically

```bash
git add data/headword_matches.jsonl
git commit -m "Add N headword-to-Wikidata matches (total: NNNN)"
git push
```

## Working Order (by total word count)

| Tier | Word Count | Count | Priority |
|------|-----------|-------|----------|
| 1 | >= 100K | 169 | Highest — major treatises |
| 2 | 50K-100K | 182 | High — significant articles |
| 3 | 20K-50K | 517 | Medium — standard articles |
| 4 | 10K-20K | 769 | Lower — shorter entries |
| 5 | 5K-10K | 1,196 | Low — brief entries |
| 6 | 1K-5K | 1,546 | Lowest — stubs and cross-refs |

## Quality Standards

- Every QID from `mcp__wikidata__search_items` — never invented
- Match the concept, not a namesake (LONDON → Q84 London, not Q24639 London Ontario)
- Historical terms should map to the concept, not the modern discipline if different
- For places: prefer the historical entity if it exists (e.g., PRUSSIA → Q38872 Kingdom of Prussia)
