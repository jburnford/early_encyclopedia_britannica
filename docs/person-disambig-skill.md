---
name: person-disambig
description: Disambiguate person entities from Encyclopedia Britannica NER output to Wikidata QIDs using MCP search. Use when continuing person disambiguation work or when the user says /person-disambig.
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, mcp__wikidata__search_items, mcp__wikidata__get_statements
argument-hint: [count-to-add | "status" | "audit"]
---

# Person Disambiguation Workflow

You are disambiguating person entity clusters from the Encyclopedia Britannica (1771-1860, 8 editions) NER output to Wikidata QIDs.

## Key Files

| File | Description |
|------|-------------|
| `data/ner/person_matches.jsonl` | **Output**: verified cluster_id -> QID matches |
| `data/ner/person_clusters.jsonl` | 74,577 person clusters from NER |
| `data/ner/person_lookup_queue.jsonl` | 10,920 clusters with >= 5 mentions, sorted by frequency |
| `data/ner/person_candidates.jsonl` | Wikidata REST API results (older, partially populated) |
| `graphrag/judge_person_batch.py` | Status script (`--status`, `--show-batch`) |
| `graphrag/disambiguate_persons.py` | Pipeline script (`--finalize` to merge) |

## Match File Format

Each line in `person_matches.jsonl` is JSON:
```json
{"cluster_id": "henry viii", "wikidata_qid": "Q38370", "wikidata_label": "Henry VIII of England", "wikidata_desc": "King of England (1491-1547)", "match_type": "mcp_verified"}
```

## Commands

If `$ARGUMENTS` is "status":
- Run `python3 graphrag/judge_person_batch.py --status`
- Count lines in person_matches.jsonl
- Report current count and remaining

If `$ARGUMENTS` is "audit":
- Pick 20 random matches from person_matches.jsonl
- Verify each QID via `mcp__wikidata__search_items`
- Report any mismatches
- Check for orphan cluster_ids not in person_clusters.jsonl

Otherwise, treat `$ARGUMENTS` as a target number of new matches to add (default: 100).

## Matching Procedure

### 1. Get unmatched clusters sorted by mentions

```python
python3 -c "
import json
matched = set()
with open('data/ner/person_matches.jsonl') as f:
    for line in f:
        matched.add(json.loads(line)['cluster_id'])
queue = []
with open('data/ner/person_lookup_queue.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj['cluster_id'] not in matched:
            queue.append(obj)
queue.sort(key=lambda x: x['total_mentions'], reverse=True)
for i, q in enumerate(queue[:100]):
    print(f'{q[\"cluster_id\"]:45s} m={q[\"total_mentions\"]:4d}')
"
```

### 2. Search Wikidata via MCP

Use `mcp__wikidata__search_items` EXCLUSIVELY for QID lookup. **Never invent or guess QIDs.**

MCP search tips:
- **Short, exact name queries work best**: "John Locke" not "John Locke English philosopher empiricism"
- **The server is intermittent**: if a query returns empty, try shorter phrasing or retry later
- **Full proper names first**, then add one disambiguator if needed: "Pierre Bouguer" -> "Pierre Bouguer astronomer"
- Send 8-10 parallel queries per batch for efficiency, but space batches if the server starts returning empty
- If the server goes cold (many consecutive empty results), pause and retry after a minute

### 3. Skip rules — do NOT match these

- **Title + single common surname**: "Dr Smith", "Mr Harris", "Sir John", "Lord North" (unless context is unambiguous like "Lord North" = PM Frederick North)
- **Bare single-word names**: "Elizabeth", "Alexander", "John", "Augustus"
- **Ordinal-only names**: "John III", "Philip I", "Peter II" (unless context is clear)
- **Ambiguous title clusters**: "of York", "of Orange", "of Buckingham", "Prince Henry", "King Edward"
- **Multi-referent clusters**: "Charles of Lorraine" (multiple dukes), "Earl Grey" (multiple earls)

However, DO match title+name clusters where the encyclopedia context makes identification clear:
- "King Alfred" -> Alfred the Great (only one famous King Alfred)
- "Bishop Warburton" -> William Warburton (the famous Bishop of Gloucester)
- "Dr Hutton" -> James Hutton (the geologist, given encyclopedia context)

### 4. Write matches

Append to person_matches.jsonl using Python:
```python
python3 -c "
import json
matches = [
    {'cluster_id': 'CLUSTER_ID', 'wikidata_qid': 'QNNN', 'wikidata_label': 'LABEL', 'wikidata_desc': 'DESC', 'match_type': 'mcp_verified'},
    # ... more matches
]
clusters = set()
with open('data/ner/person_clusters.jsonl') as f:
    for line in f:
        clusters.add(json.loads(line)['cluster_id'])
matched = set()
with open('data/ner/person_matches.jsonl') as f:
    for line in f:
        matched.add(json.loads(line)['cluster_id'])
added = 0
with open('data/ner/person_matches.jsonl', 'a') as f:
    for m in matches:
        cid = m['cluster_id']
        if cid in clusters and cid not in matched:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
            matched.add(cid)
            added += 1
        elif cid not in clusters:
            print(f'  WARNING: {cid} not in clusters')
print(f'Added {added}, total: {len(matched)}')
"
```

### 5. Validate after each batch

- Verify new cluster_ids exist in person_clusters.jsonl
- Deduplicate if needed:
```python
python3 -c "
import json
seen = {}; lines = []
with open('data/ner/person_matches.jsonl') as f:
    for line in f:
        cid = json.loads(line)['cluster_id']
        if cid not in seen:
            seen[cid] = True; lines.append(line.strip())
with open('data/ner/person_matches.jsonl', 'w') as f:
    for l in lines: f.write(l + '\n')
print(f'Deduped: {len(lines)} unique matches')
"
```

### 6. Variant awareness

Many people appear under multiple cluster_ids. When you match "Dr Hutton" to James Hutton, also check for:
- Full name variants: "james hutton"
- Alternative title variants: "mr hutton", "professor hutton"
- Use: `grep -i 'hutton' data/ner/person_clusters.jsonl | head` to find all variants

Common cluster_id patterns in this dataset:
- `dr X`, `mr X`, `mrs X`, `sir X`, `lord X` — title + surname
- `professor X`, `bishop X`, `archbishop X`, `cardinal X` — role + surname
- `m. X` — French "Monsieur" prefix (very common for French scientists/philosophers)
- `f. X` — "Father" prefix (for clergy)
- Full names: `john locke`, `isaac newton`
- Honorific full: `sir isaac newton`, `sir humphry davy`

### 7. Commit periodically

After adding 50+ matches, commit and push:
```bash
git add data/ner/person_matches.jsonl
git commit -m "Add N more person disambiguation matches (total: NNNN)"
git push
```

## Quality Standards

- Every QID MUST come from `mcp__wikidata__search_items` results — never invented
- Match confidence should be high: the Wikidata description should clearly match the encyclopedia context
- When in doubt, skip — a false match is worse than a missing one
- The encyclopedia covers 1771-1860, so people referenced are typically pre-1860 historical figures
