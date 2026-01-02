# voyage-context-3 Evaluation Report
## Embedding Model Selection for Historical Encyclopedia GraphRAG

**Date**: December 2025
**Project**: 1815 Encyclopaedia Britannica GraphRAG System
**Objective**: Evaluate whether voyage-context-3's contextual embedding feature provides value over standard voyage-3 for historical encyclopedia content.

---

## Executive Summary

We conducted a comprehensive evaluation of Voyage AI's embedding models for a GraphRAG system processing historical encyclopedias (1700-1911). Our hypothesis was that **voyage-context-3**, which embeds chunks with awareness of surrounding document context, would outperform standard **voyage-3** on long encyclopedia articles where disambiguation is important.

**Key Finding**: voyage-3 already achieves excellent performance (MRR 0.894-1.000) on encyclopedia content. The contextual feature of voyage-context-3 provides **minimal improvement** (+1.7% MRR at best) because encyclopedia chunks are topically self-contained and don't require cross-chunk context for disambiguation.

**Recommendation**: Use **voyage-3** for the Encyclopedia GraphRAG project. The simpler API, faster processing (4x), and lower complexity outweigh the marginal benefits of voyage-context-3.

---

## 1. Research Background

### 1.1 Project Context

The larger project aims to build a GraphRAG system tracking the evolution of knowledge across encyclopedias from 1700-1911, with particular attention to:
- Named Entity Recognition (NER) for people, places, concepts
- Semantic drift (words whose meanings have changed over centuries)
- Cross-article relationships and knowledge evolution

### 1.2 Data Characteristics

| Dataset | Articles | Total Words | Avg Words/Article |
|---------|----------|-------------|-------------------|
| 1815 Britannica (clean) | 18,172 | ~15M | ~825 |
| 1771 Britannica | 13,762 | ~10M | ~727 |

**Article length distribution** (1815):
- < 1,000 words: 16,500 articles (91%)
- 1,000-10,000 words: 1,469 articles (8%)
- 10,000-25,000 words: 203 articles (1%) *optimal for context testing*
- > 25,000 words: 251 articles (parsing errors - merged content)

### 1.3 Why voyage-context-3?

Voyage AI's contextual embedding model was designed for exactly this use case:
- **Long documents**: Articles up to 24,000 words (31K tokens)
- **Semantic disambiguation**: Historical terminology differs from modern usage
- **Cross-chunk awareness**: Understanding where a chunk fits in the larger article

---

## 2. Technical Specifications

### 2.1 voyage-context-3 API

```python
# Contextual embedding (chunks share context automatically)
result = client.contextualized_embed(
    inputs=[[chunk1, chunk2, chunk3, ...]],  # Nested list per document
    model="voyage-context-3",
    input_type="document",
    output_dimension=1024
)

# Query embedding (also uses contextualized_embed)
query_result = client.contextualized_embed(
    inputs=[[query]],
    model="voyage-context-3",
    input_type="query",
    output_dimension=1024
)
```

**Key insight**: No manual metadata prepending needed. Context is inferred from sibling chunks in the same inner list.

### 2.2 Token Limits

| Limit | Value | Implication |
|-------|-------|-------------|
| Per request | 120K tokens | Can embed ~3-4 long articles per call |
| Max chunks | 16K per request | Not a practical limit |
| Per document context | ~32K tokens | Articles up to ~24,000 words |
| Per chunk | 16K tokens | Not a practical limit |

**Optimal article size**: 10,000-24,000 words (~13K-32K tokens)

### 2.3 Performance Claims

Voyage AI reports voyage-context-3 outperforms:
- Standard voyage-3 by **variable margin** (depends on content type)
- Anthropic's contextual retrieval by **6.76%** on benchmarks

---

## 3. Test Methodology

### 3.1 Test 1: Leather/Tanning Domain

**Rationale**: Domain-specific test with related articles that share terminology.

**Articles** (19 target + 50 background):
- MOROCCO (13,868 words), HIDE (4,689), FULLER (4,441), TANNING (4,256)
- LEATHER (3,449), OAK (2,823), BARK (2,435), ALUM (1,709), PARCHMENT (556)

**Queries** (11 domain-specific):
```
- "vegetable tanning oak bark process pit"
- "hide preparation lime soaking unhairing"
- "currying finishing leather oil grease"
- "morocco red leather Turkey dyeing goatskin"
- "tanning liquor ooze tan yard handler"
```

### 3.2 Test 2: Long Articles with Ambiguous Queries

**Rationale**: Test articles near the 32K token limit with cross-article ambiguity.

**Articles** (3 long articles, 313 chunks):
| Article | Words | Tokens (est.) | Chunks |
|---------|-------|---------------|--------|
| GLASS | 24,275 | 31,557 | 149 |
| PRINTING | 21,200 | 27,560 | 120 |
| EXPERIMENTAL PHILOSOPHY | 8,998 | 11,697 | 44 |

**Query types**:
1. **Cross-article ambiguous**: "the French improvements to this art and their methods"
2. **Article-specific**: "melting sand with alkaline salts in a furnace"
3. **Within-article section**: "the color of glass and how to produce different tints"

### 3.3 Test 3: Truly Ambiguous Queries

**Rationale**: Target chunks that don't explicitly mention their article's topic.

**Query design**:
```python
# Matches GLASS chunk about "manufactory" - never says "glass"
"manufactory established in Lancashire commerce proprietors"

# Matches PRINTING chunk about "the art" - never says "printing"
"Laurentius first devised rough specimen of the art wooden types"

# Generic language matching either article
"the difficulties attendant upon new undertakings foreign establishments"
```

---

## 4. Results

### 4.1 Test 1: Leather/Tanning Domain

| Model | MRR | Recall@5 | Recall@10 | Time |
|-------|-----|----------|-----------|------|
| voyage-3 | 0.894 | 0.644 | 0.735 | 4.6s |
| voyage-context-3 | 0.909 | 0.508 | 0.568 | 10.4s |

**Improvement**: +1.7% MRR

**Query-by-query analysis**:
- Context wins: 2 queries
- Standard wins: 1 query
- Ties: 8 queries

**Notable**: voyage-context-3 had *worse* Recall@5/10 despite slightly better MRR, suggesting it concentrates results more narrowly.

### 4.2 Test 2: Long Articles with Ambiguous Queries

| Model | MRR | Recall@5 | Time |
|-------|-----|----------|------|
| voyage-3 | 1.000 | 0.964 | 3.6s |
| voyage-context-3 | 1.000 | 0.893 | 15.2s |

**Improvement**: 0.0% MRR (perfect tie)

**By query type**:
| Type | Standard MRR | Context MRR | Winner |
|------|--------------|-------------|--------|
| cross_article_ambiguous | 1.000 | 1.000 | TIE |
| article_specific | 1.000 | 1.000 | TIE |
| within_article_section | 1.000 | 1.000 | TIE |

**Summary**: 0 context wins, 0 standard wins, 14 ties

### 4.3 Test 3: Truly Ambiguous Queries

| Model | MRR |
|-------|-----|
| voyage-3 | 1.000 |
| voyage-context-3 | 1.000 |

**Summary**: 0 context wins, 0 standard wins, 9 ties

Even on queries specifically designed to require document context for disambiguation, both models achieved identical perfect performance.

---

## 5. Analysis

### 5.1 Why Context Doesn't Help Encyclopedia Content

**Encyclopedia articles are topically self-contained**. Unlike:
- Legal documents (where clause 47 references clause 12)
- Technical manuals (where step 5 requires context from step 1)
- Novels (where pronouns reference earlier passages)

Encyclopedia chunks contain sufficient semantic signal to identify their topic:
- Technical vocabulary (vitrification, moveable type)
- Domain-specific processes (tanning, glass-blowing)
- Historical references specific to each topic

**Example**: A chunk about "the swelling of tubes towards fire" may not say "glass," but voyage-3 correctly links it to GLASS based on the technical vocabulary alone.

### 5.2 Processing Time Trade-off

| Model | Time | Relative |
|-------|------|----------|
| voyage-3 | 3.6-4.6s | 1x |
| voyage-context-3 | 10.4-15.2s | 3-4x |

For 18,172 articles, this translates to:
- voyage-3: ~1-2 hours
- voyage-context-3: ~4-8 hours

### 5.3 API Complexity

**voyage-3**:
```python
result = client.embed(texts, model="voyage-3", input_type="document")
```

**voyage-context-3**:
```python
result = client.contextualized_embed(
    inputs=[[chunk1, chunk2, ...]],  # Must group by document
    model="voyage-context-3",
    input_type="document"
)
# Requires flattening embeddings back to chunk order
```

---

## 6. Semantic Drift Considerations

### 6.1 Historical Terminology

Words whose meanings have changed significantly since 1815:

| Term | 1815 Meaning | 2025 Meaning |
|------|--------------|--------------|
| PHLOGISTON | Combustion theory | Obsolete |
| PHYSICS | Medicine/natural philosophy | Natural science |
| BROADCAST | Scattering seeds | Radio/TV transmission |
| ENTHUSIASM | Religious fanaticism (pejorative) | Excitement (positive) |
| ENGINE | Any machine/device | Specific motor types |
| ATOM | Philosophical indivisible unit | Subatomic particles |

### 6.2 Impact on Embedding Models

Both voyage-3 and voyage-context-3 are trained on modern corpora. Neither has special handling for historical semantic drift.

**However**: Within the encyclopedia's own vocabulary, semantic consistency is maintained. A query for "combustion theory" will find PHLOGISTON because the article's own language discusses "burning," "fire," and "inflammable" in ways the model can match.

**Recommendation**: For semantic drift cases, consider:
1. Query expansion with historical synonyms
2. Metadata augmentation with time period
3. Custom fine-tuning (expensive, requires labeled data)

---

## 7. Recommendations

### 7.1 Primary Recommendation: Use voyage-3

For the Encyclopedia GraphRAG project, **voyage-3 is sufficient**.

**Rationale**:
- Achieves excellent performance (MRR 0.894-1.000)
- 3-4x faster processing
- Simpler API (no document grouping required)
- No measurable improvement from contextual embedding

### 7.2 When voyage-context-3 Might Help

Consider voyage-context-3 for:
- Legal/regulatory documents with cross-references
- Technical manuals with procedural dependencies
- Narrative content with pronoun resolution needs
- Very short chunks (< 100 words) that lack context

### 7.3 Alternative Strategies for Encyclopedia Content

Instead of contextual embeddings, invest in:

1. **Hierarchical retrieval**: Section-level embeddings for coarse retrieval, chunk-level for precision
2. **Metadata enrichment**: Prepend `[HEADWORD] [EDITION_YEAR]` to chunks
3. **Query expansion**: Include related terms for historical concepts
4. **Cross-reference graph**: Build explicit relationships between articles

### 7.4 Cost Analysis

| Model | Rate (per 1M tokens) | 18K articles (~15M tokens) |
|-------|---------------------|---------------------------|
| voyage-3 | $0.06 | ~$0.90 |
| voyage-context-3 | $0.12 | ~$1.80 |

Both are affordable. Cost is not a differentiating factor.

---

## 8. Test Scripts Reference

All test scripts are in `/home/jic823/1815EncyclopediaBritannicaNLS/`:

| Script | Purpose |
|--------|---------|
| `test_voyage_context_v2.py` | Leather/tanning domain test |
| `test_context_long_articles.py` | Long articles with ambiguous queries |
| `test_truly_ambiguous.py` | Chunks that don't mention their topic |

**Results files**:
- `leather_test_v4.log` - Leather domain results
- `context_test.log` - Long article test results
- `context_test_results.json` - Detailed query analysis

---

## 9. Conclusion

voyage-context-3's contextual embedding feature is designed for documents where individual chunks lack sufficient context for semantic understanding. Encyclopedia articles, by their nature as self-contained knowledge summaries, do not benefit from this feature.

The standard voyage-3 model provides excellent retrieval performance on historical encyclopedia content while being faster, simpler to implement, and equally accurate for our use case.

**Final verdict**: Proceed with **voyage-3** for the Encyclopedia GraphRAG system.

---

## Appendix: Sample Query Results

### Leather Domain (Test 1)
```
Query: "chamois soft white leather oil tawed"
  Expected: ['CHAMOIS', 'LEATHER']
  Standard: rank=2, got ['TANNING', 'LEATHER', 'LEATHER']
  Context:  rank=1, got ['LEATHER', 'LEATHER', 'LEATHER']
  Result: CONTEXT WINS

Query: "tanning liquor ooze tan yard handler"
  Expected: ['TANNING', 'TAN', 'BARK']
  Standard: rank=1, got ['TANNING', 'TANNING', 'OAK']
  Context:  rank=2, got ['TANNER', 'TAN', 'TANNING']
  Result: STANDARD WINS
```

### Long Articles (Test 2)
```
Query: "the French improvements to this art and their methods"
  Expected: ['GLASS', 'PRINTING']
  Standard: rank=1, got ['GLASS', 'GLASS', 'GLASS']
  Context:  rank=1, got ['PRINTING', 'PRINTING', 'PRINTING']
  Result: TIE (both correct - different article preference)

Query: "melting sand with alkaline salts in a furnace produces trans..."
  Expected: ['GLASS']
  Standard: rank=1, got ['GLASS', 'GLASS', 'GLASS']
  Context:  rank=1, got ['GLASS', 'GLASS', 'GLASS']
  Result: TIE
```
