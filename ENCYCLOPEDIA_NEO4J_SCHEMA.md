# Encyclopedia Evolution Knowledge Graph - Neo4j Schema

A knowledge graph tracking the evolution of encyclopedic knowledge from 1704-1911, designed to integrate with the existing Banks Archive GraphRAG database.

**Created:** December 21, 2025

---

## Integration Strategy

### Separation Approach

The Encyclopedia KG uses the same database as Banks but maintains separation through:

1. **Distinct Node Labels**: Encyclopedia-specific labels prefixed with `Enc_` (e.g., `Enc_Article`, `Enc_Edition`)
2. **Source Property**: All nodes have `source` property (e.g., `"britannica_5th_1815"`)
3. **Shared Entities**: Person, Place, Institution nodes are shared across projects via Wikidata QIDs

### Cross-Project Linking

Shared entities enable research questions like:
- "Which Britannica articles mention people Banks corresponded with?"
- "How did encyclopedia coverage of commodities change during the leather trade crisis?"
- "Which places mentioned in Banks' letters have encyclopedia entries?"

```cypher
// Example: Find encyclopedia articles about Banks' correspondents
MATCH (p:Person)-[:AUTHORED|RECEIVED]->(d:Document {source: "dawson_1958"})
MATCH (a:Enc_Article)-[:MENTIONS]->(p)
RETURN p.name, a.headword, a.edition_year
```

---

## Database Connection

Same database as Banks Archive:

```
URI:      neo4j://127.0.0.1:7687
User:     neo4j
Password: york2005
```

---

## Source Codes

| Source Code | Work | Years | Type |
|-------------|------|-------|------|
| `lexicon_technicum_1704` | Lexicon Technicum Vol 1 | 1704 | lexicon |
| `lexicon_technicum_1710` | Lexicon Technicum Vol 2 | 1710 | lexicon |
| `chambers_1728` | Chambers Cyclopaedia | 1728 | cyclopaedia |
| `coetlogon_1745` | Coetlogon Universal History | 1745 | universal |
| `britannica_1st_1771` | Encyclopedia Britannica 1st | 1771 | britannica |
| `britannica_2nd_1778` | Encyclopedia Britannica 2nd | 1778 | britannica |
| `britannica_3rd_1797` | Encyclopedia Britannica 3rd | 1797 | britannica |
| `britannica_4th_1810` | Encyclopedia Britannica 4th | 1810 | britannica |
| `britannica_5th_1815` | Encyclopedia Britannica 5th | 1815 | britannica |
| `britannica_6th_1823` | Encyclopedia Britannica 6th | 1823 | britannica |
| `britannica_7th_1842` | Encyclopedia Britannica 7th | 1842 | britannica |
| `britannica_8th_1860` | Encyclopedia Britannica 8th | 1860 | britannica |
| `britannica_9th_1889` | Encyclopedia Britannica 9th | 1889 | britannica |
| `britannica_11th_1911` | Encyclopedia Britannica 11th | 1911 | britannica |

---

## Schema

### Encyclopedia-Specific Nodes

```cypher
// Edition - represents a complete encyclopedia edition
(:Enc_Edition {
    id,                    // e.g., "britannica_5th_1815"
    name,                  // "Encyclopedia Britannica 5th Edition"
    year_start,            // 1815
    year_end,              // 1817
    volumes,               // 20
    work_type,             // "britannica", "lexicon", "cyclopaedia"
    publisher,
    place_of_publication
})

// Article - individual encyclopedia entry
(:Enc_Article {
    id,                    // Unique: "{source}_{headword}_{sense}"
    headword,              // "ASTRONOMY"
    headword_normalized,   // Lowercase, normalized spelling
    sense,                 // 1, 2, 3... for multiple entries with same headword
    source,                // "britannica_5th_1815"
    edition_year,          // 1815
    volume,                // 2
    start_page,            // 234
    end_page,              // 289
    word_count,            // 21543
    is_cross_reference,    // false
    text,                  // Full article text
    text_preview           // First 500 chars for display
})

// Topic - normalized topic for linking across editions
(:Enc_Topic {
    id,                    // "astronomy"
    canonical_name,        // "Astronomy"
    aliases                // ["ASTRONOMY", "Astron.", "Astronomical Science"]
})

// TextChunk - for vector embeddings (same pattern as Banks)
(:Enc_TextChunk {
    id,                    // Unique chunk ID
    text,                  // Chunk text
    embedding,             // 3072-dim vector (Gemini)
    source,                // "britannica_5th_1815"
    article_id,            // Parent article ID
    chunk_index            // Position in article
})
```

### Shared Nodes (Used by Both Banks and Encyclopedia)

These nodes are shared across projects. When loading encyclopedia data, check for existing entities by Wikidata QID or name before creating new ones.

```cypher
// Person - shared with Banks
(:Person {
    name,                  // Canonical name
    wikidata_qid,          // "Q935" (Newton)
    source,                // First source that created this node
    birth_year,
    death_year,
    occupation
})

// Place - shared with Banks
(:Place {
    name,                  // "Edinburgh"
    wikidata_qid,          // "Q23436"
    source,
    type,                  // "city", "country", "region"
    latitude,
    longitude
})

// Institution - shared with Banks
(:Institution {
    name,                  // "Royal Society"
    wikidata_qid,          // "Q123885"
    source,
    type                   // "scientific", "government", "educational"
})

// Taxon - shared with Banks (for natural history articles)
(:Taxon {
    scientific_name,
    common_name,
    source,
    wikidata_qid
})
```

---

## Relationships

### Edition Structure

```cypher
// Article belongs to edition
(a:Enc_Article)-[:IN_EDITION]->(e:Enc_Edition)

// Article is about a topic
(a:Enc_Article)-[:ABOUT]->(t:Enc_Topic)

// Article has text chunks for embeddings
(a:Enc_Article)-[:HAS_CHUNK]->(c:Enc_TextChunk)
```

### Cross-Edition Evolution

```cypher
// Direct successor (same headword, next edition)
(a1:Enc_Article)-[:EVOLVED_TO {
    headword_match: true,
    text_similarity: 0.87,    // Cosine similarity
    word_count_change: +3500  // Growth in words
}]->(a2:Enc_Article)

// Same topic, different headword (e.g., AETHER -> ETHER)
(a1:Enc_Article)-[:SAME_TOPIC_AS {
    spelling_variant: true,
    similarity: 0.92
}]->(a2:Enc_Article)
```

### Internal References

```cypher
// Cross-references within an edition
(a1:Enc_Article)-[:CROSS_REFERENCES {
    type: "see",           // "see" or "see_also"
    context: "for more details"
}]->(a2:Enc_Article)
```

### Named Entity Mentions

```cypher
// Article mentions a person
(a:Enc_Article)-[:MENTIONS {
    count: 5,
    contexts: ["discovered by", "according to"]
}]->(p:Person)

// Article mentions a place
(a:Enc_Article)-[:MENTIONS]->(pl:Place)

// Article mentions an institution
(a:Enc_Article)-[:MENTIONS]->(i:Institution)

// Article discusses a taxon (for natural history)
(a:Enc_Article)-[:DISCUSSES]->(t:Taxon)
```

---

## Indexes

```cypher
// Unique constraints
CREATE CONSTRAINT enc_article_id IF NOT EXISTS FOR (a:Enc_Article) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT enc_edition_id IF NOT EXISTS FOR (e:Enc_Edition) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT enc_topic_id IF NOT EXISTS FOR (t:Enc_Topic) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT enc_chunk_id IF NOT EXISTS FOR (c:Enc_TextChunk) REQUIRE c.id IS UNIQUE;

// Performance indexes
CREATE INDEX enc_article_headword IF NOT EXISTS FOR (a:Enc_Article) ON (a.headword);
CREATE INDEX enc_article_source IF NOT EXISTS FOR (a:Enc_Article) ON (a.source);
CREATE INDEX enc_article_year IF NOT EXISTS FOR (a:Enc_Article) ON (a.edition_year);
CREATE INDEX enc_article_headword_norm IF NOT EXISTS FOR (a:Enc_Article) ON (a.headword_normalized);

// Composite for lookups
CREATE INDEX enc_article_lookup IF NOT EXISTS FOR (a:Enc_Article) ON (a.headword, a.edition_year);

// Fulltext search
CREATE FULLTEXT INDEX enc_article_text IF NOT EXISTS FOR (a:Enc_Article) ON EACH [a.text];
CREATE FULLTEXT INDEX enc_article_headword_fulltext IF NOT EXISTS FOR (a:Enc_Article) ON EACH [a.headword];

// Vector index for semantic search
CREATE VECTOR INDEX enc_chunk_embedding IF NOT EXISTS
FOR (c:Enc_TextChunk) ON (c.embedding)
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 3072,
        `vector.similarity_function`: 'cosine'
    }
}
```

---

## Query Examples

### Within Encyclopedia Project

```cypher
// Track article evolution across editions
MATCH path = (a1:Enc_Article {headword: 'ELECTRICITY'})-[:EVOLVED_TO*]->(a2:Enc_Article)
WHERE a1.edition_year = 1771
RETURN [n in nodes(path) | {year: n.edition_year, words: n.word_count}] as evolution

// Find new topics in an edition
MATCH (a:Enc_Article {edition_year: 1889})
WHERE NOT EXISTS {
    MATCH (a)<-[:EVOLVED_TO]-(prev:Enc_Article)
}
RETURN a.headword, a.word_count
ORDER BY a.word_count DESC
LIMIT 50

// Most referenced articles
MATCH (a:Enc_Article {edition_year: 1815})<-[:CROSS_REFERENCES]-(other)
RETURN a.headword, count(other) as references
ORDER BY references DESC
LIMIT 20

// Articles by volume
MATCH (a:Enc_Article {source: 'britannica_5th_1815'})
RETURN a.volume, count(a) as articles
ORDER BY a.volume
```

### Cross-Project Queries (Banks + Encyclopedia)

```cypher
// People mentioned in both Banks letters and encyclopedia
MATCH (p:Person)-[:MENTIONED_IN|AUTHORED|RECEIVED]->(d:Document)
WHERE d.source IN ['sutro_t1', 'dawson_1958']
MATCH (a:Enc_Article)-[:MENTIONS]->(p)
RETURN p.name,
       count(DISTINCT d) as banks_documents,
       count(DISTINCT a) as encyclopedia_articles
ORDER BY banks_documents DESC
LIMIT 20

// Commodities in Banks trade data with encyclopedia articles
MATCH (c:Commodity)<-[:DISCUSSES]-(d:Document {source: 'sutro_t1'})
MATCH (a:Enc_Article)-[:MENTIONS|DISCUSSES]->(c)
RETURN c.name,
       count(DISTINCT d) as trade_documents,
       collect(DISTINCT a.headword)[0..5] as encyclopedia_headwords

// Timeline: Encyclopedia coverage vs Banks correspondence
MATCH (a:Enc_Article)-[:MENTIONS]->(p:Person {name: 'Sir Joseph Banks'})
RETURN a.edition_year as year, count(a) as articles_mentioning_banks
ORDER BY year

// Places in both datasets
MATCH (pl:Place)<-[:REFERENCES_PLACE]-(d:Document)
MATCH (a:Enc_Article)-[:MENTIONS]->(pl)
RETURN pl.name, pl.wikidata_qid,
       count(DISTINCT d) as in_banks_letters,
       count(DISTINCT a) as in_encyclopedia
ORDER BY in_banks_letters + in_encyclopedia DESC
LIMIT 30
```

### Semantic Search

```cypher
// Find encyclopedia articles semantically similar to a query
CALL db.index.vector.queryNodes('enc_chunk_embedding', 10, $query_embedding)
YIELD node, score
MATCH (a:Enc_Article)-[:HAS_CHUNK]->(node)
RETURN DISTINCT a.headword, a.edition_year, a.text_preview, max(score) as relevance
ORDER BY relevance DESC

// Combined semantic search across Banks and Encyclopedia
CALL db.index.vector.queryNodes('chunk_embedding', 5, $query_embedding)
YIELD node as banks_chunk, score as banks_score
MATCH (d:Document)-[:HAS_CHUNK]->(banks_chunk)

CALL db.index.vector.queryNodes('enc_chunk_embedding', 5, $query_embedding)
YIELD node as enc_chunk, score as enc_score
MATCH (a:Enc_Article)-[:HAS_CHUNK]->(enc_chunk)

RETURN 'banks' as source, d.summary as content, banks_score as score
UNION
RETURN 'encyclopedia' as source, a.headword + ': ' + a.text_preview as content, enc_score as score
ORDER BY score DESC
LIMIT 10
```

---

## Data Loading Pipeline

### 1. Load Editions

```python
def load_edition(tx, edition_data):
    tx.run("""
        MERGE (e:Enc_Edition {id: $id})
        SET e.name = $name,
            e.year_start = $year_start,
            e.year_end = $year_end,
            e.volumes = $volumes,
            e.work_type = $work_type
    """, **edition_data)
```

### 2. Load Articles

```python
def load_article(tx, article):
    article_id = f"{article['source']}_{article['headword']}_{article['sense']}"
    tx.run("""
        MERGE (a:Enc_Article {id: $id})
        SET a.headword = $headword,
            a.headword_normalized = toLower($headword),
            a.sense = $sense,
            a.source = $source,
            a.edition_year = $edition_year,
            a.volume = $volume,
            a.start_page = $start_page,
            a.end_page = $end_page,
            a.word_count = size(split($text, ' ')),
            a.is_cross_reference = $is_cross_reference,
            a.text = $text,
            a.text_preview = left($text, 500)

        WITH a
        MATCH (e:Enc_Edition {id: $source})
        MERGE (a)-[:IN_EDITION]->(e)
    """, id=article_id, **article)
```

### 3. Link to Shared Entities

```python
def link_person(tx, article_id, person_name, wikidata_qid=None):
    tx.run("""
        MATCH (a:Enc_Article {id: $article_id})
        MERGE (p:Person {name: $person_name})
        ON CREATE SET p.source = 'encyclopedia',
                      p.wikidata_qid = $wikidata_qid
        MERGE (a)-[:MENTIONS]->(p)
    """, article_id=article_id, person_name=person_name, wikidata_qid=wikidata_qid)
```

### 4. Create Cross-Edition Links

```python
def link_article_evolution(tx, source_edition, target_edition):
    tx.run("""
        MATCH (a1:Enc_Article {source: $source})
        MATCH (a2:Enc_Article {source: $target})
        WHERE a1.headword = a2.headword
        MERGE (a1)-[:EVOLVED_TO {headword_match: true}]->(a2)
    """, source=source_edition, target=target_edition)
```

---

## Project Isolation Queries

### Encyclopedia Only

```cypher
// All encyclopedia data
MATCH (n)
WHERE n:Enc_Article OR n:Enc_Edition OR n:Enc_Topic OR n:Enc_TextChunk
RETURN labels(n)[0] as type, count(n) as count

// Or filter by source pattern
MATCH (n)
WHERE n.source STARTS WITH 'britannica' OR n.source STARTS WITH 'lexicon'
   OR n.source STARTS WITH 'chambers' OR n.source STARTS WITH 'coetlogon'
RETURN n.source, count(n) as nodes
```

### Banks Only

```cypher
// All Banks data
MATCH (n)
WHERE n.source IN ['sutro_t1', 'dawson_1958']
RETURN n.source, count(n) as nodes
```

---

## File Locations

| File | Path | Purpose |
|------|------|---------|
| Schema | `/home/jic823/1815EncyclopediaBritannicaNLS/ENCYCLOPEDIA_NEO4J_SCHEMA.md` | This document |
| Parser | `/home/jic823/1815EncyclopediaBritannicaNLS/parse_britannica_articles.py` | Extract articles from OCR |
| KG Schema | `/home/jic823/1815EncyclopediaBritannicaNLS/britannica_kg_schema.md` | Original planning doc |
| Articles | `/home/jic823/1815EncyclopediaBritannicaNLS/articles_1815_clean.jsonl` | Parsed 1815 articles |
| Banks Schema | `/home/jic823/BanksArchivedizws/pdfs/BANKS_GRAPHRAG_DATABASE.md` | Banks database docs |

---

## Next Steps

1. **Create loader script**: `load_encyclopedia_neo4j.py`
2. **Extract named entities**: Use NER to find Person/Place/Institution mentions
3. **Generate embeddings**: Chunk articles and embed with Gemini
4. **Link editions**: Create EVOLVED_TO relationships
5. **Ground to Wikidata**: Link entities to Wikidata QIDs for cross-project joins

---

*Schema designed: December 21, 2025*
