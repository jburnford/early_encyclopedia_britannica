# Encyclopedia Evolution Knowledge Graph Schema

## Project Goal
Track the evolution of knowledge across English encyclopedias from 1704 to 1911,
covering the "long eighteenth and nineteenth centuries" of knowledge production.

### Source Works
- **Lexicon Technicum** (Harris, 1704-1710): First English technical encyclopedia
- **Cyclopaedia** (Chambers, 1728): First general English encyclopedia
- **Encyclopedia Britannica** (1771-1911): 11 editions over 140 years

Research questions enabled:
- How did understanding of a topic change over time?
- When did new topics first appear?
- Which articles expanded, contracted, or disappeared?
- How did cross-references evolve?

## Works to Include

### Pre-Britannica Encyclopedias

| Work | Years | Volumes | Author | Notes |
|------|-------|---------|--------|-------|
| Lexicon Technicum | 1704-1710 | 2 | John Harris | Technical/scientific focus |
| Cyclopaedia | 1728 | 2 | Ephraim Chambers | First general encyclopedia |
| Universal History of Arts & Sciences | 1745 | 2 | Dennis de Coetlogon | Bridges Chambers to Britannica |

### Encyclopedia Britannica Editions

| Edition | Years | Volumes | Notes |
|---------|-------|---------|-------|
| 1st | 1768-1771 | 3 | First edition, smaller scope |
| 2nd | 1777-1784 | 10 | Expanded significantly |
| 3rd | 1788-1797 | 18 | Added supplements |
| 4th | 1801-1810 | 20 | Major revision |
| 5th | 1815-1817 | 20 | Reprint of 4th with updates |
| 6th | 1820-1823 | 20 | |
| 7th | 1830-1842 | 21 | |
| 8th | 1852-1860 | 21 | |
| 9th | 1875-1889 | 25 | "Scholar's Edition" |
| 10th | 1902-1903 | 9 suppl | Supplements to 9th |
| 11th | 1910-1911 | 29 | Most famous edition |

## Neo4j Schema

### Nodes

#### Edition
```cypher
CREATE (e:Edition {
  number: 5,
  year_start: 1815,
  year_end: 1817,
  volumes: 20,
  name: "Fifth Edition"
})
```

#### Article
```cypher
CREATE (a:Article {
  id: "1815_ASTRONOMY_1",
  headword: "ASTRONOMY",
  sense: 1,                    -- For multiple articles with same headword
  edition_year: 1815,
  edition_number: 5,
  volume: 2,
  start_page: 234,
  end_page: 289,
  text_length: 123456,
  word_count: 21543,
  is_cross_reference: false,
  has_figures: true
})
```

#### Topic
Normalized topic for linking articles across editions
```cypher
CREATE (t:Topic {
  id: "astronomy",
  canonical_name: "Astronomy",
  aliases: ["ASTRONOMY", "Astron.", "Astronomical Science"]
})
```

#### Person
Named entities mentioned in articles
```cypher
CREATE (p:Person {
  id: "newton_isaac",
  name: "Isaac Newton",
  birth_year: 1643,
  death_year: 1727,
  wikidata_qid: "Q935"
})
```

#### Place
```cypher
CREATE (pl:Place {
  id: "edinburgh",
  name: "Edinburgh",
  country: "Scotland",
  geonames_id: 2650225,
  wikidata_qid: "Q23436"
})
```

### Relationships

#### Edition Structure
```cypher
(a:Article)-[:IN_EDITION]->(e:Edition)
(a:Article)-[:ABOUT]->(t:Topic)
```

#### Cross-Edition Links
```cypher
// Direct successor (same headword, next edition)
(a1:Article)-[:SUCCESSOR_OF {
  headword_match: true,
  similarity_score: 0.87
}]->(a2:Article)

// Topic continuity (different headword, same topic)
(a1:Article)-[:SAME_TOPIC]->(t:Topic)<-[:SAME_TOPIC]-(a2:Article)
```

#### Internal References
```cypher
// "See ELECTRICITY"
(a1:Article)-[:CROSS_REFERENCES {
  type: "see"
}]->(a2:Article)

// "See also MAGNETISM"
(a1:Article)-[:CROSS_REFERENCES {
  type: "see_also"
}]->(a2:Article)
```

#### Named Entities
```cypher
(a:Article)-[:MENTIONS {
  count: 5,
  contexts: ["discovered by", "according to"]
}]->(p:Person)

(a:Article)-[:MENTIONS]->(pl:Place)
```

### Indexes

```cypher
-- Performance indexes
CREATE INDEX article_headword FOR (a:Article) ON (a.headword);
CREATE INDEX article_edition FOR (a:Article) ON (a.edition_year);
CREATE INDEX article_id FOR (a:Article) ON (a.id);
CREATE INDEX topic_name FOR (t:Topic) ON (t.canonical_name);

-- Composite for lookups
CREATE INDEX article_lookup FOR (a:Article) ON (a.headword, a.edition_year);

-- Fulltext for search
CREATE FULLTEXT INDEX article_text FOR (a:Article) ON (a.text);
```

## Article Linking Strategy

### Phase 1: Exact Headword Match
Link articles with identical headwords across consecutive editions.

```python
# Example: ASTRONOMY in 1815 -> ASTRONOMY in 1823
def link_exact_matches(edition1, edition2):
    for a1 in edition1.articles:
        for a2 in edition2.articles:
            if a1.headword == a2.headword:
                create_successor_link(a1, a2)
```

### Phase 2: Fuzzy Headword Match
Handle spelling variations, modernization:
- "AETHER" -> "ETHER"
- "CHYMISTRY" -> "CHEMISTRY"
- "MOHAMMEDANISM" -> "ISLAM"

```python
def normalize_headword(headword):
    # Apply historical spelling normalization
    rules = [
        (r'^AE', 'E'),      # AETHER -> ETHER
        (r'^CH[YI]M', 'CHEM'),  # CHYMISTRY -> CHEMISTRY
        # etc.
    ]
```

### Phase 3: Topic Clustering
Use embeddings to cluster articles by semantic similarity:

```python
# Embed article texts
embeddings = model.encode([a.text for a in all_articles])

# Cluster to find topic groups
clusters = hdbscan.HDBSCAN().fit(embeddings)

# Create Topic nodes for each cluster
for cluster_id in set(clusters.labels_):
    articles = [a for a, c in zip(all_articles, clusters.labels_) if c == cluster_id]
    topic = create_topic(articles)
    for a in articles:
        link_to_topic(a, topic)
```

### Phase 4: Text Similarity Scoring
For linked articles, compute similarity metrics:

```python
def compute_similarity(a1, a2):
    return {
        'jaccard': jaccard_similarity(a1.text, a2.text),
        'cosine': cosine_similarity(embed(a1.text), embed(a2.text)),
        'length_ratio': len(a1.text) / len(a2.text),
        'shared_sentences': count_shared_sentences(a1, a2)
    }
```

## Sample Queries

### Track Article Evolution
```cypher
// How did ELECTRICITY evolve across editions?
MATCH path = (a1:Article {headword: 'ELECTRICITY'})-[:SUCCESSOR_OF*]->(a2:Article)
WHERE a1.edition_year = 1771
RETURN a1.edition_year, a2.edition_year,
       a1.word_count, a2.word_count,
       a2.word_count - a1.word_count as growth
```

### Find New Topics
```cypher
// What topics first appeared in the 9th edition?
MATCH (a:Article)-[:IN_EDITION]->(e:Edition {number: 9})
WHERE NOT EXISTS {
  MATCH (a)-[:SUCCESSOR_OF]->(prev:Article)
}
RETURN a.headword, a.word_count
ORDER BY a.word_count DESC
LIMIT 50
```

### Cross-Reference Network
```cypher
// Most referenced articles in 1815
MATCH (a:Article {edition_year: 1815})<-[:CROSS_REFERENCES]-(other)
RETURN a.headword, count(other) as references
ORDER BY references DESC
LIMIT 20
```

### Knowledge Expansion Analysis
```cypher
// Which topics grew most between 1771 and 1911?
MATCH (a1:Article)-[:ABOUT]->(t:Topic)<-[:ABOUT]-(a2:Article)
WHERE a1.edition_year = 1771 AND a2.edition_year = 1911
RETURN t.canonical_name,
       a1.word_count as words_1771,
       a2.word_count as words_1911,
       toFloat(a2.word_count) / a1.word_count as growth_factor
ORDER BY growth_factor DESC
```

## Implementation Phases

### Phase 1: Data Ingestion (Current)
- [x] Parse 1815 articles from OLMoCR JSON
- [ ] Process 1771 first edition
- [ ] Process 1810 (4th) edition
- [ ] Process 1823 (6th) edition
- [ ] Parse all editions into consistent format

### Phase 2: Article Linking
- [ ] Implement exact headword matching
- [ ] Build spelling normalization rules
- [ ] Create successor relationships

### Phase 3: Topic Clustering
- [ ] Generate article embeddings
- [ ] Cluster by semantic similarity
- [ ] Create Topic nodes

### Phase 4: Named Entity Extraction
- [ ] Extract person mentions
- [ ] Extract place mentions
- [ ] Link to Wikidata/GeoNames

### Phase 5: Graph-RAG Integration
- [ ] Store embeddings in vector index
- [ ] Build retrieval pipeline
- [ ] Create query interface

## File Locations

- Parser: `/home/jic823/1815EncyclopediaBritannicaNLS/parse_britannica_articles.py`
- 1815 Articles: `/home/jic823/1815EncyclopediaBritannicaNLS/articles_1815_clean.jsonl`
- 1815 JSONs: `/home/jic823/1815EncyclopediaBritannicaNLS/json/`
- SLURM Jobs: `/home/jic823/cluster/nibi_*_olmocr.sh`
