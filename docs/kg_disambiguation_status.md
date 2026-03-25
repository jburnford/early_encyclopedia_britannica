# Knowledge Graph — Entity Disambiguation Status

## Date: 2026-03-13

## Overall Goal

Build a knowledge graph tracking how knowledge changes across 8 editions of the Encyclopedia Britannica (1771–1860). Entity disambiguation is the prerequisite: cluster raw NER surface forms into canonical entities with stable cross-edition identities and external links (GeoNames, Wikidata).

**Priority order**: TOPONYM → COMMODITY → PERSON → ORG

---

## TOPONYM Disambiguation — 94.3% Grounded

### What's Done

**Normalization pipeline** (`graphrag/disambiguate_toponyms.py`):
- Extracted 644K mentions / 100K unique surface forms from NER output
- Case-folded, expanded abbreviations (Lond.→London, Edinb.→Edinburgh, etc.)
- Collapsed hyphens (East-Indies → East Indies, New-York → New York)
- Filtered false positives (Lat., Long., compass directions)
- Clustered into 93,200 normalized clusters

**GeoNames matching** (via Neo4j at `bolt://206.12.90.118:7687`, 6.2M Place nodes):
- 5,102 clusters matched with 10+ mentions
- Fulltext index search, exact name matching
- 270 clusters have significant alternatives (London GB vs London CA, etc.)

**Wikidata matching** (API search + SPARQL re-query):
- 817 clusters matched (Mediterranean, Peloponnesus, Tartary, Hellespont, etc.)
- Re-queried via SPARQL for ancient/historical place names missed by GeoNames

**Cleanup** (`graphrag/clean_toponym_clusters.py`):
- Fixed 48 Old World/New World primary misassignments (Canterbury NZ→GB, York CA→GB, Perth AU→GB)
- Removed 59 non-geographic Wikidata matches (insect genera, family names)
- Merged 21 spelling variant pairs (Hindoostan+Hindustan, Surry+Surrey, Strasburg+Strasbourg)

### Current Numbers (clusters with 10+ mentions)

| Source | Clusters | % |
|--------|----------|---|
| GeoNames matched | 5,102 | 81.3% |
| Wikidata matched | 817 | 13.0% |
| **Total grounded** | **5,919** | **94.3%** |
| Still unmatched | 356 | 5.7% |
| **Total** | **6,276** | 100% |

### Output Files

| File | Description |
|------|-------------|
| `data/ner/toponym_clusters_clean.jsonl` | Clean disambiguated clusters (93,200 total) |
| `data/ner/toponym_clusters_clean.csv` | Same data as CSV for review |
| `data/ner/toponym_clusters.jsonl` | Pre-cleanup version |

### Schema (per cluster in JSONL)

```json
{
  "cluster_id": "london",
  "label": "London",
  "variants": ["London", "LONDON", "Lond.", "london"],
  "total_mentions": 8807,
  "by_edition": {"1771": 312, "1778": 1045, ...},
  "article_count": 2341,
  "edition_count": 8,
  "frequency_rank": 3,
  "is_concept_headword": true,
  "match_type": "matched",
  "geonames_id": 2643743,
  "geonames_name": "London",
  "country": "GB",
  "feature_class": "P",
  "feature_code": "PPLC",
  "population": 8961989,
  "lat": 51.50853,
  "lon": -0.12574,
  "alternatives": [
    {"geonames_id": 6058560, "name": "London", "country": "CA", "population": 422324, ...}
  ]
}
```

For Wikidata-matched clusters, fields are `wikidata_qid`, `wikidata_label`, `wikidata_description` instead of GeoNames fields.

---

## Next Steps

### 1. Spot-check Wikidata QIDs (use MCP)

The Wikidata MCP server is now configured (`https://wd-mcp.wmcloud.org/mcp`) but requires a session restart to load. Use the MCP tools to:

- Verify a sample of 20-30 QID matches are correct (especially the SPARQL re-query batch)
- Use `search_items` for fuzzy matching on the 356 remaining unmatched clusters
- Top unmatched targets: Schonen (Skåne), Chalcedon (Kadıköy), Argyleshire (Argyll), Brundusium (Brindisi), Spalatro (Split), Wurtzburg (Würzburg), Marfeilles (Marseille)

### 2. COMMODITY Disambiguation

Second entity type. 196K mentions, 51K unique forms. Expected to be simpler:
- Constrained vocabulary (cotton, silk, sugar, gold...)
- No external authority needed — build a curated canonical list
- Same normalize-then-cluster approach
- Key challenge: singular/plural (slave/slaves), compound terms (sugar-cane/sugarcane/sugar cane)

### 3. PERSON Disambiguation

Hardest type. 289K mentions, 86K unique forms.
- Variant explosion: "Charles II." vs "Charles II", "Sir Isaac Newton" vs "Newton"
- Known false positive: "W. Long." (1,416 mentions) — longitude abbreviation
- Strategy: start with high-frequency persons, cluster by normalized name
- External linking: Wikidata for notable historical figures

### 4. ORG Disambiguation

Smallest type. 29K mentions, 7.8K unique forms.
- Case normalization covers many ("parliament" vs "Parliament")
- "Royal Society" vs "Royal Society of London" — real disambiguation needed
- Small enough to almost hand-curate

### 5. Neo4j Schema Design

After all 4 types are disambiguated:
- Node types: Article, Concept, ToponymEntity, CommodityEntity, PersonEntity, OrgEntity, Edition
- Temporal relationships: entity mentions per edition
- Cross-edition identity as the core structural element

### 6. GraphRAG

After graph is loaded:
- Embeddings (Voyage AI) for articles/concepts
- Vector index in Neo4j
- Temporal query patterns ("how did the article on SUGAR change between 1771 and 1860?")

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `graphrag/disambiguate_toponyms.py` | Main pipeline: normalize + GeoNames + Wikidata |
| `graphrag/clean_toponym_clusters.py` | Post-processing: OW/NW fix, bad WD removal, spelling merges |
| `graphrag/wikidata_requery.py` | SPARQL re-query for unmatched clusters |
| `graphrag/run_ner.py` | NER extraction (already complete) |
| `graphrag/build_concept_index.py` | Cross-edition concept index (already complete) |
| `graphrag/commodity_colocation.py` | Co-occurrence analysis (already complete) |

## Infrastructure

- **Neo4j GeoNames**: `bolt://206.12.90.118:7687` (6.2M Place nodes, password in `~/textasdatacolonialofficelist/.env`)
- **Wikidata MCP**: `https://wd-mcp.wmcloud.org/mcp` (configured, needs session restart)
- **NER data**: `data/ner/eb_{ed}_{year}.entities.jsonl` (8 files, 1.16M entities)
- **Concept index**: `graphrag/concept_index.json` (38,054 concepts across editions)
