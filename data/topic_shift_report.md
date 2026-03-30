# Topic Shifts in Cross-Edition Headword Index

**Date:** 2026-03-29
**Context:** During headword-to-Wikidata disambiguation, we discovered that many headwords in `data/cross_edition_index.jsonl` link articles with **different topics** across editions under the same ID. This breaks the assumption that a cross-edition entry tracks the *same* article over time, and produces incorrect links in the knowledge graph.

**Method:** We read the first 100–200 words of every edition for 20 ambiguous headwords. Topic shifts were identified by comparing opening text across editions.

## Summary of Findings

| Issue type | Count | Impact |
|------------|-------|--------|
| Headwords needing splits (different topics) | 12 | Wrong cross-edition links in KG |
| Swallowed editions (parser missed next headword) | 15 individual edition-entries | Inflated word counts, wrong content |
| Consistent entries (no action needed) | 5 | Can proceed to Wikidata grounding |

## 1. Headwords Requiring Splits

These entries link fundamentally different articles under the same ID. Each group below should become a separate entry in the cross-edition index with its own unique ID and Wikidata QID.

### BLACK (27K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| BLACK_COLOR | 1771, 1778, 1797 | Color/dyeing — "a well known colour, supposed to be owing to the absence of light" | — (generic) |
| BLACK_JOSEPH | 1810, 1815, 1823, 1842, 1860 | Joseph Black, chemist — "Dr Joseph, distinguished for his discoveries in chemistry, born 1728" | Q272512 |

### TEMPLE (32K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| TEMPLE_BUILDING | 1771, 1778 | Temple (place of worship) — "a general name for places of public worship" | Q44539 |
| TEMPLE_WILLIAM | 1797–1860 | Sir William Temple — "was born in London in the year 1628... statesman and essayist" | Q2248538 |

### DOUGLAS (20K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| DOUGLAS_TOWN | 1771 | Douglas, Isle of Man — "a port-town, and the best harbour in the Isle of Man" (20w) | Q2292301 |
| DOUGLAS_GAVIN | 1778–1860 | Gavin Douglas — "bishop of Dunkeld... third son of Archibald earl of Angus... born 1474" | Q389096 |

### BARRY (21K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| BARRY_HERALDRY | 1771 | Heraldic term — "in heraldry, is when an escutcheon is divided bar-ways" (119w) | — |
| BARRY_GIRALDUS | 1778–1815, 1842–1860 | Giraldus Cambrensis — "Girald, commonly called Giraldus Cambrensis... born at the castle of Mainarper, near Pembroke, 1146" | Q357824 |
| BARRY_JAMES | 1823 | James Barry, painter — "an eminent painter, was born at Cork, in Ireland, October 11, 1741" | Q712tried — needs search |

### WOOD (27K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| WOOD_MATERIAL | 1771, 1778 | Wood (material/craft) — 1771: "a solid substance, whereof the trunks and branches of trees consist"; 1778: wood moulding technique | — (generic) |
| WOOD_ANDREW | 1797 | Sir Andrew Wood — sea captain narrative (4.6K) | needs search |
| WOOD_ANTHONY | 1810–1860 | Anthony Wood — "an eminent biographer and antiquarian, born at Oxford in 1632" | Q691873 |

### JOHN (26K total, 7 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| JOHN_GOSPEL | 1771 | Gospel of St John — "a canonical book of the New Testament" (255w) | Q36192 |
| JOHN_PERSONS | 1778 | Dictionary of Johns — King John, John of Gaunt, John Sobieski, etc. (cross-refs) | — |
| JOHN_BAPTIST | 1797, 1815, 1823 | St John the Baptist — "the fore-runner of Jesus Christ, son of Zacharias and Elizabeth" | Q40662 |
| *(swallowed)* | 1810 | Starts with "See Felis" → LEO content | — |
| *(swallowed)* | 1842 | Starts with astronomy/longitude content | — |

### BULL (27K total, 8 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| BULL_CROSSREF | 1771 | Cross-references (zoology, astronomy, heraldry, papal bull) | — |
| BULL_JOHN | 1778–1842 | Dr John Bull — "a celebrated musician and composer, born in Somersetshire about 1563" | Q169446 (needs verify) |
| BULL_PAPAL | 1860 | Papal bull — "a letter written on parchment, sealed with lead, issued by order of the pope" | Q189867 |

### MOORE (20K total, 6 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| MOORE_EDWARD | 1797, 1815, 1860 | Edward Moore — "a dramatist... born at Abington in Berkshire in 1712" | Q1293055 |
| MOORE_JOHN | 1842 | Dr John Moore — "son of one of the clergymen of Stirling, born 1730" | needs search |
| *(swallowed)* | 1810 | Starts with "in Anatomy, tendons" | — |
| *(swallowed)* | 1823 | Starts with "a town of Dauphiny in France" | — |

### PASSION (65K total, 5 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| PASSION_RHETORIC | 1778 | Rhetorical device — "as often the effect of redoubling words" (2.9K) | Q33005760? |
| PASSION_PHILOSOPHY | 1797, 1815, 1823 | Philosophy of passions/emotions — "a word denoting every feeling of the mind occasioned by an extrinsic cause" (15K each) | Q3368369 |
| *(swallowed)* | 1810 | Starts with "in Anatomy, the eye-brow" (SUPERCILIUM content) | — |

### CLARKE (30K total, 7 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| CLARKE_SAMUEL_NONCONF | 1778, 1797, 1842 | Samuel Clarke, Nonconformist minister — "a preacher in the reign of Charles II... minister of St Bennet Fink" | Q7411116 |
| CLARKE_SAMUEL_PHIL | 1815 (partial) | Dr Samuel Clarke, philosopher (1675–1729) — appears after the nonconformist entry | Q381073 |
| CLARKE_WILLIAM | 1823 | William Clarke, divine — "born at Haghmoon-abbey in Shropshire, 1696" | needs search |
| CLARKE_ADAM | 1860 | Adam Clarke, Methodist scholar — "born of humble parents in 1762" | Q325037 (needs verify) |
| *(swallowed)* | 1810 | Starts with botany — "a genus of plants, belonging to the syngenesia class" | — |

### PHILIP (44K total, 6 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| PHILIP_BIBLICAL | 1797, 1815 | Philip, foster-brother of Antiochus Epiphanes — "was a Phrygian by birth" (11K each) | needs search |
| PHILIP_DEACON | 1823 | Philip the Deacon — "second of the seven deacons, chosen by the apostles" (10K) | Q311432 (needs verify) |
| PHILIP_APOSTLE | 1842 | Philip the Apostle — "one of the apostles, was a native of Bethsaida" (486w) | Q43719 |
| PHILIP_KINGS | 1860 | Dictionary of Philips — "name of five kings of Macedonia... five kings of Spain" (266w) | — |
| *(swallowed)* | 1810 | Starts with "a country which gives its head..." (KINGDOM content) | — |

### SIGN (32K total, 6 editions)

| Split ID | Editions | Topic | Wikidata |
|----------|----------|-------|----------|
| SIGN_DEFINITION | 1771, 1778, 1797 | Sign (general) — cross-references to algebra, astronomy, medicine (53–2.7K) | — |
| SIGN_NAVAL | 1815, 1823 | Naval signals — "NAVAL SIGNALS. When we read at our fireside the account of an engagement..." (10K each) | needs search |
| *(swallowed)* | 1810 | Starts with "an ancient Greek poet of Megara" (THEOGNIS content) | — |

## 2. Swallowed Editions (Parser Errors)

These editions have content from a **different article** that was swallowed because the parser missed the next headword. They should be removed from the cross-edition entry and ideally recovered as their correct article.

| Headword | Edition | Actual content | Likely real article |
|----------|---------|---------------|---------------------|
| JOHN | 1810 | "See Felis... Leo X Pope..." | LEO |
| JOHN | 1842 | Astronomy / longitude calculation | Unknown |
| SIGN | 1810 | "an ancient Greek poet of Megara" | THEOGNIS |
| SHOOTING | 1797 | Mid-sentence about dog collars | Unknown (tail of previous article) |
| SHOOTING | 1810 | "the person who makes his will and testament" | TESTATOR |
| PASSION | 1810 | "in Anatomy, the eye-brow" | SUPERCILIUM |
| MIGRATION | 1810 | "in Law, a writ issued in divers cases" | SUPERSEDEAS or PROHIBITION |
| MOORE | 1810 | "in Anatomy, white firm tenacious parts" | TENDONS |
| MOORE | 1823 | "a town of Dauphiny in France" | Geographic entry (MONTELIMART?) |
| CLARKE | 1810 | "a genus of plants, belonging to the syngenesia class" | CLARKIA (botanical genus) |
| PHILIP | 1810 | "a country which gives its head... a king" | KINGDOM |
| STORK | 1797 | "See ARDEA. STOVE for heating apartments..." | STOVE |
| STORK | 1810 | "See ARDEA. STOVE for heating apartments..." | STOVE |
| STORK | 1815 | "See ARDEA. STOVE for heating apartments..." | STOVE |

**Note:** The 1810 4th edition is responsible for 8 of 14 swallowed entries. This edition is a supplement with non-alphabetical volumes, which explains why headword-based parsing fails more often.

## 3. Consistent Entries (No Split Needed)

These headwords have the same topic across all (non-swallowed) editions and can be grounded directly.

| Headword | Topic | Notes | Wikidata |
|----------|-------|-------|----------|
| SHOOTING | Shooting/hunting sport | Consistent 1778–1860 (excluding swallowed 1797, 1810) | Q206989 |
| MIGRATION | Bird migration | Consistent 1778–1823 (excluding swallowed 1810) | Q216507 |
| CORONA | Optics — halos/luminous circles | Consistent all editions (brief anatomy def + long optics) | Q131559 |
| HOUND | Dogs/hound training | Consistent 1797–1860, broadens to general dog article in 1860 | — |
| COOPER | Cooperage trade | 1810–1860 consistent; 1778–1797 has mixed trade + Shaftesbury bio | Q1129337 (needs verify) |
| PRESCRIPTION | Legal prescription | Consistent topic across non-swallowed editions (also covers medical Rx briefly) | Q97358154 |
| HENRY | Biographical dictionary of Henrys | Multiple subjects, no single QID possible | — |

## 4. Recommended Actions

### Immediate (data fixes)
1. **Split the 12 headwords** listed in Section 1 into separate cross-edition index entries with new unique IDs
2. **Remove swallowed editions** from their current headword entries (Section 2)
3. **Add swallowed content** to `fix_mega_articles.py` for recovery as correct articles

### For the knowledge graph
4. **Ground the new split entries** to Wikidata QIDs where identified above
5. **Mark biographical dictionaries** (HENRY, JOHN_PERSONS, PHILIP_KINGS) as multi-subject entries that cannot be grounded to a single QID

### Future (automated detection)
6. **Embedding-based detection:** Run first 500 tokens of each edition through an embedding model, then flag entries where cosine similarity between editions drops below threshold (e.g., 0.5). This would catch topic shifts at scale across all 4,369 headwords.
7. **1810 4th edition audit:** Systematically check all 1810 entries for swallowed content, since this edition accounts for the majority of parser failures.

## 5. Files Referenced

- `data/cross_edition_index.jsonl` — current index (needs splits)
- `data/headword_matches.jsonl` — Wikidata groundings (843 matches)
- `data/ambiguous_headwords.md` — tracking file for ambiguous entries
- `scripts/fix_mega_articles.py` — manual split script (needs new entries for swallowed editions)
- `data/articles/*.articles.jsonl` — source article files
