# Named Entity Recognition & Commodity-Place Co-occurrence Analysis

## Encyclopedia Britannica, 1st–8th Editions (1771–1860)

### Overview

We extracted 1.15 million named entities from 119,851 articles across all 8 editions of the Encyclopedia Britannica using [EarlyModernNER](https://github.com/jacobpol/earlymodernner), a set of LoRA adapters fine-tuned on early modern English. Four entity types were extracted: TOPONYM (places), PERSON, ORGANIZATION, and COMMODITY.

We then performed windowed co-occurrence analysis to identify which places are discussed alongside which commodities at paragraph level (within 150 words), revealing how the Britannica's economic geography shifts across the century.

### NER Pipeline

- **Model**: Qwen3-4B-Instruct (4-bit NF4) with four specialized LoRA adapters
- **Chunking**: Articles longer than 3,000 words split into overlapping chunks (200-word overlap), reassembled with deduplication
- **Infrastructure**: 133 parallel SLURM jobs on Narval (DRAC), one per volume, using A100 MIG slices (20GB)
- **Runtime**: ~18 hours total wall time across all jobs

### Entity Counts

| Edition | Year | Articles | Entities | TOPONYM | PERSON | ORG | COMMODITY |
|---------|------|----------|----------|---------|--------|-----|-----------|
| 1st | 1771 | 8,122 | 19,077 | 12,160 | 1,166 | 484 | 5,267 |
| 2nd | 1778 | 13,559 | 101,002 | 55,567 | 24,126 | 2,206 | 19,103 |
| 3rd | 1797 | 16,481 | 152,531 | 84,330 | 38,829 | 3,350 | 26,022 |
| 4th | 1810 | 18,231 | 160,280 | 86,827 | 41,690 | 3,835 | 27,928 |
| 5th | 1815 | 18,322 | 156,212 | 85,141 | 40,485 | 3,665 | 26,921 |
| 6th | 1823 | 16,270 | 151,555 | 80,913 | 39,920 | 3,682 | 27,040 |
| 7th | 1842 | 15,879 | 189,955 | 109,524 | 45,797 | 5,059 | 29,575 |
| 8th | 1860 | 12,987 | 226,632 | 129,211 | 56,897 | 6,626 | 33,898 |

### Windowed Co-occurrence Method

Instead of article-level co-occurrence (where a 50-page article on CHEMISTRY would link "sugar" to every place mentioned anywhere in it), we use a 150-word window around each commodity mention in the raw text to find nearby toponyms. This provides paragraph-level locality and filters out incidental associations like publisher cities.

### Key Findings

#### The Eastward Shift of Commodity Geography

All major commodities show the same macro pattern: associations shift from the **Americas and Caribbean** in early editions to **India, China, and Southeast Asia** in later editions.

**Sugar**:
- 1778–1797: Caribbean dominates (Barbados, Jamaica, Mexico, Brazil, West Indies)
- 1810–1823: British Guiana enters (Demerara), India/Bengal growing
- 1842–1860: India prominent, United States appears, Java enters — truly global

**Cotton**:
- 1778–1797: Broadly distributed (Persia, China, Mexico, Africa)
- 1823: Manufacturing centers appear (Glasgow, Manchester, Lancashire)
- 1842–1860: Manchester (110 mentions) and India (199) dominate — Industrial Revolution

**Opium** (most dramatic shift):
- 1778–1823: Medical topic — Bath, Edinburgh, London (medical centers), Turkey as source
- 1842: Sudden pivot to Bengal, India, China, Calcutta — the Opium Wars
- 1860: India (56) and China (45) dominate; Canton, Patna appear

**Indigo**:
- 1778–1823: Americas (Carolina, Guatemala, Mexico, Guinea)
- 1842–1860: India/Bengal/Ganges/Calcutta — British colonial plantations

#### Place Commodity Profiles

**China**: porcelain → tea → silk → opium. By 1860, opium (51 mentions) is the 6th commodity.

**Jamaica**: Negroes/slaves persistent throughout. Sugar + rum + coffee = plantation economy. Pimento distinctly Jamaican.

**Brazil**: Gemstones (diamond, tourmaline, topaz) in early editions → plantation commodities (cotton, coffee, sugar) by 1842–1860.

**Gold Coast**: Negroes and slaves are the top commodities in *every edition* from 1797–1860.

**Canada**: Furs/beaver → timber (70 mentions by 1842) → copper, wheat, fish, coal by 1860.

**Rhubarb**: Russia/Siberia/Turkey (overland routes) → China/Yunnan/Kashgar/Mongolia by 1860 (direct knowledge of Chinese interior).

**Wool**: Consistently European (England, France, Spain). Spanish merino regions (Valencia, Murcia, Biscay) prominent throughout. English counties grow (Leicestershire, Yorkshire, Cheviot).

### Visualizations

Interactive Plotly visualizations available in `data/viz/index.html`:
- 12 commodity bump charts (place rankings over time)
- 10 place profiles (commodity mix over time)
- Combined heatmap of 6 key places

### Files

| File | Description |
|------|-------------|
| `data/ner/eb_{ed}_{year}.entities.jsonl` | Merged NER results per edition |
| `data/ner/colocation_by_commodity.json` | Windowed co-occurrence: commodity → places |
| `data/ner/colocation_by_place.json` | Windowed co-occurrence: place → commodities |
| `data/viz/index.html` | Visualization dashboard |
| `graphrag/run_ner.py` | NER extraction script |
| `graphrag/commodity_colocation.py` | Co-occurrence analysis CLI |
| `graphrag/build_colocation_data.py` | Structured data builder |
| `graphrag/visualize_colocation.py` | Visualization generator |
