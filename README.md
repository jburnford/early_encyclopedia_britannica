# Early Encyclopaedia Britannica Corpus

A parsed and searchable corpus of the first eight editions of the Encyclopaedia Britannica (1771-1860), built from OCR scans provided by the [National Library of Scotland](https://data.nls.uk/data/digitised-collections/encyclopaedia-britannica/).

**Live site:** [jburnford.github.io/early_encyclopedia_britannica](https://jburnford.github.io/early_encyclopedia_britannica/)

## The Corpus

| Edition | Year | Volumes | Articles | Words |
|---------|------|---------|----------|-------|
| 1st | 1771 | 3 | 9,336 | 1.6M |
| 2nd | 1778 | 10 | 14,415 | 5.9M |
| 3rd | 1797 | 18 | 17,312 | 10.0M |
| 4th (Supplement) | 1810 | 20 | 19,151 | 11.1M |
| 5th | 1815 | 20 | 19,208 | 11.0M |
| 6th | 1823 | 21 | 17,115 | 10.9M |
| 7th | 1842 | 21 | 16,160 | 12.0M |
| 8th | 1860 | 21 | 13,445 | 13.2M |
| **Total** | | **134** | **126,142** | **120M** |

## Project Structure

```
data/
  articles/           155 per-volume JSONL files (LIS parser output, 126K articles)
  export/             8 per-edition JSONL files (consolidated for site generation)
  ocr/                Raw OCR source files and manifests from NLS
  headword_dictionary.json        49,812 headword entries across all editions
  headword_dictionary_clean.json  48,921 entries after cleanup
  statistics.json                 Parser statistics

docs/                 Generated static site (GitHub Pages)

graphrag/             Cross-edition knowledge graph tools
  clean_headword_dict.py    Headword dictionary cleanup
  build_concept_index.py    Cross-edition concept index builder
  concept_index.json        38,073 concepts linked across editions
```

## Data Sources

The OCR text comes from the National Library of Scotland's digitised collections:

- [Encyclopaedia Britannica on NLS Data Foundry](https://data.nls.uk/data/digitised-collections/encyclopaedia-britannica/)

Articles were extracted using a custom LIS (Longest Increasing Subsequence) parser that identifies headwords in the OCR text and segments the continuous text into individual encyclopedia articles. The parser handles 18th-century typographic conventions (J/I and U/V equivalences), cross-references, and multi-volume treatises.

## Cross-Edition Concept Index

The `graphrag/concept_index.json` maps 38,073 concepts across editions, enabling temporal analysis of how knowledge evolved:

- **7,931 core concepts** appear in 6 or more editions
- Track article growth (e.g., ICHTHYOLOGY grew from 16 words in 1771 to 162,498 in 1860)
- Identify when concepts were introduced or dropped
- Trace the expansion of scientific knowledge through the Enlightenment

## License

The OCR source data is provided by the National Library of Scotland under a [Creative Commons Attribution 4.0 International Licence](https://creativecommons.org/licenses/by/4.0/).
