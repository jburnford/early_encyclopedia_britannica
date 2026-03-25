# NER-Guided Parsing Audit — March 19, 2026

## What we did

Used NER entity data (1,157,244 entities across 119,851 articles) to detect parsing errors where the parser missed article boundaries, causing one article to absorb the next.

**Method**: Cross-edition comparison of entity counts. If FRAME normally has 2 entities but has 1,234 in the 6th edition, it absorbed FRANCE (confirmed by entity content: Meuse, Burgundy, Julius Caesar).

## Results

**161 new missed-heading candidates** added to `data/missed_headings_filtered.jsonl` (698 → 859 total):
- 140 from automated NER audit script (`graphrag/ner_parsing_audit.py`)
- 21 from manual agent-confirmed investigation of unresolved cases

### Key articles recovered

| Container | Absorbed | Edition | Expected Words |
|---|---|---|---|
| FRAME | **FRANCE** | 6th (1823) | 271K |
| FRAISE | **FRANCE** | 2nd (1778) | 271K |
| CHEMISE | **CHEMISTRY** | 7th (1842) | 359K |
| CHEMISE | **CHEMISTRY** | 2nd (1778) | 359K |
| MEDIATE | **MEDICINE** | 1st (1771) | 329K |
| AGRIA | **AGRICULTURE** | 2nd (1778) | 291K |
| BOSWORTH-MARKET | **BOTANY** | 7th (1842) | 194K |
| OPTIC ANGLE | **OPTICS** | 2nd (1778) | 193K |
| SURGEON | **SURGERY** | 4th (1810) | 148K |
| LAURENTIUS | **LAW** | 5th (1815) | 143K |
| LAUSANNE | **LAW** | 6th (1823) | 143K |
| EGREMONT | **EGYPT** | 7th (1842) | 124K |
| HISTORIOGRAPHER | **HISTORY** | 5th (1815) | 116K |
| PHYSIC | **PHYSICAL GEOGRAPHY** | 7th (1842) | 99K |
| MONETARIUS | **MONEY** | 7th (1842) | 91K |
| GEOMANCY | **GEOMETRY** | 7th (1842) | 74K |
| INDEPENDENTS | **INDIA** | 4th (1810) | 61K |
| MEWING | **MEXICO** | 6th (1823) | 60K |
| RITUAL | **RIVER** | 7th (1842) | 59K |
| SURD | **SURGERY** | 7th (1842) | 148K |
| CANAL | **CANADA** | 7th (1842) | 36K |
| IVORY | **IRELAND** | 8th (1860) | — |
| SCOT | **SCOTLAND** | 8th (1860) | 220K |

### False positives identified by agents

- METEMPSYCHOSIS (3 editions) — actually mid-METAPHYSICS content, no boundary
- LIEGNITZ (8th) — mid-LIBRARIES content
- ETHIOPIA (3rd) — genuinely last article in volume
- SPAHIS → SPAIN — no heading exists in OCR text

## New script

`graphrag/ner_parsing_audit.py` — standalone analysis tool:
- Detects bloated articles via cross-edition NER comparison
- Diagnoses absorbed headwords via concept index gap analysis + late-boundary detection
- Localizes missing headwords in OCR text (12 heading patterns: titlecase, ALL-CAPS, bold, semicolon, standalone, etc.)
- Outputs candidates in `missed_headings_filtered.jsonl` format

```bash
python3 graphrag/ner_parsing_audit.py \
  --min-bloat-ratio 10 --min-entities 50 \
  --output data/ner_audit_candidates.jsonl \
  --report data/ner_parsing_audit_report.txt
```

## Output files

- `data/missed_headings_filtered.jsonl` — 859 entries (ready for parser)
- `data/ner_audit_candidates.jsonl` — 140 automated candidates
- `data/ner_parsing_audit_report.txt` — full audit report

## Next steps: re-run parser

```bash
cd /home/jic823/1815EncyclopediaBritannicaNLS

# Re-parse all editions with new headings (~2-5 min)
python3 scripts/parse_britannica.py --phase lis

# Re-export to per-edition JSONL
python3 scripts/parse_britannica.py --phase export
```

The parser's `supplementary_injection()` will automatically pick up the 161 new entries from `data/missed_headings_filtered.jsonl`. No parser code changes needed.

After re-parsing, re-run NER on affected volumes to verify entity counts normalize (FRAME should drop from 1,234 entities to ~2).
