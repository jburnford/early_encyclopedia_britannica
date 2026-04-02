# Proposed Fixes from Swallowed Article Detection

**Generated:** auto
**Total proposed fixes:** 762

## Summary

| Confidence | Count | Description |
|-----------|-------|-------------|
| HIGH | 80 | Gap match + multi-edition |
| MEDIUM | 682 | Gap match OR multi-edition + structural |
| MEDIUM-LOW | 0 | Multi-edition OR structural signal |
| LOW | 0 | Single-edition, no gap match |

### Alphabetical Adjacency Score

| Score | Count | Meaning |
|-------|-------|---------|
| 5 | 105 | Exact neighbor — shared 3+ prefix, safe to auto-apply |
| 4 | 31 | Close neighbor — shared 2+ prefix, high confidence |
| 3 | 24 | Same letter — plausible |
| 2 | 16 | Next letter or no headword — review |
| 1 | 55 | Backward but close — caution |
| 0 | 531 | Far away or wrong direction — likely false positive |

## How to Use

Review each proposed fix, then add confirmed ones to `scripts/fix_mega_articles.py`:
```python
# Example fix spec:
# (year, 'PARENT_TITLE', 'source_file_pattern', [
#     ('BREAK_HEADWORD', r'regex_pattern', after_pct),
# ]),
```

## HIGH Confidence (80 fixes)

- 🟢 **CHRISTIANA** → **CHRISTIANITY** (1815) sim=0.147 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...ch these experiments are made have no quicksilver upon them.`
  - `Christianity, however, explained and inculcated the truth of this doctrine in all its splendour and `
  - Fix: `(1815, 'CHRISTIANA', 'eb_5th_1815_v06_ENL-CRY', [('CHRISTIANITY', r'Christianity,\s+however,\s+explained', 22)])`

- 🟢 **CHRISTIANA** → **CHRISTIANITY** (1823) sim=0.149 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...ble in proportion as the matter interposed was made thinner.`
  - `Christianity, however, explained and inculcated the truth of this doctrine in all its splendour, and`
  - Fix: `(1823, 'CHRISTIANA', 'eb_6th_1823_v06_ENL-CRY', [('CHRISTIANITY', r'Christianity,\s+however,\s+explained', 26)])`

- 🟢 **PHILIP** → **PHILIP** (1815) sim=0.286 [new_headword] [gap: VARIANT] (2 eds: 1810, 1815)
  - `...Antioch, and put Philip to death, who was taken in the city.`
  - `PHILIP the apostle was a native of Bethsaida in Galilee. Jesus Christ having seen him, said to him, `
  - Fix: `(1815, 'PHILIP', 'eb_5th_1815_v16_ENL-HOR', [('PHILIP', r'PHILIP\s+the\s+apostle', 0)])`

- 🟢 **PHILIP** → **PHILIP** (1810) sim=0.287 [new_headword] [gap: EDITORIAL] (2 eds: 1810, 1815)
  - `...Antioch, and put Philip to death, who was taken in the city.`
  - `PHILIP the apostle was a native of Bethsaida in Galilee. Jesus Christ having seen him, said to him, `
  - Fix: `(1810, 'PHILIP', 'eb_4th_1810_v17_PAR-PHL', [('PHILIP', r'PHILIP\s+the\s+apostle', 0)])`

- 🟢 **PAL** → **PAL** (1810) sim=0.300 [new_headword] [gap: EDITORIAL] (4 eds: 1778, 1810, 1815, 1842)
  - `...is extravagancies. We have only some fragments of his works.`
  - `PALÆOLOGUS, MICHAEL, a very able man who was governor of Asia under the emperor Theodorus Laiçaros; `
  - Fix: `(1810, 'PAL', 'eb_4th_1810_v15_ORD-PAR', [('PAL', r'PALÆOLOGUS,\s+MICHAEL,\s+a', 31)])`

- 🟢 **PAL** → **PAL** (1815) sim=0.302 [new_headword] [gap: EDITORIAL] (4 eds: 1778, 1810, 1815, 1842)
  - `...ee CONSTANTINOPLE, from No. 145, to the end of that article.`
  - `PALÆPAPHOS (Strabo, Virgil, Pliny), a town of Cyprus, where stood a temple of Venus; and an adjoinin`
  - Fix: `(1815, 'PAL', 'eb_5th_1815_v15_NIC-CCC', [('PAL', r'PALÆPAPHOS\s+\(Strabo,\s+Virgil,', 82)])`

- 🟢 **PAL** → **PAL** (1842) sim=0.309 [new_headword] [gap: PARSING_OR_EDITORIAL] (4 eds: 1778, 1810, 1815, 1842)
  - `...ar even by Juvenal. Only some
fragments of his works remain.`
  - `PALÆOLOGUS, MICHAEL, a very able man, who was
governor of Asia under the Emperor Theodorus Lascaris;`
  - Fix: `(1842, 'PAL', 'eb_7th_1842_v16_SEV-PAN', [('PAL', r'PALÆOLOGUS,\s+MICHAEL,\s+a', 20)])`

- 🟢 **GUTTY** → **GUY** (1842) sim=0.086 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1823, 1842)
  - `...s is to be named; as gutty of sable, of gules, and so forth.`
  - `Guy, Thomas, an eminent bookseller, founder of the hospital for sick and lame in Southwark which bea`
  - Fix: `(1842, 'GUTTY', 'eb_7th_1842_v11_GRO-HYD', [('GUY', r'Guy,\s+Thomas,\s+an', 0)])`

- 🟢 **CONNOISSEUR** → **COR** (1810) sim=0.174 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1842)
  - `... any way, particularly in matters of painting and sculpture.`
  - `Cor. 2. The difference between the squares of \( CE, CG \)
the segments of the transverse diameter t`
  - Fix: `(1810, 'CONNOISSEUR', 'eb_4th_1810_v06_CON-CRY', [('COR', r'Cor\.\s+2\.\s+The', 0)])`

- 🟢 **BOURGES** → **BOUGET** (1810) sim=0.288 [new_headword] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...t in the centre of France. E. Long. 2° 30'. N. Lat. 47° 10'.`
  - `BOUGET, Dom John, an ingenious French antiquary, was born at the village of Beaumains near Falaise, `
  - Fix: `(1810, 'BOURGES', 'eb_4th_1810_v04_BOO-BRE', [('BOUGET', r'BOUGET,\s+Dom\s+John,', 0)])`

- 🟢 **BOURGES** → **BOUGET** (1815) sim=0.303 [new_headword] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...ost in the centre of France. E. Long. 2. 30. N. Lat. 47. 10.`
  - `BOUGET, Dom John, an ingenious French antiquary, was born at the village of Beaumains near Falaise, `
  - Fix: `(1815, 'BOURGES', 'eb_5th_1815_v04_ENL-BUR', [('BOUGET', r'BOUGET,\s+Dom\s+John,', 0)])`

- 🟢 **EARL** → **EAR** (1815) sim=0.328 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...EARL Marshal. See MARSHAL.`
  - `EAR is also used to signify a long cluster of flowers or seeds, produced by certain plants; usually `
  - Fix: `(1815, 'EARL', 'eb_5th_1815_v07_CUB-DIR', [('EAR', r'EAR\s+is\s+also', 25)])`

- 🟢 **EARL** → **EAR** (1810) sim=0.329 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...EARL Marshal. See MARSHAL.`
  - `EAR is also used to signify a long cluster of flowers or seeds, produced by certain plants; usually `
  - Fix: `(1810, 'EARL', 'eb_4th_1810_v07_STE-ELE', [('EAR', r'EAR\s+is\s+also', 31)])`

- 🟡 **KAZY** → **KEATE** (1815) sim=0.158 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...which under his seal are admitted as the originals in proof.`
  - `Keate, George, Esq. F.R.S., an eminent English writer, was born in 1730, and educated at Kingston sc`
  - Fix: `(1815, 'KAZY', 'eb_5th_1815_v11_ENL-LIE', [('KEATE', r'Keate,\s+George,\s+Esq\.', 0)])`

- 🟡 **KAZY** → **KEATE** (1823) sim=0.162 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...which under his seal are admitted as the originals in proof.`
  - `Keate, George, Esq. F.R.S. an eminent English writer, was born in 1730, and educated at Kingston sch`
  - Fix: `(1823, 'KAZY', 'eb_6th_1823_v11_ENL-LIE', [('KEATE', r'Keate,\s+George,\s+Esq\.', 0)])`

- 🟡 **EGYPT** → **FOR** (1815) sim=0.123 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...t in the mind of Britons which it would otherwise have done.`
  - `For a description of these stupendous and almost indestructible monuments of human grandeur, the pyr`
  - Fix: `(1815, 'EGYPT', 'eb_5th_1815_v07_CUB-DIR', [('FOR', r'For\s+a\s+description', 94)])`

- 🟡 **EGYPT** → **FOR** (1823) sim=0.124 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...reatly improved by the vigorous government of the Pacha Ali.`
  - `For a description of these stupendous monuments, the pyramids, see the article Pyramids. See also th`
  - Fix: `(1823, 'EGYPT', 'eb_6th_1823_v504_FOU-HOL', [('FOR', r'For\s+a\s+description', 93)])`

- 🟡 **EGYPT** → **FOR** (1810) sim=0.133 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...t in the mind of Britons which it would otherwise have done.`
  - `For a description of those stupendous and almost indestructible monuments of human grandeur, the pyr`
  - Fix: `(1810, 'EGYPT', 'eb_4th_1810_v07_STE-ELE', [('FOR', r'For\s+a\s+description', 94)])`

- 🟡 **LEGERDEMAIN** → **MILITARY LAW** (1810) sim=0.197 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `... and vagabonds, within the meaning of 17 Geo. III. c. 5, &c.`
  - `Military Law. See Military and Marine.

Law, John, the famous projector, was the eldest son of a gol`
  - Fix: `(1810, 'LEGERDEMAIN', 'eb_4th_1810_v11_JUN-LIE', [('MILITARY LAW', r'Military\s+Law\.\s+See', 250)])`

- 🟠 **MADURA** → **MACENAS** (1778) sim=0.196 [new_headword] [gap: VARIANT] (2 eds: 1778, 1842)
  - `...zing the country, as it passes along, with its mud, (Pliny).`
  - `MACENAS (Caius Cilnus), the great friend and counsellor of Augustus Caesar, was himself a very polit`
  - Fix: `(1778, 'MADURA', 'eb_2nd_1778_v06_BYW-IND', [('MACENAS', r'MACENAS\s+\(Caius\s+Cilnus\),', 3)])`

- 🔴 **MECKLENBURG** → **FOR** (1797) sim=0.063 [topic_change] [gap: VARIANT] (2 eds: 1797, 1860)
  - `...x-dollars. Each of these princes maintains a body of troops.`
  - `For it is probable, that there is no kind of motion but what may be referred to three easy and obvio`
  - Fix: `(1797, 'MECKLENBURG', 'eb_3rd_1797_v10_IND-MEC', [('FOR', r'For\s+it\s+is', 20)])`

- 🔴 **BLOCK** → **DANIEL** (1823) sim=0.072 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...erected them at proper stations along all their great roads.`
  - `Daniel, portrait painter, was born at Stettin in Pomerania in 1580, and gave early proofs of a good `
  - Fix: `(1823, 'BLOCK', 'eb_6th_1823_v502_AUS-CEL', [('DANIEL', r'Daniel,\s+portrait\s+painter,', 66)])`

- 🔴 **MASQUE** → **ARCHITECTURE** (1823) sim=0.086 [topic_change] [gap: OCR_GAP] (2 eds: 1810, 1823)
  - `... to be unravelled, we must leave to the reader to determine.`
  - `Architecture, is applied to certain pieces of sculpture, representing some hideous forms, grotesque,`
  - Fix: `(1823, 'MASQUE', 'eb_6th_1823_v12_ENL-ADD', [('ARCHITECTURE', r'Architecture,\s+is\s+applied', 53)])`

- 🔴 **MASQUE** → **ARCHITECTURE** (1810) sim=0.088 [topic_change] [gap: OCR_GAP] (2 eds: 1810, 1823)
  - `... to be unravelled, we must leave to the reader to determine.`
  - `Architecture, is applied to certain pieces of sculpture, representing some hideous forms, grotesque,`
  - Fix: `(1810, 'MASQUE', 'eb_4th_1810_v12_MAH-ADD', [('ARCHITECTURE', r'Architecture,\s+is\s+applied', 53)])`

- 🔴 **GEORGETOWN** → **COR** (1810) sim=0.120 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...ies 127 miles S.W. of Wilmington, and 681 from Philadelphia.`
  - `Cor. If the first of four proportional be greatest than the third, the second is greater than the fo`
  - Fix: `(1810, 'GEORGETOWN', 'eb_4th_1810_v09_FAR-GOT', [('COR', r'Cor\.\s+If\s+the', 0)])`

- 🔴 **ENGLISH LANGUAGE** → **CONTEMPORARY** (1842) sim=0.127 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1842, 1860)
  - `...of my resting? whether myn bond made not alle these things?*`
  - `Contemporary with Wycliffe was Geoffrey Chaucer, who is commonly regarded as the father of English p`
  - Fix: `(1842, 'ENGLISH LANGUAGE', 'eb_7th_1842_v09_ENG-FRA', [('CONTEMPORARY', r'Contemporary\s+with\s+Wycliffe', 36)])`

- 🔴 **HERRING** → **THOMAS** (1823) sim=0.128 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...enough to sustain a herring; otherwise the fish decay in it.`
  - `Thomas, archbishop of Canterbury, memorable for his attachment to civil and religious liberty, was t`
  - Fix: `(1823, 'HERRING', 'eb_6th_1823_v10_ENL-HYD', [('THOMAS', r'Thomas,\s+archbishop\s+of', 37)])`

- 🔴 **CIPHER** → **ORDER** (1823) sim=0.129 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...ters that answer to each other when you have fixed the dial.`
  - `Order of Cincinnatus, or the Cincinnati, a society which was established in America soon after the p`
  - Fix: `(1823, 'CIPHER', 'eb_6th_1823_v06_ENL-CRY', [('ORDER', r'Order\s+of\s+Cincinnatus,', 43)])`

- 🔴 **TROY-WEIGHT** → **FOR** (1815) sim=0.131 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...er-meat, unwrought pewter and lead, and some other articles.`
  - `For let the great circle of which A is the pole, meet the three sides in D, E, F; then F is the pole`
  - Fix: `(1815, 'TROY-WEIGHT', 'eb_5th_1815_v20_SUI-DIR', [('FOR', r'For\s+let\s+the', 23)])`

- 🔴 **PRINTING** → **STEREOTYPE PRINTING** (1823) sim=0.131 [person_bio] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...in my thinkeinge.
God be at myn ende,
And at my departyngye.`
  - `Stereotype Printing. Different persons in different countries have claimed the merit of this inventi`
  - Fix: `(1823, 'PRINTING', 'eb_6th_1823_v17_ENL-RHI', [('STEREOTYPE PRINTING', r'Stereotype\s+Printing\.\s+Different', 36)])`

- 🔴 **HERRING** → **THOMAS** (1815) sim=0.140 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...enough to sustain a herring; otherwise the fish decay in it.`
  - `Thomas, archbishop of Canterbury, memorable for his attachment to civil and religious liberty, was t`
  - Fix: `(1815, 'HERRING', 'eb_5th_1815_v10_GOT-HYD', [('THOMAS', r'Thomas,\s+archbishop\s+of', 37)])`

- 🔴 **SHARP** → **MUSIC** (1815) sim=0.147 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1842)
  - `...ove what was left,—so deeply was he engaged in calculations.`
  - `Music. See Interval.`
  - Fix: `(1815, 'SHARP', 'eb_5th_1815_v19_SCR-DVI', [('MUSIC', r'Music\.\s+See\s+Interval\.', 83)])`

- 🔴 **SCIENCE** → **FIX** (1810) sim=0.150 [topic_change] [gap: EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...n angel; unless you would be content to live with the devil.`
  - `Fix a common ewer, as A (fig. 42.) of about 12 inches high, upon a square stand B C; on one side of `
  - Fix: `(1810, 'SCIENCE', 'eb_4th_1810_v18_RUS-SCR', [('FIX', r'Fix\s+a\s+common', 49)])`

- 🔴 **SCIENCE** → **FIX** (1815) sim=0.150 [topic_change] [gap: EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...n angel; unless you would be content to live with the devil.`
  - `Fix a common ewer, as A (fig. 42.) of about 12 inches high, upon a square stand BC; on one side of w`
  - Fix: `(1815, 'SCIENCE', 'eb_5th_1815_v18_ENL-SCR', [('FIX', r'Fix\s+a\s+common', 63)])`

- 🔴 **SCIENCE** → **FIX** (1823) sim=0.150 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...n angel; unless you would be content to live with the devil.`
  - `Fix a common ewer, as A (fig. 42.) of about 12 inches high, upon a square stand BC; on one side of w`
  - Fix: `(1823, 'SCIENCE', 'eb_6th_1823_v18_ENL-SCR', [('FIX', r'Fix\s+a\s+common', 66)])`

- 🔴 **SPAIN** → **CHARLES** (1797) sim=0.156 [topic_change] [gap: PARSING_OR_EDITORIAL] (4 eds: 1797, 1810, 1815, 1823)
  - `...his daughter upon Octavio Farnese, son of the duke of Parma.`
  - `Charles had soon farther cause to be sensible of his obligations to the holy father for bringing abo`
  - Fix: `(1797, 'SPAIN', 'eb_3rd_1797_v17_TRE-STR', [('CHARLES', r'Charles\s+had\s+soon', 17)])`

- 🔴 **BRITAIN** → **TRANQUILLITY** (1797) sim=0.158 [topic_change] [gap: PARSING_OR_EDITORIAL] (4 eds: 1778, 1797, 1815, 1823)
  - `...alliance, by which means peace was again restored to Europe.`
  - `Tranquillity being thus established, the ministry proceeded to secure the dependency of the Irish pa`
  - Fix: `(1797, 'BRITAIN', 'eb_3rd_1797_v03_TRE-BYZ', [('TRANQUILLITY', r'Tranquillity\s+being\s+thus', 43)])`

- 🔴 **PARADISE** → **BIRD** (1810) sim=0.160 [topic_change] [gap: VARIANT] (2 eds: 1810, 1823)
  - `...h of imagination and invention which is perfectly wonderful.`
  - `Bird of Paradise. See the following article.`
  - Fix: `(1810, 'PARADISE', 'eb_4th_1810_v15_ORD-PAR', [('BIRD', r'Bird\s+of\s+Paradise\.', 76)])`

- 🔴 **BRITAIN** → **TRANQUILLITY** (1823) sim=0.161 [topic_change] [gap: PARSING_OR_EDITORIAL] (4 eds: 1778, 1797, 1815, 1823)
  - `...alliance, by which means peace was again restored to Europe.`
  - `Tranquillity being thus established, the ministry proceeded to secure the dependency of the Irish pa`
  - Fix: `(1823, 'BRITAIN', 'eb_6th_1823_v502_AUS-CEL', [('TRANQUILLITY', r'Tranquillity\s+being\s+thus', 7)])`

- 🔴 **CHARLOCK** → **QUEEN CHARLOTTE** (1810) sim=0.162 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...ever growing where there is a coat of grass upon the ground.`
  - `Queen Charlotte's Island, an island in the South sea, first discovered by Captain Wallis in the Dolp`
  - Fix: `(1810, 'CHARLOCK', 'eb_4th_1810_v05_CHA-CHI', [('QUEEN CHARLOTTE', r'Queen\s+Charlotte's\s+Island,', 0)])`

- 🔴 **CHARLOCK** → **QUEEN CHARLOTTE'S ISLAND** (1778) sim=0.163 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1778, 1797)
  - `...ever growing where there is a coat of grass upon the ground.`
  - `QUEEN CHARLOTTE'S ISLAND, an island in the south sea, first discovered by captain Wallis in the Dolp`
  - Fix: `(1778, 'CHARLOCK', 'eb_2nd_1778_v03_BYW-CRI', [('QUEEN CHARLOTTE'S ISLAND', r'QUEEN\s+CHARLOTTE'S\s+ISLAND,', 0)])`

- 🔴 **CHARLOCK** → **QUEEN CHARLOTTE'S ISLAND** (1797) sim=0.171 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1778, 1797)
  - `...ever growing where there is a coat of grass upon the ground.`
  - `QUEEN CHARLOTTE'S ISLAND, an island in the South Sea, first discovered by captain Wallis in the Dolp`
  - Fix: `(1797, 'CHARLOCK', 'eb_3rd_1797_v04_TRE-OMI', [('QUEEN CHARLOTTE'S ISLAND', r'QUEEN\s+CHARLOTTE'S\s+ISLAND,', 0)])`

- 🔴 **NUNDOCOMAR** → **MONTE NUOVO** (1815) sim=0.171 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...PHYSICS, Part III, Chap. IV. Of the Immortality of the Soul.`
  - `MONTE NUOVO, in the environs of Naples, blocks up the valley of Averno. "This mountain (Mr Swinburne`
  - Fix: `(1815, 'NUNDOCOMAR', 'eb_5th_1815_v15_NIC-CCC', [('MONTE NUOVO', r'MONTE\s+NUOVO,\s+in', 0)])`

- 🔴 **CINCTURE** → **POLYBIUS** (1810) sim=0.174 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...y family; Sandwich, an earldom to a branch of the Montagues.`
  - `Polybius says, that Æneas Tacticus, 2000 years ago, collected together 20 different manners of writi`
  - Fix: `(1810, 'CINCTURE', 'eb_4th_1810_v17_OBS-GEN', [('POLYBIUS', r'Polybius\s+says,\s+that', 0)])`

- 🔴 **SPAIN** → **CHARLES** (1810) sim=0.174 [topic_change] [gap: OCR_GAP] (4 eds: 1797, 1810, 1815, 1823)
  - `...onfines, all the places of strength belonging to the church.`
  - `Charles II. was succeeded by Philip V. duke of Anjou, and grandson to Louis XIV. of France, who had `
  - Fix: `(1810, 'SPAIN', 'eb_4th_1810_v19_SLE-SUG', [('CHARLES', r'Charles\s+II\.\s+was', 93)])`

- 🔴 **MANSFELD** → **PETER ERNEST** (1823) sim=0.174 [person_bio] [gap: VARIANT] (3 eds: 1810, 1815, 1823)
  - `...the circle of Upper Saxony. E. Long. 11. 41. N. Lat. 51. 38.`
  - `Peter Ernest, Count of, was descended from one of the most illustrious families in Germany, and whic`
  - Fix: `(1823, 'MANSFELD', 'eb_6th_1823_v12_ENL-ADD', [('PETER ERNEST', r'Peter\s+Ernest,\s+Count', 0)])`

- 🔴 **CHARLOCK** → **QUEEN CHARLOTTE** (1815) sim=0.175 [person_bio] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...ever growing where there is a coat of grass upon the ground.`
  - `Queen Charlotte's Island, an island in the South sea, first discovered by Captain Wallis in the Dolp`
  - Fix: `(1815, 'CHARLOCK', 'eb_5th_1815_v05_ENL-CHI', [('QUEEN CHARLOTTE', r'Queen\s+Charlotte's\s+Island,', 0)])`

- 🔴 **THEATRE** → **NOT** (1797) sim=0.176 [topic_change] [gap: VARIANT] (4 eds: 1797, 1810, 1815, 1823)
  - `...it was composed. This would save much impertinent criticism.`
  - `Not fewer than 19 playhouses had been opened before the year 1633, when Pryne published his *Histrio`
  - Fix: `(1797, 'THEATRE', 'eb_3rd_1797_v18_IND-ER', [('NOT', r'Not\s+fewer\s+than', 70)])`

- 🔴 **LIMA** → **COFFEE** (1810) sim=0.179 [topic_change] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...ns, a narrow space being left for access to the inhabitants.`
  - `Coffee-houses were not known in Lima till the year 1771, when one was opened in the street of Santo `
  - Fix: `(1810, 'LIMA', 'eb_4th_1810_v17_LIE-MAH', [('COFFEE', r'Coffee\-houses\s+were\s+not', 70)])`

- 🔴 **CIPHER** → **ORDER** (1810) sim=0.179 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...e will make this plain. Suppose the letter to be
as follows.`
  - `Order of CINNATUS, or the Cincinnati, a society which was established in America soon after the peac`
  - Fix: `(1810, 'CIPHER', 'eb_4th_1810_v17_OBS-GEN', [('ORDER', r'Order\s+of\s+CINNATUS,', 133)])`

- 🔴 **POLE** → **ASTRONOMY** (1810) sim=0.180 [topic_change] [gap: EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...d temperate vindication of the doctrines of the Reformation.`
  - `Astronomy, that point in the heavens round which the whole sphere seems to turn. It is also used for`
  - Fix: `(1810, 'POLE', 'eb_4th_1810_v16_POE-BC', [('ASTRONOMY', r'Astronomy,\s+that\s+point', 5)])`

- 🔴 **CINCTURE** → **POLYBIUS** (1815) sim=0.181 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...y family; Sandwich, an earldom to a branch of the Montagues.`
  - `Polybius says, that Æneas Tacitus, 2000 years ago, collected together 20 different manners of writin`
  - Fix: `(1815, 'CINCTURE', 'eb_5th_1815_v06_ENL-CRY', [('POLYBIUS', r'Polybius\s+says,\s+that', 0)])`

- 🔴 **NUNDOCOMAR** → **MONTE NUOVO** (1810) sim=0.182 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...physics, Part III. Chap. IV. Of the Immortality of the Soul.`
  - `MONTE NUOVO, in the environs of Naples, blocks up the valley of Averno. "This mountain (Mr Swinburne`
  - Fix: `(1810, 'NUNDOCOMAR', 'eb_4th_1810_v15_NIC-ORA', [('MONTE NUOVO', r'MONTE\s+NUOVO,\s+in', 0)])`

- 🔴 **MAY** → **THOMAS** (1815) sim=0.182 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1860)
  - `...the island abound with fish, and the cliffs with water fowl.`
  - `Thomas, an eminent English poet and historian in the 17th century, was born of an ancient but decaye`
  - Fix: `(1815, 'MAY', 'eb_5th_1815_v17_ENL-RHI', [('THOMAS', r'Thomas,\s+an\s+eminent', 55)])`

- 🔴 **HAMILTON** → **VIVE** (1810) sim=0.183 [topic_change] [gap: OCR_GAP] (3 eds: 1797, 1810, 1823)
  - `... following satirical verses were written upon this occasion:`
  - `Vive diu, felix arbor, semperque vireto
Frondibus, ut nobis talia poma feras.`
  - Fix: `(1810, 'HAMILTON', 'eb_4th_1810_v05_GOT-HER', [('VIVE', r'Vive\s+diu,\s+felix', 61)])`

- 🔴 **THEATRE** → **NOT** (1810) sim=0.184 [topic_change] [gap: PARSING_OR_EDITORIAL] (4 eds: 1797, 1810, 1815, 1823)
  - `...it was composed. This would save much impertinent criticism.`
  - `Not fewer than 19 playhouses had been opened before the year 1633, when Prynne published his Histrio`
  - Fix: `(1810, 'THEATRE', 'eb_4th_1810_v20_SUI-PRE', [('NOT', r'Not\s+fewer\s+than', 70)])`

- 🔴 **LIMA** → **COFFEE** (1823) sim=0.184 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1810, 1815, 1823)
  - `...ns, a narrow space being left for access to the inhabitants.`
  - `Coffee-houses were not known in Lima till the year 1771, when one was opened in the street of Santo `
  - Fix: `(1823, 'LIMA', 'eb_6th_1823_v12_ENL-ADD', [('COFFEE', r'Coffee\-houses\s+were\s+not', 70)])`

- 🔴 **HAMILTON** → **VIVE** (1823) sim=0.187 [topic_change] [gap: PARSING_OR_EDITORIAL] (3 eds: 1797, 1810, 1823)
  - `... following sarcastic verses were written upon this occasion:`
  - `Vive diu, felix arbor, semperque vireto
Frondibus, ut nobis talia poma feras.`
  - Fix: `(1823, 'HAMILTON', 'eb_6th_1823_v10_ENL-HYD', [('VIVE', r'Vive\s+diu,\s+felix', 70)])`

- 🔴 **BRITAIN** → **ALL** (1815) sim=0.189 [topic_change] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...of want of economy throughout the whole American department.`
  - `All this time the violent animosities between the violent parties continued; the desire of peace was`
  - Fix: `(1815, 'BRITAIN', 'eb_5th_1815_v04_ENL-BUR', [('ALL', r'All\s+this\s+time', 18)])`

- 🔴 **BRITAIN** → **ALL** (1823) sim=0.190 [topic_change] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...of want of economy throughout the whole American department.`
  - `All this time the violent animosities between the parties continued; the desire of peace was gradual`
  - Fix: `(1823, 'BRITAIN', 'eb_6th_1823_v502_AUS-CEL', [('ALL', r'All\s+this\s+time', 12)])`

- 🔴 **TROY-WEIGHT** → **FOR** (1823) sim=0.194 [topic_change] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...er-meat, unwrought pewter and lead, and some other articles.`
  - `For, dividing the latter antecedent and consequent of the proportion in the foregoing lemma by \( \c`
  - Fix: `(1823, 'TROY-WEIGHT', 'eb_6th_1823_v20_ENL-ZYG', [('FOR', r'For,\s+dividing\s+the', 13)])`

- 🔴 **PHEGOR** → **PELLANDRIUM** (1810) sim=0.196 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1797, 1810)
  - `...at Phegor was the sun presiding over the mysteries of Venus.`
  - `PELLANDRIUM, water-hemlock; a genus of plants belonging to the pentandra class. See BOTANY Index.`
  - Fix: `(1810, 'PHEGOR', 'eb_4th_1810_v17_PAR-PHL', [('PELLANDRIUM', r'PELLANDRIUM,\s+water\-hemlock;\s+a', 28)])`

- 🔴 **BRITAIN** → **ALL** (1810) sim=0.198 [topic_change] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...of want of economy throughout the whole American department.`
  - `All this time the violent animosities between the parties continued; the desire of peace was gradual`
  - Fix: `(1810, 'BRITAIN', 'eb_4th_1810_v04_BRE-BUR', [('ALL', r'All\s+this\s+time', 34)])`

- 🔴 **SIGN** → **NAVAL SIGNALS** (1797) sim=0.202 [new_headword] [gap: EDITORIAL] (3 eds: 1797, 1815, 1823)
  - `...containing a 12th part of the zodiac. See ASTRONOMY, no 318.`
  - `NAVAL SIGNALS. When we read at our fireside the account of an engagement, or other interesting opera`
  - Fix: `(1797, 'SIGN', 'eb_3rd_1797_v17_TRE-STR', [('NAVAL SIGNALS', r'NAVAL\s+SIGNALS\.\s+When', 6)])`

- 🔴 **SIGN** → **NAVAL SIGNALS** (1815) sim=0.210 [new_headword] [gap: VARIANT] (3 eds: 1797, 1815, 1823)
  - `...n containing a 12th part of the zodiac. See ASTRONOMY Index.`
  - `NAVAL SIGNALS. When we read at our fireside the account of an engagement, or other interesting opera`
  - Fix: `(1815, 'SIGN', 'eb_5th_1815_v19_SCR-DVI', [('NAVAL SIGNALS', r'NAVAL\s+SIGNALS\.\s+When', 0)])`

- 🔴 **GENUS LXXXIII** → **ORDER III** (1823) sim=0.220 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1823)
  - `...ses, bolsters, and proper supports. See the article Surgery.`
  - `ORDER III. IMPETIGINES.

Impetigines, Sauv. Class X. Ord. V. Sag. Class III. Ord. V.`
  - Fix: `(1823, 'GENUS LXXXIII', 'eb_6th_1823_v13_ENL-MIC', [('ORDER III', r'ORDER\s+III\.\s+IMPETIGINES\.', 35)])`

- 🔴 **GENUS LXXXIII** → **ORDER III** (1810) sim=0.225 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1823)
  - `...ses, bolsters, and proper supports. See the article Surgery.`
  - `ORDER III. IMPETIGINES.

Impetigines, Sauv. Clas X. Ord. V. Sag. Clas III. Ord. V.`
  - Fix: `(1810, 'GENUS LXXXIII', 'eb_4th_1810_v13_GEN-MIC', [('ORDER III', r'ORDER\s+III\.\s+IMPETIGINES\.', 78)])`

- 🔴 **MEDICINE** → **ORDER III** (1778) sim=0.235 [new_headword] [gap: OCR_GAP] (2 eds: 1778, 1815)
  - `...ses, bolsters, and proper supports. See the article Surgery.`
  - `ORDER III. IMPETIGINES.

Impetigines, Sauv. Cl. X. Ord. V. Sag. Cl. III. Ord. V.`
  - Fix: `(1778, 'MEDICINE', 'eb_2nd_1778_v06_BYW-IND', [('ORDER III', r'ORDER\s+III\.\s+IMPETIGINES\.', 75)])`

- 🔴 **BARON** → **ROBERT** (1842) sim=0.245 [new_headword] [gap: PARSING_OR_EDITORIAL] (4 eds: 1810, 1815, 1823, 1842)
  - `...t must be borne by the husband on an escutcheon of pretence.`
  - `ROBERT**, a dramatic author, who lived during the reign of Charles I. and the protectorship of Olive`
  - Fix: `(1842, 'BARON', 'eb_7th_1842_v04_SEV-BOR', [('ROBERT', r'ROBERT\*\*,\s+a\s+dramatic', 76)])`

- 🔴 **WILSON** → **THOMAS** (1810) sim=0.261 [new_headword] [gap: OCR_GAP] (2 eds: 1810, 1823)
  - `... Selass Gryph. 3. Philosophiae Ari- fici Synagogis, lib. iv.`
  - `THOMAS, lord bishop of Sodor and Man, was born in 1663, at Barton in the county of Chester. He recei`
  - Fix: `(1810, 'WILSON', 'eb_4th_1810_v20_SUI-PRE', [('THOMAS', r'THOMAS,\s+lord\s+bishop', 6)])`

- 🔴 **WILSON** → **THOMAS** (1823) sim=0.270 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1823)
  - `...er Sebast. Gryph. 3. Philosophic Aristot. Synopsis, lib. iv.`
  - `THOMAS, lord bishop of Sodor and Man, was born in 1663, at Burton, in the county of Chester. He rece`
  - Fix: `(1823, 'WILSON', 'eb_6th_1823_v20_ENL-ZYG', [('THOMAS', r'THOMAS,\s+lord\s+bishop', 6)])`

- 🔴 **MEDICINE** → **ORDER II** (1778) sim=0.273 [new_headword] [gap: OCR_GAP] (2 eds: 1778, 1815)
  - `...d with, that the tabes dorsalis almost always proves mortal.`
  - `ORDER II. INTUMESCENTIÆ.

Intumescentiae, Sauv. Clas X. Ord. II. Sag. Clas III. Ord. II.
Tumidofi, L`
  - Fix: `(1778, 'MEDICINE', 'eb_2nd_1778_v06_BYW-IND', [('ORDER II', r'ORDER\s+II\.\s+INTUMESCENTIÆ\.', 72)])`

- 🔴 **ASTRONOMY** → **HALLEY** (1823) sim=0.303 [new_headword] [gap: PARSING_OR_EDITORIAL] (3 eds: 1797, 1815, 1823)
  - `...ncessit ocumen.
Nec fax est propius mortali attingeré divos.`
  - `HALLEY.

Sect. X. Of the Libration of the Moon.`
  - Fix: `(1823, 'ASTRONOMY', 'eb_6th_1823_v03_ENL-BOO', [('HALLEY', r'HALLEY\.\s+Sect\.\s+X\.', 79)])`

- 🔴 **ASTRONOMY** → **HALLEY** (1815) sim=0.307 [new_headword] [gap: PARSING_OR_EDITORIAL] (3 eds: 1797, 1815, 1823)
  - `...cepsit acumen.
Nec fas est proprius mortali attingere divos.`
  - `HALLEY.

SECT. X. Of the Libration of the Moon.`
  - Fix: `(1815, 'ASTRONOMY', 'eb_5th_1815_v03_ASS-DIR', [('HALLEY', r'HALLEY\.\s+SECT\.\s+X\.', 87)])`

- 🔴 **HERALDRY** → **ART** (1815) sim=0.308 [new_headword] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...ngton. On June 11, 1720, he was created Vifcount Barrington.`
  - `ART. 6. Of the Cross.

The Crois is an ordinary formed by the meeting of two perpendicular with two `
  - Fix: `(1815, 'HERALDRY', 'eb_5th_1815_v10_GOT-HYD', [('ART', r'ART\.\s+6\.\s+Of', 29)])`

- 🔴 **MEDICINE** → **ORDER IV** (1778) sim=0.319 [new_headword] [gap: OCR_GAP] (2 eds: 1778, 1815)
  - `...ity hath been found to perform surprising cures in this way.`
  - `ORDER IV. APOCENOSES.

Apocenooses, Vog. Clas. II. Ord. II.
Fluxus, Sauv. Clas. IX. Sag. Clas. V.
Mo`
  - Fix: `(1778, 'MEDICINE', 'eb_2nd_1778_v06_BYW-IND', [('ORDER IV', r'ORDER\s+IV\.\s+APOCENOSES\.', 82)])`

- 🔴 **HERALDRY** → **ART** (1810) sim=0.319 [new_headword] [gap: OCR_GAP] (3 eds: 1810, 1815, 1823)
  - `...ngton. On June 11, 1720, he was created Viscount Barrington.`
  - `ART. 6. Of the Cross.

The Cross is an ordinary formed by the meeting of two perpendicular with two `
  - Fix: `(1810, 'HERALDRY', 'eb_4th_1810_v10_HER-HYD', [('ART', r'ART\.\s+6\.\s+Of', 21)])`

- 🔴 **PUTTY SOMETIMES ALSO** → **TERRA PUZZULANA** (1810) sim=0.328 [new_headword] [gap: EDITORIAL] (2 eds: 1810, 1815)
  - `...ishing and giving the last glo's to works of iron and steel.`
  - `TERRA PUZZULANA, or Pozzolana, is a grayish kind of earth used in Italy for building under water. Th`
  - Fix: `(1810, 'PUTTY SOMETIMES ALSO', 'eb_4th_1810_v17_PRO-RHI', [('TERRA PUZZULANA', r'TERRA\s+PUZZULANA,\s+or', 0)])`

- 🔴 **BRASS** → **ORDER III** (1810) sim=0.341 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1810, 1815)
  - `...xir ex aloe et rheo, the pilula stomachica, and some others.`
  - `ORDER III. HEXAGYNIA.

804. BUTOMUS, or Flowering-rush.
One species; viz. umbellatus.`
  - Fix: `(1810, 'BRASS', 'eb_4th_1810_v04_BOO-BRE', [('ORDER III', r'ORDER\s+III\.\s+HEXAGYNIA\.', 6619)])`

- 🔴 **NAVEW** → **THEORY OF NAVIGATION** (1823) sim=0.345 [new_headword] [gap: PARSING_OR_EDITORIAL] (2 eds: 1815, 1823)
  - `...not only of individual utility, but of national importance."`
  - `THEORY OF NAVIGATION.

THE motion of a ship in the water is well known to depend on the action of th`
  - Fix: `(1823, 'NAVEW', 'eb_6th_1823_v14_ENL-NIC', [('THEORY OF NAVIGATION', r'THEORY\s+OF\s+NAVIGATION\.', 74)])`

## MEDIUM Confidence (682 fixes)

- 🟢 **SLEIDAN** → **SLEIGHT** (1823) sim=-0.001 [new_headword] (2 eds: 1815, 1823)
  - `...ibri tres*; with some other historical and political pieces.`
  - `SLEIGHT of HAND. See Legerdemain.`
  - Fix: `(1823, 'SLEIDAN', 'eb_6th_1823_v19_ENL-SUG', [('SLEIGHT', r'SLEIGHT\s+of\s+HAND\.', 0)])`

- 🟢 **ARISTOXENUS** → **ARISTOTLE** (1842) sim=0.002 [topic_change] [gap: OCR_GAP]
  - `...\\
+ & 8 & 7 & 6 \\
\hline
& 1 & 2 & 3 & 6 \\
\end{array}
\]`
  - `Aristotle has no distinct treatise on the several heads of classification, or classes of Predicables`
  - Fix: `(1842, 'ARISTOXENUS', 'eb_7th_1842_v03_SEV-AST', [('ARISTOTLE', r'Aristotle\s+has\s+no', 43)])`

- 🟢 **SLEIDAN** → **SLEIGHT** (1815) sim=0.012 [new_headword] (2 eds: 1815, 1823)
  - `...libri tres; with some other historical and political pieces.`
  - `SLEIGHT of HAND. See LEGERDEMAIN.`
  - Fix: `(1815, 'SLEIDAN', 'eb_5th_1815_v19_SCR-DVI', [('SLEIGHT', r'SLEIGHT\s+of\s+HAND\.', 0)])`

- 🟢 **BELLADONA** → **BELLAI** (1810) sim=0.013 [person_bio] [gap: EDITORIAL]
  - `...ivial name of a species of Atropa. See Atropa, Botany Index.`
  - `Bellai, William du, lord of Langey, a French general,
Bellari, general, who signalized himself in th`
  - Fix: `(1810, 'BELLADONA', 'eb_4th_1810_v03_BAR-BOO', [('BELLAI', r'Bellai,\s+William\s+du,', 0)])`

- 🟢 **FLINT** → **FLINTS** (1797) sim=0.020 [new_headword] (2 eds: 1797, 1815)
  - `...standing received its direction from a supernatural impulse.`
  - `FLINTS, in the glass trade. The way of preparing flints for the nicest operations in the glass trade`
  - Fix: `(1797, 'FLINT', 'eb_3rd_1797_v07_TRE-GOA', [('FLINTS', r'FLINTS,\s+in\s+the', 28)])`

- 🟢 **FLINT** → **FLINTS** (1815) sim=0.058 [new_headword] (2 eds: 1797, 1815)
  - `...standing received its direction from a supernatural impulse.`
  - `FLINTS, in the glass trade. The way of preparing flints for the nicest operations in the glass trade`
  - Fix: `(1815, 'FLINT', 'eb_5th_1815_v08_ENL-FOR', [('FLINTS', r'FLINTS,\s+in\s+the', 37)])`

- 🟢 **NERIUM** → **NERO** (1810) sim=0.070 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...nder the 30th order, Contortae. See Botany and Dyeing Index.`
  - `Nero, Claudius Domitius Caesar, a celebrated Roman emperor, son of Caius Domitius Ahenobarbus and Ag`
  - Fix: `(1810, 'NERIUM', 'eb_4th_1810_v14_MOR-NIA', [('NERO', r'Nero,\s+Claudius\s+Domitius', 0)])`

- 🟢 **POLYXO** → **POMACE** (1797) sim=0.075 [new_headword] (2 eds: 1797, 1810)
  - `...ee by her female servants, disguised in the habit of Furies.`
  - `POMACEÆ, (pomum "an apple,") the name of the 36th order in Linnaeus's Fragments of a Natural Method,`
  - Fix: `(1797, 'POLYXO', 'eb_3rd_1797_v15_IND-RAN', [('POMACE', r'POMACEÆ,\s+\(pomum\s+"an', 0)])`

- 🟢 **PORTUGAL** → **PORTO-V** (1815) sim=0.075 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...er
Portugal, the emperor from marching directly towards him.`
  - `PORTO-Vecchio, is a sea-port town of Corsica, in the Mediterranean Sea, seated on a bay on the easte`
  - Fix: `(1815, 'PORTUGAL', 'eb_5th_1815_v17_ENL-RHI', [('PORTO-V', r'PORTO\-Vecchio,\s+is\s+a', 5)])`

- 🟢 **PORTUGAL** → **PORTO** (1823) sim=0.077 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hildren, and procured them settlements in his own dominions.`
  - `Porto-Venero, is a town of Italy, on the coast of Genoa, at the entrance of the gulf of Spezia. It i`
  - Fix: `(1823, 'PORTUGAL', 'eb_6th_1823_v17_ENL-RHI', [('PORTO', r'Porto\-Venero,\s+is\s+a', 8)])`

- 🟢 **PORTUGAL** → **PORTO-S** (1815) sim=0.080 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ch situation she continued during the remainder of her life.`
  - `PORTO-Seguro, a government of South America, on the eastern coast of Brazil; bounded on the north by`
  - Fix: `(1815, 'PORTUGAL', 'eb_5th_1815_v17_ENL-RHI', [('PORTO-S', r'PORTO\-Seguro,\s+a\s+government', 4)])`

- 🟢 **POLYXO** → **POMACE** (1810) sim=0.084 [new_headword] (2 eds: 1797, 1810)
  - `...ee by her female servants, disguised in the habit of Furies.`
  - `POMACEÆ, (pomum "an apple,") the name of the 36th order in Linnaeus's Fragments of a Natural Method,`
  - Fix: `(1810, 'POLYXO', 'eb_4th_1810_v16_POE-BC', [('POMACE', r'POMACEÆ,\s+\(pomum\s+"an', 0)])`

- 🟢 **CHRISTIANA** → **CHRISTOPHER** (1815) sim=0.103 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...d by the people, who were much more frightened than herself.`
  - `Christopher's, St., one of the Caribbee islands, in America, lying on the north-west of Nevis, and a`
  - Fix: `(1815, 'CHRISTIANA', 'eb_5th_1815_v06_ENL-CRY', [('CHRISTOPHER', r'Christopher's,\s+St\.,\s+one', 0)])`

- 🟢 **TERPANDER** → **TERRA AUSTRALIS INC** (1810) sim=0.108 [new_headword] [gap: EDITORIAL]
  - `..." Of the works of this poet only a few fragments now remain.`
  - `TERRA AUSTRALIS INCognita, a name for a large unknown continent, supposed to lie towards the south p`
  - Fix: `(1810, 'TERPANDER', 'eb_4th_1810_v20_SUI-PRE', [('TERRA AUSTRALIS INC', r'TERRA\s+AUSTRALIS\s+INCognita,', 55)])`

- 🟢 **IGNATIA** → **IGNATIUS LOYOLA** (1842) sim=0.110 [new_headword] (2 eds: 1797, 1842)
  - `... plants belonging to the pentandra class. See Botany, Index.`
  - `IGNATIUS LOYOLA, the founder of the order of Jesuits, was born at the castle of Loyola, in Biscay, i`
  - Fix: `(1842, 'IGNATIA', 'eb_7th_1842_v12_DEF-PLA', [('IGNATIUS LOYOLA', r'IGNATIUS\s+LOYOLA,\s+the', 0)])`

- 🟢 **ANGAZYA** → **ANGIOTOMY** (1797) sim=0.114 [new_headword] (2 eds: 1797, 1810)
  - `...ozambique, whether they trade in vessels of 40 tons burthen.`
  - `ANGIOTOMY, in surgery, implies the opening a vein or artery, as in bleeding; and consequently includ`
  - Fix: `(1797, 'ANGAZYA', 'eb_3rd_1797_v01_IND-COR', [('ANGIOTOMY', r'ANGIOTOMY,\s+in\s+surgery,', 0)])`

- 🟢 **TRILL** → **TRIM** (1842) sim=0.117 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ult the Methode de Chant of the Paris Conservatory of Music.`
  - `TRIM is the county town of the county of Meath in Ireland. It is situate upon the banks of the river`
  - Fix: `(1842, 'TRILL', 'eb_7th_1842_v21_SEV-ZYG', [('TRIM', r'TRIM\s+is\s+the', 0)])`

- 🟢 **ALLANTOIS** → **ALLATIUS** (1815) sim=0.118 [person_bio] [gap: OCR_GAP]
  - `...he young animals by means of the urachus. See Anatomy Index.`
  - `Allatius, Leo, keeper of the Vatican library, a native of Scio, and a celebrated writer of the 17th `
  - Fix: `(1815, 'ALLANTOIS', 'eb_5th_1815_v01_ENL-AME', [('ALLATIUS', r'Allatius,\s+Leo,\s+keeper', 0)])`

- 🟢 **ANGAZYA** → **ANGIOTOMY** (1810) sim=0.123 [new_headword] (2 eds: 1797, 1810)
  - `...ozambique, whither they trade in vessels of 40 tons burthen.`
  - `ANGIOTOMY, in Surgery, implies the opening a vein or artery, as in bleeding; and consequently includ`
  - Fix: `(1810, 'ANGAZYA', 'eb_4th_1810_v17_ART-ANS', [('ANGIOTOMY', r'ANGIOTOMY,\s+in\s+Surgery,', 0)])`

- 🟢 **STRAPADO** → **STRA** (1842) sim=0.126 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...cated. Sometimes he has to undergo three strapadoes or more.`
  - `STRAßBOURG, an arrondissement of the department of the Lower Rhine, in France. It is a rich district`
  - Fix: `(1842, 'STRAPADO', 'eb_7th_1842_v20_SEV-SUG', [('STRA', r'STRAßBOURG,\s+an\s+arrondissement', 0)])`

- 🟢 **HAND** → **HANDEL** (1810) sim=0.132 [topic_change] [gap: VARIANT]
  - `...an, making the extremity of the arm. See ANATOMY, n° 53, &c.`
  - `Handel, though yet but in his 15th year, became composer to the house; and the success of Almira, hi`
  - Fix: `(1810, 'HAND', 'eb_4th_1810_v05_GOT-HER', [('HANDEL', r'Handel,\s+though\s+yet', 0)])`

- 🟢 **ARCHYTAS** → **ARCHYTAS** (1842) sim=0.137 [new_headword] [gap: VARIANT]
  - `...the functions and privileges of chorepiscopi or rural deans.`
  - `ARCHYTAS of Tarentum was a Pythagorean philosopher, well skilled in mathematics and geography. He li`
  - Fix: `(1842, 'ARCHYTAS', 'eb_7th_1842_v03_SEV-AST', [('ARCHYTAS', r'ARCHYTAS\s+of\s+Tarentum', 0)])`

- 🟢 **PHILIPSBURG** → **PHILIST** (1797) sim=0.139 [new_headword] [gap: EDITORIAL]
  - `...49 north-east of Strasbourg. E. Long. 8. 33. N. Lat. 49. 12.`
  - `PHILISTÆA (anc. geog.), the country of the Philistines (Bible); which lay along the Mediterranean, f`
  - Fix: `(1797, 'PHILIPSBURG', 'eb_3rd_1797_v14_TRE-PLA', [('PHILIST', r'PHILISTÆA\s+\(anc\.\s+geog\.\),', 0)])`

- 🟢 **GRAVESANDE** → **GRAVEL** (1823) sim=0.141 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ers of nature with more success, or to more useful purposes.`
  - `Gravel with some loam among it, binds more firmly than the rawer kinds; and when gravel is naturally`
  - Fix: `(1823, 'GRAVESANDE', 'eb_6th_1823_v10_ENL-HYD', [('GRAVEL', r'Gravel\s+with\s+some', 0)])`

- 🟢 **MOPSUS** → **MOR** (1815) sim=0.142 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...n of Manto, as their professions and their names were alike.`
  - `MORÆA, a genus of plants belonging to the triandria clas; and in the natural method ranking under th`
  - Fix: `(1815, 'MOPSUS', 'eb_5th_1815_v14_ENL-NIC', [('MOR', r'MORÆA,\s+a\s+genus', 0)])`

- 🟢 **BELLADONA** → **BELLAC** (1810) sim=0.145 [topic_change] [gap: EDITORIAL]
  - `...rch of Mans, and a noble monument was erected to his memory.`
  - `Bellac. See Belac.`
  - Fix: `(1810, 'BELLADONA', 'eb_4th_1810_v03_BAR-BOO', [('BELLAC', r'Bellac\.\s+See\s+Belac\.', 28)])`

- 🟢 **CHRISTMAS-DAY** → **CHRISTOPHER** (1797) sim=0.150 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hich, in a short time, reduced them and the church to ashes.`
  - `Christopher's, St. one of the Caribbee islands, in America, lying to the north-west of Nevis, and ab`
  - Fix: `(1797, 'CHRISTMAS-DAY', 'eb_3rd_1797_v04_TRE-OMI', [('CHRISTOPHER', r'Christopher's,\s+St\.\s+one', 0)])`

- 🟢 **PIABUCU** → **PIACENZA** (1823) sim=0.150 [new_headword] (3 eds: 1797, 1810, 1823)
  - `... to suck the blood. It seldom exceeds four inches in length.`
  - `PIACENZA is a city of Italy, in the duchy of Parma, in E. Long. 12° 25'. N. Lat. 45°. It is a large `
  - Fix: `(1823, 'PIABUCU', 'eb_6th_1823_v16_ENL-BRE', [('PIACENZA', r'PIACENZA\s+is\s+a', 0)])`

- 🟢 **BRANCHON** → **BRAND S** (1823) sim=0.151 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...ur, seated on the river Meuse. E. Long. 4°. N. Lat. 50°. 32.`
  - `BRAND Sunday, Dimanche des Brandons, in French ecclesiastical writers, denotes the first Sunday in L`
  - Fix: `(1823, 'BRANCHON', 'eb_6th_1823_v04_ENL-BUR', [('BRAND S', r'BRAND\s+Sunday,\s+Dimanche', 0)])`

- 🟢 **BRANCHON** → **BRAND S** (1815) sim=0.152 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... seated on the river Meuse. E. Long. 4° 40' N. Lat. 50° 32'.`
  - `BRAND Sunday, Dimanche des Brandons, in French ecclesiastical writers, denotes the first Sunday in L`
  - Fix: `(1815, 'BRANCHON', 'eb_5th_1815_v04_ENL-BUR', [('BRAND S', r'BRAND\s+Sunday,\s+Dimanche', 0)])`

- 🟢 **MACCABEES** → **MACE** (1797) sim=0.153 [topic_change] [gap: VARIANT]
  - `...aving been confirmed by repeated and undeniable observation.`
  - `Mace is carminative, stomachic, and astringent; and possesses all the virtues of nutmeg, but has les`
  - Fix: `(1797, 'MACCABEES', 'eb_3rd_1797_v10_IND-MEC', [('MACE', r'Mace\s+is\s+carminative,', 7)])`

- 🟢 **WORMS** → **WORMIUS** (1810) sim=0.156 [new_headword] (2 eds: 1797, 1810)
  - `...a by which the dreadful disease hydrophobia is communicated.`
  - `WORMIUS, OLAUS, a learned Danish physician, born in 1688 at Arhusen in Jutland. After beginning his `
  - Fix: `(1810, 'WORMS', 'eb_4th_1810_v20_SUI-PRE', [('WORMIUS', r'WORMIUS,\s+OLAUS,\s+a', 61)])`

- 🟢 **BUCEPHALA** → **BUCE** (1842) sim=0.156 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...India Citerior, so called in memory of his horse Bucephalus.`
  - `BUCE, MARTIN, one of the first authors of the Reformation at Strasburg, was born in 1491, in Alsace,`
  - Fix: `(1842, 'BUCEPHALA', 'eb_7th_1842_v05_BOR-CAL', [('BUCE', r'BUCE,\s+MARTIN,\s+one', 0)])`

- 🟢 **BRANCHON** → **BRAND S** (1810) sim=0.160 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...seated on the river Meuse. E. Long. 4° 40'. N. Lat. 50° 32'.`
  - `BRAND Sunday, Dimanche des Brandons, in French ecclesiastical writers, denotes the first Sunday in L`
  - Fix: `(1810, 'BRANCHON', 'eb_4th_1810_v04_BOO-BRE', [('BRAND S', r'BRAND\s+Sunday,\s+Dimanche', 0)])`

- 🟢 **HENRY** → **HENRY** (1823) sim=0.162 [new_headword] (2 eds: 1823, 1842)
  - `...on the battle of Bannockburn has been preserved to this day.`
  - `HENRY of Susa, in Latin de Segusio, a famous civilian and canonist of the 13th century, acquired suc`
  - Fix: `(1823, 'HENRY', 'eb_6th_1823_v10_ENL-HYD', [('HENRY', r'HENRY\s+of\s+Susa,', 0)])`

- 🟢 **PIABUCU** → **PIACENZA** (1810) sim=0.165 [new_headword] (3 eds: 1797, 1810, 1823)
  - `... to suck the blood. It seldom exceeds four inches in length.`
  - `PIACENZA is a city of Italy, in the duchy of Parma, in E. Long. 10° 25'. N. Lat. 45°. It is a large `
  - Fix: `(1810, 'PIABUCU', 'eb_4th_1810_v16_PHL-HOR', [('PIACENZA', r'PIACENZA\s+is\s+a', 0)])`

- 🟢 **PIABUCU** → **PIACENZA** (1797) sim=0.169 [new_headword] (3 eds: 1797, 1810, 1823)
  - `... to suck the blood. It seldom exceeds four inches in length.`
  - `PIACENZA is a city of Italy, in the duchy of Parma, in E. Long. 10. 25. N. Lat. 45. It is a large ha`
  - Fix: `(1797, 'PIABUCU', 'eb_3rd_1797_v14_TRE-PLA', [('PIACENZA', r'PIACENZA\s+is\s+a', 0)])`

- 🟢 **MARSICONUOVO** → **MARSIGLI** (1842) sim=0.177 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `... a cathedral and five other churches, with 6790 inhabitants.`
  - `Marsigli, Louis Ferdinand Count de, an Italian geographer and naturalist, was descended of an ancien`
  - Fix: `(1842, 'MARSICONUOVO', 'eb_7th_1842_v14_SEV-MEX', [('MARSIGLI', r'Marsigli,\s+Louis\s+Ferdinand', 0)])`

- 🟢 **ARCHTREASURER** → **ARCHYTAS** (1778) sim=0.178 [topic_change] [gap: VARIANT]
  - `...y belonging to the duke of Brunswick, king of Great Britain.`
  - `**ARCHYTAS** of Tarentum, a philosopher of the Pythagorean sect, and famous for being the master of `
  - Fix: `(1778, 'ARCHTREASURER', 'eb_2nd_1778_v01_AA-AND', [('ARCHYTAS', r'\*\*ARCHYTAS\*\*\s+of\s+Tarentum,', 0)])`

- 🟢 **TELL** → **TELL-T** (1815) sim=0.180 [new_headword] (2 eds: 1810, 1815)
  - `...he association for the independence took place that instant.`
  - `TELL-Tale, a name sometimes given to the Perpetual-LOG. See that article.`
  - Fix: `(1815, 'TELL', 'eb_5th_1815_v20_SUI-DIR', [('TELL-T', r'TELL\-Tale,\s+a\s+name', 0)])`

- 🟢 **PALAMEDES** → **PAL** (1797) sim=0.186 [new_headword] (2 eds: 1797, 1823)
  - `...ed their vigilance and attention by giving them a watchword.`
  - `PALÆSTROPHYLAX, was the director of the palaestra, and the exercises performed there.or Palambang, a`
  - Fix: `(1797, 'PALAMEDES', 'eb_3rd_1797_v13_TRE-PAS', [('PAL', r'PALÆSTROPHYLAX,\s+was\s+the', 55)])`

- 🟢 **TELL** → **TELL-T** (1810) sim=0.190 [new_headword] (2 eds: 1810, 1815)
  - `...he association for the independence took place that instant.`
  - `TELL-Tale, a name sometimes given to the Perpetual Log. See that article.`
  - Fix: `(1810, 'TELL', 'eb_4th_1810_v20_SUI-PRE', [('TELL-T', r'TELL\-Tale,\s+a\s+name', 0)])`

- 🟢 **CHRISOM** → **CHRISOM** (1797) sim=0.191 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...st of this series, the fen, amounts to no more than 0 grams.`
  - `CHRISOM was not, as is said in the Encyclopedia, a face-cloth or piece of linen laid over the child'`
  - Fix: `(1797, 'CHRISOM', 'eb_3rd_1797_v501_ABE-IMP', [('CHRISOM', r'CHRISOM\s+was\s+not,', 6)])`

- 🟢 **ALEXANDER THE GREAT** → **ALEXANDER AB ALEXANDRO** (1823) sim=0.194 [new_headword] (2 eds: 1823, 1842)
  - `...rth all the heroes that ever did or will exist. See MACEDON.`
  - `ALEXANDER AB ALEXANDRO, a Neapolitan lawyer, of great learning, who flourished toward the end of the`
  - Fix: `(1823, 'ALEXANDER THE GREAT', 'eb_6th_1823_v01_ART-AME', [('ALEXANDER AB ALEXANDRO', r'ALEXANDER\s+AB\s+ALEXANDRO,', 95)])`

- 🟢 **TROUBADOURS** → **TROUGH** (1823) sim=0.195 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...e Sainte Palaye, and finished by the abbé Millot. See Music.`
  - `TROUGH, GALVANIC. See GALVANISM. For later discoveries in galvanic electricity, see ZINC.`
  - Fix: `(1823, 'TROUBADOURS', 'eb_6th_1823_v20_ENL-ZYG', [('TROUGH', r'TROUGH,\s+GALVANIC\.\s+See', 0)])`

- 🟢 **MORANT** → **MORANT-P** (1797) sim=0.196 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...upon Thomas Aylett, Esq.; who had married his only daughter.`
  - `MORANT-Point, the most easterly point or promontory of the island of Jamaica, in America. W. Lon. 75`
  - Fix: `(1797, 'MORANT', 'eb_3rd_1797_v12_TRE-NEG', [('MORANT-P', r'MORANT\-Point,\s+the\s+most', 0)])`

- 🟢 **CHRISTIANA** → **CHRISTINA** (1823) sim=0.196 [topic_change] [gap: VARIANT]
  - `...between the persons thus consecrated and their subordinates.`
  - `Christina, daughter of Gustavus Adolphus king of Sweden, was born in 1626; and succeeded to the crow`
  - Fix: `(1823, 'CHRISTIANA', 'eb_6th_1823_v06_ENL-CRY', [('CHRISTINA', r'Christina,\s+daughter\s+of', 0)])`

- 🟢 **GREEN-HOUSE** → **GREENOCK** (1823) sim=0.197 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...rge and beautiful structures, equally ornamental and useful.`
  - `Greenock, till lately, was divided into what are called the old and new parishes. Certain lands disj`
  - Fix: `(1823, 'GREEN-HOUSE', 'eb_6th_1823_v10_ENL-HYD', [('GREENOCK', r'Greenock,\s+till\s+lately,', 0)])`

- 🟢 **JEARS** → **JEBUS** (1823) sim=0.199 [new_headword] (2 eds: 1815, 1823)
  - `...which operations is called swaying, and the latter striking.`
  - `JEBUSÆI, one of the seven ancient peoples of Canaan, descendants of Jebus, Canaan's son; so warlike `
  - Fix: `(1823, 'JEARS', 'eb_6th_1823_v11_ENL-LIE', [('JEBUS', r'JEBUSÆI,\s+one\s+of', 0)])`

- 🟢 **CANN** → **CANN** (1823) sim=0.200 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...abandoned entirely before the end of the thirteenth century.`
  - `CANNÉQUINS, in commerce, white cotton cloths brought from the East Indies. They are a proper commodi`
  - Fix: `(1823, 'CANN', 'eb_6th_1823_v05_ENL-CHI', [('CANN', r'CANNÉQUINS,\s+in\s+commerce,', 111)])`

- 🟢 **JEARS** → **JEBUS** (1815) sim=0.201 [new_headword] (2 eds: 1815, 1823)
  - `...which operations is called swaying, and the latter striking.`
  - `JEBUSÆI, one of the seven ancient peoples of Canaan, descendants of Jebuhi, Canaan's son; so warlike`
  - Fix: `(1815, 'JEARS', 'eb_5th_1815_v11_ENL-LIE', [('JEBUS', r'JEBUSÆI,\s+one\s+of', 0)])`

- 🟢 **BALLAN** → **BALLAD** (1810) sim=0.205 [new_headword] (2 eds: 1797, 1810)
  - `... seated on the river Orne. E. Long. 0° 25'. N. Lat. 48° 10'.`
  - `BALLAD, a kind of song, adapted to the capacity of the lower class of people; who, being mightily ta`
  - Fix: `(1810, 'BALLAN', 'eb_4th_1810_v03_ASS-BAR', [('BALLAD', r'BALLAD,\s+a\s+kind', 0)])`

- 🟢 **TROUBADOURS** → **TROUGH** (1810) sim=0.209 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...e Sainte Palaie, and finished by the abbé Millot. See Music.`
  - `TROUGH, GALVANIC. See GALVANISM. For later discoveries in galvanic electricity, see Zinc.`
  - Fix: `(1810, 'TROUBADOURS', 'eb_4th_1810_v20_SUI-PRE', [('TROUGH', r'TROUGH,\s+GALVANIC\.\s+See', 0)])`

- 🟢 **GAUGAMELA** → **GAUGE-** (1823) sim=0.211 [new_headword] (2 eds: 1815, 1823)
  - `... whence the latter gave the name to the victory. See ARBELA.`
  - `GAUGE-point of a solid measure, the diameter of a circle whose area is equal to the solid content of`
  - Fix: `(1823, 'GAUGAMELA', 'eb_6th_1823_v09_FOR-DIR', [('GAUGE-', r'GAUGE\-point\s+of\s+a', 0)])`

- 🟢 **TROUBADOURS** → **TROUGH** (1815) sim=0.216 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...de Sainte Palae, and finished by the abbe Millet. See MUSIC.`
  - `TROUGH, GALVANIC. See GALVANISM. For later discoveries in galvanic electricity, see ZINC.`
  - Fix: `(1815, 'TROUBADOURS', 'eb_5th_1815_v20_SUI-DIR', [('TROUGH', r'TROUGH,\s+GALVANIC\.\s+See', 0)])`

- 🟢 **CHROSTASIMA** → **CHRYSIA** (1823) sim=0.216 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `..., beryl, emerald, and the topaz. See DIAMOND, CARBUNCLE, &c.`
  - `CHRYSIA, in *Ancient Geography*, a town of Mysia, on the sinus Adramyttium; extinct in Pliny's time:`
  - Fix: `(1823, 'CHROSTASIMA', 'eb_6th_1823_v06_ENL-CRY', [('CHRYSIA', r'CHRYSIA,\s+in\s+\*Ancient', 0)])`

- 🟢 **AGRIPPA** → **AGRIGENTUM** (1823) sim=0.217 [new_headword] [gap: OCR_GAP]
  - `...He adorned the city with the Pantheon, baths, aqueducts, &c.`
  - `AGRIGENTUM, in Ancient Geography, a city of Sicily, part of the site of which is now occupied by a t`
  - Fix: `(1823, 'AGRIPPA', 'eb_6th_1823_v01_ART-AME', [('AGRIGENTUM', r'AGRIGENTUM,\s+in\s+Ancient', 9)])`

- 🟢 **PHILOXENUS** → **PHILITER** (1842) sim=0.217 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...reat admirer of the dithyrambics of Philoxenus and Telestes.`
  - `PHILITER, or PHILTRE (philtrum), in Pharmacy or Chemistry, a strainer. This term is also used for a `
  - Fix: `(1842, 'PHILOXENUS', 'eb_7th_1842_v17_SEV-CON', [('PHILITER', r'PHILITER,\s+or\s+PHILTRE', 0)])`

- 🟢 **ANTIOCHETTA** → **ANTIOCHUS** (1778) sim=0.218 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ainst the island of Cyprus. E. Long. 32. 15. N. Lat. 36. 42.`
  - `ANTIOCHUS the Great, king of Syria, succeeded his brother Seleucus Ceraunus, 223 years before Christ`
  - Fix: `(1778, 'ANTIOCHETTA', 'eb_2nd_1778_v01_AA-AND', [('ANTIOCHUS', r'ANTIOCHUS\s+the\s+Great,', 0)])`

- 🟢 **GAUGAMELA** → **GAUGE-** (1815) sim=0.219 [new_headword] (2 eds: 1815, 1823)
  - `... whence the latter gave the name to the victory. See ARBELA.`
  - `GAUGE-point of a solid measure, the diameter of a circle whose area is equal to the solid content of`
  - Fix: `(1815, 'GAUGAMELA', 'eb_5th_1815_v09_FOR-CCX', [('GAUGE-', r'GAUGE\-point\s+of\s+a', 0)])`

- 🟢 **FRANCISCO** → **FRANK LANGUAGE** (1842) sim=0.227 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...e which considerably detracts from its value. Lat 26° 15' S.`
  - `FRANK LANGUAGE, Lingua Franca, a kind of jargon spoken on the Mediterranean, and particularly throug`
  - Fix: `(1842, 'FRANCISCO', 'eb_7th_1842_v10_SEV-GRO', [('FRANK LANGUAGE', r'FRANK\s+LANGUAGE,\s+Lingua', 0)])`

- 🟢 **PALAMEDES** → **PAL** (1823) sim=0.233 [new_headword] (2 eds: 1797, 1823)
  - `...is extravagancies. We have only some fragments of his works.`
  - `PALÆOLOGUS, MICHAEL, a very able man who was governor of Asia under the emperor Theodorus Lascaris; `
  - Fix: `(1823, 'PALAMEDES', 'eb_6th_1823_v15_ENL-PAR', [('PAL', r'PALÆOLOGUS,\s+MICHAEL,\s+a', 23)])`

- 🟢 **PRONG-HOE** → **PROPOSITION XXII** (1797) sim=0.236 [new_headword] (2 eds: 1797, 1823)
  - `...en to the very stalks of the plant. See Agriculture and Hoe.`
  - `PROPOSITION XXII. PROBLEM XIV.

Two sides of a spherical triangle, and an angle opposite to one of t`
  - Fix: `(1797, 'PRONG-HOE', 'eb_3rd_1797_v15_IND-RAN', [('PROPOSITION XXII', r'PROPOSITION\s+XXII\.\s+PROBLEM', 20)])`

- 🟢 **ATCHE** → **ATCHIEVEMENT** (1797) sim=0.237 [new_headword] (2 eds: 1797, 1810)
  - `...ed in Turkey, and worth only one-third of the English penny.`
  - `ATCHIEVEMENT, in heraldry, denotes the arms of a person or family, together with all the exterior or`
  - Fix: `(1797, 'ATCHE', 'eb_3rd_1797_v02_IND-BAR', [('ATCHIEVEMENT', r'ATCHIEVEMENT,\s+in\s+heraldry,', 0)])`

- 🟢 **DYRRACHIUM** → **DYS** (1815) sim=0.237 [new_headword] (2 eds: 1810, 1815)
  - `...he Spartans, discouraged strangers from settling among them.`
  - `DYSÆ, in Mythology, interior goddesses among the Saxons, being the messengers of the great Woden, wh`
  - Fix: `(1815, 'DYRRACHIUM', 'eb_5th_1815_v07_CUB-DIR', [('DYS', r'DYSÆ,\s+in\s+Mythology,', 0)])`

- 🟢 **AGRIOPHAGI** → **AGRIFOLIUM** (1823) sim=0.238 [new_headword] [gap: OCR_GAP]
  - `...rthur, Esq. his experiments to prevent the smut in wheat, 98`
  - `AGRIFOLIUM, or AQUIFOLIUM. See ILEX, BOTANY Index.

AGRIGAN, or island of St Francis Xavier, in Geog`
  - Fix: `(1823, 'AGRIOPHAGI', 'eb_6th_1823_v01_ART-AME', [('AGRIFOLIUM', r'AGRIFOLIUM,\s+or\s+AQUIFOLIUM\.', 68328)])`

- 🟢 **BUCCARI** → **BUCELLARI** (1842) sim=0.238 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...bitants are about 2000. Long. 14. 26. 12. E. Lat. 45. 18. N.`
  - `BUCELLARI, an order of soldiery under the Greek emperors, appointed to guard and distribute the ammu`
  - Fix: `(1842, 'BUCCARI', 'eb_7th_1842_v05_BOR-CAL', [('BUCELLARI', r'BUCELLARI,\s+an\s+order', 0)])`

- 🟢 **MYIODES DEUS** → **MYL** (1815) sim=0.239 [new_headword] (2 eds: 1810, 1815)
  - `...s in driving away the flies that infested the Olympic games.`
  - `MYLÆ, in Ancient Geography, a Greek city situated on an isthmus of a cognominal peninsula, on the no`
  - Fix: `(1815, 'MYIODES DEUS', 'eb_5th_1815_v14_ENL-NIC', [('MYL', r'MYLÆ,\s+in\s+Ancient', 0)])`

- 🟢 **MYIODES DEUS** → **MYL** (1810) sim=0.240 [new_headword] (2 eds: 1810, 1815)
  - `...s in driving away the flies that infested the Olympic games.`
  - `MYLÆ, in Ancient Geography, a Greek city situated on an isthmus of a cognominal peninsula, on the no`
  - Fix: `(1810, 'MYIODES DEUS', 'eb_4th_1810_v14_MOR-NIA', [('MYL', r'MYLÆ,\s+in\s+Ancient', 0)])`

- 🟢 **ATCHE** → **ATCHIEVEMENT** (1810) sim=0.247 [new_headword] (2 eds: 1797, 1810)
  - `...ed in Turkey, and worth only one-third of the English penny.`
  - `ATCHIEVEMENT, in Heraldry, denotes the arms of a person or family, together with all the exterior or`
  - Fix: `(1810, 'ATCHE', 'eb_4th_1810_v03_ASS-BAR', [('ATCHIEVEMENT', r'ATCHIEVEMENT,\s+in\s+Heraldry,', 0)])`

- 🟢 **DYRRACHIUM** → **DYS** (1810) sim=0.248 [new_headword] (2 eds: 1810, 1815)
  - `...he Spartans, discouraged strangers from settling among them.`
  - `DYSÆ, in Mythology, inferior goddesses among the Saxons, being the messengers of the great Woden, wh`
  - Fix: `(1810, 'DYRRACHIUM', 'eb_4th_1810_v07_STE-ELE', [('DYS', r'DYSÆ,\s+in\s+Mythology,', 0)])`

- 🟢 **FRANKFORT** → **FRANKLAND'S ISLANDS** (1842) sim=0.249 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... floated to New Orleans, from which it is distant 806 miles.`
  - `FRANKLAND'S ISLANDS, a cluster of small islands on the north-east coast of New Holland, about six mi`
  - Fix: `(1842, 'FRANKFORT', 'eb_7th_1842_v10_SEV-GRO', [('FRANKLAND'S ISLANDS', r'FRANKLAND'S\s+ISLANDS,\s+a', 0)])`

- 🟢 **CONRAD III** → **CONRAD** (1815) sim=0.251 [new_headword] (2 eds: 1810, 1815)
  - `...rmed from them their Guelphs and Gibbelins. He died in 1152.`
  - `CONRAD of Lichtenau, or Abbas Uspergusis, was author of an Universal Chronology from the creation to`
  - Fix: `(1815, 'CONRAD III', 'eb_5th_1815_v06_ENL-CRY', [('CONRAD', r'CONRAD\s+of\s+Lichtenau,', 0)])`

- 🟢 **PRONG-HOE** → **PROPOSITION XXII** (1823) sim=0.251 [new_headword] (2 eds: 1797, 1823)
  - `...ts, and many among us, have thought the only use of the hoe,`
  - `PROPOSITION XXII. PROBLEM XIV.

Two sides of a spherical triangle, and an angle opposite to one of t`
  - Fix: `(1823, 'PRONG-HOE', 'eb_6th_1823_v17_ENL-RHI', [('PROPOSITION XXII', r'PROPOSITION\s+XXII\.\s+PROBLEM', 28)])`

- 🟢 **BALLAN** → **BALLAD** (1797) sim=0.252 [new_headword] (2 eds: 1797, 1810)
  - `... seated on the river Orne. E. Long. 0° 20'. N. Lat. 48° 10'.`
  - `BALLAD, a kind of song, adapted to the capacity of the lower class of people; who, being mightily ta`
  - Fix: `(1797, 'BALLAN', 'eb_3rd_1797_v02_IND-BAR', [('BALLAD', r'BALLAD,\s+a\s+kind', 0)])`

- 🟢 **ATHESIS** → **ATHLET** (1815) sim=0.262 [new_headword] (2 eds: 1815, 1823)
  - `...t are called Athesini (Pliny). Its modern name is the Adige.`
  - `ATHLETÆ, in antiquity, persons of strength and agility, disciplined to perform in the public games. `
  - Fix: `(1815, 'ATHESIS', 'eb_5th_1815_v03_ASS-DIR', [('ATHLET', r'ATHLETÆ,\s+in\s+antiquity,', 0)])`

- 🟢 **TEST** → **TEST-A** (1810) sim=0.264 [new_headword] [gap: VARIANT]
  - `...lic bodies when melted. See CUPEL, under ORES, Reduction of.`
  - `TEST-Act, in Law, is the statute 25 Car. II. cap. 2, which directs all officers, civil and military,`
  - Fix: `(1810, 'TEST', 'eb_4th_1810_v20_SUI-PRE', [('TEST-A', r'TEST\-Act,\s+in\s+Law,', 0)])`

- 🟢 **WORK** → **WORK-H** (1810) sim=0.265 [new_headword] [gap: OCR_GAP]
  - `...everal articles, together with FORTIFICATION and PYROTECHNY.`
  - `WORK-House, a place where indigent, vagrant, and idle people, are set to work, and supplied with foo`
  - Fix: `(1810, 'WORK', 'eb_4th_1810_v20_SUI-PRE', [('WORK-H', r'WORK\-House,\s+a\s+place', 7)])`

- 🟢 **VILLENAGE** → **VILLI** (1810) sim=0.269 [new_headword] [gap: OCR_GAP]
  - `...s; but only, "to hold according to the custom of the manor."`
  - `VILLI, among botanists, a kind of down like short hair, with which lyme trees abound.`
  - Fix: `(1810, 'VILLENAGE', 'eb_4th_1810_v20_SUI-PRE', [('VILLI', r'VILLI,\s+among\s+botanists,', 75)])`

- 🟢 **BRILLIANTS** → **BRIM** (1815) sim=0.274 [new_headword] (2 eds: 1815, 1823)
  - `...a name given to diamonds of the finest cut. See DIAMOND.`
  - `BRIM denotes the outmost verge or edge, especially of round things. The brims of vessels are made to`
  - Fix: `(1815, 'BRILLIANTS', 'eb_5th_1815_v04_ENL-BUR', [('BRIM', r'BRIM\s+denotes\s+the', 0)])`

- 🟢 **BRILLIANTS** → **BRIM** (1823) sim=0.277 [new_headword] (2 eds: 1815, 1823)
  - `...a name given to diamonds of the finest cut. See DIAMOND.`
  - `BRIM denotes the outmost verge or edge, especially of round things. The brims of vessels are made to`
  - Fix: `(1823, 'BRILLIANTS', 'eb_6th_1823_v04_ENL-BUR', [('BRIM', r'BRIM\s+denotes\s+the', 0)])`

- 🟢 **CASSINI** → **CAST** (1815) sim=0.283 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...Cassumbaazar,`
  - `CAST is peculiarly used to denote a figure or small statue of bronze. See BRONZE.`
  - Fix: `(1815, 'CASSINI', 'eb_5th_1815_v05_ENL-CHI', [('CAST', r'CAST\s+is\s+peculiarly', 0)])`

- 🟢 **ALEXANDER THE GREAT** → **ALEXANDER AB ALEXANDRO** (1842) sim=0.290 [new_headword] (2 eds: 1823, 1842)
  - `...s Life, of which there is an English translation by Clayton.`
  - `ALEXANDER AB ALEXANDRO, a Neapolitan lawyer, of great learning, who flourished toward the end of the`
  - Fix: `(1842, 'ALEXANDER THE GREAT', 'eb_7th_1842_v02_AAL-DES', [('ALEXANDER AB ALEXANDRO', r'ALEXANDER\s+AB\s+ALEXANDRO,', 695)])`

- 🟢 **ACHABYTUS** → **ACH** (1810) sim=0.292 [new_headword] [gap: OCR_GAP]
  - `...ose of the Balearians, whom they far surpassed in dexterity.`
  - `ACHÆI, Achæans, the inhabitants of Achaia Propria. In Livy, the people of Greece; for the most part `
  - Fix: `(1810, 'ACHABYTUS', 'eb_4th_1810_v08_AAR-AGR', [('ACH', r'ACHÆI,\s+Achæans,\s+the', 52)])`

- 🟢 **ATHESIS** → **ATHLET** (1823) sim=0.301 [new_headword] (2 eds: 1815, 1823)
  - `...t are called Athetini (Pliny). Its modern name is the Adige.`
  - `ATHLETÆ, in antiquity, persons of strength and agility, disciplined to perform in the public games. `
  - Fix: `(1823, 'ATHESIS', 'eb_6th_1823_v03_ENL-BOO', [('ATHLET', r'ATHLETÆ,\s+in\s+antiquity,', 0)])`

- 🟢 **CONRAD III** → **CONRAD** (1810) sim=0.302 [new_headword] (2 eds: 1810, 1815)
  - `...rmed from them their Guelphs and Gibbelins. He died in 1152.`
  - `CONRAD of Lichtenau, or Abbas Uspurgensis, was author of an Universal Chronology from the creation t`
  - Fix: `(1810, 'CONRAD III', 'eb_4th_1810_v06_CON-CRY', [('CONRAD', r'CONRAD\s+of\s+Lichtenau,', 0)])`

- 🟢 **RETINUE** → **RETIROADE** (1797) sim=0.306 [new_headword] [gap: EDITORIAL]
  - `...wers of a prince or person of quality, chiefly in a journey.`
  - `RETIROADE, in fortification, a kind of retrenchment made in the body of a bastion, or other work, wh`
  - Fix: `(1797, 'RETINUE', 'eb_3rd_1797_v16_TRE-SCO', [('RETIROADE', r'RETIROADE,\s+in\s+fortification,', 0)])`

- 🟢 **WAR** → **WAR** (1810) sim=0.313 [new_headword] [gap: OCR_GAP]
  - `... large quantities to London. E. Long, o. 3. N. Lat. 51° 50'.`
  - `WAR, in Law, is to summon a person to appear in a court of justice.`
  - Fix: `(1810, 'WAR', 'eb_4th_1810_v20_SUI-PRE', [('WAR', r'WAR,\s+in\s+Law,', 228)])`

- 🟢 **ACHABYTTUS** → **ACH** (1823) sim=0.313 [new_headword] [gap: OCR_GAP]
  - `...ose of the Balearians, whom they far surpassed in dexterity.`
  - `ACHÆI, ACHÆANS, the inhabitants of Achaia Propria. In Livy, the people of Greece; for the most part `
  - Fix: `(1823, 'ACHABYTTUS', 'eb_6th_1823_v01_ART-AME', [('ACH', r'ACHÆI,\s+ACHÆANS,\s+the', 57)])`

- 🟢 **PAL** → **PAL** (1778) sim=0.315 [new_headword] (4 eds: 1778, 1810, 1815, 1842)
  - `... him with praises. We have only some fragments of his works.`
  - `PALÆPAPHOS, (Strabo, Virgil, Pliny), a town of Cyprus, where stood a temple of Venus; and an adjoini`
  - Fix: `(1778, 'PAL', 'eb_2nd_1778_v08_BYW-GRE', [('PAL', r'PALÆPAPHOS,\s+\(Strabo,\s+Virgil,', 45)])`

- 🟢 **CALENDER** → **CALENDER** (1815) sim=0.315 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...e rollers with a shallow indenture or engraving cut in them.`
  - `CALENDER of Monteith, a district in the southwest corner of Perthshire in Scotland, from which a bra`
  - Fix: `(1815, 'CALENDER', 'eb_5th_1815_v05_ENL-CHI', [('CALENDER', r'CALENDER\s+of\s+Monteith,', 20)])`

- 🟢 **EPHYDOR** → **EPIBAT** (1815) sim=0.323 [new_headword] (2 eds: 1815, 1823)
  - `...ntervene, they gave orders that the glass should be flopped.`
  - `EPIBATÆ, ἐπιβάται, among the Greeks, marines, or soldiers who served on board the ships of war. They`
  - Fix: `(1815, 'EPHYDOR', 'eb_5th_1815_v08_ENL-FOR', [('EPIBAT', r'EPIBATÆ,\s+ἐπιβάται,\s+among', 0)])`

- 🟢 **PALACE-C** → **PAL** (1815) sim=0.330 [new_headword] [gap: EDITORIAL]
  - `...ourt. See MARSHALSEA.

PALÆMON, or MELICERTA. See MELICERTA.`
  - `PALÆMON, Q; Rhemnius, a famous grammarian of Rome, in the reign of Tiberius. He was born of a slave `
  - Fix: `(1815, 'PALACE-C', 'eb_5th_1815_v15_NIC-CCC', [('PAL', r'PALÆMON,\s+Q;\s+Rhemnius,', 70)])`

- 🟢 **PORTSMOUTH** → **PORTO-F** (1815) sim=0.342 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...y throwing it into the fire. W. Long. 2. 5. N. Lat. 57° 50'.`
  - `PORTO-Furraio, a handsome town of Italy in the isle of Elba, with a good citadel. It is very strong,`
  - Fix: `(1815, 'PORTSMOUTH', 'eb_5th_1815_v17_ENL-RHI', [('PORTO-F', r'PORTO\-Furraio,\s+a\s+handsome', 75)])`

- 🟢 **EPHYDOR** → **EPIBAT** (1823) sim=0.345 [new_headword] (2 eds: 1815, 1823)
  - `...ntervene, they gave orders that the glass should be stopped.`
  - `EPIBATÆ, ἐπιβάται, among the Greeks, marines, or soldiers who served on board the ships of war. They`
  - Fix: `(1823, 'EPHYDOR', 'eb_6th_1823_v08_ENL-FOR', [('EPIBAT', r'EPIBATÆ,\s+ἐπιβάται,\s+among', 0)])`

- 🟢 **EUPHORBUS** → **EUPHORION** (1823) sim=0.347 [new_headword] (2 eds: 1815, 1823)
  - `...t first sight the shield of Euphorbus in the temple of Juno.`
  - `EUPHORION of Chalcis, a poet and historian, born in the 126th Olympiad. Suetonius says that Tiberius`
  - Fix: `(1823, 'EUPHORBUS', 'eb_6th_1823_v08_ENL-FOR', [('EUPHORION', r'EUPHORION\s+of\s+Chalcis,', 0)])`

- 🟢 **BATTLE** → **BATTLE** (1797) sim=0.348 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... Dantick; and the belt goes by the name of Battle gunpowder.`
  - `BATTLE, in law, or Trial by wager of Battle, a species of trial of great antiquity, but now much dis`
  - Fix: `(1797, 'BATTLE', 'eb_3rd_1797_v03_TRE-BYZ', [('BATTLE', r'BATTLE,\s+in\s+law,', 4)])`

- 🟢 **EUPHORBUS** → **EUPHORION** (1815) sim=0.348 [new_headword] (2 eds: 1815, 1823)
  - `...t first sight the shield of Euphorbus in the temple of Juno.`
  - `EUPHORION of Chalcis, a poet and historian, born in the 126th Olympiad. Suetonius says that Tiberius`
  - Fix: `(1815, 'EUPHORBUS', 'eb_5th_1815_v08_ENL-FOR', [('EUPHORION', r'EUPHORION\s+of\s+Chalcis,', 0)])`

- 🟢 **FORBES** → **FOR** (1842) sim=0.060 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...(x)`
  - `For domestic use, the meat should not be salted as soon as it comes from the market, but kept until `
  - Fix: `(1842, 'FORBES', 'eb_7th_1842_v09_ENG-FRA', [('FOR', r'For\s+domestic\s+use,', 0)])`

- 🟢 **AHAB** → **AHETULA** (1797) sim=0.061 [new_headword] (2 eds: 1797, 1815)
  - `...is son Ahaziah succeeded him, in the year of the world 3107.`
  - `AHETULA, the trivial name of a species of the coluber. See COLUBER.`
  - Fix: `(1797, 'AHAB', 'eb_3rd_1797_v01_IND-COR', [('AHETULA', r'AHETULA,\s+the\s+trivial', 0)])`

- 🟢 **AHAB** → **AHETULA** (1815) sim=0.068 [new_headword] (2 eds: 1797, 1815)
  - `...is son Ahaziah succeeded him, in the year of the world 3167.`
  - `AHETULA, the trivial name of a species of the coluber. See COLUBER.`
  - Fix: `(1815, 'AHAB', 'eb_5th_1815_v01_ENL-AME', [('AHETULA', r'AHETULA,\s+the\s+trivial', 0)])`

- 🟢 **CONSOLIDATION** → **COR** (1810) sim=0.089 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...e scar round the whole nose, and the traces of the stitches.`
  - `Cor. 3. Draw \( FK \) from the focus perpendicular to the tangent, and let \( L \) denote the parame`
  - Fix: `(1810, 'CONSOLIDATION', 'eb_4th_1810_v06_CON-CRY', [('COR', r'Cor\.\s+3\.\s+Draw', 23)])`

- 🟢 **CHROMATIC** → **CHRISTIANS** (1823) sim=0.100 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...onvex surface, which in this position was next to the light.`
  - `Christians of St John, a sect of Christians very numerous in Balsara and the neighbouring towns; whe`
  - Fix: `(1823, 'CHROMATIC', 'eb_6th_1823_v06_ENL-CRY', [('CHRISTIANS', r'Christians\s+of\s+St', 62)])`

- 🟢 **ARITHMETIC** → **ARISTOTLE** (1842) sim=0.113 [topic_change] [gap: OCR_GAP]
  - `...e rules of practice how much each debt came to at that rate.`
  - `Aristotle's discussion of friendship is open to similar objection. He has considered it in its outwa`
  - Fix: `(1842, 'ARITHMETIC', 'eb_7th_1842_v03_SEV-AST', [('ARISTOTLE', r'Aristotle's\s+discussion\s+of', 61)])`

- 🟢 **STETTIN** → **STEAM** (1842) sim=0.115 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ell and cheaply supplied. Long. 14.50. E. Lat. 52.25. 36. N.`
  - `Steam-Engine Mr Watt's Engine of Revelation.
piston, and forces it to return to the top of the cylin`
  - Fix: `(1842, 'STETTIN', 'eb_7th_1842_v20_SEV-SUG', [('STEAM', r'Steam\-Engine\s+Mr\s+Watt's', 0)])`

- 🟢 **CHROMATICS** → **CHRISTIANS** (1815) sim=0.116 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... through a glass wedge only whose angle was near 30 degrees.`
  - `CHRISTIANS, those who profess the religion of Christ: See CHRISTIANITY AND MESSIAH.—The name Christi`
  - Fix: `(1815, 'CHROMATICS', 'eb_5th_1815_v06_ENL-CRY', [('CHRISTIANS', r'CHRISTIANS,\s+those\s+who', 40)])`

- 🟢 **CHROMATICS** → **CHRISTIANS OF ST THOMAS** (1815) sim=0.129 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...tted except through the neck or anterior side of the phials.`
  - `**CHRISTIANS of St Thomas**, a sort of Christians in a peninsula of India on this side of the gulf: `
  - Fix: `(1815, 'CHROMATICS', 'eb_5th_1815_v06_ENL-CRY', [('CHRISTIANS OF ST THOMAS', r'\*\*CHRISTIANS\s+of\s+St', 49)])`

- 🟢 **CONSENTES** → **COR** (1810) sim=0.148 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `..., Venus, Mars,
Mercurius, Jovis, Neptunus, Vulcanus, Apollo.`
  - `Cor. 2. Only one circle can have the same curvature with a conic section in a given point.`
  - Fix: `(1810, 'CONSENTES', 'eb_4th_1810_v06_CON-CRY', [('COR', r'Cor\.\s+2\.\s+Only', 20)])`

- 🟢 **EALING** → **EAR** (1842) sim=0.151 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...5035, in 1811 to 5361, in 1821 to 6608, and in 1831 to 7783.`
  - `EAR. (See Anatomy.) The ear has its beauties, which a good painter ought by no means to disregard; a`
  - Fix: `(1842, 'EALING', 'eb_7th_1842_v08_DIA-VII', [('EAR', r'EAR\.\s+\(See\s+Anatomy\.\)', 0)])`

- 🟢 **PLATONIC YEAR** → **PLA** (1842) sim=0.169 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...d the sameness of things return in the same order as before.`
  - `PLAÜEN, a city of the kingdom of Saxony, in the circle of Voigtsland, and the capital of a bailiwick`
  - Fix: `(1842, 'PLATONIC YEAR', 'eb_7th_1842_v18_PLA-QUO', [('PLA', r'PLAÜEN,\s+a\s+city', 0)])`

- 🟢 **PERU** → **PERSPECTIVE** (1815) sim=0.184 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...e splendid volumes of other authors and see their frivolity.`
  - `Perspective is also used for a kind of picture or painting, frequently seen in gardens, and at the e`
  - Fix: `(1815, 'PERU', 'eb_5th_1815_v16_ENL-HOR', [('PERSPECTIVE', r'Perspective\s+is\s+also', 116)])`

- 🟢 **METHUSELAH** → **METECI** (1842) sim=0.236 [new_headword] [gap: VARIANT]
  - `...eatest age which has been attained to by any man upon earth.`
  - `METECI, a name given by the Athenians to such as had their fixed habitations in Attica, though by bi`
  - Fix: `(1842, 'METHUSELAH', 'eb_7th_1842_v14_SEV-MEX', [('METECI', r'METECI,\s+a\s+name', 0)])`

- 🟢 **DRAWING** → **DREAMS** (1842) sim=0.248 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...portsmen, denotes squirrel nests built on the tops of trees.`
  - `DREAMS are all those thoughts which pass through the mind, and those imaginary transactions in which`
  - Fix: `(1842, 'DRAWING', 'eb_7th_1842_v08_DIA-VII', [('DREAMS', r'DREAMS\s+are\s+all', 3)])`

- 🟢 **ESCUAGE** → **ESCOLAPIUS** (1815) sim=0.255 [new_headword] (2 eds: 1810, 1815)
  - `...ee the articles CHIVALRY, FEUDAL SYSTEM, and KNIGHT-SERVICE.`
  - `ESCOLAPIUS. See ÆSCULAPIUS.`
  - Fix: `(1815, 'ESCUAGE', 'eb_5th_1815_v08_ENL-FOR', [('ESCOLAPIUS', r'ESCOLAPIUS\.\s+See\s+ÆSCULAPIUS\.', 0)])`

- 🟢 **ESCUAGE** → **ESCOLAPIUS** (1810) sim=0.260 [new_headword] (2 eds: 1810, 1815)
  - `...ee the articles Chivalry, Feodal System, and Knight-Service.`
  - `ESCOLAPIUS. See Aesculapius.`
  - Fix: `(1810, 'ESCUAGE', 'eb_4th_1810_v17_ELE-FAI', [('ESCOLAPIUS', r'ESCOLAPIUS\.\s+See\s+Aesculapius\.', 0)])`

- 🟢 **ELEVATORY** → **ELVE** (1860) sim=0.263 [new_headword] (2 eds: 1842, 1860)
  - `...ing, for raising a depressed or fractured part of the skull.`
  - `ELVE, a term purely French, denoting literally a disciple or scholar—from an Italian word signifying`
  - Fix: `(1860, 'ELEVATORY', 'eb_8th_1860_v08_ADA-ENT', [('ELVE', r'ELVE,\s+a\s+term', 0)])`

- 🟢 **ANGELITES** → **ANG** (1842) sim=0.269 [new_headword] [gap: OCR_GAP]
  - `...all, and that each is God, by a participation of this deity.`
  - `ANGÈLO, MICHAEL. See Buonaroti, Michael Angelo.

ANGÈLO, St, a small but strong town of Italy, in th`
  - Fix: `(1842, 'ANGELITES', 'eb_7th_1842_v03_SEV-AST', [('ANG', r'ANGÈLO,\s+MICHAEL\.\s+See', 0)])`

- 🟢 **BOURGES** → **BOUGET** (1823) sim=0.283 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...t in the centre of France. E. Long. 2° 30'. N. Lat. 47° 10'.`
  - `BOUGET, DOM JOHN, an ingenious French antiquary, was born at the village of Beaumains near Falaise, `
  - Fix: `(1823, 'BOURGES', 'eb_6th_1823_v04_ENL-BUR', [('BOUGET', r'BOUGET,\s+DOM\s+JOHN,', 0)])`

- 🟢 **MUTUNUS** → **MUZZLE** (1810) sim=0.297 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...eity among the Romans, similar to the Priapus of the Greeks.`
  - `MUZZLE of a Gun or Mortar, the extremity at which the powder and ball is put in; and hence the muzzl`
  - Fix: `(1810, 'MUTUNUS', 'eb_4th_1810_v14_MOR-NIA', [('MUZZLE', r'MUZZLE\s+of\s+a', 0)])`

- 🟢 **ELEVATORY** → **ELVE** (1842) sim=0.297 [new_headword] (2 eds: 1842, 1860)
  - `...lied after the integuments and periosteum have been removed.`
  - `ELVE, a term purely French, though used also in our language. It signifies literally a disciple or s`
  - Fix: `(1842, 'ELEVATORY', 'eb_7th_1842_v08_DIA-VII', [('ELVE', r'ELVE,\s+a\s+term', 0)])`

- 🟢 **MUTUNUS** → **MUZZLE** (1797) sim=0.323 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...iged them to observe before the statue of this impure deity.`
  - `MUZZLE of a Gun or Mortar, the extremity at which the powder and ball is put in; and hence the muzzl`
  - Fix: `(1797, 'MUTUNUS', 'eb_3rd_1797_v12_TRE-NEG', [('MUZZLE', r'MUZZLE\s+of\s+a', 0)])`

- 🟢 **MUTUNUS** → **MUZZLE** (1823) sim=0.327 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...eity among the Romans, similar to the Priapus of the Greeks.`
  - `MUZZLE of a Gun or Mortar, the extremity at which the powder and ball is put in; and hence the muzzl`
  - Fix: `(1823, 'MUTUNUS', 'eb_6th_1823_v14_ENL-NIC', [('MUZZLE', r'MUZZLE\s+of\s+a', 0)])`

- 🟢 **MUTUNUS** → **MUZZLE** (1815) sim=0.333 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...eity among the Romans, similar to the Priapus of the Greeks.`
  - `MUZZLE of a Gun or Mortar, the extremity at which the powder and ball is put in; and hence the muzzl`
  - Fix: `(1815, 'MUTUNUS', 'eb_5th_1815_v14_ENL-NIC', [('MUZZLE', r'MUZZLE\s+of\s+a', 0)])`

- 🟡 **CEYLON** → **CHACE** (1797) sim=0.044 [new_headword] (5 eds: 1778, 1797, 1810, 1815, 1823)
  - `...ing than a piece of coarse linen wrapped about their waists.`
  - `CHACE. See Chase.`
  - Fix: `(1797, 'CEYLON', 'eb_3rd_1797_v04_TRE-OMI', [('CHACE', r'CHACE\.\s+See\s+Chase\.', 66)])`

- 🟡 **CEYLON** → **CHACE** (1778) sim=0.057 [new_headword] (5 eds: 1778, 1797, 1810, 1815, 1823)
  - `...ring than a piece of coarse linen wrapped about their waist.`
  - `CHACE. See Chase.`
  - Fix: `(1778, 'CEYLON', 'eb_2nd_1778_v03_BYW-CRI', [('CHACE', r'CHACE\.\s+See\s+Chase\.', 28)])`

- 🟡 **MARATTI** → **MOLINA** (1810) sim=0.078 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...t painter died at Rome in 1713, in the 88th year of his age.`
  - `Molina and d'Azara agree in regard to the mild qualities by which the cayou is distinguished. It eat`
  - Fix: `(1810, 'MARATTI', 'eb_4th_1810_v12_MAH-ADD', [('MOLINA', r'Molina\s+and\s+d'Azara', 9)])`

- 🟡 **CEYLON** → **CHACE** (1823) sim=0.093 [new_headword] (5 eds: 1778, 1797, 1810, 1815, 1823)
  - `... has been of late greatly neglected. See CEYLON, Supplement.`
  - `CHACE. See Chase.`
  - Fix: `(1823, 'CEYLON', 'eb_6th_1823_v502_AUS-CEL', [('CHACE', r'CHACE\.\s+See\s+Chase\.', 42)])`

- 🟡 **CEYLON** → **CHACE** (1810) sim=0.099 [new_headword] (5 eds: 1778, 1797, 1810, 1815, 1823)
  - `...d, since the island came into the possession of the British.`
  - `CHACE. See Chase.`
  - Fix: `(1810, 'CEYLON', 'eb_4th_1810_v05_BUR-CHA', [('CHACE', r'CHACE\.\s+See\s+Chase\.', 38)])`

- 🟡 **CEYLON** → **CHACE** (1815) sim=0.099 [new_headword] (5 eds: 1778, 1797, 1810, 1815, 1823)
  - `...d, since the island came into the possession of the British.`
  - `CHACE. See Chase.`
  - Fix: `(1815, 'CEYLON', 'eb_5th_1815_v05_ENL-CHI', [('CHACE', r'CHACE\.\s+See\s+Chase\.', 46)])`

- 🟡 **FILAGO** → **FORTHWITH** (1810) sim=0.134 [topic_change] [gap: VARIANT]
  - `... ranking under the 49th order, Compositae. See Botany Index.`
  - `Forthwith from the pool he rears
His mighty stature.
Paradise Lost.`
  - Fix: `(1810, 'FILAGO', 'eb_4th_1810_v08_FAI-FOR', [('FORTHWITH', r'Forthwith\s+from\s+the', 0)])`

- 🟡 **AGRICULTURE** → **ALL** (1815) sim=0.144 [topic_change] [gap: OCR_GAP]
  - `... scarce as only to be found in the libraries of the curious.`
  - `All these pressures must be balanced by the joint action of the cattle, the resistance of the bottom`
  - Fix: `(1815, 'AGRICULTURE', 'eb_5th_1815_v01_ENL-AME', [('ALL', r'All\s+these\s+pressures', 0)])`

- 🟡 **XERXES I** → **XIMEN** (1815) sim=0.152 [new_headword] [gap: EDITORIAL]
  - `... of his guards, and his distinguished favourite. See SPARTA.`
  - `XIMENÉS, FRANCIS, a justly celebrated cardinal, bishop of Toledo, and prime minister of Spain, was b`
  - Fix: `(1815, 'XERXES I', 'eb_5th_1815_v20_SUI-DIR', [('XIMEN', r'XIMENÉS,\s+FRANCIS,\s+a', 61)])`

- 🟡 **AHAZ** → **ALL** (1815) sim=0.158 [topic_change] [gap: OCR_GAP]
  - `... him in the year of the world 3287, before Jesus Christ 726.`
  - `All soils and all situations are not equally proper for this method of planting in rows, with large `
  - Fix: `(1815, 'AHAZ', 'eb_5th_1815_v01_ENL-AME', [('ALL', r'All\s+soils\s+and', 15)])`

- 🟡 **FASCINES** → **FOR** (1810) sim=0.168 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...the pales, fascines, &c. used to enclose ancient cattle, &c.`
  - `For the purpose of measuring the quantity of blood taken away, Mr White recommends a graduated vesse`
  - Fix: `(1810, 'FASCINES', 'eb_4th_1810_v08_FAI-FOR', [('FOR', r'For\s+the\s+purpose', 28)])`

- 🟡 **FIFESHIRE** → **FOR** (1810) sim=0.169 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `....

As when the force
Of subterranean wind transports a hill.`
  - `For the figures on tapestry, brocade, &c. see Tapestry, &c.`
  - Fix: `(1810, 'FIFESHIRE', 'eb_4th_1810_v08_FAI-FOR', [('FOR', r'For\s+the\s+figures', 1)])`

- 🟡 **MADURA** → **MEANDER** (1823) sim=0.257 [new_headword] (3 eds: 1778, 1823, 1860)
  - `...ce is a pearl fishery, which brings in a large sum annually.`
  - `MEANDER, in Ancient Geography, a celebrated river of Asia Minor, rising near Celene. It flows throug`
  - Fix: `(1823, 'MADURA', 'eb_6th_1823_v12_ENL-ADD', [('MEANDER', r'MEANDER,\s+in\s+Ancient', 0)])`

- 🟡 **KADESHE** → **KEMPERIA** (1797) sim=0.272 [new_headword] (2 eds: 1797, 1815)
  - `...d the north parts of Palestine. Called also Hebrews (Moses).`
  - `KEMPERIA, zedoary, in botany: A genus of the monogynia order, belonging to the monandra clafs of pla`
  - Fix: `(1797, 'KADESHE', 'eb_3rd_1797_v09_IND-LES', [('KEMPERIA', r'KEMPERIA,\s+zedoary,\s+in', 23)])`

- 🟡 **ERPETOLOGY** → **EXPLANATION OF THE PLATES** (1810) sim=0.282 [new_headword] (2 eds: 1810, 1815)
  - `...rby's Monographia apum Anglie. Latreille's treatise on Ants.`
  - `EXPLANATION OF THE PLATES.

Plate CCIII.`
  - Fix: `(1810, 'ERPETOLOGY', 'eb_4th_1810_v17_ELE-FAI', [('EXPLANATION OF THE PLATES', r'EXPLANATION\s+OF\s+THE', 186)])`

- 🟡 **ERPETOLOGY** → **EXPLANATION OF THE PLATES** (1815) sim=0.297 [new_headword] (2 eds: 1810, 1815)
  - `...by's Monographia apum Anglicæ. Latreille's treatise on Ants.`
  - `EXPLANATION OF THE PLATES.

Plate CCIII.`
  - Fix: `(1815, 'ERPETOLOGY', 'eb_5th_1815_v08_ENL-FOR', [('EXPLANATION OF THE PLATES', r'EXPLANATION\s+OF\s+THE', 155)])`

- 🟡 **MADURA** → **MEANDER** (1778) sim=0.301 [new_headword] (3 eds: 1778, 1823, 1860)
  - `... a pearl-fishery, which brings them in a large sum annually.`
  - `MEANDER, (anc. geogr.), a river rising in Phrygia from a common source with the Marfyas near Celae, `
  - Fix: `(1778, 'MADURA', 'eb_2nd_1778_v06_BYW-IND', [('MEANDER', r'MEANDER,\s+\(anc\.\s+geogr\.\),', 0)])`

- 🟡 **MADURA** → **MEANDER** (1860) sim=0.304 [new_headword] (3 eds: 1778, 1823, 1860)
  - `...land it is conjoined under the Dutch government. (See Jaya.)`
  - `MEANDER, a river which rises in Phrygia, not far from Celena; and on leaving that province forming t`
  - Fix: `(1860, 'MADURA', 'eb_8th_1860_v13_ADA-MAG', [('MEANDER', r'MEANDER,\s+a\s+river', 20)])`

- 🟡 **HELMINTHOLOGY** → **HIRUDO** (1823) sim=0.320 [new_headword] (2 eds: 1815, 1823)
  - `...b. Having no bristles on the sides of the body.`
  - `HIRUDO.
FASCIOLA.
PLANARIA.
CORDIUS.

Cuvier is uncertain whether he should place the following gene`
  - Fix: `(1823, 'HELMINTHOLOGY', 'eb_6th_1823_v10_ENL-HYD', [('HIRUDO', r'HIRUDO\.\s+FASCIOLA\.\s+PLANARIA\.', 8)])`

- 🟡 **TRUMPET** → **TULIP-T** (1797) sim=0.324 [new_headword] [gap: EDITORIAL]
  - `...er hundred, and even per root for very scarce capital sorts.`
  - `TULIP-Tree. See LIRIODENDRON.`
  - Fix: `(1797, 'TRUMPET', 'eb_3rd_1797_v18_IND-ER', [('TULIP-T', r'TULIP\-Tree\.\s+See\s+LIRIODENDRON\.', 93)])`

- 🟡 **HELMINTHOLOGY** → **HIRUDO** (1815) sim=0.334 [new_headword] (2 eds: 1815, 1823)
  - `...b. Having no bristles on the sides of the body.`
  - `HIRUDO.
FASCIOLA.
PLANARIA.
GORDIUS.

Cuvier is uncertain whether he should place the following gene`
  - Fix: `(1815, 'HELMINTHOLOGY', 'eb_5th_1815_v10_GOT-HYD', [('HIRUDO', r'HIRUDO\.\s+FASCIOLA\.\s+PLANARIA\.', 1)])`

- 🟡 **KADESHE** → **KEMPERIA** (1815) sim=0.338 [new_headword] (2 eds: 1797, 1815)
  - `... the northern parts of Palestine. Called also Hewei (Moses.)`
  - `KEMPERIA, ZEDARY, a genus of plants belonging to the monandra class; and in the natural method ranki`
  - Fix: `(1815, 'KADESHE', 'eb_5th_1815_v11_ENL-LIE', [('KEMPERIA', r'KEMPERIA,\s+ZEDARY,\s+a', 23)])`

- 🟡 **BENNET** → **CHRISTOPHER** (1815) sim=0.139 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...etters to Sir William Temple were published after his death.`
  - `Christopher, an eminent physician in the 16th century, was the son of John Bennet, of Raynton, in So`
  - Fix: `(1815, 'BENNET', 'eb_5th_1815_v03_ASS-DIR', [('CHRISTOPHER', r'Christopher,\s+an\s+eminent', 0)])`

- 🟡 **METEOROLOGY** → **NOR** (1815) sim=0.144 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...dies, 101
cealing, signs of, 103
Winter, hard, signs of, 109`
  - `Nor is it only between motives of equal force that men have the power of determining themselves. Who`
  - Fix: `(1815, 'METEOROLOGY', 'eb_5th_1815_v13_MAT-CCC', [('NOR', r'Nor\s+is\s+it', 76)])`

- 🟡 **RUSSIA** → **STATISTICS.** (1842) sim=0.172 [topic_change] [gap: VARIANT]
  - `...26 received a tolerably complete revision and concentration.`
  - `**STATISTICS.**

The Russian empire is of enormous extent, of vast resources, and of great capacity `
  - Fix: `(1842, 'RUSSIA', 'eb_7th_1842_v19_SEV-SCU', [('STATISTICS.', r'\*\*STATISTICS\.\*\*\s+The\s+Russian', 64)])`

- 🟡 **PARR** → **QUI** (1842) sim=0.181 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ilence to outrage and abuse. He might indeed say with truth,`
  - `Qui me commorit (melius non tangere, clamo)
Flebit, et insignis tota cantabitur urbe.`
  - Fix: `(1842, 'PARR', 'eb_7th_1842_v17_SEV-CON', [('QUI', r'Qui\s+me\s+commorit', 43)])`

- 🟡 **IDYLLION** → **JEARS** (1797) sim=0.244 [new_headword] (2 eds: 1797, 1810)
  - `...rm idyllion seems to be now appropriated to pastoral pieces.`
  - `JEARS or GEERS, in the sea-language, an assemblage of tackles, by which the lower yards of a ship ar`
  - Fix: `(1797, 'IDYLLION', 'eb_3rd_1797_v09_IND-LES', [('JEARS', r'JEARS\s+or\s+GEERS,', 0)])`

- 🟡 **IDYLLION** → **JEBUS** (1810) sim=0.260 [new_headword] (2 eds: 1797, 1810)
  - `...which operations is called swaying, and the latter striking.`
  - `JEBUSÆI, one of the seven ancient peoples of Canaan, descendants of Jebufl, Canaan's son; so warlike`
  - Fix: `(1810, 'IDYLLION', 'eb_4th_1810_v11_HYD-JUN', [('JEBUS', r'JEBUSÆI,\s+one\s+of', 28)])`

- 🟡 **IDYLLION** → **JEARS** (1810) sim=0.265 [new_headword] (2 eds: 1797, 1810)
  - `...rm Idyllion seems to be now appropriated to pastoral pieces.`
  - `JEARS or GEERS, in the sea language, an assemblage of tackles, by which the lower yards of a ship ar`
  - Fix: `(1810, 'IDYLLION', 'eb_4th_1810_v11_HYD-JUN', [('JEARS', r'JEARS\s+or\s+GEERS,', 0)])`

- 🟡 **IDYLLION** → **JEBUS** (1797) sim=0.285 [new_headword] (2 eds: 1797, 1810)
  - `...which operations is called swaying, and the latter striking.`
  - `JEBUSÆI, one of the seven ancient people of Canaan, descendents of Jebusi, Canaan's son; so warlike `
  - Fix: `(1797, 'IDYLLION', 'eb_3rd_1797_v09_IND-LES', [('JEBUS', r'JEBUSÆI,\s+one\s+of', 28)])`

- 🟡 **IVA** → **JUAN DE FUCA** (1815) sim=0.317 [new_headword] (2 eds: 1815, 1823)
  - `...hah is always double, and furnished with a small neat house.`
  - `JUAN DE FUCA, a strait on the north-west coast of America, was surveyed by Captain Vancouver, and th`
  - Fix: `(1815, 'IVA', 'eb_5th_1815_v11_ENL-LIE', [('JUAN DE FUCA', r'JUAN\s+DE\s+FUCA,', 5)])`

- 🟡 **IVA** → **JUAN DE FUCA** (1823) sim=0.319 [new_headword] (2 eds: 1815, 1823)
  - `...han is always double, and furnished with a small neat house.`
  - `JUAN DE FUCA, a strait on the north-west coast of America, was surveyed by Captain Vancouver, and th`
  - Fix: `(1823, 'IVA', 'eb_6th_1823_v11_ENL-LIE', [('JUAN DE FUCA', r'JUAN\s+DE\s+FUCA,', 15)])`

- 🟡 **ORATORY** → **PARTICULAR ELOCUTION** (1815) sim=0.327 [new_headword] (2 eds: 1797, 1815)
  - `...r, and therefore we need not multiply examples of them here.`
  - `PARTICULAR ELOCUTION,

Or that part of Elocution which considers the several Properties and Ornament`
  - Fix: `(1815, 'ORATORY', 'eb_5th_1815_v15_NIC-CCC', [('PARTICULAR ELOCUTION', r'PARTICULAR\s+ELOCUTION,\s+Or', 42)])`

- 🟡 **ORATORY** → **PARTICULAR ELOCUTION** (1797) sim=0.331 [new_headword] (2 eds: 1797, 1815)
  - `...r, and therefore we need not multiply examples of them here.`
  - `PARTICULAR ELOCUTION,

Or that part of Elocution which considers the several Properties and Ornament`
  - Fix: `(1797, 'ORATORY', 'eb_3rd_1797_v13_TRE-PAS', [('PARTICULAR ELOCUTION', r'PARTICULAR\s+ELOCUTION,\s+Or', 50)])`

- 🟠 **PIVAT** → **PIUS II** (1797) sim=0.043 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...in a sole, or piece of iron or brass hollowed to receive it.`
  - `PIUS II. (Æneas-Sylvius Piccolomini), was born on the 18th of October 1405, at Cortigni in Sienese, `
  - Fix: `(1797, 'PIVAT', 'eb_3rd_1797_v14_TRE-PLA', [('PIUS II', r'PIUS\s+II\.\s+\(Æneas\-Sylvius', 0)])`

- 🟠 **PIVAT** → **PIUS II** (1815) sim=0.053 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...in a hole, or piece of iron or brass hollowed to receive it.`
  - `PIUS II. (Aeneas-Sylvius Piccolomini), was born on the 18th of October 1405, at Corsigni in the Sien`
  - Fix: `(1815, 'PIVAT', 'eb_5th_1815_v16_ENL-HOR', [('PIUS II', r'PIUS\s+II\.\s+\(Aeneas\-Sylvius', 0)])`

- 🟠 **PIVAT** → **PIUS II** (1823) sim=0.062 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...in a sole, or piece of iron or brass hollowed to receive it.`
  - `PIUS II. (Æneas Sylvius Piccolomini), was born on the 18th of October 1405, at Corsigni in the Siene`
  - Fix: `(1823, 'PIVAT', 'eb_6th_1823_v16_ENL-BRE', [('PIUS II', r'PIUS\s+II\.\s+\(Æneas', 0)])`

- 🟠 **ORES** → **ORDER** (1823) sim=0.091 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...tated from the yellow solution seven grains of oxydied iron.`
  - `Order is also the title of certain ancient books, containing the divine office, with the order and m`
  - Fix: `(1823, 'ORES', 'eb_6th_1823_v15_ENL-PAR', [('ORDER', r'Order\s+is\s+also', 10)])`

- 🟠 **BOL** → **BOKHARIA** (1815) sim=0.101 [new_headword] (3 eds: 1797, 1815, 1823)
  - `... Bol died at Dort, the place of his birth, in 1681, aged 70.`
  - `BOKHARIA. See Bukharia.`
  - Fix: `(1815, 'BOL', 'eb_5th_1815_v03_ASS-DIR', [('BOKHARIA', r'BOKHARIA\.\s+See\s+Bukharia\.', 28)])`

- 🟠 **IRON-MAKING** → **IRENEUS** (1842) sim=0.108 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...en Long. 150° 30' and 153° 5' E. and Lat. 33° 40' and 50° S.`
  - `IRENEUS, St., a bishop of Lyons, was born in Greece about the year 120 of our era. He was the discip`
  - Fix: `(1842, 'IRON-MAKING', 'eb_7th_1842_v12_DEF-PLA', [('IRENEUS', r'IRENEUS,\s+St\.,\s+a', 188)])`

- 🟠 **SCUDDING** → **SCHOTT** (1842) sim=0.131 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...END OF VOLUME NINETEENTH.`
  - `Schott, *Isagoge,* p. 370; Horne, ii. p. 56.

Scholz, the most recent editor of the Greek New Testam`
  - Fix: `(1842, 'SCUDDING', 'eb_7th_1842_v19_SEV-SCU', [('SCHOTT', r'Schott,\s+\*Isagoge,\*\s+p\.', 55)])`

- 🟠 **GERMANY** → **GEORGE I** (1815) sim=0.132 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...tical Geometry, will be given under the article MEASURATION.`
  - `GEORGE I. II. and III. kings of Great Britain.—George I. the son of Ernest Augustus, duke of Brunswi`
  - Fix: `(1815, 'GERMANY', 'eb_5th_1815_v09_FOR-CCX', [('GEORGE I', r'GEORGE\s+I\.\s+II\.', 171)])`

- 🟠 **RYMER** → **RYCHOPS** (1797) sim=0.133 [new_headword] (3 eds: 1797, 1810, 1823)
  - `... Mr Nichol's Select Collection of Miscellaneous Poems, 1780.`
  - `RYCHOPS, in ornithology, a genus belonging to the order of anseres. The bill is straight; and the su`
  - Fix: `(1797, 'RYMER', 'eb_3rd_1797_v16_TRE-SCO', [('RYCHOPS', r'RYCHOPS,\s+in\s+ornithology,', 28)])`

- 🟠 **ANNUITIES** → **AND** (1823) sim=0.142 [topic_change] [gap: VARIANT]
  - `...y the multiplied observations of various subsequent authors.`
  - `And \( \bar{A} = \frac{1 - t_a v^t}{1 - t_a v^t + \frac{1}{1 + A}} + v - 1 \)`
  - Fix: `(1823, 'ANNUITIES', 'eb_6th_1823_v01_MAC-ANA', [('AND', r'And\s+\\\(\s+\\bar\{A\}', 43)])`

- 🟠 **MADURA** → **MACENAS** (1842) sim=0.142 [new_headword] (2 eds: 1778, 1842)
  - `... inhabiting the district now called Lauderdale, in Scotland.`
  - `MACENAS, Caius Cilnius, the friend and counselor of Augustus Caesar, a man whose name has become a s`
  - Fix: `(1842, 'MADURA', 'eb_7th_1842_v13_SEV-AB', [('MACENAS', r'MACENAS,\s+Caius\s+Cilnius,', 55)])`

- 🟠 **RYMER** → **RYCHOPS** (1810) sim=0.144 [new_headword] (3 eds: 1797, 1810, 1823)
  - `...E. Long. o. 50. N. Lat. 51. o.`
  - `RYCHOPS, a genus of birds belonging to the order of anseres. See Ornithology Index.`
  - Fix: `(1810, 'RYMER', 'eb_4th_1810_v18_RUS-SCR', [('RYCHOPS', r'RYCHOPS,\s+a\s+genus', 45)])`

- 🟠 **BOL** → **BOKHARIA** (1797) sim=0.145 [new_headword] (3 eds: 1797, 1815, 1823)
  - `... Bol died at Dort, the place of his birth, in 1681, aged 70.`
  - `BOKHARIA. See Bukharia.`
  - Fix: `(1797, 'BOL', 'eb_3rd_1797_v03_TRE-BYZ', [('BOKHARIA', r'BOKHARIA\.\s+See\s+Bukharia\.', 28)])`

- 🟠 **ARTOTYRITES** → **ARAU** (1823) sim=0.146 [new_headword] (2 eds: 1815, 1823)
  - `...wretchedness of human nature, and the miseries of this life.`
  - `ARAU, in Ancient Geography, a town of Baetica, in the jurisdiction of the Conventus Hispalensis: now`
  - Fix: `(1823, 'ARTOTYRITES', 'eb_6th_1823_v02_ENL-ASS', [('ARAU', r'ARAU,\s+in\s+Ancient', 28)])`

- 🟠 **BOL** → **BOKHARIA** (1823) sim=0.147 [new_headword] (3 eds: 1797, 1815, 1823)
  - `... Bol died at Dort, the place of his birth, in 1681, aged 70.`
  - `BOKHARIA. See Bukharia.`
  - Fix: `(1823, 'BOL', 'eb_6th_1823_v03_ENL-BOO', [('BOKHARIA', r'BOKHARIA\.\s+See\s+Bukharia\.', 45)])`

- 🟠 **CHRISOM** → **CHINESE WHEEL** (1797) sim=0.151 [person_bio] [gap: EDITORIAL]
  - `...ory life thou mayest be partaker of life everlasting. Amen."`
  - `Chinese Wheel is an engine employed in the province of Kiang-si, and probably through the whole empi`
  - Fix: `(1797, 'CHRISOM', 'eb_3rd_1797_v501_ABE-IMP', [('CHINESE WHEEL', r'Chinese\s+Wheel\s+is', 6)])`

- 🟠 **ONYX** → **ONALASHKA** (1810) sim=0.154 [new_headword] (2 eds: 1810, 1823)
  - `...nail of the finger. See Carnelian, under Mineralogy, p. 167.`
  - `ONALASHKA, one of the islands of the Northern Archipelago, visited by Captain Cook in his last voyag`
  - Fix: `(1810, 'ONYX', 'eb_4th_1810_v15_NIC-ORA', [('ONALASHKA', r'ONALASHKA,\s+one\s+of', 0)])`

- 🟠 **ARTOTYRITES** → **ARAU** (1815) sim=0.159 [new_headword] (2 eds: 1815, 1823)
  - `...wretchedness of human nature, and the miseries of this life.`
  - `ARAU, in Ancient Geography, a town of Baetica, in the jurisdiction of the Conventus Hispalensis: now`
  - Fix: `(1815, 'ARTOTYRITES', 'eb_5th_1815_v02_ENL-ASS', [('ARAU', r'ARAU,\s+in\s+Ancient', 28)])`

- 🟠 **ORES** → **ORDERS** (1823) sim=0.163 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `..., amounted to three grains and a half of argillaceous earth.`
  - `Orders, by way of eminency, or Holy Orders, denote a character peculiar to ecclesiastics, whereby th`
  - Fix: `(1823, 'ORES', 'eb_6th_1823_v15_ENL-PAR', [('ORDERS', r'Orders,\s+by\s+way', 10)])`

- 🟠 **MAJESTY** → **MAIL INDUCTIO** (1815) sim=0.165 [new_headword] (2 eds: 1815, 1823)
  - `...ce it signifies no more than the royalty or sovereign power.`
  - `MAIL INDUCTIO, an ancient custom for the priest and people of country-villages to go in procession t`
  - Fix: `(1815, 'MAJESTY', 'eb_5th_1815_v12_LIE-CCX', [('MAIL INDUCTIO', r'MAIL\s+INDUCTIO,\s+an', 28)])`

- 🟠 **ONYX** → **ONALASHKA** (1823) sim=0.168 [new_headword] (2 eds: 1810, 1823)
  - `...nail of the finger. See CARNELIAN, under MINERALOGY, p. 167.`
  - `ONALASHKA, one of the islands of the Northern Archipelago, visited by Captain Cook in his last voyag`
  - Fix: `(1823, 'ONYX', 'eb_6th_1823_v15_ENL-PAR', [('ONALASHKA', r'ONALASHKA,\s+one\s+of', 0)])`

- 🟠 **FEVERSHAM** → **FEBRI** (1823) sim=0.180 [new_headword] (2 eds: 1815, 1823)
  - `...ancient church was rebuilt in 1754, at the expense of 2300l.`
  - `FEBRI. SANCTÆ. FEBRI. MAGNÆ. CAMILLA. AMATA. PRO. FILIO. MALE. AFFECTO.in Farriery. See Farriery Ind`
  - Fix: `(1823, 'FEVERSHAM', 'eb_6th_1823_v08_ENL-FOR', [('FEBRI', r'FEBRI\.\s+SANCTÆ\.\s+FEBRI\.', 0)])`

- 🟠 **SOVEREIGN** → **SOU** (1815) sim=0.182 [new_headword] (2 eds: 1810, 1815)
  - `..., lords, and commons, not in any of the three estates alone.`
  - `SOU. See SOL.`
  - Fix: `(1815, 'SOVEREIGN', 'eb_5th_1815_v19_SCR-DVI', [('SOU', r'SOU\.\s+See\s+SOL\.', 28)])`

- 🟠 **EPIPHANIUS** → **EPHANY** (1842) sim=0.183 [new_headword] (2 eds: 1823, 1842)
  - `...aris in 1622. This edition was reprinted at Cologne in 1682.`
  - `EPHANY, a Christian festival, otherwise called the Manifestation of Christ to the Gentiles, observed`
  - Fix: `(1842, 'EPIPHANIUS', 'eb_7th_1842_v09_ENG-FRA', [('EPHANY', r'EPHANY,\s+a\s+Christian', 0)])`

- 🟠 **THIRST** → **THEOLOGY** (1797) sim=0.193 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...Prevention against Hunger and Thirst. See Hunger.`
  - `Theology, gion; to efface in our mind all traces and ideas of the ancients; and to fortify ourselves`
  - Fix: `(1797, 'THIRST', 'eb_3rd_1797_v18_IND-ER', [('THEOLOGY', r'Theology,\s+gion;\s+to', 28)])`

- 🟠 **EDOM** → **EDMUND I** (1815) sim=0.194 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...n Solomon's time extending to the Red sea, (1 Kings ix. 26.)`
  - `EDMUND I. and II. See (History of) England.`
  - Fix: `(1815, 'EDOM', 'eb_5th_1815_v07_CUB-DIR', [('EDMUND I', r'EDMUND\s+I\.\s+and', 28)])`

- 🟠 **RYMER** → **RYCHOPS** (1823) sim=0.197 [new_headword] (3 eds: 1797, 1810, 1823)
  - `... Mr Nichol's Select Collection of Miscellaneous Poems, 1780.`
  - `RYCHOPS, a genus of birds belonging to the order of anseres. See Ornithology Index.`
  - Fix: `(1823, 'RYMER', 'eb_6th_1823_v18_ENL-SCR', [('RYCHOPS', r'RYCHOPS,\s+a\s+genus', 28)])`

- 🟠 **EDOM** → **EDMUND I** (1823) sim=0.199 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...n Solomon's time extending to the Red sea, (1 Kings ix. 26.)`
  - `EDMUND I. and II. See (History of) England.`
  - Fix: `(1823, 'EDOM', 'eb_6th_1823_v07_ENL-ELE', [('EDMUND I', r'EDMUND\s+I\.\s+and', 28)])`

- 🟠 **SOVEREIGN** → **SOU** (1810) sim=0.204 [new_headword] (2 eds: 1810, 1815)
  - `..., lords, and commons, not in any of the three estates alone.`
  - `SOU. See Sol.`
  - Fix: `(1810, 'SOVEREIGN', 'eb_4th_1810_v19_SLE-SUG', [('SOU', r'SOU\.\s+See\s+Sol\.', 28)])`

- 🟠 **FEVERSHAM** → **FEBRI** (1815) sim=0.207 [new_headword] (2 eds: 1815, 1823)
  - `...ancient church was rebuilt in 1754, at the expense of 2300l.`
  - `FEBRI. SANCTÆ. FEBRI. MAGNÆ. CAMILLA. AMATA. PRO. FILIO. MALE. AFFECTO.Farriery. See Farriery Index.`
  - Fix: `(1815, 'FEVERSHAM', 'eb_5th_1815_v08_ENL-FOR', [('FEBRI', r'FEBRI\.\s+SANCTÆ\.\s+FEBRI\.', 0)])`

- 🟠 **BUC** → **BUANEER** (1823) sim=0.207 [new_headword] (2 eds: 1815, 1823)
  - `...s; and, 3. A work entitled The Third Universitie of England.`
  - `BUANEER, one who dries and smokes flesh or fish after the manner of the Indians. The name was partic`
  - Fix: `(1823, 'BUC', 'eb_6th_1823_v04_ENL-BUR', [('BUANEER', r'BUANEER,\s+one\s+who', 0)])`

- 🟠 **BUC** → **BUANEER** (1815) sim=0.209 [new_headword] (2 eds: 1815, 1823)
  - `...ls; and, 3. A work entitled The Third University of England.`
  - `BUANEER, one who dries and smokes flesh or fish after the manner of the Indians. The name was partic`
  - Fix: `(1815, 'BUC', 'eb_5th_1815_v04_ENL-BUR', [('BUANEER', r'BUANEER,\s+one\s+who', 0)])`

- 🟠 **GERMANY** → **GEORGE I** (1810) sim=0.211 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... cor. | 37. |
| 38. | 20. | 75. | 45. | 45. | 2 cor. | 38. |`
  - `GEORGE I. II. and III. kings of Great Britain.—George I. the son of Ernest Augustus, duke of Brunswi`
  - Fix: `(1810, 'GERMANY', 'eb_4th_1810_v09_FAR-GOT', [('GEORGE I', r'GEORGE\s+I\.\s+II\.', 163)])`

- 🟠 **ASSAYING** → **ASIARCH** (1842) sim=0.211 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ominations. For other particulars see the preceding article.`
  - `ASIARCHÆ (termed by St Paul, Chief of Asia, Acts xix. 31) were the Pagan pontiffs of Asia, chosen to`
  - Fix: `(1842, 'ASSAYING', 'eb_7th_1842_v03_SEV-AST', [('ASIARCH', r'ASIARCHÆ\s+\(termed\s+by', 131)])`

- 🟠 **MESSINA** → **MEDICAL POLICE** (1815) sim=0.225 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ases preventing cohabitation; venereal disease, leprosy, &c.`
  - `MEDICAL POLICE.

Of incomparably greater consequence, and more widely extended influence, is the sec`
  - Fix: `(1815, 'MESSINA', 'eb_5th_1815_v13_MAT-CCC', [('MEDICAL POLICE', r'MEDICAL\s+POLICE\.\s+Of', 2514)])`

- 🟠 **EPIPHANIUS** → **EPHANY** (1823) sim=0.229 [new_headword] (2 eds: 1823, 1842)
  - `...aris in 1622. This edition was reprinted at Cologne in 1682.`
  - `EPHANY, a Christian festival, otherwise called the Manifestation of Christ to the Gentiles, observed`
  - Fix: `(1823, 'EPIPHANIUS', 'eb_6th_1823_v08_ENL-FOR', [('EPHANY', r'EPHANY,\s+a\s+Christian', 2)])`

- 🟠 **FEZZAN** → **FEWEL** (1815) sim=0.233 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... for a rich man is, "that he eats bread and meat every day."`
  - `FEWEL. See FUEL.`
  - Fix: `(1815, 'FEZZAN', 'eb_5th_1815_v08_ENL-FOR', [('FEWEL', r'FEWEL\.\s+See\s+FUEL\.', 43)])`

- 🟠 **DAPHNE** → **DAHNPEPHORIA** (1823) sim=0.233 [new_headword] (2 eds: 1810, 1823)
  - `... ranking under the 31st order, Vepreculea. See Botany Index.`
  - `DAHNPEPHORIA, a festival in honour of Apollo, celebrated every ninth year by the Boeotians. It was t`
  - Fix: `(1823, 'DAPHNE', 'eb_6th_1823_v07_ENL-ELE', [('DAHNPEPHORIA', r'DAHNPEPHORIA,\s+a\s+festival', 55)])`

- 🟠 **FEZZAN** → **FEWEL** (1810) sim=0.235 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... for a rich man is, "that he eats bread and meat every day."`
  - `FEWEL. See FUEL.`
  - Fix: `(1810, 'FEZZAN', 'eb_4th_1810_v08_FAI-FOR', [('FEWEL', r'FEWEL\.\s+See\s+FUEL\.', 27)])`

- 🟠 **GERMANY** → **GEORGE I** (1823) sim=0.235 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... | 37. | 66. | 7. |
| 25. | 1. | 22. | 4. | 38. | 70. | 7. |`
  - `GEORGE I. II. and III. kings of Great Britain.

—George I. the son of Ernest Augustus, duke of Bruns`
  - Fix: `(1823, 'GERMANY', 'eb_6th_1823_v504_FOU-HOL', [('GEORGE I', r'GEORGE\s+I\.\s+II\.', 132)])`

- 🟠 **EDOM** → **EDMUND I** (1797) sim=0.240 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...in Solomon's time extending to the Red Sea (1 Kings ix. 26.)`
  - `EDMUND I. and II. See (History of) ENGLAND.`
  - Fix: `(1797, 'EDOM', 'eb_3rd_1797_v06_IND-ETH', [('EDMUND I', r'EDMUND\s+I\.\s+and', 28)])`

- 🟠 **FEZ** → **FEWEL** (1797) sim=0.241 [new_headword] (2 eds: 1778, 1797)
  - `... seated on the river Cebu, W. Long. 4° 25'. N. Lat. 33° 58'.`
  - `FEWEL. See FUEL.`
  - Fix: `(1797, 'FEZ', 'eb_3rd_1797_v07_TRE-GOA', [('FEWEL', r'FEWEL\.\s+See\s+FUEL\.', 85)])`

- 🟠 **EPIPHANIUS** → **EPHONEMA** (1815) sim=0.241 [new_headword] (2 eds: 1815, 1823)
  - `...the word epiphanias in his second epistle to Timothy, i. 10.`
  - `EPHONEMA. See ORATORY, No. 96.

EPHORA, in Medicine, a preternatural defluxion of the eyes, when the`
  - Fix: `(1815, 'EPIPHANIUS', 'eb_5th_1815_v08_ENL-FOR', [('EPHONEMA', r'EPHONEMA\.\s+See\s+ORATORY,', 17)])`

- 🟠 **FEZZAN** → **FEWEL** (1823) sim=0.241 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... for a rich man is, "that he eats bread and meat every day."`
  - `FEWEL. See FUEL.`
  - Fix: `(1823, 'FEZZAN', 'eb_6th_1823_v08_ENL-FOR', [('FEWEL', r'FEWEL\.\s+See\s+FUEL\.', 29)])`

- 🟠 **FEZ** → **FEWEL** (1778) sim=0.252 [new_headword] (2 eds: 1778, 1797)
  - `... seated on the river Cebu, W. Long. 4° 25'.
N. Lat. 33° 58'.`
  - `FEWEL. See FUEL.`
  - Fix: `(1778, 'FEZ', 'eb_2nd_1778_v04_BYW-FUZ', [('FEWEL', r'FEWEL\.\s+See\s+FUEL\.', 70)])`

- 🟠 **EPIPHANIUS** → **EPHONEMA** (1823) sim=0.254 [new_headword] (2 eds: 1815, 1823)
  - `...the word epiphanias in his second epistle to Timothy, i. 10.`
  - `EPHONEMA. See ORATORY, No. 96.

EPHORA, in Medicine, a preternatural fluxion of the eyes, when they `
  - Fix: `(1823, 'EPIPHANIUS', 'eb_6th_1823_v08_ENL-FOR', [('EPHONEMA', r'EPHONEMA\.\s+See\s+ORATORY,', 16)])`

- 🟠 **DAPHNE** → **DAHNPEPHORIA** (1810) sim=0.255 [new_headword] (2 eds: 1810, 1823)
  - `...d ranking under the 31st order, Veprecule. See Botany Index.`
  - `DAHNPEPHORIA, a festival in honour of Apollo, celebrated every ninth year by the Boeotians. It was t`
  - Fix: `(1810, 'DAPHNE', 'eb_4th_1810_v17_CRY-DYE', [('DAHNPEPHORIA', r'DAHNPEPHORIA,\s+a\s+festival', 55)])`

- 🟠 **SAVIOUR** → **SAUL** (1797) sim=0.265 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...minister the sacrament and spiritual assistance to the nuns.`
  - `SAUL the son of Kish, of the tribe of Benjamin, was the first king of the Israelites. On account of `
  - Fix: `(1797, 'SAVIOUR', 'eb_3rd_1797_v16_TRE-SCO', [('SAUL', r'SAUL\s+the\s+son', 28)])`

- 🟠 **SAVIOUR** → **SAUL** (1823) sim=0.267 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...minister the sacrament and spiritual assistance to the nuns.`
  - `SAUL the son of Kish, of the tribe of Benjamin, was the first king of the Israelites. On account of `
  - Fix: `(1823, 'SAVIOUR', 'eb_6th_1823_v18_ENL-SCR', [('SAUL', r'SAUL\s+the\s+son', 28)])`

- 🟠 **SAVIOUR** → **SAUL** (1810) sim=0.270 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...minister the sacrament and spiritual assistance to the nuns.`
  - `SAUL the son of Kish, of the tribe of Benjamin, was the first king of the Israelites. On account of `
  - Fix: `(1810, 'SAVIOUR', 'eb_4th_1810_v18_RUS-SCR', [('SAUL', r'SAUL\s+the\s+son', 28)])`

- 🟠 **SAVIOUR** → **SAUL** (1815) sim=0.283 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...minister the sacrament and spiritual assistance to the nuns.`
  - `SAUL the son of Kish, of the tribe of Benjamin, was the first king of the Israelites. On account of `
  - Fix: `(1815, 'SAVIOUR', 'eb_5th_1815_v18_ENL-SCR', [('SAUL', r'SAUL\s+the\s+son', 28)])`

- 🟠 **MESSIAH** → **MEDICAL JURISPRUDENCE** (1815) sim=0.309 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ducted on the same general principles as in the adult state.`
  - `MEDICAL JURISPRUDENCE.

During the progress of science in Europe this subject has not been altogethe`
  - Fix: `(1815, 'MESSIAH', 'eb_5th_1815_v13_MAT-CCC', [('MEDICAL JURISPRUDENCE', r'MEDICAL\s+JURISPRUDENCE\.\s+During', 20241)])`

- 🟠 **APPLICATION** → **APOGGIATURA** (1823) sim=0.337 [new_headword] (2 eds: 1815, 1823)
  - `...y means or instruments whereby this application is effected.`
  - `APOGGIATURA, in Music, a small note inserted by the practical musician, between two others, at some `
  - Fix: `(1823, 'APPLICATION', 'eb_6th_1823_v02_ENL-ASS', [('APOGGIATURA', r'APOGGIATURA,\s+in\s+Music,', 45)])`

- 🟠 **APPLICATION** → **APOGGIATURA** (1815) sim=0.338 [new_headword] (2 eds: 1815, 1823)
  - `...y means or instruments whereby this application is effected.`
  - `APOGGIATURA, in Music, a small note inserted by the practical musician, between two others, at some `
  - Fix: `(1815, 'APPLICATION', 'eb_5th_1815_v02_ENL-ASS', [('APOGGIATURA', r'APOGGIATURA,\s+in\s+Music,', 45)])`

- 🔴 **ARACHNIDES** → **ALL** (1842) sim=-0.036 [topic_change] [gap: OCR_GAP]
  - `... Tayf, which had been held by the Wahabys for sixteen years,`
  - `All female spiders, including even the erratic and webless species, are provided with reservoirs of `
  - Fix: `(1842, 'ARACHNIDES', 'eb_7th_1842_v03_SEV-AST', [('ALL', r'All\s+female\s+spiders,', 71)])`

- 🔴 **ICONOCLASTES** → **GEN** (1815) sim=-0.029 [topic_change] [gap: EDITORIAL]
  - `... parts of the Christian world by the Reformation. See IMAGE.`
  - `Gen. 5. Platystacus.

Flatylactus.`
  - Fix: `(1815, 'ICONOCLASTES', 'eb_5th_1815_v11_ENL-LIE', [('GEN', r'Gen\.\s+5\.\s+Platystacus\.', 7)])`

- 🔴 **IDIOCY** → **GEN** (1815) sim=-0.013 [topic_change] [gap: EDITORIAL]
  - `...enefit of such lunatic, his heirs, or executors. See Lunacy.`
  - `Gen. 8. Acanthotus.

Body elongated, without dorsal fin. Several spines on the back and abdomen.`
  - Fix: `(1815, 'IDIOCY', 'eb_5th_1815_v11_ENL-LIE', [('GEN', r'Gen\.\s+8\.\s+Acanthotus\.', 3)])`

- 🔴 **MYLASA** → **FOR** (1815) sim=-0.011 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...nts, which have been employed to construct a Turkish mosque.`
  - `For the highest masculine voices, which are called counter tenor, and for the tenor violin, a staff `
  - Fix: `(1815, 'MYLASA', 'eb_5th_1815_v14_ENL-NIC', [('FOR', r'For\s+the\s+highest', 0)])`

- 🔴 **ROME** → **ALL** (1842) sim=-0.007 [topic_change] [gap: OCR_GAP]
  - `... and in a few seconds extended a headless trunk before them.`
  - `All the novels of Miss Austin closely resemble each other;
but Northanger Abbey, and Sense and Sensi`
  - Fix: `(1842, 'ROME', 'eb_7th_1842_v19_SEV-SCU', [('ALL', r'All\s+the\s+novels', 19)])`

- 🔴 **ANDRISCUS** → **ALL** (1810) sim=0.002 [topic_change] [gap: OCR_GAP]
  - `...of Metellus, walking in chains before the general's chariot.`
  - `All these parts are plentifully supplied with blood vessels and nerves. Around the nymphae there are`
  - Fix: `(1810, 'ANDRISCUS', 'eb_4th_1810_v17_ART-ANS', [('ALL', r'All\s+these\s+parts', 0)])`

- 🔴 **ELMACINUS** → **VARIOUS** (1842) sim=0.007 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ranslated in whole or in part into several modern languages.`
  - `Various electrical phenomena of a very interesting kind have been observed by travellers when ascend`
  - Fix: `(1842, 'ELMACINUS', 'eb_7th_1842_v08_DIA-VII', [('VARIOUS', r'Various\s+electrical\s+phenomena', 0)])`

- 🔴 **BOURDON** → **ADD** (1823) sim=0.011 [topic_change] [gap: OCR_GAP]
  - `... the generality of the collectors. He died in 1673, aged 54.`
  - `Add the species cynosuroides, cespitosa, littoralis, levis, villosa, serrata, ciliaris, hispida, gen`
  - Fix: `(1823, 'BOURDON', 'eb_6th_1823_v04_ENL-BUR', [('ADD', r'Add\s+the\s+species', 0)])`

- 🔴 **DYNAMICS** → **III** (1810) sim=0.021 [new_headword] (2 eds: 1810, 1842)
  - `...own actions, or the exertions of their own powers or forces.`
  - `III. Of Dyeing Cotton and Linen Violet.

393. The most ordinary mode by which a violet colour is com`
  - Fix: `(1810, 'DYNAMICS', 'eb_4th_1810_v07_STE-ELE', [('III', r'III\.\s+Of\s+Dyeing', 47)])`

- 🔴 **OSORIUS** → **GREENSHANK** (1823) sim=0.029 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ion of Osorius, but the last work of Cicero on that subject.`
  - `Greenshank.—Bill straight, the lower base red; body beneath snowy; legs greenish; bill black; the lo`
  - Fix: `(1823, 'OSORIUS', 'eb_6th_1823_v15_ENL-PAR', [('GREENSHANK', r'Greenshank\.—Bill\s+straight,\s+the', 0)])`

- 🔴 **SPECIFICS** → **CHARLES** (1810) sim=0.036 [topic_change] [gap: OCR_GAP]
  - `...043 |
| White vitriol | 1.386 | 1.045 |
| Pec. sal | 1.534 |`
  - `Charles IV. had not long been seated on the throne before the potentates revolution in France involv`
  - Fix: `(1810, 'SPECIFICS', 'eb_4th_1810_v19_SLE-SUG', [('CHARLES', r'Charles\s+IV\.\s+had', 43)])`

- 🔴 **HIPPOCRATES** → **FORTUNATELY** (1842) sim=0.041 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...: 5. Numerous editions in French and other modern languages.`
  - `Fortunately for the British army, it was met, before the end of the first day's march, by the allied`
  - Fix: `(1842, 'HIPPOCRATES', 'eb_7th_1842_v11_GRO-HYD', [('FORTUNATELY', r'Fortunately\s+for\s+the', 35)])`

- 🔴 **MEDALS** → **FOR** (1842) sim=0.041 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...maller cabinet, any coin of a series is of high price, or of`
  - `For this purpose, let ABC, fig. 100, be an arch, MHTO the pier, and BUHA the loaded semicircle, whos`
  - Fix: `(1842, 'MEDALS', 'eb_7th_1842_v14_SEV-MEX', [('FOR', r'For\s+this\s+purpose,', 20)])`

- 🔴 **WHEN THE** → **EDWARD** (1797) sim=0.047 [topic_change] [gap: VARIANT]
  - `...ge $f$, marks a fluent, or the sum of fluxionary quantities.`
  - `Edward then declared, by the mouth of his chancellor, that although, in the dispute which was arisen`
  - Fix: `(1797, 'WHEN THE', 'eb_3rd_1797_v16_TRE-SCO', [('EDWARD', r'Edward\s+then\s+declared,', 0)])`

- 🔴 **NAUPLIUS** → **MER** (1815) sim=0.047 [topic_change] [gap: EDITORIAL]
  - `...o King Teuthras, to screen her from her father's resentment.`
  - `Mer. zenith dist. 34° 33' N. cof. 82369
Declination 22° 40' N.`
  - Fix: `(1815, 'NAUPLIUS', 'eb_5th_1815_v14_ENL-NIC', [('MER', r'Mer\.\s+zenith\s+dist\.', 0)])`

- 🔴 **GAZA** → **TREES** (1815) sim=0.051 [topic_change] [gap: EDITORIAL]
  - `...re, and the port is ruined. E. Long. 34. 55. N. Lat. 31. 28.`
  - `Trees that were budded last year, will now begin to push out their first shoots. Should they be infe`
  - Fix: `(1815, 'GAZA', 'eb_5th_1815_v09_FOR-CCX', [('TREES', r'Trees\s+that\s+were', 9)])`

- 🔴 **GERMANY** → **FOR** (1815) sim=0.052 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...r of her present ruler, and therefore will not be permanent.`
  - `For if AC, AD be joined, the triangles OAD, OCA, have the angle at O common to both, also the angle `
  - Fix: `(1815, 'GERMANY', 'eb_5th_1815_v09_FOR-CCX', [('FOR', r'For\s+if\s+AC,', 23)])`

- 🔴 **MASSACHUSETTS** → **TEMPORA** (1823) sim=0.052 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...contains about 1500 families. See UNITED STATES, SUPPLEMENT.`
  - `Tempora mutantur, et nos mutamur in illis.

53. About the time of the knights templars, chivalry had`
  - Fix: `(1823, 'MASSACHUSETTS', 'eb_6th_1823_v12_ENL-ADD', [('TEMPORA', r'Tempora\s+mutantur,\s+et', 57)])`

- 🔴 **BOURDEAUX** → **ALLIED** (1842) sim=0.057 [topic_change] [gap: OCR_GAP]
  - `...besides the soldiery. Long. 0. 40. 4. W. Lat. 44. 50. 14. N.`
  - `Allied to Celestrinæ, to Euphorbiaceæ, to Rosaceæ,
and to Byttneriaceæ. The berries of several speci`
  - Fix: `(1842, 'BOURDEAUX', 'eb_7th_1842_v05_BOR-CAL', [('ALLIED', r'Allied\s+to\s+Celestrinæ,', 0)])`

- 🔴 **MERCIA** → **FOR** (1815) sim=0.059 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...s, it sunk into a province, or rather was divided into many.`
  - `For, let the semiellipse ADB, and semicircle AEB, revolve about the same fixed axis AB, and thus gen`
  - Fix: `(1815, 'MERCIA', 'eb_5th_1815_v13_MAT-CCC', [('FOR', r'For,\s+let\s+the', 5)])`

- 🔴 **LEICESTERSHIRE** → **FOR** (1842) sim=0.059 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...kley. The town of Leicester, as before, returns two members.`
  - `For the most extensive collection of the works of Leibnitz, we are indebted to Louis Dutens, who in `
  - Fix: `(1842, 'LEICESTERSHIRE', 'eb_7th_1842_v13_SEV-AB', [('FOR', r'For\s+the\s+most', 65)])`

- 🔴 **CONSTAT** → **ALL** (1810) sim=0.067 [topic_change] [gap: OCR_GAP]
  - `... of the enrolment of any letters patent is called a constat.`
  - `All this time the Turks had been continuing their War with encroachments on the empire, which, had i`
  - Fix: `(1810, 'CONSTAT', 'eb_4th_1810_v06_CON-CRY', [('ALL', r'All\s+this\s+time', 28)])`

- 🔴 **UTICA** → **COL** (1810) sim=0.067 [new_headword] [gap: OCR_GAP]
  - `... the throne to his son Phraortes, after a reign of 53 years.`
  - `COL. B. A. Colonia Braccara Augusta, Brague
COL. BRYT. L. V. Colonia Berytus Legio Quinta
COL. CABE.`
  - Fix: `(1810, 'UTICA', 'eb_4th_1810_v20_SUI-PRE', [('COL', r'COL\.\s+B\.\s+A\.', 0)])`

- 🔴 **MAY** → **THOMAS** (1860) sim=0.067 [new_headword] (2 eds: 1815, 1860)
  - `...ng in some places to the height of 160 feet. Pop. (1851) 18.`
  - `THOMAS, an English historian and poet, was born in 1595 of an ancient family in Sussex. He was educa`
  - Fix: `(1860, 'MAY', 'eb_8th_1860_v14_MAG-NOT', [('THOMAS', r'THOMAS,\s+an\s+English', 52)])`

- 🔴 **OSCHOPHORIA** → **SQUACCO HERON.** (1823) sim=0.073 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...f five different things, wine, honey, cheese, meal, and oil.`
  - `**Squacco heron.**—Ferruginous; white beneath; hind head with a long white pendent crest, edged with`
  - Fix: `(1823, 'OSCHOPHORIA', 'eb_6th_1823_v15_ENL-PAR', [('SQUACCO HERON.', r'\*\*Squacco\s+heron\.\*\*—Ferruginous;\s+white', 0)])`

- 🔴 **MYAGRUM** → **THESE** (1797) sim=0.075 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...d with geese. Horses, goats, sheep, and cows, eat the plant.`
  - `These variations in the fundamental bass, as well in the chord concerning which we now treat, as in `
  - Fix: `(1797, 'MYAGRUM', 'eb_3rd_1797_v12_TRE-NEG', [('THESE', r'These\s+variations\s+in', 0)])`

- 🔴 **MEMNON** → **GENUS XCVI** (1815) sim=0.075 [new_headword] [gap: EDITORIAL]
  - `...arius's wife, and Alexander had a son by her named Hercules.`
  - `GENUS XCVI. DYSECOEA.

DEAFNESS, or Difficulty of Hearing.`
  - Fix: `(1815, 'MEMNON', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XCVI', r'GENUS\s+XCVI\.\s+DYSECOEA\.', 0)])`

- 🔴 **ALFERGAN** → **FOR** (1823) sim=0.075 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...tself contains thirty chapters, all of which are very short.`
  - `For the last four years of a lease, the same cove-
nants are generally sufficient, only they require`
  - Fix: `(1823, 'ALFERGAN', 'eb_6th_1823_v01_ART-AME', [('FOR', r'For\s+the\s+last', 0)])`

- 🔴 **PHEGOR** → **PELLANDRIUM** (1797) sim=0.078 [new_headword] (2 eds: 1797, 1810)
  - `...at Phegor was the sun presiding over the mysteries of Venus.`
  - `PELLANDRIUM, water-hemlock; a genus of the digynia order, belonging to the pentandria class of plant`
  - Fix: `(1797, 'PHEGOR', 'eb_3rd_1797_v14_TRE-PLA', [('PELLANDRIUM', r'PELLANDRIUM,\s+water\-hemlock;\s+a', 0)])`

- 🔴 **GARTH** → **PLINY** (1815) sim=0.080 [topic_change] [gap: EDITORIAL]
  - `...illness, which he bore with great patience, in January 1719.`
  - `Pliny gives a different account of the origin of grafting: he says, a husbandman willing to make a p`
  - Fix: `(1815, 'GARTH', 'eb_5th_1815_v09_FOR-CCX', [('PLINY', r'Pliny\s+gives\s+a', 0)])`

- 🔴 **BRADSHAW** → **ALLIED** (1842) sim=0.080 [topic_change] [gap: OCR_GAP]
  - `...ty, though, of course, with very considerable embellishment.`
  - `Allied to Santalaceae, Elaeocarpaceae, and Proteaceae, from which they are readily known by one or t`
  - Fix: `(1842, 'BRADSHAW', 'eb_7th_1842_v05_BOR-CAL', [('ALLIED', r'Allied\s+to\s+Santalaceae,', 11)])`

- 🔴 **LEICESTER** → **VARIETY** (1842) sim=0.080 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ported by subscriptions; a new theatre, and two new schools.`
  - `Variety in unity, or unity varied, the expressive sign of every masterpiece of nature and of art, ch`
  - Fix: `(1842, 'LEICESTER', 'eb_7th_1842_v13_SEV-AB', [('VARIETY', r'Variety\s+in\s+unity,', 20)])`

- 🔴 **VANDYCK** → **ANNE** (1842) sim=0.080 [new_headword] [gap: OCR_GAP]
  - `...wton-Averbacham, but by her second husband she had no issue.`
  - `ANNE, a thin slip of bunting hung to the mast-head, or some other conspicuous place in the ship, to `
  - Fix: `(1842, 'VANDYCK', 'eb_7th_1842_v21_SEV-ZYG', [('ANNE', r'ANNE,\s+a\s+thin', 25)])`

- 🔴 **BERWICK** → **NORTH** (1815) sim=0.084 [topic_change] [gap: VARIANT]
  - `...24946
30875
24946

Increase, 5929`
  - `North, a royal borough, and sea-port in the county of East Lothian in Scotland. W. Long. 2. 29. N. L`
  - Fix: `(1815, 'BERWICK', 'eb_5th_1815_v03_ASS-DIR', [('NORTH', r'North,\s+a\s+royal', 52)])`

- 🔴 **POECILE** → **FOR** (1823) sim=0.084 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... in the Poecile in commemoration of that celebrated victory.`
  - `For this purpose let H, expressed in feet, be the height through which a heavy body must fall in ord`
  - Fix: `(1823, 'POECILE', 'eb_6th_1823_v16_ENL-BRE', [('FOR', r'For\s+this\s+purpose', 61)])`

- 🔴 **PHIDIAS** → **TARTARIZED** (1797) sim=0.085 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ce, which consisted in keeping clean this magnificent image.`
  - `Tartarized soda, commonly called Rochel salt. E.

The Rochel salt may be prepared from purified foss`
  - Fix: `(1797, 'PHIDIAS', 'eb_3rd_1797_v14_TRE-PLA', [('TARTARIZED', r'Tartarized\s+soda,\s+commonly', 0)])`

- 🔴 **JESUITS** → **STERLET** (1823) sim=0.085 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ally suppressed and abolished by Pope Clement XIV., in 1773.`
  - `Sterlet sturgeon.—Brownish, with the sides spotted rufescens, with pale red, and the body shielded a`
  - Fix: `(1823, 'JESUITS', 'eb_6th_1823_v11_ENL-LIE', [('STERLET', r'Sterlet\s+sturgeon\.—Brownish,\s+with', 25)])`

- 🔴 **COMEDY** → **ORDER** (1810) sim=0.087 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... one to make nature known, the other to make her ridiculous.`
  - `Order of the Cards before shuffling.

1 Seven hearts
2 Nine clubs
3 Eight hearts
4 Eight spades
5 Kn`
  - Fix: `(1810, 'COMEDY', 'eb_4th_1810_v17_OBS-GEN', [('ORDER', r'Order\s+of\s+the', 15)])`

- 🔴 **BELOOCHISTAN** → **FOR** (1823) sim=0.087 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...most every species to be met with either in Europe or India.`
  - `For the information contained in this article we`
  - Fix: `(1823, 'BELOOCHISTAN', 'eb_6th_1823_v502_AUS-CEL', [('FOR', r'For\s+the\s+information', 86)])`

- 🔴 **ALMANACK** → **THEOR** (1823) sim=0.088 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...l potations, which our ancestors indulged in at that period.`
  - `Theor. VII. Cos.(a−b)+cos.(a+b)=2cos.a×cos.b.
Theor. VIII. Cos.(a−b)−cos.(a+b)=2sin.a×sin.b.

If in `
  - Fix: `(1823, 'ALMANACK', 'eb_6th_1823_v01_ART-AME', [('THEOR', r'Theor\.\s+VII\.\s+Cos\.\(a−b\)\+cos\.\(a\+b\)=2cos\.a×cos\.b\.', 11)])`

- 🔴 **ANHALT** → **FOR** (1815) sim=0.091 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ered by the Saale and Mulda; its principal trade is in beer.`
  - `For instance, of those animals who have blood-vessels and a double circulation, some respire by admi`
  - Fix: `(1815, 'ANHALT', 'eb_5th_1815_v02_ENL-ASS', [('FOR', r'For\s+instance,\s+of', 20)])`

- 🔴 **ANNULOSA** → **PORTANUS** (1823) sim=0.091 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ot issues from the seed. (Phys. des Arbres, Tom. II. p. 50.)`
  - `Portanus forceps.

Fabr. Suppl. Ent. Syst. 368.`
  - Fix: `(1823, 'ANNULOSA', 'eb_6th_1823_v02_ENL-ASS', [('PORTANUS', r'Portanus\s+forceps\.\s+Fabr\.', 184)])`

- 🔴 **WITCHCRAFT** → **FOR** (1810) sim=0.092 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...n part, L. 17 1 0 Scots.
Both, L. 34 11 0
Or L. 2 17 7 Ster.`
  - `For a considerable time after the inquisition was erected, the trials of witches (as heretics) were `
  - Fix: `(1810, 'WITCHCRAFT', 'eb_4th_1810_v20_SUI-PRE', [('FOR', r'For\s+a\s+considerable', 39)])`

- 🔴 **LEGION** → **FOR** (1815) sim=0.092 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...are told, was the first who changed all these for the eagle.`
  - `For entertaining experiments, illusions, &c. of a philosophical nature, see the articles Acoustics, `
  - Fix: `(1815, 'LEGION', 'eb_5th_1815_v11_ENL-LIE', [('FOR', r'For\s+entertaining\s+experiments,', 45)])`

- 🔴 **MEDICINE** → **GASTRITIS** (1815) sim=0.094 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...is afterwards repeated. See Ferguson's Lecl. vol. i. p. 118.`
  - `Gastritis legitima, Sauv. fp. 1. Eller. de cogn. et cur. morbi febr. xii. Haller. obf. 14. hift. 3. `
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GASTRITIS', r'Gastritis\s+legitima,\s+Sauv\.', 36)])`

- 🔴 **ARISTOXENUS** → **FOR** (1842) sim=0.095 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...rder, carrying the tens to the higher place, as in addition.`
  - `For the convenient arrangement of rhetorical arguments, Aristotle divides rhetoric into three differ`
  - Fix: `(1842, 'ARISTOXENUS', 'eb_7th_1842_v03_SEV-AST', [('FOR', r'For\s+the\s+convenient', 58)])`

- 🔴 **LEGIO VII** → **FOR** (1797) sim=0.098 [topic_change] [gap: VARIANT]
  - `...mount Tabor and the Mediterranean. Now thought to be Legume.`
  - `For entertaining experiments, illusions, &c. of a philosophical nature, see the articles Acoustics, `
  - Fix: `(1797, 'LEGIO VII', 'eb_3rd_1797_v09_IND-LES', [('FOR', r'For\s+entertaining\s+experiments,', 0)])`

- 🔴 **WARDSHIP** → **SECONDLY** (1810) sim=0.098 [topic_change] [gap: OCR_GAP]
  - `...nd it was accordingly abolished by statute 16 Car. I. c. 20.`
  - `Secondly, when the enemy is to leeward.—If the lee fleet keep close to the wind in the order of batt`
  - Fix: `(1810, 'WARDSHIP', 'eb_4th_1810_v20_SUI-PRE', [('SECONDLY', r'Secondly,\s+when\s+the', 40)])`

- 🔴 **MIDAS** → **BAR** (1778) sim=0.099 [topic_change] [gap: VARIANT]
  - `...ged, gave him a pair of asses' ears. See the article APOLLO.`
  - `Bar-bell, the smooth ovate-oblong buccinum, with an oblong and very narrow mouth. It consists of six`
  - Fix: `(1778, 'MIDAS', 'eb_2nd_1778_v07_BYW-OPT', [('BAR', r'Bar\-bell,\s+the\s+smooth', 0)])`

- 🔴 **EAGLES** → **INSTEAD** (1842) sim=0.099 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ion of effects, to import any more of them into the kingdom.`
  - `Instead of giving any more particular cases, we may observe in general, that if the intensity of the`
  - Fix: `(1842, 'EAGLES', 'eb_7th_1842_v08_DIA-VII', [('INSTEAD', r'Instead\s+of\s+giving', 28)])`

- 🔴 **LETHARGY** → **FOR** (1842) sim=0.100 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...making those who drank them forget everything that was past.`
  - `For a few years before the fatal one above mentioned, his occupations had been agreeably diversified`
  - Fix: `(1842, 'LETHARGY', 'eb_7th_1842_v13_SEV-AB', [('FOR', r'For\s+a\s+few', 28)])`

- 🔴 **SOUTH** → **ROBERT** (1842) sim=0.102 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...one of the four cardinal points from which the winds blow.`
  - `Robert, an eminent divine, was the son of William South, a merchant of London, and was born at Hackn`
  - Fix: `(1842, 'SOUTH', 'eb_7th_1842_v20_SEV-SUG', [('ROBERT', r'Robert,\s+an\s+eminent', 0)])`

- 🔴 **MEDINA** → **GENUS LXXXVI** (1815) sim=0.103 [new_headword] [gap: EDITORIAL]
  - `... set for Ovid's Metamorphoses, but they were never engraved.`
  - `GENUS LXXXVI. SCORBUTUS.

SCURVY.`
  - Fix: `(1815, 'MEDINA', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXXVI', r'GENUS\s+LXXXVI\.\s+SCORBUTUS\.', 0)])`

- 🔴 **JESUS** → **ARISTOBULUS** (1842) sim=0.103 [topic_change] [gap: OCR_GAP]
  - `...xclude from Christianity every trace of supernatural agency.`
  - `Aristobulus, his sons Alexander and Antigonus, and his two daughters, were carried away by Pompey as`
  - Fix: `(1842, 'JESUS', 'eb_7th_1842_v12_DEF-PLA', [('ARISTOBULUS', r'Aristobulus,\s+his\s+sons', 28)])`

- 🔴 **REVOLUTION** → **FOR** (1842) sim=0.103 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...inity is well cultivated. Long. 81° 25'. E. Lat. 24° 27'. N.`
  - `For an account of the experiments of Coulomb, Hutton, and Vince, the reader is referred to Hydrodyna`
  - Fix: `(1842, 'REVOLUTION', 'eb_7th_1842_v19_SEV-SCU', [('FOR', r'For\s+an\s+account', 61)])`

- 🔴 **WHEN THE** → **FOR** (1797) sim=0.105 [topic_change] [gap: VARIANT]
  - `...CXXXVII. | 203 |
| CCCCXXXVIII. | 214 |
| CCCCXXXIX. | 304 |`
  - `For these reasons, as it is said, the regency put into the hands of Edward all the forts in the coun`
  - Fix: `(1797, 'WHEN THE', 'eb_3rd_1797_v16_TRE-SCO', [('FOR', r'For\s+these\s+reasons,', 0)])`

- 🔴 **FINERY** → **INSTANCES** (1823) sim=0.106 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...**FINERING.** See Veneering.`
  - `Instances have occurred of violent inflammation excited in the stomach by the bots. An example of th`
  - Fix: `(1823, 'FINERY', 'eb_6th_1823_v08_ENL-FOR', [('INSTANCES', r'Instances\s+have\s+occurred', 45)])`

- 🔴 **PHERECYDES** → **TARTARIZED** (1797) sim=0.107 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...f which the reader will find accounts in different articles.`
  - `Tartarized kali. L.

Take of kali one pound; crystals of tartar, three pounds; distilled water, boil`
  - Fix: `(1797, 'PHERECYDES', 'eb_3rd_1797_v14_TRE-PLA', [('TARTARIZED', r'Tartarized\s+kali\.\s+L\.', 23)])`

- 🔴 **CHROMATICS** → **FOR** (1815) sim=0.107 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ur estates do, though they have the ceremony of an election.`
  - `For making his experiments, Mr Delaval used small phials of flint-glass, whose form was a parallelop`
  - Fix: `(1815, 'CHROMATICS', 'eb_5th_1815_v06_ENL-CRY', [('FOR', r'For\s+making\s+his', 49)])`

- 🔴 **PROBABILITY** → **FOR** (1842) sim=0.107 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...pending upon the manner in which the prisoners are employed.`
  - `For the sake of rendering the solution more general, let \( a \) be the number of white balls in the`
  - Fix: `(1842, 'PROBABILITY', 'eb_7th_1842_v18_PLA-QUO', [('FOR', r'For\s+the\s+sake', 31)])`

- 🔴 **ARMAGEDDON** → **ALL** (1810) sim=0.108 [topic_change] [gap: OCR_GAP]
  - `...of the gospel, and others, the mountain of apples or fruits.`
  - `All mixed circulates are derived from vulgar fractions.
tions of this kind, whose denominators are m`
  - Fix: `(1810, 'ARMAGEDDON', 'eb_4th_1810_v02_ANT-ASS', [('ALL', r'All\s+mixed\s+circulates', 0)])`

- 🔴 **BURNING** → **VILLETTE** (1815) sim=0.112 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ng there, till the flood-gates of life shut in eternal rest.`
  - `Villette, a French artist of Lyons, made a large mirror, which was bought by Tavernier and presented`
  - Fix: `(1815, 'BURNING', 'eb_5th_1815_v05_ENL-CHI', [('VILLETTE', r'Villette,\s+a\s+French', 9)])`

- 🔴 **ANDROMEDA** → **MARIS CYCLUS** (1797) sim=0.116 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...hearsing verses, and acting parts of this piece. See Abdera.`
  - `Maris Cyclus: A genus of the monogynia order, belonging to the decandria clas of plants; and in the `
  - Fix: `(1797, 'ANDROMEDA', 'eb_3rd_1797_v01_IND-COR', [('MARIS CYCLUS', r'Maris\s+Cyclus:\s+A', 13)])`

- 🔴 **MEXICO** → **FOR** (1823) sim=0.116 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...at once an extraordinary mixture of good faith and distrust.`
  - `For "if the Divine substance be infinitely extended, then will there be part of it in this place and`
  - Fix: `(1823, 'MEXICO', 'eb_6th_1823_v13_ENL-MIC', [('FOR', r'For\s+"if\s+the', 88)])`

- 🔴 **WHARTON** → **MILLS** (1823) sim=0.117 [topic_change] [gap: VARIANT]
  - `...rdship also began a play on the story of the queen of Scots.`
  - `Mills of this kind are much in use in the south of
Europe. The wheel is horizontal, and the vertical`
  - Fix: `(1823, 'WHARTON', 'eb_6th_1823_v20_ENL-ZYG', [('MILLS', r'Mills\s+of\s+this', 40)])`

- 🔴 **ICHTHYOPHAGI** → **GEN** (1815) sim=0.118 [topic_change] [gap: EDITORIAL]
  - `...e Thames at Goring, and extends to the west part of England.`
  - `Gen. 3. Amia.

Head bony, naked, rough, and furnished with futures; teeth acute, and close in the ja`
  - Fix: `(1815, 'ICHTHYOPHAGI', 'eb_5th_1815_v11_ENL-LIE', [('GEN', r'Gen\.\s+3\.\s+Amia\.', 35)])`

- 🔴 **NEBULY** → **SUN** (1823) sim=0.120 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...e outline of a bordure, ordinary, &c., is indented or waved.`
  - `Sun's declin. per Nautical Almanack = 16° 44.3
Equation to 4h 37' P.M. to 38° 30' W = + 3.4 + 1.8`
  - Fix: `(1823, 'NEBULY', 'eb_6th_1823_v14_ENL-NIC', [('SUN', r'Sun's\s+declin\.\s+per', 0)])`

- 🔴 **WAIGATZ** → **ALL** (1842) sim=0.120 [topic_change] [gap: OCR_GAP]
  - `..., Dolgoi, and Bilinor, whose products are of a similar kind.`
  - `All these results show that the power of inducing electric currents is circumferentially exerted by `
  - Fix: `(1842, 'WAIGATZ', 'eb_7th_1842_v21_SEV-ZYG', [('ALL', r'All\s+these\s+results', 0)])`

- 🔴 **MUTILATION** → **FOR** (1797) sim=0.121 [topic_change] [gap: VARIANT]
  - `... declared was the most abominable and disgraceful of crimes.`
  - `For example, let us suppose, that in the fundamental bass we have a dominant sol carrying the chord `
  - Fix: `(1797, 'MUTILATION', 'eb_3rd_1797_v12_TRE-NEG', [('FOR', r'For\s+example,\s+let', 22)])`

- 🔴 **COINAGE** → **ARCHITECTURE** (1810) sim=0.122 [topic_change] [gap: OCR_GAP]
  - `...d Segovia, the only cities where gold and silver are struck.`
  - `Architecture, a kind of dye cut diagonally, after the manner of a flight of a staircase, serving at `
  - Fix: `(1810, 'COINAGE', 'eb_4th_1810_v17_OBS-GEN', [('ARCHITECTURE', r'Architecture,\s+a\s+kind', 80)])`

- 🔴 **MELLI** → **GENUS XCI** (1815) sim=0.122 [new_headword] [gap: EDITORIAL]
  - `... It was formerly the residence of one of the English chiefs.`
  - `GENUS XCI. ICTERUS.

The JAUNDICE.`
  - Fix: `(1815, 'MELLI', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XCI', r'GENUS\s+XCI\.\s+ICTERUS\.', 9)])`

- 🔴 **IRELAND** → **STATISTICS** (1842) sim=0.122 [new_headword] [gap: VARIANT]
  - `...in a question, no longer of Irish, but of imperial interest.`
  - `STATISTICS.

The island of Ireland is of a rhomboidal shape, having its longer sides nearly in the d`
  - Fix: `(1842, 'IRELAND', 'eb_7th_1842_v12_DEF-PLA', [('STATISTICS', r'STATISTICS\.\s+The\s+island', 31)])`

- 🔴 **TELL** → **AND** (1823) sim=0.123 [topic_change] [gap: VARIANT]
  - `...ken to harshly for any offence which it can possibly commit.`
  - `And

\[
\frac{1}{a} - 1 = \frac{1-a}{a}.
\]`
  - Fix: `(1823, 'TELL', 'eb_6th_1823_v20_ENL-ZYG', [('AND', r'And\s+\\\[\s+\\frac\{1\}\{a\}', 21)])`

- 🔴 **ALMANZA** → **HERESY OF ALMARIC** (1815) sim=0.124 [new_headword] (2 eds: 1815, 1823)
  - `...s, had prevented him from seeing, or giving orders properly.`
  - `HERESY OF ALMARIC, a tenet broached in France by one Almaric, in the year 1209. It consisted in affi`
  - Fix: `(1815, 'ALMANZA', 'eb_5th_1815_v01_ENL-AME', [('HERESY OF ALMARIC', r'HERESY\s+OF\s+ALMARIC,', 0)])`

- 🔴 **JERUSALEM** → **FISHES** (1815) sim=0.124 [topic_change] [gap: VARIANT]
  - `..., and the other was done; not only here, but all over Judea.`
  - `Fishes, sense of seeing of,
hearing of,
touch of,
taste of,
smelling of,
motions of,
instruments of `
  - Fix: `(1815, 'JERUSALEM', 'eb_5th_1815_v11_ENL-LIE', [('FISHES', r'Fishes,\s+sense\s+of', 37)])`

- 🔴 **MITHRAS** → **MARINE** (1797) sim=0.125 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... hand, while with the other he plunges a dagger in his neck.`
  - `Marine acid likewise dissolves iron, and this solution is also incrustifiable.`
  - Fix: `(1797, 'MITHRAS', 'eb_3rd_1797_v12_TRE-NEG', [('MARINE', r'Marine\s+acid\s+likewise', 0)])`

- 🔴 **ALMANZA** → **HERESY OF ALMARIC** (1823) sim=0.125 [new_headword] (2 eds: 1815, 1823)
  - `...s, had prevented him from seeing, or giving orders properly.`
  - `HERESY OF ALMARIC, a tenet broached in France by one Almaric, in the year 1209. It consisted in affi`
  - Fix: `(1823, 'ALMANZA', 'eb_6th_1823_v01_ART-AME', [('HERESY OF ALMARIC', r'HERESY\s+OF\s+ALMARIC,', 0)])`

- 🔴 **MECHANICS** → **THEORY** (1823) sim=0.125 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ctures of the Kaba, and selling them to pilgrims. See CAABA.`
  - `Theory. As the sine of the plane's inclination, is to the sine of the angle formed by the direction `
  - Fix: `(1823, 'MECHANICS', 'eb_6th_1823_v13_ENL-MIC', [('THEORY', r'Theory\.\s+As\s+the', 25)])`

- 🔴 **CHRYSOLITE** → **MARINE** (1797) sim=0.127 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... says that the colour of the chrysolite is yellow like gold.`
  - `Marine society established at London.

The King of Prussia commenced hostilities in the month of Aug`
  - Fix: `(1797, 'CHRYSOLITE', 'eb_3rd_1797_v04_TRE-OMI', [('MARINE', r'Marine\s+society\s+established', 2)])`

- 🔴 **PITCAIRNE** → **DOM** (1842) sim=0.127 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...his personal history Pitcairne makes the following allusion:`
  - `Dom procul a Geneva se summovet inclita Roma,
Quod quam sit longum seit Cyphianus iter.`
  - Fix: `(1842, 'PITCAIRNE', 'eb_7th_1842_v17_SEV-CON', [('DOM', r'Dom\s+procul\s+a', 25)])`

- 🔴 **ASTRONOMY** → **HALLEY** (1797) sim=0.129 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...atque ardua calli
Scandere sublimis geniti consecit acumen.*`
  - `HALLEY.

Sir Isaac Newton having already made the great discovery of an universal and mutual deflect`
  - Fix: `(1797, 'ASTRONOMY', 'eb_3rd_1797_v501_ABE-IMP', [('HALLEY', r'HALLEY\.\s+Sir\s+Isaac', 41)])`

- 🔴 **ARISTOXENUS** → **ALL** (1842) sim=0.129 [topic_change] [gap: OCR_GAP]
  - `...what had been done would meet with a favourable acceptance."`
  - `All numbers are represented by the ten following characters:

1 2 3 4 5 6 7 8 9 0`
  - Fix: `(1842, 'ARISTOXENUS', 'eb_7th_1842_v03_SEV-AST', [('ALL', r'All\s+numbers\s+are', 38)])`

- 🔴 **CHINON** → **ALL** (1842) sim=0.129 [topic_change] [gap: OCR_GAP]
  - `...ice, and similar products. Long. 0.5.40. E. Lat. 47.11.4. N.`
  - `All bodies having a strong affinity for oxygen, and which are placed in contact with indigo and lime`
  - Fix: `(1842, 'CHINON', 'eb_7th_1842_v06_SEV-CLO', [('ALL', r'All\s+bodies\s+having', 0)])`

- 🔴 **ARMADILLA** → **ALL** (1810) sim=0.131 [topic_change] [gap: OCR_GAP]
  - `...e at Calao, a port of Lima; that of the latter at Cartagena.`
  - `All vulgar fractions, whose denominators are multiples of 2, 5, or their powers, except those alread`
  - Fix: `(1810, 'ARMADILLA', 'eb_4th_1810_v02_ANT-ASS', [('ALL', r'All\s+vulgar\s+fractions,', 45)])`

- 🔴 **HORTICULTURE** → **BROCCOLI** (1842) sim=0.131 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hey occupy; and hence they have never come into general use.`
  - `Broccoli has a close affinity to cauliflower, being, like it, of Italian origin, and differing chief`
  - Fix: `(1842, 'HORTICULTURE', 'eb_7th_1842_v11_GRO-HYD', [('BROCCOLI', r'Broccoli\s+has\s+a', 0)])`

- 🔴 **BREUGHEL** → **GOOD** (1810) sim=0.132 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...res, which generally consisted of country-dances, marriages,`
  - `Good tea, in a moderate quantity, seems to refresh and strengthen; but if taken in a recent highly f`
  - Fix: `(1810, 'BREUGHEL', 'eb_4th_1810_v04_BOO-BRE', [('GOOD', r'Good\s+tea,\s+in', 0)])`

- 🔴 **CONSTANTINE** → **ROBERT** (1842) sim=0.133 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...tines, 1123 years after its establishment at Constantinople.`
  - `Robert, a learned physician, born at Cean. He taught polite literature in that city, and acquired gr`
  - Fix: `(1842, 'CONSTANTINE', 'eb_7th_1842_v07_SEV-DIA', [('ROBERT', r'Robert,\s+a\s+learned', 5)])`

- 🔴 **GERMANICUS CAESAR** → **THEOREM** (1823) sim=0.134 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ek comedies and Latin poems, some of which are still extant.`
  - `Theorem VII.

If there be any number of quantities, and as many others, which taken two and two in a`
  - Fix: `(1823, 'GERMANICUS CAESAR', 'eb_6th_1823_v09_FOR-DIR', [('THEOREM', r'Theorem\s+VII\.\s+If', 0)])`

- 🔴 **JEFFREYS** → **BASSE** (1823) sim=0.134 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... only because she was granddaughter to the inhuman Jeffreys.`
  - `Basse sciaena, or basse. Perca labrax of Lin.—Subargenteous, with brown back, yellowish-red fins, an`
  - Fix: `(1823, 'JEFFREYS', 'eb_6th_1823_v11_ENL-LIE', [('BASSE', r'Basse\s+sciaena,\s+or', 0)])`

- 🔴 **GREEN-HOUSE** → **WEST GREENLAND** (1823) sim=0.135 [person_bio] [gap: VARIANT]
  - `...ar-houses, all carried on for exportation to a great extent.`
  - `West Greenland was first peopled by Europeans in the eighth century. At that time a company of Icela`
  - Fix: `(1823, 'GREEN-HOUSE', 'eb_6th_1823_v10_ENL-HYD', [('WEST GREENLAND', r'West\s+Greenland\s+was', 9)])`

- 🔴 **MARQUESAS ISLANDS** → **FOR** (1823) sim=0.135 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... one family, of which the eldest born is the chief or king."`
  - `For several other particulars respecting the horse,
especially on the use of that animal among the J`
  - Fix: `(1823, 'MARQUESAS ISLANDS', 'eb_6th_1823_v12_ENL-ADD', [('FOR', r'For\s+several\s+other', 34)])`

- 🔴 **GELLI** → **FOR** (1842) sim=0.135 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...Gelli died in 1563.`
  - `For lighting churches, theatres, and other public buildings, where a strong and uniform light is req`
  - Fix: `(1842, 'GELLI', 'eb_7th_1842_v10_SEV-GRO', [('FOR', r'For\s+lighting\s+churches,', 35)])`

- 🔴 **PERSIUS FLACCUS** → **NOR** (1842) sim=0.136 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...oduction to the life and writings of Persius, Leipzig, 1809.`
  - `Nor was Aga Mahommed's cruel treatment of the inhabitants of Kerman less shocking. This place was th`
  - Fix: `(1842, 'PERSIUS FLACCUS', 'eb_7th_1842_v17_SEV-CON', [('NOR', r'Nor\s+was\s+Aga', 35)])`

- 🔴 **REVIVIFICATION** → **COMMISSION OF REVIEW** (1815) sim=0.137 [new_headword] (2 eds: 1797, 1815)
  - `... must have his judgment warped by some passion or prejudice.`
  - `COMMISSION OF REVIEW, is a commission sometimes granted, in extraordinary cases, to revile the sente`
  - Fix: `(1815, 'REVIVIFICATION', 'eb_5th_1815_v17_ENL-RHI', [('COMMISSION OF REVIEW', r'COMMISSION\s+OF\s+REVIEW,', 61)])`

- 🔴 **MODESTY** → **CONSTITUENT PARTS** (1823) sim=0.137 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...or the pleasures of one, contribute to the amusement of all.`
  - `Constituent Parts. Klaproth.

| Copper | 63.7 |
|--------|------|
| Iron   | 12.7 |
| Sulphur| 19   `
  - Fix: `(1823, 'MODESTY', 'eb_6th_1823_v14_ENL-NIC', [('CONSTITUENT PARTS', r'Constituent\s+Parts\.\s+Klaproth\.', 10)])`

- 🔴 **CHUCKIAH** → **THESE** (1797) sim=0.138 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...d communion with each other, and are in fact but one church.`
  - `These wheels are from 20 to 40 feet in diameter, according to the height of the bank and consequent `
  - Fix: `(1797, 'CHUCKIAH', 'eb_3rd_1797_v501_ABE-IMP', [('THESE', r'These\s+wheels\s+are', 0)])`

- 🔴 **POGO** → **MEL** (1815) sim=0.138 [topic_change] [gap: VARIANT]
  - `...0 north by east of Bordeaux. E. Long. o. 25. N. Lat. 46. 35.`
  - `Mel. What great occasion call'd you hence to Rome?
Tiz. Freedom, which came at length, tho' slow to `
  - Fix: `(1815, 'POGO', 'eb_5th_1815_v17_ENL-RHI', [('MEL', r'Mel\.\s+What\s+great', 20)])`

- 🔴 **FIRMICUS MATERNUS** → **INSTEAD** (1823) sim=0.138 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... when such absurdities would constitute a part of his creed.`
  - `Instead of this absurd method of treatment, a feverish horse should, if possible, be put into a stab`
  - Fix: `(1823, 'FIRMICUS MATERNUS', 'eb_6th_1823_v08_ENL-FOR', [('INSTEAD', r'Instead\s+of\s+this', 20)])`

- 🔴 **GERMANDER** → **FOR** (1810) sim=0.139 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ek comedies and Latin poems, some of which are still extant.`
  - `For if DE is not parallel to BC, suppose some other line DO to be parallel to BC; then, \( AB : BD :`
  - Fix: `(1810, 'GERMANDER', 'eb_4th_1810_v09_FAR-GOT', [('FOR', r'For\s+if\s+DE', 0)])`

- 🔴 **MEDICINE** → **GENUS XIII** (1815) sim=0.139 [new_headword] [gap: EDITORIAL]
  - `...cylindrical wire, on which it vibrates as on a rolling axis.`
  - `GENUS XIII. CARDITIS.

Inflammation of the HEART.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XIII', r'GENUS\s+XIII\.\s+CARDITIS\.', 36)])`

- 🔴 **ELPHINSTON** → **EDSHEIMER** (1823) sim=0.139 [new_headword] (2 eds: 1815, 1823)
  - `...ts of Sir Thomas Fairfax, in the Bodleian library at Oxford.`
  - `EDSHEIMER, ADAM, a celebrated painter, born at Frankfort on the Maine, in 1574. He was first a disci`
  - Fix: `(1823, 'ELPHINSTON', 'eb_6th_1823_v08_ENL-FOR', [('EDSHEIMER', r'EDSHEIMER,\s+ADAM,\s+a', 28)])`

- 🔴 **SANCTION** → **FOR** (1842) sim=0.139 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...and that in pursuing the one he was combating for the other.`
  - `For the purpose of maintaining a rigid purity in speculative principles, he nominated a commission o`
  - Fix: `(1842, 'SANCTION', 'eb_7th_1842_v19_SEV-SCU', [('FOR', r'For\s+the\s+purpose', 1)])`

- 🔴 **BOTANY** → **ORDER IV** (1810) sim=0.140 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...e distempers, being highly beneficial in alleviating thirst.`
  - `ORDER IV. POLYANDRIA.

1320. Glabraria.
One species; viz. terfa. E. Indies.`
  - Fix: `(1810, 'BOTANY', 'eb_4th_1810_v04_BOO-BRE', [('ORDER IV', r'ORDER\s+IV\.\s+POLYANDRIA\.', 93)])`

- 🔴 **WEED** → **ALL** (1823) sim=0.140 [topic_change] [gap: OCR_GAP]
  - `...y of a load or vein of fine metal into an useless marcasite.`
  - `All this may be completely prevented by a few holes made in the start of each bucket. Air being at l`
  - Fix: `(1823, 'WEED', 'eb_6th_1823_v20_ENL-ZYG', [('ALL', r'All\s+this\s+may', 28)])`

- 🔴 **ARACHNIDES** → **VARIOUS** (1842) sim=0.140 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...Khalifat a duré." (Herbelot, Bibliothèque Orientale, Iaman.)`
  - `Various opinions have been entertained regarding the origin of those white, flaky, filamentous, silk`
  - Fix: `(1842, 'ARACHNIDES', 'eb_7th_1842_v03_SEV-AST', [('VARIOUS', r'Various\s+opinions\s+have', 67)])`

- 🔴 **IRON-MAKING** → **NEW** (1842) sim=0.141 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...devised system of cheap government, and even-handed justice.`
  - `New, an island in the eastern seas, north from New Britain. It is long and narrow, and extends from `
  - Fix: `(1842, 'IRON-MAKING', 'eb_7th_1842_v12_DEF-PLA', [('NEW', r'New,\s+an\s+island', 187)])`

- 🔴 **DYNAMICS** → **PART** (1810) sim=0.142 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...employed in the preparation of black vats for dyeing thread.`
  - `Part II.

74. Motions are infinitely diminished, the ultimate ratio of continually Bb to Ei is the r`
  - Fix: `(1810, 'DYNAMICS', 'eb_4th_1810_v07_STE-ELE', [('PART', r'Part\s+II\.\s+74\.', 35)])`

- 🔴 **PAN** → **ALL** (1810) sim=0.142 [topic_change] [gap: OCR_GAP]
  - `...stood it of none other than the son of Penelope and Mercury.`
  - `All other colours, as blue, &c. may be applied in the same manner. This method is the only one by wh`
  - Fix: `(1810, 'PAN', 'eb_4th_1810_v15_ORD-PAR', [('ALL', r'All\s+other\s+colours,', 17)])`

- 🔴 **PARADISE** → **STATES** (1842) sim=0.142 [topic_change] [gap: VARIANT]
  - `... which the souls of the blessed enjoy everlasting happiness.`
  - `States' Banks: 28
Union State Banks: 1

Total: 52,610,601`
  - Fix: `(1842, 'PARADISE', 'eb_7th_1842_v17_SEV-CON', [('STATES', r'States'\s+Banks:\s+28', 37)])`

- 🔴 **ORDNANCE** → **THESE** (1797) sim=0.143 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...p. 432, &c. See also Mineralogy, Part i. sect. 2. p. 61, &c.`
  - `These few hints for expressing the principal passions may, if duly attended to, suffice to direct ou`
  - Fix: `(1797, 'ORDNANCE', 'eb_3rd_1797_v13_TRE-PAS', [('THESE', r'These\s+few\s+hints', 57)])`

- 🔴 **NECESSITY** → **BEG** (1815) sim=0.143 [topic_change] [gap: EDITORIAL]
  - `...use the felony, which the killing would otherwise amount to.`
  - `Beg. of eclipse at Greenwich per Naut. Alm. 9h 23' 45'
Ship's longitude in time - 7° 19' 12"`
  - Fix: `(1815, 'NECESSITY', 'eb_5th_1815_v14_ENL-NIC', [('BEG', r'Beg\.\s+of\s+eclipse', 8)])`

- 🔴 **METHODISTS** → **MAY** (1823) sim=0.143 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ho use violent medicines, and pretended secrets or nostrums.`
  - `May not the appearance of the aurora borealis be owing to the union of oxygen and hydrogen by the in`
  - Fix: `(1823, 'METHODISTS', 'eb_6th_1823_v13_ENL-MIC', [('MAY', r'May\s+not\s+the', 17)])`

- 🔴 **ELPHINSTON** → **EDSHEIMER** (1815) sim=0.144 [new_headword] (2 eds: 1815, 1823)
  - `...ts of Sir Thomas Fairfax, in the Bodleian library at Oxford.`
  - `EDSHEIMER, ADAM, a celebrated painter, born at Frankfort on the Main, in 1574. He was first a discip`
  - Fix: `(1815, 'ELPHINSTON', 'eb_5th_1815_v08_ENL-FOR', [('EDSHEIMER', r'EDSHEIMER,\s+ADAM,\s+a', 28)])`

- 🔴 **ALBIN** → **PARTS** (1823) sim=0.144 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...AVG. Annona Augusti

A. N. F. F. Annum Novum Faustum Felicem`
  - `Parts

(c) The weather of the sails is the angle which the surface forms with the plane in which the`
  - Fix: `(1823, 'ALBIN', 'eb_6th_1823_v13_ENL-MIC', [('PARTS', r'Parts\s+\(c\)\s+The', 28)])`

- 🔴 **RETURN** → **THESE** (1797) sim=0.146 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... total effective; wanting to complete the establishment, &c.`
  - `These sentiments of a future state, conceived in a savage and a rude period, could not long prevail `
  - Fix: `(1797, 'RETURN', 'eb_3rd_1797_v16_TRE-SCO', [('THESE', r'These\s+sentiments\s+of', 6)])`

- 🔴 **ALIEN** → **FOR** (1815) sim=0.146 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...gland, and so called from their belonging to foreign abbeys.`
  - `For example, if the proposed equation be

$$x^3 - \frac{7}{4}x^2 + \frac{3}{4}x - 6 = 0.$$`
  - Fix: `(1815, 'ALIEN', 'eb_5th_1815_v01_ENL-AME', [('FOR', r'For\s+example,\s+if', 12)])`

- 🔴 **MAGNETISM** → **CLAUDIAN** (1815) sim=0.146 [new_headword] (2 eds: 1810, 1815)
  - `...ientia torpent
Membra fame, venaque fitis consumit apertas."`
  - `CLAUDIAN.

In the 16th century, the philosophers of modern times first began to speculate about the `
  - Fix: `(1815, 'MAGNETISM', 'eb_5th_1815_v12_LIE-CCX', [('CLAUDIAN', r'CLAUDIAN\.\s+In\s+the', 34)])`

- 🔴 **MEDICINE** → **GENUS XI** (1815) sim=0.146 [new_headword] [gap: EDITORIAL]
  - `...xes about five inches in diameter, and one and a half broad.`
  - `GENUS XI. PNEUMONIA.
Febris pneumonia, Hoffm. II. 136.

Sp. I. PERIPNEUMONIA.
Peripneumony, or Infla`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XI', r'GENUS\s+XI\.\s+PNEUMONIA\.', 34)])`

- 🔴 **OU-POEY-TSE** → **GREATER OUSE** (1815) sim=0.146 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... fluxions of the eyes and ears, and in many other disorders.`
  - `GREATER OUSE, a river which rises near Fitwell in Oxfordshire, and proceeds to Buckingham, Stony-Str`
  - Fix: `(1815, 'OU-POEY-TSE', 'eb_5th_1815_v15_NIC-CCC', [('GREATER OUSE', r'GREATER\s+OUSE,\s+a', 45)])`

- 🔴 **EVIL-MERODACH** → **EULER** (1823) sim=0.146 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...this prince was immediately succeeded by his son Belshazzar.`
  - `Euler, Leonard, professor of mathematics, member of the imperial academy of Petersburgh, ancient dir`
  - Fix: `(1823, 'EVIL-MERODACH', 'eb_6th_1823_v08_ENL-FOR', [('EULER', r'Euler,\s+Leonard,\s+professor', 0)])`

- 🔴 **JESUS** → **DIFFERENT** (1842) sim=0.146 [topic_change] [gap: VARIANT]
  - `...Judea being reduced to a subordinate principality. n. c. 63.`
  - `Different ground has been taken up by infidels in modern times, who, while they have been constraine`
  - Fix: `(1842, 'JESUS', 'eb_7th_1842_v12_DEF-PLA', [('DIFFERENT', r'Different\s+ground\s+has', 27)])`

- 🔴 **MARCIANITES** → **GENUS** (1810) sim=0.147 [topic_change] [gap: VARIANT]
  - `... copy of St Luke he threw out the two first chapters entire.`
  - `Genus 3. Moschus, Musk.

Horns wanting; front teeth eight in the lower jaw; tusks solitary in the up`
  - Fix: `(1810, 'MARCIANITES', 'eb_4th_1810_v12_MAH-ADD', [('GENUS', r'Genus\s+3\.\s+Moschus,', 23)])`

- 🔴 **ANGLES** → **ALL** (1815) sim=0.147 [topic_change] [gap: OCR_GAP]
  - `...honour of giving the name of Anglia to England. See England.`
  - `All the veins which bring back the blood from the upper extremities, and from the head and breast, p`
  - Fix: `(1815, 'ANGLES', 'eb_5th_1815_v02_ENL-ASS', [('ALL', r'All\s+the\s+veins', 0)])`

- 🔴 **COTTON** → **SWITZERLAND** (1842) sim=0.147 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ovement of Children, which has been several times reprinted.`
  - `Switzerland has been an exporting country for many years; and the Swiss goods, particularly fine twe`
  - Fix: `(1842, 'COTTON', 'eb_7th_1842_v07_SEV-DIA', [('SWITZERLAND', r'Switzerland\s+has\s+been', 11)])`

- 🔴 **LEE** → **NATHANIEL** (1842) sim=0.147 [new_headword] (3 eds: 1810, 1823, 1842)
  - `...ide the lee-side, and the larboard or left the weather-side.`
  - `NATHANIEL, a dramatic poet of the eighteenth century, was the son of a clergyman, who gave him a lib`
  - Fix: `(1842, 'LEE', 'eb_7th_1842_v13_SEV-AB', [('NATHANIEL', r'NATHANIEL,\s+a\s+dramatic', 61)])`

- 🔴 **SCOTLAND** → **FOR** (1842) sim=0.147 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...SECTION V.

A.D. 1306 TO 1436.`
  - `For the accomplishment of such ends, it was first necessary to exhibit a wholesome example of retrib`
  - Fix: `(1842, 'SCOTLAND', 'eb_7th_1842_v19_SEV-SCU', [('FOR', r'For\s+the\s+accomplishment', 20)])`

- 🔴 **UTICA** → **MEDIA** (1810) sim=0.148 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...nking under the 11th order, Sarmenitaceae. See Botany Index.`
  - `MEDIA, now the province of Ghilan in Persia, once the seat of a potent empire, was bounded, accordin`
  - Fix: `(1810, 'UTICA', 'eb_4th_1810_v20_SUI-PRE', [('MEDIA', r'MEDIA,\s+now\s+the', 0)])`

- 🔴 **BOW** → **ALLIED** (1842) sim=0.148 [topic_change] [gap: OCR_GAP]
  - `...k the bow-line is to slacken it when the wind becomes large.`
  - `Allied on the one hand to Salicaceae; and on the other to Saxifragaceae (Cunoniaceae): to Vochysiace`
  - Fix: `(1842, 'BOW', 'eb_7th_1842_v05_BOR-CAL', [('ALLIED', r'Allied\s+on\s+the', 43)])`

- 🔴 **GARNET** → **THESE** (1797) sim=0.149 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... be baked, in a pot well luted, in a potter's kiln 24 hours.`
  - `These observations, in the elegant performance whence they are extracted, the author illustrates by `
  - Fix: `(1797, 'GARNET', 'eb_3rd_1797_v07_TRE-GOA', [('THESE', r'These\s+observations,\s+in', 32)])`

- 🔴 **ENTOMOLOGY** → **ALL** (1815) sim=0.149 [topic_change] [gap: OCR_GAP]
  - `...72. Phalena. Antennae introrsum crassiores.

IV. NEUROPTERA.`
  - `All the polite arts having been buried under the ruins of the Roman empire, the art of engraving on `
  - Fix: `(1815, 'ENTOMOLOGY', 'eb_5th_1815_v08_ENL-FOR', [('ALL', r'All\s+the\s+polite', 9)])`

- 🔴 **RAMUS** → **PETER** (1823) sim=0.149 [topic_change] [gap: VARIANT]
  - `...llamenta; but both kinds are generally denominated surculus.`
  - `Peter, was one of the most famous professors of the 16th century. He was born in Picardy in 1515. A `
  - Fix: `(1823, 'RAMUS', 'eb_6th_1823_v17_ENL-RHI', [('PETER', r'Peter,\s+was\s+one', 0)])`

- 🔴 **TRIUMPH** → **THEOR. IX.** (1823) sim=0.149 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ed in the public places, and rejoicings appeared everywhere.`
  - `**Theor. IX.**

All the angles of a spherical triangle are together greater than two, and less than `
  - Fix: `(1823, 'TRIUMPH', 'eb_6th_1823_v20_ENL-ZYG', [('THEOR. IX.', r'\*\*Theor\.\s+IX\.\*\*\s+All', 4)])`

- 🔴 **BRITAIN** → **ORDER IV. POLYANDRIA.** (1823) sim=0.150 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ute diseases, being highly beneficial in alleviating thirst.`
  - `**ORDER IV. POLYANDRIA.**

1320. *Glabraria.*
One species; viz. tersa. E. Indies.`
  - Fix: `(1823, 'BRITAIN', 'eb_6th_1823_v502_AUS-CEL', [('ORDER IV. POLYANDRIA.', r'\*\*ORDER\s+IV\.\s+POLYANDRIA\.\*\*', 120)])`

- 🔴 **OU-POEY-TSE** → **GREATER OUSE** (1823) sim=0.150 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... fluxions of the eyes and ears, and in many other disorders.`
  - `GREATER OUSE, a river which rises near Fitwell in Oxfordshire, and proceeds to Buckingham, Stony-Str`
  - Fix: `(1823, 'OU-POEY-TSE', 'eb_6th_1823_v15_ENL-PAR', [('GREATER OUSE', r'GREATER\s+OUSE,\s+a', 20)])`

- 🔴 **CAMOENS** → **EARLY** (1823) sim=0.151 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... the cities or suburbs, as they shall judge most convenient.`
  - `Early in his life the misfortunes of the poet began. In his infancy, Simon Vaz de Caamans, his fathe`
  - Fix: `(1823, 'CAMOENS', 'eb_6th_1823_v05_ENL-CHI', [('EARLY', r'Early\s+in\s+his', 0)])`

- 🔴 **GORDIAN KNOT** → **FOR** (1842) sim=0.151 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...is sword, and thus either accomplished or eluded the oracle.`
  - `For a further account of the Hottentots, see the article *Africa*, vol. i. part ii. p. 226. Under th`
  - Fix: `(1842, 'GORDIAN KNOT', 'eb_7th_1842_v10_SEV-GRO', [('FOR', r'For\s+a\s+further', 0)])`

- 🔴 **TROY-WEIGHT** → **COR** (1810) sim=0.152 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...er-meat, unwrought pewter and lead, and some other articles.`
  - `Cor. In any triangle the greater angle is subtended by the greater side; and conversely. For if the `
  - Fix: `(1810, 'TROY-WEIGHT', 'eb_4th_1810_v02_THE-AND', [('COR', r'Cor\.\s+In\s+any', 23)])`

- 🔴 **GEORGE** → **COR** (1810) sim=0.152 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...mental purposes, to the churches they had planted in Canada.`
  - `Cor. One circumference of a circle cannot intersect another in more than two points, for if they cou`
  - Fix: `(1810, 'GEORGE', 'eb_4th_1810_v09_FAR-GOT', [('COR', r'Cor\.\s+One\s+circumference', 15)])`

- 🔴 **SANCTION** → **WILLIAM** (1842) sim=0.152 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... of Suffolk, and Wyatt the principal mover of the rebellion.`
  - `William the Conqueror now turned his attention to the north, where his authority had not yet been pr`
  - Fix: `(1842, 'SANCTION', 'eb_7th_1842_v19_SEV-SCU', [('WILLIAM', r'William\s+the\s+Conqueror', 3)])`

- 🔴 **HOLLAND** → **NEW** (1842) sim=0.153 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `..., von Z. M. Koning in Gravenhage; Almanach der 1834, Weimar.`
  - `New, the largest island in the world, reaching from $10^\circ$ to $40^\circ$ south latitude, and bet`
  - Fix: `(1842, 'HOLLAND', 'eb_7th_1842_v11_GRO-HYD', [('NEW', r'New,\s+the\s+largest', 93)])`

- 🔴 **TROGLODYTES** → **COR** (1810) sim=0.154 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...have only an abridgement by Justin, flourished about 41 B.C.`
  - `Cor. 1. If DE be drawn, the angle AED is a right angle, and DE being therefore at right angles to ev`
  - Fix: `(1810, 'TROGLODYTES', 'eb_4th_1810_v20_SUI-PRE', [('COR', r'Cor\.\s+1\.\s+If', 28)])`

- 🔴 **DYNAMICS** → **FOR** (1810) sim=0.154 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ed, as a smaller quantity of tin would thus be precipitated.`
  - `For, let m be the number of particles in the system. Suppose any particle to move uniformly in any d`
  - Fix: `(1810, 'DYNAMICS', 'eb_4th_1810_v07_STE-ELE', [('FOR', r'For,\s+let\s+m', 45)])`

- 🔴 **FRANCE** → **MOREOVER** (1823) sim=0.154 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...sion of part of Artois, which had been acquired by Louis XI.`
  - `Moreover, from the first period of their assembling, the commons made every effort to augment their `
  - Fix: `(1823, 'FRANCE', 'eb_6th_1823_v09_FOR-DIR', [('MOREOVER', r'Moreover,\s+from\s+the', 8)])`

- 🔴 **CHRYSOPHYLLUM** → **THEORY** (1797) sim=0.155 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ociety of Jesuits suppressed by the Pope's bull August 25th.`
  - `Theory of chemistry defined, 21.

Thermometers: its use, 103. Wedgewood's improvement, 104.`
  - Fix: `(1797, 'CHRYSOPHYLLUM', 'eb_3rd_1797_v04_TRE-OMI', [('THEORY', r'Theory\s+of\s+chemistry', 5906)])`

- 🔴 **MORAI** → **FOR** (1815) sim=0.155 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...s not admitted, though they have also morais common to both.`
  - `For 34 Years. New moon. 1st Quart. Full Moon. 2d Quart.`
  - Fix: `(1815, 'MORAI', 'eb_5th_1815_v14_ENL-NIC', [('FOR', r'For\s+34\s+Years\.', 0)])`

- 🔴 **MARIA** → **FOR** (1810) sim=0.156 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... a tower and a cloister. W. Long. 5°. 33'. N. Lat. 36°. 35'.`
  - `For an account of the common goat, we refer our readers to Buffon and Mr Pennant's British Zoology, `
  - Fix: `(1810, 'MARIA', 'eb_4th_1810_v12_MAH-ADD', [('FOR', r'For\s+an\s+account', 28)])`

- 🔴 **MOMUS** → **CONSTITUENT PARTS.** (1823) sim=0.156 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... mask from his face, and holding a small figure in his hand.`
  - `**Constituent Parts.** Chenevix.

|                |           |
|----------------|-----------|
| Ox`
  - Fix: `(1823, 'MOMUS', 'eb_6th_1823_v14_ENL-NIC', [('CONSTITUENT PARTS.', r'\*\*Constituent\s+Parts\.\*\*\s+Chenevix\.', 0)])`

- 🔴 **IDLENESS** → **MEDITERRANEAN** (1815) sim=0.157 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...he public for any offence that such his inmate might commit.`
  - `Mediterranean flying fish.—The ventral fins reaching to the tail. The general length of this species`
  - Fix: `(1815, 'IDLENESS', 'eb_5th_1815_v11_ENL-LIE', [('MEDITERRANEAN', r'Mediterranean\s+flying\s+fish\.—The', 7)])`

- 🔴 **POLIANTHES** → **SHE** (1797) sim=0.158 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...a most excellent flavour, scarce inferior to oil of jasmine.`
  - `She all night long her amorous descent sung.

It is indeed a better example of the proper use of the`
  - Fix: `(1797, 'POLIANTHES', 'eb_3rd_1797_v15_IND-RAN', [('SHE', r'She\s+all\s+night', 15)])`

- 🔴 **MAGNETISM** → **CLAUDIAN** (1810) sim=0.158 [new_headword] (2 eds: 1810, 1815)
  - `...entia torpens
Membra fame, venalque sitis consumit apertas."`
  - `CLAUDIAN.

In the 16th century, the philosophers of modern times first began to speculate about the `
  - Fix: `(1810, 'MAGNETISM', 'eb_4th_1810_v17_LIE-MAH', [('CLAUDIAN', r'CLAUDIAN\.\s+In\s+the', 30)])`

- 🔴 **AGRICULTURE** → **NOR** (1815) sim=0.159 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hree feet two inches higher than the bottom of the head A.B.`
  - `Nor must we here omit to mention, that the justly celebrated Linnæus and his disciples have performe`
  - Fix: `(1815, 'AGRICULTURE', 'eb_5th_1815_v01_ENL-AME', [('NOR', r'Nor\s+must\s+we', 0)])`

- 🔴 **CHARLOCK** → **QUEEN** (1823) sim=0.159 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ever growing where there is a coat of grass upon the ground.`
  - `Queen CHARLOTTE's ISLAND, an island in the South sea, first discovered by Captain Wallis in the Dolp`
  - Fix: `(1823, 'CHARLOCK', 'eb_6th_1823_v05_ENL-CHI', [('QUEEN', r'Queen\s+CHARLOTTE's\s+ISLAND,', 0)])`

- 🔴 **POPE** → **ALEXANDER** (1810) sim=0.160 [topic_change] [gap: OCR_GAP]
  - `...e subjection and control of Bonaparte. See FRANCE and ITALY.`
  - `Alexander, a celebrated English poet, descended from a respectable family, was born the 8th of June `
  - Fix: `(1810, 'POPE', 'eb_4th_1810_v16_POE-BC', [('ALEXANDER', r'Alexander,\s+a\s+celebrated', 72)])`

- 🔴 **GARIZIM** → **CARLAND** (1815) sim=0.160 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...imon, the fourth in succession of the Asmonaeans (Josephus).`
  - `CARLAND, a fort of chaplet made of flowers, feathers, and sometimes precious stones, worn on the hea`
  - Fix: `(1815, 'GARIZIM', 'eb_5th_1815_v09_FOR-CCX', [('CARLAND', r'CARLAND,\s+a\s+fort', 0)])`

- 🔴 **ELEUSINIA** → **THESE** (1797) sim=0.161 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...0 years, and were at last abolished by Theodosius the Great.`
  - `These bottles, thus broken in large discharges, seem always to break, or to be struck through, nearl`
  - Fix: `(1797, 'ELEUSINIA', 'eb_3rd_1797_v06_IND-ETH', [('THESE', r'These\s+bottles,\s+thus', 31)])`

- 🔴 **GEOMETRY** → **PHIL** (1797) sim=0.161 [topic_change] [gap: EDITORIAL]
  - `...be, the illuminated hemisphere, the sun being in the zenith.`
  - `Phil. Trans. No. 456. p. 321, or Martyn's Abr. Vol. VIII. p. 352.`
  - Fix: `(1797, 'GEOMETRY', 'eb_3rd_1797_v07_TRE-GOA', [('PHIL', r'Phil\.\s+Trans\.\s+No\.', 111)])`

- 🔴 **OU-POEY-TSE** → **GREATER OUSE** (1797) sim=0.161 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... fluxions of the eyes and ears, and in many other disorders.`
  - `GREATER OUSE, a river which rises near Fitwell in Oxfordshire, and proceeds to Buckingham, Stony-Str`
  - Fix: `(1797, 'OU-POEY-TSE', 'eb_3rd_1797_v13_TRE-PAS', [('GREATER OUSE', r'GREATER\s+OUSE,\s+a', 11)])`

- 🔴 **MENES** → **GENUS CXX** (1815) sim=0.161 [new_headword] [gap: EDITORIAL]
  - `... furlongs broad, and caused it to run through the mountains.`
  - `GENUS CXX. ENURESIS.

An involuntary FLUX of URINE.`
  - Fix: `(1815, 'MENES', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CXX', r'GENUS\s+CXX\.\s+ENURESIS\.', 0)])`

- 🔴 **FRANCE** → **ALL** (1823) sim=0.161 [topic_change] [gap: OCR_GAP]
  - `... and to substitute manly and martial exercises in its place.`
  - `All these victories, however, as well as many others said to have been gained by the Romans, were no`
  - Fix: `(1823, 'FRANCE', 'eb_6th_1823_v09_FOR-DIR', [('ALL', r'All\s+these\s+victories,', 0)])`

- 🔴 **BRAHMINS** → **MARTIUS** (1842) sim=0.161 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...festivals the Brahmins exhibit the same species of idolatry,`
  - `Martius and Lindley attribute to this a fleshy albumen; St Hilaire and Kunth say there is none. Some`
  - Fix: `(1842, 'BRAHMINS', 'eb_7th_1842_v05_BOR-CAL', [('MARTIUS', r'Martius\s+and\s+Lindley', 14)])`

- 🔴 **KINGDOM OF LEON AND CASTILLE UNITED** → **STATISTICS** (1842) sim=0.161 [new_headword] [gap: VARIANT]
  - `... to raise her above her former rank in the scale of nations.`
  - `STATISTICS.

The position and boundaries of Spain have already been described. Its extent north and `
  - Fix: `(1842, 'KINGDOM OF LEON AND CASTILLE UNITED', 'eb_7th_1842_v20_SEV-SUG', [('STATISTICS', r'STATISTICS\.\s+The\s+position', 92)])`

- 🔴 **ENTOMOLOGY** → **THOMSON** (1815) sim=0.162 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...
G. Mouth furnished with jaws. Hind-legs formed for leaping.`
  - `Thomson's Winter.

For we shall ever find some peculiar beauty to admire, even in the slightest prod`
  - Fix: `(1815, 'ENTOMOLOGY', 'eb_5th_1815_v08_ENL-FOR', [('THOMSON', r'Thomson's\s+Winter\.\s+For', 8)])`

- 🔴 **FRANCE** → **PHILIP** (1823) sim=0.162 [topic_change] [gap: VARIANT]
  - `...yment of the bills drawn upon them by their army in America.`
  - `Philip now endeavoured to secure himself against the power of his rival by alliances, and by purchas`
  - Fix: `(1823, 'FRANCE', 'eb_6th_1823_v09_FOR-DIR', [('PHILIP', r'Philip\s+now\s+endeavoured', 3)])`

- 🔴 **ECATESIA** → **SECOND LAW** (1810) sim=0.163 [person_bio] [gap: OCR_GAP]
  - `..., because of the great number of hecatombs sacrificed in it.`
  - `Second Law of Motion.

Every change of motion is proportional to the force impressed, and it is made`
  - Fix: `(1810, 'ECATESIA', 'eb_4th_1810_v07_STE-ELE', [('SECOND LAW', r'Second\s+Law\s+of', 20)])`

- 🔴 **EBIONITES** → **EAVES-D** (1823) sim=0.163 [new_headword] (2 eds: 1815, 1823)
  - `...of the Nazarenes, and even in those used by the Cerinthians.`
  - `EAVES-Droppers, are such persons as stand under the eaves, or walls, and windows of a house, by nigh`
  - Fix: `(1823, 'EBIONITES', 'eb_6th_1823_v07_ENL-ELE', [('EAVES-D', r'EAVES\-Droppers,\s+are\s+such', 57)])`

- 🔴 **ORICHALCUM** → **FOR** (1842) sim=0.163 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ome remains and fragments of his Hexapla. He ought not to be`
  - `For as the tint produced at P by AB at an inclination of 90°, which is its maximum tint, is equal to`
  - Fix: `(1842, 'ORICHALCUM', 'eb_7th_1842_v16_SEV-PAN', [('FOR', r'For\s+as\s+the', 4)])`

- 🔴 **REMPHAN** → **ACTION OF REMOVING** (1823) sim=0.164 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...ds his countrymen with having borne the Star of their deity.`
  - `ACTION OF REMOVING, in Scots Law. See LAW, No. clxvii. 18.`
  - Fix: `(1823, 'REMPHAN', 'eb_6th_1823_v17_ENL-RHI', [('ACTION OF REMOVING', r'ACTION\s+OF\s+REMOVING,', 76)])`

- 🔴 **CHRYSA** → **PORTABLE** (1797) sim=0.165 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ed into Finland.

1156 The city of Moscow in Russia founded.`
  - `Portable furnaces, 600, et seq. See Furnaces.

Poveshell's vessels of use in chemistry, 503, 504. Re`
  - Fix: `(1797, 'CHRYSA', 'eb_3rd_1797_v04_TRE-OMI', [('PORTABLE', r'Portable\s+furnaces,\s+600,', 5061)])`

- 🔴 **BRAVO** → **ALL** (1823) sim=0.165 [topic_change] [gap: OCR_GAP]
  - `...miles distant from Magadoxo. E. Long. 41° 35' N. Lat. 1° 0'.`
  - `All the sorts of aloe dissolve in pure spirit, proof spirit, and proof spirit diluted with half its `
  - Fix: `(1823, 'BRAVO', 'eb_6th_1823_v04_ENL-BUR', [('ALL', r'All\s+the\s+sorts', 28)])`

- 🔴 **LAURA** → **POET-LAUREATE** (1842) sim=0.165 [new_headword] (2 eds: 1797, 1842)
  - `... Kedron; and the laura of the Towers, near the river Jordan.`
  - `POET-LAUREATE, an officer of the household of the kings of Britain, whose business consists only in `
  - Fix: `(1842, 'LAURA', 'eb_7th_1842_v13_SEV-AB', [('POET-LAUREATE', r'POET\-LAUREATE,\s+an\s+officer', 11)])`

- 🔴 **LIGATURES** → **VAN HELMONT** (1842) sim=0.165 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...eration that it can be produced at a very trifling expense."`
  - `Van Helmont appears to have discovered another powerful phosphorus; and Baldwin of Misnia, in 1677, `
  - Fix: `(1842, 'LIGATURES', 'eb_7th_1842_v13_SEV-AB', [('VAN HELMONT', r'Van\s+Helmont\s+appears', 36)])`

- 🔴 **FERULA** → **PARTIAL** (1810) sim=0.166 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...x. The drug asafoetida is obtained from a species of ferula.`
  - `Partial blindness is a symptom of several diseases in the horse: it usually attends great weakness, `
  - Fix: `(1810, 'FERULA', 'eb_4th_1810_v08_FAI-FOR', [('PARTIAL', r'Partial\s+blindness\s+is', 28)])`

- 🔴 **POINT** → **MEL** (1815) sim=0.166 [topic_change] [gap: VARIANT]
  - `...or expected at the close of an epigram. See Poetry, No. 169.`
  - `Mel. But we must beg our bread in climes unknown,
Beneath the scorching or the freezing zone;
And so`
  - Fix: `(1815, 'POINT', 'eb_5th_1815_v17_ENL-RHI', [('MEL', r'Mel\.\s+But\s+we', 58)])`

- 🔴 **NUNCUPATIVE** → **MONTE NUOVO** (1797) sim=0.167 [person_bio] [gap: EDITORIAL]
  - `...taphysics, Part III. Ch. iv. Of the Immortality of the Soul.`
  - `Monte Nuovo, in the environs of Naples, blocks up the valley of Averno. "This mountain (Mr Swinburne`
  - Fix: `(1797, 'NUNCUPATIVE', 'eb_3rd_1797_v13_TRE-PAS', [('MONTE NUOVO', r'Monte\s+Nuovo,\s+in', 50)])`

- 🔴 **DYNAMICS** → **ALL** (1810) sim=0.167 [topic_change] [gap: OCR_GAP]
  - `... account of the predominance of the yellow colouring matter.`
  - `All the perpendiculars, such as PR, on one side of the plane CDFE, being equal to all those on the o`
  - Fix: `(1810, 'DYNAMICS', 'eb_4th_1810_v07_STE-ELE', [('ALL', r'All\s+the\s+perpendiculars,', 43)])`

- 🔴 **MAMMON** → **SEVERAL** (1810) sim=0.168 [topic_change] [gap: OCR_GAP]
  - `... grow in hell; that foil may best
Deserve the precious bane.`
  - `Several of these animals have been brought into Europe. Buffon gives an account of one, and Dr Parso`
  - Fix: `(1810, 'MAMMON', 'eb_4th_1810_v12_MAH-ADD', [('SEVERAL', r'Several\s+of\s+these', 28)])`

- 🔴 **ANNULOSA** → **PORTUNUS** (1823) sim=0.168 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hose about the base of the bud, losing their primary figure,`
  - `Portunus marmoreus.

Leach, Edin. Encycl. vii. 390.`
  - Fix: `(1823, 'ANNULOSA', 'eb_6th_1823_v02_ENL-ASS', [('PORTUNUS', r'Portunus\s+marmoreus\.\s+Leach,', 182)])`

- 🔴 **FRANCE** → **FOR** (1823) sim=0.168 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...itory of Guienne, and annexed it to the dominions of France.`
  - `For 40 years the principles of liberty had been disseminated with eagerness in France by some men of`
  - Fix: `(1823, 'FRANCE', 'eb_6th_1823_v09_FOR-DIR', [('FOR', r'For\s+40\s+years', 7)])`

- 🔴 **PHILOSOPHY** → **INSTEAD** (1823) sim=0.168 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... τῷ τοιαύτῳ ἀγαθῷ πληροῖται ἡ βασιλεία τῆς ἡσυχίας τοιαύτης.`
  - `Instead of giving a literal and bald translation of this advertisement, which runs exactly in the st`
  - Fix: `(1823, 'PHILOSOPHY', 'eb_6th_1823_v16_ENL-BRE', [('INSTEAD', r'Instead\s+of\s+giving', 211)])`

- 🔴 **SETTING** → **ACT OF SETTLEMENT** (1823) sim=0.168 [new_headword] (2 eds: 1815, 1823)
  - `...of a dog particularly trained to that purpose. See Shooting.`
  - `ACT OF SETTLEMENT, in British history, a name given to the statute 12 and 13 Will. III. cap. 2, wher`
  - Fix: `(1823, 'SETTING', 'eb_6th_1823_v19_ENL-SUG', [('ACT OF SETTLEMENT', r'ACT\s+OF\s+SETTLEMENT,', 45)])`

- 🔴 **AVENA** → **HELIOCENTRIC** (1797) sim=0.169 [topic_change] [gap: EDITORIAL]
  - `...seases, coughs, hoarseness, and exulcerations of the fauces.`
  - `Heliocentric circles of the planets, the same with their orbits round the sun, 311. Heliocentric lat`
  - Fix: `(1797, 'AVENA', 'eb_3rd_1797_v02_IND-BAR', [('HELIOCENTRIC', r'Heliocentric\s+circles\s+of', 15)])`

- 🔴 **MOISTURE** → **ALKALINE** (1810) sim=0.169 [topic_change] [gap: OCR_GAP]
  - `... to the drink, being given, the weight of a human body is le`
  - `Alkaline cinnabar of De Born is found at the same place; is of a bright red colour, foliated fractur`
  - Fix: `(1810, 'MOISTURE', 'eb_4th_1810_v17_MIC-MOR', [('ALKALINE', r'Alkaline\s+cinnabar\s+of', 0)])`

- 🔴 **REMPHAN** → **ACTION OF REMOVING** (1815) sim=0.169 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...ds his countrymen with having borne the Star of their deity.`
  - `ACTION OF REMOVING, in Scots Law, See LAW, No. clxvii. 18.`
  - Fix: `(1815, 'REMPHAN', 'eb_5th_1815_v17_ENL-RHI', [('ACTION OF REMOVING', r'ACTION\s+OF\s+REMOVING,', 75)])`

- 🔴 **SANCTION** → **EDWARD** (1842) sim=0.169 [topic_change] [gap: VARIANT]
  - `...nd Fisher bishop of Rochester acted with the same integrity.`
  - `Edward was succeeded in 978 by Ethelred, the unconscious cause of his untimely fate. When the latter`
  - Fix: `(1842, 'SANCTION', 'eb_7th_1842_v19_SEV-SCU', [('EDWARD', r'Edward\s+was\s+succeeded', 0)])`

- 🔴 **CRANGANORE** → **ARCHBISHOP NICOLSON** (1842) sim=0.170 [person_bio] [gap: VARIANT]
  - `...nd retook the place in 1791. Long. 76. 5. E. Lat. 10. 15. N.`
  - `Archbishop Nicolson mentions an historical production of the same learned author. "I have likewise s`
  - Fix: `(1842, 'CRANGANORE', 'eb_7th_1842_v07_SEV-DIA', [('ARCHBISHOP NICOLSON', r'Archbishop\s+Nicolson\s+mentions', 0)])`

- 🔴 **WM FARQUHARSON** → **RUBEOLA** (1810) sim=0.171 [topic_change] [gap: OCR_GAP]
  - `...be now read; and it was read accordingly, and is as follows:`
  - `Rubeola variolodes, Sauv. sp. 3.

Description. This disease begins with a cold stage, which is soon `
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('RUBEOLA', r'Rubeola\s+variolodes,\s+Sauv\.', 0)])`

- 🔴 **MEDICINE** → **GENUS XIV** (1815) sim=0.171 [new_headword] [gap: EDITORIAL]
  - `... beam CT, in order to prevent the beam from rising too high.`
  - `GENUS XIV. PERITONITIS.

Inflammation of the PERITONÆUM.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XIV', r'GENUS\s+XIV\.\s+PERITONITIS\.', 36)])`

- 🔴 **CARTHAGE** → **NEW** (1815) sim=0.172 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...s contraria, fluitibus undas,
Arma armi. VIRG. Aen. iv. 627.`
  - `New CARTHAGE, a considerable town of Mexico, in the province of Costa Rica. It is a very rich tradin`
  - Fix: `(1815, 'CARTHAGE', 'eb_5th_1815_v05_ENL-CHI', [('NEW', r'New\s+CARTHAGE,\s+a', 93)])`

- 🔴 **CALEDONIA** → **NEW CALEDONIA** (1778) sim=0.173 [new_headword] (2 eds: 1778, 1797)
  - `...em any provisions; so they were obliged to leave it in 1700.`
  - `NEW CALEDONIA, an island in the south-sea, lately discovered by captain Cook, and, next to New Holla`
  - Fix: `(1778, 'CALEDONIA', 'eb_2nd_1778_v03_BYW-CRI', [('NEW CALEDONIA', r'NEW\s+CALEDONIA,\s+an', 0)])`

- 🔴 **RETZIA** → **THESE** (1797) sim=0.173 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...d to as a matter of faith, wherein reason has nothing to do.`
  - `These good men are engaged in various amusements, according to the taste and genius of each. Orpheus`
  - Fix: `(1797, 'RETZIA', 'eb_3rd_1797_v16_TRE-SCO', [('THESE', r'These\s+good\s+men', 18)])`

- 🔴 **PENNANT** → **THESE** (1797) sim=0.173 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...m the good fortune of my life I will attempt the execution."`
  - `These valuable volumes are drawn up by Mr Pennant in the manner of his introduction to the Arctic Zo`
  - Fix: `(1797, 'PENNANT', 'eb_3rd_1797_v502_IND-ZEM', [('THESE', r'These\s+valuable\s+volumes', 60)])`

- 🔴 **COMPASS** → **SECOND PART HENRY** (1810) sim=0.173 [person_bio] [gap: OCR_GAP]
  - `...the variation of the needle itself. See Compass and Dialing.`
  - `Second Part Henry IV. Act i. sc. 6.
The strongest objection that can lie against a comparison is, th`
  - Fix: `(1810, 'COMPASS', 'eb_4th_1810_v17_OBS-GEN', [('SECOND PART HENRY', r'Second\s+Part\s+Henry', 62)])`

- 🔴 **SPARTA** → **CHARLES** (1810) sim=0.173 [topic_change] [gap: OCR_GAP]
  - `...his daughter upon Octavio Farnese, son of the duke of Parma.`
  - `Charles had soon farther cause to be sensible of his obligations to the holy father for bringing abo`
  - Fix: `(1810, 'SPARTA', 'eb_4th_1810_v19_SLE-SUG', [('CHARLES', r'Charles\s+had\s+soon', 242)])`

- 🔴 **JERICHO** → **GEN** (1815) sim=0.173 [topic_change] [gap: EDITORIAL]
  - `...e commerce of Raha, which is no more than a ruinous village.`
  - `Gen. 12. Chimæra.

Head sharp-pointed; spiracles solitary, in four divisions under the neck; mouth u`
  - Fix: `(1815, 'JERICHO', 'eb_5th_1815_v11_ENL-LIE', [('GEN', r'Gen\.\s+12\.\s+Chimæra\.', 0)])`

- 🔴 **CHROMATICS** → **PLINY** (1815) sim=0.174 [topic_change] [gap: EDITORIAL]
  - `...ents, we have any reason to believe he ever did make use of.`
  - `Pliny the younger, who was governor of Pontus and Bithynia between the years 103 and 105, gives a ve`
  - Fix: `(1815, 'CHROMATICS', 'eb_5th_1815_v06_ENL-CRY', [('PLINY', r'Pliny\s+the\s+younger,', 43)])`

- 🔴 **BLOOD** → **NOR** (1842) sim=0.174 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...hrown into prison, where he died on the 24th of August 1680.`
  - `Nor have the blind been less distinguished in the practice of the arts than in science and literatur`
  - Fix: `(1842, 'BLOOD', 'eb_7th_1842_v04_SEV-BOR', [('NOR', r'Nor\s+have\s+the', 45)])`

- 🔴 **PAINTING** → **SCHOOLS** (1810) sim=0.175 [topic_change] [gap: VARIANT]
  - `...y thing connected with science or the liberal arts.
History.`
  - `Schools. tal qualities and accessories of the art; and if he had superiors, it consisted in this, th`
  - Fix: `(1810, 'PAINTING', 'eb_4th_1810_v15_ORD-PAR', [('SCHOOLS', r'Schools\.\s+tal\s+qualities', 39)])`

- 🔴 **BARON** → **ROBERT** (1823) sim=0.175 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...t must be borne by the husband on an escutcheon of pretence.`
  - `ROBERT, a dramatic author, who lived during the reign of Charles I. and the protectorship of Oliver `
  - Fix: `(1823, 'BARON', 'eb_6th_1823_v03_ENL-BOO', [('ROBERT', r'ROBERT,\s+a\s+dramatic', 76)])`

- 🔴 **BURNING** → **SIG** (1815) sim=0.176 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...in riot those gathering sorrows which he knew not to subdue.`
  - `Sig. Mondini, Bianchini, and Maffei, have written treatises express to account for the cause of so e`
  - Fix: `(1815, 'BURNING', 'eb_5th_1815_v05_ENL-CHI', [('SIG', r'Sig\.\s+Mondini,\s+Bianchini,', 0)])`

- 🔴 **CONSECRATION** → **FOR** (1815) sim=0.176 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...fidelibus recepta: on the one side of the head, dea,
or Θεά.`
  - `For \( AC^2 : BC^2 :: CL \cdot Dh : CL \cdot DH :: Db : DH. \)`
  - Fix: `(1815, 'CONSECRATION', 'eb_5th_1815_v06_ENL-CRY', [('FOR', r'For\s+\\\(\s+AC\^2', 15)])`

- 🔴 **MELEAGER** → **GENUS LXXXIX** (1815) sim=0.176 [new_headword] [gap: EDITORIAL]
  - `...hem into the order they are in at present, in the year 1380.`
  - `GENUS LXXXIX. FRAMBOESIA.

The Yaws.`
  - Fix: `(1815, 'MELEAGER', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXXIX', r'GENUS\s+LXXXIX\.\s+FRAMBOESIA\.', 11)])`

- 🔴 **LEATHER** → **GAME** (1778) sim=0.177 [topic_change] [gap: VARIANT]
  - `...9. Impress. See IMPRESSING.

10. Insurance. See INSURANCE.`
  - `Game-Laws. See the article GAME.

Sir William Blackstone, treating of the alterations in our laws, a`
  - Fix: `(1778, 'LEATHER', 'eb_2nd_1778_v06_BYW-IND', [('GAME', r'Game\-Laws\.\s+See\s+the', 3219)])`

- 🔴 **EBIONITES** → **EAVES-D** (1815) sim=0.177 [new_headword] (2 eds: 1815, 1823)
  - `...ul's epistles, whom they treated with the utmost disrespect.`
  - `EAVES-Droppers, are such persons as stand under the eaves, on walls, and windows of a house, by nigh`
  - Fix: `(1815, 'EBIONITES', 'eb_5th_1815_v07_CUB-DIR', [('EAVES-D', r'EAVES\-Droppers,\s+are\s+such', 21)])`

- 🔴 **EULOGY** → **REPRODUCTIVE POWER OF REPTILES.** (1823) sim=0.178 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...um on any person, on account of some virtue or good quality.`
  - `**Reproductive Power of Reptiles.**—Many of the animals belonging to the order of reptiles undergo v`
  - Fix: `(1823, 'EULOGY', 'eb_6th_1823_v08_ENL-FOR', [('REPRODUCTIVE POWER OF REPTILES.', r'\*\*Reproductive\s+Power\s+of', 55)])`

- 🔴 **LAURA** → **POET LAUREATE** (1823) sim=0.178 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... Cedron; the laura of the Towers, near the river Jordan, &c.`
  - `POET LAUREATE, an officer of the household of the kings of Britain, whose business consists only in `
  - Fix: `(1823, 'LAURA', 'eb_6th_1823_v11_ENL-LIE', [('POET LAUREATE', r'POET\s+LAUREATE,\s+an', 11)])`

- 🔴 **ARLES** → **FOR** (1823) sim=0.179 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...d makes it not very wholesome. E. Long. 4° 48'. N. Lat. 43°.`
  - `For the same reason, if any circulating decimal, not a multiple of 3, be divided by 3, the quotient `
  - Fix: `(1823, 'ARLES', 'eb_6th_1823_v02_ENL-ASS', [('FOR', r'For\s+the\s+same', 35)])`

- 🔴 **END OF THE FIFTH VOLUME** → **PERCUSSION** (1810) sim=0.180 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...| |
|---------|---|
| CXLII. CXLIII. | 774 |
| CXLIV. | 79 |`
  - `Percussion.

The production of heat by striking together flint and steel, is a well known fact. The `
  - Fix: `(1810, 'END OF THE FIFTH VOLUME', 'eb_4th_1810_v05_CHA-CHI', [('PERCUSSION', r'Percussion\.\s+The\s+production', 28)])`

- 🔴 **HEMICYCLE** → **FOR** (1823) sim=0.180 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...un's declination, the day of the month, hour of the day, &c.`
  - `For preserving these animals, Mr Barbut advises that they be drowned in brandy or other spirits, tak`
  - Fix: `(1823, 'HEMICYCLE', 'eb_6th_1823_v10_ENL-HYD', [('FOR', r'For\s+preserving\s+these', 20)])`

- 🔴 **SCOTLAND** → **ENGLAND** (1842) sim=0.180 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...lloway, marched against Perth, then in possession of Edward.`
  - `England and France were now at peace, and Henry the Foreign Eighth and Francis the First united in a`
  - Fix: `(1842, 'SCOTLAND', 'eb_7th_1842_v19_SEV-SCU', [('ENGLAND', r'England\s+and\s+France', 20)])`

- 🔴 **STOURBRIDGE** → **INSTEAD** (1842) sim=0.180 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...3431, in 1811 to 4072; in 1821 to 5090, and in 1831 to 6148.`
  - `Instead of using the principle of recoil, the force of steam, issuing with violence as we see it fro`
  - Fix: `(1842, 'STOURBRIDGE', 'eb_7th_1842_v20_SEV-SUG', [('INSTEAD', r'Instead\s+of\s+using', 0)])`

- 🔴 **FRANCE** → **PARTITION TREATY** (1823) sim=0.181 [person_bio] [gap: PARSING_OR_EDITORIAL]
  - `...n of despotism, gave way to the noble enthusiasm of liberty.`
  - `Partition Treaty between the Courts in Concert, concluded and signed at Pavia, in the month of July `
  - Fix: `(1823, 'FRANCE', 'eb_6th_1823_v09_FOR-DIR', [('PARTITION TREATY', r'Partition\s+Treaty\s+between', 13)])`

- 🔴 **PRESCRIPTIONS** → **SIGNETUR** (1823) sim=0.181 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...3.
Socci Spissati Hysocyami, gr. 4.
Syrupi q. s. Fiat bolus.`
  - `Signetur. To be taken just as the next hot fit is coming on.`
  - Fix: `(1823, 'PRESCRIPTIONS', 'eb_6th_1823_v17_ENL-RHI', [('SIGNETUR', r'Signetur\.\s+To\s+be', 84)])`

- 🔴 **GREGORY** → **STATISTICAL VIEW OF INDEPENDENT GREECE** (1842) sim=0.181 [new_headword] [gap: VARIANT]
  - `...ore them up under the most dreadful privations and reverses.`
  - `STATISTICAL VIEW OF INDEPENDENT GREECE.

It was settled by the conference of London in March 1829, t`
  - Fix: `(1842, 'GREGORY', 'eb_7th_1842_v10_SEV-GRO', [('STATISTICAL VIEW OF INDEPENDENT GREECE', r'STATISTICAL\s+VIEW\s+OF', 215)])`

- 🔴 **SHRUB** → **PHILOSOPHICAL TRANSACTIONS** (1797) sim=0.182 [person_bio] [gap: EDITORIAL]
  - `...ion, they must be taken off down to the level of the ground.`
  - `Philosophical Transactions, no 165.`
  - Fix: `(1797, 'SHRUB', 'eb_3rd_1797_v17_TRE-STR', [('PHILOSOPHICAL TRANSACTIONS', r'Philosophical\s+Transactions,\s+no', 28)])`

- 🔴 **COACH** → **THESE** (1797) sim=0.182 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ems inclined to give the credit of the invention to Hungary.`
  - `These properties show that this is a volatile oil, and consequently it is probable that camphor is c`
  - Fix: `(1797, 'COACH', 'eb_3rd_1797_v501_ABE-IMP', [('THESE', r'These\s+properties\s+show', 14)])`

- 🔴 **BRITAIN** → **EARLY** (1823) sim=0.182 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... conquered the rich Dutch settlement of Java in August 1811.`
  - `Early in 1812 the Prince regent who had now become reconciled to the Tory ministers, invited his old`
  - Fix: `(1823, 'BRITAIN', 'eb_6th_1823_v502_AUS-CEL', [('EARLY', r'Early\s+in\s+1812', 38)])`

- 🔴 **TROJA** → **FOR** (1823) sim=0.182 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...avels, vol. 3d, 8vo. and Edinburgh Review, vol. 6th, p. 257.`
  - `For let the great circle of which A is the pole, meet the three sides in D, E, F; then F is the pole`
  - Fix: `(1823, 'TROJA', 'eb_6th_1823_v20_ENL-ZYG', [('FOR', r'For\s+let\s+the', 10)])`

- 🔴 **GEORGE** → **SCHOLIUM.** (1842) sim=0.182 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...om rocks, so that it is not safe for any vessel to enter it.`
  - `**Scholium.** It is manifest that the homologous sides are opposite to the equal angles.`
  - Fix: `(1842, 'GEORGE', 'eb_7th_1842_v10_SEV-GRO', [('SCHOLIUM.', r'\*\*Scholium\.\*\*\s+It\s+is', 22)])`

- 🔴 **GOLDEN** → **OROBIO** (1810) sim=0.183 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...rpheus is said to have had the muse Calliope for his mother.`
  - `OROBIO, Don Balthasar, a celebrated Jew of Spain. He was carefully educated in Judaism by his parent`
  - Fix: `(1810, 'GOLDEN', 'eb_4th_1810_v15_ORD-PAR', [('OROBIO', r'OROBIO,\s+Don\s+Balthasar,', 10)])`

- 🔴 **ELECTRICITY** → **HIEROGLYPHICKS** (1823) sim=0.183 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...age, and was taken by the Romans during the first Punic war.`
  - `HIEROGLYPHICKS.

A. DEITIES`
  - Fix: `(1823, 'ELECTRICITY', 'eb_6th_1823_v504_FOU-HOL', [('HIEROGLYPHICKS', r'HIEROGLYPHICKS\.\s+A\.\s+DEITIES', 90)])`

- 🔴 **LAURA** → **POET LAUREATE** (1810) sim=0.184 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... Cedron; the laura of the Towers, near the river Jordan, &c.`
  - `POET LAUREATE, an officer of the household of the kings of Britain, whose business consists only in `
  - Fix: `(1810, 'LAURA', 'eb_4th_1810_v11_JUN-LIE', [('POET LAUREATE', r'POET\s+LAUREATE,\s+an', 11)])`

- 🔴 **PALMUS** → **ALL** (1810) sim=0.184 [topic_change] [gap: OCR_GAP]
  - `...eadth of it. The Greek palmus was called doran. See MEASURE.`
  - `All the colours being ground, they are placed in a small heap on a piece of glass, which is covered `
  - Fix: `(1810, 'PALMUS', 'eb_4th_1810_v15_ORD-PAR', [('ALL', r'All\s+the\s+colours', 0)])`

- 🔴 **CINCTURE** → **GREEK** (1810) sim=0.184 [topic_change] [gap: EDITORIAL]
  - `...e true, is from the laurus caffia. See Laurus, Botany Index.`
  - `Greek historian, wrote a history of the eastern empire, during the reigns of John and Manuel Comnene`
  - Fix: `(1810, 'CINCTURE', 'eb_4th_1810_v17_OBS-GEN', [('GREEK', r'Greek\s+historian,\s+wrote', 0)])`

- 🔴 **CIOTAT** → **ORDER** (1797) sim=0.186 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...r family, and yourself. Live with honour, or die with glory.`
  - `Order of the cards before the 1st shuffle.

Ace spade i a d u y i
Ten diamonds a l e u l
Eight heart`
  - Fix: `(1797, 'CIOTAT', 'eb_3rd_1797_v05_TRE-DIA', [('ORDER', r'Order\s+of\s+the', 54)])`

- 🔴 **LEE** → **NATHANIEL** (1810) sim=0.186 [new_headword] (3 eds: 1810, 1823, 1842)
  - `...Lee Stone. See Lee-Penny.

Lee Way. See Navigation.`
  - `NATHANIEL, a very eminent dramatic poet of the last century, was the son of a clergyman, who gave hi`
  - Fix: `(1810, 'LEE', 'eb_4th_1810_v11_JUN-LIE', [('NATHANIEL', r'NATHANIEL,\s+a\s+very', 18)])`

- 🔴 **MATERIA MEDICA** → **FOR** (1810) sim=0.186 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...rva Roseae rubrae, L. Conserva Roseae, D. Conserve of roses.`
  - `For obtaining the juices of vegetables or fruits, or the oils of seeds, &c., recourse is had to expr`
  - Fix: `(1810, 'MATERIA MEDICA', 'eb_4th_1810_v12_MAH-ADD', [('FOR', r'For\s+obtaining\s+the', 36)])`

- 🔴 **NIAGARA** → **COR** (1810) sim=0.188 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...heir expedition, will sometimes return without going to war.`
  - `Cor. 1. Hence the weights of bodies do not depend upon their forms and textures. For if the weights `
  - Fix: `(1810, 'NIAGARA', 'eb_4th_1810_v14_MOR-NIA', [('COR', r'Cor\.\s+1\.\s+Hence', 23)])`

- 🔴 **HAMINTON** → **VIVE** (1815) sim=0.188 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... following laudative verses were written upon this occasion:`
  - `Vive diu, felix arbor, semperque vireto
Frondibus, ut nobis talia poma feras.`
  - Fix: `(1815, 'HAMINTON', 'eb_5th_1815_v10_GOT-HYD', [('VIVE', r'Vive\s+diu,\s+felix', 55)])`

- 🔴 **PRESCRIPTION** → **SIGNETUR** (1815) sim=0.188 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...3.
Succi Spissati Hydacyami, gr. 4.
Syrupi q. s. Fiat bolus.`
  - `Signetur. To be taken just as the next hot fit is coming on.`
  - Fix: `(1815, 'PRESCRIPTION', 'eb_5th_1815_v17_ENL-RHI', [('SIGNETUR', r'Signetur\.\s+To\s+be', 58)])`

- 🔴 **DOLLOND** → **ALL** (1823) sim=0.188 [topic_change] [gap: OCR_GAP]
  - `...ix of all the Papers referred to, 3d edit. 4to. Lond. 1808.)`
  - `All the officers, clerks, artificers, and labourers of the civil establishments of the navy, are ent`
  - Fix: `(1823, 'DOLLOND', 'eb_6th_1823_v501_EIG-DUR', [('ALL', r'All\s+the\s+officers,', 63)])`

- 🔴 **PUBLIUS SYRUS** → **OAK PUCERON** (1810) sim=0.189 [new_headword] (2 eds: 1810, 1823)
  - `...ences, written in iambics, and placed in alphabetical order.`
  - `OAK PUCERON, a name given by naturalists to a very remarkable species of animal of the puceron kind.`
  - Fix: `(1810, 'PUBLIUS SYRUS', 'eb_4th_1810_v17_PRO-RHI', [('OAK PUCERON', r'OAK\s+PUCERON,\s+a', 0)])`

- 🔴 **PUBLIUS SYRUS** → **OAK PUCERON** (1823) sim=0.189 [new_headword] (2 eds: 1810, 1823)
  - `...ences, written in iambics, and placed in alphabetical order.`
  - `OAK PUCERON, a name given by naturalists to a very remarkable species of animal of the puceron kind.`
  - Fix: `(1823, 'PUBLIUS SYRUS', 'eb_6th_1823_v17_ENL-RHI', [('OAK PUCERON', r'OAK\s+PUCERON,\s+a', 0)])`

- 🔴 **HOLLAND** → **DISPUTES** (1842) sim=0.189 [topic_change] [gap: VARIANT]
  - `...ding system, which has since been followed by other nations.`
  - `Disputes had arisen between the states of Holland on one hand, and the Prince of Orange and the smal`
  - Fix: `(1842, 'HOLLAND', 'eb_7th_1842_v11_GRO-HYD', [('DISPUTES', r'Disputes\s+had\s+arisen', 48)])`

- 🔴 **MATERIA MEDICA** → **ORDER** (1810) sim=0.190 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ling water; but the evaporation is in this way very tedious.`
  - `Order 6. POLYGYNIA.

165. HELLEBORUS NIGER, E. L. D. MELAMPODIUM. Black hellebore. See Botany, p. 21`
  - Fix: `(1810, 'MATERIA MEDICA', 'eb_4th_1810_v12_MAH-ADD', [('ORDER', r'Order\s+6\.\s+POLYGYNIA\.', 40)])`

- 🔴 **LEE** → **NATHANIEL** (1823) sim=0.190 [new_headword] (3 eds: 1810, 1823, 1842)
  - `...Lee-Stone. See Lee-Penny.

Lee-Way. See Navigation.`
  - `NATHANIEL, a very eminent dramatic poet of the last century, was the son of a clergyman, who gave hi`
  - Fix: `(1823, 'LEE', 'eb_6th_1823_v11_ENL-LIE', [('NATHANIEL', r'NATHANIEL,\s+a\s+very', 40)])`

- 🔴 **HISTORY** → **COMPOSITION OF HISTORY** (1842) sim=0.190 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ersities of Louvain, Douay, Sorbonne, Alcala, and Salamanca.`
  - `COMPOSITION OF HISTORY.

History has been defined, philosophy teaching by examples. But this definit`
  - Fix: `(1842, 'HISTORY', 'eb_7th_1842_v12_DEF-PLA', [('COMPOSITION OF HISTORY', r'COMPOSITION\s+OF\s+HISTORY\.', 38)])`

- 🔴 **POPERY** → **PHILIPS** (1842) sim=0.190 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... or else tha zurn
Will quite bego before ch' avs half n dom!`
  - `Philips is said to have resented this treatment by threats of personal chastisement to Pope, and eve`
  - Fix: `(1842, 'POPERY', 'eb_7th_1842_v18_PLA-QUO', [('PHILIPS', r'Philips\s+is\s+said', 337)])`

- 🔴 **LULA** → **ULLI** (1778) sim=0.191 [new_headword] (4 eds: 1778, 1797, 1810, 1823)
  - `...ast, by Pithia Lapmark on the south, and Norway on the west.`
  - `ULLI (John Baptist), the most celebrated and most excellent musician that has appeared in France sin`
  - Fix: `(1778, 'LULA', 'eb_2nd_1778_v06_BYW-IND', [('ULLI', r'ULLI\s+\(John\s+Baptist\),', 15)])`

- 🔴 **LULA** → **ULLI** (1797) sim=0.191 [new_headword] (4 eds: 1778, 1797, 1810, 1823)
  - `...ast, by Pithia Lapmark on the south, and Norway on the west.`
  - `ULLI (John Baptist), the most celebrated and most excellent musician that has appeared in France sin`
  - Fix: `(1797, 'LULA', 'eb_3rd_1797_v10_IND-MEC', [('ULLI', r'ULLI\s+\(John\s+Baptist\),', 20)])`

- 🔴 **HOLLAND** → **NEW HOLLAND** (1797) sim=0.192 [new_headword] (2 eds: 1797, 1810)
  - `...rns brackish; and if they are shallow, they soon become dry.`
  - `NEW HOLLAND, the largest island in the world, reaching from 10 to 44 deg. S. Lat. and between 110 an`
  - Fix: `(1797, 'HOLLAND', 'eb_3rd_1797_v08_IND-HYD', [('NEW HOLLAND', r'NEW\s+HOLLAND,\s+the', 0)])`

- 🔴 **ROOD** → **ALL** (1810) sim=0.192 [topic_change] [gap: OCR_GAP]
  - `...cting in any other direction which may engage our attention.`
  - `All that we propose to deliver on this subject at present may be included in the following propositi`
  - Fix: `(1810, 'ROOD', 'eb_4th_1810_v17_RHI-RUS', [('ALL', r'All\s+that\s+we', 39)])`

- 🔴 **DREAMS** → **CAN** (1842) sim=0.192 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...ing ceased to be relished, are no longer produced.
DREDGING.`
  - `Can this cockpit hold
The vasty fields of France? Or may we cram
Within this wooden O, the very casq`
  - Fix: `(1842, 'DREAMS', 'eb_7th_1842_v08_DIA-VII', [('CAN', r'Can\s+this\s+cockpit', 62)])`

- 🔴 **COMBUSTIO PECUNIARIA** → **CHARMING** (1797) sim=0.193 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... deficient in value; because mixed with copper or brass, &c.`
  - `Charming flowers, adorn the bosom of my shepherdess.

It seems quite unnecessary to give any further`
  - Fix: `(1797, 'COMBUSTIO PECUNIARIA', 'eb_3rd_1797_v05_TRE-DIA', [('CHARMING', r'Charming\s+flowers,\s+adorn', 0)])`

- 🔴 **FIN** → **ALL** (1810) sim=0.193 [topic_change] [gap: OCR_GAP]
  - `....
get by heart the three
curves. The smith Newton's Express.`
  - `All these animals may occasionally swallow poison, 
and the treatment in these cases must depend in `
  - Fix: `(1810, 'FIN', 'eb_4th_1810_v08_FAI-FOR', [('ALL', r'All\s+these\s+animals', 475)])`

- 🔴 **AGRICULTURE** → **COLUMELLA** (1815) sim=0.193 [topic_change] [gap: EDITORIAL]
  - `...ll be pointed out on the semicircle by the straight edge CF.`
  - `Columella, who flourished in the reign of the emperor Claudius, wrote 12 books on husbandry, replete`
  - Fix: `(1815, 'AGRICULTURE', 'eb_5th_1815_v01_ENL-AME', [('COLUMELLA', r'Columella,\s+who\s+flourished', 0)])`

- 🔴 **NEWTONIAN PHILOSOPHY** → **FOR** (1815) sim=0.193 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...11} \quad \frac{11}{12} \quad \frac{12}{13}.
\end{align*}
\]`
  - `For this and his other lemmas Sir Isaac makes the following apology: "These lemmas are premised, to `
  - Fix: `(1815, 'NEWTONIAN PHILOSOPHY', 'eb_5th_1815_v14_ENL-NIC', [('FOR', r'For\s+this\s+and', 38)])`

- 🔴 **ROOF** → **ALL** (1823) sim=0.193 [topic_change] [gap: OCR_GAP]
  - `...cting in any other direction which may engage our attention.`
  - `All that we propose to deliver on this subject at present may be included in the following propositi`
  - Fix: `(1823, 'ROOF', 'eb_6th_1823_v18_ENL-SCR', [('ALL', r'All\s+that\s+we', 39)])`

- 🔴 **POLTROON** → **FOR** (1842) sim=0.194 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...o Mornicus, calls him an elegant, acute, and learned writer.`
  - `For a farther demonstration of the same principle, see Mill's Commerce Defended, p. 80.`
  - Fix: `(1842, 'POLTROON', 'eb_7th_1842_v18_PLA-QUO', [('FOR', r'For\s+a\s+farther', 28)])`

- 🔴 **JELLY** → **GEN** (1815) sim=0.195 [topic_change] [gap: EDITORIAL]
  - `...her of the above-mentioned broths, or any other warm liquor.`
  - `Gen. 2. Tetradon.

Tetradon.`
  - Fix: `(1815, 'JELLY', 'eb_5th_1815_v11_ENL-LIE', [('GEN', r'Gen\.\s+2\.\s+Tetradon\.', 6)])`

- 🔴 **BREAM** → **ORDER** (1823) sim=0.195 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...the tide has ebbed from her, or by docking, or by careening.`
  - `Order I. Monogynia.

798. Laurus. Cal. o. Cor. 6-petala, calycina. Bacc 1-sperma. Nectarii glandulae`
  - Fix: `(1823, 'BREAM', 'eb_6th_1823_v04_ENL-BUR', [('ORDER', r'Order\s+I\.\s+Monogynia\.', 6)])`

- 🔴 **DYNAMICS** → **INSTEAD** (1842) sim=0.195 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...tant ratio of the increments of the ordinates and abscissas.`
  - `Instead of woad, and the other ingredients just mentioned, indigo is deprived of an atom of oxygen, `
  - Fix: `(1842, 'DYNAMICS', 'eb_7th_1842_v08_DIA-VII', [('INSTEAD', r'Instead\s+of\s+woad,', 49)])`

- 🔴 **FARRIERY** → **ORIGIN** (1810) sim=0.196 [topic_change] [gap: EDITORIAL]
  - `...---

**Larger...**
FAR R I E R Y.`
  - `Origin and Insertion.

From the side of the breast-bone, and the cartilages of the six last true rib`
  - Fix: `(1810, 'FARRIERY', 'eb_4th_1810_v08_FAI-FOR', [('ORIGIN', r'Origin\s+and\s+Insertion\.', 46)])`

- 🔴 **HEMP** → **CHINESE HEMP** (1797) sim=0.197 [person_bio] [gap: EDITORIAL]
  - `...      | 10½d. or 10½d.|
| 3 from do.        | 12½d.        |`
  - `Chinese Hemp, a newly discovered species of Cannabis, of which an account is given in the 72nd volum`
  - Fix: `(1797, 'HEMP', 'eb_3rd_1797_v08_IND-HYD', [('CHINESE HEMP', r'Chinese\s+Hemp,\s+a', 79)])`

- 🔴 **FUST** → **ARCHITECTURE** (1810) sim=0.197 [topic_change] [gap: OCR_GAP]
  - `...ng from the copying of manuscripts. See History of Printing.`
  - `Architecture, the shaft of a column, or the part comprehended between the base and the capital, call`
  - Fix: `(1810, 'FUST', 'eb_4th_1810_v01_JOH-MAS', [('ARCHITECTURE', r'Architecture,\s+the\s+shaft', 45)])`

- 🔴 **ANNUITIES** → **ORDER** (1823) sim=0.197 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...nches of the Peach, and ash-coloured on the larger branches.`
  - `Order II. MACROURA.

Tribe 1. PAGURII. Gen. 46. Albunea, 47. Remipes, 48. Hippa, 49. Pagurus.`
  - Fix: `(1823, 'ANNUITIES', 'eb_6th_1823_v01_MAC-ANA', [('ORDER', r'Order\s+II\.\s+MACROURA\.', 54)])`

- 🔴 **GUELPHS** → **GELDERLAND** (1810) sim=0.198 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...el, gave his name to the Gibelins. See the article Gibelins.`
  - `Gelderland, one of the united provinces, bounded on the west by Utrecht and Holland, on the east by `
  - Fix: `(1810, 'GUELPHS', 'eb_4th_1810_v05_GOT-HER', [('GELDERLAND', r'Gelderland,\s+one\s+of', 28)])`

- 🔴 **MOGODORE** → **CONSTITUENT PARTS** (1815) sim=0.198 [person_bio] [gap: VARIANT]
  - `...Its entrance is defended by a fort well furnished with guns.`
  - `Constituent Parts.

| Panzenberg | Dolomieu |
|------------|----------|
| Pure carbone | 90       | `
  - Fix: `(1815, 'MOGODORE', 'eb_5th_1815_v14_ENL-NIC', [('CONSTITUENT PARTS', r'Constituent\s+Parts\.\s+\|', 4)])`

- 🔴 **AI** → **STRAW** (1797) sim=0.199 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `... poetical, the part is oftentimes to be taken for the whole.`
  - `Straw should be prepared for the dunghill, by being laid under cattle, and sufficiently moistened. W`
  - Fix: `(1797, 'AI', 'eb_3rd_1797_v01_IND-COR', [('STRAW', r'Straw\s+should\s+be', 28)])`

- 🔴 **CHIMNEY** → **THESE** (1797) sim=0.199 [topic_change] [gap: PARSING_OR_EDITORIAL]
  - `...f those objects of misery, the unfortunate chimney-sweepers.`
  - `These experiments have been since repeated by Dr Pearson, assisted by Mr Cuthbertson. He produced, b`
  - Fix: `(1797, 'CHIMNEY', 'eb_3rd_1797_v501_ABE-IMP', [('THESE', r'These\s+experiments\s+have', 32)])`

- 🔴 **BIBLIOGRAPHY** → **VII** (1860) sim=0.200 [new_headword] (2 eds: 1842, 1860)
  - `...t be rendered profitable either to rulers or their subjects.`
  - `VII. Of Bibliographical Dictionaries and Catalogues.

The works which fall to be considered under th`
  - Fix: `(1860, 'BIBLIOGRAPHY', 'eb_8th_1860_v04_LIS-EXT', [('VII', r'VII\.\s+Of\s+Bibliographical', 77)])`

- 🔴 **NUNDINAL** → **MONTE NUOVO** (1823) sim=0.201 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...physics, Part III. Chap. IV. Of the Immortality of the Soul.`
  - `MONTE NUOVO, in the environs of Naples, blocks up the valley of Averno. "This mountain (Mr Swinburne`
  - Fix: `(1823, 'NUNDINAL', 'eb_6th_1823_v15_ENL-PAR', [('MONTE NUOVO', r'MONTE\s+NUOVO,\s+in', 28)])`

- 🔴 **OTHO** → **VENIUS** (1810) sim=0.202 [new_headword] (2 eds: 1810, 1815)
  - `...d with virulence by some, but Cicero ably defended it, &c.**`
  - `VENIUS,** a very celebrated Dutch painter. He was descended of a considerable family in Leyden, and `
  - Fix: `(1810, 'OTHO', 'eb_4th_1810_v15_ORD-PAR', [('VENIUS', r'VENIUS,\*\*\s+a\s+very', 20)])`

- 🔴 **GOLDEN** → **OROBUS** (1810) sim=0.203 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ed in Egypt, as well as those of Pythagoras many ages after.`
  - `OROBUS, bitter vetch, a genus of plants belong-
ing to the diadelphia class; and in the natural meth`
  - Fix: `(1810, 'GOLDEN', 'eb_4th_1810_v15_ORD-PAR', [('OROBUS', r'OROBUS,\s+bitter\s+vetch,', 13)])`

- 🔴 **MONEY** → **XIII** (1823) sim=0.207 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...e their vows, without obliging themselves to any new reform.`
  - `XIII. NICKEL Genus.

1. Species. Copper Coloured Nickel.`
  - Fix: `(1823, 'MONEY', 'eb_6th_1823_v14_ENL-NIC', [('XIII', r'XIII\.\s+NICKEL\s+Genus\.', 55)])`

- 🔴 **AUSTRALASIA** → **III** (1860) sim=0.208 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...days in peace.—D'Entrecasteaux, Labillardière, Flinders, &c.`
  - `III. New Guinea, or Papua, is, after Australia, not only the first in point of magnitude, but claims`
  - Fix: `(1860, 'AUSTRALASIA', 'eb_8th_1860_v04_LIS-EXT', [('III', r'III\.\s+New\s+Guinea,', 14)])`

- 🔴 **PROVIDENCE** → **NEW** (1842) sim=0.210 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...E. lat. 9° 10' S. And a third in long. 78° E. lat. 54° 5' N.`
  - `NEW, one of the Bahama Islands, in the West Indies, situated between longitude 77° 10' and 77° 38' w`
  - Fix: `(1842, 'PROVIDENCE', 'eb_7th_1842_v18_PLA-QUO', [('NEW', r'NEW,\s+one\s+of', 84)])`

- 🔴 **MINERALOGY** → **III** (1815) sim=0.211 [new_headword] (2 eds: 1778, 1815)
  - `...atches quartz slightly. Brittle. Spec. grav. 3.514 to 3.530.`
  - `III. The acumination, in which are also to be considered the parts of the acumination and the determ`
  - Fix: `(1815, 'MINERALOGY', 'eb_5th_1815_v14_ENL-NIC', [('III', r'III\.\s+The\s+acumination,', 32)])`

- 🔴 **DYNAMICS** → **III** (1842) sim=0.211 [new_headword] (2 eds: 1810, 1842)
  - `...tio of the altitude be to the altitude fy; Retarding Forces.`
  - `III.—Processes for dyeing Silk Blue.

Silk is dyed blue with indigo alone, without any proportion of`
  - Fix: `(1842, 'DYNAMICS', 'eb_7th_1842_v08_DIA-VII', [('III', r'III\.—Processes\s+for\s+dyeing', 50)])`

- 🔴 **AUSTRALASIA** → **III** (1842) sim=0.212 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...ed together. (D'Entrecasteaux, Labillardiere, Flinders, &c.)`
  - `III. This great island is, after New Holland, not only the first in point of magnitude, but claims a`
  - Fix: `(1842, 'AUSTRALASIA', 'eb_7th_1842_v04_SEV-BOR', [('III', r'III\.\s+This\s+great', 35)])`

- 🔴 **SIGN** → **NAVAL SIGNALS** (1823) sim=0.213 [new_headword] (3 eds: 1797, 1815, 1823)
  - `...n containing a 12th part of the zodiac. See Astronomy Index.`
  - `NAVAL SIGNALS. When we read at our fireside the account of an engagement, or other interesting opera`
  - Fix: `(1823, 'SIGN', 'eb_6th_1823_v19_ENL-SUG', [('NAVAL SIGNALS', r'NAVAL\s+SIGNALS\.\s+When', 0)])`

- 🔴 **BLIND** → **THOMSON** (1815) sim=0.214 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...resent deity, and taste
The joy of God to see a happy world.`
  - `THOMSON.

Much labour has been bestowed to investigate, both from reason a priori and from experimen`
  - Fix: `(1815, 'BLIND', 'eb_5th_1815_v03_ASS-DIR', [('THOMSON', r'THOMSON\.\s+Much\s+labour', 9)])`

- 🔴 **MEXICO** → **III** (1860) sim=0.214 [new_headword] (2 eds: 1842, 1860)
  - `...t was suppressed, however, by Comonfort on the 22d of March.`
  - `III.—STATISTICS OF MEXICO.

Mexico is bounded on the N. by California, New Mexico, and Texas; E. by `
  - Fix: `(1860, 'MEXICO', 'eb_8th_1860_v14_MAG-NOT', [('III', r'III\.—STATISTICS\s+OF\s+MEXICO\.', 29)])`

- 🔴 **UTICA** → **MEDICAGO** (1810) sim=0.215 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...eople, to declare unto them his word; (Deut. v. 5, iii. 19.)`
  - `MEDICAGO, SNAIL TREFOIL, a genus of plants belonging to the diadelphia clas, and in the natural meth`
  - Fix: `(1810, 'UTICA', 'eb_4th_1810_v20_SUI-PRE', [('MEDICAGO', r'MEDICAGO,\s+SNAIL\s+TREFOIL,', 0)])`

- 🔴 **UTICA** → **MEDIANA** (1810) sim=0.216 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...be considered, as also the manners, &c., of the inhabitants.`
  - `MEDIANA, the name of a vein or little vessel, made by the union of the cephalic and basilic, in the `
  - Fix: `(1810, 'UTICA', 'eb_4th_1810_v20_SUI-PRE', [('MEDIANA', r'MEDIANA,\s+the\s+name', 0)])`

- 🔴 **CONCORD** → **FORM OF CONCORD** (1810) sim=0.219 [new_headword] (2 eds: 1810, 1815)
  - `...ble to the ear, whether applied in succession or consonance.`
  - `FORM OF CONCORD, in ecclesiastical history, a standard-book among the Lutherans composed at Torgau, `
  - Fix: `(1810, 'CONCORD', 'eb_4th_1810_v06_CON-CRY', [('FORM OF CONCORD', r'FORM\s+OF\s+CONCORD,', 28)])`

- 🔴 **LAURA** → **POET LAUREATE** (1815) sim=0.222 [new_headword] (3 eds: 1810, 1815, 1823)
  - `... Cedron; the laura of the Towers, near the river Jordan, &c.`
  - `POET LAUREATE, an officer of the household of the kings of Britain, whose business consists only in `
  - Fix: `(1815, 'LAURA', 'eb_5th_1815_v11_ENL-LIE', [('POET LAUREATE', r'POET\s+LAUREATE,\s+an', 9)])`

- 🔴 **AEROSTATION** → **MONTGOLFIER'S B** (1823) sim=0.222 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...VERSAILLES
B.

CHARLES & ROBERTS B.
CAMP DE MARS.`
  - `MONTGOLFIER'S B.
FAUXBOURG OF ST GERMAIN.
wood, and its diameter should be somewhat less than that o`
  - Fix: `(1823, 'AEROSTATION', 'eb_6th_1823_v01_ART-AME', [('MONTGOLFIER'S B', r'MONTGOLFIER'S\s+B\.\s+FAUXBOURG', 92)])`

- 🔴 **HOLLAND** → **NEW HOLLAND** (1810) sim=0.223 [new_headword] (2 eds: 1797, 1810)
  - `...rns brackish; and if they are shallow, they soon become dry.`
  - `NEW HOLLAND, the largest island in the world, reaching from 10 to 44° S. Lat. and between 110 and 15`
  - Fix: `(1810, 'HOLLAND', 'eb_4th_1810_v10_HER-HYD', [('NEW HOLLAND', r'NEW\s+HOLLAND,\s+the', 2)])`

- 🔴 **MATCHING** → **DURA** (1810) sim=0.226 [new_headword] (2 eds: 1810, 1815)
  - `...everal sizes, or to the services on which they are employed.`
  - `DURA and PIA MATER, the names given by anatomists to the two membranes which surround the brain. See`
  - Fix: `(1810, 'MATCHING', 'eb_4th_1810_v12_MAH-ADD', [('DURA', r'DURA\s+and\s+PIA', 55)])`

- 🔴 **MEDICINE** → **GENUS LXIII** (1815) sim=0.226 [new_headword] [gap: EDITORIAL]
  - `...large proportion of fat meat, such as pork steaks or butter.`
  - `GENUS LXIII. HYSTERIA.

HYSTERICS.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXIII', r'GENUS\s+LXIII\.\s+HYSTERIA\.', 79)])`

- 🔴 **CONCORD** → **FORM OF CONCORD** (1815) sim=0.227 [new_headword] (2 eds: 1810, 1815)
  - `...ble to the ear, whether applied in succession or consonance.`
  - `FORM OF CONCORD, in ecclesiastical history, a standard book among the Lutherans, compiled at Torgau,`
  - Fix: `(1815, 'CONCORD', 'eb_5th_1815_v06_ENL-CRY', [('FORM OF CONCORD', r'FORM\s+OF\s+CONCORD,', 3)])`

- 🔴 **SETTING** → **ACT OF SETTLEMENT** (1815) sim=0.227 [new_headword] (2 eds: 1815, 1823)
  - `...s of a dog peculiarly trained to that purpose. See Shooting.`
  - `ACT OF SETTLEMENT, in British history, a name given to the statute 12 and 13 Will. III. cap. 2, wher`
  - Fix: `(1815, 'SETTING', 'eb_5th_1815_v19_SCR-DVI', [('ACT OF SETTLEMENT', r'ACT\s+OF\s+SETTLEMENT,', 45)])`

- 🔴 **PARR** → **SAMUEL** (1842) sim=0.227 [new_headword] (2 eds: 1842, 1860)
  - `...y, printed by John Wayland, 1545, 4to, reprinted 1561, 12mo.`
  - `SAMUEL, a critic, metaphysician, theologian, and one of the most learned classical scholars of the a`
  - Fix: `(1842, 'PARR', 'eb_7th_1842_v17_SEV-CON', [('SAMUEL', r'SAMUEL,\s+a\s+critic,', 1)])`

- 🔴 **LAURA** → **POET-LAUREATE** (1797) sim=0.228 [new_headword] (2 eds: 1797, 1842)
  - `... Cedron; the laura of the Towers, near the river Jordan, &c.`
  - `POET-LAUREATE, an officer of the household of the kings of Britain, whose business consists only in `
  - Fix: `(1797, 'LAURA', 'eb_3rd_1797_v09_IND-LES', [('POET-LAUREATE', r'POET\-LAUREATE,\s+an\s+officer', 9)])`

- 🔴 **HUDSON** → **WILLIAM** (1815) sim=0.229 [new_headword] (2 eds: 1815, 1823)
  - `...s. Such was the unfortunate end of this adventurous mariner!`
  - `WILLIAM, a celebrated English botanist, was born at Westmoreland about 1730. He was bound apprentice`
  - Fix: `(1815, 'HUDSON', 'eb_5th_1815_v10_GOT-HYD', [('WILLIAM', r'WILLIAM,\s+a\s+celebrated', 2)])`

- 🔴 **LONDON** → **III** (1815) sim=0.229 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...eldom less than 2000 hogs, which are fed entirely on grains.`
  - `III. City and Liberties of WESTMINSTER. The city of Westminster derives its name from a minster, or `
  - Fix: `(1815, 'LONDON', 'eb_5th_1815_v12_LIE-CCX', [('III', r'III\.\s+City\s+and', 70)])`

- 🔴 **LULA** → **ULLI** (1823) sim=0.230 [new_headword] (4 eds: 1778, 1797, 1810, 1823)
  - `...ast, by Pithia Lapmark on the south, and Norway on the west.`
  - `ULLI, JOHN BAPTIST, the most celebrated and most excellent musician that has appeared in France sinc`
  - Fix: `(1823, 'LULA', 'eb_6th_1823_v12_ENL-ADD', [('ULLI', r'ULLI,\s+JOHN\s+BAPTIST,', 28)])`

- 🔴 **BRIDGE** → **EXPLANATION OF THE PLATES** (1823) sim=0.232 [new_headword] (2 eds: 1823, 1860)
  - `...nt reason to be proud, for a long series of successive ages.`
  - `EXPLANATION OF THE PLATES.

Plate XLIII. fig. 1. If AB represent the distance of any two particles o`
  - Fix: `(1823, 'BRIDGE', 'eb_6th_1823_v502_AUS-CEL', [('EXPLANATION OF THE PLATES', r'EXPLANATION\s+OF\s+THE', 79)])`

- 🔴 **ATOM** → **EARTH MOON D** (1815) sim=0.233 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...e it is denominated the Epicurean Philosophy. See Epicurean.`
  - `EARTH
MOON
Distance of the Moon from the Earth

Fifth Satellite thrice the distance of the Fourth`
  - Fix: `(1815, 'ATOM', 'eb_5th_1815_v03_ASS-DIR', [('EARTH MOON D', r'EARTH\s+MOON\s+Distance', 4)])`

- 🔴 **MATCHING** → **DURA** (1815) sim=0.233 [new_headword] (2 eds: 1810, 1815)
  - `...everal fizes, or to the services on which they are employed.`
  - `DURA and PIA MATER, the names given by anatomists to the two membranes which surround the brain. See`
  - Fix: `(1815, 'MATCHING', 'eb_5th_1815_v12_LIE-CCX', [('DURA', r'DURA\s+and\s+PIA', 55)])`

- 🔴 **ATOMIC THEORY** → **ANNUITIES** (1823) sim=0.233 [new_headword] [gap: OCR_GAP]
  - `..., or of the artificial warmth of the Conservatory.
ADDENDUM.`
  - `ANNUITIES. As an addition to the article ANNUITIES, we beg to insert here an expeditious method of c`
  - Fix: `(1823, 'ATOMIC THEORY', 'eb_6th_1823_v03_ENL-BOO', [('ANNUITIES', r'ANNUITIES\.\s+As\s+an', 92)])`

- 🔴 **MENANDER** → **GENUS CXIV** (1815) sim=0.234 [new_headword] [gap: EDITORIAL]
  - `...s,
"Veniebat greatu delicatulo et languido."
Lib. v. fab. 2.`
  - `GENUS CXIV. STRABISMUS.

SQUINTING.`
  - Fix: `(1815, 'MENANDER', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CXIV', r'GENUS\s+CXIV\.\s+STRABISMUS\.', 7)])`

- 🔴 **OMEN** → **DRYDEN** (1810) sim=0.235 [new_headword] (2 eds: 1810, 1823)
  - `... upon the weft:
The ninth is good for travel, bad for theft.`
  - `DRYDEN.

From this coincidence of the superstition of the Roman poet with that of the natives of Mul`
  - Fix: `(1810, 'OMEN', 'eb_4th_1810_v15_NIC-ORA', [('DRYDEN', r'DRYDEN\.\s+From\s+this', 87)])`

- 🔴 **MEDICINE** → **ORDER III** (1815) sim=0.235 [new_headword] (2 eds: 1778, 1815)
  - `...ses, bolsters, and proper supports. See the article SURGERY.`
  - `ORDER III. IMPETIGINES.

Impetigines, Sauv. Clas X. Ord. V. Sag. Clas III. Ord. V.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('ORDER III', r'ORDER\s+III\.\s+IMPETIGINES\.', 86)])`

- 🔴 **BRIDGE** → **EXPLANATION OF THE PLATES** (1860) sim=0.235 [new_headword] (2 eds: 1823, 1860)
  - `...hering in of a new earth, wherein righteousness is to dwell.`
  - `EXPLANATION OF THE PLATES.

Plate CX. Helleborus foetidus, Stinking Hellebore, belonging to the Nat.`
  - Fix: `(1860, 'BRIDGE', 'eb_8th_1860_v05_ADA-BUR', [('EXPLANATION OF THE PLATES', r'EXPLANATION\s+OF\s+THE', 630)])`

- 🔴 **LULA** → **ULLI** (1810) sim=0.236 [new_headword] (4 eds: 1778, 1797, 1810, 1823)
  - `...ast, by Pithia Lapmark on the south, and Norway on the west.`
  - `ULLI, JOHN BAPTIST, the most celebrated and most excellent musician that has appeared in France sinc`
  - Fix: `(1810, 'LULA', 'eb_4th_1810_v17_LIE-MAH', [('ULLI', r'ULLI,\s+JOHN\s+BAPTIST,', 28)])`

- 🔴 **GUINEA** → **NEW GUINEA** (1842) sim=0.236 [new_headword] (4 eds: 1797, 1815, 1823, 1842)
  - `...med Benin, after the principal state. See the article BENIN.`
  - `NEW GUINEA, or Papua. See AUSTRALASIA.

a gold coin, struck and current in Britain. The value or rat`
  - Fix: `(1842, 'GUINEA', 'eb_7th_1842_v11_GRO-HYD', [('NEW GUINEA', r'NEW\s+GUINEA,\s+or', 70)])`

- 🔴 **LONDON** → **III** (1810) sim=0.238 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...eldom less than 2000 hogs, which are fed entirely on grains.`
  - `III. City and Liberties of Westminster. The city of Westminster derives its name from a minster, or `
  - Fix: `(1810, 'LONDON', 'eb_4th_1810_v17_LIE-MAH', [('III', r'III\.\s+City\s+and', 69)])`

- 🔴 **MENIPPEAN** → **GENUS CXXI** (1815) sim=0.240 [new_headword] [gap: EDITORIAL]
  - `...holicon of Spain. It is esteemed a masterpiece for the time.`
  - `GENUS CXXI. GONORRHŒA.

Gonorrhœa, Sauv. gen 208. Lin. 200. Vog. 118. Sag. 204.`
  - Fix: `(1815, 'MENIPPEAN', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CXXI', r'GENUS\s+CXXI\.\s+GONORRHŒA\.', 20)])`

- 🔴 **OTHO** → **VENIUS** (1815) sim=0.241 [new_headword] (2 eds: 1810, 1815)
  - `...sed with virulence by some, but Cicero ably defended it, &c.`
  - `VENIUS, a very celebrated Dutch painter. He was descended of a considerable family in Leyden, and wa`
  - Fix: `(1815, 'OTHO', 'eb_5th_1815_v15_NIC-CCC', [('VENIUS', r'VENIUS,\s+a\s+very', 28)])`

- 🔴 **ECONOMISTS** → **III** (1842) sim=0.241 [new_headword] (2 eds: 1842, 1860)
  - `...at his revenue increases. See the article Political Economy.`
  - `III. In the remarks which we have to offer on the doctrines of this sect, we must content ourselves `
  - Fix: `(1842, 'ECONOMISTS', 'eb_7th_1842_v08_DIA-VII', [('III', r'III\.\s+In\s+the', 65)])`

- 🔴 **PRUSSIA** → **STATISTICS** (1842) sim=0.241 [new_headword] [gap: VARIANT]
  - `...ority in the archiepiscopal province over which he presided.`
  - `STATISTICS.

The kingdom of Prussia is situated in the northern part of Germany. It is bounded on th`
  - Fix: `(1842, 'PRUSSIA', 'eb_7th_1842_v18_PLA-QUO', [('STATISTICS', r'STATISTICS\.\s+The\s+kingdom', 72)])`

- 🔴 **ECONOMISTS** → **III** (1860) sim=0.241 [new_headword] (2 eds: 1842, 1860)
  - `...at his revenue increases. See the article Political Economy.`
  - `III. In the remarks which we have to offer on the doctrines of this sect, we must content ourselves `
  - Fix: `(1860, 'ECONOMISTS', 'eb_8th_1860_v08_ADA-ENT', [('III', r'III\.\s+In\s+the', 65)])`

- 🔴 **BARON** → **ROBERT** (1810) sim=0.242 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...t must be borne by the husband on an escutcheon of pretence.`
  - `ROBERT, a dramatic author, who lived during the reign of Charles I. and the protectorship of Oliver `
  - Fix: `(1810, 'BARON', 'eb_4th_1810_v03_BAR-BOO', [('ROBERT', r'ROBERT,\s+a\s+dramatic', 81)])`

- 🔴 **FLAMSTEED** → **JOHN** (1810) sim=0.244 [new_headword] (3 eds: 1810, 1823, 1842)
  - `...rought, is still preserved in the manor house near the town.`
  - `JOHN, an eminent English astronomer, and the first who obtained the appointment of astronomer-royal,`
  - Fix: `(1810, 'FLAMSTEED', 'eb_4th_1810_v08_FAI-FOR', [('JOHN', r'JOHN,\s+an\s+eminent', 0)])`

- 🔴 **LONDON** → **III** (1823) sim=0.250 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...seldom less than 2000 hogs, which are fed entirely on grain.`
  - `III. City and Liberties of Westminster. The city of Westminster derives its name from a minster, or `
  - Fix: `(1823, 'LONDON', 'eb_6th_1823_v19_ENL-SUG', [('III', r'III\.\s+City\s+and', 70)])`

- 🔴 **LONDON** → **III** (1797) sim=0.252 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...0 hogs constantly grunting, and kept entirely on the grains.`
  - `III. City and Liberties of Westminster. The city of Westminster derives its name from a minster, or `
  - Fix: `(1797, 'LONDON', 'eb_3rd_1797_v10_IND-MEC', [('III', r'III\.\s+City\s+and', 70)])`

- 🔴 **MAGIC** → **DRYDEN** (1810) sim=0.252 [new_headword] (2 eds: 1810, 1815)
  - `...kling fury roll,
When all the god came rushing on her soul."`
  - `DRYDEN.

In answer to this, it is to be observed, that the temple of Apollo at Cumæ was an immense e`
  - Fix: `(1810, 'MAGIC', 'eb_4th_1810_v17_LIE-MAH', [('DRYDEN', r'DRYDEN\.\s+In\s+answer', 71)])`

- 🔴 **SWEDEN** → **STATISTICS** (1842) sim=0.253 [new_headword] [gap: VARIANT]
  - `...erve his situation, and transmit the crown to his posterity.`
  - `STATISTICS.

Sweden and Norway form together one geographical region, situated between 45° and 32° E`
  - Fix: `(1842, 'SWEDEN', 'eb_7th_1842_v21_SEV-ZYG', [('STATISTICS', r'STATISTICS\.\s+Sweden\s+and', 75)])`

- 🔴 **BRECONSHIRE** → **TRANSVERSE SECTION** (1823) sim=0.255 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...lt, or it must be capable at least of containing 75 bushels.`
  - `TRANSVERSE SECTION of the FINISHED PART of the BREAKWATER.

High Water Spring Tides
Low Water Spring`
  - Fix: `(1823, 'BRECONSHIRE', 'eb_6th_1823_v502_AUS-CEL', [('TRANSVERSE SECTION', r'TRANSVERSE\s+SECTION\s+of', 32)])`

- 🔴 **OMEN** → **DRYDEN** (1823) sim=0.261 [new_headword] (2 eds: 1810, 1823)
  - `... upon the west:
The ninth is good for travel, bad for theft.`
  - `DRYDEN.

From this coincidence of the superstition of the Roman poet with that of the natives of Mul`
  - Fix: `(1823, 'OMEN', 'eb_6th_1823_v15_ENL-PAR', [('DRYDEN', r'DRYDEN\.\s+From\s+this', 87)])`

- 🔴 **MEDICINE** → **ORDER II** (1815) sim=0.262 [new_headword] (2 eds: 1778, 1815)
  - `... without it the best remedies will prove altogether useless.`
  - `ORDER II. INTUMESCENTIAE.

Intumescentiae, Sauv. Clas X. Ord. II. Sag. Clas III. Ord. II.
Tumidof, L`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('ORDER II', r'ORDER\s+II\.\s+INTUMESCENTIAE\.', 83)])`

- 🔴 **GOLDEN** → **OROBANCHE** (1810) sim=0.266 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...s a poet, but also as a theologian, and founder of religion.`
  - `OROBANCHE, a genus of plants belonging to the didynamia clas; and in the natural method ranking unde`
  - Fix: `(1810, 'GOLDEN', 'eb_4th_1810_v15_ORD-PAR', [('OROBANCHE', r'OROBANCHE,\s+a\s+genus', 7)])`

- 🔴 **STALE** → **ANIMATED STALK** (1797) sim=0.267 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...STALE is also a name for the urine of cattle.`
  - `ANIMATED STALK. This remarkable animal was found by Mr Ives at Cuddalore; and he mentions several ki`
  - Fix: `(1797, 'STALE', 'eb_3rd_1797_v17_TRE-STR', [('ANIMATED STALK', r'ANIMATED\s+STALK\.\s+This', 28)])`

- 🔴 **STALE** → **ANIMATED STALK** (1810) sim=0.267 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...STALE is also a name for the urine of cattle.`
  - `ANIMATED STALK. This remarkable animal was found by Mr Ives at Cuddalore; and he mentions several ki`
  - Fix: `(1810, 'STALE', 'eb_4th_1810_v19_SLE-SUG', [('ANIMATED STALK', r'ANIMATED\s+STALK\.\s+This', 28)])`

- 🔴 **BARON** → **ROBERT** (1815) sim=0.268 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...muft be borne by the hufband on an efcutcheon of pre- tence.`
  - `ROBERT, a dramatic author, who lived during the reign of Charles I. and the protectorship of Oliver `
  - Fix: `(1815, 'BARON', 'eb_5th_1815_v03_ASS-DIR', [('ROBERT', r'ROBERT,\s+a\s+dramatic', 80)])`

- 🔴 **STALE** → **ANIMATED STALK** (1823) sim=0.269 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...STALE is also a name for the urine of cattle.`
  - `ANIMATED STALK. This remarkable animal was found by Mr Ives at Cuddalore: and he mentions several ki`
  - Fix: `(1823, 'STALE', 'eb_6th_1823_v19_ENL-SUG', [('ANIMATED STALK', r'ANIMATED\s+STALK\.\s+This', 28)])`

- 🔴 **HEBRIDES** → **NEW HEBRIDES** (1810) sim=0.272 [new_headword] (2 eds: 1810, 1842)
  - `...ave been adopted, and are gradually carrying into execution.`
  - `NEW HEBRIDES, a cluster of islands lying in the Great South Sea, or Pacific ocean. The northern isla`
  - Fix: `(1810, 'HEBRIDES', 'eb_4th_1810_v05_GOT-HER', [('NEW HEBRIDES', r'NEW\s+HEBRIDES,\s+a', 89)])`

- 🔴 **STALE** → **ANIMATED STALK** (1815) sim=0.272 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...STALE is also a name for the urine of cattle.`
  - `ANIMATED STALK. This remarkable animal was found by Mr Ives at Cuddalore: and he mentions several ki`
  - Fix: `(1815, 'STALE', 'eb_5th_1815_v19_SCR-DVI', [('ANIMATED STALK', r'ANIMATED\s+STALK\.\s+This', 28)])`

- 🔴 **BIBLIOGRAPHY** → **VII** (1842) sim=0.273 [new_headword] (2 eds: 1842, 1860)
  - `...t be rendered profitable either to rulers or their subjects.`
  - `VII. Of Bibliographical Dictionaries and Catalogues.

The works which fall to be considered under th`
  - Fix: `(1842, 'BIBLIOGRAPHY', 'eb_7th_1842_v04_SEV-BOR', [('VII', r'VII\.\s+Of\s+Bibliographical', 86)])`

- 🔴 **NELSON** → **NAVIGATION N** (1815) sim=0.274 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ould have made a distinguished figure in the house of peers.`
  - `NAVIGATION
Navigation, Navigation of the Ancients. See Phoenicia and Inland Navigation.`
  - Fix: `(1815, 'NELSON', 'eb_5th_1815_v14_ENL-NIC', [('NAVIGATION N', r'NAVIGATION\s+Navigation,\s+Navigation', 54)])`

- 🔴 **ADRIAN** → **POPE** (1842) sim=0.275 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...trembling, shivering, dying,
And wit and humour are no more!`
  - `POPE.

Some fragments of his Latin poetry are still extant, and there are Greek verses of his in the`
  - Fix: `(1842, 'ADRIAN', 'eb_7th_1842_v02_AAL-DES', [('POPE', r'POPE\.\s+Some\s+fragments', 61)])`

- 🔴 **MAMMALIA** → **ORDER I** (1860) sim=0.277 [new_headword] (2 eds: 1842, 1860)
  - `...process will assuredly be to his, if not to their advantage.`
  - `ORDER I.—QUADRUMANA.

QUADRUMANOUS, OR FOUR-HANDED ANIMALS.`
  - Fix: `(1860, 'MAMMALIA', 'eb_8th_1860_v14_MAG-NOT', [('ORDER I', r'ORDER\s+I\.—QUADRUMANA\.\s+QUADRUMANOUS,', 28)])`

- 🔴 **MAMMALIA** → **ORDER I** (1842) sim=0.278 [new_headword] (2 eds: 1842, 1860)
  - `...process will assuredly be to his, if not to their advantage.`
  - `ORDER I.—QUADRUMANA.

QUADRUMANOUS, OR FOUR-HANDED ANIMALS.`
  - Fix: `(1842, 'MAMMALIA', 'eb_7th_1842_v14_SEV-MEX', [('ORDER I', r'ORDER\s+I\.—QUADRUMANA\.\s+QUADRUMANOUS,', 21)])`

- 🔴 **MEDICINE** → **GENUS LXXII** (1815) sim=0.280 [new_headword] [gap: EDITORIAL]
  - `...by entirely destroying the action of the chylopoeic viterra.`
  - `GENUS LXXII. PNEUMATOSIS.

EMPHYSEMA, or Windy Swelling.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXII', r'GENUS\s+LXXII\.\s+PNEUMATOSIS\.', 83)])`

- 🔴 **AUSTRALASIA** → **VIII** (1823) sim=0.280 [new_headword] (2 eds: 1823, 1860)
  - `...hens were seen by the French. (See Cook, Labillardiere, &c.)`
  - `VIII. Though these Islands geographically belong to New Zealand, Australasia, the natives are, in th`
  - Fix: `(1823, 'AUSTRALASIA', 'eb_6th_1823_v502_AUS-CEL', [('VIII', r'VIII\.\s+Though\s+these', 61)])`

- 🔴 **AUSTRALASIA** → **VIII** (1860) sim=0.280 [new_headword] (2 eds: 1823, 1860)
  - `...d hens were seen by the French.—See Cook, Labillardière, &c.`
  - `VIII. Though these islands geographically belong to New Zealand, Australasia, the natives are, in th`
  - Fix: `(1860, 'AUSTRALASIA', 'eb_8th_1860_v04_LIS-EXT', [('VIII', r'VIII\.\s+Though\s+these', 23)])`

- 🔴 **CHRONICLE** → **VIII** (1815) sim=0.282 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...e, he might, and very probably would, conceal his authority.`
  - `VIII. The history of the discovery of the Marbles is obscure and unsatisfactory.`
  - Fix: `(1815, 'CHRONICLE', 'eb_5th_1815_v06_ENL-CRY', [('VIII', r'VIII\.\s+The\s+history', 60)])`

- 🔴 **AUSTRALASIA** → **III** (1823) sim=0.282 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...ed together. (D'Entrecasteaux, Labillardiere, Flinders, &c.)`
  - `III. This great Island is, after New Holland, not only the first in point of magnitude, but claims a`
  - Fix: `(1823, 'AUSTRALASIA', 'eb_6th_1823_v502_AUS-CEL', [('III', r'III\.\s+This\s+great', 32)])`

- 🔴 **MEDICINE** → **GENUS LXXIII** (1815) sim=0.283 [new_headword] [gap: EDITORIAL]
  - `...e. In some instances it is followed even by a complete cure.`
  - `GENUS LXXIII. TYMPANITES.

TYMPANY.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXIII', r'GENUS\s+LXXIII\.\s+TYMPANITES\.', 83)])`

- 🔴 **PARR** → **SAMUEL** (1860) sim=0.285 [new_headword] (2 eds: 1842, 1860)
  - `...d. She died in child-bed in 1548, at the age of thirty-five.`
  - `SAMUEL, a very distinguished scholar and an acute thinker, was born at Harrow-on-the-Hill on the 15t`
  - Fix: `(1860, 'PARR', 'eb_8th_1860_v17_PRI-PLA', [('SAMUEL', r'SAMUEL,\s+a\s+very', 0)])`

- 🔴 **DICKINSON** → **GREAT** (1778) sim=0.286 [new_headword] (2 eds: 1778, 1797)
  - `...asy transition, the seat itself has also acquired that name.`
  - `GREAT. adj. A relative word, denoting largeness of quantity, number, &c. serving to augment the valu`
  - Fix: `(1778, 'DICKINSON', 'eb_2nd_1778_v04_BYW-FUZ', [('GREAT', r'GREAT\.\s+adj\.\s+A', 20)])`

- 🔴 **GARDENING** → **III** (1797) sim=0.286 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...tions, their contrasts, are more important than their forms.`
  - `III. WATER. All inland water is either running or stagnated. When stagnated, it forms a lake or a po`
  - Fix: `(1797, 'GARDENING', 'eb_3rd_1797_v07_TRE-GOA', [('III', r'III\.\s+WATER\.\s+All', 15)])`

- 🔴 **ARUNDELIAN MARBLES** → **III** (1810) sim=0.286 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...iting in the time of Ptolemy Philadelphus was not on stones.`
  - `III. "The chronicle does not appear to have been engraved by public authority."`
  - Fix: `(1810, 'ARUNDELIAN MARBLES', 'eb_4th_1810_v02_ANT-ASS', [('III', r'III\.\s+"The\s+chronicle', 36)])`

- 🔴 **MEDICINE** → **GENUS XXVIII** (1815) sim=0.286 [new_headword] [gap: EDITORIAL]
  - `...es, especially cinchona and cold drink, are the most proper.`
  - `GENUS XXVIII. VARIOLA.

The SMALLPOX.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XXVIII', r'GENUS\s+XXVIII\.\s+VARIOLA\.', 46)])`

- 🔴 **GARDENING** → **III** (1823) sim=0.286 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...xions, their contrasts, are more important than their forms.`
  - `III. WATER. All inland water is either running or stagnated. When stagnated, it forms a lake or a po`
  - Fix: `(1823, 'GARDENING', 'eb_6th_1823_v09_FOR-DIR', [('III', r'III\.\s+WATER\.\s+All', 18)])`

- 🔴 **EXCUBIAE** → **LETTERS OF EXCULPATION** (1810) sim=0.287 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... to the equites, they were obliged to have an eye over them.`
  - `LETTERS OF EXCULPATION, in Scots Law, a writ or summons issued by authority of the court of justicia`
  - Fix: `(1810, 'EXCUBIAE', 'eb_4th_1810_v17_ELE-FAI', [('LETTERS OF EXCULPATION', r'LETTERS\s+OF\s+EXCULPATION,', 28)])`

- 🔴 **EXCUBIAE** → **LETTERS OF EXCULPATION** (1797) sim=0.288 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... to the equites, they were obliged to have an eye over them.`
  - `LETTERS OF EXCULPATION, in Scots law, a writ or summons issued by authority of the court of justicia`
  - Fix: `(1797, 'EXCUBIAE', 'eb_3rd_1797_v07_TRE-GOA', [('LETTERS OF EXCULPATION', r'LETTERS\s+OF\s+EXCULPATION,', 28)])`

- 🔴 **CHRONICLE** → **VIII** (1810) sim=0.288 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...e, he might, and very probably would, conceal his authority.`
  - `VIII. The history of the discovery of the Marbles is obscure and unsatisfactory.`
  - Fix: `(1810, 'CHRONICLE', 'eb_4th_1810_v17_OBS-GEN', [('VIII', r'VIII\.\s+The\s+history', 60)])`

- 🔴 **GARDENING** → **III** (1815) sim=0.288 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...xions, their contrasts, are more important than their forms.`
  - `III. WATER. All inland water is either running or stagnated. When stagnated, it forms a lake or a po`
  - Fix: `(1815, 'GARDENING', 'eb_5th_1815_v09_FOR-CCX', [('III', r'III\.\s+WATER\.\s+All', 19)])`

- 🔴 **MAGIC** → **DRYDEN** (1815) sim=0.288 [new_headword] (2 eds: 1810, 1815)
  - `...kling fury roll,
When all the god came rushing on her soul."`
  - `DRYDEN.

In answer to this, it is to be observed, that the temple of Apollo at Cumæ was an immense e`
  - Fix: `(1815, 'MAGIC', 'eb_5th_1815_v12_LIE-CCX', [('DRYDEN', r'DRYDEN\.\s+In\s+answer', 72)])`

- 🔴 **ARUNDELIAN MARBLES** → **III** (1797) sim=0.289 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...iting in the time of Ptolemy Philadelphus was not on stones.`
  - `III. "The chronicle does not appear to have been engraved by public authority."`
  - Fix: `(1797, 'ARUNDELIAN MARBLES', 'eb_3rd_1797_v02_IND-BAR', [('III', r'III\.\s+"The\s+chronicle', 35)])`

- 🔴 **MATERIA MEDICA** → **CLASS VI** (1815) sim=0.289 [new_headword] (2 eds: 1810, 1815)
  - `...z. their expelling worms from the bowels. See ANTHELMINTICS.`
  - `CLASS VI. ERRHINES.

158 Definition of errhines.`
  - Fix: `(1815, 'MATERIA MEDICA', 'eb_5th_1815_v12_LIE-CCX', [('CLASS VI', r'CLASS\s+VI\.\s+ERRHINES\.', 24)])`

- 🔴 **MEDICINE** → **GENUS LV** (1815) sim=0.289 [new_headword] [gap: EDITORIAL]
  - `...d other medicines, it has brought about a complete recovery.`
  - `GENUS LV. PALPITATIO.

PALPITATION of the HEART.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LV', r'GENUS\s+LV\.\s+PALPITATIO\.', 74)])`

- 🔴 **CHEMISTRY** → **III** (1823) sim=0.289 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...is dried and reduced to powder, it produces the same effect.`
  - `III. Of Matters peculiar to Animals in the Amphibious Class.`
  - Fix: `(1823, 'CHEMISTRY', 'eb_6th_1823_v501_EIG-DUR', [('III', r'III\.\s+Of\s+Matters', 97)])`

- 🔴 **CHRONICLE** → **VIII** (1797) sim=0.290 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...e, he might, and very probably would, conceal his authority.`
  - `VIII. The history of the discovery of the Marbles is obscure and unsatisfactory.`
  - Fix: `(1797, 'CHRONICLE', 'eb_3rd_1797_v04_TRE-OMI', [('VIII', r'VIII\.\s+The\s+history', 1)])`

- 🔴 **IMPOTENCE** → **GENUS CXXI** (1810) sim=0.290 [new_headword] [gap: VARIANT]
  - `...emper seems to exceed that of every other medicine whatever.`
  - `GENUS CXXI. GONORRHOEA.

Gonorrhoea, Sauv. gen. 208. Lin. 200. Vog. 118. Sag. 204.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('GENUS CXXI', r'GENUS\s+CXXI\.\s+GONORRHOEA\.', 57)])`

- 🔴 **DICTIONARY** → **GREAT** (1810) sim=0.290 [new_headword] (4 eds: 1810, 1815, 1842, 1860)
  - `...asy transition, the seat itself has also acquired that name.`
  - `GREAT. adj. A relative word, denoting largeness of quantity, number, &c. serving to augment the valu`
  - Fix: `(1810, 'DICTIONARY', 'eb_4th_1810_v17_CRY-DYE', [('GREAT', r'GREAT\.\s+adj\.\s+A', 13)])`

- 🔴 **CHRONICLE** → **VIII** (1823) sim=0.290 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...e, he might, and very probably would, conceal his authority.`
  - `VIII. The history of the discovery of the Marbles is obscure and unsatisfactory.`
  - Fix: `(1823, 'CHRONICLE', 'eb_6th_1823_v501_EIG-DUR', [('VIII', r'VIII\.\s+The\s+history', 63)])`

- 🔴 **DICKINSON** → **GREAT** (1797) sim=0.291 [new_headword] (2 eds: 1778, 1797)
  - `...asy transition, the seat itself has also acquired that name.`
  - `GREAT. adj. A relative word, denoting largeness of quantity, number, &c. serving to augment the valu`
  - Fix: `(1797, 'DICKINSON', 'eb_3rd_1797_v06_IND-ETH', [('GREAT', r'GREAT\.\s+adj\.\s+A', 18)])`

- 🔴 **ARUNDELIAN MARBLES** → **III** (1823) sim=0.291 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...iting in the time of Ptolemy Philadelphus was not on stones.`
  - `III. "The chronicle does not appear to have been engraved by public authority."`
  - Fix: `(1823, 'ARUNDELIAN MARBLES', 'eb_6th_1823_v02_ENL-ASS', [('III', r'III\.\s+"The\s+chronicle', 36)])`

- 🔴 **EXCUBIAE** → **LETTERS OF EXCULPATION** (1823) sim=0.291 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... to the equites, they were obliged to have an eye over them.`
  - `LETTERS OF EXCULPATION, in Scots Law, a writ or summons issued by authority of the court of justicia`
  - Fix: `(1823, 'EXCUBIAE', 'eb_6th_1823_v08_ENL-FOR', [('LETTERS OF EXCULPATION', r'LETTERS\s+OF\s+EXCULPATION,', 7)])`

- 🔴 **DICTIONARY** → **GREAT** (1815) sim=0.292 [new_headword] (4 eds: 1810, 1815, 1842, 1860)
  - `...asy transition, the seat itself has also acquired that name.`
  - `GREAT, adj. A relative word, denoting largeness of quantity, number, &c. serving to augment the valu`
  - Fix: `(1815, 'DICTIONARY', 'eb_5th_1815_v15_NIC-CCC', [('GREAT', r'GREAT,\s+adj\.\s+A', 11)])`

- 🔴 **WM FARQUHARSON** → **GENUS XLI** (1810) sim=0.293 [new_headword] [gap: VARIANT]
  - `...ry influe, assisted by a diet of asses milk and vegetables."`
  - `GENUS XLI. DYSENTERIA.

The Dysentery.`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS XLI', r'GENUS\s+XLI\.\s+DYSENTERIA\.', 43)])`

- 🔴 **MEDICINE** → **GENUS XLI** (1815) sim=0.293 [new_headword] [gap: EDITORIAL]
  - `...lar ilium, assisted by a diet of ashes milk and vegetables."`
  - `GENUS XLI. DYSENTERIA.

The Dysentery.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XLI', r'GENUS\s+XLI\.\s+DYSENTERIA\.', 65)])`

- 🔴 **IMPOTENCE** → **ORDER V** (1810) sim=0.294 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ng sleep, may be cured by tonics and a mild cooling regimen.`
  - `ORDER V. EPISCHESSES.

GENUS CXXII. OBSTIPATIO.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('ORDER V', r'ORDER\s+V\.\s+EPISCHESSES\.', 83)])`

- 🔴 **LULA** → **ULLY** (1815) sim=0.294 [new_headword] (3 eds: 1778, 1797, 1815)
  - `...his own composition, till his death, which happened in 1687.`
  - `ULLY, RAYMOND, a writer on alchemy, surnamed the Enlightened Doctor, was born in the island of Major`
  - Fix: `(1815, 'LULA', 'eb_5th_1815_v12_LIE-CCX', [('ULLY', r'ULLY,\s+RAYMOND,\s+a', 45)])`

- 🔴 **BRAZIL** → **III** (1842) sim=0.294 [new_headword] (2 eds: 1842, 1860)
  - `...es with the caiman the terror and hatred of the inhabitants.`
  - `III. Statistics.—In the first division of this sketch we Statistics have pointed out how Brazil was `
  - Fix: `(1842, 'BRAZIL', 'eb_7th_1842_v05_BOR-CAL', [('III', r'III\.\s+Statistics\.—In\s+the', 55)])`

- 🔴 **ARITHMETIC** → **CHA** (1810) sim=0.295 [new_headword] [gap: OCR_GAP]
  - `...23 10

Balance due me L. 34 10`
  - `CHA.
In multiplication, two numbers are given, and it is required to find how much the first amounts`
  - Fix: `(1810, 'ARITHMETIC', 'eb_4th_1810_v02_ANT-ASS', [('CHA', r'CHA\.\s+In\s+multiplication,', 7)])`

- 🔴 **MEDICINE** → **GENUS LXXXV** (1815) sim=0.295 [new_headword] [gap: EDITORIAL]
  - `...ious in scrophula than cold bathing, especially sea-bathing.`
  - `GENUS LXXXV. SYPHILIS.

LUES VENEREA, or French Pox.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXXV', r'GENUS\s+LXXXV\.\s+SYPHILIS\.', 87)])`

- 🔴 **EXCUBIAE** → **LETTERS OF EXCULPATION** (1815) sim=0.296 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `... to the equites, they were obliged to have an eye over them.`
  - `LETTERS OF EXCULPATION, in Scots Law, a writ or summons issued by authority of the court of judiciar`
  - Fix: `(1815, 'EXCUBIAE', 'eb_5th_1815_v08_ENL-FOR', [('LETTERS OF EXCULPATION', r'LETTERS\s+OF\s+EXCULPATION,', 28)])`

- 🔴 **MATERIA MEDICA** → **CLASS VI** (1810) sim=0.297 [new_headword] (2 eds: 1810, 1815)
  - `...epared chalk, and the same proportion of prepared red coral.`
  - `CLASS VI. WORMS. Order 2. MOLLUSCA.

18. HIRUDO MEDICINALIS. Medicinal leech. See HELMINTHOLOGY Inde`
  - Fix: `(1810, 'MATERIA MEDICA', 'eb_4th_1810_v12_MAH-ADD', [('CLASS VI', r'CLASS\s+VI\.\s+WORMS\.', 0)])`

- 🔴 **MEMPHIS** → **GENUS CXI** (1815) sim=0.297 [new_headword] [gap: EDITORIAL]
  - `...a forbearance of all mercurials, are the speediest remedies.`
  - `GENUS CXI. MUTITAS.

DUMBNESS.`
  - Fix: `(1815, 'MEMPHIS', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CXI', r'GENUS\s+CXI\.\s+MUTITAS\.', 18002)])`

- 🔴 **DICTIONARY** → **GREAT** (1842) sim=0.297 [new_headword] (4 eds: 1810, 1815, 1842, 1860)
  - `...easy transition the seat itself has also acquired that name.`
  - `GREAT, adj. A relative word, denoting largeness of quantity, number, &c., serving to augment the val`
  - Fix: `(1842, 'DICTIONARY', 'eb_7th_1842_v08_DIA-VII', [('GREAT', r'GREAT,\s+adj\.\s+A', 13)])`

- 🔴 **DICTIONARY** → **GREAT** (1860) sim=0.297 [new_headword] (4 eds: 1810, 1815, 1842, 1860)
  - `...easy transition the seat itself has also acquired that name.`
  - `GREAT, adj. A relative word, denoting largeness of quantity, number, &c., serving to augment the val`
  - Fix: `(1860, 'DICTIONARY', 'eb_8th_1860_v15_MIL-NAV', [('GREAT', r'GREAT,\s+adj\.\s+A', 16)])`

- 🔴 **ENTOMOLOGY** → **PENTAMERA** (1842) sim=0.298 [new_headword] (2 eds: 1842, 1860)
  - `...at sections, according to the number of joints in the tarsi.`
  - `PENTAMERA.

All the Tarsi composed of Five Joints.`
  - Fix: `(1842, 'ENTOMOLOGY', 'eb_7th_1842_v09_ENG-FRA', [('PENTAMERA', r'PENTAMERA\.\s+All\s+the', 17)])`

- 🔴 **ENTOMOLOGY** → **PENTAMERA** (1860) sim=0.298 [new_headword] (2 eds: 1842, 1860)
  - `...at sections, according to the number of joints in the tarsi.`
  - `PENTAMERA.

All the Tarsi composed of Five Joints.`
  - Fix: `(1860, 'ENTOMOLOGY', 'eb_8th_1860_v09_ENT-FRA', [('PENTAMERA', r'PENTAMERA\.\s+All\s+the', 29)])`

- 🔴 **GEOMETRY** → **III** (1860) sim=0.298 [new_headword] (2 eds: 1842, 1860)
  - `...\( L'E' = 2692'7 \)

\( L'F' = 2032'7 \)`
  - `III. Between any two finite magnitudes of the same kind there subsists a certain relation, in respec`
  - Fix: `(1860, 'GEOMETRY', 'eb_8th_1860_v10_ADA-GRA', [('III', r'III\.\s+Between\s+any', 26)])`

- 🔴 **GARDENING** → **III** (1810) sim=0.299 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...xions, their contrasts, are more important than their forms.`
  - `III. WATER. All inland water is either running or stagnated. When stagnated, it forms a lake or a po`
  - Fix: `(1810, 'GARDENING', 'eb_4th_1810_v09_FAR-GOT', [('III', r'III\.\s+WATER\.\s+All', 74)])`

- 🔴 **WALSINGHAM** → **THOMAS** (1810) sim=0.299 [new_headword] [gap: OCR_GAP]
  - `... north-north-east of London. E. Long. o. 53. N. Lat. 52. 56.`
  - `THOMAS, an English Benedictine monk of the monastery of St. Alban's, who lived about the year 1450. `
  - Fix: `(1810, 'WALSINGHAM', 'eb_4th_1810_v20_SUI-PRE', [('THOMAS', r'THOMAS,\s+an\s+English', 0)])`

- 🔴 **MEDICINE** → **GENUS V** (1815) sim=0.299 [new_headword] [gap: EDITORIAL]
  - `...aring it easily, its relieving delirium, and inducing sleep.`
  - `GENUS V. TYPHUS; the Typhous FEVER.
Typhus, Sauv. gen. 82. Sag. 677.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS V', r'GENUS\s+V\.\s+TYPHUS;', 23)])`

- 🔴 **ARUNDELIAN MARBLES** → **III** (1815) sim=0.301 [new_headword] (4 eds: 1797, 1810, 1815, 1823)
  - `...iting in the time of Ptolemy Philadelphus was not on stones.`
  - `III. "The chronicle does not appear to have been engraved by public authority."`
  - Fix: `(1815, 'ARUNDELIAN MARBLES', 'eb_5th_1815_v02_ENL-ASS', [('III', r'III\.\s+"The\s+chronicle', 36)])`

- 🔴 **CHEMISTRY** → **III** (1815) sim=0.301 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...is dried and reduced to powder, it produces the same effect.`
  - `III. Of
III. Of Matters peculiar to Animals in the Amphibious Clafs.`
  - Fix: `(1815, 'CHEMISTRY', 'eb_5th_1815_v05_ENL-CHI', [('III', r'III\.\s+Of\s+III\.', 96)])`

- 🔴 **BERNOULLI** → **VIII** (1860) sim=0.301 [new_headword] (2 eds: 1842, 1860)
  - `...its advantageous application to various economical purposes.`
  - `VIII. Bernoulli, James, younger brother of the preceding, and the second of this name, was born at B`
  - Fix: `(1860, 'BERNOULLI', 'eb_8th_1860_v04_LIS-EXT', [('VIII', r'VIII\.\s+Bernoulli,\s+James,', 84)])`

- 🔴 **CHEMISTRY** → **III** (1810) sim=0.302 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...is dried and reduced to powder, it produces the same effect.`
  - `III. Of
III. Of Matters peculiar to Animals in the Amphibious Class.`
  - Fix: `(1810, 'CHEMISTRY', 'eb_4th_1810_v05_CHA-CHI', [('III', r'III\.\s+Of\s+III\.', 96)])`

- 🔴 **WM FARQUHARSON** → **GENUS LIV** (1810) sim=0.302 [new_headword] [gap: VARIANT]
  - `...d other medicines, it has brought about a complete recovery.`
  - `GENUS LIV. PALPITATIO.

PALPITATION OF THE HEART.`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS LIV', r'GENUS\s+LIV\.\s+PALPITATIO\.', 75)])`

- 🔴 **WM FARQUHARSON** → **GENUS LII** (1810) sim=0.304 [new_headword] [gap: VARIANT]
  - `...commending it at least in every obdurate instance of chorea.`
  - `GENUS LII. RAPHANIA.

Raphania, Lin. 155. Vog. 143. Lin. Amoen. Acad. vol. vi.
Convulvio raphania, S`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS LII', r'GENUS\s+LII\.\s+RAPHANIA\.', 70)])`

- 🔴 **IMPOTENCE** → **ORDER IV** (1810) sim=0.305 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...city has been found to perform surprising cures in this way.`
  - `ORDER IV. APOCENOSES.

Apocenoae, Vog. Clas. II. Ord. II.
Fluxus, Sauv. Clas IX. Sag. Clas V.
Morbi `
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('ORDER IV', r'ORDER\s+IV\.\s+APOCENOSES\.', 39)])`

- 🔴 **WM FARQUHARSON** → **GENUS XXXIII** (1810) sim=0.305 [new_headword] [gap: VARIANT]
  - `...is intention we have often employed it with great advantage.`
  - `GENUS XXXIII. URTICARIA.

NETTLE-RASH.`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS XXXIII', r'GENUS\s+XXXIII\.\s+URTICARIA\.', 10)])`

- 🔴 **LOGIC** → **VII** (1778) sim=0.307 [new_headword] (2 eds: 1778, 1815)
  - `... the exercise of faculties so imperfect and limited as ours.`
  - `VII. Before we conclude this chapter, it may not be improper to take notice of the distinction of it`
  - Fix: `(1778, 'LOGIC', 'eb_2nd_1778_v06_BYW-IND', [('VII', r'VII\.\s+Before\s+we', 46)])`

- 🔴 **MELITENSIS TERRA** → **GENUS XC** (1815) sim=0.308 [new_headword] [gap: EDITORIAL]
  - `...English physician; the dose is four ounces every sixth hour.`
  - `GENUS XC. TRICHOMA.

The PLICA POLONICA, or Plaited Hair.`
  - Fix: `(1815, 'MELITENSIS TERRA', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XC', r'GENUS\s+XC\.\s+TRICHOMA\.', 44955)])`

- 🔴 **LULA** → **ULLY** (1797) sim=0.309 [new_headword] (3 eds: 1778, 1797, 1815)
  - `...his own composition, till his death, which happened in 1687.`
  - `ULLY (Raymond), a famous writer, surnamed the Enlightened Doctor, was born in the island of Majorca `
  - Fix: `(1797, 'LULA', 'eb_3rd_1797_v10_IND-MEC', [('ULLY', r'ULLY\s+\(Raymond\),\s+a', 45)])`

- 🔴 **CARPENTRY** → **III** (1823) sim=0.310 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...5}{6} \) is a very good mean for representing these results.`
  - `III. ELEMENTS OF CARPENTRY.

"Carpentry is the art of framing timber for the purposes of Architectur`
  - Fix: `(1823, 'CARPENTRY', 'eb_6th_1823_v502_AUS-CEL', [('III', r'III\.\s+ELEMENTS\s+OF', 7)])`

- 🔴 **GEOMETRY** → **III** (1842) sim=0.310 [new_headword] (2 eds: 1842, 1860)
  - `...$G'E'$ or $G'T' = 330°0$

$L'E' = 2692°7$`
  - `III. Between any two finite magnitudes of the same kind there subsists a certain relation, in respec`
  - Fix: `(1842, 'GEOMETRY', 'eb_7th_1842_v10_SEV-GRO', [('III', r'III\.\s+Between\s+any', 36)])`

- 🔴 **IMPOTENCE** → **GENUS CXI** (1810) sim=0.311 [new_headword] [gap: VARIANT]
  - `...a forbearance of all mercurials, are the speediest remedies.`
  - `GENUS CXI. MUTITAS.

DUMBNESS.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('GENUS CXI', r'GENUS\s+CXI\.\s+MUTITAS\.', 4)])`

- 🔴 **MEDICINE** → **GENUS XXXIII** (1815) sim=0.311 [new_headword] [gap: EDITORIAL]
  - `...is intention we have often employed it with great advantage.`
  - `GENUS XXXIII. URTICARIA.

NETTLE-RASH.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XXXIII', r'GENUS\s+XXXIII\.\s+URTICARIA\.', 55)])`

- 🔴 **MEMORY** → **GENUS CVI** (1815) sim=0.311 [new_headword] [gap: EDITORIAL]
  - `...ntly prevented it where it would otherwise have taken place.`
  - `GENUS CVI. NOSTALGIA.

Vehement Desire of revisiting one's Country.`
  - Fix: `(1815, 'MEMORY', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CVI', r'GENUS\s+CVI\.\s+NOSTALGIA\.', 3736)])`

- 🔴 **POLITICAL** → **MAY** (1823) sim=0.311 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ple subsist, and out of which all taxes...
POLITICAL ECONOMY`
  - `MAY be defined the science which relates to the production, multiplication and distribution of Wealt`
  - Fix: `(1823, 'POLITICAL', 'eb_6th_1823_v17_ENL-RHI', [('MAY', r'MAY\s+be\s+defined', 0)])`

- 🔴 **MEXICO** → **III** (1842) sim=0.311 [new_headword] (2 eds: 1842, 1860)
  - `...Congress as yet in the least degree countenanced the revolt.`
  - `III.—STATISTICS OF MEXICO.

The republic of Mexico is bounded on the east and south-east by the Gulf`
  - Fix: `(1842, 'MEXICO', 'eb_7th_1842_v14_SEV-MEX', [('III', r'III\.—STATISTICS\s+OF\s+MEXICO\.', 91)])`

- 🔴 **ADRIAN** → **POPE** (1810) sim=0.312 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...trembling, shivering, dying,
And wit and humour are no more!`
  - `POPE.

Some fragments of his Latin poetry are still extant, and there are Greek verses of his in the`
  - Fix: `(1810, 'ADRIAN', 'eb_4th_1810_v08_AAR-AGR', [('POPE', r'POPE\.\s+Some\s+fragments', 22)])`

- 🔴 **ADRIAN** → **POPE** (1815) sim=0.312 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...rembling, thriv'ring, dying,
And wit and humour are no more!`
  - `POPE.

Some fragments of his Latin poetry are still extant, and there are Greek verses of his in the`
  - Fix: `(1815, 'ADRIAN', 'eb_5th_1815_v01_ENL-AME', [('POPE', r'POPE\.\s+Some\s+fragments', 45)])`

- 🔴 **MALACIA** → **ORDER IV** (1823) sim=0.312 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...city has been found to perform surprising cures in this way.`
  - `ORDER IV. APOCENOSES.

Apocenos, Vog. Class II. Ord. II.
Fluxus, Sauv. Class IX. Sag. Class V.
Morbi`
  - Fix: `(1823, 'MALACIA', 'eb_6th_1823_v13_ENL-MIC', [('ORDER IV', r'ORDER\s+IV\.\s+APOCENOSES\.', 26)])`

- 🔴 **BOTANY** → **NATURAL CLASSIFICATION OF PLANTS** (1842) sim=0.312 [new_headword] (2 eds: 1823, 1842)
  - `...uggested, and the attempts that have been made, respecting a`
  - `NATURAL CLASSIFICATION OF PLANTS.

"The sexual system of Linnaeus lays no claim to the merit of bein`
  - Fix: `(1842, 'BOTANY', 'eb_7th_1842_v05_BOR-CAL', [('NATURAL CLASSIFICATION OF PLANTS', r'NATURAL\s+CLASSIFICATION\s+OF', 70)])`

- 🔴 **IMPOTENCE** → **GENUS CXX** (1810) sim=0.313 [new_headword] [gap: VARIANT]
  - `...he bread, she had a stool or two every day more than common.`
  - `GENUS CXX. ENURESIS.

An involuntary Flux of Urine.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('GENUS CXX', r'GENUS\s+CXX\.\s+ENURESIS\.', 54)])`

- 🔴 **NATURAL HISTORY** → **III** (1823) sim=0.313 [new_headword] (2 eds: 1810, 1823)
  - `...Z. Palmae, - 14

Total, 14,807 (c).`
  - `III. IN THE MINERAL KINGDOM.

Minerals are divided into four great classes, viz. Earths and Stones, `
  - Fix: `(1823, 'NATURAL HISTORY', 'eb_6th_1823_v14_ENL-NIC', [('III', r'III\.\s+IN\s+THE', 51)])`

- 🔴 **BOTANY** → **NATURAL CLASSIFICATION OF PLANTS** (1823) sim=0.313 [new_headword] (2 eds: 1823, 1842)
  - `...uggested, and the attempts that have been made, respecting a`
  - `NATURAL CLASSIFICATION OF PLANTS.

The sexual system of Linneaus lays no claim to the Sexual System,`
  - Fix: `(1823, 'BOTANY', 'eb_6th_1823_v502_AUS-CEL', [('NATURAL CLASSIFICATION OF PLANTS', r'NATURAL\s+CLASSIFICATION\s+OF', 9)])`

- 🔴 **MEDICINE** → **GENUS XXXVIII** (1815) sim=0.314 [new_headword] [gap: EDITORIAL]
  - `...Simmons has seen it evidently of great use in several cases.`
  - `GENUS XXXVIII. HÆMORRHOIS.

HÆMORRHOIDS, or PILES.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS XXXVIII', r'GENUS\s+XXXVIII\.\s+HÆMORRHOIS\.', 60)])`

- 🔴 **ACADEMY** → **III** (1860) sim=0.314 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...liberality of Brendelius, formerly protochirurgus at Vienna.`
  - `III. ECCLESIASTICAL ACADEMIES. Under this head may be mentioned the academy at Bologna in Italy, ins`
  - Fix: `(1860, 'ACADEMY', 'eb_8th_1860_v02_ADA-GEN', [('III', r'III\.\s+ECCLESIASTICAL\s+ACADEMIES\.', 12)])`

- 🔴 **MEDICINE** → **ORDER V** (1778) sim=0.315 [new_headword] [gap: OCR_GAP]
  - `...g sleep, may be cured by tonics, and a mild cooling regimen.`
  - `ORDER V. EPISCHESSES.

CCIX. OBSTIPATIO; COSTIVENESS.`
  - Fix: `(1778, 'MEDICINE', 'eb_2nd_1778_v06_BYW-IND', [('ORDER V', r'ORDER\s+V\.\s+EPISCHESSES\.', 83)])`

- 🔴 **TRIM** → **PROBLEM** (1810) sim=0.317 [new_headword] (2 eds: 1810, 1815)
  - `...p and the effort of the wind upon her sails. See SEAMANSHIP.`
  - `PROBLEM.
Having given the sum of any two quantities and also their difference, to find each of the q`
  - Fix: `(1810, 'TRIM', 'eb_4th_1810_v20_SUI-PRE', [('PROBLEM', r'PROBLEM\.\s+Having\s+given', 20)])`

- 🔴 **MEMORY** → **GENUS CIV** (1815) sim=0.317 [new_headword] [gap: EDITORIAL]
  - `...t is merely prompted by an uncommon and inexplicable desire.`
  - `GENUS CIV. SATYRIASIS.
Satyriasis, Sauv. gen. 228. Lin. 81. Sag. 340.`
  - Fix: `(1815, 'MEMORY', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CIV', r'GENUS\s+CIV\.\s+SATYRIASIS\.', 3730)])`

- 🔴 **TRIM** → **PROBLEM** (1815) sim=0.317 [new_headword] (2 eds: 1810, 1815)
  - `...p and the effort of the wind upon her sails. See SEAMANSHIP.`
  - `PROBLEM.

Having given the sum of any two quantities and also their difference, to find each of the `
  - Fix: `(1815, 'TRIM', 'eb_5th_1815_v20_SUI-DIR', [('PROBLEM', r'PROBLEM\.\s+Having\s+given', 45)])`

- 🔴 **NATURAL HISTORY** → **III** (1810) sim=0.318 [new_headword] (2 eds: 1810, 1823)
  - `...Z. Palmae, - 14

Total, 14,827 (c).`
  - `III. IN THE MINERAL KINGDOM.

Minerals are divided into four great classes, viz. Earths and Stones, `
  - Fix: `(1810, 'NATURAL HISTORY', 'eb_4th_1810_v14_MOR-NIA', [('III', r'III\.\s+IN\s+THE', 53)])`

- 🔴 **WM FARQUHARSON** → **GENUS LVIII** (1810) sim=0.319 [new_headword] [gap: VARIANT]
  - `... too uncertain and too dangerous to be employed in practice.`
  - `GENUS LVIII. PYROSIS.

The Heart-Burn.`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS LVIII', r'GENUS\s+LVIII\.\s+PYROSIS\.', 82)])`

- 🔴 **ADRIAN** → **POPE** (1823) sim=0.319 [new_headword] (4 eds: 1810, 1815, 1823, 1842)
  - `...trembling, shiv'ring, dying,
And wit and humour are no more!`
  - `POPE.

Some fragments of his Latin poetry are still extant, and there are Greek verses of his in the`
  - Fix: `(1823, 'ADRIAN', 'eb_6th_1823_v01_ART-AME', [('POPE', r'POPE\.\s+Some\s+fragments', 45)])`

- 🔴 **PUTTY SOMETIMES ALSO** → **TERRA PUZZOLANA** (1797) sim=0.320 [new_headword] [gap: EDITORIAL]
  - `...ishing and giving the last gloss to works of iron and steel.`
  - `TERRA PUZZOLANA, or Pozzolana, is a greyish kind of earth used in Italy for building under water. Th`
  - Fix: `(1797, 'PUTTY SOMETIMES ALSO', 'eb_3rd_1797_v15_IND-RAN', [('TERRA PUZZOLANA', r'TERRA\s+PUZZOLANA,\s+or', 0)])`

- 🔴 **PARR** → **THOMAS** (1810) sim=0.320 [new_headword] [gap: OCR_GAP]
  - `...mo. 3. Other Meditations, Prayers, Letters, &c. unpublished.`
  - `THOMAS, or OLD PARR, a remarkable Englishman, who lived in the reigns of ten kings and queens; marri`
  - Fix: `(1810, 'PARR', 'eb_4th_1810_v15_ORD-PAR', [('THOMAS', r'THOMAS,\s+or\s+OLD', 10)])`

- 🔴 **THEOGNIS** → **NUMEN** (1810) sim=0.320 [new_headword] (2 eds: 1810, 1815)
  - `...affirms of Cumberland, that "he excels all men in fixing the`
  - `NUMEN, ET VIM DRORUM; deinde aliquo tempore, patefactis terrae faucibus, ex illis abditis sedibus ev`
  - Fix: `(1810, 'THEOGNIS', 'eb_4th_1810_v20_SUI-PRE', [('NUMEN', r'NUMEN,\s+ET\s+VIM', 5)])`

- 🔴 **BRITAIN** → **ORDER IV** (1823) sim=0.320 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... been sometimes used with the same intentions as the leaves.`
  - `ORDER IV. POLYGAMIA NECESSARIA.

985. MILLERIA.
Two species; viz. quinquiflora, biflora. Panama, Ver`
  - Fix: `(1823, 'BRITAIN', 'eb_6th_1823_v502_AUS-CEL', [('ORDER IV', r'ORDER\s+IV\.\s+POLYGAMIA', 130)])`

- 🔴 **NAVEW** → **THEORY OF NAVIGATION** (1815) sim=0.321 [new_headword] (2 eds: 1815, 1823)
  - `...not only of individual utility, but of national importance."`
  - `THEORY OF NAVIGATION.

The motion of a ship in the water is well known to depend on the action of th`
  - Fix: `(1815, 'NAVEW', 'eb_5th_1815_v14_ENL-NIC', [('THEORY OF NAVIGATION', r'THEORY\s+OF\s+NAVIGATION\.', 16)])`

- 🔴 **PUTTY SOMETIMES ALSO** → **TERRA PUZZULANA** (1815) sim=0.321 [new_headword] (2 eds: 1810, 1815)
  - `...ishing and giving the last glost to works of iron and steel.`
  - `TERRA PUZZULANA, or Pozzolana, is a grayish kind of earth used in Italy for building under water. Th`
  - Fix: `(1815, 'PUTTY SOMETIMES ALSO', 'eb_5th_1815_v17_ENL-RHI', [('TERRA PUZZULANA', r'TERRA\s+PUZZULANA,\s+or', 0)])`

- 🔴 **ACADEMY** → **III** (1842) sim=0.321 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...liberality of Brenndellius, formerly protosurgeon at Vienna.`
  - `III. Ecclesiastical Academies. Under this head may be mentioned the academy at Bologna in Italy, ins`
  - Fix: `(1842, 'ACADEMY', 'eb_7th_1842_v02_AAL-DES', [('III', r'III\.\s+Ecclesiastical\s+Academies\.', 7)])`

- 🔴 **ACADEMY** → **VIII** (1778) sim=0.322 [new_headword] (3 eds: 1778, 1842, 1860)
  - `...a HISTORICAE LUSITANAE, INSTITUTA VI. IDUS DECEMBRIS MDCCXX.`
  - `VIII. Academies of Antiquities; as that at Cortona in Italy, and at Upsal in Sweden. The first is de`
  - Fix: `(1778, 'ACADEMY', 'eb_2nd_1778_v01_AA-AND', [('VIII', r'VIII\.\s+Academies\s+of', 52)])`

- 🔴 **LULA** → **ULLY** (1778) sim=0.322 [new_headword] (3 eds: 1778, 1797, 1815)
  - `...his own composition, till his death, which happened in 1687.`
  - `ULLY (Raymond), a famous writer, surnamed the Enlightened Deuter, was born in the island of Majorca `
  - Fix: `(1778, 'LULA', 'eb_2nd_1778_v06_BYW-IND', [('ULLY', r'ULLY\s+\(Raymond\),\s+a', 35)])`

- 🔴 **DICKINSON** → **TALL** (1797) sim=0.322 [new_headword] (2 eds: 1778, 1797)
  - `...g attended with peculiar degrees of guilt; as, high treason.`
  - `TALL. adj. Something elevated to a considerable degree in a perpendicular direction. Opposed to low.`
  - Fix: `(1797, 'DICKINSON', 'eb_3rd_1797_v06_IND-ETH', [('TALL', r'TALL\.\s+adj\.\s+Something', 39)])`

- 🔴 **IMPOTENCE** → **GENUS CXV** (1810) sim=0.322 [new_headword] [gap: VARIANT]
  - `...hat in both these respects, which also facilitates the cure.`
  - `GENUS CXV. CONTRACTURA.

Contractions of the Limbs.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('GENUS CXV', r'GENUS\s+CXV\.\s+CONTRACTURA\.', 36)])`

- 🔴 **DICTIONARY** → **TALL** (1810) sim=0.322 [new_headword] (2 eds: 1810, 1815)
  - `...g attended with peculiar degrees of guilt; as, high treason.`
  - `TALL. adj. Something elevated to a considerable degree in a perpendicular direction. Opposed to low.`
  - Fix: `(1810, 'DICTIONARY', 'eb_4th_1810_v17_CRY-DYE', [('TALL', r'TALL\.\s+adj\.\s+Something', 35)])`

- 🔴 **DICTIONARY** → **TALL** (1815) sim=0.322 [new_headword] (2 eds: 1810, 1815)
  - `...g attended with peculiar degrees of guilt; as, high treason.`
  - `TALL. adj. Something elevated to a considerable degree in a perpendicular direction. Opposed to low.`
  - Fix: `(1815, 'DICTIONARY', 'eb_5th_1815_v15_NIC-CCC', [('TALL', r'TALL\.\s+adj\.\s+Something', 32)])`

- 🔴 **ACADEMY** → **III** (1823) sim=0.322 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...d Brussels, both of which have published their Transactions.`
  - `III. Academies of the Fine Arts. In 1778, an academy of painting and sculpture was established at Tu`
  - Fix: `(1823, 'ACADEMY', 'eb_6th_1823_v01_MAC-ANA', [('III', r'III\.\s+Academies\s+of', 6)])`

- 🔴 **GEOMETRY** → **XVII** (1842) sim=0.323 [new_headword] (2 eds: 1842, 1860)
  - `... is of the same length. Therefore we have BE or bE = 1474.5.`
  - `XVII. By mixing, when the sum of the first and second is to their difference, as the sum of the thir`
  - Fix: `(1842, 'GEOMETRY', 'eb_7th_1842_v10_SEV-GRO', [('XVII', r'XVII\.\s+By\s+mixing,', 38)])`

- 🔴 **ZEALAND** → **NEW ZEALAND** (1823) sim=0.324 [new_headword] (2 eds: 1810, 1823)
  - `... Scheldt, on the south; and by the German ocean on the west.`
  - `NEW ZEALAND, a country of Asia, in the South Pacific ocean, first discovered by Tasman, the Dutch na`
  - Fix: `(1823, 'ZEALAND', 'eb_6th_1823_v20_ENL-ZYG', [('NEW ZEALAND', r'NEW\s+ZEALAND,\s+a', 28)])`

- 🔴 **MEDICAL JURISPRUDENCE** → **III** (1860) sim=0.324 [new_headword] (2 eds: 1842, 1860)
  - `...the event was due to spontaneous changes in the living body.`
  - `III.—FORGERY AND FALSIFICATION OF DOCUMENTS.—Forgery. This may be of two kinds:`
  - Fix: `(1860, 'MEDICAL JURISPRUDENCE', 'eb_8th_1860_v14_MAG-NOT', [('III', r'III\.—FORGERY\s+AND\s+FALSIFICATION', 4)])`

- 🔴 **MENDICANTS** → **GENUS CXV** (1815) sim=0.325 [new_headword] [gap: EDITORIAL]
  - `...hat in both these respects, which also facilitates the cure.`
  - `GENUS CXV. CONTRACTURA.

Contractions of the Limbs.`
  - Fix: `(1815, 'MENDICANTS', 'eb_5th_1815_v13_MAT-CCC', [('GENUS CXV', r'GENUS\s+CXV\.\s+CONTRACTURA\.', 10303)])`

- 🔴 **BOYLE** → **ORDER II** (1823) sim=0.325 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... and goats eat it. Swine refuse it. Cows are not fond of it.`
  - `ORDER II. DIGYNIA.

259. CRUZITA.
One species; viz. hispanica.`
  - Fix: `(1823, 'BOYLE', 'eb_6th_1823_v04_ENL-BUR', [('ORDER II', r'ORDER\s+II\.\s+DIGYNIA\.', 266)])`

- 🔴 **MEDICINE** → **ORDER VI** (1778) sim=0.326 [new_headword] [gap: OCR_GAP]
  - `...ce any effect. This patient also dreaded the sight of a dog.`
  - `ORDER VI. VESANIAE.

Paranoia, Veg. Clas IX.
Deliria, Sauv. Clas. VIII. Ord. III. Sag. Clas XI. Ord.`
  - Fix: `(1778, 'MEDICINE', 'eb_2nd_1778_v06_BYW-IND', [('ORDER VI', r'ORDER\s+VI\.\s+VESANIAE\.', 71)])`

- 🔴 **POETRY** → **ABCD** (1815) sim=0.326 [new_headword] [gap: OCR_GAP]
  - `...espects since his death by an ingenious person of that city.`
  - `ABCD (fig. 99.) is an iron cylinder, truly bored within, and evacuated at top like a cup. EFGH is an`
  - Fix: `(1815, 'POETRY', 'eb_5th_1815_v17_ENL-RHI', [('ABCD', r'ABCD\s+\(fig\.\s+99\.\)', 170)])`

- 🔴 **HERALDRY** → **ART** (1823) sim=0.327 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...ly retained this bearing without any alteration or addition.`
  - `ART. 2. OF MODERN DIFFERENCES.

The modern differences which the English have adopted not only for t`
  - Fix: `(1823, 'HERALDRY', 'eb_6th_1823_v10_ENL-HYD', [('ART', r'ART\.\s+2\.\s+OF', 10)])`

- 🔴 **BERNOULLI** → **VIII** (1842) sim=0.327 [new_headword] (2 eds: 1842, 1860)
  - `...its advantageous application to various economical purposes.`
  - `VIII. BERNOULLI, James, younger brother of the preceding, and the second of this name, was born at B`
  - Fix: `(1842, 'BERNOULLI', 'eb_7th_1842_v04_SEV-BOR', [('VIII', r'VIII\.\s+BERNOULLI,\s+James,', 83)])`

- 🔴 **IMPOTENCE** → **GENUS CXIV** (1810) sim=0.328 [new_headword] [gap: VARIANT]
  - `...and then it cannot by any pains whatever be totally removed.`
  - `GENUS CXIV. STRABISMUS.

SQUINTING.`
  - Fix: `(1810, 'IMPOTENCE', 'eb_4th_1810_v13_GEN-MIC', [('GENUS CXIV', r'GENUS\s+CXIV\.\s+STRABISMUS\.', 11)])`

- 🔴 **MEDICINE** → **GENUS IX** (1815) sim=0.328 [new_headword] [gap: EDITORIAL]
  - `...t is the complete separation of the diseased from the found.`
  - `GENUS IX. PHRENITIS.

PHRENSY, or Inflammation of the BRAIN.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS IX', r'GENUS\s+IX\.\s+PHRENITIS\.', 32)])`

- 🔴 **WILL** → **LUCAS PEPYS** (1810) sim=0.329 [new_headword] [gap: EDITORIAL]
  - `... ravages at least, if not to the existence, of the smallpox.`
  - `LUCAS PEPYS, PRESIDENT.

Royal College of Physicians,`
  - Fix: `(1810, 'WILL', 'eb_4th_1810_v20_SUI-PRE', [('LUCAS PEPYS', r'LUCAS\s+PEPYS,\s+PRESIDENT\.', 30)])`

- 🔴 **MEDICINE** → **GENUS LXIV** (1815) sim=0.329 [new_headword] [gap: EDITORIAL]
  - `... of the alimentary canal, by cathartics frequently repeated.`
  - `GENUS LXIV. HYDROPHOBIA.

The Dread of Water.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXIV', r'GENUS\s+LXIV\.\s+HYDROPHOBIA\.', 79)])`

- 🔴 **BRENTFORD** → **ORDER III** (1823) sim=0.329 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...r ex aloe et rheo, the pilulae stomachicae, and some others.`
  - `ORDER III. HEXAGYNIA.

804. BUTOMUS, or Flowering rush.
One species; viz. * umbellatus.`
  - Fix: `(1823, 'BRENTFORD', 'eb_6th_1823_v04_ENL-BUR', [('ORDER III', r'ORDER\s+III\.\s+HEXAGYNIA\.', 26015)])`

- 🔴 **HEMP** → **ORDER IV** (1823) sim=0.330 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...order of Mollusca contains 32 genera, and about
433 species.`
  - `ORDER IV. ZOOPHYTA.

The creatures ranked under this order seem to hold
a middle rank between animal`
  - Fix: `(1823, 'HEMP', 'eb_6th_1823_v10_ENL-HYD', [('ORDER IV', r'ORDER\s+IV\.\s+ZOOPHYTA\.', 342)])`

- 🔴 **GEOMETRY** → **XVII** (1860) sim=0.330 [new_headword] (2 eds: 1842, 1860)
  - `... to the westward, and the latter 30° to the eastward, of DE.`
  - `XVII. By mixing, when the sum of the first and second is to their difference, as the sum of the thir`
  - Fix: `(1860, 'GEOMETRY', 'eb_8th_1860_v10_ADA-GRA', [('XVII', r'XVII\.\s+By\s+mixing,', 28)])`

- 🔴 **ACADEMY** → **VIII** (1842) sim=0.331 [new_headword] (3 eds: 1778, 1842, 1860)
  - `...of the Sittientes at Bologna. We are not aware of any other.`
  - `VIII. ACADEMIES OF HISTORY. The first of these to which we shall advert, is the Royal Academy of Por`
  - Fix: `(1842, 'ACADEMY', 'eb_7th_1842_v02_AAL-DES', [('VIII', r'VIII\.\s+ACADEMIES\s+OF', 54)])`

- 🔴 **ZEALAND** → **NEW ZEALAND** (1810) sim=0.332 [new_headword] (2 eds: 1810, 1823)
  - `... Scheldt, on the south; and by the German ocean on the west.`
  - `NEW ZEALAND, a country of Asia, in the South Pacific ocean, first discovered by Tasman, the Dutch na`
  - Fix: `(1810, 'ZEALAND', 'eb_4th_1810_v20_SUI-PRE', [('NEW ZEALAND', r'NEW\s+ZEALAND,\s+a', 28)])`

- 🔴 **AMES** → **WILLIAM** (1842) sim=0.332 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ican should contemplate with gratitude and patriotic pride."`
  - `WILLIAM, D. D. a learned independent divine, celebrated for his controversial writings, was born in `
  - Fix: `(1842, 'AMES', 'eb_7th_1842_v02_AAL-DES', [('WILLIAM', r'WILLIAM,\s+D\.\s+D\.', 72)])`

- 🔴 **CARPENTRY** → **III** (1842) sim=0.332 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...}{8} \) is a 
very good mean for representing these results.`
  - `III.—ELEMENTS OF CARPENTRY.

Definition. "Carpentry is the art of framing timber for the pur-
poses `
  - Fix: `(1842, 'CARPENTRY', 'eb_7th_1842_v06_SEV-CLO', [('III', r'III\.—ELEMENTS\s+OF\s+CARPENTRY\.', 30)])`

- 🔴 **CHESS** → **METHODS OF GIVING CHECK-MATE** (1842) sim=0.332 [new_headword] (2 eds: 1842, 1860)
  - `...ub, and the second and fifth were won by the Edinburgh Club.`
  - `METHODS OF GIVING CHECK-MATE.

1. With a Rook and King against a King.`
  - Fix: `(1842, 'CHESS', 'eb_7th_1842_v06_SEV-CLO', [('METHODS OF GIVING CHECK-MATE', r'METHODS\s+OF\s+GIVING', 32)])`

- 🔴 **CARPENTRY** → **III** (1860) sim=0.332 [new_headword] (3 eds: 1823, 1842, 1860)
  - `...}{8} \) is a 
very good mean for representing these results.`
  - `III.—ELEMENTS OF CARPENTRY.

Definition. "Carpentry is the art of framing timber for the pur-
poses `
  - Fix: `(1860, 'CARPENTRY', 'eb_8th_1860_v06_ADA-CLI', [('III', r'III\.—ELEMENTS\s+OF\s+CARPENTRY\.', 18)])`

- 🔴 **MECHANICS** → **PROP** (1815) sim=0.334 [new_headword] (2 eds: 1815, 1823)
  - `...\frac{A \times Aa + B \times Bb + C \times Cc}{A + B + C}
\]`
  - `PROP. IV.

163. To find the centre of inertia of a straight line, composed of material particles.`
  - Fix: `(1815, 'MECHANICS', 'eb_5th_1815_v13_MAT-CCC', [('PROP', r'PROP\.\s+IV\.\s+163\.', 63)])`

- 🔴 **MEDICINE** → **GENUS LXXVI** (1815) sim=0.334 [new_headword] [gap: EDITORIAL]
  - `...but particularly diuretics, are often employed with success.`
  - `GENUS LXXVI. HYDROCEPHALUS.

WATER in the HEAD.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXXVI', r'GENUS\s+LXXVI\.\s+HYDROCEPHALUS\.', 84)])`

- 🔴 **CHESS** → **METHODS OF GIVING CHECK-MATE** (1860) sim=0.334 [new_headword] (2 eds: 1842, 1860)
  - `...ub, and the second and fifth were won by the Edinburgh Club.`
  - `METHODS OF GIVING CHECK-MATE.

1. With a Rook and King against a King.`
  - Fix: `(1860, 'CHESS', 'eb_8th_1860_v06_ADA-CLI', [('METHODS OF GIVING CHECK-MATE', r'METHODS\s+OF\s+GIVING', 65)])`

- 🔴 **GENUS LXIV** → **ORDER IV** (1823) sim=0.335 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ly any effect. This patient also dreaded the sight of a dog.`
  - `ORDER IV. VESANIAE.

Paranoiae, Vog. Class IX.
Deliria, Sauv. Class VIII. Ord. III. Sag. Class XI.
O`
  - Fix: `(1823, 'GENUS LXIV', 'eb_6th_1823_v13_ENL-MIC', [('ORDER IV', r'ORDER\s+IV\.\s+VESANIAE\.', 25)])`

- 🔴 **MEDICINE** → **GENUS LXVIII** (1815) sim=0.336 [new_headword] [gap: EDITORIAL]
  - `...engthen their whole frame and secure them against a relapse.`
  - `GENUS LXVIII. ONEIRODYNYA.

UNEASINESS in SLEEP.`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LXVIII', r'GENUS\s+LXVIII\.\s+ONEIRODYNYA\.', 82)])`

- 🔴 **BRAZIL** → **III** (1860) sim=0.336 [new_headword] (2 eds: 1842, 1860)
  - `...rive in Brazil at all so well as the larger kinds of cattle.`
  - `III. Statistics.—The population of Brazil has been variously estimated at different times, and indee`
  - Fix: `(1860, 'BRAZIL', 'eb_8th_1860_v05_ADA-BUR', [('III', r'III\.\s+Statistics\.—The\s+population', 23)])`

- 🔴 **MINERALOGY** → **III** (1778) sim=0.337 [new_headword] (2 eds: 1778, 1815)
  - `... like a star on the septaria, thence called stella septarii.`
  - `III. Calcareous earth satiated with the acid of common salt. Sal ammoniacum fixum naturale.`
  - Fix: `(1778, 'MINERALOGY', 'eb_2nd_1778_v07_BYW-OPT', [('III', r'III\.\s+Calcareous\s+earth', 64)])`

- 🔴 **BRASS** → **ORDER III** (1815) sim=0.337 [new_headword] (2 eds: 1810, 1815)
  - `... only officinal preparation of them, to be made by infusion.`
  - `ORDER III. TRIGYNYIA.

894. Cucubalus, or Berry-bearing Chickweed.`
  - Fix: `(1815, 'BRASS', 'eb_5th_1815_v04_ENL-BUR', [('ORDER III', r'ORDER\s+III\.\s+TRIGYNYIA\.', 881)])`

- 🔴 **MEDICINE** → **GENUS LVIII** (1815) sim=0.337 [new_headword] [gap: EDITORIAL]
  - `... too uncertain and too dangerous to be employed in practice.`
  - `GENUS LVIII. PYROSIS.
The Heart-Burn.

Pyrosis, Sauv. gen. 200. Sag. 158.
Soda, Lin. 47. Vog. 154.
S`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('GENUS LVIII', r'GENUS\s+LVIII\.\s+PYROSIS\.', 76)])`

- 🔴 **BREWING** → **ORDER III** (1823) sim=0.338 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... only officinal preparation of them, to be made by infusion.`
  - `ORDER III. TRIGYNIA.

894. CUCUBALUS, or Berry-bearing Chickweed.`
  - Fix: `(1823, 'BREWING', 'eb_6th_1823_v04_ENL-BUR', [('ORDER III', r'ORDER\s+III\.\s+TRIGYNIA\.', 839)])`

- 🔴 **ACADEMY** → **VIII** (1860) sim=0.338 [new_headword] (3 eds: 1778, 1842, 1860)
  - `... of the Siciences at Bologna. We are not aware of any other.`
  - `VIII. Academies of History. The first of these to which we shall advert, is the Royal Academy of Por`
  - Fix: `(1860, 'ACADEMY', 'eb_8th_1860_v02_ADA-GEN', [('VIII', r'VIII\.\s+Academies\s+of', 70)])`

- 🔴 **WM FARQUHARSON** → **MEDICINE** (1810) sim=0.341 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...such as gargarisms or blisters for affections of the throat.`
  - `MEDICINE.

this city about the beginning of the year 1801, and appears to have made inconsiderable p`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('MEDICINE', r'MEDICINE\.\s+this\s+city', 0)])`

- 🔴 **DICTIONARY** → **IMMEDIATELY** (1810) sim=0.341 [new_headword] (2 eds: 1810, 1815)
  - `..., a dictionary of the English language ought to be compiled.`
  - `IMMEDIATELY. adv. of time.

1. Instantly, without delay. Always employed to denote future time, and `
  - Fix: `(1810, 'DICTIONARY', 'eb_4th_1810_v17_CRY-DYE', [('IMMEDIATELY', r'IMMEDIATELY\.\s+adv\.\s+of', 0)])`

- 🔴 **THEOGNIS** → **NUMEN** (1815) sim=0.342 [new_headword] (2 eds: 1810, 1815)
  - `...irms of Cumberland, that "he excels all men in fixing on the`
  - `NUMEN, ET VIM DEORUM; deinde aliquo tempore, patefactis terrae faucibus, ex illis abditis fedibus ev`
  - Fix: `(1815, 'THEOGNIS', 'eb_5th_1815_v20_SUI-DIR', [('NUMEN', r'NUMEN,\s+ET\s+VIM', 24)])`

- 🔴 **DICTIONARY** → **IMMEDIATELY** (1815) sim=0.343 [new_headword] (2 eds: 1810, 1815)
  - `..., a dictionary of the English language ought to be compiled.`
  - `IMMEDIATELY, adv. of time.
1. Instantly, without delay. Always employed to denote future time, and n`
  - Fix: `(1815, 'DICTIONARY', 'eb_5th_1815_v15_NIC-CCC', [('IMMEDIATELY', r'IMMEDIATELY,\s+adv\.\s+of', 0)])`

- 🔴 **DUNBAR** → **WILLIAM** (1842) sim=0.343 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ng the country part of the parish, amounted in 1831 to 4735.`
  - `WILLIAM**, the most eminent of all the early Scottish poets, appears to have been born about the mid`
  - Fix: `(1842, 'DUNBAR', 'eb_7th_1842_v08_DIA-VII', [('WILLIAM', r'WILLIAM\*\*,\s+the\s+most', 0)])`

- 🔴 **DICKINSON** → **TALL** (1778) sim=0.344 [new_headword] (2 eds: 1778, 1797)
  - `...ed.
tended with peculiar degrees of guilt; as, high treason.`
  - `TALL. adj. Something elevated to a considerable degree in a perpendicular direction. Opposed to low.`
  - Fix: `(1778, 'DICKINSON', 'eb_2nd_1778_v04_BYW-FUZ', [('TALL', r'TALL\.\s+adj\.\s+Something', 40)])`

- 🔴 **MONEY** → **XIII** (1810) sim=0.344 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...nd with muriatic acid he obtained from it a sympathetic ink.`
  - `XIII. NICKEL Genus.

1. Species. Copper-coloured Nickel.`
  - Fix: `(1810, 'MONEY', 'eb_4th_1810_v17_MIC-MOR', [('XIII', r'XIII\.\s+NICKEL\s+Genus\.', 254)])`

- 🔴 **GENUS XCVIII** → **ORDER II** (1810) sim=0.345 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `... and also warm bathing, especially in the natural hot baths.`
  - `ORDER II. DYSOREXIAE.

SECT. I. APPETITUS ERRONEI.`
  - Fix: `(1810, 'GENUS XCVIII', 'eb_4th_1810_v13_GEN-MIC', [('ORDER II', r'ORDER\s+II\.\s+DYSOREXIAE\.', 60)])`

- 🔴 **VANDYCK** → **ANNES** (1842) sim=0.345 [new_headword] [gap: OCR_GAP]
  - `... the thimble, upon which it turns about as the wind changes.`
  - `ANNES, an arrondissement in the department of Morbihan in France, extending over 638 square miles, c`
  - Fix: `(1842, 'VANDYCK', 'eb_7th_1842_v21_SEV-ZYG', [('ANNES', r'ANNES,\s+an\s+arrondissement', 33)])`

- 🔴 **ZODIAC** → **ORDER IV** (1797) sim=0.346 [new_headword] [gap: PARSING_OR_EDITORIAL]
  - `...ita, Haliotis, Patella, Dentalium, Serpula, Teredo, Sabella.`
  - `ORDER IV. The Zoophyta, are compound animals,

furnished with a kind of flowers, and having a vege-`
  - Fix: `(1797, 'ZODIAC', 'eb_3rd_1797_v18_IND-ER', [('ORDER IV', r'ORDER\s+IV\.\s+The', 82)])`

- 🔴 **GRAMMAR** → **ALONE** (1823) sim=0.346 [new_headword] (2 eds: 1815, 1823)
  - `... anon withouten more abode."

"Anon in all the haste I can."`
  - `ALONE and ONLY are resolved into ALL ONE, and ONE-LIKE. In the Dutch, EEN is one; and ALL EEN alone;`
  - Fix: `(1823, 'GRAMMAR', 'eb_6th_1823_v10_ENL-HYD', [('ALONE', r'ALONE\s+and\s+ONLY', 65)])`

- 🔴 **LOGIC** → **VII** (1815) sim=0.347 [new_headword] (2 eds: 1778, 1815)
  - `...s creatures:
"Therefore he is a Being of infinite goodness."`
  - `VII. These two species take in the whole class of conditional syllogisms, and include all the possib`
  - Fix: `(1815, 'LOGIC', 'eb_5th_1815_v12_LIE-CCX', [('VII', r'VII\.\s+These\s+two', 50)])`

- 🔴 **MONEY** → **XIII** (1815) sim=0.347 [new_headword] (3 eds: 1810, 1815, 1823)
  - `...nd with muriatic acid he obtained from it a sympathetic ink.`
  - `XIII. NICKEL Genus:

1. Species. Copper-coloured Nickel.`
  - Fix: `(1815, 'MONEY', 'eb_5th_1815_v14_ENL-NIC', [('XIII', r'XIII\.\s+NICKEL\s+Genus:', 212)])`

- 🔴 **MINES** → **STATE OF THE GLOBE DURING THE FORMATION OF THE NEW RED SANDSTONES** (1842) sim=0.347 [new_headword] [gap: VARIANT]
  - `...Gloucestershire were completed before the date of that rock.`
  - `STATE OF THE GLOBE DURING THE FORMATION OF THE NEW RED SANDSTONES.`
  - Fix: `(1842, 'MINES', 'eb_7th_1842_v15_SEV-NAV', [('STATE OF THE GLOBE DURING THE FORMATION OF THE NEW RED SANDSTONES', r'STATE\s+OF\s+THE', 421)])`

- 🔴 **WM FARQUHARSON** → **GENUS XXXIV** (1810) sim=0.348 [new_headword] [gap: VARIANT]
  - `...cally cured, by his having no further symptoms of a relapse.`
  - `GENUS XXXIV. PEMPHIGUS.

Pemphigus, Sauv. gen. 93. Sag. 291.
Morta, Lin. i.
Febris bullola, Vog. 41.`
  - Fix: `(1810, 'WM FARQUHARSON', 'eb_4th_1810_v13_MAT-GEN', [('GENUS XXXIV', r'GENUS\s+XXXIV\.\s+PEMPHIGUS\.', 12)])`

- 🔴 **MEDICINE** → **ORDER IV** (1815) sim=0.348 [new_headword] (2 eds: 1778, 1815)
  - `...ly any effect. This patient also dreaded the sight of a dog.`
  - `ORDER IV. VESANIAE.

Paranoiae, Vog. Clas. IX.
Deliria, Sauv. Clas. VIII. Ord. III. Sag. Clas. XI.
O`
  - Fix: `(1815, 'MEDICINE', 'eb_5th_1815_v13_MAT-CCC', [('ORDER IV', r'ORDER\s+IV\.\s+VESANIAE\.', 81)])`

- 🔴 **ANGOLA** → **EXPLANATION OF PLATE XXXII** (1810) sim=0.349 [new_headword] (2 eds: 1810, 1815)
  - `...be found useful to a person whose eye is naturally too flat.`
  - `EXPLANATION OF PLATE XXXII.

Fig. 1. Shows the Lachrymal Canals, after the Common Teguments and Bone`
  - Fix: `(1810, 'ANGOLA', 'eb_4th_1810_v17_ART-ANS', [('EXPLANATION OF PLATE XXXII', r'EXPLANATION\s+OF\s+PLATE', 563)])`

- 🔴 **ANGOLA** → **EXPLANATION OF PLATE XXXII** (1815) sim=0.349 [new_headword] (2 eds: 1810, 1815)
  - `...be found useful to a person whose eye is naturally too flat.`
  - `EXPLANATION OF PLATE XXXII.

Fig. 1. Shows the Lachrymal Canals, after the Common Teguments and Bone`
  - Fix: `(1815, 'ANGOLA', 'eb_5th_1815_v02_ENL-ASS', [('EXPLANATION OF PLATE XXXII', r'EXPLANATION\s+OF\s+PLATE', 695)])`

- 🔴 **GRAMMAR** → **ALONE** (1815) sim=0.349 [new_headword] (2 eds: 1815, 1823)
  - `...t anon withouten more abode."
"Anon in all the halfe I can."`
  - `ALONE and ONLY are resolved into ALL ONE, and ONE-LIKE. In the Dutch, EEN is one; and ALL EEN alone;`
  - Fix: `(1815, 'GRAMMAR', 'eb_5th_1815_v10_GOT-HYD', [('ALONE', r'ALONE\s+and\s+ONLY', 81)])`
