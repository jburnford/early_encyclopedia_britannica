# Audit Report: 1778 (2nd Edition) Encyclopedia Britannica

## Executive Summary

- **Total Articles Analyzed**: 17,219
- **Total Characters**: 63,779,362
- **Average Article Length**: 3,704 characters
- **Date Generated**: 2026-01-03

### Data Source Overview

The 1778 2nd Edition data is organized in 11 JSON files:
- **vol0.json**: Supplementary content (2,623 articles) - not listed in main index
- **vol1.json - vol10.json**: Main encyclopedia volumes (14,596 articles)

**Note**: vol0 contains unique headwords that do not overlap with volumes 1-10,
suggesting it may be appendix material, index entries, or alternative extractions.

### Issue Overview

| Issue Type | Count | Severity |
|------------|-------|----------|
| Articles Outside Alphabetical Range | 163 | HIGH |
| Unusually Short Articles (<50 chars) | 18 | LOW |
| Unusually Long Articles (>50K chars) | 245 | HIGH |
| Duplicate Headwords | 0 | LOW |
| Alphabetical Order Issues | 21 | LOW |
| OCR/Parsing Errors | 25 | LOW |

---

## Volume Statistics

| Volume | Articles | Min Length | Max Length | Avg Length |
|--------|----------|------------|------------|------------|
| vol0 | 2,623 | 50 | 100,088 | 7,777 |
| vol1 | 2,026 | 45 | 276,430 | 2,129 |
| vol10 | 1,977 | 42 | 334,679 | 3,223 |
| vol2 | 1,736 | 49 | 198,514 | 2,053 |
| vol3 | 2,031 | 44 | 221,315 | 2,094 |
| vol4 | 1,526 | 15 | 355,325 | 2,966 |
| vol5 | 1,483 | 15 | 192,937 | 2,972 |
| vol6 | 981 | 13 | 606,579 | 4,508 |
| vol7 | 921 | 13 | 413,093 | 3,974 |
| vol8 | 803 | 42 | 384,556 | 4,414 |
| vol9 | 1,112 | 48 | 608,622 | 3,882 |

---

## 1. Articles Outside Alphabetical Range

**Severity: HIGH**

**Count: 163**

Articles whose headwords don't match the expected letter range for their volume.

### vol1 (Expected: ('A', 'AST'))
**12 issues found**

- `vol1#2014`: **CHAP**
- `vol1#2015`: **END OF THE FIRST VOLUME**
- `vol1#2016`: **ENIGMATOGRAPHY**
- `vol1#2017`: **EXPLANATION OF PLATE XVIII**
- `vol1#2018`: **EXPLANATION OF PLATE XXI**
- `vol1#2019`: **EXPLANATION OF THE PLATES OF OSTEOLGY**
- `vol1#2020`: **HERESY OF ALMARIC**
- `vol1#2021`: **NEW ANDALUSIA**
- `vol1#2022`: **PLATE XV**
- `vol1#2023`: **PLATE XVI**
- ... and 2 more

### vol2 (Expected: ('AST', 'BZZ'))
**13 issues found**

- `vol2#1723`: **DIANDRIA MONOGYNYIA**
- `vol2#1724`: **DIDYNAMIA ANGIOSPERMIA**
- `vol2#1725`: **END OF THE SECOND VOLUME**
- `vol2#1726`: **FREE BENCH**
- `vol2#1727`: **GYNANDRIA PENTANDRIA**
- `vol2#1728`: **HEAD-BOROUGH ALSO**
- `vol2#1729`: **ICOSANDRIA POLYGAMIA**
- `vol2#1730`: **MONODELPHIA POLYANDRIA**
- `vol2#1731`: **MONOECIA TETRANDRIA**
- `vol2#1732`: **SOURING**
- ... and 3 more

### vol3 (Expected: ('C', 'CZZ'))
**16 issues found**

- `vol3#0`: **ARTIFICIAL CORUSCATIONS MAY ALSO BE PRODUCED BY**
- `vol3#1`: **BLACK**
- `vol3#2017`: **END OF THE THIRD VOLUME**
- `vol3#2018`: **ISO'ER**
- `vol3#2019`: **MONEY-CHANGER**
- `vol3#2020`: **NEW CALEDONIA**
- `vol3#2021`: **PASTILS**
- `vol3#2022`: **PRECEPT OF CLARE CONSTAT**
- `vol3#2023`: **QUEEN CHARLOTTE'S ISLAND**
- `vol3#2024`: **QUEEN CHARLOTTE'S ISLANDS**
- ... and 6 more

### vol4 (Expected: ('D', 'FZZ'))
**13 issues found**

- `vol4#0`: **APHORISMS**
- `vol4#1`: **AREWELL-CAPE**
- `vol4#2`: **CASE II**
- `vol4#1516`: **HAROLD IN THE MEAN TIME INCREASED HIS POPULARITY BY ALL POSSIBLE**
- `vol4#1517`: **ISLANDS OF DISAPPOINTMENT**
- `vol4#1518`: **MOE**
- `vol4#1519`: **MONEY-TABLE**
- `vol4#1520`: **PILE**
- `vol4#1521`: **ST DAVID'S**
- `vol4#1522`: **THIS ACCIDENT PROVED THE**
- ... and 3 more

### vol5 (Expected: ('G', 'JZZ'))
**34 issues found**

- `vol5#0`: **AMONG PAINTERS IT**
- `vol5#1`: **COLOUR-MAKING**
- `vol5#2`: **COROLLARY**
- `vol5#3`: **DCA**
- `vol5#4`: **END OF THE FIFTH VOLUME**
- `vol5#1454`: **KF**
- `vol5#1455`: **MORAL GOOD**
- `vol5#1456`: **MOST OF THE ABOVE PROBLEMS MAY ALSO BE PERFORMED BY**
- `vol5#1457`: **PROPOSITION II**
- `vol5#1458`: **PROPOSITION III**
- ... and 24 more

### vol6 (Expected: ('K', 'MED'))
**23 issues found**

- `vol6#0`: **ADYNAMIE**
- `vol6#1`: **AFTER THE PATIENT HAS BY THIS**
- `vol6#2`: **CLIV**
- `vol6#3`: **CXIV**
- `vol6#4`: **END OF THE SIXTH VOLUME**
- `vol6#5`: **EXERCISE AND ABSTINENCE ARE THE**
- `vol6#6`: **GENERAL OBSERVATIONS**
- `vol6#7`: **GENUS LXVII**
- `vol6#8`: **HITHERTO MAHOMET HAD PROPAGATED HIS RELIGION BY FAIR**
- `vol6#9`: **INDEX

ABORTION**
- ... and 13 more

### vol7 (Expected: ('MED', 'OPT'))
**31 issues found**

- `vol7#0`: **BD**
- `vol7#1`: **BECAUSE THE EQUABLE DESCRIPTION OF AREAS**
- `vol7#2`: **CLASS II**
- `vol7#3`: **CLASS III**
- `vol7#4`: **CLASS IV**
- `vol7#5`: **DEFINITIONS OF SEVERAL TECHNICAL TERMS**
- `vol7#6`: **FINNED-FOOTED WATER-BIRDS**
- `vol7#7`: **FIRST ORDER**
- `vol7#8`: **FOURTH CLASS**
- `vol7#9`: **GENERAL REMARK**
- ... and 21 more

### vol8 (Expected: ('OPT', 'POE'))
**13 issues found**

- `vol8#0`: **APIAS**
- `vol8#1`: **FEWER ERRORS HAVE BEEN COMMITTED IN THE**
- `vol8#2`: **GEMMA**
- `vol8#3`: **GRE-HOUND**
- `vol8#4`: **HYDRAULIC ORGAN**
- `vol8#5`: **INDEX

ACACIA**
- `vol8#6`: **MARCVS**
- `vol8#7`: **MORTALS**
- `vol8#798`: **RESTITVRENT**
- `vol8#799`: **SIGNOR FIDO**
- ... and 3 more

### vol9 (Expected: ('POI', 'SCU'))
**8 issues found**

- `vol9#0`: **ELIZABETH HAVING THUS FOUND**
- `vol9#1`: **END OF THE NINTH VOLUME**
- `vol9#2`: **GALBA HAVING BEEN BROUGHT TO THE EMPIRE BY**
- `vol9#3`: **HEPBURN IS SAID ALSO TO HAVE GAINED AN ASCENDENCY OVER THE REGENT BY**
- `vol9#4`: **NUMBER**
- `vol9#1109`: **THESE RESOLUTIONS OF THE DIET WERE BY NO**
- `vol9#1110`: **THIS PUMP IS MANAGED BY**
- `vol9#1111`: **TOGA PRETEXTA**


---

## 2. Unusually Short Articles (<50 characters)

**Severity: LOW**

**Count: 18**

These may indicate parsing errors or incomplete OCR extraction.

### Examples (sorted by length)

- `vol6#182`: **LANDSCAPE** (13 chars)
  - Preview: "See LANDSKIP...."
- `vol7#833`: **OLD-WIFE FISH** (13 chars)
  - Preview: "See BALISTES...."
- `vol4#768`: **EMBASSADOR** (15 chars)
  - Preview: "See Ambassador...."
- `vol5#499`: **HAIMSUCKEN** (15 chars)
  - Preview: "see HAMBSECKEN...."
- `vol10#983`: **SUSPENSION** (42 chars)
  - Preview: "in Scots law. See Law, No clxxxv. 5, 6, 7...."
- `vol8#138`: **PACKAGE** (42 chars)
  - Preview: "is a small duty of one penny in the pound,..."
- `vol3#1284`: **COGITATION** (44 chars)
  - Preview: "a term used by some for the act of thinking...."
- `vol1#255`: **ACGIAH-SARAI** (45 chars)
  - Preview: "a town on the north shore of the Caspian sea...."
- `vol1#1927`: **ASCALON** (45 chars)
  - Preview: "an ancient city, and one of the five  Vol. I...."
- `vol4#1128`: **FARDING-DEAL** (45 chars)
  - Preview: "the fourth part of an acre of land. See Acre...."
- `vol8#687`: **PINNATED LEAVES** (46 chars)
  - Preview: "in botany. See BOTANY, p. 1296, col. 2, n° 59...."
- `vol1#464`: **ADVERSARY** (47 chars)
  - Preview: "a person who is an enemy to or opposes another...."
- `vol1#1833`: **ARMOURER** (47 chars)
  - Preview: "a person who makes or deals in arms and armour...."
- `vol3#1815`: **CRIBBAGE** (47 chars)
  - Preview: "a game at cards, to be learnt only by practice...."
- `vol9#155`: **PRECOGNITION** (48 chars)
  - Preview: "in Scots law. See Law, Part III. No clxxxvi. 43...."
- `vol9#304`: **PRUNES** (48 chars)
  - Preview: "are plumbs dried in the sunshine, or in an oven...."
- `vol2#1515`: **BRONCHOCELE** (49 chars)
  - Preview: "a tumour rising in the anterior part of the neck...."
- `vol5#1251`: **INTESTATE** (49 chars)
  - Preview: "in law, a person that dies without making a will...."

---

## 3. Unusually Long Articles (>50,000 characters)

**Severity: HIGH**

**Count: 245**

Long articles fall into two categories:

1. **Legitimate Treatises**: Major encyclopedic entries like SCOTLAND, CHEMISTRY, AGRICULTURE
2. **Potential Merge Errors**: Articles whose headwords are sentence fragments

### Potential Merge Errors (18 articles)
These headwords appear to be sentence fragments, suggesting parsing issues:

- `vol9#3`: **HEPBURN IS SAID ALSO TO HAVE GAINED AN ASCENDENCY OVER THE REGENT BY** (608,622 chars)
- `vol4#1516`: **HAROLD IN THE MEAN TIME INCREASED HIS POPULARITY BY ALL POSSIBLE** (355,325 chars)
- `vol10#1217`: **THERE IS STILL ANOTHER** (300,850 chars)
- `vol9#2`: **GALBA HAVING BEEN BROUGHT TO THE EMPIRE BY** (288,622 chars)
- `vol5#1481`: **WHILE THE ROMANS THUS EMPLOYED ALL** (192,937 chars)
- `vol1#1003`: **AMERICA IS BY NO** (130,456 chars)
- `vol7#916`: **THIS OPERATION OF ADJUSTING THE METALS TO THE MONEY OF ACCOUNT** (117,595 chars)
- `vol10#1826`: **WHEN THE DOCTOR WANTED TO EXTRACT INFLAMMABLE AIR FROM METALS BY** (110,137 chars)
- `vol6#1`: **AFTER THE PATIENT HAS BY THIS** (104,863 chars)
- `vol10#1830`: **WHILE THE STATES-GENERAL WERE EMPLOYED IN WAYS AND** (98,289 chars)
- `vol9#0`: **ELIZABETH HAVING THUS FOUND** (81,739 chars)
- `vol4#422`: **DISEASES OF THE FEET** (75,969 chars)
- `vol10#1825`: **WHEN THE** (71,774 chars)
- `vol1#2019`: **EXPLANATION OF THE PLATES OF OSTEOLGY** (62,930 chars)
- `vol10#1232`: **THIS IS INDEED THE ONLY RATIONAL END WHICH CAN BY ANY OF THESE** (61,066 chars)
- `vol6#5`: **EXERCISE AND ABSTINENCE ARE THE** (57,031 chars)
- `vol8#802`: **THIS IS BY NO** (53,046 chars)
- `vol4#1525`: **WHEN THE WEEPING IS BY THESE** (51,491 chars)

### Likely Legitimate Treatises (227 articles)
These are expected long entries covering major topics:

- `vol6#6`: **GENERAL OBSERVATIONS** (606,579 chars)
- `vol9#1091`: **SCOTLAND** (540,800 chars)
- `vol7#10`: **HISTORY** (413,093 chars)
- `vol8#8`: **OPTICS** (384,556 chars)
- `vol8#796`: **POETRY** (368,938 chars)
- `vol10#1765`: **WAR** (334,679 chars)
- `vol1#561`: **AGRICULTURE** (276,430 chars)
- `vol6#252`: **LAW** (236,111 chars)
- `vol3#904`: **CHEMISE** (221,315 chars)
- `vol1#1644`: **ARABIA** (203,302 chars)
- `vol2#66`: **ATTICA** (198,514 chars)
- `vol10#994`: **SWEDEN** (194,650 chars)
- `vol4#661`: **EARL** (192,581 chars)
- `vol10#691`: **SPAIN** (185,895 chars)
- `vol6#957`: **MECHANICS** (184,557 chars)
- `vol3#501`: **CARTHAGE** (183,639 chars)
- `vol4#1402`: **FRAISE** (177,015 chars)
- `vol7#128`: **METAPHYSICS** (174,699 chars)
- `vol6#195`: **LANGUAGE** (174,693 chars)
- `vol4#714`: **EGYPT** (173,038 chars)
- `vol8#318`: **PARTICULAR ELOCUTION** (172,001 chars)
- `vol3#1540`: **CONSTABLE** (157,480 chars)
- `vol5#1315`: **ITALY** (156,737 chars)
- `vol2#1731`: **MONOECIA TETRANDRIA** (144,252 chars)
- `vol6#655`: **MACEDON** (137,919 chars)
- `vol1#2023`: **PLATE XVI** (135,060 chars)
- `vol5#1271`: **IRELAND** (134,844 chars)
- `vol7#539`: **NAVIGATION** (132,423 chars)
- `vol5#982`: **HYDROSCOPE** (131,494 chars)
- `vol7#149`: **MEXICO** (125,477 chars)
- `vol1#1533`: **APIS** (124,139 chars)
- `vol1#1708`: **ARCHITECTURE** (119,436 chars)
- `vol2#1160`: **BOOK** (118,880 chars)
- `vol8#532`: **PERSIA** (118,053 chars)
- `vol9#9`: **POLAND** (117,571 chars)
- `vol6#919`: **MATERIA MEDICA** (111,721 chars)
- `vol9#64`: **PONTUS** (110,534 chars)
- `vol8#587`: **PHARISEES** (105,606 chars)
- `vol6#526`: **LONDON** (104,408 chars)
- `vol5#1387`: **JEWS** (101,632 chars)
- `vol0#1477`: **HISPANIA** (100,088 chars)
- `vol0#1717`: **LIXUS** (100,085 chars)
- `vol0#2356`: **SATURN** (100,085 chars)
- `vol0#1940`: **NOTH** (100,082 chars)
- `vol0#143`: **AFRICA** (100,080 chars)
- `vol0#1880`: **MOUNTAINS** (100,070 chars)
- `vol0#1204`: **ERASISTRATUS** (100,051 chars)
- `vol0#1846`: **MINES** (100,050 chars)
- `vol0#58`: **ABSTRUCTION** (100,044 chars)
- `vol0#697`: **BRODY** (100,042 chars)

---

## 4. Duplicate Headwords

**Severity: LOW**

**Count: 0 unique headwords with duplicates**

Same headword appearing multiple times (may be intentional cross-references or errors).


---

## 5. Alphabetical Order Issues

**Severity: LOW**

**Count: 21**

Large gaps or backward jumps in alphabetical sequence.

### Large Alphabetical Gaps (21 issues)
Suspicious jumps that might indicate missing articles.

- `vol1#2020`: **EXPLANATION OF THE PLATES OF OSTEOLGY** -> **HERESY OF ALMARIC**
- `vol1#2021`: **HERESY OF ALMARIC** -> **NEW ANDALUSIA**
- `vol1#2024`: **PLATE XVI** -> **THOUGH WE CAN BY NO**
- `vol2#1730`: **ICOSANDRIA POLYGAMIA** -> **MONODELPHIA POLYANDRIA**
- `vol2#1732`: **MONOECIA TETRANDRIA** -> **SOURING**
- `vol3#2018`: **END OF THE THIRD VOLUME** -> **ISO'ER**
- `vol3#2019`: **ISO'ER** -> **MONEY-CHANGER**
- `vol3#2025`: **QUEEN CHARLOTTE'S ISLANDS** -> **TABLES OF COMBINATIONS**
- `vol3#2029`: **THOMSON'S WINTER** -> **WHEN THE COLOUR OF GOLD IS BY ANY**
- `vol4#1518`: **ISLANDS OF DISAPPOINTMENT** -> **MOE**
- `vol4#1520`: **MONEY-TABLE** -> **PILE**
- `vol4#1521`: **PILE** -> **ST DAVID'S**
- `vol5#1457`: **MOST OF THE ABOVE PROBLEMS MAY ALSO BE PERFORMED BY** -> **PROPOSITION II**
- `vol5#1478`: **PROPOSITION XXX** -> **SCHLUTER RECOMMENDS MECHANICAL**
- `vol5#1479`: **SCHLUTER RECOMMENDS MECHANICAL** -> **WATER MAY ALSO BE RAISED BY**
- `vol6#968`: **MEDICAGO** -> **PUSTULES ARE SELDOM PERFECTLY CURED BY**
- `vol6#969`: **PUSTULES ARE SELDOM PERFECTLY CURED BY** -> **SAUVAGES**
- `vol7#12`: **HISTORY OF MUSIC** -> **MEDINA**
- `vol8#1`: **APIAS** -> **FEWER ERRORS HAVE BEEN COMMITTED IN THE**
- `vol8#6`: **INDEX

ACACIA** -> **MARCVS**
- ... and 1 more


---

## 6. OCR/Parsing Errors in Headwords

**Severity: LOW**

**Count: 25**

Headwords with suspicious characters, formatting, or structure.

### Excessively long headword (possible sentence) (8 instances)

- `vol10#1216`: **THERE ARE TWO PRINCIPAL REASONS WHY THE SEA DOTH NOT INCREASE BY**
- `vol10#1232`: **THIS IS INDEED THE ONLY RATIONAL END WHICH CAN BY ANY OF THESE**
- `vol10#1826`: **WHEN THE DOCTOR WANTED TO EXTRACT INFLAMMABLE AIR FROM METALS BY**
- `vol4#1516`: **HAROLD IN THE MEAN TIME INCREASED HIS POPULARITY BY ALL POSSIBLE**
- `vol6#977`: **UPON THIS PRINCIPLE THE PROPORTION OF THE POWER TO THE WEIGHT IT SUSTAINS BY**
- `vol7#916`: **THIS OPERATION OF ADJUSTING THE METALS TO THE MONEY OF ACCOUNT**
- `vol7#917`: **TOBACCO IS MADE UP INTO ROLLS BY THE INHABITANTS OF THE INTERIOR PARTS OF AMERICA BY**
- `vol9#3`: **HEPBURN IS SAID ALSO TO HAVE GAINED AN ASCENDENCY OVER THE REGENT BY**

### Starts with 'THIS' (likely sentence fragment) (7 instances)

- `vol10#1232`: **THIS IS INDEED THE ONLY RATIONAL END WHICH CAN BY ANY OF THESE**
- `vol3#2027`: **THIS IS THE BEST**
- `vol4#1522`: **THIS ACCIDENT PROVED THE**
- `vol4#1523`: **THIS DISTEMPER IS TO BE CURED BY THESE**
- `vol7#916`: **THIS OPERATION OF ADJUSTING THE METALS TO THE MONEY OF ACCOUNT**
- `vol8#802`: **THIS IS BY NO**
- `vol9#1110`: **THIS PUMP IS MANAGED BY**

### Starts with 'WHEN' (likely sentence fragment) (6 instances)

- `vol10#1825`: **WHEN THE**
- `vol10#1826`: **WHEN THE DOCTOR WANTED TO EXTRACT INFLAMMABLE AIR FROM METALS BY**
- `vol3#2029`: **WHEN THE COLOUR OF GOLD IS BY ANY**
- `vol4#1525`: **WHEN THE WEEPING IS BY THESE**
- `vol6#978`: **WHEN THE PATIENT CAN BY NO OTHER**
- `vol7#920`: **WHEN THE DELIVERY COULD NOT BE ACCOMPLISHED BY OTHER**

### Roman numeral only (possible subsection header) (4 instances)

- `vol6#2`: **CLIV**
- `vol6#3`: **CXIV**
- `vol6#616`: **LXXXV**
- `vol6#980`: **XLIV**

### Starts with 'THESE' (likely sentence fragment) (3 instances)

- `vol10#1221`: **THESE ARE THE**
- `vol6#974`: **THESE WERE THE ONLY**
- `vol9#1109`: **THESE RESOLUTIONS OF THE DIET WERE BY NO**


---

## Recommendations

### Priority Actions

1. **HIGH**: Fix sentence fragment headwords (21 found)
   - These are likely parsing errors where article text was mistakenly captured as headword
   - Review parser logic for handling edge cases at article boundaries
2. **HIGH**: Review articles outside expected alphabetical ranges (163 found)
   - Many appear to be appendix content (PLATE explanations, END OF VOLUME markers)
   - Consider excluding these from main article index or creating separate appendix category
3. **HIGH**: Investigate 17 potential article merge errors
   - These very long articles have sentence fragments as headwords
   - Likely represent incorrectly merged content from adjacent articles
4. **MEDIUM**: Review OCR quality for headwords (25 issues)
   - Focus on excessively long headwords and sentence fragments
5. **LOW**: Review alphabetical ordering issues (21 found)
   - May indicate missing articles or incorrect sorting

### Structural Observations

1. **vol0.json** contains 2,623 unique articles not found in volumes 1-10
   - Purpose unclear: may be appendix, index, or alternative OCR extraction
   - Recommend investigating source and deciding on inclusion/exclusion

2. **Appendix content** scattered across volumes:
   - PLATE explanations, END OF VOLUME markers, INDEX sections
   - Consider separating front/back matter from main article content

3. **Short articles** are mostly valid cross-references:
   - Examples: 'See LANDSKIP', 'See BALISTES', 'See Ambassador'
   - These are intentional encyclopedia cross-references, not errors

### Data Quality Score

- **Quality Score**: 97.3%
- **Grade**: A
- **Total Issues Found**: 472
- **Issues per Article**: 0.027

**Note**: This score reflects parsing/structural issues only. OCR accuracy of article content
requires separate evaluation through text quality analysis.