# Ambiguous / Unmatched Headwords — Classification

Top ~95 unmatched headwords from cross_edition_index, classified by per-edition word count patterns.

**Key insight**: When a headword has <1K words in most editions but one giant outlier (10K+), it's a **swallowed article** — the parser failed to detect the next headword and concatenated subsequent articles into this one.

## SWALLOWED MEGA-ARTICLES — One edition has a massive outlier

These need parser fixes. The outlier edition swallowed subsequent articles.

| # | Headword | Total | Pattern | Swallowed edition(s) |
|---|----------|-------|---------|---------------------|
| 1 | ORDER | 268K | 1771:135, 1778:**62K**, 1797:**63K**, 1815:**64K**, 1823:**79K**, 1842:782 | 1778-1823 all ~60-79K — likely swallowed tail of ORATORY |
| 2 | ZYGOMATICUS | 146K | 1778:**146K**, 1797:23, 1810:21, 1815:21, 1823:21 | 1778 swallowed ~146K of subsequent content |
| 3 | MATERA | 138K | 1778:32, 1797:32, 1810:**46K**, 1815:**29K**, 1823:**63K**, 1860:124 | 1810-1823 swallowed (minor Italian town = 32 words) |
| 4 | GENUS IX | 123K | 1797:**104K**, 1823:**19K** | Medical subsection, not standalone headword |
| 5 | WEEK | 89K | 1771:223, 1778:59, 1810:**88K**, 1815:37, 1823:47 | 1810 swallowed ~88K |
| 6 | FLUX | 88K | 1771:35, 1778:5K, 1797:5K, 1810:927, 1815:924, 1823:925, 1842:134, 1860:**74K** | 1860 swallowed ~74K |
| 7 | STRAIN | 87K | 1771:13, 1797-1823:468, 1842:**85K** | 1842 swallowed ~85K |
| 8 | THUS | 83K | 1797:**62K**, 1810:**21K**, 1815:94, 1823:68 | 1797 mid-sentence fragment, swallowed ~62K |
| 9 | PHYSIC | 83K | 1778-1823:32 each, 1842:**83K** | 1842 swallowed ~83K (real entry is 32-word cross-ref) |
| 10 | PART | 82K | 1778:128, 1797:251, 1810:**81K**, 1815:278, 1823:253 | 1810 swallowed ~81K |
| 11 | PERSONIFYING | 79K | 1771:55, 1778:56, 1797:**20K**, 1810:**21K**, 1815:**22K**, 1842:**16K** | 1797-1842 consistently ~16-22K — may be legit article on rhetoric? Check text |
| 12 | GENUS | 79K | 1771:254, 1778:691, 1797:1.5K, 1810-1815:396, 1823:**75K**, 1842:325, 1860:140 | 1823 swallowed ~75K |
| 13 | ABYSS | 72K | 1771:61, 1778:789, 1797:952, 1810-1815:~940, 1823:**68K**, 1842:278, 1860:311 | 1823 swallowed ~68K |
| 14 | SCOT | 66K | 1771:29, 1778-1823:362 each, 1842:314, 1860:**63K** | 1860 swallowed ~63K |
| 15 | WHITE | 64K | 1771:26, 1778-1823:~35-191, 1842:**62K**, 1860:1.2K | 1842 swallowed ~62K |
| 16 | BOND | 47K | 1771:**44K**, 1778-1860:~600-655 | 1771 swallowed ~44K |
| 17 | VOCAL | 37K | 1771-1815:36, 1842:**37K** | 1842 swallowed ~37K |
| 18 | ROMANO | 44K | 1778-1815:247, 1823:**42K**, 1842:479, 1860:61 | 1823 swallowed ~42K (real entry is ~247 words on Giulio Romano) |
| 19 | NET | 37K | 1778:442, 1797:302, 1810-1823:~1.6K, 1842:**32K** | 1842 swallowed ~32K |
| 20 | SLAUGHTER | 42K | 1810:**14K**, 1815:**14K**, 1823:**14K** | All 3 editions ~14K — check if legit or swallowed |
| 21 | CENTER | 44K | 1778:106, 1797:160, 1810:**15K**, 1815:**15K**, 1823:**15K** | 1810-1823 consistently ~15K — check if legit geometry article |
| 22 | BURNING | 43K | 1771:793, 1778:4.8K, 1797:5.7K, 1810:**13K**, 1815:**13K**, 1823:5.7K | 1810-1815 doubled — possible swallowing |
| 23 | INDIAN | 52K | 1810:**52K**, 1842:14 | 1810 swallowed ~52K (mid-sentence fragment) |
| 24 | SWEDEN IS BY NO | 47K | 1815:**23K**, 1823:**24K** | Mid-sentence parsing break in both editions |
| 25 | POLITICAL | 41K | 1797:1.3K, 1810:**20K**, 1815:1.2K, 1823:**19K**, 1842:67 | 1810/1823 swallowed POLITICAL ARITHMETIC content |
| 26 | LOGARITHMS OF NUMBERS | 60K | 1810:**18K**, 1815:**19K**, 1823:**23K** | Numerical log tables, not articles |
| 27 | PERSHORE | 35K | 1842:**20K**, 1860:**15K** | Market town, ~200 words max — swallowed |
| 28 | ENGRAILED | 34K | 1778-1823:~40, 1842:**15K**, 1860:**19K** | 1842/1860 swallowed (real entry is 40 words on heraldry) |
| 29 | STONE-MASONRY | 32K | 1823:6K, 1860:**26K** | 1860 starts with math formulas — swallowed |
| 30 | LOGARITHMIC CURVE | 29K | 1771:714, 1778:697, 1797:**28K** | 1797 swallowed ~28K |
| 31 | HOUND | 35K | 1797-1823:~3.6K, 1842:7K, 1860:**13K** | Actually opens as "Hour, in chronology" — mislabeled headword |

## CONSISTENT ARTICLES — Legitimate across editions, need disambiguation

These have **consistent word counts** across multiple editions, suggesting a real article topic.

| # | Headword | Total | Per-edition pattern | What it likely is | Matchable? |
|---|----------|-------|--------------------|--------------------|-----------|
| 1 | EXCHANGE | 130K | 7-9K in 1771-1815, 28-35K in 1823-1860 | Commerce/bills of exchange | Maybe Q276454 (bill of exchange) |
| 2 | POLAR SEAS | 108K | 1823:95K, 1842:13K | Arctic exploration article | Maybe Q213390 |
| 3 | MENSURATION | 93K | 17-23K across 5 editions | Geometry/measurement | Q12453 (measurement) |
| 4 | VOLTAIC ELECTRICITY | 86K | 1842:36K, 1860:50K | Galvanism/electrochemistry | Matchable |
| 5 | NEWTONIAN PHILOSOPHY | 80K | 13-17K in 1778-1823, 1.6K in 1842 | Newtonian physics | Matchable |
| 6 | NATURAL HISTORY | 78K | 1.8-20K across 8 editions | Natural history discipline | Q7205 |
| 7 | COALERY | 74K | 5-18K across 5 editions | Coal mining | Q5384 (coal mining) |
| 8 | SPECIFICS | 73K | 96-21K across 6 editions | Medical specifics | Ambiguous |
| 9 | SHOOTING | 70K | 2-24K across 7 editions | Gunnery + sport | Ambiguous (military vs sport) |
| 10 | ROPES | 67K | 15-18K across 4 editions | Rope manufacturing | Q54341483 (rope) |
| 11 | PROPERTY | 65K | 65-14K across 8 editions | Legal/philosophical concept | Ambiguous |
| 12 | STEAM NAVIGATION | 65K | 1842:26K, 1860:39K | Steamship technology | Q39804 (steamship) |
| 13 | PASSION | 65K | 2.9-15.6K across 5 editions | Rhetoric/literary concept | Ambiguous |
| 14 | PRESCRIPTION | 64K | 325-33K, variable | Scots law concept | Inconsistent — check 1810/1815 for swallowing |
| 15 | BROWN | 63K | 2.8-12K across 7 editions | Robert Browne (Brownists founder) | Q1232972 |
| 16 | PLANTING | 62K | 1.2-29K, grows over time | Agriculture | Growing article, ambiguous |
| 17 | SECT | 58K | 18-18K, variable | Religious sects | Ambiguous |
| 18 | FUNDING SYSTEM | 57K | 17-22K across 3 editions | Sinking fund / national debt | Matchable |
| 19 | POLE | 55K | 10-11K in 1778-1823, ~1K in 1842/60 | Reginald Pole (cardinal) | Q313028 |
| 20 | DEAF AND DUMB | 54K | 18-19K across 3 editions | Deaf education | Matchable |
| 21 | POLITICAL ARITHMETIC | 54K | 1842:54K, 1860:210 | Demography/statistics (William Petty) | Q1640824 |
| 22 | ARTS | 54K | 17-19K across 3 editions | Fine arts | Q735 (art) |
| 23 | WEIGHTS AND MEASURES | 53K | 1.2-21K across 4 editions | Metrology | Q47574 |
| 24 | IGNATIUS | 53K | 1-37K, variable | St Ignatius of Antioch | Q44436 — but 1797 has 37K (check for swallowing) |
| 25 | UNITED PROVINCES | 52K | 5.9-19K across 5 editions | Dutch Republic | Q170072 |
| 26 | TRIGONOMETRICAL SURVEY | 50K | 1823:17K, 1842:17K | Ordnance Survey | Matchable |
| 27 | SAVAGE | 48K | 3.2-9K across 7 editions | Richard Savage (poet) | Q553138 |
| 28 | SPECTRE | 46K | 8.6-18K across 4 editions | Apparitions/ghosts | Ambiguous |
| 29 | ELEMENTS OF MUSIC | 45K | 1810:12K, 1823:33K | Music theory treatise | Q11401 (music theory) |
| 30 | HUNTER | 45K | 4.6-9K across 6 editions | Article on horses for hunting | Ambiguous |
| 31 | MIGRATION | 45K | 15-13K across 6 editions | General migration concept | Ambiguous |
| 32 | PHILIP | 44K | 10-11K in 1797-1823, <500 later | Biblical/historical Philip | Ambiguous (which Philip?) |
| 33 | READING | 44K | 26 in 1771, 8-8.5K in 1778-1823 | Reading, Berkshire (town) | Q161491 |
| 34 | COMBINATION | 43K | 21-7.5K across 7 editions | Mathematical combinations | Ambiguous |
| 35 | BRUTE | 43K | 13-6.7K across 8 editions | Zoological definition | Too generic |
| 36 | ARSURA | 42K | 10.5K in 1797-1823, 118-120 in 1842/60 | Gold/silver assaying term | Swallowed in 1797-1823? Or legit 10K article? |
| 37 | GARDEN | 42K | 4.4-11.9K across 6 editions | Gardening | Q1107656 (garden) |
| 38 | CUSTOM | 42K | 88-9.1K across 7 editions | Legal customs | Ambiguous |
| 39 | POST | 42K | 108-15.8K across 8 editions | Postal service | Q178777 (mail) |
| 40 | PLANTERSHIP | 42K | 8.3K across 6 editions (consistent) | Sugar plantation management | Unique topic, maybe skip |
| 41 | SUPPER | 41K | 94-8.7K across 6 editions | Dietary article | Too generic |
| 42 | PREROGATIVE | 41K | 6.5-7K across 6 editions | Royal prerogative (legal) | Q3042529 |
| 43 | PHILOSOPHIZING | 40K | 1810:4.6K, 1815:**22K**, 1823:13K | Rules of philosophizing → PHILOSOPHY | Swallowed into PHILOSOPHY |
| 44 | MINE | 40K | 1.4-17K across 6 editions | Mining | Q820477 (mine) |
| 45 | PROVIDENCE | 39K | 16-8.6K across 8 editions | Divine providence (theology) | Q8371 (divine providence) |
| 46 | INJECTION | 38K | 157-8.6K across 7 editions | Medical/anatomical injection | Ambiguous |
| 47 | APPARITIONS | 38K | 1842:19K, 1860:19K | Ghosts/spectral illusions | Q49833 (ghost) |
| 48 | RAMSAY | 38K | 7.6-9.1K across 6 editions | Allan Ramsay (Scots poet) | Q1251746 |
| 49 | KING | 38K | 1.1-6.6K across 8 editions | Definition of kingship | Ambiguous |
| 50 | LOCK | 37K | 14-16.8K, grows over time | Lock mechanism | Q44167 (lock) |
| 51 | PORISM | 37K | 7-8K across 5 editions | Mathematical concept | Q846744 |
| 52 | INDEPENDENTS | 37K | 104-6K across 8 editions | Congregationalism | Q178169 |
| 53 | SMITH | 37K | 216-18K, grows over time | Edmund Smith (poet) → but 1860 has 18K, may be Adam Smith | Check 1860 text |
| 54 | SIMSON | 33K | 5.3-5.7K across 6 editions | Robert Simson (mathematician) | Q555384 |
| 55 | BASALTES | 31K | 193-7.5K across 6 editions | Basalt (mineral) | Q43338 |
| 56 | VEGA | 30K | 74-14K, variable | Lope de Vega (poet) | Q166263 |
| 57 | CANIS | 29K | 42-17K, shrinks over time | Dog genus (Canis) | Q16868 |
| 58 | ECONOMISTS | 31K | 1842:15K, 1860:15K | Physiocrats | Q187655 |
| 59 | MATERIALISTS | 30K | 147-15K across 3 editions | Materialism (philosophy) | Q7098 |
| 60 | COTTAGE SYSTEM | 31K | 10-11K across 3 editions | Land allotment for poor | Unique topic |
| 61 | STONES AND EARTHS | 34K | 1810:16K, 1815:17K | Chemical analysis of minerals | Subsection of Mineralogy |
| 62 | PRACTICE OF NAVIGATION | 33K | 1823:17K, 1842:17K | Navigation manual | Subsection |
| 63 | NATIONAL EDUCATION | 29K | 1842:47, 1860:29K | Public education system | 1860 swallowed? Or legit article |

## ADDITIONAL AMBIGUOUS HEADWORDS (from 20-50K range, Mar 29 2026)

Found during headword disambiguation session — these need further investigation.

| # | Headword | Total | Eds | Issue |
|---|----------|-------|-----|-------|
| 64 | SHOOTING | 70K | 7 | Ambiguous: gunnery vs sport hunting |
| 65 | PROPERTY | 65K | 8 | Too generic: legal vs philosophical concept |
| 66 | PASSION | 65K | 5 | Rhetoric/literary concept, no clear single QID |
| 67 | PRESCRIPTION | 64K | 7 | Scots law vs general legal concept — check content |
| 68 | PLANTING | 62K | 7 | Ambiguous: agriculture vs plantation system |
| 69 | SECT | 58K | 8 | Generic: religious sects, no single QID |
| 70 | MIGRATION | 45K | 6 | Ambiguous: human vs animal vs general concept |
| 71 | PHILIP | 44K | 6 | Which Philip? Biblical, Macedonian, or other |
| 72 | HUNTER | 45K | 6 | Check text — horses for hunting? Or a person? |
| 73 | CUSTOM | 42K | 7 | Ambiguous: legal customs vs cultural customs |
| 74 | INTEREST | 36K | 8 | Ambiguous: economic interest vs general concept |
| 75 | SOCIETY | 36K | 8 | Too generic |
| 76 | KING | 38K | 8 | Definition of kingship — too generic |
| 77 | CAMPBELL | 32K | 6 | Which Campbell? Need to check text |
| 78 | CLARKE | 30K | 7 | Which Clarke? Samuel Clarke likely (philosopher) |
| 79 | HENRY | 30K | 8 | Which Henry? Ambiguous across many people |
| 80 | ADAM | 33K | 7 | Biblical Adam? Or Robert Adam? Check text |
| 81 | SIGN | 32K | 6 | Ambiguous: mathematical, linguistic, or general |
| 82 | COLD | 31K | 7 | Ambiguous: physics concept vs medical condition |
| 83 | GENERAL | 30K | 6 | Too generic |
| 84 | CORONA | 28K | 7 | Ambiguous: astronomical or anatomical |
| 85 | BLACK | 27K | 8 | Ambiguous: Joseph Black (chemist)? Or color concept? |
| 86 | WOOD | 27K | 8 | Ambiguous: material vs Robert Wood (traveller) |
| 87 | BULL | 27K | 8 | Ambiguous: papal bull vs George Bull (bishop) |
| 88 | VARIATION | 27K | 7 | Mathematical/musical concept — check which |
| 89 | HAIR | 27K | 7 | Natural history of hair — too generic |
| 90 | COOPER | 27K | 8 | Ambiguous: trade (barrel-making) vs person |
| 91 | CREATION | 26K | 8 | Theological/philosophical — too broad |
| 92 | JOHN | 26K | 7 | Which John? Too many possibilities |

## NEXT STEPS

1. **Parser fixes needed** for the 31 swallowed mega-articles (Section 1)
2. **Quick wins for matching** (Section 2, "Matchable" column): NATURAL HISTORY, VOLTAIC ELECTRICITY, COALERY, STEAM NAVIGATION, POLE, BROWN, SAVAGE, RAMSAY, SIMSON, VEGA, BASALTES, CANIS, ECONOMISTS, MATERIALISTS, READING, PORISM, INDEPENDENTS, PREROGATIVE, etc.
3. **Read first 100 words** in the swallowed edition to determine what content was absorbed (helps identify the parser bug)
4. **Genuinely generic** entries (PROPERTY, PASSION, COMBINATION, BRUTE, SUPPER, etc.) should be skipped
5. **Ambiguous persons** (Section 3): CAMPBELL, CLARKE, HENRY, ADAM, BLACK, COOPER, JOHN — need to read article text to identify the subject
