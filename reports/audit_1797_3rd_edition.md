# Quality Audit Report: 1797 Encyclopedia Britannica (3rd Edition)

**Generated:** 2026-01-03 18:55:43
**Total Articles Analyzed:** 18,016
**Volumes:** 18

---

## Executive Summary

This audit examines the 1797 Third Edition of the Encyclopaedia Britannica for quality issues 
in the parsed HTML corpus. The analysis identified several categories of problems that may 
affect research usability.

| Issue Category | Count | Severity |
|----------------|-------|----------|
| Articles Outside Alphabetical Range | 325 | HIGH |
| OCR/Parsing Errors | 158 | HIGH |
| Unusually Short Articles (<10 words) | 37 | MEDIUM |
| Significant Alphabetical Gaps | 63 | LOW |
| Duplicate Articles | 0 | - |

---

## 1. Articles Outside Alphabetical Range [SEVERITY: HIGH]

Each volume of the encyclopedia covers a specific alphabetical range. Articles appearing 
outside their expected range indicate parsing errors where content was misclassified or 
headwords were incorrectly extracted.

**Total Found:** 325 articles

### By Volume Breakdown:


#### Volume 1 (A-ANG) - Expected letters: A
**18 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-BESIDES_THE_CONNECTION_OF_THE_SEVERAL` | BESIDES THE CONNECTION OF THE SEVERAL VE... | B | 33 |
| `article-CORRECTED` | CORRECTED | C | 104 |
| `article-DEFINITIONS` | DEFINITIONS | D | 171 |
| `article-EDICULA` | EDICULA | E | 146 |
| `article-EXPLANATION_OF_PLATE_XXIII` | EXPLANATION OF PLATE XXIII | E | 539 |
| `article-EXPLANATION_OF_PLATE_XXIV` | EXPLANATION OF PLATE XXIV | E | 352 |
| `article-EXPLANATION_OF_THE_PLATES_OF_OSTEOLGY` | EXPLANATION OF THE PLATES OF OSTEOLGY | E | 613 |
| `article-HERESY_OF_ALMARIC` | HERESY OF ALMARIC | H | 119 |
| `article-INFLAMMABLE_AIR_HAVING_BEEN_AT_FIRST_` | INFLAMMABLE AIR HAVING BEEN AT FIRST PRO... | I | 461 |
| `article-INFLAMMABLE_AIR_PROCURED_BY` | INFLAMMABLE AIR PROCURED BY | I | 590 |
| `article-PART_PART_II` | PART PART II | P | 140 |
| `article-PLATE_XXI` | PLATE XXI | P | 408 |
| `article-PLATE_XXV` | PLATE XXV | P | 577 |
| `article-PLATE_XXVIII` | PLATE XXVIII | P | 702 |
| `article-RAW_FISH_PRODUCES_DEPHLOGISTICATED_AI` | RAW FISH PRODUCES DEPHLOGISTICATED AIR B... | R | 330 |

*...and 3 more articles*

#### Volume 2 (ANG-BAR) - Expected letters: AB
**6 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-NOTWITHSTANDING_ALL_THESE_DISCOVERIES` | NOTWITHSTANDING ALL THESE DISCOVERIES BY | N | 260 |
| `article-PROOF` | PROOF | P | 108 |
| `article-TA` | TA | T | 126 |
| `article-TEIN` | TEIN | T | 327 |
| `article-THOUGH_WE_CAN_BY_NO` | THOUGH WE CAN BY NO | T | 768 |
| `article-THUS_IO` | THUS IO | T | 334 |

#### Volume 3 (BAR-BZO) - Expected letters: B
**25 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-DIANDRIA_MONOGYNYA` | DIANDRIA MONOGYNYA | D | 152 |
| `article-DIDYNAMIA_ANGIOSPERMIA` | DIDYNAMIA ANGIOSPERMIA | D | 172 |
| `article-END_OF_THE_THIRD_VOLUME` | END OF THE THIRD VOLUME | E | 397 |
| `article-ESSENTIAL_CHARACTER` | ESSENTIAL CHARACTER | E | 159 |
| `article-FLOATING_BRIDGE` | FLOATING BRIDGE | F | 136 |
| `article-FREE_BENCH` | FREE BENCH | F | 25 |
| `article-GENERAL_TERMS` | GENERAL TERMS | G | 107 |
| `article-GYNANDRIA_PENTANDRIA` | GYNANDRIA PENTANDRIA | G | 142 |
| `article-HEAD_BOROUGH_ALSO` | HEAD BOROUGH ALSO | H | 22 |
| `article-ICOSANDRIA_POLYGAMIA` | ICOSANDRIA POLYGAMIA | I | 150 |
| `article-JOURNAL` | JOURNAL | J | 86 |
| `article-KFG` | KFG | K | 433 |
| `article-LORD_CHATHAM_WAS_BY_NO` | LORD CHATHAM WAS BY NO | L | 650 |
| `article-MONODELPHIA_POLYANDRIA` | MONODELPHIA POLYANDRIA | M | 129 |
| `article-MONOECIA_TETRANDRIA` | MONOECIA TETRANDRIA | M | 108 |

*...and 10 more articles*

#### Volume 4 (CAA-CIC) - Expected letters: C
**25 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ACID_OF_BORAX` | ACID OF BORAX | A | 675 |
| `article-ACIDS` | ACIDS | A | 11 |
| `article-ALCOHOL` | ALCOHOL | A | 20 |
| `article-ALKALIES` | ALKALIES | A | 609 |
| `article-BATTLE` | BATTLE | B | 196 |
| `article-EARTHS` | EARTHS | E | 46 |
| `article-END_OF_THE_FOURTH_VOLUME` | END OF THE FOURTH VOLUME | E | 491 |
| `article-HAVING_EXAMINED_THE_SUBSTANCE_BY` | HAVING EXAMINED THE SUBSTANCE BY | H | 130 |
| `article-METALS` | METALS | M | 22 |
| `article-NATURAL_BODIES` | NATURAL BODIES | N | 870 |
| `article-NEW_CAMBRIDGE` | NEW CAMBRIDGE | N | 264 |
| `article-OILS` | OILS | O | 69 |
| `article-OSTEOCELLA` | OSTEOCELLA | O | 324 |
| `article-QUEEN_CHARLOTTE_S_ISLAND` | QUEEN CHARLOTTE S ISLAND | Q | 81 |
| `article-QUEEN_CHARLOTTE_S_ISLANDS` | QUEEN CHARLOTTE S ISLANDS | Q | 142 |

*...and 10 more articles*

#### Volume 5 (CIC-DIA) - Expected letters: CD
**12 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ARTIFICIAL_CORUSCATIONS_MAY_ALSO_BE_P` | ARTIFICIAL CORUSCATIONS MAY ALSO BE PROD... | A | 196 |
| `article-ELECTRICAL_CIRCUIT` | ELECTRICAL CIRCUIT | E | 613 |
| `article-END_OF_THE_FIFTH_VOLUME` | END OF THE FIFTH VOLUME | E | 163 |
| `article-HALF_DECK` | HALF DECK | H | 64 |
| `article-O` | O | O | 240 |
| `article-POSITIVE_LOSS` | POSITIVE LOSS | P | 34 |
| `article-RELATIVE_PROFIT_IS_WHAT` | RELATIVE PROFIT IS WHAT | R | 279 |
| `article-SHAKESPEARE` | SHAKESPEARE | S | 33 |
| `article-SINCE_THE_DISCOVERY_OF_THE_POSSIBILIT` | SINCE THE DISCOVERY OF THE POSSIBILITY O... | S | 939 |
| `article-THIS_IS_THE_BEST` | THIS IS THE BEST | T | 892 |
| `article-WATER_IS_FOUND_TO_SUSPEND_THE_RESIN_B` | WATER IS FOUND TO SUSPEND THE RESIN BY | W | 486 |
| `article-YET_LET_US_ENQUIRE_WHAT` | YET LET US ENQUIRE WHAT | Y | 344 |

#### Volume 6 (DIA-ETH) - Expected letters: DE
**22 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-APHORISMS` | APHORISMS | A | 507 |
| `article-ARCHITECTO_ROBERTO_ADAM` | ARCHITECTO ROBERTO ADAM | A | 520 |
| `article-BROAD` | BROAD | B | 193 |
| `article-CONDUCTING_POWER_OF_VARIOUS_SUBSTANCE` | CONDUCTING POWER OF VARIOUS SUBSTANCES A... | C | 333 |
| `article-GEOMETRICAL_FIGURES_BEAUTIFULLY_SHOWN` | GEOMETRICAL FIGURES BEAUTIFULLY SHOWN BY | G | 634 |
| `article-HAROLD_IN_THE_MEAN_TIME_INCREASED_HIS` | HAROLD IN THE MEAN TIME INCREASED HIS PO... | H | 993 |
| `article-HAVING_ENDEAVOURED_TO_ASSIGN_THE_EFFI` | HAVING ENDEAVOURED TO ASSIGN THE EFFICIE... | H | 679 |
| `article-HAVING_THUS_DESCRIBED_VERY_PARTICULAR` | HAVING THUS DESCRIBED VERY PARTICULARLY ... | H | 1 |
| `article-ISLANDS_OF_DISAPPOINTMENT` | ISLANDS OF DISAPPOINTMENT | I | 119 |
| `article-LET_US_GIVE_YET_ANOTHER_INSTANCE_OF_T` | LET US GIVE YET ANOTHER INSTANCE OF THE | L | 433 |
| `article-LLENCHUS` | LLENCHUS | L | 32 |
| `article-MECHANICAL_DIVISION` | MECHANICAL DIVISION | M | 814 |
| `article-MOE` | MOE | M | 122 |
| `article-NARROW` | NARROW | N | 121 |
| `article-PHYSCON_HAVING_BY_THIS` | PHYSCON HAVING BY THIS | P | 382 |

*...and 7 more articles*

#### Volume 7 (ETM-GOA) - Expected letters: EFG
**41 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ADC` | ADC | A | 29 |
| `article-DH` | DH | D | 119 |
| `article-LETTERS_OF_EXONERATION` | LETTERS OF EXONERATION | L | 41 |
| `article-MONEY_TABLE` | MONEY TABLE | M | 768 |
| `article-MOST_OF_THE_ABOVE_PROBLEMS_MAY_ALSO_B` | MOST OF THE ABOVE PROBLEMS MAY ALSO BE P... | M | 807 |
| `article-ONTARABIA` | ONTARABIA | O | 94 |
| `article-OUR_AUTHOR` | OUR AUTHOR | O | 170 |
| `article-PAINTING_ON_GLASS_BY` | PAINTING ON GLASS BY | P | 874 |
| `article-PLEURI` | PLEURI | P | 51 |
| `article-PROPOSITION_II` | PROPOSITION II | P | 514 |
| `article-PROPOSITION_III` | PROPOSITION III | P | 131 |
| `article-PROPOSITION_IV` | PROPOSITION IV | P | 510 |
| `article-PROPOSITION_LI` | PROPOSITION LI | P | 176 |
| `article-PROPOSITION_LII` | PROPOSITION LII | P | 159 |
| `article-PROPOSITION_LIII` | PROPOSITION LIII | P | 862 |

*...and 26 more articles*

#### Volume 8 (GOB-HYD) - Expected letters: GH
**28 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ADJECTIVES` | ADJECTIVES | A | 84 |
| `article-ADVERBS` | ADVERBS | A | 74 |
| `article-AGO` | AGO | A | 125 |
| `article-AGRICULTURE` | AGRICULTURE | A | 460 |
| `article-ARISTAEUS` | ARISTAEUS | A | 50 |
| `article-ARTICLES` | ARTICLES | A | 53 |
| `article-CONDAR` | CONDAR | C | 469 |
| `article-CONJUNCTIONS` | CONJUNCTIONS | C | 69 |
| `article-CONVEYS_TO_THE_MIND_OF_THE_READER_THE` | CONVEYS TO THE MIND OF THE READER THE VE... | C | 109 |
| `article-END_OF_THE_EIGHTH_VOLUME` | END OF THE EIGHTH VOLUME | E | 166 |
| `article-FEE` | FEE | F | 186 |
| `article-FINGAL` | FINGAL | F | 448 |
| `article-INTERJECTIONS` | INTERJECTIONS | I | 25 |
| `article-MDCCCLXXXIII` | MDCCCLXXXIII | M | 123 |
| `article-MORAL_GOOD` | MORAL GOOD | M | 764 |

*...and 13 more articles*

#### Volume 9 (Hydrostatics-LES) - Expected letters: HIJKL
**9 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-AMONG_PAINTERS_IT` | AMONG PAINTERS IT | A | 31 |
| `article-END_OF_THE_NINTH_VOLUME` | END OF THE NINTH VOLUME | E | 344 |
| `article-GENERAL_OBSERVATIONS` | GENERAL OBSERVATIONS | G | 497 |
| `article-POET_LAUREATE` | POET LAUREATE | P | 871 |
| `article-RELIGIOUS_INFIDELITY` | RELIGIOUS INFIDELITY | R | 258 |
| `article-SOME_MYTHOLOGISTS_SUPPOSE_THAT_JUNO` | SOME MYTHOLOGISTS SUPPOSE THAT JUNO | S | 33 |
| `article-ST_JANUARIUS` | ST JANUARIUS | S | 87 |
| `article-STOCK_JOBBING` | STOCK JOBBING | S | 76 |
| `article-WATER_MAY_ALSO_BE_RAISED_BY` | WATER MAY ALSO BE RAISED BY | W | 679 |

#### Volume 10 (LES-MEC) - Expected letters: LM
**13 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ANOTHER_METHOD_OF_ACCUMULATING_FORCE_` | ANOTHER METHOD OF ACCUMULATING FORCE IS ... | A | 924 |
| `article-HITHERTO_MAHOMET_HAD_PROPAGATED_HIS_R` | HITHERTO MAHOMET HAD PROPAGATED HIS RELI... | H | 753 |
| `article-OUTH` | OUTH | O | 172 |
| `article-OUVAIN` | OUVAIN | O | 723 |
| `article-REPARATION_IS_PERFORMED_BY` | REPARATION IS PERFORMED BY | R | 218 |
| `article-ST_MAWES` | ST MAWES | S | 123 |
| `article-THIS_MAY_BE_EASILY_ACCOMPLISHED_BY` | THIS MAY BE EASILY ACCOMPLISHED BY | T | 83 |
| `article-ULLI` | ULLI | U | 113 |
| `article-ULLY` | ULLY | U | 141 |
| `article-UMBAGO` | UMBAGO | U | 11 |
| `article-UMBARIS` | UMBARIS | U | 13 |
| `article-UMBRIAL` | UMBRIAL | U | 16 |
| `article-UMBRIUS` | UMBRIUS | U | 40 |

#### Volume 11 (Medals-Midwifery) - Expected letters: M
**30 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-AABC` | AABC | A | 228 |
| `article-ADYNAMIAE` | ADYNAMIAE | A | 300 |
| `article-AFTER_THE_PATIENT_HAS_BY_THIS` | AFTER THE PATIENT HAS BY THIS | A | 463 |
| `article-AMONG_OTHER` | AMONG OTHER | A | 171 |
| `article-ARTHRODYNA` | ARTHRODYNA | A | 318 |
| `article-CEPHALALGIA` | CEPHALALGIA | C | 15 |
| `article-CLASS_II` | CLASS II | C | 351 |
| `article-CLASS_III` | CLASS III | C | 126 |
| `article-DEPRAVED_VISION` | DEPRAVED VISION | D | 46 |
| `article-DIFFICULTY_OF_DISCHARGING_URINE` | DIFFICULTY OF DISCHARGING URINE | D | 470 |
| `article-DUMBNESS` | DUMBNESS | D | 446 |
| `article-END_OF_THE_ELEVENTH_VOLUME` | END OF THE ELEVENTH VOLUME | E | 121 |
| `article-EXERCISE_AND_ABSTINENCE_ARE_THE` | EXERCISE AND ABSTINENCE ARE THE | E | 997 |
| `article-FALLING_SICKNESS` | FALLING SICKNESS | F | 541 |
| `article-HYPOCHONDRIAC_AFFECTION` | HYPOCHONDRIAC AFFECTION | H | 364 |

*...and 15 more articles*

#### Volume 12 (MEI-NEG) - Expected letters: MN
**11 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-AMPUTATION_IS_NOT_THE_ONLY` | AMPUTATION IS NOT THE ONLY | A | 767 |
| `article-DEFINITIONS_OF_SEVERAL_TECHNICAL_TERM` | DEFINITIONS OF SEVERAL TECHNICAL TERMS | D | 829 |
| `article-ELEMENTS_OF_MUSIC` | ELEMENTS OF MUSIC | E | 234 |
| `article-END_OF_THE_TWELFTH_VOLUME` | END OF THE TWELFTH VOLUME | E | 193 |
| `article-HENRIADE` | HENRIADE | H | 96 |
| `article-ONE_GREAT` | ONE GREAT | O | 242 |
| `article-REMARK` | REMARK | R | 264 |
| `article-RULE_II` | RULE II | R | 541 |
| `article-SAXA` | SAXA | S | 66 |
| `article-THIS_OPERATION_OF_ADJUSTING_THE_METAL` | THIS OPERATION OF ADJUSTING THE METALS T... | T | 680 |
| `article-UT` | UT | U | 69 |

#### Volume 13 (NEH-PAS) - Expected letters: NOP
**14 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-AVIUM_ORDINES` | AVIUM ORDINES | A | 22 |
| `article-BECAUSE_THE_EQUABLE_DESCRIPTION_OF_AR` | BECAUSE THE EQUABLE DESCRIPTION OF AREAS | B | 906 |
| `article-DRYDEN` | DRYDEN | D | 777 |
| `article-END_OF_THE_THIRTEENTH_VOLUME_ERRATA` | END OF THE THIRTEENTH VOLUME ERRATA | E | 86 |
| `article-GREATER_OUSE` | GREATER OUSE | G | 971 |
| `article-HISTORY` | HISTORY | H | 951 |
| `article-MARCVS` | MARCVS | M | 11 |
| `article-RESTITVERVNT` | RESTITVERVNT | R | 672 |
| `article-TIMES` | TIMES | T | 84 |
| `article-UNDINA` | UNDINA | U | 52 |
| `article-UNDINAL` | UNDINAL | U | 148 |
| `article-UNDOCOMAR` | UNDOCOMAR | U | 335 |
| `article-VOLUME_XIII` | VOLUME XIII | V | 254 |
| `article-WHITE_ORDER` | WHITE ORDER | W | 202 |

#### Volume 14 (PAS-PLA) - Expected letters: P
**14 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ACCOMMODATES_ITSELF_TO_DIFFERENT_CIRC` | ACCOMMODATES ITSELF TO DIFFERENT CIRCUMS... | A | 745 |
| `article-BEFORE_MAN_HAD_RECOURSE_TO_AGRICULTUR` | BEFORE MAN HAD RECOURSE TO AGRICULTURE A... | B | 190 |
| `article-BOTH_CALAMINE_AND_TUTTY_ACT_ONLY_BY` | BOTH CALAMINE AND TUTTY ACT ONLY BY | B | 125 |
| `article-CASES_OF_DOUBLE_ELECTIVE_ATTRACTIONS` | CASES OF DOUBLE ELECTIVE ATTRACTIONS | C | 259 |
| `article-END_OF_THE_FOURTEENTH_VOLUME` | END OF THE FOURTEENTH VOLUME | E | 605 |
| `article-FEWER_ERRORS_HAVE_BEEN_COMMITTED_IN_T` | FEWER ERRORS HAVE BEEN COMMITTED IN THE | F | 349 |
| `article-INDEED_THE_SPANIARDS_APPEAR_BY_NO` | INDEED THE SPANIARDS APPEAR BY NO | I | 367 |
| `article-OPIUM_WAS_FORMERLY_PURIFIED_BY` | OPIUM WAS FORMERLY PURIFIED BY | O | 33 |
| `article-STORAX_WAS_FORMERLY_DIRECTED_TO_BE_PU` | STORAX WAS FORMERLY DIRECTED TO BE PURIF... | S | 123 |
| `article-THIS_ASSEMBLING_OF_THE_INDIVIDUAL_OBJ` | THIS ASSEMBLING OF THE INDIVIDUAL OBJECT... | T | 460 |
| `article-THIS_MAY_BE_IN_SOME_CASES_AN_USEFUL` | THIS MAY BE IN SOME CASES AN USEFUL | T | 591 |
| `article-THOUGH_IT_IS_BY_NO` | THOUGH IT IS BY NO | T | 390 |
| `article-WHEREVER_THE_ECONOMY_OF_LIVING_BODIES` | WHEREVER THE ECONOMY OF LIVING BODIES | W | 349 |
| `article-WHILE_THE_CONTEMPLATION_OF_THESE_APPE` | WHILE THE CONTEMPLATION OF THESE APPEARA... | W | 50 |

#### Volume 15 (PLA-RAM) - Expected letters: PQR
**23 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-AMONG_THE_ROMANS_PROFESSION` | AMONG THE ROMANS PROFESSION | A | 28 |
| `article-ASSERTORIS` | ASSERTORIS | A | 15 |
| `article-AZD` | AZD | A | 317 |
| `article-COROLLARIES` | COROLLARIES | C | 308 |
| `article-COROLLARY` | COROLLARY | C | 234 |
| `article-DANIEL_PULTENEY` | DANIEL PULTENEY | D | 932 |
| `article-END_OF_THE_FIFTEENTH_VOLUME` | END OF THE FIFTEENTH VOLUME | E | 117 |
| `article-EXPERIMENTS_ON_THIS_SUBJECT_ARE_BY_NO` | EXPERIMENTS ON THIS SUBJECT ARE BY NO | E | 462 |
| `article-FOG` | FOG | F | 140 |
| `article-GULIELMO_BRIDGEN` | GULIELMO BRIDGEN | G | 595 |
| `article-SEC_ORATORY` | SEC ORATORY | S | 12 |
| `article-SOAMES` | SOAMES | S | 316 |
| `article-TERRA_PUZZOLANA` | TERRA PUZZOLANA | T | 442 |
| `article-THESE_RESOLUTIONS_OF_THE_DIET_WERE_BY` | THESE RESOLUTIONS OF THE DIET WERE BY NO | T | 537 |
| `article-THIS_COMBINATION_OF_AIR_WITH_WATER_IS` | THIS COMBINATION OF AIR WITH WATER IS VE... | T | 978 |

*...and 8 more articles*

#### Volume 16 (RAN-SCO) - Expected letters: RS
**9 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-CITIES_OF_REFUGE` | CITIES OF REFUGE | C | 396 |
| `article-END_OF_THE_SIXTEENTH_VOLUME` | END OF THE SIXTEENTH VOLUME | E | 313 |
| `article-GALBA_HAVING_BEEN_BROUGHT_TO_THE_EMPI` | GALBA HAVING BEEN BROUGHT TO THE EMPIRE ... | G | 179 |
| `article-JOHN_BALIOL` | JOHN BALIOL | J | 522 |
| `article-NOVA_SCOTIA` | NOVA SCOTIA | N | 711 |
| `article-THAT_THIS` | THAT THIS | T | 538 |
| `article-THERE_ARE_BY_NO` | THERE ARE BY NO | T | 576 |
| `article-THIS_IS_BY_NO` | THIS IS BY NO | T | 45 |
| `article-THIS_PROPORTION_WILL_BE_FOUND_BY_TREA` | THIS PROPORTION WILL BE FOUND BY TREATIN... | T | 678 |

#### Volume 17 (SCO-STR) - Expected letters: S
**17 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-BEFORE_WE_PROCEED_TO_THE_DESCRIPTION_` | BEFORE WE PROCEED TO THE DESCRIPTION OF ... | B | 649 |
| `article-CONCERNING_THE_DECOMPOSITION_OF_SOAP_` | CONCERNING THE DECOMPOSITION OF SOAP BY | C | 371 |
| `article-END_OF_THE_SEVENTEENTH_VOLUME` | END OF THE SEVENTEENTH VOLUME | E | 419 |
| `article-HENRY_STEPHENS` | HENRY STEPHENS | H | 134 |
| `article-ISAIAH_BY` | ISAIAH BY | I | 80 |
| `article-MECHANICS` | MECHANICS | M | 72 |
| `article-OCULAR_SPECTRA` | OCULAR SPECTRA | O | 162 |
| `article-OS_SPHENOIDES` | OS SPHENOIDES | O | 12 |
| `article-PAPHLAGONIA` | PAPHLAGONIA | P | 87 |
| `article-PAPHOS` | PAPHOS | P | 964 |
| `article-PAPIAS` | PAPIAS | P | 70 |
| `article-PAPILIO` | PAPILIO | P | 237 |
| `article-PROBLEM` | PROBLEM | P | 997 |
| `article-ROBERT_STEPHENS` | ROBERT STEPHENS | R | 961 |
| `article-THERE_ARE_THE` | THERE ARE THE | T | 606 |

*...and 2 more articles*

#### Volume 18 (STR-ZYM) - Expected letters: STUVWXYZ
**8 articles outside range:**

| Article ID | Headword | First Letter | Word Count |
|------------|----------|--------------|------------|
| `article-ALL_THE_EFFUSED_BLOOD_OUGHT_THEN_TO_B` | ALL THE EFFUSED BLOOD OUGHT THEN TO BE W... | A | 597 |
| `article-BRITISH_WOOL_SOCIETY` | BRITISH WOOL SOCIETY | B | 381 |
| `article-HITHERTO_THESE_UNHALLOWED` | HITHERTO THESE UNHALLOWED | H | 1 |
| `article-INCAL` | INCAL | I | 14 |
| `article-ISLE_OF_WIGHT` | ISLE OF WIGHT | I | 299 |
| `article-JONATHAN_SWIFT` | JONATHAN SWIFT | J | 541 |
| `article-NAVAL_TACTICS` | NAVAL TACTICS | N | 884 |
| `article-PLANE_TRIGONOMETRY` | PLANE TRIGONOMETRY | P | 383 |

---

## 2. OCR/Parsing Errors [SEVERITY: HIGH]

These articles have headwords that appear to be sentence fragments, section markers, 
plate descriptions, or other text incorrectly parsed as article titles.


### Sentence fragment (104 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 1 | `article-BESIDES_THE_CONNECTION_OF_THE_SE` | BESIDES THE CONNECTION OF THE SEVERAL VERTEBR... | 33 |
| 1 | `article-EXPLANATION_OF_THE_PLATES_OF_OST` | EXPLANATION OF THE PLATES OF OSTEOLGY | 613 |
| 1 | `article-INFLAMMABLE_AIR_HAVING_BEEN_AT_F` | INFLAMMABLE AIR HAVING BEEN AT FIRST PRODUCED... | 461 |
| 1 | `article-RAW_FISH_PRODUCES_DEPHLOGISTICAT` | RAW FISH PRODUCES DEPHLOGISTICATED AIR BY | 330 |
| 1 | `article-WHEN_THE_PERSPIRATION_IS_BY_ANY` | WHEN THE PERSPIRATION IS BY ANY | 941 |
| 1 | `article-WINGS_OR_OARS_ARE_THE_ONLY` | WINGS OR OARS ARE THE ONLY | 347 |
| 2 | `article-ANGLE_OF_REFRACTION_NOW_GENERALL` | ANGLE OF REFRACTION NOW GENERALLY | 328 |
| 2 | `article-NOTWITHSTANDING_ALL_THESE_DISCOV` | NOTWITHSTANDING ALL THESE DISCOVERIES BY | 260 |
| 2 | `article-THOUGH_WE_CAN_BY_NO` | THOUGH WE CAN BY NO | 768 |
| 3 | `article-BRETHREN_AND_SISTERS_OF_THE_FREE` | BRETHREN AND SISTERS OF THE FREE SPIRIT | 654 |
| 3 | `article-END_OF_THE_THIRD_VOLUME` | END OF THE THIRD VOLUME | 397 |
| 3 | `article-LORD_CHATHAM_WAS_BY_NO` | LORD CHATHAM WAS BY NO | 650 |

*...and 92 more*

### Truncated at BY (41 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 1 | `article-BESIDES_THE_CONNECTION_OF_THE_SE` | BESIDES THE CONNECTION OF THE SEVERAL VERTEBR... | 33 |
| 1 | `article-INFLAMMABLE_AIR_HAVING_BEEN_AT_F` | INFLAMMABLE AIR HAVING BEEN AT FIRST PRODUCED... | 461 |
| 1 | `article-INFLAMMABLE_AIR_PROCURED_BY` | INFLAMMABLE AIR PROCURED BY | 590 |
| 1 | `article-RAW_FISH_PRODUCES_DEPHLOGISTICAT` | RAW FISH PRODUCES DEPHLOGISTICATED AIR BY | 330 |
| 2 | `article-NOTWITHSTANDING_ALL_THESE_DISCOV` | NOTWITHSTANDING ALL THESE DISCOVERIES BY | 260 |
| 4 | `article-HAVING_EXAMINED_THE_SUBSTANCE_BY` | HAVING EXAMINED THE SUBSTANCE BY | 130 |
| 4 | `article-SOLUTIONS_EFFECTED_BY` | SOLUTIONS EFFECTED BY | 433 |
| 4 | `article-THEIR_HOUSES_ARE_MADE_IN_THE_WAT` | THEIR HOUSES ARE MADE IN THE WATER COLLECTED ... | 260 |
| 4 | `article-THIS_MAY_BE_OBTAINED_FROM_FUEL_B` | THIS MAY BE OBTAINED FROM FUEL BY | 447 |
| 5 | `article-ARTIFICIAL_CORUSCATIONS_MAY_ALSO` | ARTIFICIAL CORUSCATIONS MAY ALSO BE PRODUCED ... | 196 |
| 5 | `article-WATER_IS_FOUND_TO_SUSPEND_THE_RE` | WATER IS FOUND TO SUSPEND THE RESIN BY | 486 |
| 6 | `article-CONDUCTING_POWER_OF_VARIOUS_SUBS` | CONDUCTING POWER OF VARIOUS SUBSTANCES ASCERT... | 333 |

*...and 29 more*

### Section marker (28 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 7 | `article-PROPOSITION_II` | PROPOSITION II | 514 |
| 7 | `article-PROPOSITION_III` | PROPOSITION III | 131 |
| 7 | `article-PROPOSITION_IV` | PROPOSITION IV | 510 |
| 7 | `article-PROPOSITION_LI` | PROPOSITION LI | 176 |
| 7 | `article-PROPOSITION_LII` | PROPOSITION LII | 159 |
| 7 | `article-PROPOSITION_LIII` | PROPOSITION LIII | 862 |
| 7 | `article-PROPOSITION_VI` | PROPOSITION VI | 180 |
| 7 | `article-PROPOSITION_VII` | PROPOSITION VII | 455 |
| 7 | `article-PROPOSITION_VIII` | PROPOSITION VIII | 222 |
| 7 | `article-PROPOSITION_XL` | PROPOSITION XL | 210 |
| 7 | `article-PROPOSITION_XLI` | PROPOSITION XLI | 134 |
| 7 | `article-PROPOSITION_XLII` | PROPOSITION XLII | 205 |

*...and 16 more*

### Starts with determiner (23 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 3 | `article-THESE_REMONSTRANCES_WERE_BY_NO` | THESE REMONSTRANCES WERE BY NO | 405 |
| 3 | `article-THIS_REPLY_DID_NOT_BY_ANY` | THIS REPLY DID NOT BY ANY | 221 |
| 4 | `article-THIS_MAY_BE_OBTAINED_FROM_FUEL_B` | THIS MAY BE OBTAINED FROM FUEL BY | 447 |
| 5 | `article-THIS_IS_THE_BEST` | THIS IS THE BEST | 892 |
| 6 | `article-THIS_ACCIDENT_PROVED_THE` | THIS ACCIDENT PROVED THE | 155 |
| 6 | `article-THIS_WILL_EASILY_APPEAR_FROM_CON` | THIS WILL EASILY APPEAR FROM CONSIDERING THE | 255 |
| 7 | `article-THIS_DISTEMPER_IS_TO_BE_CURED_BY` | THIS DISTEMPER IS TO BE CURED BY THESE | 775 |
| 7 | `article-THIS_METHOD_OF_PRODUCING_COLD_BY` | THIS METHOD OF PRODUCING COLD BY | 362 |
| 8 | `article-THAT_PART_OF_MEDICINE_WHICH_SHOW` | THAT PART OF MEDICINE WHICH SHOWS THE | 76 |
| 10 | `article-THIS_MAY_BE_EASILY_ACCOMPLISHED_` | THIS MAY BE EASILY ACCOMPLISHED BY | 83 |
| 11 | `article-THESE_WERE_THE_ONLY` | THESE WERE THE ONLY | 219 |
| 12 | `article-THIS_OPERATION_OF_ADJUSTING_THE_` | THIS OPERATION OF ADJUSTING THE METALS TO THE... | 680 |

*...and 11 more*

### Volume ending marker (14 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 3 | `article-END_OF_THE_THIRD_VOLUME` | END OF THE THIRD VOLUME | 397 |
| 4 | `article-END_OF_THE_FOURTH_VOLUME` | END OF THE FOURTH VOLUME | 491 |
| 5 | `article-END_OF_THE_FIFTH_VOLUME` | END OF THE FIFTH VOLUME | 163 |
| 6 | `article-END_OF_THE_SIXTH_VOLUME` | END OF THE SIXTH VOLUME | 286 |
| 7 | `article-END_OF_THE_SEVENTH_VOLUME` | END OF THE SEVENTH VOLUME | 172 |
| 8 | `article-END_OF_THE_EIGHTH_VOLUME` | END OF THE EIGHTH VOLUME | 166 |
| 9 | `article-END_OF_THE_NINTH_VOLUME` | END OF THE NINTH VOLUME | 344 |
| 11 | `article-END_OF_THE_ELEVENTH_VOLUME` | END OF THE ELEVENTH VOLUME | 121 |
| 12 | `article-END_OF_THE_TWELFTH_VOLUME` | END OF THE TWELFTH VOLUME | 193 |
| 13 | `article-END_OF_THE_THIRTEENTH_VOLUME_ERR` | END OF THE THIRTEENTH VOLUME ERRATA | 86 |
| 14 | `article-END_OF_THE_FOURTEENTH_VOLUME` | END OF THE FOURTEENTH VOLUME | 605 |
| 15 | `article-END_OF_THE_FIFTEENTH_VOLUME` | END OF THE FIFTEENTH VOLUME | 117 |

*...and 2 more*

### Botanical classification header (12 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 2 | `article-ANGIOSPERMIA` | ANGIOSPERMIA | 42 |
| 3 | `article-DIANDRIA_MONOGYNYA` | DIANDRIA MONOGYNYA | 152 |
| 3 | `article-DIDYNAMIA_ANGIOSPERMIA` | DIDYNAMIA ANGIOSPERMIA | 172 |
| 3 | `article-GYNANDRIA_PENTANDRIA` | GYNANDRIA PENTANDRIA | 142 |
| 3 | `article-ICOSANDRIA_POLYGAMIA` | ICOSANDRIA POLYGAMIA | 150 |
| 3 | `article-MONOECIA_TETRANDRIA` | MONOECIA TETRANDRIA | 108 |
| 3 | `article-SYNGENESIA_POLYGAMIA_AEQUALIS` | SYNGENESIA POLYGAMIA AEQUALIS | 166 |
| 6 | `article-DIANDRIA` | DIANDRIA | 66 |
| 8 | `article-GYNANDRIA` | GYNANDRIA | 98 |
| 12 | `article-MONADELPHIA` | MONADELPHIA | 56 |
| 12 | `article-MONOGYNYA` | MONOGYNYA | 50 |
| 18 | `article-TETRANDRIA` | TETRANDRIA | 127 |

### Plate description (4 articles)

| Volume | Article ID | Headword | Words |
|--------|------------|----------|-------|
| 1 | `article-EXPLANATION_OF_PLATE_XXIII` | EXPLANATION OF PLATE XXIII | 539 |
| 1 | `article-EXPLANATION_OF_PLATE_XXIV` | EXPLANATION OF PLATE XXIV | 352 |
| 1 | `article-EXPLANATION_OF_THE_PLATES_OF_OST` | EXPLANATION OF THE PLATES OF OSTEOLGY | 613 |
| 1 | `article-PLATE_XXI` | PLATE XXI | 408 |

---

## 3. Unusually Short Articles [SEVERITY: MEDIUM]

Articles with fewer than 10 words may indicate incomplete parsing, OCR errors, 
or cross-references that were not properly handled.

**Total Found:** 37 articles

| Volume | Article ID | Headword | Word Count |
|--------|------------|----------|------------|
| 6 | `article-HAVING_THUS_DESCRIBED_VERY_PARTI` | HAVING THUS DESCRIBED VERY PARTICULARLY  | 1 |
| 7 | `article-FORTUNE` | FORTUNE | 1 |
| 18 | `article-HITHERTO_THESE_UNHALLOWED` | HITHERTO THESE UNHALLOWED | 1 |
| 18 | `article-VITIS` | VITIS | 1 |
| 1 | `article-AGRIGENTUM` | AGRIGENTUM | 2 |
| 3 | `article-BETHSAIDA` | BETHSAIDA | 2 |
| 3 | `article-BISCHOF` | BISCHOF | 2 |
| 3 | `article-BITUMEN_JUDAIUM` | BITUMEN JUDAIUM | 2 |
| 4 | `article-CANAAAN` | CANAAAN | 2 |
| 7 | `article-EULER` | EULER | 2 |
| 8 | `article-HANDEL` | HANDEL | 2 |
| 10 | `article-MASCICOT` | MASCICOT | 2 |
| 10 | `article-MASSILLON` | MASSILLON | 2 |
| 14 | `article-PHILOSTRATUS` | PHILOSTRATUS | 2 |
| 18 | `article-TURGOT` | TURGOT | 2 |
| 3 | `article-BREVIARY` | BREVIARY | 3 |
| 12 | `article-MONAD` | MONAD | 3 |
| 18 | `article-WOLFRAM` | WOLFRAM | 3 |
| 2 | `article-ANTHELION` | ANTHELION | 4 |
| 8 | `article-HOBBES` | HOBBES | 4 |
| 10 | `article-MASSANIELLO` | MASSANIELLO | 4 |
| 17 | `article-SHAKE` | SHAKE | 4 |
| 9 | `article-LADOGNA` | LADOGNA | 5 |
| 14 | `article-PIKE` | PIKE | 5 |
| 15 | `article-PORT` | PORT | 5 |

*...and 12 more*

---

## 4. Longest Articles (Potential Merged Content)

These are the longest articles in the edition. While treatises are expected to be long,
unusually lengthy articles may indicate multiple articles merged together.

**Note:** No articles exceeded 10,000 words, which would be a strong indicator of merging.
The longest articles appear to be legitimate treatises.

| Volume | Article ID | Headword | Word Count | Type |
|--------|------------|----------|------------|------|
| 8 | `article-HAIL` | HAIL | 999 | dictionary |
| 2 | `article-ANTIPAROS` | ANTIPAROS | 998 | treatise |
| 10 | `article-MARSAIS` | MARSAIS | 998 | dictionary |
| 2 | `article-ARCHANGEL` | ARCHANGEL | 997 | geographical |
| 7 | `article-GLYCIIRHIZA` | GLYCIIRHIZA | 997 | dictionary |
| 11 | `article-EXERCISE_AND_ABSTINENCE_ARE` | EXERCISE AND ABSTINENCE ARE THE | 997 | treatise |
| 17 | `article-PROBLEM` | PROBLEM | 997 | treatise |
| 2 | `article-BALANCE` | BALANCE | 996 | treatise |
| 8 | `article-GRACE` | GRACE | 996 | treatise |
| 17 | `article-SPARTA` | SPARTA | 996 | treatise |
| 4 | `article-CHRISTINA` | CHRISTINA | 995 | treatise |
| 4 | `article-CHRYSALIS` | CHRYSALIS | 995 | treatise |
| 1 | `article-AGARICUS` | AGARICUS | 994 | treatise |
| 4 | `article-CAOUTCHOUC` | CAOUTCHOUC | 993 | treatise |
| 4 | `article-CELTIS` | CELTIS | 993 | dictionary |
| 6 | `article-HAROLD_IN_THE_MEAN_TIME_INC` | HAROLD IN THE MEAN TIME INCREASED H | 993 | treatise |
| 2 | `article-ARCHITECTURE` | ARCHITECTURE | 992 | treatise |
| 3 | `article-BEAD` | BEAD | 992 | dictionary |
| 4 | `article-CATALOGUE` | CATALOGUE | 992 | dictionary |
| 2 | `article-APOLLONIA` | APOLLONIA | 991 | geographical |

---

## 5. Duplicate Articles [SEVERITY: -]

**No duplicate headwords were found in the edition.**

This indicates good uniqueness in the article parsing.

---

## 6. Alphabetical Gaps [SEVERITY: LOW]

Large jumps in alphabetical order may indicate missing articles. This analysis looks for
gaps where the second letter of consecutive headwords jumps by more than 5 positions.

**Total Significant Gaps:** 63

### Notable Gaps by Volume:


#### Volume 1 (3 gaps)

| From | To | Gap Size |
|------|-----|----------|
| A LEE | AAHUS | 33 |
| EDICULA | EXPLANATION OF PLATE XXIII | 20 |
| PART PART II | PLATE XXI | 11 |

#### Volume 2 (2 gaps)

| From | To | Gap Size |
|------|-----|----------|
| ADVENTURE | AMIERS | 9 |
| BARBADDOES | BISHOP S AUKLAND | 8 |

#### Volume 3 (2 gaps)

| From | To | Gap Size |
|------|-----|----------|
| FLOATING BRIDGE | FREE BENCH | 6 |
| GENERAL TERMS | GYNANDRIA PENTANDRIA | 20 |

#### Volume 4 (4 gaps)

| From | To | Gap Size |
|------|-----|----------|
| ACIDS | ALCOHOL | 9 |
| EARTHS | END OF THE FOURTH VOLUME | 13 |
| OILS | OSTEOCELLA | 10 |
| SEMIMETALS | SOLUTIONS EFFECTED BY | 10 |

#### Volume 6 (3 gaps)

| From | To | Gap Size |
|------|-----|----------|
| DIVORCE | DOBUNI | 6 |
| LET US GIVE YET ANOTHER INSTAN | LLENCHUS | 7 |
| MECHANICAL DIVISION | MOE | 10 |

#### Volume 7 (5 gaps)

| From | To | Gap Size |
|------|-----|----------|
| END OF THE SEVENTH VOLUME | ETNA | 6 |
| ONTARABIA | OUR AUTHOR | 7 |
| PAINTING ON GLASS BY | PLEURI | 11 |
| PLEURI | PROPOSITION II | 6 |
| THIS METHOD OF PRODUCING COLD  | TRIULI | 10 |

#### Volume 8 (8 gaps)

| From | To | Gap Size |
|------|-----|----------|
| AGRICULTURE | ARISTAEUS | 11 |
| GAUTIMALA | GOBIET | 14 |
| HIVITES | HO KIEN FOU | 6 |
| HOYE | HU QUANG | 6 |
| MDCCCLXXXIII | MORAL GOOD | 11 |
| PARTICLES | POSSESSIVES | 14 |
| SCHLUTTER RECOMMENDS MECHANICA | SUBSTANTIVES | 18 |
| THAT PART OF MEDICINE WHICH SH | TRIFLIROR | 10 |

#### Volume 9 (3 gaps)

| From | To | Gap Size |
|------|-----|----------|
| JEZRAEL | JOAB | 10 |
| JOTAPATA | JUBA | 6 |
| LESSONS | LUBACH | 16 |

#### Volume 10 (1 gaps)

| From | To | Gap Size |
|------|-----|----------|
| LOZENGES | LUBEC | 6 |

#### Volume 11 (5 gaps)

| From | To | Gap Size |
|------|-----|----------|
| AFTER THE PATIENT HAS BY THIS | AMONG OTHER | 7 |
| CEPHALALGIA | CLASS II | 7 |
| DIFFICULTY OF DISCHARGING URIN | DUMBNESS | 12 |
| END OF THE ELEVENTH VOLUME | EXERCISE AND ABSTINENCE ARE TH | 10 |
| WHETHER THERE BE ANY OTHER | WORMS | 7 |

#### Volume 12 (3 gaps)

| From | To | Gap Size |
|------|-----|----------|
| MAGNETICAL NEEDLES | MIERIS | 8 |
| MOYLE | MUCILAGE | 6 |
| REMARK | RULE II | 16 |

#### Volume 13 (3 gaps)

| From | To | Gap Size |
|------|-----|----------|
| NIZAM | NOAH | 6 |
| NOX | NUAYHAS | 6 |
| O RICOLI | OAK | 33 |

#### Volume 14 (1 gaps)

| From | To | Gap Size |
|------|-----|----------|
| BEFORE MAN HAD RECOURSE TO AGR | BOTH CALAMINE AND TUTTY ACT ON | 10 |

#### Volume 15 (6 gaps)

| From | To | Gap Size |
|------|-----|----------|
| AMONG THE ROMANS PROFESSION | ASSERTORIS | 6 |
| ASSERTORIS | AZD | 7 |
| END OF THE FIFTEENTH VOLUME | EXPERIMENTS ON THIS SUBJECT AR | 10 |
| RAMUS | ROWLEY RAGG | 14 |
| SEC ORATORY | SOAMES | 10 |
| THUS MAY THE CHIEF CIRCUMSTANC | TOGA PRETEXTA | 7 |

#### Volume 16 (2 gaps)

| From | To | Gap Size |
|------|-----|----------|
| RIVINIA | ROAD | 6 |
| ROYSTON | RUBENS | 6 |

#### Volume 17 (2 gaps)

| From | To | Gap Size |
|------|-----|----------|
| OCULAR SPECTRA | OS SPHENOIDES | 16 |
| PAPILIO | PROBLEM | 17 |

#### Volume 18 (10 gaps)

| From | To | Gap Size |
|------|-----|----------|
| SARCOCELE | SPHERICAL TRIGONOMETRY | 15 |
| TITUS VESPASIANUS | TOAD | 6 |
| UDDER | UKRAINE | 7 |
| UTRICULARIA | UZ | 6 |
| VIVIPAROUS | VOCATIVE | 6 |
| VOWEL | VULCAN | 6 |
| WITZENBERG | WOAD | 6 |
| YELLOW | YORKSHIRE | 10 |

*...and 2 more gaps*

---

## Recommendations

### High Priority Fixes:

1. **Sentence Fragment Headwords:** Review and correct 104+ articles where sentence 
   fragments were incorrectly parsed as headwords. These typically end with "BY" or 
   contain common words like "THE", "OF", "IS".

2. **Volume/Section Markers:** Remove or reclassify 14 "END OF THE X VOLUME" entries 
   and 28 "PROPOSITION/CLASS/RULE" section markers that should not be independent articles.

3. **Out-of-Range Articles:** Investigate 325 articles appearing in volumes outside 
   their alphabetical range - these may need to be moved or their headwords corrected.

### Medium Priority:

4. **Very Short Articles:** Review 37 articles with fewer than 10 words to determine 
   if they are cross-references, OCR fragments, or legitimate brief definitions.

5. **Botanical Headers:** Verify that 12 Linnaean classification headers 
   (DIANDRIA MONOGYNYA, etc.) are properly categorized.

### Low Priority:

6. **Alphabetical Gaps:** Some gaps are natural (e.g., no English words start with 
   certain letter combinations), but unusually large gaps may warrant investigation.

---

## Appendix: Volume Structure

| Volume | Range | Articles | Treatises |
|--------|-------|----------|-----------|
| 1 | A-ANG | 1,408 | 74 |
| 2 | ANG-BAR | 1,448 | 64 |
| 3 | BAR-BZO | 1,509 | 102 |
| 4 | CAA-CIC | 1,253 | 74 |
| 5 | CIC-DIA | 1,584 | 100 |
| 6 | DIA-ETH | 922 | 58 |
| 7 | ETM-GOA | 1,112 | 119 |
| 8 | GOB-HYD | 1,013 | 87 |
| 9 | Hydrostatics-LES | 1,119 | 76 |
| 10 | LES-MEC | 832 | 76 |
| 11 | Medals-Midwifery | 275 | 51 |
| 12 | MEI-NEG | 590 | 81 |
| 13 | NEH-PAS | 893 | 81 |
| 14 | PAS-PLA | 611 | 66 |
| 15 | PLA-RAM | 739 | 80 |
| 16 | RAN-SCO | 768 | 61 |
| 17 | SCO-STR | 784 | 80 |
| 18 | STR-ZYM | 1,156 | 101 |

---

*Report generated by automated analysis. Manual verification recommended for critical issues.*
