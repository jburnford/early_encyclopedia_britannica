# Swallowed Article Detections

**Date:** 2026-04-02
**Method:** Consecutive paragraph embedding similarity (voyage-4-large)
**Thresholds:** per-category (mid_word/mid_sentence: all, new_headword: <0.35, person_bio/topic_change: <0.20)
**Total detections:** 23088 (2201 multi-edition, 20887 single-edition)

## Summary by Classification

| Type | Count | Threshold | Description |
|------|-------|-----------|-------------|
| mid_word | 251 | all | Starts mid-word — definitely swallowed |
| mid_sentence | 1196 | all | Starts mid-sentence — likely swallowed |
| new_headword | 5189 | <0.35 | ALLCAPS heading — missed headword |
| person_bio | 1291 | <0.20 | Person name — swallowed biography |
| topic_change | 15161 | <0.20 | Topic change — may be legitimate |

## Cross-Edition Breaks (HIGH CONFIDENCE)

**752 unique breaks** appearing in 2+ editions (2201 total detections):

- **BLOOD** → AVENGER (7 editions: 1778, 1797, 1810, 1815, 1823, 1842, 1860) best sim=0.100 [topic_change]
- **ADAM** → ADAM (5 editions: 1778, 1797, 1823, 1842, 1860) best sim=0.113 [topic_change]
- **BRITAIN** → THIS (5 editions: 1778, 1797, 1810, 1815, 1823) best sim=0.137 [topic_change]
- **BRITAIN** → THE (5 editions: 1778, 1810, 1815, 1823, 1860) best sim=0.139 [topic_change]
- **BRITAIN** → DOMESTIC (5 editions: 1797, 1815, 1823, 1842, 1860) best sim=0.163 [topic_change]
- **CARPI** → CARPI (5 editions: 1797, 1810, 1815, 1823, 1842) best sim=0.111 [topic_change]
- **CEYLON** → CHACE (5 editions: 1778, 1797, 1810, 1815, 1823) best sim=0.044 [new_headword]
- **EGYPT** → THE (5 editions: 1797, 1810, 1815, 1823, 1842) best sim=0.137 [topic_change]
- **ENGLAND** → SINCE (5 editions: 1778, 1797, 1810, 1815, 1823) best sim=0.146 [topic_change]
- **KING** → KING (5 editions: 1771, 1778, 1797, 1810, 1842) best sim=0.154 [person_bio]
- **ROME** → WHILE (5 editions: 1778, 1797, 1810, 1815, 1842) best sim=0.178 [topic_change]
- **ADRIAN** → ANIMULA (4 editions: 1810, 1815, 1823, 1842) best sim=0.125 [topic_change]
- **ADRIAN** → POPE (4 editions: 1810, 1815, 1823, 1842) best sim=0.275 [new_headword]
- **AGRICULTURE** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.155 [topic_change]
- **AMERICA** → THE (4 editions: 1797, 1815, 1823, 1842) best sim=0.130 [topic_change]
- **AMPLIFICATION** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.131 [topic_change]
- **ANDREW** → THE (4 editions: 1771, 1810, 1842, 1860) best sim=0.094 [topic_change]
- **ANDREW'S** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.167 [topic_change]
- **ANGELICS** → THE (4 editions: 1771, 1797, 1810, 1860) best sim=0.083 [topic_change]
- **ANTIMONY** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.161 [topic_change]
- **ARBA** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.169 [topic_change]
- **ARSURA** → THE (4 editions: 1810, 1815, 1823, 1860) best sim=0.105 [topic_change]
- **ARUNDELIAN MARBLES** → III (4 editions: 1797, 1810, 1815, 1823) best sim=0.286 [new_headword]
- **BARON** → ROBERT (4 editions: 1810, 1815, 1823, 1842) best sim=0.175 [new_headword]
- **BATH** → KNIGHTS (4 editions: 1797, 1810, 1823, 1860) best sim=0.048 [topic_change]
- **BOSCOVICH** → AFTER (4 editions: 1810, 1815, 1823, 1860) best sim=0.149 [topic_change]
- **BOXING** → BOXING (4 editions: 1797, 1810, 1815, 1823) best sim=0.173 [topic_change]
- **BRADFORD** → JOHN (4 editions: 1810, 1815, 1823, 1860) best sim=0.165 [topic_change]
- **BRITAIN** → TRANQUILLITY (4 editions: 1778, 1797, 1815, 1823) best sim=0.157 [topic_change]
- **BRITAIN** → HAVING (4 editions: 1797, 1810, 1815, 1823) best sim=0.182 [topic_change]
- **BRITAIN** → DURING (4 editions: 1810, 1815, 1823, 1860) best sim=0.160 [topic_change]
- **BRITAIN** → PARLIAMENT (4 editions: 1810, 1815, 1823, 1860) best sim=0.180 [topic_change]
- **BROUGH** → BROUGHTON (4 editions: 1797, 1810, 1815, 1823) best sim=0.137 [person_bio]
- **BUXTON** → BUXTON (4 editions: 1797, 1810, 1815, 1823) best sim=0.170 [person_bio]
- **CANDIA** → CANDIA (4 editions: 1797, 1810, 1815, 1823) best sim=0.179 [topic_change]
- **CHRONICLE** → VIII (4 editions: 1797, 1810, 1815, 1823) best sim=0.282 [new_headword]
- **CHURCHILL** → CHURCHILL (4 editions: 1797, 1810, 1815, 1823) best sim=0.137 [topic_change]
- **COLCHESTER** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.061 [topic_change]
- **CONNECTICUT** → CONNECTICUT (4 editions: 1797, 1810, 1815, 1823) best sim=0.108 [topic_change]
- **COOK** → FROM (4 editions: 1797, 1810, 1815, 1823) best sim=0.109 [topic_change]
- **CROSS** → ENGLISH (4 editions: 1778, 1810, 1815, 1823) best sim=0.119 [topic_change]
- **DELAWARE** → THE DUTCH (4 editions: 1797, 1810, 1815, 1823) best sim=0.122 [person_bio]
- **DICTIONARY** → GREAT (4 editions: 1810, 1815, 1842, 1860) best sim=0.290 [new_headword]
- **EGYPT** → THUS (4 editions: 1797, 1810, 1815, 1823) best sim=0.147 [topic_change]
- **EGYPT** → NOTWITHSTANDING (4 editions: 1797, 1810, 1815, 1823) best sim=0.165 [topic_change]
- **EXCUBIAE** → LETTERS OF EXCULPATION (4 editions: 1797, 1810, 1815, 1823) best sim=0.287 [new_headword]
- **GARDENING** → III (4 editions: 1797, 1810, 1815, 1823) best sim=0.286 [new_headword]
- **GEORGE** → GEORGE (4 editions: 1797, 1810, 1823, 1842) best sim=0.109 [topic_change]
- **GERMANY** → THE REFORMATION (4 editions: 1797, 1810, 1815, 1823) best sim=0.151 [person_bio]
- **GUINEA** → NEW GUINEA (4 editions: 1797, 1815, 1823, 1842) best sim=0.152 [person_bio]
- **HANOVER** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.170 [topic_change]
- **HOLLAND** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.163 [topic_change]
- **IRELAND** → MATTERS (4 editions: 1797, 1810, 1815, 1823) best sim=0.154 [topic_change]
- **IRELAND** → SOON (4 editions: 1797, 1810, 1815, 1823) best sim=0.172 [topic_change]
- **JAMES** → JAMESONE (4 editions: 1797, 1810, 1815, 1823) best sim=0.117 [person_bio]
- **LANDEN** → JOHN (4 editions: 1810, 1815, 1823, 1842) best sim=0.163 [topic_change]
- **LEO** → LEO (4 editions: 1778, 1797, 1810, 1815) best sim=0.060 [topic_change]
- **LEWIS** → LOUIS (4 editions: 1810, 1815, 1823, 1842) best sim=0.158 [topic_change]
- **LIMASSOL** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.149 [topic_change]
- **LONDON** → III (4 editions: 1797, 1810, 1815, 1823) best sim=0.229 [new_headword]
- **LULA** → ULLI (4 editions: 1778, 1797, 1810, 1823) best sim=0.191 [new_headword]
- **MAHOMETANISM** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.183 [topic_change]
- **MALTA** → THE (4 editions: 1778, 1810, 1815, 1823) best sim=0.193 [topic_change]
- **MUTUNUS** → MUZZLE (4 editions: 1797, 1810, 1815, 1823) best sim=0.297 [new_headword]
- **OU-POEY-TSE** → GREATER OUSE (4 editions: 1797, 1810, 1815, 1823) best sim=0.146 [new_headword]
- **PAL** → PAL (4 editions: 1778, 1810, 1815, 1842) best sim=0.300 [new_headword]
- **PHILIPPI** → THE (4 editions: 1778, 1797, 1810, 1815) best sim=0.165 [topic_change]
- **PHILOSOPHY** → HAD (4 editions: 1797, 1810, 1823, 1842) best sim=0.068 [topic_change]
- **PROJECTILES** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.062 [topic_change]
- **RAIN** → RAIN (4 editions: 1797, 1810, 1815, 1823) best sim=0.130 [topic_change]
- **REFLECTION** → CIRCULAR INSTRUMENT (4 editions: 1797, 1815, 1823, 1842) best sim=0.154 [person_bio]
- **SAURIN** → SAURIN (4 editions: 1797, 1810, 1815, 1823) best sim=0.191 [person_bio]
- **SAVIOUR** → SAUL (4 editions: 1797, 1810, 1815, 1823) best sim=0.265 [new_headword]
- **SCOT** → SCOT (4 editions: 1778, 1797, 1810, 1815) best sim=0.109 [person_bio]
- **SCOTLAND** → THIS (4 editions: 1778, 1810, 1815, 1823) best sim=0.162 [topic_change]
- **SHORE** → JANE (4 editions: 1810, 1815, 1823, 1842) best sim=0.159 [topic_change]
- **SPAIN** → CHARLES (4 editions: 1797, 1810, 1815, 1823) best sim=0.156 [topic_change]
- **STALE** → ANIMATED STALK (4 editions: 1797, 1810, 1815, 1823) best sim=0.267 [new_headword]
- **STIRLING** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.043 [topic_change]
- **THEATRE** → NOT (4 editions: 1797, 1810, 1815, 1823) best sim=0.176 [topic_change]
- **TIDE** → THE (4 editions: 1797, 1810, 1815, 1823) best sim=0.082 [topic_change]
- **YEAR** → NEW YEAR (4 editions: 1797, 1810, 1815, 1823) best sim=0.103 [person_bio]
- **ABAS** → SCHAH (3 editions: 1810, 1815, 1823) best sim=0.152 [topic_change]
- **ABEL** → ABEL-K (3 editions: 1810, 1815, 1823) best sim=0.290 [new_headword]
- **ABYSSINIA** → THE ABYSSINIANS (3 editions: 1810, 1815, 1823) best sim=0.169 [person_bio]
- **ACADEMY** → VIII (3 editions: 1778, 1842, 1860) best sim=0.322 [new_headword]
- **ACADEMY** → III (3 editions: 1823, 1842, 1860) best sim=0.314 [new_headword]
- **ALLEGRO** → ALLEIN (3 editions: 1810, 1815, 1823) best sim=0.046 [person_bio]
- **ALPS** → LOWER ALPS (3 editions: 1810, 1815, 1823) best sim=0.150 [person_bio]
- **AMERICA** → NOTWITHSTANDING (3 editions: 1797, 1815, 1823) best sim=0.158 [topic_change]
- **AMERICA** → BUT (3 editions: 1797, 1815, 1823) best sim=0.160 [topic_change]
- **AMMON** → AMMONIUS (3 editions: 1810, 1815, 1823) best sim=0.180 [person_bio]
- **ANDREW'S** → ANDREWS (3 editions: 1797, 1810, 1815) best sim=0.079 [person_bio]
- **ARCH** → THE (3 editions: 1771, 1797, 1842) best sim=0.178 [topic_change]
- **ARREOYS** → THE (3 editions: 1823, 1842, 1860) best sim=0.070 [topic_change]
- **ARTEMISIA** → MUGWORT (3 editions: 1810, 1815, 1823) best sim=0.127 [person_bio]
- **ASIA** → FROM (3 editions: 1810, 1815, 1823) best sim=0.158 [topic_change]
- **ASTRONOMY** → HALLEY (3 editions: 1797, 1815, 1823) best sim=0.129 [new_headword]
- **AUSTRALASIA** → III (3 editions: 1823, 1842, 1860) best sim=0.208 [new_headword]
- **BACON** → ROGER (3 editions: 1810, 1815, 1823) best sim=0.175 [topic_change]
- **BOL** → BOKHARIA (3 editions: 1797, 1815, 1823) best sim=0.101 [new_headword]
- **BOURBON** → BOURBON (3 editions: 1778, 1797, 1860) best sim=0.125 [topic_change]
- **BOURGES** → BOUGET (3 editions: 1810, 1815, 1823) best sim=0.283 [new_headword]
- **BOW** → BOWS (3 editions: 1810, 1815, 1823) best sim=0.190 [topic_change]
- **BRANCHON** → BRAND S (3 editions: 1810, 1815, 1823) best sim=0.151 [new_headword]
- **BRITAIN** → LORD TEMPLE (3 editions: 1810, 1815, 1823) best sim=0.170 [person_bio]
- **BRITAIN** → ONE (3 editions: 1810, 1815, 1823) best sim=0.179 [topic_change]
- **BRITAIN** → EXCEPTING (3 editions: 1810, 1815, 1823) best sim=0.187 [topic_change]
- **BRITAIN** → ALL (3 editions: 1810, 1815, 1823) best sim=0.189 [topic_change]
- **BUCKINGHAM** → GEORGE VILLIERS (3 editions: 1810, 1815, 1823) best sim=0.175 [person_bio]
- **BURTON** → ROBERT (3 editions: 1810, 1815, 1823) best sim=0.171 [topic_change]
- **BYNG** → FROM (3 editions: 1797, 1810, 1823) best sim=0.131 [topic_change]
- **CAIRNS** → QUAMQUAM (3 editions: 1797, 1810, 1815) best sim=0.096 [topic_change]
- **CANINE** → CANINI (3 editions: 1810, 1815, 1823) best sim=0.184 [person_bio]
- **CARPENTRY** → III (3 editions: 1823, 1842, 1860) best sim=0.310 [new_headword]
- **CHEMISTRY** → III (3 editions: 1810, 1815, 1823) best sim=0.289 [new_headword]
- **CHIVALRY** → THE (3 editions: 1823, 1842, 1860) best sim=0.081 [topic_change]
- **CHIVALRY** → SOMETIMES (3 editions: 1823, 1842, 1860) best sim=0.145 [topic_change]
- **CIPHER** → ORDER (3 editions: 1810, 1815, 1823) best sim=0.077 [topic_change]
- **COMITIA** → THE (3 editions: 1771, 1810, 1860) best sim=0.074 [topic_change]
- **CONNECTICUT** → ABOUT (3 editions: 1797, 1810, 1815) best sim=0.166 [topic_change]
- **CONNECTICUT** → THE (3 editions: 1810, 1823, 1860) best sim=0.159 [topic_change]
- **CONQUEST** → COR (3 editions: 1815, 1823, 1860) best sim=0.046 [topic_change]
- **COOPER** → ANTHONY ASHLEY (3 editions: 1810, 1815, 1823) best sim=0.014 [person_bio]
- **COS** → WHETSTONE (3 editions: 1810, 1815, 1823) best sim=0.136 [topic_change]
- **CUSTOM-HOUSE** → CUT (3 editions: 1778, 1810, 1823) best sim=0.151 [topic_change]
- **DEGRADATION** → PAINTING (3 editions: 1810, 1823, 1842) best sim=0.118 [topic_change]
- **DELAWARE** → UNDER (3 editions: 1810, 1815, 1823) best sim=0.136 [topic_change]
- **DRAKE** → DRAKE (3 editions: 1778, 1815, 1823) best sim=0.123 [topic_change]
- **DUCTILITY** → DUDLEY (3 editions: 1810, 1815, 1823) best sim=0.142 [person_bio]
- **EDOM** → EDMUND I (3 editions: 1797, 1815, 1823) best sim=0.194 [new_headword]
- **EGYPT** → FOR (3 editions: 1810, 1815, 1823) best sim=0.123 [topic_change]
- **ELECTRICITY** → SINCE (3 editions: 1778, 1797, 1860) best sim=0.133 [topic_change]
- **ETOLIA** → THE (3 editions: 1810, 1815, 1823) best sim=0.006 [topic_change]
- **ETON** → THE (3 editions: 1810, 1815, 1823) best sim=0.061 [topic_change]
- **EUROPE** → INDEX (3 editions: 1810, 1815, 1823) best sim=0.328 [new_headword]
- **FACTION** → FACTITIOUS (3 editions: 1810, 1815, 1823) best sim=0.140 [topic_change]
- **FAIR ISLE** → FAIR (3 editions: 1810, 1815, 1823) best sim=0.132 [topic_change]
- **FALCONER** → WILLIAM (3 editions: 1810, 1815, 1823) best sim=0.074 [topic_change]
- **FEZZAN** → FEWEL (3 editions: 1810, 1815, 1823) best sim=0.233 [new_headword]
- **FIRE** → FIRE-EATER. (3 editions: 1810, 1815, 1823) best sim=0.187 [topic_change]
- **FLAMSTEED** → JOHN (3 editions: 1810, 1823, 1842) best sim=0.185 [topic_change]
- **FOOD** → THE (3 editions: 1823, 1842, 1860) best sim=0.163 [topic_change]
- **FRANCE** → THE (3 editions: 1810, 1815, 1823) best sim=0.053 [topic_change]
- **FRANCE** → AFTER (3 editions: 1810, 1815, 1823) best sim=0.168 [topic_change]
- **FRANCE** → WHILE FRANCE (3 editions: 1810, 1815, 1823) best sim=0.179 [person_bio]
- **FREDERICK II** → FREDERICK (3 editions: 1810, 1815, 1823) best sim=0.114 [topic_change]
- **GEOGRAPHY** → WITH (3 editions: 1810, 1815, 1823) best sim=0.159 [topic_change]
- **GEORGETOWN** → COR (3 editions: 1810, 1815, 1823) best sim=0.081 [topic_change]
- **GERMANY** → GEORGE I (3 editions: 1810, 1815, 1823) best sim=0.132 [new_headword]
- **GOTHS** → END OF THE NINTH VOLUME (3 editions: 1810, 1815, 1823) best sim=0.067 [new_headword]
- **HAMILTON** → VIVE (3 editions: 1797, 1810, 1823) best sim=0.170 [topic_change]
- **HERALDRY** → ART (3 editions: 1810, 1815, 1823) best sim=0.308 [new_headword]
- **HERODOTUS** → WHAT (3 editions: 1810, 1815, 1823) best sim=0.122 [topic_change]
- **HISTORY** → HOWEVER (3 editions: 1797, 1810, 1823) best sim=0.111 [topic_change]
- **HUNTER** → HUNTER (3 editions: 1810, 1823, 1842) best sim=0.110 [person_bio]
- **ISIS** → THAMES (3 editions: 1810, 1815, 1823) best sim=0.139 [topic_change]
- **KIOF** → KIPPIS (3 editions: 1810, 1815, 1823) best sim=0.154 [person_bio]
- **LAURA** → POET LAUREATE (3 editions: 1810, 1815, 1823) best sim=0.178 [new_headword]
- **LEE** → NATHANIEL (3 editions: 1810, 1823, 1842) best sim=0.147 [new_headword]
- **LIFE** → LIFE (3 editions: 1810, 1815, 1823) best sim=0.118 [topic_change]
- **LIMA** → COFFEE (3 editions: 1810, 1815, 1823) best sim=0.179 [topic_change]
- **LOGARITHMS** → SECT (3 editions: 1810, 1815, 1823) best sim=0.307 [new_headword]
- **LULA** → ULLY (3 editions: 1778, 1797, 1815) best sim=0.294 [new_headword]
- **MADURA** → MEANDER (3 editions: 1778, 1823, 1860) best sim=0.257 [new_headword]
- **MANCHESTER** → THE (3 editions: 1823, 1842, 1860) best sim=0.140 [topic_change]
- **MANSFELD** → PETER ERNEST (3 editions: 1810, 1815, 1823) best sim=0.174 [person_bio]
- **METAPHYSICS** → PART II (3 editions: 1810, 1815, 1823) best sim=0.231 [new_headword]
- **MONEY** → XIII (3 editions: 1810, 1815, 1823) best sim=0.207 [new_headword]
- **NAPLES** → THE (3 editions: 1797, 1810, 1815) best sim=0.175 [topic_change]
- **NOSTRADAMUS** → NOSTRA (3 editions: 1797, 1815, 1823) best sim=0.068 [topic_change]
- **OLYMPIA** → ANCIENT GEOGRAPHY (3 editions: 1810, 1815, 1823) best sim=0.133 [person_bio]
- **PAISLEY** → THE (3 editions: 1810, 1815, 1823) best sim=0.156 [topic_change]
- **PARENT** → PARENT (3 editions: 1797, 1810, 1823) best sim=0.142 [topic_change]
- **PARIS** → PARIS (3 editions: 1797, 1810, 1823) best sim=0.104 [topic_change]
- **PARTHENIUM** → END OF THE FIFTEENTH VOLUME (3 editions: 1810, 1815, 1823) best sim=0.189 [new_headword]
- **PATRICK** → PATRICK (3 editions: 1797, 1810, 1842) best sim=0.164 [person_bio]
- **PETER** → PETER-PENCE (3 editions: 1797, 1810, 1815) best sim=0.117 [topic_change]
- **PIABUCU** → PIACENZA (3 editions: 1797, 1810, 1823) best sim=0.150 [new_headword]
- **PICRIUM** → PICTLAND (3 editions: 1810, 1815, 1823) best sim=0.129 [topic_change]
- **PIVAT** → PIUS II (3 editions: 1797, 1815, 1823) best sim=0.043 [new_headword]
- **PIVAT** → PIUS (3 editions: 1797, 1815, 1823) best sim=0.124 [topic_change]
- **POCOCKE** → THE (3 editions: 1797, 1810, 1860) best sim=0.042 [topic_change]
- **POETRY** → SOMETIMES (3 editions: 1797, 1810, 1815) best sim=0.164 [topic_change]
- **POLE** → ASTRONOMY (3 editions: 1810, 1815, 1823) best sim=0.168 [topic_change]
- **PRINTING** → STEREOTYPE PRINTING (3 editions: 1810, 1815, 1823) best sim=0.125 [person_bio]
- **PRINTING** → THE (3 editions: 1810, 1842, 1860) best sim=0.170 [topic_change]
- **PROCESS** → CHEMISTRY (3 editions: 1810, 1815, 1823) best sim=0.191 [topic_change]
- **PULSE** → BOTANY (3 editions: 1810, 1815, 1823) best sim=0.144 [topic_change]
- **REMPHAN** → ACTION OF REMOVING (3 editions: 1810, 1815, 1823) best sim=0.164 [new_headword]
- **REVERIE** → REVERSAL (3 editions: 1778, 1810, 1815) best sim=0.102 [topic_change]
- **RHIZOBALUS** → END OF THE SEVENTEENTH VOLUME (3 editions: 1810, 1815, 1823) best sim=0.221 [new_headword]
- **ROOKE** → THE (3 editions: 1810, 1823, 1860) best sim=0.072 [topic_change]
- **RUSSIA** → THE (3 editions: 1815, 1823, 1842) best sim=0.143 [topic_change]
- **RYMER** → RYCHOPS (3 editions: 1797, 1810, 1823) best sim=0.133 [new_headword]
- **SABLE** → CAPE (3 editions: 1810, 1815, 1823) best sim=0.144 [topic_change]
- **SANCTORIUS** → SANCTUARY (3 editions: 1797, 1810, 1842) best sim=0.186 [topic_change]
- **SCIENCE** → FIX (3 editions: 1810, 1815, 1823) best sim=0.150 [topic_change]
- **SCOTLAND** → THE (3 editions: 1810, 1815, 1842) best sim=0.094 [topic_change]
- **SENNAAR** → SENNERTUS (3 editions: 1797, 1810, 1823) best sim=0.124 [person_bio]
- **SIGN** → NAVAL SIGNALS (3 editions: 1797, 1815, 1823) best sim=0.202 [new_headword]
- **SIMONIDES** → THERE (3 editions: 1797, 1810, 1815) best sim=0.151 [topic_change]
- **SLEEP** → THAT (3 editions: 1810, 1815, 1823) best sim=0.120 [topic_change]
- **SLEEP** → THE (3 editions: 1810, 1815, 1823) best sim=0.052 [topic_change]
- **SMYRNA** → THE (3 editions: 1810, 1815, 1823) best sim=0.175 [topic_change]
- **SPAIN** → WHEN (3 editions: 1810, 1815, 1823) best sim=0.193 [topic_change]
- **STEWARD** → COURT (3 editions: 1810, 1815, 1823) best sim=0.197 [topic_change]
- **TAYLOR** → TAYLOR (3 editions: 1797, 1810, 1823) best sim=0.178 [topic_change]
- **TRENT** → COUNCIL (3 editions: 1810, 1815, 1823) best sim=0.121 [topic_change]
- **TROUBADOURS** → TROUGH (3 editions: 1810, 1815, 1823) best sim=0.195 [new_headword]
- **VEGETABLE PHYSIOLOGY** → EXPLANATION OF PLATES DXLI (3 editions: 1810, 1815, 1823) best sim=0.214 [new_headword]
- **WATER-WORKS** → THE (3 editions: 1810, 1815, 1823) best sim=0.170 [topic_change]
- **WILLIAM** → SWEET (3 editions: 1797, 1810, 1815) best sim=0.120 [topic_change]
- **WORMS** → WORMS (3 editions: 1810, 1815, 1823) best sim=0.168 [topic_change]
- **ABA** → ABAS (2 editions: 1797, 1810) best sim=0.120 [person_bio]
- **ABORIGINES** → THE ABORIGINES (2 editions: 1797, 1815) best sim=0.174 [person_bio]
- **ABYSSINIA** → THE (2 editions: 1810, 1823) best sim=0.173 [topic_change]
- **ACT** → DRAMATIC POETRY (2 editions: 1810, 1860) best sim=0.180 [person_bio]
- **AGON** → AGON (2 editions: 1842, 1860) best sim=0.107 [topic_change]
- **AGRICOLA** → GEORGE (2 editions: 1810, 1823) best sim=0.123 [topic_change]
- **AGRICULTURE** → PART II (2 editions: 1778, 1797) best sim=0.282 [new_headword]
- **AGRICULTURE** → FIG (2 editions: 1797, 1815) best sim=0.122 [topic_change]
- **AGRICULTURE** → REGULUS (2 editions: 1797, 1815) best sim=0.129 [topic_change]
- **AGRICULTURE** → DURING (2 editions: 1797, 1815) best sim=0.129 [topic_change]
- **AGRICULTURE** → AFTER (2 editions: 1797, 1815) best sim=0.127 [topic_change]
- **AGRICULTURE** → ABOUT (2 editions: 1797, 1815) best sim=0.160 [topic_change]
- **AGRICULTURE** → THE SHEATH (2 editions: 1797, 1815) best sim=0.085 [person_bio]
- **AGRICULTURE** → FROM (2 editions: 1797, 1815) best sim=0.185 [topic_change]
- **AGRICULTURE** → SECT (2 editions: 1810, 1823) best sim=0.164 [topic_change]
- **AGYNIANI** → THE (2 editions: 1797, 1815) best sim=0.140 [topic_change]
- **AHAB** → AHETULA (2 editions: 1797, 1815) best sim=0.061 [new_headword]
- **ALEXANDER** → ALEXANDER (2 editions: 1778, 1797) best sim=0.133 [topic_change]
- **ALEXANDER THE GREAT** → ALEXANDER AB ALEXANDRO (2 editions: 1823, 1842) best sim=0.194 [new_headword]
- **ALLIER** → LET (2 editions: 1823, 1860) best sim=0.128 [topic_change]
- **ALMAMON** → BUT (2 editions: 1823, 1842) best sim=0.003 [topic_change]
- **ALMANZA** → HERESY OF ALMARIC (2 editions: 1815, 1823) best sim=0.124 [new_headword]
- **ALTING** → ALTITUDE (2 editions: 1810, 1815) best sim=0.111 [topic_change]
- **AMERICA** → THREE (2 editions: 1815, 1823) best sim=0.142 [topic_change]
- **AMERICA** → DURING (2 editions: 1815, 1823) best sim=0.175 [topic_change]
- **ANDALUSIA** → THE (2 editions: 1797, 1810) best sim=0.115 [topic_change]
- **ANDEUSE** → THE (2 editions: 1815, 1823) best sim=0.109 [topic_change]
- **ANDREA** → THE (2 editions: 1810, 1815) best sim=0.093 [topic_change]
- **ANDREWS** → THE (2 editions: 1842, 1860) best sim=0.144 [topic_change]
- **ANDROGYNES** → THE (2 editions: 1810, 1815) best sim=0.100 [topic_change]
- **ANDROS** → THE (2 editions: 1842, 1860) best sim=0.141 [topic_change]
- **ANECDOTE** → THE (2 editions: 1810, 1823) best sim=0.153 [topic_change]
- **ANEMOMETER** → THE (2 editions: 1810, 1823) best sim=0.120 [topic_change]
- **ANEMOSCOPE** → THE (2 editions: 1797, 1810) best sim=0.157 [topic_change]
- **ANGAZYA** → ANGIOTOMY (2 editions: 1797, 1810) best sim=0.114 [new_headword]
- **ANGEL** → THE (2 editions: 1815, 1860) best sim=0.155 [topic_change]
- **ANGLESEA** → THE (2 editions: 1842, 1860) best sim=0.069 [topic_change]
- **ANGLING** → THE (2 editions: 1823, 1860) best sim=0.183 [topic_change]
- **ANGOLA** → EXPLANATION OF PLATE XXXII (2 editions: 1810, 1815) best sim=0.349 [new_headword]
- **ANGOY** → THE (2 editions: 1810, 1815) best sim=0.122 [topic_change]
- **ANIMAL KINGDOM** → YOU (2 editions: 1842, 1860) best sim=0.071 [topic_change]
- **ANIMAL KINGDOM** → THE (2 editions: 1842, 1860) best sim=0.080 [topic_change]
- **ANJENGO** → THE (2 editions: 1815, 1823) best sim=0.151 [topic_change]
- **ANNAND** → THE (2 editions: 1815, 1823) best sim=0.077 [topic_change]
- **ANNE** → THE (2 editions: 1810, 1823) best sim=0.087 [topic_change]
- **ANNUITIES** → THE (2 editions: 1823, 1860) best sim=0.081 [topic_change]
- **APPLICATION** → APOGGIATURA (2 editions: 1815, 1823) best sim=0.337 [new_headword]
- **ARABIA** → THIS (2 editions: 1797, 1815) best sim=0.153 [topic_change]
- **ARAGON** → THE (2 editions: 1842, 1860) best sim=0.035 [topic_change]
- **ARC** → THE (2 editions: 1842, 1860) best sim=0.044 [topic_change]
- **ARCHITRAVE** → SUPERIOR (2 editions: 1842, 1860) best sim=0.179 [topic_change]
- **ARCHITRAVE** → THE (2 editions: 1842, 1860) best sim=0.103 [topic_change]
- **ARCHYTAS** → THE (2 editions: 1810, 1823) best sim=0.150 [topic_change]
- **ARETHUSA** → THE (2 editions: 1771, 1810) best sim=0.150 [topic_change]
- **ARITHMETIC** → THE (2 editions: 1842, 1860) best sim=0.079 [topic_change]
- **ARKWRIGHT** → WHEN (2 editions: 1823, 1842) best sim=0.131 [topic_change]
- **ARMENIA** → THE (2 editions: 1797, 1815) best sim=0.130 [topic_change]
- **ARTOTYRITES** → ARAU (2 editions: 1815, 1823) best sim=0.146 [new_headword]
- **ARTS** → THE (2 editions: 1842, 1860) best sim=0.180 [topic_change]
- **ASSAYING** → THE (2 editions: 1842, 1860) best sim=0.065 [topic_change]
- **ASTRONOMY** → PART III (2 editions: 1810, 1815) best sim=0.322 [new_headword]
- **ATCHE** → ATCHIEVEMENT (2 editions: 1797, 1810) best sim=0.237 [new_headword]
- **ATHANASIUS** → THE (2 editions: 1810, 1823) best sim=0.051 [topic_change]
- **ATHESIS** → ATHLET (2 editions: 1815, 1823) best sim=0.262 [new_headword]
- **ATLAS** → THE (2 editions: 1778, 1810) best sim=0.099 [topic_change]
- **ATTERBURY** → THE (2 editions: 1778, 1860) best sim=0.098 [topic_change]
- **AUSTRALASIA** → VIII (2 editions: 1823, 1860) best sim=0.280 [new_headword]
- **AVIARY** → AVICENNA (2 editions: 1810, 1823) best sim=0.125 [topic_change]
- **BACKER** → BACK (2 editions: 1797, 1810) best sim=0.085 [topic_change]
- **BALLAN** → BALLAD (2 editions: 1797, 1810) best sim=0.205 [new_headword]
- **BAMFF** → THE (2 editions: 1810, 1823) best sim=0.188 [topic_change]
- **BANQUET** → BANSTICKLE (2 editions: 1778, 1797) best sim=0.123 [topic_change]
- **BARBARY** → BARBATELLI (2 editions: 1810, 1823) best sim=0.183 [person_bio]
- **BAROMETER** → BEFORE (2 editions: 1842, 1860) best sim=0.184 [topic_change]
- **BARONET** → BARONI (2 editions: 1810, 1823) best sim=0.184 [person_bio]
- **BEAN** → BEAN-COD (2 editions: 1842, 1860) best sim=0.173 [topic_change]
- **BECK** → BECK (2 editions: 1810, 1842) best sim=0.182 [person_bio]
- **BECK** → DAVID (2 editions: 1815, 1823) best sim=0.193 [topic_change]
- **BEDFORDSHIRE** → THE (2 editions: 1842, 1860) best sim=0.164 [topic_change]
- **BENNAVENTA** → BENNET (2 editions: 1810, 1823) best sim=0.152 [person_bio]
- **BENTLEY** → WHEN BENTLEY (2 editions: 1842, 1860) best sim=0.167 [person_bio]
- **BERNOULLI** → VIII (2 editions: 1842, 1860) best sim=0.301 [new_headword]
- **BEVERLEY** → JOHN (2 editions: 1842, 1860) best sim=0.123 [topic_change]
- **BIANA** → BIANCHI (2 editions: 1810, 1823) best sim=0.180 [person_bio]
- **BIBLIOGRAPHY** → VII (2 editions: 1842, 1860) best sim=0.200 [new_headword]
- **BLEACHING** → PART II (2 editions: 1810, 1815) best sim=0.298 [new_headword]
- **BLEACHING** → INDEX (2 editions: 1815, 1823) best sim=0.246 [new_headword]
- **BLIND** → THUS (2 editions: 1778, 1823) best sim=0.129 [topic_change]
- **BLOCK** → BLOCK (2 editions: 1797, 1810) best sim=0.174 [person_bio]
- **BLOCK** → DANIEL (2 editions: 1815, 1823) best sim=0.060 [topic_change]
- **BONA** → BONA DEA (2 editions: 1778, 1842) best sim=0.172 [person_bio]
- **BOND** → LAW (2 editions: 1810, 1823) best sim=0.171 [topic_change]
- **BONNET** → BONNEVAL (2 editions: 1810, 1823) best sim=0.090 [person_bio]
- **BOTANY** → NATURAL CLASSIFICATION OF PLANTS (2 editions: 1823, 1842) best sim=0.312 [new_headword]
- **BOYCE** → THE (2 editions: 1842, 1860) best sim=0.065 [topic_change]
- **BRACCIOLINI** → THE (2 editions: 1810, 1842) best sim=0.167 [topic_change]
- **BRAHE** → THE (2 editions: 1815, 1860) best sim=0.124 [topic_change]
- **BRANDY** → THE (2 editions: 1815, 1860) best sim=0.080 [topic_change]
- **BRASS** → ORDER III (2 editions: 1810, 1815) best sim=0.337 [new_headword]
- **BRAZIL** → III (2 editions: 1842, 1860) best sim=0.294 [new_headword]
- **BREDA** → BREDA (2 editions: 1810, 1815) best sim=0.156 [topic_change]
- **BRENT** → BRENT (2 editions: 1810, 1823) best sim=0.176 [topic_change]
- **BRIANCONNOIS** → WINDSOR ALE. (2 editions: 1815, 1823) best sim=0.129 [topic_change]
- **BRIAREUS** → SCURVY-GRASS ALE. (2 editions: 1815, 1823) best sim=0.189 [topic_change]
- **BRIDGE** → EXPLANATION OF THE PLATES (2 editions: 1823, 1860) best sim=0.232 [new_headword]
- **BRILLIANTS** → BRIM (2 editions: 1815, 1823) best sim=0.274 [new_headword]
- **BRITAIN** → FRANCE (2 editions: 1810, 1823) best sim=0.149 [topic_change]
- **BRITAIN** → BUT (2 editions: 1823, 1860) best sim=0.187 [topic_change]
- **BRITAIN** → THESE (2 editions: 1842, 1860) best sim=0.195 [topic_change]
- **BRODERA** → BROKE (2 editions: 1815, 1823) best sim=0.130 [person_bio]
- **BROME** → THE (2 editions: 1842, 1860) best sim=0.089 [topic_change]
- **BROWN** → BROWN (2 editions: 1810, 1815) best sim=0.119 [topic_change]
- **BUC** → BUANEER (2 editions: 1815, 1823) best sim=0.207 [new_headword]
- **BUCEPHALA** → BUCEROS (2 editions: 1810, 1815) best sim=0.084 [topic_change]
- **BULL** → SEE BOS (2 editions: 1810, 1823) best sim=0.072 [person_bio]
- **CABANIS** → CABBAGE. (2 editions: 1842, 1860) best sim=0.069 [topic_change]
- **CALCUTTA** → BEFORE (2 editions: 1815, 1823) best sim=0.123 [topic_change]
- **CALCUTTA** → THE (2 editions: 1815, 1823) best sim=0.174 [topic_change]
- **CALEDONIA** → NEW CALEDONIA (2 editions: 1778, 1797) best sim=0.136 [person_bio]
- **CAMBER** → FRENCH (2 editions: 1810, 1842) best sim=0.152 [topic_change]
- **CARTHAGE** → NEW CARTHAGE (2 editions: 1810, 1823) best sim=0.179 [person_bio]
- **CASSEL** → CASSIA (2 editions: 1815, 1860) best sim=0.125 [topic_change]
- **CATHERINE PARR** → COUNT GREGORY ORLOFF (2 editions: 1815, 1823) best sim=0.176 [person_bio]
- **CATHOLICON** → CATILINE (2 editions: 1810, 1815) best sim=0.089 [person_bio]
- **CEMENT** → CENCHRUS. (2 editions: 1810, 1815) best sim=0.184 [topic_change]
- **CEYLON** → THE (2 editions: 1823, 1860) best sim=0.146 [topic_change]
- **CHAPLET** → CHAPMAN (2 editions: 1810, 1815) best sim=0.147 [person_bio]
- **CHARLOCK** → QUEEN CHARLOTTE'S ISLAND (2 editions: 1778, 1797) best sim=0.163 [new_headword]
- **CHARLOCK** → QUEEN CHARLOTTE (2 editions: 1810, 1815) best sim=0.162 [person_bio]
- **CHEMISTRY** → SECT (2 editions: 1797, 1815) best sim=0.321 [new_headword]
- **CHEMISTRY** → INDEX (2 editions: 1810, 1815) best sim=0.162 [new_headword]
- **CHESS** → METHODS OF GIVING CHECK-MATE (2 editions: 1842, 1860) best sim=0.332 [new_headword]
- **CHESTER** → THE (2 editions: 1842, 1860) best sim=0.102 [topic_change]
- **CHICHESTER** → SECT (2 editions: 1810, 1842) best sim=0.150 [topic_change]
- **CHILD** → BARTHOLOME (2 editions: 1778, 1823) best sim=0.138 [person_bio]
- **CHILD** → BARTHOLINE (2 editions: 1810, 1815) best sim=0.140 [person_bio]
- **CHINA** → WITH (2 editions: 1778, 1810) best sim=0.185 [topic_change]
- **CHRISTIANA** → ONE (2 editions: 1815, 1823) best sim=0.048 [topic_change]
- **CHRISTIANA** → BUT (2 editions: 1815, 1823) best sim=0.066 [topic_change]
- **CHRISTIANA** → ENHARMONIC (2 editions: 1815, 1823) best sim=-0.002 [topic_change]
- **CHRISTIANA** → THE (2 editions: 1815, 1823) best sim=0.020 [topic_change]
- **CHRISTIANA** → SUCH (2 editions: 1815, 1823) best sim=0.075 [topic_change]
- **CHRISTIANA** → ABOUT (2 editions: 1815, 1823) best sim=0.103 [topic_change]
- **CHRISTIANA** → FROM (2 editions: 1815, 1823) best sim=0.111 [topic_change]
- **CHRISTIANA** → HERE (2 editions: 1815, 1823) best sim=0.101 [topic_change]
- **CHRISTIANA** → THE CHRISTIAN (2 editions: 1815, 1823) best sim=0.098 [person_bio]
- **CHRISTIANA** → THESE (2 editions: 1815, 1823) best sim=0.122 [topic_change]
- **CHRISTIANA** → CHRISTIANITY (2 editions: 1815, 1823) best sim=0.147 [topic_change]
- **CHRISTIANA** → THIS (2 editions: 1815, 1823) best sim=0.092 [topic_change]
- **CHRISTIANA** → SIR ISAAC NEWTON (2 editions: 1815, 1823) best sim=0.160 [person_bio]
- **CHRISTIANA** → THOUGH (2 editions: 1815, 1823) best sim=0.169 [topic_change]
- **CHRISTIANA** → BAYLE (2 editions: 1815, 1823) best sim=0.192 [topic_change]
- **CHRISTIANA** → BEFORE (2 editions: 1815, 1823) best sim=0.132 [topic_change]
- **CHRISTINA** → SANTA (2 editions: 1778, 1797) best sim=0.144 [topic_change]
- **CHRONOLOGY** → THE (2 editions: 1778, 1815) best sim=0.115 [topic_change]
- **CHRYSALIS** → THE (2 editions: 1815, 1823) best sim=0.173 [topic_change]
- **CINCTURE** → THE (2 editions: 1810, 1815) best sim=0.040 [topic_change]
- **CINCTURE** → THE CINQUE (2 editions: 1810, 1815) best sim=0.095 [person_bio]
- **CINCTURE** → PRACTITIONERS (2 editions: 1810, 1815) best sim=0.132 [topic_change]
- **CINCTURE** → THE PERUVIAN (2 editions: 1810, 1815) best sim=0.149 [person_bio]
- **CINCTURE** → FACTITIOUS (2 editions: 1810, 1815) best sim=0.152 [topic_change]
- **CINCTURE** → BARK (2 editions: 1810, 1815) best sim=0.152 [topic_change]
- **CINCTURE** → POLYBIUS (2 editions: 1810, 1815) best sim=0.174 [topic_change]
- **CINCTURE** → WHEN (2 editions: 1810, 1815) best sim=0.185 [topic_change]
- **CINCTURE** → WATER (2 editions: 1810, 1815) best sim=0.186 [topic_change]
- **CINCTURE** → BUT (2 editions: 1810, 1815) best sim=0.188 [topic_change]
- **CLACKMANNANSHIRE** → THE (2 editions: 1842, 1860) best sim=0.009 [topic_change]
- **CLYTIA** → VOL (2 editions: 1810, 1815) best sim=0.030 [topic_change]
- **COMPANY** → THE (2 editions: 1815, 1823) best sim=0.126 [topic_change]
- **COMPARISON** → THE (2 editions: 1842, 1860) best sim=0.198 [topic_change]
- **CONCORD** → FORM OF CONCORD (2 editions: 1810, 1815) best sim=0.219 [new_headword]
- **CONJURATION** → PROP (2 editions: 1842, 1860) best sim=0.164 [topic_change]
- **CONNOISSEUR** → COR (2 editions: 1810, 1842) best sim=0.174 [topic_change]
- **CONON** → LET (2 editions: 1810, 1823) best sim=0.113 [topic_change]
- **CONRAD III** → CONRAD (2 editions: 1810, 1815) best sim=0.251 [new_headword]
- **COOPER** → COOPER (2 editions: 1842, 1860) best sim=0.024 [person_bio]
- **COPHTI** → THE COPTS (2 editions: 1797, 1815) best sim=0.156 [person_bio]
- **CORDOVA** → NEW CORDOVA (2 editions: 1797, 1815) best sim=0.039 [person_bio]
- **CORNWALL** → THE (2 editions: 1842, 1860) best sim=0.146 [topic_change]
- **COVENANT** → ARK (2 editions: 1815, 1823) best sim=0.156 [topic_change]
- **CROWN** → CROWNE (2 editions: 1815, 1823) best sim=0.180 [person_bio]
- **DAMASKEENING** → DAMEOPRE (2 editions: 1797, 1810) best sim=0.188 [topic_change]
- **DAPHNE** → DAHNPEPHORIA (2 editions: 1810, 1823) best sim=0.233 [new_headword]
- **DEFORMITY** → AND (2 editions: 1810, 1815) best sim=0.163 [topic_change]
- **DELAWARE** → THE DELAWARE STATE (2 editions: 1797, 1823) best sim=0.193 [person_bio]
- **DESCENT** → DESCHAMPS (2 editions: 1810, 1823) best sim=0.120 [topic_change]
- **DICKINSON** → GREAT (2 editions: 1778, 1797) best sim=0.286 [new_headword]
- **DICKINSON** → TALL (2 editions: 1778, 1797) best sim=0.322 [new_headword]
- **DICTIONARY** → TALL (2 editions: 1810, 1815) best sim=0.322 [new_headword]
- **DICTIONARY** → IMMEDIATELY (2 editions: 1810, 1815) best sim=0.341 [new_headword]
- **DIGGES** → THE (2 editions: 1842, 1860) best sim=0.174 [topic_change]
- **DISCORD** → DISCORD (2 editions: 1797, 1810) best sim=0.193 [topic_change]
- **DISPERSION** → DISPERSION (2 editions: 1797, 1810) best sim=0.117 [topic_change]
- **DISTILLATION** → THIS (2 editions: 1823, 1842) best sim=0.147 [topic_change]
- **DIVISIBILITY** → SINCE (2 editions: 1842, 1860) best sim=0.175 [topic_change]
- **DOIG** → HIS (2 editions: 1842, 1860) best sim=0.055 [topic_change]
- **DRAWBACK** → DRAW (2 editions: 1778, 1810) best sim=0.146 [topic_change]
- **DYNAMICS** → III (2 editions: 1810, 1842) best sim=0.021 [new_headword]
- **DYNAMICS** → THE (2 editions: 1810, 1842) best sim=0.086 [topic_change]
- **DYNAMICS** → THIS (2 editions: 1810, 1842) best sim=0.121 [topic_change]
- **DYNAMICS** → BUT (2 editions: 1810, 1842) best sim=0.162 [topic_change]
- **DYNAMICS** → CHAPTER VI (2 editions: 1842, 1860) best sim=0.294 [new_headword]
- **DYNAMICS** → CHAPTER V (2 editions: 1842, 1860) best sim=0.292 [new_headword]
- **DYRRACHIUM** → DYS (2 editions: 1810, 1815) best sim=0.237 [new_headword]
- **EARL** → EAR (2 editions: 1810, 1815) best sim=0.328 [new_headword]
- **EAVES** → ONE (2 editions: 1815, 1823) best sim=0.114 [topic_change]
- **EBIONITES** → EAVES-D (2 editions: 1815, 1823) best sim=0.163 [new_headword]
- **ECHO** → THE (2 editions: 1778, 1823) best sim=0.183 [topic_change]
- **ECONOMISTS** → III (2 editions: 1842, 1860) best sim=0.241 [new_headword]
- **EGYPT** → SECTION V (2 editions: 1823, 1842) best sim=0.163 [new_headword]
- **ELECTRICITY** → PART VI (2 editions: 1810, 1815) best sim=0.334 [new_headword]
- **ELECTRICITY** → THE (2 editions: 1842, 1860) best sim=-0.045 [topic_change]
- **ELECTRICITY** → WHEN (2 editions: 1842, 1860) best sim=-0.013 [topic_change]
- **ELECTRICITY** → THESE (2 editions: 1842, 1860) best sim=0.056 [topic_change]
- **ELECTRICITY** → AMONG (2 editions: 1842, 1860) best sim=0.010 [topic_change]
- **ELECTRICITY** → FROM (2 editions: 1842, 1860) best sim=0.025 [topic_change]
- **ELECTRICITY** → ALTHOUGH (2 editions: 1842, 1860) best sim=0.124 [topic_change]
- **ELECTRICITY** → SECT (2 editions: 1842, 1860) best sim=-0.020 [topic_change]
- **ELECTRICITY** → THAT (2 editions: 1842, 1860) best sim=0.063 [topic_change]
- **ELECTRICITY** → BUT (2 editions: 1842, 1860) best sim=0.041 [topic_change]
- **ELECTRICITY** → SEVERAL (2 editions: 1842, 1860) best sim=0.031 [topic_change]
- **ELEVATORY** → THE (2 editions: 1842, 1860) best sim=0.114 [topic_change]
- **ELEVATORY** → ELVE (2 editions: 1842, 1860) best sim=0.263 [new_headword]
- **ELPHINSTON** → EDSHEIMER (2 editions: 1815, 1823) best sim=0.139 [new_headword]
- **EMINENCE** → EMINENCE (2 editions: 1810, 1823) best sim=0.190 [topic_change]
- **ENGLAND** → EDWARD (2 editions: 1778, 1823) best sim=0.169 [topic_change]
- **ENGLISH LANGUAGE** → CONTEMPORARY (2 editions: 1842, 1860) best sim=0.112 [topic_change]
- **ENTOMOLOGY** → PENTAMERA (2 editions: 1842, 1860) best sim=0.298 [new_headword]
- **EPHESUS** → THE (2 editions: 1810, 1815) best sim=0.095 [topic_change]
- **EPHYDOR** → EPIBAT (2 editions: 1815, 1823) best sim=0.323 [new_headword]
- **EPICURUS** → WINGS (2 editions: 1810, 1823) best sim=0.139 [topic_change]
- **EPILOBUM** → WINGS (2 editions: 1810, 1823) best sim=0.113 [topic_change]
- **EPIPHANIUS** → EPHONEMA (2 editions: 1815, 1823) best sim=0.241 [new_headword]
- **EPIPHANIUS** → EPHANY (2 editions: 1823, 1842) best sim=0.183 [new_headword]
- **EPISCOPACY** → WINGS (2 editions: 1810, 1823) best sim=0.060 [topic_change]
- **ERPETOLOGY** → EXPLANATION OF THE PLATES (2 editions: 1810, 1815) best sim=0.282 [new_headword]
- **ESCUAGE** → ESCOLAPIUS (2 editions: 1810, 1815) best sim=0.255 [new_headword]
- **ESCURIAL** → THIS (2 editions: 1810, 1823) best sim=0.157 [topic_change]
- **ESDRAS** → THIS (2 editions: 1810, 1815) best sim=0.026 [topic_change]
- **EUPHORBUS** → EUPHORION (2 editions: 1815, 1823) best sim=0.347 [new_headword]
- **EURYSTHEUS** → RIBS (2 editions: 1810, 1815) best sim=0.136 [topic_change]
- **EVANGELISTS** → THE (2 editions: 1810, 1815) best sim=0.120 [topic_change]
- **EVE** → EVELYN (2 editions: 1810, 1815) best sim=0.134 [person_bio]
- **EXPLOSION** → EXPO (2 editions: 1815, 1860) best sim=0.164 [topic_change]
- **FEVERSHAM** → FEBRI (2 editions: 1815, 1823) best sim=0.180 [new_headword]
- **FEZ** → FEWEL (2 editions: 1778, 1797) best sim=0.241 [new_headword]
- **FIBULA** → FICINUS (2 editions: 1810, 1815) best sim=0.146 [person_bio]
- **FIELDING** → ELYSIAN FIELDS (2 editions: 1815, 1823) best sim=0.112 [person_bio]
- **FIFESHIRE** → FIGURE (2 editions: 1810, 1823) best sim=0.196 [topic_change]
- **FINAL** → GEOGRAPHY (2 editions: 1815, 1823) best sim=0.138 [topic_change]
- **FINGAL** → THE (2 editions: 1810, 1823) best sim=0.153 [topic_change]
- **FLAMSTEED** → INFLAMMATION (2 editions: 1810, 1823) best sim=0.022 [topic_change]
- **FLEURY** → THE (2 editions: 1842, 1860) best sim=0.084 [topic_change]
- **FLINT** → FLINTS (2 editions: 1797, 1815) best sim=0.020 [new_headword]
- **FLINT** → SEEING (2 editions: 1797, 1815) best sim=0.093 [topic_change]
- **FLINTSHIRE** → THE (2 editions: 1810, 1823) best sim=0.113 [topic_change]
- **FONTEVRAUD** → EXAMPLE (2 editions: 1810, 1823) best sim=0.034 [topic_change]
- **FOOT** → THE (2 editions: 1815, 1860) best sim=0.048 [topic_change]
- **FORDWICH** → END OF THE EIGHTH VOLUME (2 editions: 1815, 1823) best sim=0.119 [new_headword]
- **FORFAR** → MANY (2 editions: 1823, 1842) best sim=0.176 [topic_change]
- **FRANCE** → WHEN (2 editions: 1823, 1842) best sim=0.144 [topic_change]
- **FRANCE** → BUT (2 editions: 1823, 1842) best sim=0.161 [topic_change]
- **FRANKLIN** → FRANKLIN (2 editions: 1810, 1823) best sim=0.163 [topic_change]
- **FREE** → FREE (2 editions: 1810, 1823) best sim=0.180 [topic_change]
- **GABALE** → THE TURKS (2 editions: 1815, 1823) best sim=0.157 [person_bio]
- **GARTH** → GARUMNA (2 editions: 1797, 1810) best sim=0.119 [topic_change]
- **GASCOIGNE** → THE (2 editions: 1810, 1815) best sim=0.043 [topic_change]
- **GAUGAMELA** → GAUGE- (2 editions: 1815, 1823) best sim=0.211 [new_headword]
- **GAUNTLOPE** → THE GAULS (2 editions: 1815, 1823) best sim=0.170 [person_bio]
- **GED** → ABOUT (2 editions: 1815, 1823) best sim=0.040 [topic_change]
- **GEM** → INDEX TO PART III (2 editions: 1815, 1823) best sim=0.310 [new_headword]
- **GENUS LXXXIII** → ORDER III (2 editions: 1810, 1823) best sim=0.220 [new_headword]
- **GEOMETRY** → III (2 editions: 1842, 1860) best sim=0.298 [new_headword]
- **GEOMETRY** → XVII (2 editions: 1842, 1860) best sim=0.323 [new_headword]
- **GEORGE** → KING GEORGE (2 editions: 1810, 1823) best sim=0.167 [person_bio]
- **GRAMMAR** → ALONE (2 editions: 1815, 1823) best sim=0.346 [new_headword]
- **GRAMMAR** → HYPOCRISY (2 editions: 1842, 1860) best sim=0.082 [topic_change]
- **GRAMMAR** → THAT (2 editions: 1842, 1860) best sim=0.188 [topic_change]
- **GRAY** → LADY JANE (2 editions: 1810, 1815) best sim=0.153 [person_bio]
- **GREECE** → THE (2 editions: 1842, 1860) best sim=0.149 [topic_change]
- **GROVE** → HENRY (2 editions: 1810, 1815) best sim=0.173 [topic_change]
- **GUTTY** → GUY (2 editions: 1823, 1842) best sim=0.086 [person_bio]
- **HAY** → HAYES (2 editions: 1810, 1823) best sim=0.085 [person_bio]
- **HEAT** → ONE (2 editions: 1842, 1860) best sim=0.082 [topic_change]
- **HEBRIDES** → NEW HEBRIDES (2 editions: 1810, 1842) best sim=0.127 [person_bio]
- **HELMINTHOLOGY** → HIRUDO (2 editions: 1815, 1823) best sim=0.320 [new_headword]
- **HENRY** → THE (2 editions: 1823, 1842) best sim=0.089 [topic_change]
- **HENRY** → HENRY (2 editions: 1823, 1842) best sim=0.162 [new_headword]
- **HERCULANEUM** → THE (2 editions: 1823, 1842) best sim=0.168 [topic_change]
- **HERRING** → THOMAS (2 editions: 1815, 1823) best sim=0.128 [topic_change]
- **HILL** → HILL (2 editions: 1810, 1823) best sim=0.054 [person_bio]
- **HIMILCO** → THE (2 editions: 1842, 1860) best sim=0.180 [topic_change]
- **HOLLAND** → NEW HOLLAND (2 editions: 1797, 1810) best sim=0.192 [new_headword]
- **HOUSE** → CHEAP (2 editions: 1797, 1823) best sim=0.195 [topic_change]
- **HOWE** → HOWE (2 editions: 1823, 1860) best sim=0.176 [topic_change]
- **HUDSON** → WILLIAM (2 editions: 1815, 1823) best sim=0.152 [topic_change]
- **HUNTING** → THE MEXICANS (2 editions: 1797, 1815) best sim=0.155 [person_bio]
- **ICE ICE** → BLINK (2 editions: 1815, 1823) best sim=0.186 [topic_change]
- **IDYLLION** → JEARS (2 editions: 1797, 1810) best sim=0.244 [new_headword]
- **IDYLLION** → JEBUS (2 editions: 1797, 1810) best sim=0.260 [new_headword]
- **IGNATIA** → IGNATIUS LOYOLA (2 editions: 1797, 1842) best sim=0.077 [person_bio]
- **ILA** → HISTORY (2 editions: 1797, 1823) best sim=0.124 [topic_change]
- **INFORMER** → INFRINGEMENT (2 editions: 1815, 1823) best sim=0.179 [topic_change]
- **IRELAND** → THE (2 editions: 1842, 1860) best sim=0.074 [topic_change]
- **IVA** → JUAN DE FUCA (2 editions: 1815, 1823) best sim=0.317 [new_headword]
- **JEARS** → JEBUS (2 editions: 1815, 1823) best sim=0.199 [new_headword]
- **JOHN** → JOHN (2 editions: 1797, 1810) best sim=0.192 [topic_change]
- **JUAN FERNANDEZ** → JUBA (2 editions: 1815, 1823) best sim=0.047 [topic_change]
- **JURY** → JUSSICA (2 editions: 1797, 1815) best sim=0.185 [topic_change]
- **KADESHE** → KEMPERIA (2 editions: 1797, 1815) best sim=0.272 [new_headword]
- **KASSON** → KASTRIL (2 editions: 1815, 1823) best sim=0.104 [topic_change]
- **KAZY** → KEATE (2 editions: 1815, 1823) best sim=0.158 [person_bio]
- **KILDARE** → THE (2 editions: 1842, 1860) best sim=0.082 [topic_change]
- **KNOT** → KNOT (2 editions: 1778, 1797) best sim=0.186 [topic_change]
- **LAHOR** → LAINEZ (2 editions: 1810, 1823) best sim=0.054 [person_bio]
- **LANGUEDOC** → WHERE (2 editions: 1810, 1815) best sim=0.076 [topic_change]
- **LANGUET** → TROJAM (2 editions: 1797, 1810) best sim=0.163 [topic_change]
- **LAPLAND** → THE (2 editions: 1797, 1823) best sim=0.102 [topic_change]
- **LAURA** → POET-LAUREATE (2 editions: 1797, 1842) best sim=0.165 [new_headword]
- **LAW** → CUSTOM (2 editions: 1797, 1823) best sim=0.174 [topic_change]
- **LEAGUE ALSO** → LEAK (2 editions: 1815, 1823) best sim=0.077 [topic_change]
- **LEGERDEMAIN** → CUSTOM (2 editions: 1810, 1815) best sim=0.190 [topic_change]
- **LEGERDEMAIN** → MILITARY LAW (2 editions: 1810, 1815) best sim=0.197 [person_bio]
- **LEGHORN** → GIVE (2 editions: 1810, 1823) best sim=0.060 [topic_change]
- **LEITRIM** → THE (2 editions: 1842, 1860) best sim=0.183 [topic_change]
- **LEUCTRA** → LEVEL (2 editions: 1815, 1823) best sim=0.000 [topic_change]
- **LIFE** → VEGETABLE LIFE (2 editions: 1810, 1823) best sim=0.168 [person_bio]
- **LIGAMENT** → LIGARIUS (2 editions: 1810, 1815) best sim=0.158 [person_bio]
- **LIMERICK** → DONALD (2 editions: 1810, 1823) best sim=0.070 [topic_change]
- **LOGIC** → VII (2 editions: 1778, 1815) best sim=0.307 [new_headword]
- **LONDONDERRY** → THE (2 editions: 1842, 1860) best sim=0.094 [topic_change]
- **LONGFORD** → THE (2 editions: 1842, 1860) best sim=0.147 [topic_change]
- **LOUTH** → THE (2 editions: 1842, 1860) best sim=0.163 [topic_change]
- **LUCERNE** → BOTANY (2 editions: 1810, 1823) best sim=0.107 [topic_change]
- **MADAGASCAR** → MADDER (2 editions: 1778, 1797) best sim=0.162 [topic_change]
- **MADURA** → MACENAS (2 editions: 1778, 1842) best sim=0.142 [new_headword]
- **MAELSTROM** → MAFF (2 editions: 1778, 1823) best sim=0.094 [topic_change]
- **MAGIC** → DRYDEN (2 editions: 1810, 1815) best sim=0.252 [new_headword]
- **MAGNETISM** → CLAUDIAN (2 editions: 1810, 1815) best sim=0.146 [new_headword]
- **MAJESTY** → MAIL INDUCTIO (2 editions: 1815, 1823) best sim=0.165 [new_headword]
- **MAJOR** → MAJOR (2 editions: 1810, 1823) best sim=0.174 [person_bio]
- **MAMMALIA** → ORDER I (2 editions: 1842, 1860) best sim=0.277 [new_headword]
- **MAN** → MAN (2 editions: 1810, 1823) best sim=0.173 [topic_change]
- **MANICHEES** → THE (2 editions: 1823, 1842) best sim=0.052 [topic_change]
- **MANILLA** → MANILIUS (2 editions: 1810, 1823) best sim=0.100 [person_bio]
- **MANUAL** → THE (2 editions: 1810, 1823) best sim=0.162 [topic_change]
- **MAROLLES** → THE (2 editions: 1810, 1823) best sim=0.038 [topic_change]
- **MARTIN** → MARTIN (2 editions: 1797, 1810) best sim=0.089 [topic_change]
- **MASON** → WILLIAM (2 editions: 1810, 1815) best sim=0.050 [topic_change]
- **MASQUE** → ARCHITECTURE (2 editions: 1810, 1823) best sim=0.086 [topic_change]
- **MATCHING** → DURA (2 editions: 1810, 1815) best sim=0.226 [new_headword]
- **MATERIA MEDICA** → CLASS VI (2 editions: 1810, 1815) best sim=0.289 [new_headword]
- **MATERIA MEDICA AND PHARMACY** → THE (2 editions: 1815, 1823) best sim=0.136 [topic_change]
- **MAURITIUS** → THE (2 editions: 1842, 1860) best sim=0.146 [topic_change]
- **MAY** → THOMAS (2 editions: 1815, 1860) best sim=0.067 [new_headword]
- **MEATH** → THE (2 editions: 1842, 1860) best sim=0.154 [topic_change]
- **MECHANICS** → PROP (2 editions: 1815, 1823) best sim=0.152 [topic_change]
- **MECHANICS** → THE (2 editions: 1823, 1842) best sim=0.098 [topic_change]
- **MECHANICS** → BUT (2 editions: 1823, 1842) best sim=0.155 [topic_change]
- **MECKLENBURG** → FOR (2 editions: 1797, 1860) best sim=0.063 [topic_change]
- **MEDICAL JURISPRUDENCE** → III (2 editions: 1842, 1860) best sim=0.324 [new_headword]
- **MEDICAL POLICE** → INDEX (2 editions: 1810, 1823) best sim=0.233 [new_headword]
- **MEDICINE** → MADAME NOUFFER (2 editions: 1778, 1810) best sim=0.146 [person_bio]
- **MEDICINE** → ORDER III (2 editions: 1778, 1815) best sim=0.235 [new_headword]
- **MEDICINE** → ORDER II (2 editions: 1778, 1815) best sim=0.262 [new_headword]
- **MEDICINE** → ORDER IV (2 editions: 1778, 1815) best sim=0.319 [new_headword]
- **METAPHYSICS** → SECT (2 editions: 1778, 1823) best sim=0.088 [topic_change]
- **METAPHYSICS** → CHAMONT (2 editions: 1778, 1823) best sim=0.146 [topic_change]
- **METAPHYSICS** → GONZALEZ (2 editions: 1778, 1823) best sim=0.104 [topic_change]
- **METELLUS** → THIS (2 editions: 1810, 1842) best sim=0.064 [topic_change]
- **METEOR** → FROM (2 editions: 1810, 1823) best sim=0.172 [topic_change]
- **MEXICO** → III (2 editions: 1842, 1860) best sim=0.214 [new_headword]
- **MICKLE** → DURING (2 editions: 1810, 1815) best sim=0.133 [topic_change]
- **MICKLE** → THE (2 editions: 1810, 1815) best sim=0.147 [topic_change]
- **MICKLE** → ABOUT (2 editions: 1810, 1815) best sim=0.186 [topic_change]
- **MIDDLESEX** → THE (2 editions: 1842, 1860) best sim=0.088 [topic_change]
- **MINERALOGY** → III (2 editions: 1778, 1815) best sim=0.211 [new_headword]
- **MINERALOGY** → THE (2 editions: 1815, 1842) best sim=0.179 [topic_change]
- **MINOS II** → COLOUR (2 editions: 1810, 1823) best sim=-0.023 [topic_change]
- **MINT** → SEE (2 editions: 1810, 1815) best sim=0.082 [topic_change]
- **MISCHNA** → COLOUR (2 editions: 1815, 1823) best sim=0.155 [topic_change]
- **MOGULS** → SUBSPECIES (2 editions: 1810, 1823) best sim=0.163 [topic_change]
- **MOIVRE** → COLOUR (2 editions: 1810, 1823) best sim=0.124 [topic_change]
- **MONAGHAN** → COLOUR (2 editions: 1815, 1823) best sim=0.151 [topic_change]
- **MONASTEREVAN** → THE (2 editions: 1810, 1815) best sim=-0.018 [topic_change]
- **MONEY** → PLATES CCCLII (2 editions: 1810, 1815) best sim=0.301 [new_headword]
- **MONGOLIA** → BUT (2 editions: 1842, 1860) best sim=0.081 [topic_change]
- **MONK** → MONK (2 editions: 1778, 1823) best sim=-0.001 [topic_change]
- **MORAL PHILOSOPHY** → PART III (2 editions: 1810, 1823) best sim=0.346 [new_headword]
- **MUTIUS** → THE (2 editions: 1823, 1842) best sim=0.111 [topic_change]
- **MYIODES DEUS** → MYL (2 editions: 1810, 1815) best sim=0.239 [new_headword]
- **MYXINE** → THOUGH (2 editions: 1810, 1815) best sim=0.173 [topic_change]
- **NAME** → ACCORDING (2 editions: 1810, 1815) best sim=0.110 [topic_change]
- **NAME** → NAMES (2 editions: 1810, 1815) best sim=0.115 [topic_change]
- **NAME** → THE HINDOOS (2 editions: 1810, 1815) best sim=0.136 [person_bio]
- **NAME** → BUT (2 editions: 1810, 1815) best sim=0.146 [topic_change]
- **NAN-KING** → THE (2 editions: 1810, 1815) best sim=0.137 [topic_change]
- **NANCOWRY** → OANNES (2 editions: 1810, 1815) best sim=0.152 [topic_change]
- **NAPIER** → THE EGYPTIANS (2 editions: 1810, 1815) best sim=0.091 [person_bio]
- **NATURAL HISTORY** → III (2 editions: 1810, 1823) best sim=0.313 [new_headword]
- **NAVEW** → THEORY OF NAVIGATION (2 editions: 1815, 1823) best sim=0.321 [new_headword]
- **NAVIGATION** → THE (2 editions: 1797, 1860) best sim=0.161 [topic_change]
- **NELSON** → THE (2 editions: 1810, 1823) best sim=0.127 [topic_change]
- **NEWTONIAN PHILOSOPHY** → VOL (2 editions: 1815, 1823) best sim=0.088 [topic_change]
- **NUNDOCOMAR** → MONTE NUOVO (2 editions: 1810, 1815) best sim=0.171 [new_headword]
- **OMEN** → DRYDEN (2 editions: 1810, 1823) best sim=0.235 [new_headword]
- **ONYX** → ONALASHKA (2 editions: 1810, 1823) best sim=0.154 [new_headword]
- **OPHIOLOGY** → EXPLANATION OF PLATES CCCLXXI (2 editions: 1810, 1815) best sim=0.292 [new_headword]
- **OPUNTIA** → THE (2 editions: 1797, 1823) best sim=0.108 [topic_change]
- **ORANGE** → ORATION (2 editions: 1810, 1823) best sim=0.129 [topic_change]
- **ORATORY** → THE (2 editions: 1778, 1797) best sim=0.048 [topic_change]
- **ORATORY** → BUT (2 editions: 1797, 1823) best sim=0.126 [topic_change]
- **ORATORY** → PARTICULAR ELOCUTION (2 editions: 1797, 1815) best sim=0.327 [new_headword]
- **ORDERS** → THE (2 editions: 1815, 1860) best sim=0.098 [topic_change]
- **ORVIETO** → ORYZA (2 editions: 1810, 1823) best sim=0.089 [topic_change]
- **OTHO** → VENIUS (2 editions: 1810, 1815) best sim=0.202 [new_headword]
- **PALAMEDES** → PAL (2 editions: 1797, 1823) best sim=0.186 [new_headword]
- **PARADISE** → BIRD (2 editions: 1810, 1823) best sim=0.160 [topic_change]
- **PARR** → SAMUEL (2 editions: 1842, 1860) best sim=0.227 [new_headword]
- **PASSAU** → PASSERAT (2 editions: 1778, 1797) best sim=0.127 [topic_change]
- **PATRICK** → PATRICK SIMON (2 editions: 1815, 1823) best sim=0.131 [person_bio]
- **PEGU** → THE (2 editions: 1778, 1797) best sim=0.168 [topic_change]
- **PENITENTIARY** → PENMAN (2 editions: 1810, 1815) best sim=0.186 [topic_change]
- **PENN** → THE (2 editions: 1842, 1860) best sim=0.111 [topic_change]
- **PERSPECTIVE** → PERSPECTIVE (2 editions: 1797, 1823) best sim=0.179 [topic_change]
- **PHEGOR** → PELLANDRIUM (2 editions: 1797, 1810) best sim=0.078 [new_headword]
- **PHILIP** → PHILIP (2 editions: 1810, 1815) best sim=0.286 [new_headword]
- **PHILOLOGY** → PERHAPS (2 editions: 1815, 1823) best sim=0.138 [topic_change]
- **PIASTUS** → THE (2 editions: 1797, 1810) best sim=0.141 [topic_change]
- **PICRIUM** → PICET (2 editions: 1810, 1823) best sim=0.133 [person_bio]
- **PIGMENTS** → PIGNEROL (2 editions: 1797, 1815) best sim=0.120 [topic_change]
- **PIGMENTS** → PIGNUT (2 editions: 1797, 1815) best sim=0.180 [topic_change]
- **PILE** → THIS (2 editions: 1778, 1797) best sim=0.194 [topic_change]
- **PIMENTO** → THIS (2 editions: 1810, 1815) best sim=0.187 [topic_change]
- **PISMIRES** → PISO (2 editions: 1810, 1815) best sim=0.124 [person_bio]
- **PIVAT** → SUM PIUS (2 editions: 1815, 1823) best sim=0.153 [person_bio]
- **PNEUMATICS** → THERE (2 editions: 1842, 1860) best sim=0.097 [topic_change]
- **POETRY** → THE (2 editions: 1797, 1860) best sim=0.154 [topic_change]
- **POETRY** → NEXT (2 editions: 1810, 1815) best sim=0.180 [topic_change]
- **POLYTHEISM** → THE (2 editions: 1842, 1860) best sim=0.053 [topic_change]
- **POLYXO** → POMACE (2 editions: 1797, 1810) best sim=0.075 [new_headword]
- **POPE** → POPE (2 editions: 1797, 1823) best sim=0.169 [topic_change]
- **POPE** → CHAPTER V (2 editions: 1842, 1860) best sim=0.328 [new_headword]
- **POPULATION** → SWIFT (2 editions: 1842, 1860) best sim=0.077 [topic_change]
- **POPULATION** → BUT (2 editions: 1842, 1860) best sim=0.102 [topic_change]
- **POPULATION** → WHO (2 editions: 1842, 1860) best sim=0.112 [topic_change]
- **POPULATION** → THE (2 editions: 1842, 1860) best sim=0.087 [topic_change]
- **POPULATION** → WHY (2 editions: 1842, 1860) best sim=0.143 [topic_change]
- **POPULATION** → TAKING (2 editions: 1842, 1860) best sim=0.159 [topic_change]
- **PORCH** → TABLE (2 editions: 1810, 1815) best sim=0.184 [topic_change]
- **PORTUGAL** → THE (2 editions: 1823, 1842) best sim=0.155 [topic_change]
- **POTSDAM** → POTT (2 editions: 1797, 1823) best sim=0.160 [person_bio]
- **PRINCIPLE** → THE (2 editions: 1810, 1815) best sim=0.096 [topic_change]
- **PRINTING** → PRINCIPAL RAY (2 editions: 1810, 1815) best sim=0.072 [topic_change]
- **PRINTING** → WHAT (2 editions: 1810, 1815) best sim=0.104 [topic_change]
- **PRINTING** → THIS (2 editions: 1810, 1815) best sim=0.144 [topic_change]
- **PRINTING** → PRINCIPAL POINT (2 editions: 1810, 1815) best sim=0.117 [topic_change]
- **PRINTING** → WHENCE (2 editions: 1810, 1815) best sim=0.174 [topic_change]
- **PRONG-HOE** → PROPOSITION XXII (2 editions: 1797, 1823) best sim=0.236 [new_headword]
- **PRONUNCIATION** → LET (2 editions: 1815, 1823) best sim=0.144 [topic_change]
- **PUBLIUS SYRUS** → OAK PUCERON (2 editions: 1810, 1823) best sim=0.189 [new_headword]
- **PULSE** → PULTENEY (2 editions: 1815, 1823) best sim=-0.044 [person_bio]
- **PUTTY SOMETIMES ALSO** → TERRA PUZZULANA (2 editions: 1810, 1815) best sim=0.321 [new_headword]
- **QUANG-TONG** → QUANTITY (2 editions: 1797, 1815) best sim=0.174 [topic_change]
- **RACK** → RADCLIFFE (2 editions: 1842, 1860) best sim=0.110 [person_bio]
- **RAMSAY** → RAMSAY (2 editions: 1815, 1823) best sim=0.178 [topic_change]
- **RAMUS** → RAMUS (2 editions: 1810, 1815) best sim=0.132 [person_bio]
- **RAY** → RAY (2 editions: 1810, 1823) best sim=0.047 [topic_change]
- **RESISTANCE** → THE (2 editions: 1815, 1842) best sim=0.159 [topic_change]
- **RESISTANCE OF FLUIDS** → BUT (2 editions: 1823, 1860) best sim=0.169 [topic_change]
- **RETORT** → THE (2 editions: 1797, 1823) best sim=0.086 [topic_change]
- **REVENUE** → BESIDES (2 editions: 1778, 1797) best sim=0.172 [topic_change]
- **REVIVIFICATION** → COMMISSION OF REVIEW (2 editions: 1797, 1815) best sim=0.137 [new_headword]
- **RIVER** → THE (2 editions: 1842, 1860) best sim=0.061 [topic_change]
- **ROSETTO** → ROSICRUCIANS (2 editions: 1797, 1810) best sim=0.170 [topic_change]
- **ROSSANO** → ROS SOLIS (2 editions: 1815, 1823) best sim=0.122 [person_bio]
- **ROUSSILLON** → VOL (2 editions: 1815, 1823) best sim=0.146 [topic_change]
- **RUSSIA** → DURING (2 editions: 1815, 1823) best sim=0.131 [topic_change]
- **RUSSIA** → THE FINNS (2 editions: 1815, 1823) best sim=0.180 [person_bio]
- **SANTA CRUZ** → SANTALUM (2 editions: 1810, 1815) best sim=0.134 [topic_change]
- **SCOTLAND** → DURING (2 editions: 1815, 1842) best sim=0.076 [topic_change]
- **SCRIMZEOR** → THE (2 editions: 1815, 1823) best sim=0.185 [topic_change]
- **SCRIPTURE** → THE (2 editions: 1810, 1860) best sim=0.068 [topic_change]
- **SECUNDINES** → SECUNDUS (2 editions: 1797, 1810) best sim=0.063 [topic_change]
- **SEGOVIA** → SEGREANT (2 editions: 1797, 1810) best sim=0.191 [topic_change]
- **SERVICE** → CHORAL SERVICE (2 editions: 1797, 1823) best sim=0.175 [person_bio]
- **SERVICE** → CHORAL (2 editions: 1810, 1815) best sim=0.176 [topic_change]
- **SETTING** → ACT OF SETTLEMENT (2 editions: 1815, 1823) best sim=0.168 [new_headword]
- **SHANNON** → ANTHONY TIMOTHY DOLTHEAD (2 editions: 1842, 1860) best sim=0.035 [person_bio]
- **SHARP** → MUSIC (2 editions: 1815, 1842) best sim=0.147 [topic_change]
- **SHIRAZ** → THE (2 editions: 1842, 1860) best sim=0.076 [topic_change]
- **SHIRT** → NOW (2 editions: 1797, 1810) best sim=0.041 [topic_change]
- **SHORT** → SHORT-HAND WRITING. (2 editions: 1797, 1823) best sim=0.176 [topic_change]
- **SHORT** → THE (2 editions: 1842, 1860) best sim=0.084 [topic_change]
- **SLEEP** → ONE (2 editions: 1810, 1823) best sim=0.187 [topic_change]
- **SLEEP** → SUCH (2 editions: 1815, 1823) best sim=0.064 [topic_change]
- **SLEIDAN** → SLEIGHT (2 editions: 1815, 1823) best sim=-0.001 [new_headword]
- **SLEIDAN** → THOUGH (2 editions: 1815, 1823) best sim=0.053 [topic_change]
- **SLOANE** → WITH (2 editions: 1815, 1823) best sim=0.087 [topic_change]
- **SMELLING** → WITH (2 editions: 1810, 1815) best sim=0.150 [topic_change]
- **SMOLLETT** → HAVING (2 editions: 1842, 1860) best sim=0.160 [topic_change]
- **SOLID** → SOLIPUGA (2 editions: 1810, 1823) best sim=0.170 [topic_change]
- **SOVEREIGN** → SOU (2 editions: 1810, 1815) best sim=0.182 [new_headword]
- **SPEECH** → SPEEDWELL (2 editions: 1797, 1810) best sim=0.126 [topic_change]
- **SPEECH** → SPEED (2 editions: 1797, 1810) best sim=0.143 [topic_change]
- **SPRAT** → SEE (2 editions: 1815, 1823) best sim=0.119 [topic_change]
- **STEWARD** → STEWARD (2 editions: 1810, 1823) best sim=0.191 [topic_change]
- **STRENGTH OF MATERIALS** → GALILEO (2 editions: 1842, 1860) best sim=0.128 [topic_change]
- **SUGILLATION** → END OF THE NINETEENTH VOLUME (2 editions: 1810, 1823) best sim=0.144 [new_headword]
- **SURGERY** → SECT (2 editions: 1810, 1815) best sim=0.243 [new_headword]
- **SURNAME** → WHEN (2 editions: 1823, 1842) best sim=0.044 [topic_change]
- **SURRENDER** → PLATE (2 editions: 1810, 1815) best sim=0.085 [topic_change]
- **TELESCOPE** → THE (2 editions: 1823, 1860) best sim=0.097 [topic_change]
- **TELL** → TELL-T (2 editions: 1810, 1815) best sim=0.180 [new_headword]
- **TENNANT** → HIS (2 editions: 1842, 1860) best sim=0.167 [topic_change]
- **TEST ACT** → TEST (2 editions: 1797, 1823) best sim=0.153 [topic_change]
- **THEOGNIS** → NUMEN (2 editions: 1810, 1815) best sim=0.320 [new_headword]
- **THOMSON** → THAT (2 editions: 1797, 1842) best sim=0.058 [topic_change]
- **TIPPERARY** → THE (2 editions: 1842, 1860) best sim=0.111 [topic_change]
- **TRIM** → PROBLEM (2 editions: 1810, 1815) best sim=0.317 [new_headword]
- **TRINIDAD** → THE (2 editions: 1815, 1860) best sim=0.110 [topic_change]
- **TROY-WEIGHT** → FOR (2 editions: 1815, 1823) best sim=0.131 [topic_change]
- **TURNING** → PLATE DXL (2 editions: 1815, 1823) best sim=0.266 [new_headword]
- **TYPE-FOUNDING** → THE (2 editions: 1842, 1860) best sim=0.195 [topic_change]
- **USQUEBAUGH** → VOL (2 editions: 1810, 1815) best sim=0.198 [new_headword]
- **WATERFORD** → THE (2 editions: 1842, 1860) best sim=0.186 [topic_change]
- **WEBSTER** → THE (2 editions: 1823, 1860) best sim=0.119 [topic_change]
- **WESTMEATH** → THE (2 editions: 1842, 1860) best sim=0.128 [topic_change]
- **WEXFORD** → THE (2 editions: 1842, 1860) best sim=0.117 [topic_change]
- **WICKLOW** → THE (2 editions: 1842, 1860) best sim=0.119 [topic_change]
- **WILSON** → THOMAS (2 editions: 1810, 1823) best sim=0.261 [new_headword]
- **WIT** → JOHN (2 editions: 1810, 1815) best sim=0.104 [topic_change]
- **WOOD** → ROTTEN (2 editions: 1810, 1815) best sim=0.134 [person_bio]
- **WORMS** → WORMING (2 editions: 1797, 1810) best sim=0.095 [topic_change]
- **WORMS** → WORMIUS (2 editions: 1797, 1810) best sim=0.156 [new_headword]
- **ZEALAND** → NEW ZEALAND (2 editions: 1810, 1823) best sim=0.324 [new_headword]
- **ZYMOSIMETER** → FINIS (2 editions: 1810, 1823) best sim=0.191 [new_headword]
- **ZYMOSIMETER** → PART I (2 editions: 1810, 1815) best sim=0.279 [new_headword]

## Single-Edition Breaks

**20887 detections** in only one edition (lower confidence):

### mid_word (251)

- **HUNGARY** (1860) para 42→43 sim=-0.023
  - `...reach honours and fame higher even than those of his father.`
  - `o his character as a philosopher, his genius will probably be more appreciated, ...`

- **PAUL** (1797) para 62→63 sim=-0.021
  - `...rge, and a magnificent monument was erected to his memorial.`
  - `sea language, is a short bar of wood on iron, fixed close to the capstan or wind...`

- **ORATORY** (1797) para 10→11 sim=-0.020
  - `...crocodiles; and if he escapes unhurt, he is deemed innocent.`
  - `dus of the equestrian order, who was succeeded by others; some of whose lives ar...`

- **PAUL** (1810) para 62→63 sim=-0.017
  - `...harge, and a magnificent monument was erected to his memory.`
  - `sea language, is a short bar of wood or iron, fixed close to the capstern or win...`

- **PAUL** (1815) para 61→62 sim=-0.013
  - `...harge, and a magnificent monument was erected to his memory.`
  - `sea language, is a short bar of wood or iron, fixed close to the capstern or win...`

- **ANCOURT** (1778) para 0→615 sim=0.018
  - `...d at last into nine. This last edition is the most complete.`
  - `b. During this process, the salivary glands being gently compressed by the contr...`

- **PAUL** (1823) para 65→66 sim=0.027
  - `...harge, and a magnificent monument was erected to his memory.`
  - `sea language is a short bar of wood or iron, fixed close to the capstan or windl...`

- **POLIANTHES** (1842) para 76→272 sim=0.050
  - `... against which it seems impracticable to guard successfully.`
  - `vi. The laws and experiments related in the sixth section belong to M. Fresnel. ...`

- **ANDREAS** (1778) para 1→665 sim=0.053
  - `...l those who write against the Mahometans quote it very much.`
  - `u. The penis has three pair of muscles, the ereciones, acceleratores, and transv...`

- **BULL** (1842) para 11→1018 sim=0.056
  - `... doors of temples were sometimes adorned with golden bullae.`
  - `now a minister aware of the evil tendency of our orders in Regency, council, and...`

- **ANDROS** (1778) para 1→739 sim=0.063
  - `...site of the ancient city. E. Long. 25° 30'. N. Lat. 37° 50'.`
  - `g. The trunk of the aorta, when it has reached the last vertebra lumborum, or th...`

- **FERG** (1810) para 0→785 sim=0.079
  - `...his prints of that kind are greatly esteemed by the curious.`
  - `d. BALSAM d. Balsam of Copaiva. See Expectorants.  In flatulent colic or gripes....`

- **PAK PATTAN** (1860) para 197→198 sim=0.087
  - `...lusca, laurus, and fulcatus), remarkable for their elegance.`
  - `man; yet nothing worth remembering occurred till the death of Leonardo, in the a...`

- **ANGLUS** (1778) para 0→867 sim=0.088
  - `...e restoration of Charles II., but in what year is uncertain.`
  - `n. The portio mollis of the seventh pair is distributed through the cochlea, the...`

- **POST** (1778) para 6→7 sim=0.088
  - `...al independent offices would only serve to ruin one another.`
  - `war, any fort of ground, fortified or not, where a body of men can be in a condi...`

- **MEDICINE** (1815) para 915→916 sim=0.088
  - `...z. 1731. Osservazioni di Targ. Tozetti, Racolta 1ma, p. 176.`
  - `ly lower the point o, but will bring it forward, and nearer the proper position ...`

- **MITYLENE** (1797) para 0→1241 sim=0.089
  - `...ch had been made between Mithridates and Sylla. See Metelin.`
  - `d. Blackish-brown.  2. With fine scales, a. White. b. Whitish-yellow. c. Reddish...`

- **ANDRONA** (1778) para 3→737 sim=0.090
  - `...Andronicus of Cyrrhus, built, at Athens, an octagon.`
  - `e. The aorta, after having given off at its curvature the carotids and subclavia...`

- **BOREL** (1823) para 0→192 sim=0.092
  - `... cum brevi omnium conspiscillorum historia. He died in 1678.`
  - `go is sold off, an account of sales is drawn out, in order to be transmitted to ...`

- **EISLEBEN** (1842) para 1→130 sim=0.096
  - `... for rent in arrear, or holding over his term, and the like.`
  - `et les bras collés contre les hanches. J'ai copié toute cette curieuse série de ...`

- **DRAMA** (1860) para 53→54 sim=0.096
  - `...ecimens of the Grecian comedy, both in action and character.`
  - `c and e are long tapering spades for digging out the middle and bottom splits, I...`

- **POLEMO** (1842) para 0→265 sim=0.099
  - `...ious pleasures, and devote himself to the pursuit of wisdom.`
  - `iii. The measurements related in the third section, which appear to show that, a...`

- **BOND** (1797) para 1→2 sim=0.106
  - `... 8vo. 2. Commentarii in sex satyras Persii, Lond. 1614, 8vo.`
  - `law, is a deed whereby the obligor obliges himself, his heirs, executors, and ad...`

- **SAXONY** (1797) para 4→5 sim=0.110
  - `...* See Reformation, No. 8.  † See Porteclairn, No. 23, 24.`
  - `ing, an important foreign commerce is carried on. A Saxony-great addition has be...`

- **ANDREA** (1778) para 0→661 sim=0.113
  - `...iles distant, and where the natives have not this distemper.`
  - `q. The penis is invested by the common integuments, but the cutis is reflected b...`

- **PARIS** (1815) para 0→1 sim=0.114
  - `...storie; manuscript. Besides many other things in manuscript.`
  - `son of Priam, king of Troy, by Hecuba, also named Alexander. He was decreed, eve...`

- **ANCONA** (1778) para 1→610 sim=0.116
  - `...nean it is scarce visible. E. Long. 15° 5'. N. Lat. 43° 36'.`
  - `i. The saliva, like all the other humours of the body, is found to be different ...`

- **MYCONUS** (1797) para 0→515 sim=0.119
  - `...ne, an island in the Archipelago. E. Long. 25° 6'. Lat. 37°.`
  - `la si ut re mi fa sol la,  which is (85) the scale of the minor mode of la in as...`

- **DIS** (1810) para 1→2 sim=0.121
  - `...ry, and the making of stays. E. Long. 1. 16. N. Lat. 52. 25.`
  - `god of the Gauls, the same as Pluto the god of hell. The inhabitants of Gaul sup...`

- **DIS** (1815) para 1→2 sim=0.125
  - `...si, and the making of flays. E. Long. i. 16. N. Lat. 52. 25.`
  - `god of the Gauls, the fame as Pluto the god of hell. The inhabitants of Gaul sup...`

  ... and 221 more

### mid_sentence (1196)

- **HUNGARY** (1860) para 35→36 sim=0.006
  - `...ject to the Hungarian crown, and to threaten Hungary Proper.`
  - `occasion to vindicate any one circumstance of my character and conduct." If by t...`

- **FOOTE** (1810) para 6→1012 sim=0.012
  - `...as privately interred in the cloisters of Westminster abbey.`
  - `with the general expression \( \frac{U}{V} \), it appears that \( U = 1 \), and ...`

- **PROME** (1842) para 0→282 sim=0.013
  - `...ls are lodged during the rains. Long. 95. E. Lat. 18. 50. N.`
  - `from which $c$ may be computed, and thence $h$ by means of one of the equations ...`

- **OLDENBURG** (1860) para 9→723 sim=0.017
  - `...lish from the French and Latin, under the anagram Grubendot.`
  - `versal ridges, and an anterior and posterior talon, the latter being more develo...`

- **CONNECTICUT** (1842) para 8→255 sim=0.020
  - `...assachusetts; and Hartford, Middletown, &c., in Connecticut.`
  - `therefore \(2\text{FHP} + \text{FHf} = 2\text{FHP} + \text{FHf}\), and hence \(2...`

- **FONTAINES** (1815) para 0→733 sim=0.021
  - `...of his works, and another catalogue of writings against him.`
  - `therefore \( my - x = 0, \) and \( y = \frac{x}{m}. \)...`

- **PHOTOGRAPHY** (1860) para 84→85 sim=0.025
  - `...assium ..... 5 gr. - Alcohol ..... ½ dr. - Water ..... 1 oz.`
  - `first discovered and stated the leading principles of Greek syntax, has rested u...`

- **VIDA** (1860) para 4→5 sim=0.025
  - `...VIE  VIE`
  - `cular of St Mark at Mantua. Here, however, he did not long remain, but removed t...`

- **GRAMPONDS** (1815) para 0→398 sim=0.027
  - `...ish church, which is at Creed about a quarter of a mile off.`
  - `well as the manner in which the place of these words is supplied in the language...`

- **FOLKESTONE** (1860) para 0→189 sim=0.029
  - `...s born here in 1578. Market-day, Thursday. Pop. (1851) 6726.`
  - `therefore also,  \[ \frac{x}{a} = \frac{1}{2} \left( \frac{y + \sqrt{y^2 - a^2}}...`

- **GUY** (1810) para 0→1 sim=0.032
  - `...one who could prove themselves in any degree related to him.`
  - `rope used to keep steady any weighty body whilst it is hoisting or lowering, par...`

- **COMITIA** (1842) para 20→131 sim=0.035
  - `...e might be prepared for martial service, the *tribus rus*...`
  - `whence, making \( A_1 = 36^\circ 39' 20" \),  \[ \begin{align*} \varepsilon &= (...`

- **GUY** (1815) para 0→1 sim=0.036
  - `...one who could prove themselves in any degree related to him.`
  - `rope used to keep steady any weighty body whilst it is hoisting or lowering, par...`

- **DIS** (1823) para 0→1 sim=0.041
  - `...y a separation, detachment, &c., as disposing, distributing.`
  - `town of Norfolk, seated on the river Waveney, on the side of a hill. It is a nea...`

- **JANUARY** (1860) para 13→14 sim=0.041
  - `...d its walls, and destroyed its temple and palaces with fire.`
  - `ments at their death. "Although," continues Golownin, "the keepers of bagnios ar...`

- **DIS** (1810) para 0→1 sim=0.049
  - `...fy a separation, detachment, &c. as disposing, distributing.`
  - `town of Norfolk, seated on the river Waveney, on the side of a hill. It is a nea...`

- **HUNGARY** (1860) para 38→39 sim=0.050
  - `...ace was actually concluded for the term of ten years (1444).`
  - `quiesced, in after life, in his first early conclusions—the very immaturity of w...`

- **BUSH** (1810) para 0→1 sim=0.052
  - `...nce, &c. in verse, Lond. by Pinson, 4to. 8. Carmina diversa.`
  - `term used for several shrubs of the same kind growing close together: thus we sa...`

- **HOOKE** (1823) para 17→18 sim=0.054
  - `...[595]  HOO`
  - `lation of Ramsay's Travels of Cyrus, in 4to; in 1733 he revised a Translation of...`

- **ASTRONOMY** (1860) para 401→402 sim=0.054
  - `...\[ P + (r - R) \cdot \frac{P}{P - p} \]`
  - `from which expression the solar ecliptic limits may be readily computed. The res...`

- **HAND** (1810) para 5→6 sim=0.055
  - `...nd the affair ended in the total dissolution of the academy.`
  - `falconry, is used for the foot of the hawk. To have a clean, strong, slender, gl...`

- **BRIGITTINS** (1815) para 0→3593 sim=0.055
  - `...d at Lisbon. The revenues were reckoned at £495l. per annum.`
  - `tripolium. * A. leaves strap-spear-shaped, fleshy, smooth, 3-fibred; calyx cæsæ ...`

- **CONSISTENTES** (1842) para 0→375 sim=0.055
  - `...prayers, but who were not admitted to receive the sacrament.`
  - `hence \( CL \cdot DH = DE \cdot CM. \)...`

- **CONN** (1815) para 0→129 sim=0.058
  - `...ted of late, and 330 parishes. The principal town is Galway.`
  - `therefore \( Ff + 2AF = Ff + 2af \);...`

- **MINT** (1778) para 4→5 sim=0.059
  - `...endered all such marks of much less consequence than before.`
  - `botany. See Mentha....`

- **DIS** (1815) para 0→1 sim=0.060
  - `...y a separation, detachment, &c. as disjoining, distributing.`
  - `town of Norfolk, seated on the river Waveney, on the side of a hill. It is a nea...`

- **ARCHANGEL** (1797) para 0→1 sim=0.061
  - `...th rank in the celestial hierarchy. See ANGEL and HIERARCHY.`
  - `city of Russia, in the province of Dwina, situated on the east side of the river...`

- **BOUHOURS** (1823) para 0→620 sim=0.062
  - `... year 1668, with a dedication prefixed to James II.'s queen.`
  - `leaves obvate-wedge-shaped, slightly dented, panicle and naked; stem shrubby. Af...`

- **ARCHANGEL** (1810) para 0→1 sim=0.063
  - `...th rank in the celestial hierarchy. See Angel and Hierarchy.`
  - `city of Russia, in the province of Dwina, situated on the east side of the river...`

- **REVENUE** (1797) para 32→33 sim=0.064
  - `...expense of the present war, be necessarily rendered greater.`
  - `hunting, a fleshy lump formed chiefly by a cluster of whitish worms on the head ...`

  ... and 1166 more

### new_headword (4751)

- **FOUR BOROUGHS' COURT** (1842) para 2→3 sim=-0.053
  - `...nd burgesses of free burghs, on a simple charge of ten days.`
  - `FOUR SADDLE ISLAND, an island in the Mergui Archipelago, about six miles in circ...`

- **PUBLIUS SYRUS** (1842) para 0→1 sim=-0.033
  - `... Orelli, Leipzig, 1822, or that of C. Zell, Stuttgart, 1829.`
  - `PUNA, a town of Hindustan, in the province of Bengal, sixty-three miles east fro...`

- **DAUBENTON** (1797) para 0→1 sim=-0.027
  - `...eral orations of his, and a life of St Francois Regis, 12mo.`
  - `CIRCULATING DECIMALS, called also recurring or repeating decimals, are those in ...`

- **ELEUTHEROLACONES** (1842) para 0→1 sim=-0.022
  - `...robably because the others had gradually become depopulated.`
  - `ELEVATION, the same with altitude or height.  Elevation of the Host, in the chur...`

- **LACCADIVES** (1860) para 16→17 sim=-0.020
  - `...er than those produced with the ordinary gold thread. (c.t.)`
  - `LACEDÉMON. See SPARTA.  LACEPÈDE, Bernard - Germain - Étienne de la Ville-sur-Il...`

- **FORCING** (1810) para 2→3 sim=-0.019
  - `...his place is famous for excellent trouts in its river Stour.`
  - `END OF THE EIGHTH VOLUME. ERRATA IN FLUXIONS.  Page. Col. Line. 700 1 27 for Mau...`

- **TIMUR** (1860) para 1→2 sim=-0.019
  - `...ently been written by C. R. Markham (Hakluyt Society), 1860.`
  - `TIN. See Mining....`

- **MACAU** (1842) para 0→0 sim=-0.017
  - `...e left bank of the Garonne, and containing 1800 inhabitants.`
  - `MACCABEES, two apocryphal books of Scripture, containing the history of Judas an...`

- **AGGERS-HERRED** (1778) para 0→1 sim=-0.015
  - `...three juridical places; namely, Acher, West Barum, and Ager.`
  - `AGESILAUS, king of the Lacedemonians, the son of Archidamus, was raised to the t...`

- **HOB-NOB** (1860) para 0→1 sim=-0.014
  - `...nce came to be used as an invitation to reciprocal drinking.`
  - `HOCHÉ, LAZARE, one of the noblest spirits and ablest generals of the French Repu...`

- **KENNICOTT** (1815) para 3→4 sim=-0.011
  - `...published, the volume having been completed from his papers.`
  - `KENO. See KINO....`

- **ASSOCIATION** (1823) para 11→12 sim=-0.011
  - `...eriod to his life in September 1817. See Africa, Supplement.`
  - `ASSOILZIE, in Law, to absolve or free.  ASSONANCE, in Rhetoric and Poetry, a ter...`

- **EXETER** (1860) para 0→1 sim=-0.010
  - `...several almshouses. It is connected with Bristol by railway.`
  - `EXFOLIATION; in Surgery, the separation of a piece of dead bone from the living ...`

- **NEWTY FORT** (1842) para 0→1 sim=-0.008
  - `...rn bank of a small river. Long. 73° 40'. E. Lat. 15° 56'. N.`
  - `NEW YEAR'S GIFTS, presents made on the first day of the new year. Nonius Marcell...`

- **KENNET** (1823) para 4→5 sim=-0.002
  - `...published, the volume having been completed from his papers.`
  - `KENO. See KINO....`

- **MONK** (1810) para 0→1 sim=-0.001
  - `...ervations on Military and Political Affairs," a small folio.`
  - `MONK Fish. See SQUALUS, Ichthyology Index.  MONK'S Head, or Wolf's bone. See ACO...`

- **MAINTENON, MADAME DE** (1860) para 4→5 sim=0.000
  - `...a society of literary antiquaries who have assumed his name.`
  - `MAIZE, or Indian Corn, the Zea Mays of botanists, a monocotyledonous grass of th...`

- **SAAVEDRA** (1842) para 0→1 sim=0.001
  - `...onde de Lemos. The particular day of his death is not known.`
  - `SABAGAN Islands, a group of small islands in the Red Sea. Long. 41° 54' E. Lat. ...`

- **SERVANT** (1815) para 7→246 sim=0.003
  - `...r the condition of servants by the law of Scotland, see LAW.`
  - `PROBLEM III. Required the sum of the infinite series \[ \frac{x}{1 \cdot 2} + \f...`

- **BETTINELLI** (1823) para 16→17 sim=0.005
  - `...whelms it on all sides.—See Biographie Universelle, Tom. IV.`
  - `BEYKANEER or BICANERE, a principality of Asia, situate in the north-west of Hind...`

- **MOOR** (1860) para 0→1 sim=0.006
  - `...iderable trade in corn, horses, and cattle. Pop. about 7000.`
  - `JAMES, an eminent Greek scholar, was born at Glasgow on the 22d of June 1712. Hi...`

- **POLICE** (1860) para 113→113 sim=0.010
  - `...beds, to the height of 500 feet above the present sea-level.`
  - `VII. Metropolitan Police.—State of Instruction of Persons taken into Custody, 18...`

- **CANUTE** (1860) para 1→2 sim=0.011
  - `...of his reign, and was buried in the monastery at Winchester.`
  - `CANVAS (French *canvres*; Greek *σάρπας*, hemp); a strong kind of cloth made of ...`

- **CLAYTON** (1860) para 1→2 sim=0.012
  - `...Liturgy, 1756, 8vo; 12. A Vindication, Part III., 1756, 8vo.`
  - `CLAZOMENÆ (Kelissman), a town of Ionia, and a member of the Ionian Dodecapolis, ...`

- **THLASPI** (1771) para 0→1 sim=0.015
  - `...ridate-mustard; and the bursa pastoris, or shepherd's purse.`
  - `THOMÆANS, Thomists, or Christians of St Thomas, a people of the East-Indies, who...`

- **NORIS** (1797) para 3→4 sim=0.015
  - `...ublished at Verona, in 1729 and 1730, in five volumes folio.`
  - `NORRKOPING, a town of Sweden, in the province of East Gothland, in east longitud...`

- **JERVIS** (1860) para 2→3 sim=0.015
  - `...cent; Lord Brougham's Statesmen of the Times of George III.)`
  - `JESI (the ancient Etrium), a walled city of the Papal States, on the left bank o...`

- **TZETZES** (1860) para 4→5 sim=0.016
  - `... antiquities. Pop. of North Uist, 3302; of South Uist, 4006.`
  - `UKRAINE, a large district of European Russia, formerly, as it name denotes, the ...`

- **MANLIUS** (1778) para 1→2 sim=0.017
  - `...eople, than it was for the people to bear with his severity.`
  - `MANKA, in the materia medica, the juice of certain trees of the ash kind, either...`

- **WINTON** (1860) para 4→5 sim=0.018
  - `...ntemporary, is composed of the same heterogeneous materials.`
  - `WIRE-ROPE. (See end of Rope-Making.)...`

  ... and 4721 more

### person_bio (1044)

- **PAK PATTAN** (1860) para 154→154 sim=-0.069
  - `...e, although all the component parts may be perfect; and Raf-`
  - `The Mactras and Tellens are also comparatively modern groups; most of the suppos...`

- **FRET** (1842) para 1→638 sim=-0.064
  - `...practised in roofs which are fretted over with plaster work.`
  - `At Laon had vanished Napoleon's last hope of retrieving his fortunes in the fiel...`

- **SERVANDONI** (1842) para 0→148 sim=-0.054
  - `...empt an enumeration of all his performances and exhibitions.`
  - `Genus Homalopsis. Body bulky, head very thick; muzzle short and rounded; eyes an...`

- **FRANKLIN, BENJAMIN** (1842) para 17→610 sim=-0.034
  - `...he heads of the different states in which they are situated.`
  - `At Borisow, which had thus been regained by Oudinot, the passage of the Berezina...`

- **BOULANGER** (1842) para 7→365 sim=-0.031
  - `...onument of finely-polished marble was erected to his memory.`
  - `After Don, we unite *Viviania* (*Macrea*, Lindl., and *Cassarea*, St Hil.) to th...`

- **ALCASSAR** (1815) para 0→1811 sim=-0.029
  - `...ated, and their king slain. W. Long. 12. 35. N. Lat. 35. 15.`
  - `Firstly, The quantity of cream obtained from the first-drawn cup was, in every c...`

- **PAK PATTAN** (1860) para 256→256 sim=-0.018
  - `... which, in plain words, is just what the objectors demanded.`
  - `In Zygobatus (fig. 39), the middle series of teeth is less broad; and a still na...`

- **FRATRICELLI** (1842) para 3→622 sim=-0.017
  - `...FRATICIDE, the crime of murdering a brother.`
  - `As Russia desired Poland, and Prussia Saxony, so Austria had her eye continually...`

- **ELECTRICITY** (1860) para 427→428 sim=-0.015
  - `...on the position and electricity of its poles have been made.`
  - `The Albanians now invited Ahmad Pasha Khursheed to assume the reins of governmen...`

- **ELECTRICITY** (1860) para 405→405 sim=-0.013
  - `...the character of an extravagant, cruel, and voluptuous king.`
  - `Mr Sievright of Meggetland fitted up a tourmaline so as to bring the action of i...`

- **GALE** (1823) para 0→1 sim=-0.012
  - `...other, they say that the one ship gales away from the other.`
  - `Dr John, an eminent and learned minister among the Baptists, was born at London ...`

- **MECHANICS** (1823) para 117→118 sim=-0.011
  - `...of the power makes with a line at right angles to the plane.`
  - `No Christian dares go to Mecca; not that the approach to it is prohibited by any...`

- **EU** (1860) para 0→2073 sim=-0.010
  - `... a crypt containing numerous monuments of the Artois family.`
  - `Genus Agromyza, Fallen. Antennae deflexed and porrect, the third joint orbicular...`

- **PAK PATTAN** (1860) para 139→140 sim=-0.009
  - `...arge size, in the eocene of Hampshire and miocene of Vienna.`
  - `If Julius was adapted for Michel Angelo, Leo X. was peculiarly so for Raffaello;...`

- **ODONTOLOGY** (1860) para 289→289 sim=-0.007
  - `...rlands and Holland, we must specify that of Van der Chilje.)`
  - `The Delphinus griseus has five teeth on each side of the lower jaw; but they soo...`

- **INDIANA** (1860) para 15→837 sim=-0.005
  - `... persons. The value of church property is 1,512,485 dollars.`
  - `The Mugil cephalus, or Mediterranean Grey Mullet, is distinguished from the Engl...`

- **GALE** (1815) para 0→1 sim=-0.004
  - `...other, they say that the one ship gales away from the other.`
  - `Dr John, an eminent and learned minister among the Baptists, was born at London ...`

- **ROSLIN** (1823) para 1→2 sim=-0.001
  - `...dinburgh, nor Segrave from being rescued from his captivity.`
  - `Rosmarinus, Rosemary, a genus of plants belonging to the diandra class, and in t...`

- **ELECTRICITY** (1860) para 441→442 sim=-0.001
  - `...manner, the laws by which the crystallization was regulated.`
  - `The Beys became divided in their wishes; one party being desirous of co-operatin...`

- **HENRY IV** (1810) para 47→48 sim=0.002
  - `...ur,*   he said he would make use of it with this alteration,`
  - `Prince Henry, notwithstanding his indifference in matrimonial matters, applied h...`

- **INDULGENCE** (1860) para 2→870 sim=0.006
  - `...spute became identified with the history of the Reformation.`
  - `The Coryphaene are strong, active, and voracious fishes. While swimming rapidly,...`

- **OLDHAM** (1860) para 8→740 sim=0.006
  - `...Edition of the English Poets, edited by Robert Bell, London.`
  - `In Phacochoerus (fig. 122) only the two premolars (p 3 and 4) are developed; in ...`

- **GRANADA** (1842) para 17→274 sim=0.011
  - `...Granada, New, a province of South America. See Venezuela.`
  - `That Mr Horne Tooke's principles will apply exactly to the conjunctions of every...`

- **PULTENEY** (1842) para 0→157 sim=0.012
  - `... his only son had died before him, the title became extinct.`
  - `In Prussia the state imposes on all parents the strict obligation of sending the...`

- **BELLADONA** (1810) para 0→1 sim=0.013
  - `...ivial name of a species of Atropa. See Atropa, Botany Index.`
  - `Bellai, William du, lord of Langey, a French general, Bellari, general, who sign...`

- **ESTREMADURA** (1842) para 1→1682 sim=0.013
  - `...rs the desolation they had spread in their advancing course.`
  - `Genus Chrysogaster, Meig. Antennae nutant, the third joint compressed, orbicular...`

- **ORMSKIRK** (1860) para 0→1775 sim=0.013
  - `...Parliamentary troops with great slaughter. Pop. (1851) 5548.`
  - `When Sir David Brewster discovered the system of rings in quartz, he found the t...`

- **MONTGOMERYSHIRE** (1860) para 12→440 sim=0.016
  - `...he 13th October 1828, in the seventy-fourth year of his age.`
  - `Total Fixed Issues: £3,303,357  * This bank has stopped payments. | Bank | Avera...`

- **JEWELL** (1810) para 2→3 sim=0.018
  - `...t to be kept chained in all parish churches for public use."`
  - `Jewel Blocks, in the sea language, a name given to two small blocks which are su...`

- **GORDIANUS** (1842) para 8→34 sim=0.018
  - `...s; also Caspar, Historia trium Gordianorum, Deventer, 1697.)`
  - `Hope Island, an island in the South Pacific Ocean, discovered by Le Maire and Sc...`

  ... and 1014 more

### topic_change (13645)

- **THEBES** (1860) para 7→129 sim=-0.083
  - `...e assimilated, in no small degree, to the Spartan character.`
  - `(J. D-R-N.)...`

- **ELECTRICITY** (1860) para 473→474 sim=-0.078
  - `...he wood used in these experiments was beach and cherry tree.`
  - `Boolak, the port of Cairo, is a considerable and flourishing town, having two re...`

- **HYPERBATON** (1842) para 2→146 sim=-0.075
  - `... juxta Ascanius, magnum spes altera Roma: Precedunt castris.`
  - `While the sulphuric-acid hygrometer displays considerable ingenuity, the other i...`

- **PAK PATTAN** (1860) para 140→141 sim=-0.071
  - `...s of *Modiola*, found in the cretaceous and tertiary strata.`
  - `In some life of him an attempt was made to prove that he caught cold by hurrying...`

- **PHOTOGRAPHY** (1860) para 96→97 sim=-0.068
  - `...lacial acetic acid ..... 3 dr. - Distilled water ..... 8 oz.`
  - `The very structure of inflected language shows us that we cannot measure futurit...`

- **PALESTINE** (1860) para 90→648 sim=-0.065
  - `...ition of Protestants in Syria and other parts of the empire.`
  - `Sub-order 1.—AMPHICHELIA.  Crocodiles closely resembling in general form the lon...`

- **INDULT** (1860) para 0→1 sim=-0.064
  - `...tter have the disposal of the benefices depending upon them.`
  - `**INDUS.** This great river of Asia has its rise in Thibet, at the N. of the Kai...`

- **BRADY** (1842) para 1→679 sim=-0.063
  - `...by his eldest son, who was a clergyman at Tooting in Surrey.`
  - `Order 152. Homalinea. R. Brown.  Perianth with a short tube, the limb 4-15-parti...`

- **HYGINUS** (1860) para 0→876 sim=-0.062
  - `...us Constituentibus, which have been several times reprinted.`
  - `\[ \frac{dS}{dh} = l + h \frac{dl}{dh} + 2mh = 0; \]...`

- **BOURCHIER** (1842) para 0→421 sim=-0.060
  - `...hony Wood says it was usually acted at Calais after vespers.`
  - `Order 51. Brexiaceae. Lindl.  Sepals five, small, persistent, cohering at the ba...`

- **BRUNSWICK** (1860) para 9→1224 sim=-0.060
  - `...ssible to vessels drawing not more than 13 feet at low tide.`
  - `The first step taken by the new ministry after the meeting of parliament was to ...`

- **MONTMORENCY** (1860) para 1→443 sim=-0.059
  - `... beheaded in the Hotel de Ville of Toulouse in October 1632.`
  - `**Carry forward**   £3,394,852   £3,254,464   £3,940,719  | Bank                ...`

- **SERINGAPATAM** (1842) para 1→241 sim=-0.058
  - `... and from Delhi 1321 miles. Long. 76° 51' E. Lat. 12° 26' N.`
  - `\[ \frac{a + bx}{1 - ax - bx} = \frac{a + bx}{(1 - px)(1 - qx)}. \]...`

- **FIRDUSI** (1860) para 0→479 sim=-0.058
  - `... and verse was published by Mr James Atkinson, London, 1833.`
  - `\[ U = Q (1 + n \sin^2 l); \]...`

- **PAK PATTAN** (1860) para 141→141 sim=-0.055
  - `...ur with fifty quarters in their arms to sit in his presence.`
  - `The *Trigoniidae* are represented in the lower Silurian strata by *Lyrodema* (fi...`

- **CIDARIS** (1860) para 4→623 sim=-0.054
  - `...r there are remains of a Roman town, supposed to be Cartela.`
  - `1796-1838 Möhler, Ger. "Symbolism." 1781-1838 A. von Chamisso, Ger. Natural Scie...`

- **ELECTRICITY** (1860) para 434→435 sim=-0.049
  - `...ing is the list of those in which he detected this property:`
  - `Mohammad 'Alee now possessed the title of Governor of Egypt, but beyond the wall...`

- **EARL** (1842) para 0→343 sim=-0.048
  - `...ed to his successors, though the reason has long ago failed.`
  - `\[ A \times V^2 - v^2 = B \times u^2 - U^2, \]...`

- **ROCHESTER** (1842) para 2→422 sim=-0.047
  - `...he population amounted in 1821 to 8795, and in 1831 to 9811.`
  - `Then we get the swell with sufficient precision for any point H between F and D,...`

- **HUNGARY** (1860) para 48→48 sim=-0.047
  - `...p pen at one end, and an instrument of erasure at the other.`
  - `Mohammed, who vowed to unfurl the banner of the prophet on the ramparts of Belgr...`

- **PAK PATTAN** (1860) para 216→217 sim=-0.045
  - `...anting in the deposit where the Conodonts are most abundant.`
  - `In 1711, there existed a school, of which Kneller was the head, whilst Vertue th...`

- **BOULTER** (1810) para 0→1013 sim=-0.044
  - `...monument of finely polished marble is erected to his memory.`
  - `† Cufute nonnulla.  Sect. II. Flores pentapetali, inferi....`

- **PAK PATTAN** (1860) para 269→269 sim=-0.044
  - `...nnual rent of all the property within the burgh was £99,628.`
  - `The last term signifies a form and structure of tail illustrated by fig. 42, and...`

- **ROPE** (1860) para 206→436 sim=-0.044
  - `...ine room 90 feet long, 19 feet wide, and 9 feet high. (A.A.)`
  - `The contempt with which the character of this unfortunate emperor has been loade...`

- **MYLASA** (1810) para 0→254 sim=-0.043
  - `...nts, which have been employed to construct a Turkish mosque.`
  - `Again, the ♯ above the highest G of the treble staff is placed on a leger line a...`

- **FLORIAN** (1860) para 0→366 sim=-0.043
  - `...dition of Florian's works is that of 1812, in 16 vols. 18mo.`
  - `We may add that the transit duties, that is, the tax imposed on transmission int...`

- **MONTENEGRO** (1860) para 1→402 sim=-0.043
  - `...oes not seem to have been restored by the Romans. Pop. 6600.`
  - `---  1 We earnestly recommend those who may have any doubts in regard to this co...`

- **COL** (1823) para 32→935 sim=-0.042
  - `...ice, Elche in Spain D. Decuriones D. C. A. Divus. Caes. Aug.`
  - `437. The mode of bringing the sails back against the wind, which Mr Beatson inve...`

- **AUBENAS** (1860) para 1→260 sim=-0.042
  - `... retired to Geneva, where he died in 1620, at the age of 80.`
  - `Let us consider the general expression  \[ \int \int \frac{d \phi d \cos \theta ...`

- **ARACHNIDES** (1842) para 87→87 sim=-0.040
  - `... repulsed with severe loss, by the well-directed fire of the`
  - `The external appearance varies considerably in the same species, according to th...`

  ... and 13615 more
