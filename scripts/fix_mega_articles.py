#!/usr/bin/env python3
"""Fix specific mega-articles that swallowed neighboring entries.

Each fix is hand-specified based on manual analysis of the article text.
This is a one-time cleanup script, not a general-purpose tool.

Usage:
    python scripts/fix_mega_articles.py --dry-run    # preview changes
    python scripts/fix_mega_articles.py              # apply changes
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

ARTICLES_DIR = REPO_DIR / "data" / "articles"


def find_split_point(text, pattern, after_pct=0):
    """Find the character position where `pattern` starts a new article.

    Returns the char offset in `text` where the split should occur
    (everything before this point stays in the current article,
     everything from this point starts the new article).

    after_pct: only match occurrences after this percentage through the text.
    """
    min_pos = int(len(text) * after_pct / 100)
    for m in re.finditer(pattern, text):
        if m.start() >= min_pos:
            # Back up to start of the line (after previous \n\n)
            pos = text.rfind('\n\n', 0, m.start())
            if pos == -1:
                pos = m.start()  # no paragraph break — split at match
            else:
                pos += 2  # skip the \n\n
            return pos
    return None


def split_article(articles, idx, splits):
    """Split articles[idx] at the given split points.

    splits: list of (new_title, pattern, after_pct) tuples, in order.
    Returns list of new articles to replace articles[idx].
    """
    original = articles[idx]
    text = original['text']
    result = []

    # Find all split positions
    positions = []  # (char_pos, new_title)
    for new_title, pattern, after_pct in splits:
        pos = find_split_point(text, pattern, after_pct)
        if pos is not None:
            positions.append((pos, new_title))

    # Sort by position
    positions.sort(key=lambda x: x[0])

    # Create article slices
    prev_pos = 0
    prev_title = original['title']

    for pos, new_title in positions:
        if pos == 0:
            # pos=0 means "rename the host article" — don't emit an empty chunk
            prev_title = new_title
            continue
        if pos <= prev_pos:
            continue
        chunk_text = text[prev_pos:pos].strip()
        if chunk_text:
            art = dict(original)
            art['title'] = prev_title
            art['text'] = chunk_text
            art['word_count'] = len(chunk_text.split())
            art['char_start'] = original['char_start'] + prev_pos
            art['char_end'] = original['char_start'] + pos
            art['article_id'] = f"{original['article_id']}_{len(result)}" if result else original['article_id']
            art['heading_pattern'] = 'mega_split_manual'
            result.append(art)
        prev_pos = pos
        prev_title = new_title

    # Final chunk
    chunk_text = text[prev_pos:].strip()
    if chunk_text:
        art = dict(original)
        art['title'] = prev_title
        art['text'] = chunk_text
        art['word_count'] = len(chunk_text.split())
        art['char_start'] = original['char_start'] + prev_pos
        art['char_end'] = original['char_end']
        art['article_id'] = f"{original['article_id']}_{len(result)}"
        art['heading_pattern'] = 'mega_split_manual'
        result.append(art)

    return result


# ============================================================================
# Fix specifications: (year, title, file_pattern, splits)
# Each split is (new_title, regex_pattern, min_percent)
# ============================================================================

FIXES = [
    # BOSWORTH-MARKET swallowed BOTAL and BOTANY
    (1860, 'BOSWORTH-MARKET', 'eb_8th_1860_v05', [
        ('BOTAL', r'BOTAL,', 0),
        ('BOTANY', r'VEGETABLE ORGANOGRAPHY AND PHYSIOLOGY', 0),
    ]),

    # UNIVERSITY OF PARIS swallowed all university sub-articles (1860)
    (1860, 'UNIVERSITY OF PARIS', 'eb_8th_1860_v21', [
        ('UNIVERSITIES (English)', r'ENGLISH UNIVERSITIES', 10),
        ('UNIVERSITY OF OXFORD', r'UNIVERSITY OF OXFORD', 10),
        ('UNIVERSITY OF CAMBRIDGE', r'UNIVERSITY OF CAMBRIDGE', 20),
        ('UNIVERSITY OF LONDON', r'UNIVERSITY OF LONDON', 35),
        ('UNIVERSITIES (Scottish)', r'SCOTTISH UNIVERSITIES', 50),
        ('UNIVERSITY OF GLASGOW', r'UNIVERSITY OF GLASGOW', 58),
        ('UNIVERSITY OF ABERDEEN', r'UNIVERSITY OF ABERDEEN', 63),
        ('UNIVERSITY OF EDINBURGH', r'UNIVERSITY OF EDINBURGH', 68),
        ('UNIVERSITY OF DUBLIN', r'UNIVERSITY OF DUBLIN', 75),
        ('UNIVERSITIES (Colonial)', r'COLONIAL UNIVERSITIES', 88),
        ('UNIVERSITY OF FRANCE', r'UNIVERSITY OF FRANCE', 90),
    ]),

    # UNIVERSITY OF PARIS swallowed sub-articles (1842)
    (1842, 'UNIVERSITY OF PARIS', 'eb_7th_1842_v21', [
        ('UNIVERSITIES (English)', r'ENGLISH UNIVERSITIES', 10),
        ('UNIVERSITY OF LONDON', r'UNIVERSITY OF LONDON', 40),
        ('UNIVERSITIES (Scottish)', r'SCOTI.H UNIVERSITIES', 48),
        ('UNIVERSITY OF ABERDEEN', r'UNIVERSITY OF ABERDEEN', 60),
        ('UNIVERSITY OF EDINBURGH', r'UNIVERSITY OF EDINBURGH', 70),
        ('UNIVERSITY OF DUBLIN', r'UNIVERSITY OF DUBLIN', 78),
        ('UNIVERSITY OF FRANCE', r'ROYAL UNIVERSITY OF FRANCE', 88),
    ]),

    # MINERALOGY swallowed GEOLOGY (1842)
    (1842, 'MINERALOGY', 'eb_7th_1842_v15', [
        ('GEOLOGY', r'OBJECTS OF GEOLOGICAL SCIENCE', 60),
    ]),

    # SCOTLAND IS BY NO → should be SCOTLAND (1815 broken headword)
    (1815, 'SCOTLAND IS BY NO', 'eb_5th_1815_v18', [
        # Just rename, don't split — it's the real SCOTLAND article with a broken title
    ]),

    # ANTAGONISTS OF HOBBIESTS → part of DISSERTATIONS (1842 broken headword)
    # This is actually a fragment of the Dissertations prelim material
    # Just rename it
    (1842, 'ANTAGONISTS OF HOBBIESTS', 'eb_7th_1842_v01', []),

    # CLOCK AND WATCH WORK — broken headword, probably CLOCKS
    (1842, 'CLOCK AND WATCH WORK', 'eb_7th_1842_v06', []),

    # HYDRODYNAMICS swallowed INDEX and DIRECTIONS (1810, 1815, 1823)
    (1810, 'HYDRODYNAMICS', 'eb_4th_1810_v10', []),
    (1815, 'HYDRODYNAMICS', 'eb_5th_1815_v10', []),
    (1823, 'HYDRODYNAMICS', 'eb_6th_1823_v10', []),

    # ================================================================
    # VARIANT→SWALLOW reclassifications (found Mar 28, 2026)
    # These were classified as VARIANT by classify_gaps.py but are
    # actually swallowed articles — the parser missed the headword
    # boundary (ALLCAPS headword, OCR-damaged headword, or stripped).
    # ================================================================

    # --- 1778 2nd edition ---

    # FORGER (6w) swallowed FORGERY (811w)
    (1778, 'FORGER', 'eb_2nd_1778_v04', [
        ('FORGERY', r'Forgery,', 0),
    ]),

    # --- 1797 3rd edition ---

    # ACOUSTIC (short def) swallowed ACOUSTICS (16,349w)
    (1797, 'ACOUSTIC', 'eb_3rd_1797_v01', [
        ('ACOUSTICS', r'ACOUSTICS', 0),
    ]),

    # ANDREW (short bio) swallowed ST ANDREW'S (2,526w)
    (1797, 'ANDREW', 'eb_3rd_1797_v01', [
        ("ANDREW'S", r"Andrew.s \(St\)", 0),
    ]),

    # BAPTIST (175w) swallowed BAPTISTS (862w)
    (1797, 'BAPTIST', 'eb_3rd_1797_v02', [
        ('BAPTISTS', r'BAPTISTS,', 0),
    ]),

    # CHARLES (biography entries) swallowed CHARLES V (1,549w)
    (1797, 'CHARLES', 'eb_3rd_1797_v04', [
        ('CHARLES V', r'Charles V\.', 0),
    ]),

    # CHURCH (1,016w) swallowed CHURCHILL (4,574w)
    (1797, 'CHURCH', 'eb_3rd_1797_v04', [
        ('CHURCHILL', r'Churchill,', 20),
    ]),

    # GRACE (sub-entries) swallowed GRACES (255w)
    (1797, 'GRACE', 'eb_3rd_1797_v08', [
        ('GRACES', r'Graces, Grati', 80),
    ]),

    # LOMBARD (93w) swallowed LOMBARDS (5,827w)
    (1797, 'LOMBARD', 'eb_3rd_1797_v10', [
        ('LOMBARDS', r'LOMBARDS,', 0),
    ]),

    # NEEDLE (def + manufacturing) swallowed NEEDLES (3,729w)
    (1797, 'NEEDLE', 'eb_3rd_1797_v12', [
        ('NEEDLES', r'Needles make', 0),
    ]),

    # PALES (39w) swallowed PALESTINE (4,232w)
    (1797, 'PALES', 'eb_3rd_1797_v13', [
        ('PALESTINE', r'PALESTINE,', 0),
    ]),

    # PRESBYTER (149w) swallowed PRESBYTERIANS (4,051w)
    (1797, 'PRESBYTER', 'eb_3rd_1797_v15', [
        ('PRESBYTERIANS', r'Presbyterians,', 0),
    ]),

    # RIGHT (148w) swallowed RIGHTS (3,518w)
    (1797, 'RIGHT', 'eb_3rd_1797_v16', [
        ('RIGHTS', r'Rights, in the common', 20),
    ]),

    # --- 1810 4th edition ---

    # AERONAUT (17w) swallowed AERONAUTICA (22,913w)
    (1810, 'AERONAUT', 'eb_4th_1810_v08', [
        ('AERONAUTICA', r'Aeronautica,', 0),
    ]),

    # BAPTIST (175w) swallowed BAPTISTS (861w)
    (1810, 'BAPTIST', 'eb_4th_1810_v03', [
        ('BAPTISTS', r'BAPTISTS,', 0),
    ]),

    # BENEDICTINS swallowed BENEDICTION (285w) at end
    (1810, 'BENEDICTINS', 'eb_4th_1810_v03', [
        ('BENEDICTION', r'Benediction is also used for an ecclesiastical', 60),
    ]),

    # BUCHAN (33w) swallowed BUCHANAN (1,871w)
    (1810, 'BUCHAN', 'eb_4th_1810_v04', [
        ('BUCHANAN', r'Buchanan, George', 0),
    ]),

    # CHURCH (1,016w) swallowed CHURCHILL (4,865w)
    (1810, 'CHURCH', 'eb_4th_1810_v17', [
        ('CHURCHILL', r'Churchill, Sir Winston', 20),
    ]),

    # GRACE swallowed GRACES (255w)
    (1810, 'GRACE', 'eb_4th_1810_v05', [
        ('GRACES', r'Graces, Grati', 80),
    ]),

    # INVERNESS (city) swallowed INVERNESS-SHIRE (1,384w)
    (1810, 'INVERNESS', 'eb_4th_1810_v11', [
        ('INVERNESS-SHIRE', r'Inverness-Shire,', 10),
    ]),

    # JUDGE (18w) swallowed JUDGES (1,989w)
    (1810, 'JUDGE', 'eb_4th_1810_v11', [
        ('JUDGES', r'Judges, in Jewish', 0),
    ]),

    # MEDICIS swallowed MEDICI (2,462w)
    (1810, 'MEDICIS', 'eb_4th_1810_v13', [
        ('MEDICI', r'MEDICI, LORENZO', 10),
    ]),

    # ROPE (97w) swallowed ROPES (15,486w)
    (1810, 'ROPE', 'eb_4th_1810_v17', [
        ('ROPES', r'Ropes are made', 0),
    ]),

    # SOUTH (one-liner) swallowed SOUTH SEA (394w)
    (1810, 'SOUTH', 'eb_4th_1810_v19', [
        ('SOUTH SEA', r'SOUTH Sea, or Pacific', 30),
    ]),

    # STAR (sub-entries) swallowed STAR-BOARD (260w)
    (1810, 'STAR', 'eb_4th_1810_v19', [
        ('STAR-BOARD', r'STAR-Board,', 60),
    ]),

    # STEWARD (184w) swallowed STEWART (1,458w)
    (1810, 'STEWARD', 'eb_4th_1810_v19', [
        ('STEWART', r'STEWART', 40),
    ]),

    # STONE (282w) swallowed STONES (1,141w)
    (1810, 'STONE', 'eb_4th_1810_v19', [
        ('STONES', r'STONES, in Natural History', 50),
    ]),

    # WHITE (sub-entries) swallowed WHITEFIELD (462w)
    (1810, 'WHITE', 'eb_4th_1810_v20', [
        ('WHITEFIELD', r'Whitefield, George', 20),
    ]),

    # --- 1815 5th edition ---

    # BAPTIST (175w) swallowed BAPTISTS (860w)
    (1815, 'BAPTIST', 'eb_5th_1815_v03', [
        ('BAPTISTS', r'BAPTISTS,', 0),
    ]),

    # BATTERING (102w) swallowed BATTERING-RAM (447w)
    (1815, 'BATTERING', 'eb_5th_1815_v03', [
        ('BATTERING-RAM', r'BATTERING-Ram,', 10),
    ]),

    # BIRD (508w) swallowed BIRD-CALL (4,620w)
    (1815, 'BIRD', 'eb_5th_1815_v03', [
        ('BIRD-CALL', r'Bird-Call,', 5),
    ]),

    # BUCHAN (33w) swallowed BUCHANAN (1,872w)
    (1815, 'BUCHAN', 'eb_5th_1815_v04', [
        ('BUCHANAN', r'Buchanan, George', 0),
    ]),

    # CHURCH (1,016w) swallowed CHURCHILL (4,908w)
    (1815, 'CHURCH', 'eb_5th_1815_v06', [
        ('CHURCHILL', r'Churchill, Sir Winston', 20),
    ]),

    # FOUNTAIN (144w) swallowed FOUNTAIN-TREE (1,531w)
    (1815, 'FOUNTAIN', 'eb_5th_1815_v09', [
        ('FOUNTAIN-TREE', r'FOUNTAIN-Tree,', 5),
    ]),

    # GREEN (243w) swallowed GREEN-CLOTH (at 2%) and GREEN-HOUSE (at 15%)
    (1815, 'GREEN', 'eb_5th_1815_v10', [
        ('GREEN-CLOTH', r'GREEN-Cloth,', 0),
        ('GREEN-HOUSE', r'GREEN-House,', 10),
    ]),

    # GUN-SMITH (short def) swallowed GUN-SMITHERY (7,985w)
    (1815, 'GUN-SMITH', 'eb_5th_1815_v10', [
        ('GUN-SMITHERY', r'GUN-Smithery,', 0),
    ]),

    # INVERNESS (city) swallowed INVERNESS-SHIRE (1,393w)
    (1815, 'INVERNESS', 'eb_5th_1815_v11', [
        ('INVERNESS-SHIRE', r'INVERNESS-Shire,', 10),
    ]),

    # MEDICIS swallowed MEDICI (2,465w)
    (1815, 'MEDICIS', 'eb_5th_1815_v13', [
        ('MEDICI', r'Medici, Lorenzo', 10),
    ]),

    # NEEDLES (manufacturing) swallowed NEEDLE (sub-entries, 592w) at end
    (1815, 'NEEDLES', 'eb_5th_1815_v14', [
        ('NEEDLE', r'Needle Fish', 70),
    ]),

    # RIGHT (148w) swallowed RIGHTS (3,553w)
    (1815, 'RIGHT', 'eb_5th_1815_v18', [
        ('RIGHTS', r'Rights, in the common', 20),
    ]),

    # SOUTH (one-liner) swallowed SOUTH SEA (395w)
    (1815, 'SOUTH', 'eb_5th_1815_v19', [
        ('SOUTH SEA', r'SOUTH Sea, or Pacific', 30),
    ]),

    # STAR (sub-entries) swallowed STAR-BOARD (260w)
    (1815, 'STAR', 'eb_5th_1815_v19', [
        ('STAR-BOARD', r'STAR-Board,', 60),
    ]),

    # VARI (21w) swallowed VARIATION (6,124w)
    (1815, 'VARI', 'eb_5th_1815_v20', [
        ('VARIATION', r'VARIATION of the Compass', 0),
    ]),

    # WEAVING (32w) swallowed WEAVING-LOOM (3,267w)
    (1815, 'WEAVING', 'eb_5th_1815_v20', [
        ('WEAVING-LOOM', r'WEAVING-Loom,', 0),
    ]),

    # --- 1823 6th edition ---

    # BATTERING (102w) swallowed BATTERING-RAM (445w)
    (1823, 'BATTERING', 'eb_6th_1823_v03', [
        ('BATTERING-RAM', r'BATTERING-Ram,', 10),
    ]),

    # GOOD (sub-entries) swallowed GOOD HOPE (3,444w)
    (1823, 'GOOD', 'eb_6th_1823_v09', [
        ('GOOD HOPE', r'Good Hope.*promontory', 10),
    ]),

    # MEDICIS swallowed MEDICI (2,462w)
    (1823, 'MEDICIS', 'eb_6th_1823_v13', [
        ('MEDICI', r'Medici, Lorenzo', 10),
    ]),

    # NEEDLE (manufacturing) swallowed NEEDLES (2,428w)
    (1823, 'NEEDLE', 'eb_6th_1823_v14', [
        ('NEEDLES', r'Needles.*sharp pointed rocks', 90),
    ]),

    # RIGHT (148w) swallowed RIGHTS (2,933w)
    (1823, 'RIGHT', 'eb_6th_1823_v18', [
        ('RIGHTS', r'Rights, in the common', 20),
    ]),

    # ROBERT (bio entries) swallowed ROBERTSON (2,570w)
    (1823, 'ROBERT', 'eb_6th_1823_v18', [
        ('ROBERTSON', r'Robertson, Dr', 10),
    ]),

    # --- 1842 7th edition ---

    # ACCIDENT (sub-entries) swallowed ACCIDENTAL (2,173w)
    (1842, 'ACCIDENT', 'eb_7th_1842_v02', [
        ('ACCIDENTAL', r'Accidental, in Philosophy', 5),
    ]),

    # ADMIRAL (history) swallowed ADMIRALTY (3,901w)
    (1842, 'ADMIRAL', 'eb_7th_1842_v02', [
        ('ADMIRALTY', r'Admiralty, High Court', 30),
    ]),

    # AYR (town) swallowed AYRSHIRE (1,989w)
    (1842, 'AYR', 'eb_7th_1842_v04', [
        ('AYRSHIRE', r'AYRSHIRE,', 20),
    ]),

    # BUKHARIA → actually contains BUKHARA article (different spelling)
    (1842, 'BUKHARIA', 'eb_7th_1842_v05', [
        ('BUKHARA', r'Bukhara, or Bochhara, an extensive region', 0),
    ]),

    # CLACKMANNAN (town) swallowed CLACKMANNANSHIRE (3,647w)
    (1842, 'CLACKMANNAN', 'eb_7th_1842_v06', [
        ('CLACKMANNANSHIRE', r'Clackmannanshire,', 0),
    ]),

    # LOMBARDS (history) swallowed LOMBARDY (19,845w) — headword stripped by OCR
    (1842, 'LOMBARDS', 'eb_7th_1842_v13', [
        ('LOMBARDY', r'This interesting part of Italy was in remote periods', 15),
    ]),

    # PEMBROKE (town) swallowed PEMBROKESHIRE (1,756w)
    (1842, 'PEMBROKE', 'eb_7th_1842_v17', [
        ('PEMBROKESHIRE', r'PEMBROKESHIRE', 5),
    ]),

    # PHILIPPI (battle) swallowed PHILIPPINES (2,265w)
    (1842, 'PHILIPPI', 'eb_7th_1842_v17', [
        ('PHILIPPINES', r'Philippines\.', 0),
    ]),

    # YORK: article starts with YORKSHIRE county content, then YORK city at ~73%
    # Headword "YORK" was stripped by OCR.
    # Split trick: YORKSHIRE at 0% renames the host, YORK at 60% splits the city.
    (1842, 'YORK', 'eb_7th_1842_v21', [
        ('YORKSHIRE', r'an English county', 0),
        ('YORK', r'very ancient city, the capital of the county', 60),
    ]),

    # ================================================================
    # PARSING_OR_EDITORIAL → SWALLOWED (found Mar 28-29, 2026)
    # These are articles the parser missed, swallowed by adjacent entries.
    # ================================================================

    # --- 1797 3rd edition ---
    (1797, 'RAIN', 'eb_3rd_1797_v15', [
        ('RAINBOW', r'Rainbow', 70),
    ]),

    # --- 1810 4th edition ---
    (1810, 'BAILLIE', 'eb_4th_1810_v03', [
        ('BAILLY', r'Bailly, Jean Sylvain', 25),
    ]),
    (1810, 'DENMARK', 'eb_4th_1810_v17', [
        ('DENNIS', r'Dennis, John', 85),
    ]),
    (1810, 'MIDWIFERY', 'eb_4th_1810_v17', [
        ('MIEL', r'Miel, Jan', 90),
    ]),

    # --- 1815 5th edition ---
    (1815, 'CAMPANULA', 'eb_5th_1815_v05', [
        ('CAMPBELL', r'Campbell, Archibald', 0),
    ]),
    (1815, 'CHARLES MARTEL', 'eb_5th_1815_v05', [
        ('CHARLES V', r'Charles V.*emperor.*king of Spain.*was son of Philip', 0),
    ]),
    (1815, 'PLATONISM', 'eb_5th_1815_v16', [
        ('PLAUTUS', r'Plautus, Marcus', 50),
        ('PLAYHOUSE', r'Playhouse', 55),
    ]),
    (1815, 'RAIN', 'eb_5th_1815_v17', [
        ('RAINBOW', r'Rainbow', 70),
    ]),
    (1815, 'ROPES', 'eb_5th_1815_v18', [
        ('ROSA', r'Rosa, Salvator', 90),
    ]),
    (1815, 'ROUSSILLON', 'eb_5th_1815_v18', [
        ('ROUSSEAU', r'Rousseau, John-James', 20),
    ]),

    # --- 1823 6th edition ---
    (1823, 'ABYSS', 'eb_6th_1823_v01', [
        ('ABYSSINIA', r'Abyssinia', 0),
    ]),
    (1823, 'AETIUS', 'eb_6th_1823_v01', [
        ('AETNA', r'Aetna', 0),
    ]),
    # GASSENDI was recovered from OCR by recover_from_ocr.py
    (1823, 'CAMMIN', 'eb_6th_1823_v05', [
        ('CAMOENS', r'Camoens, Louis', 0),
    ]),
    (1823, 'CAMPANULA', 'eb_6th_1823_v05', [
        ('CAMPBELL', r'Campbell, Archibald', 0),
    ]),
    (1823, 'CLARK', 'eb_6th_1823_v06', [
        ('CLARKE', r'Clarke, William', 80),
    ]),
    (1823, 'DANMONII', 'eb_6th_1823_v07', [
        ('DANTE', r'Dante, Aligheri', 5),
    ]),
    (1823, 'JOHN', 'eb_6th_1823_v11', [
        ('JOHNSON', r'Johnson, Ben', 10),
    ]),
    (1823, 'MATERA', 'eb_6th_1823_v12', [
        ('MATERIA MEDICA AND PHARMACY', r'MATERIA MEDICA AND PHARMACY', 0),
    ]),
    (1823, 'MORLACHIA', 'eb_6th_1823_v14', [
        ('MORNAY', r'Mornay, Philippe', 60),
    ]),
    (1823, 'PERSPECTIVE', 'eb_6th_1823_v16', [
        ('PERTH', r'Perth, the capital', 85),
    ]),
    (1823, 'RICE', 'eb_6th_1823_v18', [
        ('RICHARDSON', r'Richardson, Jonathan', 60),
    ]),
    (1823, 'VERMILION', 'eb_6th_1823_v20', [
        ('VERMIN', r'Vermin,', 0),
    ]),

    # --- 1842 7th edition ---
    (1842, 'BREAKERS', 'eb_7th_1842_v05', [
        ('BREAKWATER', r'Breakwater', 70),
    ]),
    (1842, 'JOINTS', 'eb_7th_1842_v12', [
        ('JONES', r'Jones, Inigo', 50),
    ]),
    (1842, 'LAKE', 'eb_7th_1842_v13', [
        ('LALANDE', r'Lalande, Joseph Jerome', 20),
    ]),
    (1842, 'LESLEY', 'eb_7th_1842_v13', [
        ('LESLIE', r'Leslie, Charles', 30),
    ]),
    (1842, 'MONTE MAGGIORE', 'eb_7th_1842_v15', [
        ('MONTESQUIEU', r'Montesquieu, Charles', 0),
    ]),

    # ================================================================
    # MEGA-ARTICLE SWALLOWERS — Session Mar 29, 2026
    # Large articles that swallowed significant adjacent content.
    # ================================================================

    # SCOT (1860) — Reginald Scot bio (759 chars) + SCOTLAND history (63K)
    (1860, 'SCOT', 'eb_8th_1860_v19', [
        ('SCOTLAND', r'HISTORY OF SCOTLAND', 0),
    ]),

    # ROMANO (1823) — Giulio Romano bio (~1477 chars) + ROME history (42K)
    (1823, 'ROMANO', 'eb_6th_1823_v18', [
        ('ROME', r'ROM[EÈ],.*(?:ancient|celebrated).*city.*Italy', 0),
    ]),

    # BOND (1771) — 40w definition swallowed everything through BOOK-KEEPING (42K)
    (1771, 'BOND', 'eb_1st_1771_v01_AAB', [
        ('BOOK-KEEPING', r'BOOK-KEEPING', 1),
    ]),

    # MATERA (1810) — 32w town swallowed MATERIA MEDICA (v12, not v13)
    (1810, 'MATERA', 'eb_4th_1810_v12', [
        ('MATERIA MEDICA AND PHARMACY', r'MATERIA MEDICA', 0),
    ]),
    # MATERA (1815) — same pattern
    (1815, 'MATERA', 'eb_5th_1815_v12', [
        ('MATERIA MEDICA AND PHARMACY', r'MATERIA MEDICA', 0),
    ]),

    # ENGRAILED (1842) — heraldry term swallowed ENGRAVING (~15K)
    (1842, 'ENGRAILED', 'eb_7th_1842_v09', [
        ('ENGRAVING', r'Engraving being properly a branch of sculpture', 0),
    ]),
    # ENGRAILED (1860) — same pattern
    (1860, 'ENGRAILED', 'eb_8th_1860_v08', [
        ('ENGRAVING', r'Engraving,', 1),
    ]),

    # PERSHORE (1842) — market town swallowed PERSIA (~20K)
    (1842, 'PERSHORE', 'eb_7th_1842_v17', [
        ('PERSIA', r'From the remotest period of antiquity Persia', 0),
    ]),
    # PERSHORE (1860) — same
    (1860, 'PERSHORE', 'eb_8th_1860_v17', [
        ('PERSIA', r'In illustration of this remark', 0),
    ]),

    # NET (1842) — net definition (~200w) swallowed NETHERLANDS (~32K)
    (1842, 'NET', 'eb_7th_1842_v16', [
        ('NETHERLANDS', r'The decisive battle of Gembloux', 0),
    ]),

    # ZYGOMATICUS (1778) — swallowed entire Vol 10 Appendix (146K)
    # Real ZYGOMATICUS is ~18w definition. Rest is Appendix.
    (1778, 'ZYGOMATICUS', 'eb_2nd_1778_v10_IND-WOO_alt2', [
        ('APPENDIX', r'APPENDIX:', 0),
    ]),

    # PERSONIFYING (1797) — swallowed PERSPECTIVE (~18K)
    (1797, 'PERSONIFYING', 'eb_3rd_1797_v14', [
        ('PERSPECTIVE', r'\*\*PERSPECTIVE\.\*\*', 0),
    ]),

    # BURNING (1810) — swallowed BURNISHING, BURNLEY, BURNS
    (1810, 'BURNING', 'eb_4th_1810_v05', [
        ('BURNS', r'\*\*Burns, Robert\*\*', 30),
    ]),

    # SLAUGHTER (1810-1823) — actually SLAVERY content
    # SLAVE is 37w stub, SLAUGHTER is 14K of slavery content
    # Relabel handled in RELABELS section below

    # INDIAN (1810) — tail of INDIA, merge handled in MERGES section below

    # ================================================================
    # RAG-DISCOVERED SWALLOWED ARTICLES — Session Mar 31, 2026
    # Found via semantic search over embedded corpus
    # ================================================================

    # DRYANDER (1860) swallowed DRYDEN + DRY ROT
    (1860, 'DRYANDER', 'eb_8th_1860_v08', [
        ('DRYDEN', r'Dryden, John, an illustrious English poet', 5),
        ('DRY ROT', r'DRY ROT,\n\nA most destructive', 40),
    ]),

    # ================================================================
    # SWALLOWED ARTICLES — Session Mar 29, 2026
    # These 28 gaps were classified SWALLOWED in gap_classifications.
    # 2 false positives excluded (FORTIFICATION/FOUNDERY plate label,
    # SQUARE-RIGGED/STARCH plate label).
    # ================================================================

    # --- ABEL (1810, 1815, 1823) swallowed ABELARD ---
    (1810, 'ABEL', 'eb_4th_1810_v08', [
        ('ABELARD', r'ABELARD, Peter', 5),
    ]),
    (1815, 'ABEL', 'eb_5th_1815_v01', [
        ('ABELARD', r'ABELARD, Peter', 5),
    ]),
    (1823, 'ABEL', 'eb_6th_1823_v01', [
        ('ABELARD', r'ABELARD, Peter', 5),
    ]),

    # --- NORTH (1797) swallowed NORTHAMPTON + NORTHAMPTONSHIRE ---
    (1797, 'NORTH', 'eb_3rd_1797_v13', [
        ('NORTHAMPTON', r'NORTHAMPTON, a town in England', 80),
        ('NORTHAMPTONSHIRE', r'Northamptonshire, a county of England', 90),
    ]),
    # --- NORTH (1810) swallowed NORTHAMPTON + NORTHAMPTONSHIRE ---
    (1810, 'NORTH', 'eb_4th_1810_v15', [
        ('NORTHAMPTON', r'NORTHAMPTON, a town in England', 75),
        ('NORTHAMPTONSHIRE', r'NORTHAMPTONSHIRE, a county of England', 90),
    ]),

    # --- CUSTOM (1778, 1810) swallowed CUSTOMS ---
    (1778, 'CUSTOM', 'eb_2nd_1778_v03', [
        ('CUSTOMS', r'CUSTOMS, in political economy', 80),
    ]),
    (1810, 'CUSTOM', 'eb_4th_1810_v17', [
        ('CUSTOMS', r'CUSTOMS, in political economy', 80),
    ]),

    # --- PARR (1797) swallowed PARTISAN, PARTNERSHIP, PARTRIDGE ---
    (1797, 'PARR', 'eb_3rd_1797_v13', [
        ('PARTISAN', r'PARTISAN, in the art', 55),
        ('PARTNERSHIP', r'PARTNERSHIP, is a contract', 58),
        ('PARTRIDGE', r'PARTRIDGE, in ornithology', 70),
    ]),

    # --- ICE ICE (1815) swallowed ICE-HOUSE ---
    (1815, 'ICE ICE', 'eb_5th_1815_v11', [
        ('ICE-HOUSE', r'ICE-HOUSE, a repository', 60),
    ]),

    # --- MEDICAL JURISPRUDENCE (1842) swallowed MEDICINE ---
    (1842, 'MEDICAL JURISPRUDENCE', 'eb_7th_1842_v14', [
        ('MEDICINE', r'\nMEDICINE\.\n\nMedicine, in its most extended', 55),
    ]),

    # --- MEASURE (1823) swallowed MECHANICS + MECHANISM ---
    (1823, 'MEASURE', 'eb_6th_1823_v13', [
        ('MECHANICS', r'\nMECHANICS\.\n\n1\. Mechanics is the science', 5),
        ('MECHANISM', r'MECHANISM, either the construction', 65),
    ]),

    # --- JENA (1842) swallowed JEROME + JERSEY ---
    (1842, 'JENA', 'eb_7th_1842_v12', [
        ('JEROME', r'JEROME, St, in Latin', 15),
        ('JERSEY', r'JERSEY, one of a group of islands', 20),
    ]),

    # --- DALMATIA (1842) swallowed DALRYMPLE ---
    (1842, 'DALMATIA', 'eb_7th_1842_v07', [
        ('DALRYMPLE', r'DALRYMPLE, JAMES, Viscount', 3),
    ]),

    # --- SPECIFICS (1797) swallowed SPIRITUAL ---
    (1797, 'SPECIFICS', 'eb_3rd_1797_v17', [
        ('SPIRITUAL', r'SPIRITUAL, in general', 85),
    ]),

    # --- DRAWING (1815) swallowed DRAYTON ---
    (1815, 'DRAWING', 'eb_5th_1815_v07', [
        ('DRAYTON', r'DRAYTON, MICHAEL', 90),
    ]),

    # --- EUXINUS PONTUS (1842) swallowed EVIDENCE ---
    (1842, 'EUXINUS PONTUS', 'eb_7th_1842_v09', [
        ('EVIDENCE', r'EVIDENCE, in Philosophy', 90),
    ]),

    # --- ORDEAL (1797) swallowed ORDER ---
    (1797, 'ORDEAL', 'eb_3rd_1797_v13', [
        ('ORDER', r'ORDER, in architecture', 90),
    ]),

    # --- JOACHIMITES (1842) swallowed JOHNSON ---
    (1842, 'JOACHIMITES', 'eb_7th_1842_v12', [
        ('JOHNSON', r'JOHNSON, or Jonson, Ben', 40),
    ]),

    # --- KING (1842) swallowed KING'S COUNTY ---
    (1842, 'KING', 'eb_7th_1842_v12', [
        ("KING'S COUNTY", r"KING'S COUNTY, an inland county", 60),
    ]),

    # --- JONES (1842) swallowed JOSEPHUS ---
    (1842, 'JONES', 'eb_7th_1842_v12', [
        ('JOSEPHUS', r'JOSEPHUS, the celebrated historian', 75),
    ]),

    # --- ASSOCIATION (1823) swallowed ASSUAN + ASSUMPSIT ---
    (1823, 'ASSOCIATION', 'eb_6th_1823_v03', [
        ('ASSUAN', r'ASSUAN See SYENE', 40),
        ('ASSUMPSIT', r'ASSUMPSIT, in the Law of England', 43),
    ]),

    # --- MEDALLIONS (1823) swallowed MEDIA ---
    (1823, 'MEDALLIONS', 'eb_6th_1823_v13', [
        ('MEDIA', r'MEDIA, now the province of Ghilan', 5),
    ]),

    # --- STORK (1797, 1810, 1815) swallowed STOVE ---
    (1797, 'STORK', 'eb_3rd_1797_v17', [
        ('STOVE', r'STOVE for heating apartments', 0),
    ]),
    (1810, 'STORK', 'eb_4th_1810_v19', [
        ('STOVE', r'STOVE for heating apartments', 0),
    ]),
    (1815, 'STORK', 'eb_5th_1815_v19', [
        ('STOVE', r'STOVE for heating apartments', 0),
    ]),
]


# ============================================================================
# MERGE specifications: (year, source_title, target_title, file_pattern)
# Source article text is appended to target article, then source is deleted.
# Used when the parser split a single article at a mid-text heading.
# ============================================================================

MERGES = [
    # ORDER is the tail of ORATORY — parser split at "§ 2. Of Order."
    (1778, 'ORDER', 'ORATORY', 'eb_2nd_1778_v08'),
    (1797, 'ORDER', 'ORATORY', 'eb_3rd_1797_v13'),
    (1815, 'ORDER', 'ORATORY', 'eb_5th_1815_v15'),
    (1823, 'ORDER', 'ORATORY', 'eb_6th_1823_v15'),

    # PART (1810) is the tail of ORATORY in the same volume
    (1810, 'PART', 'ORATORY', 'eb_4th_1810_v15_NIC'),

    # INDIAN (1810) is the tail of INDIA
    (1810, 'INDIAN', 'INDIA', 'eb_4th_1810_v11'),
]


# ============================================================================
# DELETE specifications: (year, title, file_pattern, min_word_count)
# Removes misattributed articles that are fragments of other articles.
# Only deletes if word_count >= min_word_count (safety check).
# ============================================================================

DELETES = [
    # WEEK (1810 v13) — 88K fragment of MEDICINE article (MEDICINE already exists at 26K)
    (1810, 'WEEK', 'eb_4th_1810_v13_MAT', 50000),

    # STRAIN (1842 v16) — 85K fragment of ORNITHOLOGY (ORNITHOLOGY already 24K in same file)
    (1842, 'STRAIN', 'eb_7th_1842_v16', 50000),

    # WHITE (1842 v13) — 62K fragment of MAGNETISM article
    (1842, 'WHITE', 'eb_7th_1842_v13_SEV', 50000),

    # AAA (1823) — contributor key + Dissertations, not a real article
    (1823, 'AAA', 'eb_6th_1823_v01', 50000),

    # VOCAL (1842 v08) — 37K fragment of English history (mid-sentence, about Henry)
    (1842, 'VOCAL', 'eb_7th_1842_v08', 30000),

    # THUS (1797 v06) — 62K false headword, tail of ETHIOPIA
    (1797, 'THUS', 'eb_3rd_1797_v06', 50000),

    # THUS (1810 v17) — 21K false headword, tail of RUSSIA content
    (1810, 'THUS', 'eb_4th_1810_v17_RHI', 15000),

    # THUS (1797 v16) — 7.4K false headword, tail of SCOTLAND
    (1797, 'THUS', 'eb_3rd_1797_v16', 5000),

    # GENUS IX — medical subsection, not a standalone article
    (1797, 'GENUS IX', 'eb_3rd_1797_v11', 50000),
    (1823, 'GENUS IX', 'eb_6th_1823_v13', 10000),

    # LOGARITHMS OF NUMBERS — numerical log tables, not articles
    (1810, 'LOGARITHMS OF NUMBERS', 'eb_4th_1810_v17_LIE', 10000),
    (1815, 'LOGARITHMS OF NUMBERS', 'eb_5th_1815_v12', 10000),
    (1823, 'LOGARITHMS OF NUMBERS', 'eb_6th_1823_v12', 10000),
]


# ============================================================================
# RELABEL specifications: (year, old_title, new_title, file_pattern, min_wc)
# Renames an article that has the wrong headword.
# ============================================================================

RELABELS = [
    # SWEDEN IS BY NO (1815, 1823) — broken headword, should be SWEDEN
    (1815, 'SWEDEN IS BY NO', 'SWEDEN', 'eb_5th_1815_v20', 10000),
    (1823, 'SWEDEN IS BY NO', 'SWEDEN', 'eb_6th_1823_v20', 10000),

    # SLAUGHTER (1810-1823) — actually SLAVERY content (14K each)
    # SLAVE is a 37w stub; SLAUGHTER got all the slavery article text
    (1810, 'SLAUGHTER', 'SLAVERY', 'eb_4th_1810_v17_OLD', 10000),
    (1815, 'SLAUGHTER', 'SLAVERY', 'eb_5th_1815_v19', 10000),
    (1823, 'SLAUGHTER', 'SLAVERY', 'eb_6th_1823_v19', 10000),

    # AMERICA IS BY NO (1778) — broken headword, should be AMERICA
    (1778, 'AMERICA IS BY NO', 'AMERICA', 'eb_2nd_1778_v01_AA', 10000),
]


def process_fix(year, title, file_pattern, splits, dry_run=False):
    """Apply a single fix."""
    # Find the file
    matches = list(ARTICLES_DIR.glob(f"{file_pattern}*.articles.jsonl"))
    if not matches:
        print(f"  WARNING: No file matching {file_pattern}")
        return 0

    for filepath in matches:
        if filepath.suffix == '.bak':
            continue
        with open(filepath, 'r') as f:
            articles = [json.loads(line) for line in f if line.strip()]

        # Find the article
        target_idx = None
        for i, a in enumerate(articles):
            if a['title'] == title and a['edition_year'] == year:
                target_idx = i
                break

        if target_idx is None:
            continue

        art = articles[target_idx]

        # Handle rename-only (no splits)
        if not splits:
            if title == 'SCOTLAND IS BY NO':
                print(f"  RENAME: '{title}' → 'SCOTLAND' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'SCOTLAND'
            elif title == 'ANTAGONISTS OF HOBBIESTS':
                print(f"  RENAME: '{title}' → 'DISSERTATIONS' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'DISSERTATIONS'
            elif title == 'CLOCK AND WATCH WORK':
                print(f"  RENAME: '{title}' → 'CLOCKS' ({art['word_count']:,}w)")
                if not dry_run:
                    articles[target_idx]['title'] = 'CLOCKS'
            elif title == 'HYDRODYNAMICS':
                # Strip trailing INDEX and DIRECTIONS if present
                text = art['text']
                idx_match = re.search(r'\n\nINDEX[,.]', text)
                dir_match = re.search(r'\n\nDIRECTIONS FOR PLACING', text)
                cut_point = None
                if idx_match:
                    cut_point = idx_match.start()
                elif dir_match:
                    cut_point = dir_match.start()
                if cut_point:
                    old_wc = art['word_count']
                    art['text'] = text[:cut_point].strip()
                    art['word_count'] = len(art['text'].split())
                    print(f"  TRIM: '{title}' trailing matter removed ({old_wc:,}w → {art['word_count']:,}w)")
                    if not dry_run:
                        articles[target_idx] = art
                else:
                    print(f"  SKIP: '{title}' — no trailing INDEX found")
                    return 0
            else:
                print(f"  SKIP: '{title}' — no splits defined")
                return 0

            if not dry_run:
                with open(filepath, 'w') as f:
                    for a in articles:
                        f.write(json.dumps(a, ensure_ascii=False) + '\n')
            return 1

        # Apply splits
        new_articles = split_article(articles, target_idx, splits)

        if len(new_articles) <= 1:
            # Check if it's a rename (single article with different title)
            if new_articles and new_articles[0]['title'] != title:
                print(f"  RENAME: '{title}' → '{new_articles[0]['title']}' ({new_articles[0]['word_count']:,}w)")
            else:
                print(f"  WARNING: No splits found for {title}")
                return 0

        if len(new_articles) > 1:
            print(f"  SPLIT: '{title}' ({art['word_count']:,}w) → {len(new_articles)} articles:")
            for na in new_articles:
                print(f"    {na['title']:40s} {na['word_count']:>8,}w")

        if not dry_run:
            # Replace the original article with the split versions
            articles[target_idx:target_idx + 1] = new_articles
            with open(filepath, 'w') as f:
                for a in articles:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')

        return len(new_articles) - 1  # excess articles added

    print(f"  WARNING: Article '{title}' not found in {file_pattern}")
    return 0


def process_merge(year, source_title, target_title, file_pattern, dry_run=False):
    """Merge source article text into target article, then delete source."""
    matches = list(ARTICLES_DIR.glob(f"{file_pattern}*.articles.jsonl"))
    if not matches:
        print(f"  WARNING: No file matching {file_pattern}")
        return 0

    for filepath in matches:
        if filepath.suffix == '.bak':
            continue
        with open(filepath, 'r') as f:
            articles = [json.loads(line) for line in f if line.strip()]

        source_idx = target_idx = None
        for i, a in enumerate(articles):
            if a['title'] == source_title and a['edition_year'] == year:
                source_idx = i
            if a['title'] == target_title and a['edition_year'] == year:
                target_idx = i

        if source_idx is None or target_idx is None:
            continue

        source = articles[source_idx]
        target = articles[target_idx]

        old_target_wc = target['word_count']
        new_text = target['text'] + '\n\n' + source['text']
        new_wc = len(new_text.split())

        print(f"  MERGE: '{source_title}' ({source['word_count']:,}w) → '{target_title}' "
              f"({old_target_wc:,}w → {new_wc:,}w)")

        if not dry_run:
            articles[target_idx]['text'] = new_text
            articles[target_idx]['word_count'] = new_wc
            articles[target_idx]['char_end'] = source['char_end']
            del articles[source_idx]
            with open(filepath, 'w') as f:
                for a in articles:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
        return 1

    print(f"  WARNING: Could not find both '{source_title}' and '{target_title}' in {file_pattern}")
    return 0


def process_delete(year, title, file_pattern, min_wc, dry_run=False):
    """Delete a misattributed article (only if word_count >= min_wc)."""
    matches = list(ARTICLES_DIR.glob(f"{file_pattern}*.articles.jsonl"))
    if not matches:
        print(f"  WARNING: No file matching {file_pattern}")
        return 0

    for filepath in matches:
        if filepath.suffix == '.bak':
            continue
        with open(filepath, 'r') as f:
            articles = [json.loads(line) for line in f if line.strip()]

        target_idx = None
        for i, a in enumerate(articles):
            if a['title'] == title and a['edition_year'] == year and a['word_count'] >= min_wc:
                target_idx = i
                break

        if target_idx is None:
            continue

        art = articles[target_idx]
        print(f"  DELETE: '{title}' ({art['word_count']:,}w) — misattributed fragment")

        if not dry_run:
            del articles[target_idx]
            with open(filepath, 'w') as f:
                for a in articles:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
        return 1

    print(f"  WARNING: '{title}' not found (or below {min_wc}w threshold) in {file_pattern}")
    return 0


def process_relabel(year, old_title, new_title, file_pattern, min_wc, dry_run=False):
    """Rename a mislabeled article."""
    matches = list(ARTICLES_DIR.glob(f"{file_pattern}*.articles.jsonl"))
    if not matches:
        print(f"  WARNING: No file matching {file_pattern}")
        return 0

    for filepath in matches:
        if filepath.suffix == '.bak':
            continue
        with open(filepath, 'r') as f:
            articles = [json.loads(line) for line in f if line.strip()]

        for i, a in enumerate(articles):
            if a['title'] == old_title and a['edition_year'] == year and a['word_count'] >= min_wc:
                print(f"  RELABEL: '{old_title}' → '{new_title}' ({a['word_count']:,}w)")
                if not dry_run:
                    articles[i]['title'] = new_title
                    with open(filepath, 'w') as f:
                        for a in articles:
                            f.write(json.dumps(a, ensure_ascii=False) + '\n')
                return 1

    print(f"  WARNING: '{old_title}' not found in {file_pattern}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Fix mega-articles")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    total_changes = 0

    # Phase 1: Splits
    print(f"{'DRY RUN: ' if args.dry_run else ''}Phase 1: Splitting {len(FIXES)} mega-articles...\n")
    for year, title, file_pattern, splits in FIXES:
        print(f"\n{year} {title}:")
        changes = process_fix(year, title, file_pattern, splits, dry_run=args.dry_run)
        total_changes += changes

    # Phase 2: Merges (must run after splits since splits may affect same files)
    print(f"\n\n{'DRY RUN: ' if args.dry_run else ''}Phase 2: Merging {len(MERGES)} split articles...\n")
    for year, source, target, file_pattern in MERGES:
        print(f"\n{year} {source} → {target}:")
        changes = process_merge(year, source, target, file_pattern, dry_run=args.dry_run)
        total_changes += changes

    # Phase 3: Deletes
    print(f"\n\n{'DRY RUN: ' if args.dry_run else ''}Phase 3: Deleting {len(DELETES)} misattributed articles...\n")
    for year, title, file_pattern, min_wc in DELETES:
        print(f"\n{year} {title}:")
        changes = process_delete(year, title, file_pattern, min_wc, dry_run=args.dry_run)
        total_changes += changes

    # Phase 4: Relabels
    print(f"\n\n{'DRY RUN: ' if args.dry_run else ''}Phase 4: Relabeling {len(RELABELS)} mislabeled articles...\n")
    for year, old_title, new_title, file_pattern, min_wc in RELABELS:
        print(f"\n{year} {old_title} → {new_title}:")
        changes = process_relabel(year, old_title, new_title, file_pattern, min_wc, dry_run=args.dry_run)
        total_changes += changes

    print(f"\n\nTotal changes: {total_changes}")


if __name__ == "__main__":
    main()
