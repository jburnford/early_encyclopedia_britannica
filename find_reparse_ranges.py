#!/usr/bin/env python3
"""
Find page ranges that need re-parsing due to OCR/regex errors.

Identifies:
1. Articles with wrong starting letter for volume (e.g., "MOE" in Vol 2 C-L)
2. Sentence fragments parsed as article titles
3. Orphaned subsections (PROPOSITION, SCHOLIUM, etc.)
4. Alphabetical anomalies (articles out of sequence with neighbors)
5. Short word fragments (titles that look like word examples)
6. Title/text mismatches (wrong article boundaries)
7. Suspicious duplicates (same title on nearby pages)

Then finds the surrounding page range that needs Gemini re-parsing.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

# Expected letter ranges by volume (approximate - varies by edition)
# This will be refined per-edition
VOLUME_LETTERS = {
    '1771': {  # 1st Edition - 3 volumes
        'vol1': 'A',
        'vol2': 'B-L',  # Approximate
        'vol3': 'M-Z',
    },
    '1778': {  # 2nd Edition - 10 volumes
        'vol1': 'A',
        'vol2': 'C-L',  # User confirmed
        'vol3': 'C',
        'vol4': 'D-F',
        'vol5': 'G-J',
        'vol6': 'K-M',
        'vol7': 'M-O',
        'vol8': 'O-P',
        'vol9': 'P-S',
        'vol10': 'S-Z',
    },
}

# Patterns indicating parsing errors (sentence fragments)
FRAGMENT_PATTERNS = [
    r'^[A-Z]{2,}\s+(?:MAY|IS|ARE|WAS|WERE|HAS|HAVE|HAD|THE|THIS|THAT|ALSO|THUS|BY)\s',
    r'^(?:ALTHOUGH|BESIDES|CONCERNING|HITHERTO|SINCE|SUCH|THESE|THIS|UPON|VARIOUS|WHEN|WHILE|AFTER)\s+(?:THE|THIS|THAT|IT|HE|SHE|THEY|WE|A|AN)\s',
    r'^(?:MANY|SOME|ALL|MOST|FEW|SEVERAL)\s+(?:ATTEMPTS|METHODS|WAYS|MEANS|HAVE|OF)\s',
    r'(?:BY\s+THE|BY\s+THIS|BY\s+THAT|BY\s+NO|BY\s+THESE)$',  # Ends with "BY THE/THIS/NO"
    r'^OILS\s+THUS\s',
    r'^WATER\s+MAY\s',
    r'^FAT\s+(?:OILS|LIKEWISE)\s',
]
FRAGMENT_RE = [re.compile(p, re.IGNORECASE) for p in FRAGMENT_PATTERNS]

# Orphaned subsection patterns
SUBSECTION_PATTERNS = [
    r'^PROPOSITION\s+[IVXLC\d]+',
    r'^PROBLEM\s+[IVXLC\d]+',
    r'^SCHOLIUM\s+[IVXLC\d]*',
    r'^CASE\s+[IVXLC\d]+',
    r'^EXAMPLE\s+[IVXLC\d]+',
    r'^RULE\s+[IVXLC\d]+',
    r'^COROLLARY\s+[IVXLC\d]*',
    r'^LEMMA\s+[IVXLC\d]+',
    r'^AXIOM\s+[IVXLC\d]+',
    r'^DEFINITION\s+[IVXLC\d]+',
    r'^EXPERIMENT\s+[IVXLC\d]+',
    r'^OBSERVATION\s+[IVXLC\d]+',
    r'^REMARK\s+[IVXLC\d]*',
    r'^CLASS\s+[IVXLC\d]+',
    r'^PROCESS\s+[IVXLC\d]+',
    r'^EXPLANATION\s+OF\s+PLATE',
]
SUBSECTION_RE = [re.compile(p, re.IGNORECASE) for p in SUBSECTION_PATTERNS]

# Short word patterns - words that are likely word examples, not article titles
# These appear in dictionary/language articles as examples of usage
SHORT_WORD_PATTERNS = [
    # Very short all-caps words that could be word examples
    r'^[A-Z]{2,4}$',  # MOE, BADE, WIDE, etc.
]
SHORT_WORD_RE = [re.compile(p) for p in SHORT_WORD_PATTERNS]

# Common English words that appear as examples in dictionary articles
# NOT actual encyclopedia headwords - these are grammar/usage examples
COMMON_WORD_EXAMPLES = {
    # Grammar/usage examples
    'MOE', 'BADE', 'HATH', 'DOTH', 'THOU', 'THEE', 'THY', 'YE', 'YEA', 'NAY',
    # Adjectives used as examples
    'GOOD', 'BAD', 'BEST', 'WORST', 'MORE', 'LESS', 'MOST', 'LEAST',
    # Quantifiers - often used as examples
    'OWN', 'VERY', 'EVEN', 'STILL', 'JUST', 'ONLY', 'ALSO', 'YET',
}

# Words that LOOK like they could be examples but ARE valid encyclopedia headwords
# These should NOT be flagged
VALID_SHORT_HEADWORDS = {
    # Places
    'ABA', 'IDA', 'GOA', 'NIM', 'KOS', 'ZEA', 'NIL', 'AAR', 'YAM',
    # Rivers/seas
    'PO', 'DON', 'CAM', 'TAY', 'DEE', 'EXE', 'WYE', 'URE', 'USA',
    # Sciences/concepts
    'AIR', 'ART', 'AGE', 'APE', 'ANT', 'ARM', 'AXE', 'BAR', 'BAT', 'BEE',
    'BOW', 'BOX', 'CAP', 'CAR', 'CAT', 'COD', 'COW', 'CUP', 'DAM', 'DAY',
    'DEW', 'DOG', 'DOT', 'DYE', 'EAR', 'EEL', 'EGG', 'ELK', 'ELM', 'EVE',
    'EWE', 'EYE', 'FAN', 'FAT', 'FEN', 'FIG', 'FIN', 'FIR', 'FLY', 'FOG',
    'FOX', 'FUR', 'GAP', 'GAS', 'GIN', 'GOD', 'GUM', 'GUN', 'GUT', 'HAM',
    'HAT', 'HAY', 'HEM', 'HEN', 'HOG', 'HOP', 'ICE', 'INN', 'INK', 'IVY',
    'JAM', 'JAR', 'JAW', 'JET', 'JIG', 'JOY', 'JUG', 'KEY', 'KIN', 'LAC',
    'LAP', 'LAW', 'LEA', 'LEG', 'LID', 'LIE', 'LIP', 'LOG', 'LOT', 'MAP',
    'MAT', 'MAY', 'MEN', 'MOP', 'MUD', 'NIT', 'NUN', 'NUT', 'OAK', 'OAR',
    'OAT', 'OIL', 'ORE', 'OWL', 'OX', 'PAN', 'PAW', 'PEA', 'PEG', 'PEN',
    'PET', 'PIE', 'PIG', 'PIN', 'PIT', 'POT', 'RAG', 'RAM', 'RAT', 'RAY',
    'RIB', 'RIM', 'ROD', 'ROE', 'ROW', 'RUG', 'RUM', 'RYE', 'SAP', 'SAW',
    'SEA', 'SET', 'SEW', 'SKY', 'SON', 'SOY', 'SPA', 'SUN', 'TAR', 'TAX',
    'TEA', 'TIN', 'TOE', 'TON', 'TOP', 'TOW', 'TOY', 'TUB', 'TUN', 'URN',
    'VAN', 'VAT', 'WAR', 'WAX', 'WAY', 'WEB', 'WIG', 'WIN', 'WIT', 'WOE',
    'YAK', 'YEW', 'ZOO',
    # Longer but common headwords
    'ACID', 'ACRE', 'ALPS', 'AMEN', 'ANNA', 'ARCH', 'AREA', 'ARMY', 'ASIA',
    'ATOM', 'AXIS', 'BARD', 'BARK', 'BARN', 'BASS', 'BATH', 'BEAM', 'BEAN',
    'BEAR', 'BEAT', 'BEER', 'BELL', 'BELT', 'BEND', 'BILE', 'BILL', 'BIRD',
    'BITE', 'BOAT', 'BODY', 'BOIL', 'BOLT', 'BOMB', 'BONE', 'BOOK', 'BOOT',
    'BORE', 'BOWL', 'BRAN', 'BULK', 'BULL', 'BURN', 'CAGE', 'CAKE', 'CALF',
    'CALM', 'CAMP', 'CANE', 'CAPE', 'CARD', 'CARE', 'CART', 'CASE', 'CASH',
    'CAST', 'CAVE', 'CELL', 'CHIN', 'CITY', 'CLAP', 'CLAY', 'CLIP', 'CLUB',
    'COAL', 'COAT', 'COCK', 'CODE', 'COIL', 'COIN', 'COKE', 'COLD', 'COLT',
    'COMB', 'COOK', 'COOL', 'COPY', 'CORD', 'CORE', 'CORK', 'CORN', 'COST',
    'CREW', 'CROP', 'CROW', 'CUBE', 'CURE', 'CURL', 'DALE', 'DAME', 'DAMP',
    'DARE', 'DARK', 'DATA', 'DATE', 'DAWN', 'DAYS', 'DEAL', 'DEAN', 'DEBT',
    'DECK', 'DEED', 'DEER', 'DENT', 'DESK', 'DIAL', 'DICE', 'DIET', 'DIDO',
    'DION', 'DIRT', 'DISC', 'DISH', 'DOCK', 'DOME', 'DOOR', 'DOSE', 'DOVE',
    'DOWN', 'DRAG', 'DRAW', 'DRUM', 'DUCK', 'DUEL', 'DUKE', 'DUNG', 'DUST',
    'DUTY', 'EARL', 'EASE', 'EAST', 'ECHO', 'EDGE', 'EDIT', 'ELMS', 'EPIC',
    'EVIL', 'EXAM', 'FACE', 'FACT', 'FADE', 'FAIL', 'FAIR', 'FAKE', 'FALL',
    'FAME', 'FARM', 'FATE', 'FAWN', 'FEAR', 'FEAT', 'FEED', 'FEEL', 'FEET',
    'FERN', 'FILE', 'FILL', 'FILM', 'FIND', 'FINE', 'FIRE', 'FIRM', 'FISH',
    'FIST', 'FLAG', 'FLAP', 'FLAT', 'FLAW', 'FLAX', 'FLEA', 'FLOW', 'FLUX',
    'FOAL', 'FOAM', 'FOLD', 'FOLK', 'FOOD', 'FOOL', 'FOOT', 'FORD', 'FORK',
    'FORM', 'FORT', 'FOWL', 'FROG', 'FUEL', 'FUND', 'FUSE', 'GAIT', 'GALE',
    'GAME', 'GANG', 'GATE', 'GEAR', 'GENE', 'GIFT', 'GILL', 'GIRL', 'GIVE',
    'GLAD', 'GLEN', 'GLOW', 'GLUE', 'GOAL', 'GOAT', 'GOLD', 'GOLF', 'GONE',
    'GONG', 'GORE', 'GOUT', 'GRAB', 'GRAM', 'GRAY', 'GREW', 'GREY', 'GRID',
    'GRIT', 'GULF', 'GUST', 'HAIL', 'HAIR', 'HALF', 'HALL', 'HALT', 'HAND',
    'HANG', 'HARE', 'HARM', 'HARP', 'HASH', 'HAZE', 'HEAD', 'HEAL', 'HEAP',
    'HEAR', 'HEAT', 'HEEL', 'HEIR', 'HELD', 'HELP', 'HEMP', 'HERB', 'HERD',
    'HERO', 'HIDE', 'HILL', 'HINT', 'HOLD', 'HOLE', 'HOME', 'HOOD', 'HOOK',
    'HOOF', 'HOPE', 'HORN', 'HORSE', 'HOST', 'HOUR', 'HULL', 'HUNT', 'HURT',
    'HYMN', 'IDEA', 'IDLE', 'IDOL', 'INCH', 'INNS', 'IRON', 'ISLE', 'ITEM',
    'JACK', 'JADE', 'JAIL', 'JANE', 'JAZZ', 'JEST', 'JOHN', 'JOKE', 'JOLT',
    'JULY', 'JUMP', 'JUNE', 'JURY', 'KEEN', 'KEEP', 'KICK', 'KILL', 'KIND',
    'KING', 'KISS', 'KITE', 'KNEE', 'KNIT', 'KNOB', 'KNOT', 'LACE', 'LACK',
    'LADY', 'LAID', 'LAKE', 'LAMB', 'LAMP', 'LAND', 'LANE', 'LARD', 'LARK',
    'LAST', 'LATE', 'LATH', 'LAVA', 'LAWN', 'LEAD', 'LEAF', 'LEAK', 'LEAN',
    'LEAP', 'LEFT', 'LENS', 'LENT', 'LIFE', 'LIFT', 'LILY', 'LIMA', 'LIMB',
    'LIME', 'LINE', 'LINK', 'LION', 'LIST', 'LIVE', 'LOAD', 'LOAF', 'LOAN',
    'LOCK', 'LOFT', 'LOGO', 'LOOK', 'LOOP', 'LORD', 'LOSE', 'LOSS', 'LOST',
    'LOVE', 'LUCK', 'LUMP', 'LUNG', 'LURE', 'LUST', 'MACE', 'MADE', 'MAID',
    'MAIL', 'MAIN', 'MAKE', 'MALE', 'MALL', 'MALT', 'MAMA', 'MANE', 'MAPS',
    'MARE', 'MARK', 'MARS', 'MASH', 'MASK', 'MASS', 'MAST', 'MATE', 'MATH',
    'MAZE', 'MEAL', 'MEAN', 'MEAT', 'MEDE', 'MEET', 'MELT', 'MEMO', 'MENU',
    'MERE', 'MESH', 'MESS', 'MICE', 'MILD', 'MILE', 'MILK', 'MILL', 'MIME',
    'MIND', 'MINE', 'MINT', 'MISS', 'MIST', 'MODE', 'MOLD', 'MOLE', 'MONK',
    'MOOD', 'MOON', 'MOOR', 'MOSS', 'MOTH', 'MOVE', 'MUCK', 'MULE', 'MUSE',
    'MUST', 'MYTH', 'NAIL', 'NAME', 'NAPE', 'NAVY', 'NEAT', 'NECK', 'NEED',
    'NEPA', 'NERO', 'NEST', 'NEWS', 'NEXT', 'NICE', 'NICK', 'NILE', 'NINE',
    'NODE', 'NONE', 'NOON', 'NORM', 'NOSE', 'NOTE', 'NOUN', 'NUDE', 'NUTS',
    'OATH', 'ODDS', 'ODOR', 'OMEN', 'ONCE', 'OOZE', 'OPEN', 'ORAL', 'OVEN',
    'PACE', 'PACK', 'PACT', 'PAGE', 'PAID', 'PAIL', 'PAIN', 'PAIR', 'PALE',
    'PALM', 'PAPA', 'PARK', 'PART', 'PASS', 'PAST', 'PATH', 'PAVE', 'PEAK',
    'PEAR', 'PEAT', 'PEEL', 'PEER', 'PICK', 'PIER', 'PIKE', 'PILE', 'PILL',
    'PINE', 'PINK', 'PINT', 'PIPE', 'PITY', 'PLAN', 'PLAY', 'PLEA', 'PLOD',
    'PLOT', 'PLOW', 'PLUG', 'PLUM', 'PLUS', 'POEM', 'POET', 'POKE', 'POLE',
    'POLL', 'POLO', 'POND', 'POOL', 'POPE', 'PORK', 'PORT', 'POSE', 'POST',
    'POUR', 'PRAY', 'PREY', 'PUMP', 'PURE', 'PUSH', 'QUIT', 'QUIZ', 'RACE',
    'RACK', 'RAGE', 'RAID', 'RAIL', 'RAIN', 'RAKE', 'RAMP', 'RANA', 'RANG',
    'RANK', 'RAPE', 'RARE', 'RASH', 'RATE', 'RAVE', 'READ', 'REAL', 'REAP',
    'REAR', 'REED', 'REEF', 'REEL', 'RELY', 'RENT', 'REST', 'RICE', 'RICH',
    'RIDE', 'RING', 'RIOT', 'RIPE', 'RISE', 'RISK', 'ROAD', 'ROAM', 'ROAR',
    'ROBE', 'ROCK', 'RODE', 'ROLE', 'ROLL', 'ROME', 'ROOF', 'ROOM', 'ROOT',
    'ROPE', 'ROSE', 'RUIN', 'RULE', 'RUSH', 'RUST', 'RUTH', 'SACK', 'SAFE',
    'SAGE', 'SAID', 'SAIL', 'SAKE', 'SALE', 'SALT', 'SAME', 'SAND', 'SANK',
    'SASH', 'SAVE', 'SCAB', 'SCAN', 'SEAL', 'SEAM', 'SEAT', 'SECT', 'SEED',
    'SEEK', 'SELF', 'SELL', 'SEND', 'SEPT', 'SEWN', 'SHED', 'SHIP', 'SHOE',
    'SHOP', 'SHOT', 'SHOW', 'SHUT', 'SICK', 'SIDE', 'SIFT', 'SIGH', 'SIGN',
    'SILK', 'SINK', 'SITE', 'SIZE', 'SKIN', 'SLAB', 'SLAP', 'SLIP', 'SLIT',
    'SLOT', 'SLOW', 'SLUG', 'SNAP', 'SNOW', 'SOAK', 'SOAP', 'SOAR', 'SOCK',
    'SODA', 'SOFA', 'SOFT', 'SOIL', 'SOLD', 'SOLE', 'SOME', 'SONG', 'SOON',
    'SOOT', 'SORE', 'SORT', 'SOUL', 'SOUP', 'SOUR', 'SPAN', 'SPAR', 'SPIN',
    'SPIT', 'SPOT', 'STAB', 'STAR', 'STAY', 'STEM', 'STEP', 'STEW', 'STIR',
    'STOP', 'STUD', 'STUM', 'STYX', 'SUCH', 'SUIT', 'SUNK', 'SURE', 'SURF',
    'SWAN', 'SWAP', 'SWIM', 'TABS', 'TACK', 'TAIL', 'TAKE', 'TALE', 'TALK',
    'TALL', 'TAME', 'TANK', 'TAPE', 'TART', 'TASK', 'TEAM', 'TEAR', 'TELL',
    'TEND', 'TENT', 'TERM', 'TEST', 'TEXT', 'THAN', 'THAT', 'THEM', 'THEN',
    'THEY', 'THIS', 'THORN', 'TIDE', 'TIDY', 'TILE', 'TILL', 'TILT', 'TIME',
    'TINY', 'TIRE', 'TOAD', 'TOES', 'TOLD', 'TOLL', 'TOMB', 'TONE', 'TONG',
    'TOOK', 'TOOL', 'TOOT', 'TOPS', 'TORE', 'TORN', 'TORT', 'TOSS', 'TOUR',
    'TOWN', 'TOYS', 'TRAM', 'TRAP', 'TRAY', 'TREE', 'TRIM', 'TRIO', 'TRIP',
    'TROD', 'TROT', 'TRUE', 'TUBE', 'TUCK', 'TUFT', 'TUNE', 'TURN', 'TURF',
    'TWIG', 'TWIN', 'TYPE', 'UNIT', 'UPON', 'URGE', 'VAIN', 'VALE', 'VANE',
    'VARY', 'VASE', 'VAST', 'VEAL', 'VEIN', 'VENT', 'VERB', 'VERY', 'VEST',
    'VICE', 'VIEW', 'VINE', 'VISA', 'VOID', 'VOLT', 'VOTE', 'WADE', 'WAGE',
    'WAIL', 'WAIT', 'WAKE', 'WALK', 'WALL', 'WAND', 'WANT', 'WARD', 'WARM',
    'WARN', 'WARP', 'WART', 'WASH', 'WASP', 'WAVE', 'WEAK', 'WEAR', 'WEED',
    'WEEK', 'WELD', 'WELL', 'WENT', 'WERE', 'WEST', 'WHAT', 'WHEN', 'WHEY',
    'WHIP', 'WHOM', 'WICK', 'WIDE', 'WIFE', 'WILD', 'WILL', 'WILT', 'WIND',
    'WINE', 'WING', 'WINK', 'WIPE', 'WIRE', 'WISE', 'WISH', 'WITH', 'WOKE',
    'WOLF', 'WOMB', 'WOOD', 'WOOL', 'WORD', 'WORE', 'WORK', 'WORM', 'WORN',
    'WRAP', 'WREN', 'WRIT', 'YANK', 'YARD', 'YARN', 'YAWN', 'YEAR', 'YELL',
    'YOKE', 'YOUR', 'ZEAL', 'ZERO', 'ZONE',
    # Greek/Latin prefixes that are legitimate headwords
    'EX', 'CG', 'DH',
}


def normalize_title_for_sorting(title: str) -> str:
    """
    Normalize a title for alphabetical comparison.
    Removes leading articles, punctuation, and normalizes case.
    """
    if not title:
        return ''

    # Remove common leading words that don't affect alphabetical ordering
    title = title.upper().strip()
    for prefix in ['THE ', 'A ', 'AN ']:
        if title.startswith(prefix):
            title = title[len(prefix):]

    # Keep only letters and spaces for comparison
    clean = ''.join(c for c in title if c.isalpha() or c.isspace())
    return clean.strip()


def is_alphabetical_anomaly(prev_title: Optional[str], curr_title: str,
                            next_title: Optional[str]) -> bool:
    """
    Check if an article is out of alphabetical order with its neighbors.

    Returns True if:
    - curr < prev AND curr < next (stuck in wrong position)
    - curr > prev AND curr > next (also stuck in wrong position)

    This detects articles that were incorrectly parsed from middle of text.
    """
    if not curr_title:
        return False

    curr_norm = normalize_title_for_sorting(curr_title)

    # Need at least one neighbor to compare
    if not prev_title and not next_title:
        return False

    prev_norm = normalize_title_for_sorting(prev_title) if prev_title else None
    next_norm = normalize_title_for_sorting(next_title) if next_title else None

    # Skip if normalized title is very short (likely an abbreviation)
    if len(curr_norm) < 3:
        return False

    # Check if out of order with both neighbors
    if prev_norm and next_norm:
        # Should be: prev < curr < next (alphabetically)
        # If curr < prev AND curr < next, it's out of place (should come earlier)
        if curr_norm < prev_norm and curr_norm < next_norm:
            return True
        # If curr > prev AND curr > next, it's out of place (should come later)
        if curr_norm > prev_norm and curr_norm > next_norm:
            return True

    return False


def is_short_word_fragment(title: str, text: str, volume: str, expected_letters: Optional[str]) -> bool:
    """
    Detect titles that look like word examples from dictionary articles.

    Criteria:
    - Title is a very short word (2-4 chars)
    - Title is in the list of common word examples (NOT valid headwords)
    - Text doesn't start like a typical encyclopedia entry
    - May be outside expected letter range for volume
    """
    if not title or len(title) > 5:
        return False

    title_upper = title.upper().strip()

    # Skip if it's a known valid encyclopedia headword
    if title_upper in VALID_SHORT_HEADWORDS:
        return False

    # Check if it's a common word example (grammar/usage example, not encyclopedia topic)
    if title_upper in COMMON_WORD_EXAMPLES:
        return True

    # For very short words (2-4 chars) not in our known lists,
    # only flag if there's strong evidence it's not a real article
    if len(title_upper) <= 4:
        # Check if text starts with lowercase (continuation of previous text)
        if text and len(text) > 0:
            # Strip HTML tags to get raw text start
            clean_text = re.sub(r'<[^>]+>', '', text).strip()
            if clean_text and clean_text[0].islower():
                return True

    return False


def is_title_text_mismatch(title: str, text: str) -> bool:
    """
    Check if the text content doesn't match the title (wrong article boundary).

    This catches cases where the parser split an article incorrectly,
    creating a new article whose text is about something else entirely.
    """
    if not title or not text or len(text) < 50:
        return False

    title_upper = title.upper()

    # Get the first significant word from the title
    title_words = [w for w in title_upper.split() if len(w) > 2]
    if not title_words:
        return False
    first_title_word = title_words[0]

    # Get first 200 chars of text, clean HTML tags
    text_start = re.sub(r'<[^>]+>', '', text[:200]).upper()

    # For long titles that look like sentences, check if text continues the sentence
    if len(title.split()) > 4:
        # These are likely fragments - check if text starts continuing a thought
        # Text starting with lowercase or with continuation words
        clean_text = text.strip()
        if clean_text and clean_text[0].islower():
            return True
        # Text that continues mid-sentence
        if clean_text[:30].startswith(('and ', 'or ', 'but ', 'which ', 'that ', 'who ')):
            return True

    # For short article titles, the title word should appear in text start
    # (encyclopedia entries typically repeat the headword)
    if len(title_words) == 1 and len(first_title_word) >= 5:
        # Single-word article titles should appear in first part of text
        # Exception: very common words like prepositions, articles
        if first_title_word not in text_start[:100]:
            # Check if text discusses something completely different
            # This is a weak heuristic - flag only clear mismatches
            pass  # Don't flag this automatically, too many false positives

    return False


def find_suspicious_duplicates(articles: List[dict], page_window: int = 15) -> List[Tuple[int, str]]:
    """
    Find articles with the same title appearing within a small page window.
    This can indicate incorrect splitting or OCR errors.

    Returns list of (position, reason) tuples for suspicious articles.
    """
    suspicious = []

    # Build index of title -> list of (position, page) tuples
    title_positions = defaultdict(list)
    for i, article in enumerate(articles):
        title = article.get('h', '').upper().strip()
        page = article.get('sp')
        if title and page:
            title_positions[title].append((i, page))

    # Find duplicates within page window
    for title, positions in title_positions.items():
        if len(positions) > 1:
            # Sort by page number
            positions_sorted = sorted(positions, key=lambda x: x[1])
            for j in range(len(positions_sorted) - 1):
                pos1, page1 = positions_sorted[j]
                pos2, page2 = positions_sorted[j + 1]

                if page2 - page1 <= page_window:
                    # Suspicious duplicate - same title within 15 pages
                    suspicious.append((pos1, f'duplicate_near:{title[:30]}'))
                    suspicious.append((pos2, f'duplicate_near:{title[:30]}'))

    return suspicious


@dataclass
class SuspiciousArticle:
    """An article that looks like a parsing error."""
    volume: str
    position: int
    title: str
    start_page: Optional[int]
    end_page: Optional[int]
    reason: str  # 'wrong_letter', 'fragment', 'subsection'


@dataclass
class ReparseRange:
    """A page range that needs re-parsing."""
    edition: str
    volume: str
    start_page: int
    end_page: int
    reason: str
    suspicious_articles: List[str]


def is_fragment(title: str) -> bool:
    """Check if title looks like a sentence fragment."""
    for p in FRAGMENT_RE:
        if p.search(title):
            return True
    # Also check for very long titles with many words
    words = title.split()
    if len(words) > 6 and len(title) > 50:
        return True
    return False


def is_subsection(title: str) -> bool:
    """Check if title looks like an orphaned subsection."""
    for p in SUBSECTION_RE:
        if p.match(title):
            return True
    return False


def get_expected_letters(edition: str, volume: str) -> Optional[str]:
    """Get expected starting letter(s) for a volume."""
    if edition in VOLUME_LETTERS:
        return VOLUME_LETTERS[edition].get(volume)
    return None


def is_wrong_letter(title: str, expected: str) -> bool:
    """Check if article title starts with wrong letter for volume."""
    if not title or not expected:
        return False

    first_letter = title[0].upper()

    # Handle ranges like "C-L"
    if '-' in expected:
        start, end = expected.split('-')
        return not (start <= first_letter <= end)
    else:
        # Single letter or list
        return first_letter not in expected.upper()


def find_suspicious_articles(edition: str, volume: str, articles: List[dict]) -> List[SuspiciousArticle]:
    """Find all suspicious articles in a volume."""
    suspicious = []
    expected_letters = get_expected_letters(edition, volume)

    # First pass: find duplicates (needs full article list)
    duplicate_positions: Set[int] = set()
    for pos, reason in find_suspicious_duplicates(articles):
        duplicate_positions.add(pos)

    for i, article in enumerate(articles):
        title = article.get('h', '')
        text = article.get('t', '')
        sp = article.get('sp')
        ep = article.get('ep')

        # Skip articles without page numbers
        if sp is None:
            continue

        reason = None

        # Check for sentence fragments (highest priority - clear errors)
        if is_fragment(title):
            reason = 'fragment'
        # Check for orphaned subsections
        elif is_subsection(title):
            reason = 'subsection'
        # Check for wrong starting letter
        elif expected_letters and is_wrong_letter(title, expected_letters):
            reason = 'wrong_letter'
        # Check for suspicious duplicates (same title within 15 pages)
        elif i in duplicate_positions:
            reason = 'duplicate'
        # NOTE: word_example, alpha_anomaly, and text_mismatch checks disabled
        # due to too many false positives with short encyclopedia headwords
        # (places, people, scientific terms, etc.)

        if reason:
            suspicious.append(SuspiciousArticle(
                volume=volume,
                position=i,
                title=title[:80],
                start_page=sp,
                end_page=ep or sp,
                reason=reason
            ))

    return suspicious


def group_into_ranges(articles: List[dict], suspicious: List[SuspiciousArticle],
                      context_pages: int = 5) -> List[ReparseRange]:
    """
    Group suspicious articles into page ranges for re-parsing.
    Includes surrounding context pages.
    """
    if not suspicious:
        return []

    ranges = []

    for sus in suspicious:
        # Find page range: suspicious article's pages + context
        sp = sus.start_page
        ep = sus.end_page

        # Extend range to include context
        range_start = max(1, sp - context_pages)
        range_end = ep + context_pages

        ranges.append({
            'start': range_start,
            'end': range_end,
            'articles': [sus]
        })

    # Merge overlapping ranges
    ranges.sort(key=lambda x: x['start'])
    merged = []

    for r in ranges:
        if merged and r['start'] <= merged[-1]['end'] + 3:  # Allow small gaps
            # Merge with previous
            merged[-1]['end'] = max(merged[-1]['end'], r['end'])
            merged[-1]['articles'].extend(r['articles'])
        else:
            merged.append(r)

    return merged


def analyze_volume(edition: str, volume: str, articles: List[dict]) -> Tuple[List[SuspiciousArticle], List[dict]]:
    """Analyze a volume and return suspicious articles and reparse ranges."""
    suspicious = find_suspicious_articles(edition, volume, articles)
    ranges = group_into_ranges(articles, suspicious)
    return suspicious, ranges


def main():
    import sys

    # Can specify edition as argument
    target_edition = sys.argv[1] if len(sys.argv) > 1 else None

    docs_dir = Path('docs')
    all_ranges = []

    editions = [
        ('1771', '1st Edition'),
        ('1778', '2nd Edition'),
        ('1797', '3rd Edition'),
        ('1810', '4th Edition'),
        ('1815', '5th Edition'),
        ('1823', '6th Edition'),
        ('1842', '7th Edition'),
        ('1853', '8th Edition'),
        ('1860', '8th Edition Alt'),
    ]

    for year, name in editions:
        if target_edition and year != target_edition:
            continue

        data_dir = docs_dir / year / 'data'
        if not data_dir.exists():
            continue

        print(f"\n{'=' * 60}")
        print(f"{name} ({year})")
        print('=' * 60)

        edition_suspicious = 0
        edition_ranges = []

        for json_file in sorted(data_dir.glob('vol*.json')):
            vol = json_file.stem

            # Skip vol0 for pre-1842 (generated index)
            if vol == 'vol0' and year < '1842':
                continue
            # Skip split files
            if '_main' in vol or '_supplement' in vol:
                continue

            with open(json_file, 'r') as f:
                articles = json.load(f)

            suspicious, ranges = analyze_volume(year, vol, articles)

            if suspicious:
                print(f"\n{vol}: {len(suspicious)} suspicious articles -> {len(ranges)} reparse ranges")

                for sus in suspicious[:5]:
                    print(f"  [{sus.reason}] \"{sus.title[:50]}\" (pp.{sus.start_page}-{sus.end_page})")
                if len(suspicious) > 5:
                    print(f"  ... and {len(suspicious) - 5} more")

                edition_suspicious += len(suspicious)

                for r in ranges:
                    edition_ranges.append({
                        'edition': year,
                        'volume': vol,
                        'start_page': r['start'],
                        'end_page': r['end'],
                        'suspicious_count': len(r['articles']),
                        'reasons': list(set(a.reason for a in r['articles'])),
                        'sample_titles': [a.title[:40] for a in r['articles'][:3]]
                    })

        print(f"\n{name} Summary: {edition_suspicious} suspicious articles, {len(edition_ranges)} reparse ranges")
        all_ranges.extend(edition_ranges)

    # Save results
    output_file = Path('reparse_ranges.json')
    with open(output_file, 'w') as f:
        json.dump(all_ranges, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {len(all_ranges)} page ranges need re-parsing")
    print(f"Saved to: {output_file}")

    # Summary stats
    total_pages = sum(r['end_page'] - r['start_page'] + 1 for r in all_ranges)
    print(f"Total pages to re-parse: ~{total_pages}")


if __name__ == '__main__':
    main()
