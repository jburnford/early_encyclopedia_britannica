"""
Volume letter ranges for each Encyclopedia Britannica edition.

Used by ArticleValidator to check if articles are in the correct volume.

Note: Some editions have gaps in volume numbers or contain appendix/supplement volumes
that don't follow alphabetical ordering. Volume 0 in many editions contains index
or supplementary content.
"""

# Volume ranges: {edition_year: {volume_num: (start_letter, end_letter)}}
# Letters are inclusive. Use first letter of headword for comparison.

VOLUME_RANGES = {
    1771: {
        1: ('A', 'B'),   # A-B
        2: ('C', 'L'),   # C-L
        3: ('M', 'Z'),   # M-Z
    },

    1778: {
        # Note: Volume 3 not found in parsed data
        1: ('A', 'A'),   # A-AST
        2: ('A', 'B'),   # Astronomy-BZO (overlaps with A)
        4: ('D', 'F'),   # D-F (no C volume?)
        5: ('G', 'J'),   # G-J
        6: ('K', 'M'),   # K-Medicine
        7: ('M', 'O'),   # Medicines-Optics
        8: ('O', 'P'),   # Optics-Poetry
        9: ('P', 'S'),   # POI-SCU
        10: ('S', 'Z'),  # SCU-Appendix (contains appendix material)
    },

    1797: {
        1: ('A', 'A'),   # A-ANG
        2: ('A', 'B'),   # ANG-BAR
        3: ('B', 'B'),   # BAR-BZO
        4: ('C', 'C'),   # CAA-CIC
        5: ('C', 'D'),   # CIC-DIA
        6: ('D', 'E'),   # DIA-ETH
        7: ('E', 'G'),   # ETM-GOA
        8: ('G', 'H'),   # GOB-HYD
        9: ('H', 'L'),   # Hydrostatics-LES
        10: ('L', 'M'),  # LES-MEC
        11: ('M', 'M'),  # Medals-Midwifery
        12: ('M', 'N'),  # MEI-NEG
        13: ('N', 'P'),  # NEH-PAS
        14: ('P', 'P'),  # PAS-PLA
        15: ('P', 'R'),  # PLA-RAM
        16: ('R', 'S'),  # RAN-SCO
        17: ('S', 'S'),  # SCO-STR
        18: ('S', 'Z'),  # STR-ZYM
    },

    1810: {
        # No volume ranges extracted from index - use broad ranges
        # 4th edition had 20 volumes
        1: ('A', 'A'),
        2: ('A', 'B'),
        3: ('B', 'B'),
        4: ('B', 'C'),
        5: ('C', 'C'),
        6: ('C', 'D'),
        7: ('D', 'E'),
        8: ('E', 'F'),
        9: ('F', 'G'),
        10: ('G', 'H'),
        11: ('H', 'I'),
        12: ('I', 'L'),
        13: ('L', 'M'),
        14: ('M', 'M'),
        15: ('M', 'N'),
        16: ('N', 'P'),
        17: ('P', 'P'),
        18: ('P', 'R'),
        19: ('R', 'S'),
        20: ('S', 'Z'),
    },

    1815: {
        1: ('A', 'A'),   # A-AME
        2: ('A', 'A'),   # America-ASS
        3: ('A', 'B'),   # ASS-BOO
        4: ('B', 'B'),   # BOO-BUR
        5: ('B', 'C'),   # BUR-CHI
        6: ('C', 'C'),   # CHI-Crystallization
        7: ('C', 'E'),   # CTE-Electricity
        8: ('E', 'F'),   # ELE-FOR
        9: ('F', 'G'),   # FOR-GOT
        10: ('G', 'H'),  # GOT-Hydrodynamics
        11: ('H', 'L'),  # HYD-LIE
        12: ('L', 'M'),  # LIE-Materia
        13: ('M', 'M'),  # MAT-MIC
        14: ('M', 'N'),  # MIC-NIC
        15: ('N', 'P'),  # NIC-PAR
        16: ('P', 'P'),  # PAR-Poetry
        17: ('P', 'R'),  # Poetry-RHI
        18: ('R', 'S'),  # RHI-Scripture
        # Note: Volume 19 missing from parsed data
        20: ('S', 'Z'),  # SUI-ZYM
    },

    1823: {
        # Note: Volumes 1, 5, 6 not in parsed data
        2: ('A', 'A'),   # America-ASS
        3: ('A', 'B'),   # ASS-BOO
        4: ('B', 'B'),   # BOO-BUR
        7: ('C', 'E'),   # CTE-Electricity
        8: ('E', 'F'),   # ELE-FOR
        9: ('F', 'G'),   # FOR-GOT
        10: ('G', 'H'),  # GOT-Hydrodynamics
        11: ('H', 'L'),  # HYD-LIE
        12: ('L', 'M'),  # LIE-Materia
        13: ('M', 'M'),  # MAT-MIC
        14: ('M', 'N'),  # MIC-NIC
        15: ('N', 'P'),  # NIC-PAR
        16: ('P', 'P'),  # PAR-Poetry
        17: ('P', 'R'),  # Poetry-RHI
        18: ('R', 'S'),  # RHI-Scripture
        19: ('S', 'S'),  # Scripture-SUG
        20: ('S', 'Z'),  # SUI-ZYM
    },

    1842: {
        # Note: Volume 1 not in parsed data
        2: ('A', 'A'),   # A-Anatomy
        3: ('A', 'A'),   # Anatomy-Astronomy
        4: ('A', 'B'),   # Astronomy-BOR
        5: ('B', 'C'),   # BOR-CAL
        6: ('C', 'C'),   # CAL-Clock
        7: ('C', 'D'),   # CLO-Dialling
        8: ('D', 'E'),   # DIA-England
        9: ('E', 'F'),   # England-FRA
        10: ('F', 'G'),  # France-GRO
        11: ('G', 'H'),  # Grotius-HYD
        12: ('H', 'K'),  # Hydrodynamics-KYR
        13: ('L', 'M'),  # LAB-Magnetism
        14: ('M', 'M'),  # Magnetism-Mexico
        15: ('M', 'N'),  # MEY-Navigation
        16: ('N', 'P'),  # Navigation-PAN
        17: ('P', 'P'),  # PAN-Plastic
        18: ('P', 'Q'),  # PLA-QUI
        19: ('R', 'S'),  # RAB-SCU
        20: ('S', 'S'),  # Sculpture-SUR
        21: ('S', 'Z'),  # Surveying-ZYM
    },

    1860: {
        # Note: Volumes 1, 6 not in parsed data
        2: ('A', 'A'),   # A-Anatomy
        3: ('A', 'A'),   # Anatomy-Astronomy
        4: ('A', 'B'),   # Astronomy-BOM
        5: ('B', 'B'),   # Bombay-BUR
        7: ('C', 'D'),   # CLI-DIA
        8: ('D', 'E'),   # Diamond-Entail
        9: ('E', 'F'),   # Entomology-FRA
        10: ('F', 'G'),  # France-GRA
        11: ('G', 'H'),  # GRA-HUM
        12: ('H', 'J'),  # Hume-JOM
        13: ('J', 'M'),  # Jonah-MAG
        14: ('M', 'M'),  # Magnetism-MIH
        15: ('M', 'N'),  # Milan-NAV
        16: ('N', 'O'),  # Navigation-Ornithology
        17: ('O', 'P'),  # ORO-Plato
        18: ('P', 'R'),  # PLA-REI
        19: ('R', 'S'),  # Reid-Scythia
        20: ('S', 'S'),  # Seamanship-SZO
        21: ('T', 'Z'),  # T-ZWO
    },
}


def get_volume_range(edition_year: int, volume_num: int) -> tuple[str, str] | None:
    """
    Get the letter range for a specific volume.

    Args:
        edition_year: Year of the edition (e.g., 1771)
        volume_num: Volume number

    Returns:
        Tuple of (start_letter, end_letter) or None if not found
    """
    if edition_year not in VOLUME_RANGES:
        return None
    if volume_num not in VOLUME_RANGES[edition_year]:
        return None
    return VOLUME_RANGES[edition_year][volume_num]


def check_headword_in_range(
    headword: str,
    edition_year: int,
    volume_num: int
) -> tuple[bool, str | None]:
    """
    Check if a headword falls within the expected range for its volume.

    Args:
        headword: The article headword
        edition_year: Year of the edition
        volume_num: Volume number

    Returns:
        Tuple of (is_in_range, reason_if_not)
    """
    vol_range = get_volume_range(edition_year, volume_num)
    if vol_range is None:
        # Unknown range, can't validate
        return True, None

    if not headword:
        return True, None

    first_letter = headword[0].upper()
    if not first_letter.isalpha():
        # Non-alphabetic headwords can't be range-checked
        return True, None

    start_letter, end_letter = vol_range

    if first_letter < start_letter:
        return False, f"'{headword[:20]}' starts with '{first_letter}' but volume {volume_num} starts at '{start_letter}'"

    if first_letter > end_letter:
        return False, f"'{headword[:20]}' starts with '{first_letter}' but volume {volume_num} ends at '{end_letter}'"

    return True, None


def get_expected_volume(headword: str, edition_year: int) -> int | None:
    """
    Given a headword, find which volume it should be in.

    Args:
        headword: The article headword
        edition_year: Year of the edition

    Returns:
        Expected volume number or None if can't determine
    """
    if edition_year not in VOLUME_RANGES:
        return None

    if not headword:
        return None

    first_letter = headword[0].upper()
    if not first_letter.isalpha():
        return None

    for volume_num, (start, end) in VOLUME_RANGES[edition_year].items():
        if start <= first_letter <= end:
            return volume_num

    return None
