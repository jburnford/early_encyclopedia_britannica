#!/usr/bin/env python3
"""
Parser for Encyclopedia Britannica and related encyclopedias.
Extracts individual articles from OLMoCR JSON files.

Supports multiple editions:
- Lexicon Technicum (1704-1710)
- Chambers Cyclopaedia (1728)
- Encyclopedia Britannica editions (1771-1911)

Features:
- Skips front matter (title pages, preface)
- Detects potential running header errors
- Handles multiple senses per headword
- Maps articles to source pages
- Tracks edition year for cross-edition linking
"""

import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Set
import argparse


def load_index_headwords(index_path: str) -> Set[str]:
    """
    Load headwords from an index JSONL file for use as validation whitelist.

    Args:
        index_path: Path to index JSONL file (e.g., output_v2/index_1842.jsonl)

    Returns:
        Set of uppercase headwords that are main entries with volume references.
    """
    headwords = set()
    if not os.path.exists(index_path):
        return headwords

    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Only main entries with volume references
                if entry.get('entry_type') == 'main' and entry.get('references'):
                    term = entry.get('term', '').upper().strip()
                    if len(term) > 1:  # Skip single letters
                        headwords.add(term)

    return headwords


@dataclass
class Article:
    """Represents a single encyclopedia article."""
    headword: str
    text: str
    volume: int
    start_char: int
    end_char: int
    edition_year: int = 0  # Year of this edition
    edition_name: str = ""  # e.g., "Britannica 5th"
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    sense_number: int = 1  # For multiple articles with same headword
    is_cross_reference: bool = False
    potential_header_error: bool = False
    error_reason: str = ""


@dataclass
class ParserStats:
    """Statistics from parsing."""
    total_articles: int = 0
    cross_references: int = 0
    multi_sense_headwords: int = 0
    potential_header_errors: int = 0
    skipped_front_matter: int = 0


class BritannicaParser:
    """Parser for Encyclopedia Britannica and related encyclopedias."""

    # Pattern for article headwords: ALL-CAPS word(s) followed by comma (dictionary entry)
    # Updated to handle optional markdown bolding (**)
    ARTICLE_PATTERN = re.compile(
        r'(?:^|\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?,\s+',
        re.MULTILINE
    )

    # Pattern for treatise headwords: ALL-CAPS followed by period and newlines (major essays)
    # Modified to allow markdown bolding and horizontal rules
    TREATISE_PATTERN = re.compile(
        r'(?:\n+|---\s*\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\.\s*\n+',
        re.MULTILINE
    )

    # Pattern for treatise without period: just headword on its own line
    # e.g., "MEDICINE\n\nMedicine is generally defined..."
    TREATISE_NO_PERIOD_PATTERN = re.compile(
        r'(?:\n+|---\s*\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\n\n(?=[A-Z])',
        re.MULTILINE
    )

    # Special pattern for cases where headword is immediately repeated at start of text
    # e.g. "ANATOMY\n\nANATOMY is the art..."
    TREATISE_REPEATED_PATTERN = re.compile(
        r'(?:\n+|---\s*\n+)(?:\*\*)?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)(?:\*\*)?\n\n(?=(?:\*\*)?\1\b)',
        re.IGNORECASE | re.MULTILINE
    )

    # Special pattern for LAW treatise in 1771 which lacks standard formatting
    # "LAW may be defined..."
    LAW_PATTERN = re.compile(
        r'\n\n(LAW)(?=\s+may\s+be\s+defined)',
        re.MULTILINE
    )

    # Pattern for Title Case headwords (used in 1842 edition where OCR renders some
    # headwords as "Aalen," instead of "AALEN,")
    # Matches: Aalen, a city... or Aaron, high-priest...
    # NOTE: This pattern produces many false positives - use with index whitelist!
    TITLECASE_ARTICLE_PATTERN = re.compile(
        r'(?:^|\n+)([A-Z][a-z]+(?:[\s\-][A-Za-z]+)*),\s+[a-z]',
        re.MULTILINE
    )

    # Pattern to extract volume number from metadata Source-File
    VOLUME_FROM_METADATA = re.compile(r'Volume\s+(\d+)', re.IGNORECASE)

    # Editions that use mixed case headwords (Title Case in addition to ALL-CAPS)
    # These editions need index-based validation to avoid false positives
    EDITIONS_WITH_TITLECASE = {1842, 1860}

    # Edition configurations
    EDITIONS = {
        1704: {"name": "Lexicon Technicum", "type": "lexicon"},
        1710: {"name": "Lexicon Technicum Vol 2", "type": "lexicon"},
        1728: {"name": "Chambers Cyclopaedia", "type": "cyclopaedia"},
        1745: {"name": "Coetlogon Universal History", "type": "universal"},
        1771: {"name": "Britannica 1st", "type": "britannica"},
        1778: {"name": "Britannica 2nd", "type": "britannica"},
        1797: {"name": "Britannica 3rd", "type": "britannica"},
        1810: {"name": "Britannica 4th", "type": "britannica"},
        1815: {"name": "Britannica 5th", "type": "britannica"},
        1823: {"name": "Britannica 6th", "type": "britannica"},
        1842: {"name": "Britannica 7th", "type": "britannica"},
        1860: {"name": "Britannica 8th", "type": "britannica"},
        1889: {"name": "Britannica 9th", "type": "britannica"},
        1911: {"name": "Britannica 11th", "type": "britannica"},
    }

    # Cross-reference pattern
    CROSS_REF_PATTERN = re.compile(r'^See\s+[A-Z]', re.IGNORECASE)

    # Suspicious patterns that might be running headers
    HEADER_SUSPICIONS = [
        # Very short articles (< 50 chars) that aren't cross-references
        ('short_non_crossref', lambda art: len(art.text.strip()) < 50 and not art.is_cross_reference),
        # Headword appears twice in quick succession (within 500 chars)
        ('duplicate_nearby', None),  # Handled specially
        # Article text starts with the same headword (running header slip)
        ('starts_with_headword', lambda art: art.text.strip().upper().startswith(art.headword)),
    ]

    # Known front matter patterns to skip (title pages, prefaces, etc.)
    FRONT_MATTER_PATTERNS = [
        r'^OR$',  # Part of "Encyclopaedia Britannica: OR, A DICTIONARY"
        r'^DICTIONARY',
        r'^OF$',
        r'^ARTS',
        r'^SCIENCES',
        r'^FOR\s+ARCHIBALD',
        r'^GALE',
        r'^CURTIS',
        r'^PREFACE',
        r'^ENCYCLOP',
        r'^LONDON',
        r'^EDINBURGH',
        r'^PRINTED',
        r'^VOL\b',
        r'^VOLUME\b',
    ]

    # Known major treatises in 1771 edition to protect from fragmentation
    TREATISES_1771 = {
        'AGRICULTURE', 'ALGEBRA', 'ANATOMY', 'ARCHITECTURE', 'ARITHMETIC',
        'ASTRONOMY', 'BOOK-KEEPING', 'BOTANY', 'CHEMISTRY', 'COMMERCE',
        'FARRIERY', 'GEOGRAPHY', 'GEOMETRY', 'GRAMMAR', 'HISTORY',
        'LAW', 'LOGIC', 'MECHANICS', 'MEDICINE', 'METAPHYSICS',
        'MIDWIFERY', 'MORAL PHILOSOPHY', 'MUSIC', 'NAVIGATION',
        'OPTICS', 'PERSPECTIVE', 'PHARMACY', 'PHILOSOPHY', 'PNEUMATICS',
        'POETRY', 'RELIGION', 'SURGERY', 'TANNING', 'WATCH-WORK'
    }

    # Volume letter ranges for 1771 edition
    VOL_RANGES_1771 = {
        1: ('A', 'B'),
        2: ('C', 'L'),
        3: ('M', 'Z')
    }

    # Volume letter ranges for 1815 edition (5th)
    # Based on filenames
    VOL_RANGES_1815 = {
        1: ('A', 'A'),   # A - AME
        2: ('A', 'A'),   # AME - ASS (mostly A)
        3: ('A', 'B'),   # ASS - BOO
        4: ('B', 'B'),   # BOO - BUR
        5: ('B', 'C'),   # BUR - CHI
        6: ('C', 'C'),   # CHI - Crystallization
        7: ('C', 'E'),   # CTE - Electricity
        8: ('E', 'F'),   # ELE - FOR
        9: ('F', 'G'),   # FOR - GOT
        10: ('G', 'H'),  # GOT - Hydrodynamics
        11: ('H', 'L'),  # HYD - LIE
        12: ('L', 'M'),  # LIE - Materia medica
        13: ('M', 'M'),  # MAT - MIC
        14: ('M', 'N'),  # MIC - NIC
        15: ('N', 'P'),  # NIC - PAR
        16: ('P', 'P'),  # PAR - Poetry
        17: ('P', 'R'),  # Poetry - RHI
        18: ('R', 'S'),  # RHI - Scripture
        19: ('S', 'S'),  # Missing volume, presumed SCR - SUI
        20: ('S', 'Z')   # SUI - ZYM
    }

    # Major treatises to protect (1815)
    # Includes 1771 list + items from 5th ed filenames
    TREATISES_1815 = TREATISES_1771.union({
        'CRYSTALLIZATION', 'ELECTRICITY', 'HYDRODYNAMICS', 'MATERIA MEDICA', 
        'SCRIPTURE', 'THEOLOGY', 'MINERALOGY', 'GEOLOGY' # Common 19th century additions
    })

    def __init__(self, edition_year: int = 1815, verbose: bool = False,
                 index_headwords: set = None):
        """
        Initialize the parser.

        Args:
            edition_year: Year of the edition to parse
            verbose: Whether to print verbose output
            index_headwords: Optional set of valid headwords from index (uppercase).
                            If provided, enables Title Case matching for editions
                            that use mixed case headwords (1842, 1860).
        """
        self.edition_year = edition_year
        self.edition_info = self.EDITIONS.get(edition_year, {"name": f"Unknown {edition_year}", "type": "unknown"})
        self.verbose = verbose
        self.stats = ParserStats()
        self.index_headwords = index_headwords or set()

    def parse_volume(self, json_path: str, volume_override: int = None) -> list[Article]:
        """Parse a single volume JSON file into articles."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            return []

        entry = data[0]
        text = entry['text']
        page_numbers = entry['attributes'].get('pdf_page_numbers', [])

        # Use override if provided
        if volume_override is not None:
            return self._parse_text(text, page_numbers, volume_override, json_path)

        # Extract volume from filename
        filename = Path(json_path).stem
        vol_match = re.search(r'Volume\s+(\d+)', filename)
        volume = int(vol_match.group(1)) if vol_match else 0

        return self._parse_text(text, page_numbers, volume, json_path)

    def parse_jsonl_entry(self, entry: dict) -> list[Article]:
        """Parse a single JSONL entry (one PDF) into articles."""
        text = entry.get('text', '')
        if not text:
            return []

        page_numbers = entry.get('attributes', {}).get('pdf_page_numbers', [])

        # Extract volume from metadata Source-File
        source_file = entry.get('metadata', {}).get('Source-File', '')
        vol_match = self.VOLUME_FROM_METADATA.search(source_file)
        volume = int(vol_match.group(1)) if vol_match else 0

        return self._parse_text(text, page_numbers, volume, source_file)

    def _parse_text(self, text: str, page_numbers: list, volume: int, source: str) -> list[Article]:
        """Core parsing logic for text content."""
        # Find all article matches (dictionary entries with comma)
        dict_matches = list(self.ARTICLE_PATTERN.finditer(text))

        # For editions with mixed-case OCR (1842, 1860), also find Title Case headwords
        # but ONLY if they match entries in the index (to avoid false positives)
        titlecase_matches = []
        if self.edition_year in self.EDITIONS_WITH_TITLECASE and self.index_headwords:
            for m in self.TITLECASE_ARTICLE_PATTERN.finditer(text):
                headword_upper = m.group(1).upper()
                # Only accept if it's a known headword from the index
                if headword_upper in self.index_headwords:
                    titlecase_matches.append(m)
            if self.verbose:
                print(f"    Found {len(titlecase_matches)} validated Title Case matches")

        # Combine ALL-CAPS and validated Title Case matches
        dict_matches = dict_matches + titlecase_matches

        # Find treatise matches (major essays with period)
        treatise_matches = list(self.TREATISE_PATTERN.finditer(text))

        # Find treatise matches without period (e.g., MEDICINE\n\n)
        treatise_no_period = list(self.TREATISE_NO_PERIOD_PATTERN.finditer(text))

        # Find repeated headword pattern
        repeated_matches = list(self.TREATISE_REPEATED_PATTERN.finditer(text))

        # Find LAW matches
        law_matches = list(self.LAW_PATTERN.finditer(text))

        # Combine and sort by position
        # Each match needs to track: (start_pos, end_pos, headword, is_treatise)
        all_entries = []
        for m in dict_matches:
            headword = m.group(1).upper()
            
            # Volume letter range validation
            vol_ranges = self.VOL_RANGES_1771 if self.edition_year == 1771 else self.VOL_RANGES_1815
            treatises = self.TREATISES_1771 if self.edition_year == 1771 else self.TREATISES_1815
            
            if self.edition_year in (1771, 1815) and volume in vol_ranges:
                start_let, end_let = vol_ranges[volume]
                first_char = headword[0]
                # Check if char is within range (inclusive)
                if not (start_let <= first_char <= end_let):
                    # Exception for major treatises which are allowed in any volume (sometimes)
                    # or if the range is imperfect. 
                    # But for dictionary entries, strictness is good.
                    if headword not in treatises:
                        # Allow slight overlap (e.g. AME ending, next volume starts AME)
                        # But 'U' in 'A-B' volume is definitely wrong.
                        continue

            all_entries.append((m.start(), m.end(), m.group(1), False, m))

        # Common false positive words for treatise patterns
        # Removed BOOK, CHAPTER, SECTION to avoid skipping real articles like BOOK-KEEPING
        skip_words = {'ILLUSTRATED', 'VOLUMES', 'CASE', 'TABLE', 'PART', 'PLATE',
                     'FIGURE', 'DIRECT', 'INDIRECT', 'GENERAL', 'USES', 'PRINCIPLES',
                     'APPENDIX', 'INDEX', 'CONTENTS'}

        treatise_candidates = treatise_matches + treatise_no_period + repeated_matches + law_matches

        for m in treatise_candidates:
            headword = m.group(1).upper()

            # Skip very short or obviously non-article headwords
            if len(headword) < 3:
                continue

            # Check skip words
            headword_parts = set(re.split(r'[\s\-]+', headword))
            if any(w in headword_parts for w in skip_words):
                continue

            # Special handling for editions
            treatises = self.TREATISES_1771 if self.edition_year == 1771 else self.TREATISES_1815

            if self.edition_year in (1771, 1815):
                # ONLY allow treatise matches if they are in our major whitelist
                # This prevents section headers within Anatomy/Law/etc from breaking articles
                if headword not in treatises:
                    continue

            # Specific check for structural headers (Book/Chapter/Section + Number)
            if re.match(r'^(?:BOOK|CHAPTER|SECTION|SECT|PART|PLATE|FIG|FIGURE)\s+[IVX0-9]+', headword, re.IGNORECASE):
                continue
            
            # Check for "Part I." or similar immediately before or after the match
            # Look back 50 chars and forward 50 chars
            structural_headers = r'\b(?:Part|PART|PLATE|Plate|Fig|FIG|SECTION|SECT|BOOK|CHAPTER|Page|PAGE)\.?\s+[IVX0-9]+'
            lookback = text[max(0, m.start()-50):m.start()]
            lookahead = text[m.end():min(len(text), m.end()+50)]
            if re.search(structural_headers, lookback) or re.search(structural_headers, lookahead):
                continue

            # Filter out section titles that start with a number (e.g. "BOROUGHS.\n\n11 A borough...")
            # The match includes the \n\n at the end, so text[m.end()] is the start of body
            body_preview = text[m.end():m.end()+10].strip()
            if body_preview:
                if body_preview[0].isdigit():
                    continue
                
                # Filter out running headers interrupting sentences (body continues with lowercase)
                # But allow common definition starters like "is", "may", "signifies"
                if body_preview[0].islower():
                    first_word = body_preview.split()[0].lower()
                    allowed_starters = {'is', 'are', 'may', 'can', 'signifies', 'denotes', 'contains', 'includes', 'consists'}
                    # Remove punctuation
                    first_word = first_word.strip('.,;:')
                    if first_word not in allowed_starters:
                        continue

            all_entries.append((m.start(), m.end(), headword, True, m))

        # Sort by start position
        all_entries.sort(key=lambda x: x[0])

        # Deduplicate matches that are very close (within 20 chars) or have same start
        unique_entries = []
        for e in all_entries:
            if not unique_entries or e[0] > unique_entries[-1][0] + 20:
                unique_entries.append(e)
            else:
                # If they are very close, prefer treatise over dict entry
                if e[3] and not unique_entries[-1][3]:
                    unique_entries[-1] = e
        all_entries = unique_entries

        # Filter out empty/short treatises (e.g. "See ASTRONOMY." followed immediately by next headword)
        # Treatises should have substantial content. If < 50 chars, it's likely a false positive
        # or a cross-reference that got detected as a treatise title.
        filtered_entries = []
        for i in range(len(all_entries)):
            entry = all_entries[i]
            # (start, end, headword, is_treatise, match)
            is_treatise = entry[3]

            if is_treatise:
                # Calculate content length
                content_start = entry[1]
                if i + 1 < len(all_entries):
                    content_end = all_entries[i+1][0]
                else:
                    content_end = len(text)
                
                # Check if content is effectively empty or too short
                content = text[content_start:content_end].strip()
                if len(content) < 50: 
                    continue
            
            filtered_entries.append(entry)
        all_entries = filtered_entries

        # Convert back to match-like format for filtering
        matches = [e[4] for e in all_entries]

        # Filter out front matter
        valid_matches = self._filter_front_matter(matches, text)

        # Build articles
        articles = []
        headword_counts = {}  # Track sense numbers

        for i, match in enumerate(valid_matches):
            original_headword = match.group(1)
            headword = original_headword.upper()
            start_char = match.end()  # Start after the "HEADWORD, "

            # End is the start of the next article, or end of text
            if i + 1 < len(valid_matches):
                end_char = valid_matches[i + 1].start()
            else:
                end_char = len(text)

            article_text = text[start_char:end_char].strip()

            # Track sense numbers for duplicate headwords
            headword_counts[headword] = headword_counts.get(headword, 0) + 1
            sense = headword_counts[headword]
            
            # Heuristic: If we hit the same major treatise headword again in the same volume,
            # it's likely a running header error. We should skip it and merge text.
            treatises = self.TREATISES_1771 if self.edition_year == 1771 else self.TREATISES_1815
            
            if self.edition_year in (1771, 1815) and headword in treatises and sense > 1:
                # Add text to previous article if available
                if articles and articles[-1].headword == headword:
                    articles[-1].text += "\n\n" + article_text
                    articles[-1].end_char = end_char
                    # Update end page
                    _, new_end_page = self._get_page_range(match.start(), end_char, page_numbers)
                    if new_end_page:
                        articles[-1].end_page = new_end_page
                continue

            if sense > 1:
                self.stats.multi_sense_headwords += 1

            # Check for cross-reference
            is_crossref = bool(self.CROSS_REF_PATTERN.match(article_text))
            if is_crossref:
                self.stats.cross_references += 1

            # Get page range
            start_page, end_page = self._get_page_range(
                match.start(), end_char, page_numbers
            )

            article = Article(
                headword=headword,
                text=article_text,
                volume=volume,
                start_char=match.start(),
                end_char=end_char,
                edition_year=self.edition_year,
                edition_name=self.edition_info["name"],
                start_page=start_page,
                end_page=end_page,
                sense_number=sense,
                is_cross_reference=is_crossref,
            )

            # Check for potential header errors
            self._check_header_errors(article, articles)

            articles.append(article)
            self.stats.total_articles += 1

        return articles

    def _filter_front_matter(self, matches: list, text: str) -> list:
        """Filter out front matter matches.

        Strategy: Find the first legitimate short headword (1-4 chars, all letters)
        that looks like a dictionary entry (A, AA, AB, AACH, etc.) and treat
        everything before it as front matter.
        """
        valid = []
        articles_start_idx = None

        # First pass: find where articles actually start
        # Look for short alphabetical headwords that are real dictionary entries
        for i, match in enumerate(matches):
            headword = match.group(1)

            # Skip multi-line headwords (contain newlines) - these are front matter artifacts
            if '\n' in headword:
                continue

            # Skip known front matter patterns
            is_front_matter = False
            for pattern in self.FRONT_MATTER_PATTERNS:
                if re.match(pattern, headword, re.IGNORECASE):
                    is_front_matter = True
                    break
            if is_front_matter:
                continue

            # Look for the first legitimate article headword
            # Criteria: 1-6 chars, purely alphabetic (no spaces), and article text looks real
            if (1 <= len(headword) <= 6 and
                headword.replace("'", "").replace("-", "").isalpha()):
                # Check that the article text is substantial (> 20 chars after headword)
                article_preview = text[match.end():match.end()+100]
                if len(article_preview.strip()) > 20:
                    articles_start_idx = i
                    break

        if articles_start_idx is None:
            # Fallback: just skip obvious front matter patterns
            articles_start_idx = 0

        # Second pass: collect valid articles
        for i, match in enumerate(matches):
            if i < articles_start_idx:
                self.stats.skipped_front_matter += 1
                continue

            headword = match.group(1)

            # Skip multi-line headwords even after start
            if '\n' in headword:
                self.stats.skipped_front_matter += 1
                continue

            valid.append(match)

        return valid

    def _get_page_range(self, start_char: int, end_char: int,
                        page_numbers: list) -> tuple[Optional[int], Optional[int]]:
        """Get the page range for a character range."""
        start_page = None
        end_page = None

        for page_start, page_end, page_num in page_numbers:
            # Skip empty page markers
            if page_start == 0 and page_end == 0:
                continue

            if page_start <= start_char <= page_end:
                start_page = page_num
            if page_start <= end_char <= page_end:
                end_page = page_num
                break
            # If we've passed the end, use last valid page
            if page_start > end_char and end_page is None:
                end_page = page_num - 1 if page_num > 1 else 1
                break

        return start_page, end_page

    def _check_header_errors(self, article: Article, prev_articles: list):
        """Check for patterns that might indicate running header errors.

        NOTE: Multi-sense entries (same headword, different subjects) are NORMAL
        in encyclopedias and should NOT be flagged as errors. We only flag
        true anomalies that suggest OCR running header slip-through.
        """
        reasons = []

        # Check: Very short non-cross-reference (< 20 chars is suspicious)
        text_len = len(article.text.strip())
        if text_len < 20 and not article.is_cross_reference:
            reasons.append(f"very_short_entry_{text_len}chars")

        # Check: Single word that's suspiciously common in headers (not articles)
        suspicious_words = {'PART', 'VOL', 'VOLUME', 'PLATE', 'FIGURE', 'PAGE',
                          'CHAPTER', 'SECTION', 'INDEX', 'CONTENTS'}
        if article.headword in suspicious_words:
            reasons.append("common_header_word")

        # Check: Headword is just a number or pure Roman numeral (page/chapter markers)
        # Be careful: CID, CIVIL, MIX, VIM, etc. are real words, not numerals
        # Only flag very short pure-numeral patterns (I, II, III, IV, V, VI, VII, VIII, IX, X, etc.)
        # or Arabic numerals
        if re.match(r'^\d+$', article.headword):
            reasons.append("arabic_numeral_headword")
        elif re.match(r'^[IVX]{1,4}$', article.headword):  # Only flag short Roman numerals I-XIII
            # Exclude real words that look like Roman numerals
            real_words = {'VI', 'MIX', 'VIM', 'MIC', 'MID', 'VIA', 'VIC', 'VII'}
            if article.headword not in real_words:
                reasons.append("short_roman_numeral")

        # Check: Article text is ONLY the headword repeated (clear header slip)
        clean_text = article.text.strip().upper()
        if clean_text == article.headword or clean_text.startswith(article.headword + ' ' + article.headword):
            reasons.append("repeated_headword_only")

        if reasons:
            article.potential_header_error = True
            article.error_reason = "; ".join(reasons)
            self.stats.potential_header_errors += 1

    def parse_all_volumes(self, json_dir: str) -> dict[int, list[Article]]:
        """Parse all volume JSON files in a directory.

        Handles duplicate volume files by picking the largest one per volume number.
        """
        json_path = Path(json_dir)

        # Group files by volume number, pick largest per volume
        volume_files = {}
        for json_file in json_path.glob("Volume *.json"):
            vol_match = re.search(r'Volume\s+(\d+)', json_file.stem)
            if vol_match:
                vol_num = int(vol_match.group(1))
                file_size = json_file.stat().st_size
                if vol_num not in volume_files or file_size > volume_files[vol_num][1]:
                    volume_files[vol_num] = (json_file, file_size)

        if self.verbose:
            print(f"Found {len(volume_files)} unique volumes (deduped from {len(list(json_path.glob('Volume *.json')))} files)")

        results = {}
        for vol_num in sorted(volume_files.keys()):
            json_file, _ = volume_files[vol_num]
            if self.verbose:
                print(f"Parsing {json_file.name}...")

            articles = self.parse_volume(str(json_file))
            results[vol_num] = articles

        return results

    def parse_all_jsonl(self, jsonl_dir: str) -> dict[int, list[Article]]:
        """Parse all JSONL OCR result files in a directory.

        Each JSONL file may contain multiple entries (one per PDF).
        Deduplicates by volume, picking the largest entry per volume.
        """
        jsonl_path = Path(jsonl_dir)

        # Collect all entries from all JSONL files
        # Key: volume number, Value: (entry dict, file size proxy)
        volume_entries = {}

        for jsonl_file in jsonl_path.glob("*.jsonl"):
            if self.verbose:
                print(f"Reading {jsonl_file.name}...")

            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract volume from metadata
                    source_file = entry.get('metadata', {}).get('Source-File', '')
                    vol_match = self.VOLUME_FROM_METADATA.search(source_file)
                    if not vol_match:
                        continue

                    vol_num = int(vol_match.group(1))
                    text_len = len(entry.get('text', ''))

                    # Keep largest entry per volume
                    if vol_num not in volume_entries or text_len > volume_entries[vol_num][1]:
                        volume_entries[vol_num] = (entry, text_len, source_file)

        if self.verbose:
            print(f"Found {len(volume_entries)} unique volumes")

        # Parse each volume
        results = {}
        for vol_num in sorted(volume_entries.keys()):
            entry, _, source = volume_entries[vol_num]
            if self.verbose:
                print(f"Parsing Volume {vol_num} from {Path(source).name}...")

            articles = self.parse_jsonl_entry(entry)
            results[vol_num] = articles

        return results

    def export_articles(self, articles: list[Article], output_path: str,
                        format: str = 'jsonl'):
        """Export articles to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            if format == 'jsonl':
                for art in articles:
                    record = {
                        'headword': art.headword,
                        'sense': art.sense_number,
                        'edition_year': art.edition_year,
                        'edition_name': art.edition_name,
                        'volume': art.volume,
                        'start_page': art.start_page,
                        'end_page': art.end_page,
                        'is_cross_reference': art.is_cross_reference,
                        'potential_error': art.potential_header_error,
                        'error_reason': art.error_reason if art.potential_header_error else None,
                        'text': art.text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            elif format == 'summary':
                for art in articles:
                    flag = '!' if art.potential_header_error else ' '
                    xref = 'X' if art.is_cross_reference else ' '
                    sense = f".{art.sense_number}" if art.sense_number > 1 else ""
                    f.write(f"{flag}{xref} V{art.volume:02d} p{art.start_page or 0:04d} {art.headword}{sense}\n")

    def print_stats(self):
        """Print parsing statistics."""
        print(f"\n=== Parsing Statistics ===")
        print(f"Total articles: {self.stats.total_articles:,}")
        print(f"Cross-references: {self.stats.cross_references:,}")
        print(f"Multi-sense headwords: {self.stats.multi_sense_headwords:,}")
        print(f"Potential header errors: {self.stats.potential_header_errors:,}")
        print(f"Skipped front matter: {self.stats.skipped_front_matter:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse Encyclopedia Britannica OCR output into articles'
    )
    parser.add_argument('json_dir', nargs='?', help='Directory containing volume JSON files')
    parser.add_argument('-y', '--year', type=int, default=1815,
                        help='Edition year (default: 1815)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-f', '--format', choices=['jsonl', 'summary'],
                        default='jsonl', help='Output format')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--errors-only', action='store_true',
                        help='Only output potential header errors')
    parser.add_argument('--list-editions', action='store_true',
                        help='List known editions and exit')

    args = parser.parse_args()

    if args.list_editions:
        print("Known editions:")
        for year, info in sorted(BritannicaParser.EDITIONS.items()):
            print(f"  {year}: {info['name']} ({info['type']})")
        return

    if not args.json_dir:
        parser.error("json_dir is required when not using --list-editions")

    britannica = BritannicaParser(edition_year=args.year, verbose=args.verbose)
    print(f"Parsing {britannica.edition_info['name']} ({args.year})...")

    # Auto-detect format: check for .jsonl files (OCR results) vs .json files (converted)
    input_path = Path(args.json_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))
    json_files = list(input_path.glob("Volume *.json"))

    if jsonl_files and not json_files:
        print(f"Detected JSONL format ({len(jsonl_files)} files)")
        all_volumes = britannica.parse_all_jsonl(args.json_dir)
    elif json_files:
        print(f"Detected JSON format ({len(json_files)} files)")
        all_volumes = britannica.parse_all_volumes(args.json_dir)
    else:
        print(f"Error: No .jsonl or Volume *.json files found in {args.json_dir}")
        return

    # Flatten to single list
    all_articles = []
    for vol_num in sorted(all_volumes.keys()):
        all_articles.extend(all_volumes[vol_num])

    # Filter if errors-only
    if args.errors_only:
        all_articles = [a for a in all_articles if a.potential_header_error]
        print(f"\nFiltered to {len(all_articles)} potential errors")

    britannica.print_stats()

    # Output
    if args.output:
        britannica.export_articles(all_articles, args.output, args.format)
        print(f"\nExported to {args.output}")
    else:
        # Print sample
        print(f"\n=== Sample Articles ===")
        for art in all_articles[:10]:
            flag = "!" if art.potential_header_error else " "
            print(f"{flag} {art.headword} (V{art.volume}, p{art.start_page}): {art.text[:100]}...")

        if args.errors_only or britannica.stats.potential_header_errors > 0:
            print(f"\n=== Potential Header Errors ===")
            errors = [a for a in all_articles if a.potential_header_error]
            for art in errors[:20]:
                print(f"  {art.headword} (V{art.volume}): {art.error_reason}")
                print(f"    Text: {art.text[:80]}...")


if __name__ == '__main__':
    main()
