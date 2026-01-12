#!/usr/bin/env python3
"""
Volume-level alphabetic outlier detection for Encyclopedia Britannica.

Uses the explicit volume ranges (e.g., "NIC-PAR") from volumes_*.jsonl
to detect articles that appear in the wrong volume based on their headword.

This catches clustered outliers that neighbor-based detection misses.
For example: A run of 10 "E" articles in a volume covering "M-N" would all
be detected as outliers, even though they're in alphabetical order with each other.

Auto-classifies outliers as:
- MERGE: Sentence fragments, section headers, Roman numerals, genus entries
- DELETE: Publisher notices, front matter, index entries
- RENAME: OCR errors where headword is garbled
- KEEP: Valid articles that may be geographic subdivisions
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VolumeRange:
    """Parsed volume range with start/end headwords."""
    volume_num: int
    raw_range: str
    start_headword: str
    end_headword: str
    is_valid: bool = True
    derived: bool = False  # True if range was derived from article distribution

    # Special end boundaries that aren't alphabetic limits
    NON_ALPHABETIC_ENDS = {
        'APPENDIX', 'SUPPLEMENT', 'INDEX', 'ATLAS', 'PLATES',
        'GENERAL INDEX', 'ADDENDA', 'ERRATA', 'CONTENTS'
    }

    @classmethod
    def from_metadata(cls, vol_num: int, raw_range: str) -> 'VolumeRange':
        """Parse a range like 'NIC-PAR' or 'America-ASS'."""
        if not raw_range or raw_range in ('Part', 'Preliminary', 'Dissertations', ''):
            return cls(vol_num, raw_range, '', '', is_valid=False)

        # Handle single-letter/word ranges like "C" or "A-B"
        if '-' in raw_range:
            parts = raw_range.split('-', 1)
            start = parts[0].strip().upper()
            end = parts[1].strip().upper()

            # Check if end is a non-alphabetic boundary (e.g., "SCU-Appendix")
            # In this case, the volume goes from start to Z
            if end in cls.NON_ALPHABETIC_ENDS:
                end = ''  # Will be derived from articles
        else:
            # Single entry like "C" or "Burning" - just the start, need to derive end
            start = raw_range.strip().upper()
            end = ''  # Will be derived from articles

        return cls(vol_num, raw_range, start, end, is_valid=bool(start))

    @classmethod
    def from_articles(cls, vol_num: int, articles: list[dict]) -> 'VolumeRange':
        """Derive range from actual article distribution."""
        sorted_arts = sorted(articles, key=lambda a: a.get('headword', '').upper())

        # Find first and last valid alphabetic headwords
        first_hw = ''
        last_hw = ''

        for a in sorted_arts:
            hw = a.get('headword', '')
            if hw and hw[0].isalpha() and not is_structural_entry(hw):
                first_hw = hw.upper()
                break

        for a in reversed(sorted_arts):
            hw = a.get('headword', '')
            if hw and hw[0].isalpha() and not is_structural_entry(hw):
                last_hw = hw.upper()
                break

        if first_hw and last_hw:
            return cls(vol_num, f"{first_hw[:3]}-{last_hw[:3]} (derived)",
                      first_hw, last_hw, is_valid=True, derived=True)
        return cls(vol_num, '', '', '', is_valid=False)

    def update_end_from_articles(self, articles: list[dict]) -> None:
        """Update end boundary from article distribution if not set."""
        if self.end_headword:
            return

        sorted_arts = sorted(articles, key=lambda a: a.get('headword', '').upper())
        for a in reversed(sorted_arts):
            hw = a.get('headword', '')
            if hw and hw[0].isalpha() and not is_structural_entry(hw):
                self.end_headword = hw.upper()
                self.derived = True
                break

    def contains_headword(self, headword: str) -> tuple[bool, str]:
        """Check if headword falls within this volume's range.

        Returns: (is_in_range, reason_if_not)
        """
        if not self.is_valid:
            return True, ""  # Can't validate, assume OK

        hw_upper = headword.upper()

        # Skip non-alphabetic headwords
        if not hw_upper or not hw_upper[0].isalpha():
            return True, ""

        # Compare to start boundary
        if hw_upper < self.start_headword:
            return False, f"'{headword}' comes before '{self.start_headword}'"

        # Compare to end boundary
        if self.end_headword and hw_upper > self.end_headword:
            # Allow flexibility for abbreviated end boundaries
            # e.g., "PAR" should allow "PARISH"
            end_prefix = self.end_headword[:3]
            if not hw_upper.startswith(end_prefix[:len(hw_upper)]):
                return False, f"'{headword}' comes after '{self.end_headword}'"

        return True, ""


def is_structural_entry(headword: str) -> bool:
    """Check if headword is a structural entry (END OF, EXPLANATION, etc.)."""
    structural = [
        'END OF', 'FINIS', 'ERRATA', 'CORRIGENDA', 'ADDENDA',
        'CONTENTS', 'INDEX', 'PREFACE', 'INTRODUCTION',
        'EXPLANATION', 'DIRECTIONS', 'APPENDIX'
    ]
    hw_upper = headword.upper()
    return any(hw_upper.startswith(s) for s in structural)


@dataclass
class OutlierClassification:
    """Auto-classification of an outlier with confidence."""
    decision: str  # MERGE, DELETE, RENAME, KEEP
    confidence: str  # high, medium, low
    reason: str
    merge_target: Optional[str] = None


@dataclass
class Outlier:
    """A detected volume outlier article."""
    article_id: str
    headword: str
    edition_year: int
    volume_num: int
    start_page: int
    end_page: int
    word_count: int
    volume_range: str
    effective_start: str  # Actual start boundary used
    effective_end: str    # Actual end boundary used
    reason: str
    text_preview: str
    text_end: str
    classification: Optional[OutlierClassification] = None
    merge_candidates: list = field(default_factory=list)
    prev_articles: list = field(default_factory=list)
    next_articles: list = field(default_factory=list)


class OutlierClassifier:
    """Auto-classify detected outliers based on patterns."""

    # Patterns indicating fragments/section headers (high confidence MERGE)
    FRAGMENT_PATTERNS = [
        r'^(SECT\.?|SECTION)\s*[IVXLC\d]+',  # SECT. I, SECTION 2
        r'^(CHAP\.?|CHAPTER)\s*[IVXLC\d]+',  # CHAP. I, CHAPTER 2
        r'^(PART)\s*[IVXLC\d]+',  # PART I, PART 2
        r'^(PLATE|PLATES?)\s*[IVXLC\d]*',  # PLATE I, PLATES
        r'^(BOOK)\s*[IVXLC\d]+',  # BOOK I
        r'^[IVXLC]+[\.\,\s]',  # Roman numerals: I., II., III, etc.
        r'^[IVXLC]+$',  # Just Roman numerals: VII, VIII, etc.
        r'^[A-Z]\.\s',  # Single letter sections: A., B.
        r'^OF\s+THE\s+',  # "OF THE ..." sentence fragments
        r'^AND\s+',  # Starts with "AND"
        r'^THE\s+',  # Starts with "THE" (often sentence fragments)
        r'^OR\s+',  # Starts with "OR"
        r'^IN\s+THE\s+',  # "IN THE ..."
        r'^\d+[\.\)]\s',  # Numbered sections: 1., 2), etc.
        r'^(GENUS|ORDER|CLASS|SPECIES|FAMILY)\s+',  # Taxonomic sections
        r'^(PROBLEM|PROPOSITION|THEOREM|COROLLARY|LEMMA|AXIOM|RULE|CASE)\s*[IVXLC\d]*',  # Math sections
        r'^(PROCESS|METHOD|EXPERIMENT)\s*[IVXLC\d]+',  # Scientific sections
        r'^(EXPLANATION|DESCRIPTION)\s+OF\s+(THE\s+)?(PLATE|PLATES)',  # Plate captions
        r'^(USES|PROPERTIES|EFFECTS)\s+OF\s+THE',  # Subsection headers
        r'^(CLASSIS|ORDO|DIVISIO)\s*[IVXLC\d]*',  # Latin taxonomic
        r'^(GENERAL|PARTICULAR|SPECIFIC)\s+(OBSERVATIONS|REMARKS|DESCRIPTION)',  # Headers
        r'^WORKS\s+BY\s+',  # Bibliography entries
        r'^TABLE\s+OF\s+',  # Table captions
        r'^\w+\s+(LIKEWISE|ALSO|TOO)\s*$',  # "X LIKEWISE" sentence fragments
        r'^(THIS|THAT|THESE|THOSE|SUCH|WHICH|WHEN|WHERE|WHILE|ALTHOUGH|SINCE|BESIDES|HITHERTO|NOTWITHSTANDING)\s+',  # Sentence starters
        r'^(HAVING|BEING|AFTER|BEFORE|UPON|DURING)\s+',  # Participle phrases
    ]

    # Anatomical/botanical Latin terms (medium confidence MERGE - subsections)
    LATIN_SUBSECTION_PATTERNS = [
        r'^MUSCUL[IO]',  # MUSCULI, MUSCULUS
        r'^(RECTUS|OBLIQUUS|TRANSVERSAL|DELTOID|PECTOR|LATIS|BICEP|TRICEP|TENSOR|EXTENSOR|FLEXOR)',  # Muscles
        r'^(OBTURATOR|PYRIFORM|POPLITE|DIAPHRAGM|INTEROSS|LUMBRIC|THENAR|HYPOTHENAR)',  # More muscles
        r'^(INTESTIN|PANCREA|SPLEEN|VESIC|PULMON|CEREBR|MEDULL)',  # Organs
        r'^(PIA|DURA)\s+MATER',  # Brain membranes
        r'^(DIANDRIA|TRIANDRIA|TETRANDRIA|PENTANDRIA|HEXANDRIA|OCTANDRIA|DECANDRIA|DODECANDRIA|ICOSANDRIA|POLYANDRIA)',  # Linnaean classes
        r'^(MONOGYN|DIGYN|TRIGYN|TETRAGYN|PENTAGYN|HEXAGYN|POLYGYN)',  # Linnaean orders
        r'^(DIDYNAMIA|TETRADYNAMIA|MONADELPHIA|DIADELPHIA|POLYADELPHIA|SYNGENESIA|GYNANDRIA|MONOECIA|DIOECIA)',  # More Linnaean
        r'^(ARANEA|ARGYRONETA|PHALANGIUM|SCORPIO)',  # Arachnids
    ]

    # Patterns for publisher notices / front matter (high confidence DELETE)
    DELETE_PATTERNS = [
        r'^(MACKENZIE|GRANT|BALLANTYNE|LONGMAN|MURRAY)',  # Publishers
        r'^(PRINTER|PRINTERS|PRINTED\s+BY)',
        r'^(LONDON|EDINBURGH|GLASGOW)[\.\,\:]',  # City headers
        r'^INDEX\s+(TO|OF)',
        r'^(ERRATA|CORRIGENDA|ADDENDA)',
        r'^(CONTENTS|TABLE\s+OF\s+CONTENTS)',
        r'^(PREFACE|INTRODUCTION|ADVERTISEMENT)',
        r'^(LIST\s+OF\s+(PLATES|ILLUSTRATIONS|MAPS))',
        r'^(END\s+OF|FINIS)',
        r'^(VOL\.?|VOLUME)\s*[IVXLC\d]+',  # Volume headers
    ]

    # OCR error patterns (medium confidence RENAME)
    OCR_ERROR_PATTERNS = [
        r'^[^AEIOU]{5,}$',  # No vowels = likely garbled
        r'^[A-Z]{1,2}\d+',  # Letters followed by numbers
        r'[\*\#\@\$\%]',  # Special characters in headword
        r'^\.+',  # Starts with periods
    ]

    def __init__(self):
        self.fragment_re = [re.compile(p, re.IGNORECASE) for p in self.FRAGMENT_PATTERNS]
        self.latin_re = [re.compile(p, re.IGNORECASE) for p in self.LATIN_SUBSECTION_PATTERNS]
        self.delete_re = [re.compile(p, re.IGNORECASE) for p in self.DELETE_PATTERNS]
        self.ocr_error_re = [re.compile(p) for p in self.OCR_ERROR_PATTERNS]

    def classify(self, outlier: Outlier) -> OutlierClassification:
        """Auto-classify an outlier based on patterns."""
        headword = outlier.headword
        text = outlier.text_preview
        word_count = outlier.word_count

        # Check for fragment patterns (MERGE)
        for pattern in self.fragment_re:
            if pattern.match(headword):
                return OutlierClassification(
                    decision='MERGE',
                    confidence='high',
                    reason=f'Headword matches fragment pattern: {pattern.pattern}',
                    merge_target=self._find_merge_target(outlier)
                )

        # Check for delete patterns (DELETE)
        for pattern in self.delete_re:
            if pattern.match(headword):
                return OutlierClassification(
                    decision='DELETE',
                    confidence='high',
                    reason=f'Headword matches non-content pattern: {pattern.pattern}'
                )

        # Check for Latin anatomical/botanical subsection patterns (medium confidence MERGE)
        for pattern in self.latin_re:
            if pattern.match(headword):
                return OutlierClassification(
                    decision='MERGE',
                    confidence='medium',
                    reason=f'Latin technical term (subsection): {pattern.pattern}',
                    merge_target=self._find_merge_target(outlier)
                )

        # Very short text (< 50 words) that looks like a fragment
        if word_count < 50:
            text_lower = text.lower() if text else ''

            # First check for cross-references - these should be KEPT, not merged
            # Cross-refs often start with lowercase (e.g., "a name given by... See X")
            if 'see ' in text_lower[:200] or 'see the article' in text_lower:
                return OutlierClassification(
                    decision='KEEP',
                    confidence='medium',
                    reason='Cross-reference entry (contains "see" directive)'
                )

            # Check if it starts without capital (sentence fragment)
            # But only if it's NOT a cross-reference
            if text and len(text) > 0 and text[0].islower():
                # Additional check: if text is a definition starting with article/preposition, might be valid
                # e.g., "a name given by..." could be a valid definition
                first_word = text.split()[0].lower() if text.split() else ''
                if first_word in ('a', 'an', 'the', 'in', 'on', 'at', 'of', 'or', 'is', 'are', 'was', 'were'):
                    # Could be a valid short definition - check word count
                    if word_count > 10:
                        return OutlierClassification(
                            decision='KEEP',
                            confidence='medium',
                            reason='Short entry with valid definition structure'
                        )
                return OutlierClassification(
                    decision='MERGE',
                    confidence='high',
                    reason='Short text starting with lowercase (sentence fragment)',
                    merge_target=self._find_merge_target(outlier)
                )

            # Very short entries might be index entries
            if word_count < 20:
                return OutlierClassification(
                    decision='MERGE',
                    confidence='medium',
                    reason='Very short entry, likely fragment',
                    merge_target=self._find_merge_target(outlier)
                )

        # Check for OCR errors
        for pattern in self.ocr_error_re:
            if pattern.search(headword):
                return OutlierClassification(
                    decision='RENAME',
                    confidence='medium',
                    reason=f'Headword appears garbled (OCR error): {pattern.pattern}'
                )

        # Geographic subdivision patterns (KEEP)
        geo_prefixes = ['ST ', 'ST. ', 'SAINT ', 'FORT ', 'PORT ', 'CAPE ', 'MOUNT ', 'NEW ', 'LAKE ', 'ISLE ']
        for prefix in geo_prefixes:
            if headword.startswith(prefix):
                return OutlierClassification(
                    decision='KEEP',
                    confidence='medium',
                    reason=f'Geographic prefix "{prefix}" - may be valid subdivision'
                )

        # Default: unclear, needs manual review
        return OutlierClassification(
            decision='REVIEW',
            confidence='low',
            reason='Could not auto-classify - needs manual review'
        )

    def _find_merge_target(self, outlier: Outlier) -> Optional[str]:
        """Find the most likely merge target."""
        # Prefer previous articles (fragments usually belong to preceding article)
        for prev in outlier.prev_articles:
            if prev.get('headword'):
                return prev['headword']
        # Fall back to next articles
        for nxt in outlier.next_articles:
            if nxt.get('headword'):
                return nxt['headword']
        return None


def load_volume_ranges(output_dir: Path) -> dict[int, dict[int, VolumeRange]]:
    """Load volume ranges from volumes_*.jsonl files."""
    ranges = {}

    for vol_file in output_dir.glob('volumes_*.jsonl'):
        year = int(vol_file.stem.split('_')[1])
        ranges[year] = {}

        with open(vol_file) as f:
            for line in f:
                vol = json.loads(line)
                vol_num = vol.get('volume_num', 0)
                raw_range = vol.get('letter_range', '')
                ranges[year][vol_num] = VolumeRange.from_metadata(vol_num, raw_range)

    return ranges


def load_articles(jsonl_path: Path) -> list[dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def detect_volume_outliers(
    articles: list[dict],
    edition_year: int,
    volume_ranges: dict[int, VolumeRange],
    classifier: OutlierClassifier
) -> list[Outlier]:
    """Detect articles outside their volume's alphabetic range."""

    # Group by volume
    by_volume = defaultdict(list)
    for art in articles:
        vol = art.get('volume_num', 0)
        by_volume[vol].append(art)

    outliers = []

    for vol_num in sorted(by_volume.keys()):
        vol_articles = sorted(
            by_volume[vol_num],
            key=lambda a: (a.get('start_page', 0), a.get('headword', ''))
        )

        vol_range = volume_ranges.get(vol_num)

        # Try to derive range from articles if not available or incomplete
        if not vol_range or not vol_range.is_valid:
            vol_range = VolumeRange.from_articles(vol_num, vol_articles)
            if vol_range.is_valid:
                volume_ranges[vol_num] = vol_range
        elif not vol_range.end_headword:
            # Have start but no end - derive end from articles
            vol_range.update_end_from_articles(vol_articles)

        if not vol_range or not vol_range.is_valid:
            continue

        # Skip volumes that span the entire alphabet (likely index/supplement)
        if vol_range.start_headword and vol_range.end_headword:
            if vol_range.start_headword[0] == 'A' and vol_range.end_headword[0] == 'Z':
                continue

        for i, art in enumerate(vol_articles):
            headword = art.get('headword', '')
            if not headword:
                continue

            in_range, reason = vol_range.contains_headword(headword)

            if not in_range:
                # Get context (prev/next articles)
                prev_arts = vol_articles[max(0, i-3):i]
                next_arts = vol_articles[i+1:i+4]

                text = art.get('text', '')
                outlier = Outlier(
                    article_id=art.get('article_id', ''),
                    headword=headword,
                    edition_year=edition_year,
                    volume_num=vol_num,
                    start_page=art.get('start_page', 0),
                    end_page=art.get('end_page', 0),
                    word_count=art.get('word_count', len(text.split())),
                    volume_range=vol_range.raw_range,
                    effective_start=vol_range.start_headword,
                    effective_end=vol_range.end_headword,
                    reason=reason,
                    text_preview=text[:500] if text else '',
                    text_end=text[-300:] if text and len(text) > 300 else '',
                    prev_articles=[
                        {
                            'headword': a.get('headword'),
                            'start_page': a.get('start_page'),
                            'end_page': a.get('end_page')
                        }
                        for a in prev_arts
                    ],
                    next_articles=[
                        {
                            'headword': a.get('headword'),
                            'start_page': a.get('start_page')
                        }
                        for a in next_arts
                    ],
                    merge_candidates=[
                        {
                            'article_id': a.get('article_id'),
                            'headword': a.get('headword'),
                            'start_page': a.get('start_page'),
                            'word_count': a.get('word_count', 0),
                            'direction': 'previous' if a in prev_arts else 'next'
                        }
                        for a in (list(reversed(prev_arts)) + next_arts)[:6]
                    ]
                )

                # Auto-classify
                outlier.classification = classifier.classify(outlier)
                outliers.append(outlier)

    return outliers


def analyze_edition(
    jsonl_path: Path,
    volume_ranges: dict[int, dict[int, VolumeRange]],
    classifier: OutlierClassifier
) -> dict:
    """Analyze a single edition for volume outliers."""
    articles = load_articles(jsonl_path)
    edition_year = int(jsonl_path.stem.split('_')[1])

    # Get or create volume ranges for this edition
    vol_ranges = volume_ranges.get(edition_year, {})
    if edition_year not in volume_ranges:
        volume_ranges[edition_year] = vol_ranges

    outliers = detect_volume_outliers(articles, edition_year, vol_ranges, classifier)

    # Count by classification
    class_counts = defaultdict(int)
    for o in outliers:
        if o.classification:
            class_counts[o.classification.decision] += 1

    # Track which volumes have valid ranges
    volumes_with_ranges = {
        vol_num: {
            'range': vr.raw_range,
            'start': vr.start_headword,
            'end': vr.end_headword,
            'derived': vr.derived
        }
        for vol_num, vr in vol_ranges.items()
        if vr.is_valid
    }

    return {
        'edition_year': edition_year,
        'total_articles': len(articles),
        'volumes_with_ranges': volumes_with_ranges,
        'outliers': [asdict(o) for o in outliers],
        'classification_summary': dict(class_counts)
    }


def main():
    output_dir = Path('output_v2')

    # Load volume ranges from metadata
    print("Loading volume ranges from metadata...")
    volume_ranges = load_volume_ranges(output_dir)

    # Show loaded ranges
    for year in sorted(volume_ranges.keys()):
        valid_vols = [v for v in volume_ranges[year].values() if v.is_valid]
        print(f"  {year}: {len(valid_vols)} volumes with metadata ranges")

    classifier = OutlierClassifier()
    all_results = []

    print("\nDetecting volume outliers (deriving missing ranges from articles)...")
    for jsonl_file in sorted(output_dir.glob('articles_*.jsonl')):
        if 'backup' in jsonl_file.name:
            continue

        print(f"  Analyzing {jsonl_file.name}...")
        result = analyze_edition(jsonl_file, volume_ranges, classifier)
        all_results.append(result)

        year = result['edition_year']
        count = len(result['outliers'])
        summary = result['classification_summary']
        vol_info = result.get('volumes_with_ranges', {})
        derived_count = sum(1 for v in vol_info.values() if v.get('derived'))
        print(f"    {year}: {count} outliers | {len(vol_info)} volumes ({derived_count} derived) | {summary}")

    # Summary
    print("\n" + "="*70)
    print("VOLUME OUTLIER DETECTION SUMMARY")
    print("="*70)

    total_outliers = 0
    total_by_class = defaultdict(int)

    for result in all_results:
        year = result['edition_year']
        count = len(result['outliers'])
        total_outliers += count

        for decision, cnt in result['classification_summary'].items():
            total_by_class[decision] += cnt

        print(f"\n{year}: {count} outliers")

        # Group by volume
        by_vol = defaultdict(list)
        for o in result['outliers']:
            by_vol[o['volume_num']].append(o)

        for vol_num in sorted(by_vol.keys())[:5]:  # Show first 5 volumes
            vol_outliers = by_vol[vol_num]
            print(f"  Vol {vol_num} ({vol_outliers[0]['volume_range']}): {len(vol_outliers)} outliers")
            for o in vol_outliers[:3]:
                cls = o.get('classification', {})
                decision = cls.get('decision', '?')
                confidence = cls.get('confidence', '?')
                print(f"    p.{o['start_page']:4d} | {o['headword'][:25]:25s} | {decision} ({confidence})")
            if len(vol_outliers) > 3:
                print(f"    ... and {len(vol_outliers) - 3} more")

        remaining = len(by_vol) - 5
        if remaining > 0:
            print(f"  ... and {remaining} more volumes with outliers")

    print("\n" + "-"*70)
    print("CLASSIFICATION SUMMARY")
    print("-"*70)
    for decision in ['MERGE', 'DELETE', 'RENAME', 'KEEP', 'REVIEW']:
        count = total_by_class.get(decision, 0)
        if count > 0:
            pct = 100 * count / total_outliers if total_outliers > 0 else 0
            print(f"  {decision:10s}: {count:5d} ({pct:.1f}%)")

    print(f"\nTotal: {total_outliers} outliers across all editions")

    # Save detailed results
    output_file = Path('llm_corrections/outliers/volume_outliers.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Also save a summary CSV for quick analysis
    summary_file = Path('llm_corrections/outliers/volume_outliers_summary.csv')
    with open(summary_file, 'w') as f:
        f.write('edition_year,volume_num,volume_range,headword,page,word_count,decision,confidence,reason\n')
        for result in all_results:
            for o in result['outliers']:
                cls = o.get('classification', {})
                f.write(f"{o['edition_year']},{o['volume_num']},\"{o['volume_range']}\",")
                f.write(f"\"{o['headword']}\",{o['start_page']},{o['word_count']},")
                f.write(f"{cls.get('decision', '')},{cls.get('confidence', '')},\"{cls.get('reason', '')}\"\n")

    print(f"Summary CSV saved to: {summary_file}")


if __name__ == '__main__':
    main()
