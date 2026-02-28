#!/usr/bin/env python3
"""Clean the headword dictionary for use in the Britannica GraphRAG.

Reads headword_dictionary.json and produces headword_dictionary_clean.json:
1. Remove sentence fragments (keys with 5+ words)
2. Remove stop-word starts (THIS, WHEN, ALL, THERE, etc.)
3. Remove structural artifacts (END OF, VOLVME, JNDEX standalone, etc.)
4. Fix corrupted display headwords (J-for-I, V-for-U still in headword field)
5. Split aliases (entries with ", OR ")
6. Merge hyphen/space duplicates
7. Remove 1-2 char noise (keep those with edition_count >= 3)
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import defaultdict


def normalize_sort_key(headword: str) -> str:
    """Replicate the parser's sort key normalization."""
    key = headword.upper()
    key = key.replace('U', 'V').replace('I', 'J')
    key = unicodedata.normalize('NFKD', key)
    key = key.encode('ASCII', 'ignore').decode('ASCII')
    key = re.sub(r"['\-]", '', key)
    key = re.sub(r'\s+', ' ', key).strip()
    return key


def reverse_sort_key(sort_key: str) -> str:
    """Best-effort reverse of 18th-century sort key to display form.

    This handles the common cases:
    - J before a vowel → I (JNDEX → INDEX)
    - V before a consonant → U (VBJQVJTY → UBIQUITY)
    But preserves legitimate J and V usage where possible.
    """
    result = list(sort_key)
    vowels = set('AEJOUY')  # Note: in sort-key space, I→J already

    for i, ch in enumerate(result):
        next_ch = result[i + 1] if i + 1 < len(result) else None

        if ch == 'J':
            # J before a vowel (A,E,O,U,Y) or at end of word → likely I
            # But J before consonant is legitimate J (already mapped from I→J)
            # Actually in sort-key: I→J, so ALL J were originally I or J
            # We want: J before A,E,O,V,Y → I (was originally I before vowel)
            # J before consonant → keep J (was originally I before consonant,
            #   but that's wrong — original I before consonant should stay I)
            # The trick: in English, I before consonant is common (IN, IT, IS)
            #   while J before consonant is rare (only in borrowed words)
            # So: J → I unless it's word-initial before a vowel AND looks like
            #   a legitimate J word (JACK, JAM, etc.)
            # Simplest correct approach: J → I always, then fix known J-words
            result[i] = 'I'

        elif ch == 'V':
            # V before consonant → U (VNDER → UNDER, VBJQVJTY → UBIQUITY)
            # V before vowel → keep V (VALE, VINE)
            if next_ch and next_ch not in 'AEIJOUY':
                result[i] = 'U'
            elif next_ch is None:
                # V at end of word: could be U (PERV→PERU) but rare
                # Keep as V by default
                pass

    return ''.join(result)


# Known words that start with J (not I) — expand as needed
LEGITIMATE_J_WORDS = {
    'JACK', 'JACOB', 'JACOBIN', 'JACOBINS', 'JACOBITE', 'JACOBITES',
    'JACOBUS', 'JAFFA', 'JAGO', 'JAGUAR', 'JAIL', 'JAILER',
    'JALAP', 'JAM', 'JAMAICA', 'JAMBA', 'JAMES', 'JAMESON',
    'JANISSARIES', 'JANISSARY', 'JANUARY', 'JAPAN', 'JAPANESE',
    'JAR', 'JARGON', 'JASMINE', 'JASPER', 'JAUNTING', 'JAVA',
    'JAVELIN', 'JAW', 'JAY', 'JEALOUSY', 'JEDDAH', 'JELLY',
    'JENNET', 'JENNY', 'JERICHO', 'JERSEY', 'JERUSALEM', 'JESSAMINE',
    'JEST', 'JESUIT', 'JESUITS', 'JESUS', 'JET', 'JEW', 'JEWEL',
    'JEWELLERY', 'JEWELRY', 'JEWS', 'JIG', 'JILT', 'JOB', 'JOCKEY',
    'JOIN', 'JOINER', 'JOINT', 'JOINTS', 'JOKE', 'JOLLY', 'JONAH',
    'JORDAN', 'JOURNAL', 'JOURNALISM', 'JOURNEY', 'JOURNEYMAN',
    'JOUST', 'JOY', 'JUBILEE', 'JUDAH', 'JUDGE', 'JUDGES',
    'JUDGMENT', 'JUDICATURE', 'JUDICIAL', 'JUDICIARY', 'JUG',
    'JUGGLER', 'JUGULAR', 'JUICE', 'JULIAN', 'JULY', 'JUMP',
    'JUNCTION', 'JUNE', 'JUNGLE', 'JUNIOR', 'JUNIPER', 'JUNK',
    'JUNO', 'JUPITER', 'JURISDICTION', 'JURISPRUDENCE', 'JUROR',
    'JURY', 'JUST', 'JUSTICE', 'JUSTICIARY', 'JUSTIFICATION',
    'JUVENILE', 'JUXTAPOSITION',
}


def fix_display_headword(sort_key: str, current_headword: str) -> str:
    """Fix a corrupted display headword where headword == sort_key.

    Only called when the headword field still looks like a sort key
    (contains J-for-I or V-for-U patterns).
    """
    if sort_key != current_headword:
        return current_headword  # Already corrected

    candidate = reverse_sort_key(sort_key)

    # Restore legitimate J for known words
    first_word = candidate.split()[0] if candidate else ''
    for jword in LEGITIMATE_J_WORDS:
        if candidate.startswith(jword.replace('J', 'I')):
            # Check if the J version is more plausible
            candidate = jword + candidate[len(jword):]
            break

    # Title case for multi-word entries
    if ' ' in candidate:
        # Keep as uppercase (encyclopedia convention)
        pass

    return candidate


def is_corrupted_headword(sort_key: str, headword: str) -> bool:
    """Check if a headword still contains J-for-I or V-for-U artifacts."""
    if sort_key != headword:
        return False  # Already has a different display form

    # Check for V-before-consonant (strong signal of corruption)
    if re.search(r'V[BCDFGHJKLMNPQRSTVWXYZ]', headword):
        return True

    # Check for J-before-vowel that's NOT a legitimate J word
    if re.search(r'J[AEOU]', headword):
        first_word = headword.split()[0]
        if first_word not in LEGITIMATE_J_WORDS:
            return True

    return False


STOP_WORD_STARTS = {
    'THIS', 'THJS', 'WHEN', 'ALL', 'THERE', 'THESE', 'THOSE', 'SOME',
    'FROM', 'WERE', 'HAVE', 'BEEN', 'THEY', 'THAT', 'WITH', 'WHICH',
    'WHAT', 'UPON', 'INTO', 'ALSO', 'THUS', 'SUCH',
}

STRUCTURAL_PATTERNS = [
    r'^END OF',
    r'^VOLVME$',
    r'^JNDEX$',
    r'^JNDEX [A-Z]',  # "JNDEX ABORTJON" etc.
    r'^PLATE EXPLANATJON',
    r'^DJRECTJONS FOR',
    r'^FJNJS$',
    r'^ERRАТА$',
    r'^ERRATA',
    r'^ADDENDA',
    r'^CORRIGENDA',
    r'^CONTENTS',
    r'^PREFACE',
    r'^ADVERTISEMENT',
    r'^ENCYCLOP',
    r'^SUPPLEMENT',
    r'^TABLE OF',
    r'^LIST OF',
]

STRUCTURAL_RE = re.compile('|'.join(STRUCTURAL_PATTERNS))


def clean_dictionary(input_path: str, output_path: str) -> dict:
    """Clean the headword dictionary and return stats."""
    with open(input_path) as f:
        raw = json.load(f)

    stats = defaultdict(int)
    stats['input_entries'] = len(raw)
    cleaned = {}
    removed_keys = set()

    # Pass 1: Remove obvious junk
    for key, entry in raw.items():
        headword = entry['headword']
        edition_count = entry.get('edition_count', 0)
        words = key.split()

        # 1. Sentence fragments (5+ words)
        if len(words) >= 5:
            stats['removed_sentence_fragments'] += 1
            removed_keys.add(key)
            continue

        # 2. Stop-word starts
        if words and words[0] in STOP_WORD_STARTS:
            stats['removed_stop_word_starts'] += 1
            removed_keys.add(key)
            continue

        # 3. Structural artifacts
        if STRUCTURAL_RE.search(key):
            stats['removed_structural_artifacts'] += 1
            removed_keys.add(key)
            continue

        # 7. Short noise (1-2 chars with low edition count)
        if len(key.replace(' ', '')) <= 2 and edition_count < 3:
            stats['removed_short_noise'] += 1
            removed_keys.add(key)
            continue

        cleaned[key] = dict(entry)  # Copy

    # Pass 2: Fix corrupted display headwords
    for key, entry in cleaned.items():
        if is_corrupted_headword(key, entry['headword']):
            old_hw = entry['headword']
            entry['headword'] = fix_display_headword(key, entry['headword'])
            if old_hw != entry['headword']:
                stats['fixed_corrupted_headwords'] += 1

    # Pass 3: Split aliases (", OR " entries)
    to_add = {}
    to_remove = set()
    for key, entry in list(cleaned.items()):
        headword = entry['headword']
        if ', OR ' in headword:
            parts = [p.strip() for p in headword.split(', OR ')]
            parts = [p for p in parts if p]  # Remove empty
            if len(parts) >= 2:
                # Primary = first part
                primary = parts[0]
                aliases = parts[1:]
                entry['headword'] = primary
                entry['aliases'] = aliases
                stats['split_aliases'] += 1

                # Create alias entries pointing back
                for alias in aliases:
                    alias_key = normalize_sort_key(alias)
                    if alias_key not in cleaned and alias_key not in to_add:
                        to_add[alias_key] = {
                            'headword': alias,
                            'sources': entry.get('sources', []),
                            'editions': entry.get('editions', []),
                            'source_count': entry.get('source_count', 0),
                            'edition_count': entry.get('edition_count', 0),
                            'alias_of': primary,
                        }
                        stats['alias_entries_created'] += 1
        # Also handle ", or " (lowercase)
        elif ', or ' in headword:
            parts = [p.strip() for p in headword.split(', or ')]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                primary = parts[0]
                aliases = parts[1:]
                entry['headword'] = primary
                entry['aliases'] = aliases
                stats['split_aliases'] += 1

                for alias in aliases:
                    alias_key = normalize_sort_key(alias)
                    if alias_key not in cleaned and alias_key not in to_add:
                        to_add[alias_key] = {
                            'headword': alias,
                            'sources': entry.get('sources', []),
                            'editions': entry.get('editions', []),
                            'source_count': entry.get('source_count', 0),
                            'edition_count': entry.get('edition_count', 0),
                            'alias_of': primary,
                        }
                        stats['alias_entries_created'] += 1

    cleaned.update(to_add)

    # Pass 4: Merge hyphen/space duplicates
    # Group by normalized form (remove hyphens)
    norm_groups = defaultdict(list)
    for key in cleaned:
        norm = cleaned[key]['headword'].replace('-', ' ').replace('  ', ' ').strip().upper()
        norm_groups[norm].append(key)

    merge_remove = set()
    for norm, keys in norm_groups.items():
        if len(keys) <= 1:
            continue
        # Keep the one with highest edition_count
        keys.sort(key=lambda k: cleaned[k].get('edition_count', 0), reverse=True)
        primary_key = keys[0]
        for secondary_key in keys[1:]:
            sec = cleaned[secondary_key]
            # Add as alias
            if 'aliases' not in cleaned[primary_key]:
                cleaned[primary_key]['aliases'] = []
            cleaned[primary_key]['aliases'].append(sec['headword'])
            # Merge editions
            primary_editions = set(cleaned[primary_key].get('editions', []))
            sec_editions = set(sec.get('editions', []))
            merged = sorted(primary_editions | sec_editions)
            cleaned[primary_key]['editions'] = merged
            cleaned[primary_key]['edition_count'] = len(merged)
            # Merge sources
            primary_sources = set(cleaned[primary_key].get('sources', []))
            sec_sources = set(sec.get('sources', []))
            cleaned[primary_key]['sources'] = sorted(primary_sources | sec_sources)
            merge_remove.add(secondary_key)
            stats['merged_hyphen_space_dupes'] += 1

    for key in merge_remove:
        del cleaned[key]

    # Pass 5: Deduplicate and clean aliases
    for key, entry in cleaned.items():
        if 'aliases' in entry:
            primary = entry['headword'].upper()
            # Remove self-references and duplicates
            unique = []
            seen = {primary}
            for a in entry['aliases']:
                a_upper = a.upper()
                if a_upper not in seen:
                    seen.add(a_upper)
                    unique.append(a)
            if unique:
                entry['aliases'] = unique
            else:
                del entry['aliases']

    stats['output_entries'] = len(cleaned)

    # Write output
    with open(output_path, 'w') as f:
        json.dump(cleaned, f, indent=1, ensure_ascii=False)

    return dict(stats)


def main():
    base = Path(__file__).parent.parent
    input_path = base / 'data' / 'headword_dictionary.json'
    output_path = base / 'data' / 'headword_dictionary_clean.json'

    print(f"Reading {input_path}...")
    stats = clean_dictionary(str(input_path), str(output_path))

    print(f"\nCleaning results:")
    print(f"  Input entries:              {stats['input_entries']:>6}")
    print(f"  Removed sentence fragments: {stats.get('removed_sentence_fragments', 0):>6}")
    print(f"  Removed stop-word starts:   {stats.get('removed_stop_word_starts', 0):>6}")
    print(f"  Removed structural:         {stats.get('removed_structural_artifacts', 0):>6}")
    print(f"  Removed short noise:        {stats.get('removed_short_noise', 0):>6}")
    print(f"  Fixed corrupted headwords:  {stats.get('fixed_corrupted_headwords', 0):>6}")
    print(f"  Split aliases:              {stats.get('split_aliases', 0):>6}")
    print(f"  Alias entries created:      {stats.get('alias_entries_created', 0):>6}")
    print(f"  Merged hyphen/space dupes:  {stats.get('merged_hyphen_space_dupes', 0):>6}")
    print(f"  Output entries:             {stats['output_entries']:>6}")
    print(f"\nWrote {output_path}")


if __name__ == '__main__':
    main()
