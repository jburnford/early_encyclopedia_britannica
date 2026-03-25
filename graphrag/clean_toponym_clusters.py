#!/usr/bin/env python3
"""
Post-processing cleanup for toponym clusters.

Fixes:
1. Old World / New World primary misassignment (Canterbury NZ → Canterbury GB)
2. Bad Wikidata matches (insect genera, family names, etc.)
3. Spelling variant merges (Hindoostan + Hindustan, Surry + Surrey, etc.)
"""

import json
import re
from pathlib import Path
from collections import Counter

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"
INPUT_PATH = NER_DIR / "toponym_clusters.jsonl"
OUTPUT_PATH = NER_DIR / "toponym_clusters_clean.jsonl"
CSV_PATH = NER_DIR / "toponym_clusters_clean.csv"

# --- Old World countries (pre-1860 primary referents) ---

OLD_WORLD = {
    # Europe
    'GB', 'IE', 'FR', 'DE', 'IT', 'ES', 'PT', 'NL', 'AT', 'CH', 'GR', 'TR',
    'PL', 'CZ', 'HU', 'RO', 'BG', 'HR', 'RS', 'BA', 'ME', 'MK', 'AL',
    'LT', 'LV', 'EE', 'FI', 'SE', 'NO', 'DK', 'IS', 'BE', 'LU', 'SK', 'SI',
    'UA', 'BY', 'RU', 'GE', 'AM', 'AZ', 'CY', 'MT',
    # Middle East / North Africa
    'EG', 'LY', 'TN', 'DZ', 'MA', 'IR', 'IQ', 'SY', 'LB', 'JO', 'IL', 'PS',
    'SA', 'YE', 'OM', 'AE', 'QA', 'BH', 'KW',
    # South / East Asia
    'IN', 'PK', 'BD', 'LK', 'NP', 'BT', 'MM', 'TH', 'VN', 'KH', 'LA',
    'MY', 'ID', 'PH', 'CN', 'JP', 'KR', 'KP', 'MN', 'TW', 'AF',
    # Africa
    'ET', 'KE', 'TZ', 'UG', 'RW', 'BI', 'CD', 'CG', 'CM', 'NG', 'GH',
    'SN', 'ML', 'NE', 'TD', 'CF', 'GA', 'GQ', 'AO', 'MZ', 'MG', 'ZW',
    'ZM', 'MW', 'BW', 'NA', 'ZA', 'LS', 'SZ', 'SO', 'DJ', 'ER', 'SD',
    'SS', 'LR', 'SL', 'GN', 'GW', 'CI', 'BF', 'TG', 'BJ', 'MR',
}

NEW_WORLD = {
    'US', 'CA', 'AU', 'NZ',
    'BR', 'AR', 'CL', 'CO', 'VE', 'PE', 'EC', 'BO', 'PY', 'UY',
    'GY', 'SR', 'GF', 'MX', 'GT', 'BZ', 'HN', 'SV', 'NI', 'CR', 'PA',
    'CU', 'JM', 'HT', 'DO', 'TT', 'BB', 'BS', 'AG', 'DM', 'GD', 'KN', 'LC', 'VC',
}

# Countries that ARE the New World place (Mexico = Mexico, Jamaica = Jamaica)
# These should NOT be flipped even though they're in NEW_WORLD
COUNTRY_NAMES = {
    'mexico': 'MX', 'jamaica': 'JM', 'cuba': 'CU', 'brazil': 'BR',
    'peru': 'PE', 'chile': 'CL', 'colombia': 'CO', 'venezuela': 'VE',
    'panama': 'PA', 'haiti': 'HT', 'barbados': 'BB', 'trinidad': 'TT',
    'bahamas': 'BS', 'dominica': 'DM', 'grenada': 'GD',
    'georgia': 'GE',  # The country Georgia, not the US state
    'virginia': 'US',  # No — Virginia is the US state in this corpus
    'palestine': 'PS',
}

# --- Spelling variant merges ---
# target_cluster_id: [source_cluster_ids_to_merge_into_it]
SPELLING_MERGES = {
    'hindustan': ['hindoostan'],
    'surrey': ['surry'],
    'st petersburg': ['peterburgh', 'petersburgh', 'peterburg'],
    'toulouse': ['thoulouse'],
    'nijmegen': ['nimeguen', 'nimwegen'],
    'brunswick': ['brunswic', 'brunswic'],
    'strasbourg': ['strasburg', 'strasburg'],
    'etruria': ['hetruria'],
    'smolensk': ['smolensko'],
    'franche comté': ['franche comte', 'franche compte'],
    'phoenicia': ['phenicia'],
    'ratisbon': ['ratibon'],
    'west indies': ['west india'],
    'east indies': ['east india'],
    'kamtschatka': ['kamtchatka'],  # keep the more common spelling
}

# --- Wikidata geographic description terms ---

GEO_TERMS = {
    'island', 'islands', 'archipelago', 'atoll',
    'region', 'regions', 'area', 'unit',  # "regional unit"
    'sea', 'ocean', 'river', 'rivers', 'lake', 'lakes', 'waterway',
    'mountain', 'mountains', 'range', 'hill', 'hills', 'peak', 'volcano',
    'peninsula', 'gulf', 'strait', 'bay', 'cape', 'coast', 'shore',
    'kingdom', 'empire', 'caliphate', 'sultanate', 'khanate',
    'province', 'county', 'state', 'territory', 'colony', 'colonies',
    'city', 'town', 'village', 'settlement', 'port', 'frazione',
    'country', 'nation', 'republic', 'duchy', 'department', 'principality',
    'continent', 'subcontinent',
    'historical', 'ancient', 'former', 'medieval', 'classical',
    'district', 'commune', 'municipality', 'borough', 'parish', 'quarter',
    'satrapy', 'prefecture', 'voivodeship', 'canton', 'shire',
    'desert', 'valley', 'plain', 'plateau', 'steppe', 'forest',
    'channel', 'passage', 'inlet', 'fjord', 'lagoon', 'marsh', 'delta',
    'federal', 'autonomous', 'community',  # "autonomous community"
    'polity', 'civilization', 'nationality',  # "nationality and autonomous community"
    'capital', 'capitals',
    'banks', 'bank',  # "city on the banks of"
    'northern', 'southern', 'eastern', 'western',  # directional in place descriptions
    'greece', 'italy', 'spain', 'france', 'turkey', 'india', 'china',
    'russia', 'england', 'germany', 'austria', 'tunisia', 'egypt',
    'syria', 'iran', 'iraq', 'pakistan', 'japan',  # country names in descriptions
    'europe', 'asia', 'africa', 'americas',
    'roman', 'greek', 'ottoman', 'byzantine', 'persian', 'british', 'french',
    'german', 'spanish', 'portuguese', 'dutch',
}


def load_clusters():
    clusters = []
    with open(INPUT_PATH) as f:
        for line in f:
            clusters.append(json.loads(line))
    return clusters


def fix_old_new_world(clusters):
    """Swap primary/alternative when Old World place should be primary."""
    fixed = 0
    for rec in clusters:
        if rec.get('match_type') != 'matched' or not rec.get('alternatives'):
            continue

        primary_country = rec.get('country', '')
        label_lower = rec['label'].lower()

        # Skip if the label IS a New World country name
        if label_lower in COUNTRY_NAMES:
            expected = COUNTRY_NAMES[label_lower]
            if primary_country == expected:
                continue

        # Check if primary is New World
        if primary_country not in NEW_WORLD:
            continue

        # Find best Old World alternative
        old_world_alts = [a for a in rec['alternatives'] if a.get('country') in OLD_WORLD]
        if not old_world_alts:
            continue

        # Swap: demote current primary to alternative, promote best Old World
        best_ow = max(old_world_alts, key=lambda a: a.get('population') or 0)

        # Build the old primary as an alternative entry
        old_primary = {
            'geonames_id': rec.get('geonames_id'),
            'name': rec.get('geonames_name'),
            'country': rec.get('country'),
            'feature_class': rec.get('feature_class'),
            'feature_code': rec.get('feature_code'),
            'population': rec.get('population'),
            'lat': rec.get('lat'),
            'lon': rec.get('lon'),
        }

        # Promote Old World candidate to primary
        rec['geonames_id'] = best_ow['geonames_id']
        rec['geonames_name'] = best_ow['name']
        rec['country'] = best_ow['country']
        rec['feature_class'] = best_ow['feature_class']
        rec['feature_code'] = best_ow['feature_code']
        rec['population'] = best_ow['population']
        rec['lat'] = best_ow['lat']
        rec['lon'] = best_ow['lon']

        # Rebuild alternatives: remove the promoted one, add the demoted one
        new_alts = [a for a in rec['alternatives'] if a.get('geonames_id') != best_ow['geonames_id']]
        new_alts.insert(0, old_primary)
        rec['alternatives'] = new_alts

        fixed += 1

    print(f"  Fixed {fixed} Old World / New World primary swaps")
    return clusters


def fix_wikidata_matches(clusters):
    """Remove Wikidata matches that are clearly not geographic entities.

    Strategy:
    - If description contains "genus of", "family name", "given name",
      "video game", "TV series" etc. → definitely not a place
    - If description contains geographic terms → keep
    - If it's a concept headword appearing in 5+ editions → keep regardless
      (the encyclopedia has an article about it, so it's a real place)
    - Otherwise, if description is empty or ambiguous → keep if concept headword
    """
    removed = 0
    kept_as_concept = 0

    # Patterns that indicate definitely NOT a place
    NOT_GEO_PATTERNS = [
        'genus of', 'genus in', 'family name', 'given name', 'surname',
        'video game', 'tv series', 'television', 'film', 'novel', 'song',
        'album', 'band', 'magazine', 'newspaper',
        'protein', 'enzyme', 'chemical', 'mineral species',
        'species of', 'breed of',
    ]

    for rec in clusters:
        if rec.get('match_type') != 'wikidata':
            continue

        desc = (rec.get('wikidata_description') or '').lower()
        desc_words = set(re.findall(r'\w+', desc))

        # Check if it's obviously NOT a place
        is_not_geo = any(pat in desc for pat in NOT_GEO_PATTERNS)

        if is_not_geo:
            # But keep if it's a concept headword in 5+ editions
            # (the encyclopedia article IS about the place, Wikidata just
            # returned the wrong entity)
            if rec.get('is_concept_headword') and rec.get('edition_count', 0) >= 5:
                rec['wikidata_note'] = f'bad_match_kept_as_concept (was: {desc})'
                kept_as_concept += 1
            else:
                rec['match_type'] = 'none'
                rec.pop('wikidata_qid', None)
                rec.pop('wikidata_label', None)
                rec.pop('wikidata_description', None)
                removed += 1

    print(f"  Removed {removed} non-geographic Wikidata matches")
    print(f"  Kept {kept_as_concept} bad Wikidata matches because they're concept headwords (need re-linking)")
    return clusters


def merge_spelling_variants(clusters):
    """Merge known spelling variant clusters."""
    by_id = {r['cluster_id']: r for r in clusters}
    merged_count = 0
    to_remove = set()

    for target_id, source_ids in SPELLING_MERGES.items():
        target = by_id.get(target_id)
        if not target:
            continue

        for src_id in source_ids:
            src = by_id.get(src_id)
            if not src:
                continue

            # Merge source into target
            target['variants'].extend(src['variants'])
            target['total_mentions'] += src['total_mentions']
            target['article_count'] += src['article_count']

            # Merge edition counts
            for ed, cnt in src.get('by_edition', {}).items():
                ed_key = str(ed) if isinstance(ed, int) else ed
                target_ed = target.get('by_edition', {})
                if ed_key in target_ed:
                    target_ed[ed_key] += cnt
                else:
                    target_ed[ed_key] = cnt

            target['edition_count'] = len(target.get('by_edition', {}))

            # If source had a concept headword flag, propagate
            if src.get('is_concept_headword') and not target.get('is_concept_headword'):
                target['is_concept_headword'] = True
                target['concept_label'] = src.get('concept_label')

            to_remove.add(src_id)
            merged_count += 1

    clusters = [r for r in clusters if r['cluster_id'] not in to_remove]

    # Re-sort and re-rank
    clusters.sort(key=lambda r: r['total_mentions'], reverse=True)
    for i, rec in enumerate(clusters):
        rec['frequency_rank'] = i + 1

    print(f"  Merged {merged_count} spelling variant pairs, removed {len(to_remove)} duplicate clusters")
    return clusters


def save_clusters(clusters, path):
    with open(path, 'w') as f:
        for rec in clusters:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')
    print(f"  Saved {len(clusters):,} clusters to {path}")


def save_csv(clusters, path):
    import csv
    editions = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'rank', 'cluster_id', 'label', 'total_mentions', 'article_count',
            'edition_count', 'is_concept_headword', 'match_type',
            'country', 'geonames_id', 'population', 'lat', 'lon',
            'wikidata_qid', 'wikidata_description',
            'alternatives_summary', 'variants',
            *[str(y) for y in editions],
        ])
        for r in clusters:
            alts = r.get('alternatives', [])
            alt_str = '; '.join(
                f"{a['country']}(pop={a.get('population') or 0:,})"
                for a in alts
            ) if alts else ''

            w.writerow([
                r['frequency_rank'],
                r['cluster_id'],
                r['label'],
                r['total_mentions'],
                r['article_count'],
                r['edition_count'],
                r.get('is_concept_headword', False),
                r.get('match_type', ''),
                r.get('country', ''),
                r.get('geonames_id', ''),
                r.get('population', ''),
                r.get('lat', ''),
                r.get('lon', ''),
                r.get('wikidata_qid', ''),
                r.get('wikidata_description', ''),
                alt_str,
                '; '.join(r['variants'][:10]),
                *[r['by_edition'].get(str(y), r['by_edition'].get(y, 0)) for y in editions],
            ])

    print(f"  Saved CSV to {path}")


def print_summary(clusters, min_mentions=10):
    """Print summary stats after cleanup."""
    above = [r for r in clusters if r['total_mentions'] >= min_mentions]
    total_mentions = sum(r['total_mentions'] for r in clusters)

    by_type = Counter(r.get('match_type', 'none') for r in above)
    grounded = sum(1 for r in above if r.get('match_type') in ('matched', 'wikidata'))
    grounded_mentions = sum(r['total_mentions'] for r in above if r.get('match_type') in ('matched', 'wikidata'))
    with_alts = sum(1 for r in above if r.get('match_type') == 'matched' and r.get('alternatives'))

    print(f"\n=== CLEAN TOPONYM SUMMARY (>= {min_mentions} mentions) ===")
    print(f"  Clusters: {len(above):,} / {len(clusters):,} total")
    print(f"  Grounded: {grounded:,} ({100*grounded/len(above):.1f}%)")
    print(f"  With alternatives: {with_alts:,}")
    print(f"  Grounded mentions: {grounded_mentions:,} / {total_mentions:,} ({100*grounded_mentions/total_mentions:.1f}%)")
    print(f"  Match types: {dict(by_type)}")

    # Show the top 10 with alternatives after cleanup
    alt_recs = [r for r in above if r.get('alternatives')]
    alt_recs.sort(key=lambda r: r['total_mentions'], reverse=True)
    print(f"\n  Top 10 with significant alternatives:")
    for r in alt_recs[:10]:
        alts = ', '.join(f"{a['country']}({a.get('population') or 0:,})" for a in r['alternatives'][:3])
        print(f"    {r['label']} ({r['total_mentions']:,}): primary={r['country']}({r.get('population') or 0:,}) alts=[{alts}]")

    # Show top 10 still unmatched
    unmatched = [r for r in above if r.get('match_type') == 'none']
    unmatched.sort(key=lambda r: r['total_mentions'], reverse=True)
    print(f"\n  Top 10 still unmatched:")
    for r in unmatched[:10]:
        concept = ' [C]' if r.get('is_concept_headword') else ''
        print(f"    {r['label']}: {r['total_mentions']:,} mentions{concept}")


def main():
    print("Loading clusters...")
    clusters = load_clusters()
    print(f"  {len(clusters):,} clusters loaded")

    print("\nFix 1: Old World / New World primary assignment...")
    clusters = fix_old_new_world(clusters)

    print("\nFix 2: Remove non-geographic Wikidata matches...")
    clusters = fix_wikidata_matches(clusters)

    print("\nFix 3: Merge spelling variants...")
    clusters = merge_spelling_variants(clusters)

    print("\nSaving results...")
    save_clusters(clusters, OUTPUT_PATH)
    save_csv(clusters, CSV_PATH)

    print_summary(clusters)

    print("\nDone!")


if __name__ == '__main__':
    main()
