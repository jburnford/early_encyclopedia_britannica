#!/usr/bin/env python3
"""
Re-query Wikidata for unmatched and badly-matched toponym clusters.

Uses the Wikidata SPARQL endpoint to search specifically for geographic entities,
avoiding the genus-of-insects problem from the basic search API.

Strategy:
1. For each unmatched label, query SPARQL for items that:
   - Have a matching label or alias in English
   - Are instances of geographic/political entity types
2. Pick the best match based on sitelinks count (proxy for notability)
"""

import json
import re
import time
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"
INPUT_PATH = NER_DIR / "toponym_clusters_clean.jsonl"
OUTPUT_PATH = NER_DIR / "toponym_clusters_clean.jsonl"  # overwrite in place

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Geographic entity types (P31 = instance of)
# We search for items that are instances of any subclass of these
GEO_TYPES = """
  wd:Q515         # city
  wd:Q532         # village
  wd:Q486972      # human settlement
  wd:Q3624078     # sovereign state
  wd:Q3024240     # historical country
  wd:Q28171280    # ancient civilization
  wd:Q6256        # country
  wd:Q35657       # U.S. state
  wd:Q82794       # geographic region
  wd:Q34763       # peninsula
  wd:Q23442       # island
  wd:Q4022        # river
  wd:Q8502        # mountain
  wd:Q165         # sea
  wd:Q9430        # ocean
  wd:Q39816       # valley
  wd:Q35509       # cave
  wd:Q23397       # lake
  wd:Q34038       # waterfall
  wd:Q185113      # bay
  wd:Q180874      # strait
  wd:Q217403      # cape
  wd:Q133056      # desert
  wd:Q46831       # mountain range
  wd:Q41176       # building
  wd:Q16917       # hospital
  wd:Q3914       # school
  wd:Q35127       # website
  wd:Q15284       # municipality
  wd:Q1093829     # city-state
  wd:Q1620908     # historical region
  wd:Q1496967     # territorial entity
  wd:Q56061        # administrative territorial entity
  wd:Q7930989     # city/town
  wd:Q1549591     # big city
  wd:Q174844      # province
  wd:Q1352230     # historical province
  wd:Q6465        # gulf
  wd:Q12280       # bridge
  wd:Q190107      # plateau
  wd:Q36784       # volcano
  wd:Q46841       # harbor
  wd:Q83620       # canale
  wd:Q928830      # metro station
  wd:Q1233637     # polity
  wd:Q3024240     # historical country
  wd:Q133442      # colony
  wd:Q160016      # palatinate
  wd:Q164142      # duchy
  wd:Q1250464     # shire
  wd:Q1187580     # historical geographic area
"""


def sparql_search(label: str):
    """Search Wikidata SPARQL for geographic entities matching label."""
    # Escape for SPARQL
    escaped = label.replace('"', '\\"').replace("'", "\\'")

    query = f"""
    SELECT ?item ?itemLabel ?itemDescription ?sitelinks WHERE {{
      ?item rdfs:label "{escaped}"@en .
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 0)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT 10
    """

    params = urlencode({'query': query, 'format': 'json'})
    url = f"{WIKIDATA_SPARQL}?{params}"
    req = Request(url, headers={
        'User-Agent': 'EncyclopediaBritannicaKG/1.0 (research project)',
        'Accept': 'application/sparql-results+json',
    })

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get('results', {}).get('bindings', [])
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        return None


def sparql_search_alias(label: str):
    """Fallback: search by alias (skos:altLabel) instead of rdfs:label."""
    escaped = label.replace('"', '\\"').replace("'", "\\'")

    query = f"""
    SELECT ?item ?itemLabel ?itemDescription ?sitelinks WHERE {{
      ?item skos:altLabel "{escaped}"@en .
      ?item wikibase:sitelinks ?sitelinks .
      FILTER(?sitelinks > 0)
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC(?sitelinks)
    LIMIT 10
    """

    params = urlencode({'query': query, 'format': 'json'})
    url = f"{WIKIDATA_SPARQL}?{params}"
    req = Request(url, headers={
        'User-Agent': 'EncyclopediaBritannicaKG/1.0 (research project)',
        'Accept': 'application/sparql-results+json',
    })

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return data.get('results', {}).get('bindings', [])
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
        return None


# Terms in Wikidata descriptions that indicate a place
GEO_DESC_TERMS = {
    'island', 'islands', 'archipelago', 'atoll',
    'region', 'unit', 'area',
    'sea', 'ocean', 'river', 'lake', 'waterway',
    'mountain', 'range', 'hill', 'peak', 'volcano',
    'peninsula', 'gulf', 'strait', 'bay', 'cape', 'coast',
    'kingdom', 'empire', 'caliphate', 'sultanate',
    'province', 'county', 'state', 'territory', 'colony',
    'city', 'town', 'village', 'settlement', 'port', 'frazione',
    'country', 'nation', 'republic', 'duchy', 'department',
    'continent', 'subcontinent',
    'historical', 'ancient', 'former', 'medieval', 'classical',
    'district', 'commune', 'municipality', 'borough', 'parish',
    'satrapy', 'prefecture', 'canton', 'shire',
    'desert', 'valley', 'plain', 'plateau', 'steppe',
    'channel', 'passage', 'inlet', 'fjord', 'lagoon', 'delta',
    'autonomous', 'community', 'polity', 'civilization',
    'capital', 'greece', 'italy', 'spain', 'france', 'turkey',
    'india', 'china', 'russia', 'england', 'germany', 'austria',
    'europe', 'asia', 'africa', 'roman', 'greek', 'ottoman',
    'byzantine', 'persian', 'british', 'french', 'german', 'spanish',
    'administrative', 'geographic',
    'ruins', 'archaeological', 'site', 'fortress', 'castle',
    'battle', 'siege',  # battle sites are places
}

NOT_GEO_PATTERNS = [
    'genus of', 'genus in', 'family name', 'given name', 'surname',
    'video game', 'tv series', 'film', 'novel', 'song', 'album',
    'protein', 'enzyme', 'chemical', 'species of', 'breed of',
    'taxon', 'taxonomic',
]


def pick_best_geo_result(results):
    """From SPARQL results, pick the most likely geographic entity."""
    for r in results:
        desc = (r.get('itemDescription', {}).get('value', '') or '').lower()
        desc_words = set(re.findall(r'\w+', desc))

        # Skip if clearly not geographic
        if any(pat in desc for pat in NOT_GEO_PATTERNS):
            continue

        # Accept if description contains geographic terms
        if desc_words & GEO_DESC_TERMS:
            qid = r['item']['value'].split('/')[-1]
            label = r.get('itemLabel', {}).get('value', '')
            return qid, label, desc

        # Accept if description is empty but has many sitelinks (notable entity)
        sitelinks = int(r.get('sitelinks', {}).get('value', 0))
        if sitelinks >= 10 and not any(pat in desc for pat in NOT_GEO_PATTERNS):
            qid = r['item']['value'].split('/')[-1]
            label = r.get('itemLabel', {}).get('value', '')
            return qid, label, desc

    return None, None, None


def main():
    print("Loading clean clusters...")
    clusters = []
    with open(INPUT_PATH) as f:
        for line in f:
            clusters.append(json.loads(line))

    # Find clusters that need re-querying
    to_requery = []
    for rec in clusters:
        if rec['total_mentions'] < 10:
            continue
        # Unmatched
        if rec.get('match_type') == 'none':
            to_requery.append(rec)
        # Bad Wikidata match (kept as concept)
        elif rec.get('wikidata_note', '').startswith('bad_match_kept_as_concept'):
            to_requery.append(rec)

    print(f"  {len(to_requery)} clusters to re-query")

    matched = 0
    failed = 0
    errors = 0

    for i, rec in enumerate(to_requery):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(to_requery)} ({matched} matched, {failed} failed, {errors} errors)")

        label = rec['label']

        # Try exact label match first
        results = sparql_search(label)
        if results is None:
            errors += 1
            time.sleep(2)
            continue

        qid, wd_label, desc = pick_best_geo_result(results)

        # If no match on label, try alias search
        if qid is None:
            time.sleep(0.5)
            results = sparql_search_alias(label)
            if results is None:
                errors += 1
                time.sleep(2)
                continue
            qid, wd_label, desc = pick_best_geo_result(results)

        if qid:
            rec['match_type'] = 'wikidata'
            rec['wikidata_qid'] = qid
            rec['wikidata_label'] = wd_label
            rec['wikidata_description'] = desc
            rec.pop('wikidata_note', None)
            matched += 1
        else:
            failed += 1

        # Rate limit: be polite to WDQS
        time.sleep(1.0)

    print(f"\nRe-query complete:")
    print(f"  Matched: {matched}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")

    # Save updated clusters
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        for rec in clusters:
            json.dump(rec, f, ensure_ascii=False)
            f.write('\n')

    # Summary
    above = [r for r in clusters if r['total_mentions'] >= 10]
    grounded = sum(1 for r in above if r.get('match_type') in ('matched', 'wikidata'))
    still_none = sum(1 for r in above if r.get('match_type') == 'none')
    print(f"\n  Grounded: {grounded}/{len(above)} ({100*grounded/len(above):.1f}%)")
    print(f"  Still unmatched: {still_none}")

    # Show remaining unmatched
    unmatched = [r for r in above if r.get('match_type') == 'none']
    unmatched.sort(key=lambda r: r['total_mentions'], reverse=True)
    print(f"\n  Top 20 still unmatched:")
    for r in unmatched[:20]:
        c = ' [C]' if r.get('is_concept_headword') else ''
        print(f"    {r['label']}: {r['total_mentions']:,} mentions, {r['edition_count']} eds{c}")

    print("\nDone!")


if __name__ == '__main__':
    main()
