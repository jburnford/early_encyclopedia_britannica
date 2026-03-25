#!/usr/bin/env python3
"""
Enrich toponym clusters with Wikidata QIDs and validate all matches.

1. For GeoNames-matched clusters: look up QID via P1566 (GeoNames ID property)
2. For all clusters with QIDs: pull P31 (instance of) + sitelinks count
3. Flag non-geographic entities for review

Uses the public Wikidata SPARQL endpoint in batches.
"""

import json
import time
import urllib.request
import urllib.parse
import sys
from pathlib import Path
from collections import defaultdict

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
CLUSTERS_FILE = Path("data/ner/toponym_clusters_clean.jsonl")
OUTPUT_FILE = Path("data/ner/toponym_clusters_enriched.jsonl")
REVIEW_FILE = Path("data/ner/toponym_qid_review.txt")
BATCH_SIZE = 150  # VALUES clause limit for SPARQL

# Geographic P31 classes (and their subclasses) that are valid matches
# We'll check if ANY P31 value is geographic
GEO_KEYWORDS = {
    'city', 'town', 'village', 'country', 'state', 'province', 'region',
    'county', 'island', 'river', 'lake', 'sea', 'ocean', 'mountain',
    'peninsula', 'strait', 'bay', 'gulf', 'cape', 'desert', 'valley',
    'continent', 'territory', 'colony', 'empire', 'kingdom', 'duchy',
    'republic', 'principality', 'prefecture', 'department', 'district',
    'municipality', 'borough', 'parish', 'commune', 'port', 'capital',
    'archipelago', 'plateau', 'plain', 'canal', 'channel', 'harbor',
    'harbour', 'volcano', 'hill', 'forest', 'palace', 'castle',
    'sovereign', 'nation', 'realm', 'metropolis', 'settlement',
    'historical', 'ancient', 'former', 'shire', 'constituency',
    'geographic', 'geographical', 'toponym', 'place', 'location',
    'administrative', 'subdivision', 'federal', 'autonomous',
    'governorate', 'voivodeship', 'oblast', 'krai', 'landkreis',
    'arrondissement', 'canton', 'comarca', 'regency', 'subdistrict',
    'creek', 'stream', 'tributary', 'waterfall', 'spring', 'oasis',
    'swamp', 'marsh', 'lagoon', 'reef', 'atoll', 'coast', 'shore',
    'range', 'pass', 'gorge', 'canyon', 'cliff', 'cave', 'glacier',
    'megalopolis', 'agglomeration', 'urban', 'rural', 'hamlet',
    'neighborhood', 'quarter', 'ward', 'precinct',
    'body of water', 'watercourse', 'landform', 'terrain',
}


def sparql_query(query, retries=3):
    """Execute SPARQL query against Wikidata endpoint."""
    params = urllib.parse.urlencode({
        'query': query,
        'format': 'json'
    })
    url = f"{SPARQL_ENDPOINT}?{params}"
    headers = {
        'User-Agent': 'EncyclopediaBritannicaKG/1.0 (research project)',
        'Accept': 'application/sparql-results+json'
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  SPARQL error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  SPARQL failed after {retries} attempts: {e}")
                return None


def batch_geonames_to_qid(geonames_ids):
    """Look up Wikidata QIDs for a list of GeoNames IDs via P1566."""
    results = {}
    batches = [geonames_ids[i:i+BATCH_SIZE] for i in range(0, len(geonames_ids), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches):
        values = " ".join(f'"{gid}"' for gid in batch)
        query = f"""
SELECT ?geonamesId ?item ?itemLabel ?sitelinks WHERE {{
  VALUES ?geonamesId {{ {values} }}
  ?item wdt:P1566 ?geonamesId .
  ?item wikibase:sitelinks ?sitelinks .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
        print(f"  GeoNames→QID batch {batch_num+1}/{len(batches)} ({len(batch)} IDs)...")
        data = sparql_query(query)
        if data and 'results' in data:
            for row in data['results']['bindings']:
                gid = row['geonamesId']['value']
                qid = row['item']['value'].split('/')[-1]
                label = row.get('itemLabel', {}).get('value', '')
                sitelinks = int(row.get('sitelinks', {}).get('value', 0))
                # Keep highest-sitelinks match if multiple
                if gid not in results or sitelinks > results[gid].get('sitelinks', 0):
                    results[gid] = {
                        'wikidata_qid': qid,
                        'wikidata_label': label,
                        'sitelinks': sitelinks
                    }

        time.sleep(1.5)  # Rate limiting

    return results


def batch_validate_qids(qids):
    """Check P31 (instance of) for a list of QIDs to validate they're geographic."""
    results = {}
    batches = [qids[i:i+BATCH_SIZE] for i in range(0, len(qids), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches):
        values = " ".join(f'wd:{qid}' for qid in batch)
        query = f"""
SELECT ?item ?itemLabel ?itemDescription ?instanceLabel ?sitelinks WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P31 ?instance . }}
  ?item wikibase:sitelinks ?sitelinks .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
        print(f"  Validate QIDs batch {batch_num+1}/{len(batches)} ({len(batch)} QIDs)...")
        data = sparql_query(query)
        if data and 'results' in data:
            for row in data['results']['bindings']:
                qid = row['item']['value'].split('/')[-1]
                if qid not in results:
                    results[qid] = {
                        'label': row.get('itemLabel', {}).get('value', ''),
                        'description': row.get('itemDescription', {}).get('value', ''),
                        'sitelinks': int(row.get('sitelinks', {}).get('value', 0)),
                        'instance_of': []
                    }
                inst = row.get('instanceLabel', {}).get('value', '')
                if inst and inst not in results[qid]['instance_of']:
                    results[qid]['instance_of'].append(inst)

        time.sleep(1.5)

    return results


def is_geographic(instance_of_labels, description=''):
    """Check if any P31 label or description suggests a geographic entity."""
    all_text = ' '.join(instance_of_labels).lower() + ' ' + description.lower()
    return any(kw in all_text for kw in GEO_KEYWORDS)


def main():
    # Load clusters
    print("Loading clusters...")
    clusters = []
    with open(CLUSTERS_FILE) as f:
        for line in f:
            clusters.append(json.loads(line))
    print(f"  {len(clusters)} total clusters")

    # Separate by match type
    geonames_clusters = [c for c in clusters if c.get('geonames_id')]
    wikidata_clusters = [c for c in clusters if c.get('wikidata_qid') and not c.get('geonames_id')]
    print(f"  {len(geonames_clusters)} GeoNames-matched")
    print(f"  {len(wikidata_clusters)} Wikidata-only")

    # Step 1: Get QIDs for GeoNames-matched clusters
    print("\n=== Step 1: GeoNames → Wikidata QID lookup ===")
    geonames_ids = list(set(str(c['geonames_id']) for c in geonames_clusters))
    print(f"  {len(geonames_ids)} unique GeoNames IDs to look up")
    geonames_qids = batch_geonames_to_qid(geonames_ids)
    print(f"  {len(geonames_qids)} QIDs found")

    # Apply QIDs to clusters
    geonames_enriched = 0
    for c in clusters:
        gid = str(c.get('geonames_id', ''))
        if gid in geonames_qids:
            info = geonames_qids[gid]
            c['wikidata_qid'] = info['wikidata_qid']
            c['wikidata_label'] = info['wikidata_label']
            c['wikidata_sitelinks'] = info['sitelinks']
            geonames_enriched += 1
    print(f"  Enriched {geonames_enriched} clusters with QIDs")

    # Step 2: Validate ALL QIDs (GeoNames-derived + Wikidata-only)
    print("\n=== Step 2: Validate all QIDs ===")
    all_qids = list(set(
        c['wikidata_qid'] for c in clusters
        if c.get('wikidata_qid') and c['wikidata_qid'] != 'NOT_A_PLACE'
    ))
    print(f"  {len(all_qids)} unique QIDs to validate")
    validation = batch_validate_qids(all_qids)
    print(f"  {len(validation)} QIDs validated")

    # Step 3: Flag non-geographic entities
    print("\n=== Step 3: Flag non-geographic entities ===")
    flagged = []
    for c in clusters:
        qid = c.get('wikidata_qid')
        if not qid or qid == 'NOT_A_PLACE':
            continue
        if qid in validation:
            v = validation[qid]
            c['wikidata_instance_of'] = v['instance_of']
            c['wikidata_description'] = v.get('description', '')
            if 'wikidata_sitelinks' not in c:
                c['wikidata_sitelinks'] = v['sitelinks']
            if not is_geographic(v['instance_of'], v.get('description', '')):
                c['qid_flagged'] = True
                flagged.append(c)

    print(f"  {len(flagged)} clusters flagged as potentially non-geographic")

    # Step 4: Write enriched output
    print(f"\n=== Step 4: Writing output ===")
    with open(OUTPUT_FILE, 'w') as f:
        for c in clusters:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(clusters)} clusters to {OUTPUT_FILE}")

    # Step 5: Write review file
    with open(REVIEW_FILE, 'w') as f:
        # Summary stats
        has_qid = sum(1 for c in clusters if c.get('wikidata_qid'))
        has_geonames = sum(1 for c in clusters if c.get('geonames_id'))
        f.write("=== TOPONYM QID ENRICHMENT REPORT ===\n\n")
        f.write(f"Total clusters: {len(clusters)}\n")
        f.write(f"With GeoNames ID: {has_geonames}\n")
        f.write(f"With Wikidata QID: {has_qid}\n")
        f.write(f"GeoNames→QID lookups successful: {geonames_enriched}\n")
        f.write(f"Flagged non-geographic: {len(flagged)}\n\n")

        # Flagged entities
        f.write("=== FLAGGED NON-GEOGRAPHIC ENTITIES ===\n")
        f.write("These QIDs may be wrong — their P31 doesn't match any geographic class.\n\n")
        flagged.sort(key=lambda x: -x.get('total_mentions', 0))
        for c in flagged:
            qid = c['wikidata_qid']
            v = validation.get(qid, {})
            f.write(f"  {c['cluster_id']:30s} {qid:12s} "
                    f"inst={v.get('instance_of', [])!r:60s} "
                    f"desc={v.get('description', '')[:50]:50s} "
                    f"mentions={c.get('total_mentions', 0)}\n")

        # Stats on sitelinks distribution
        f.write("\n\n=== LOW SITELINKS (potential wrong-entity matches) ===\n")
        f.write("Clusters with QIDs that have very few sitelinks (< 20) and 10+ mentions.\n\n")
        low_sitelinks = [
            c for c in clusters
            if c.get('wikidata_qid')
            and c.get('wikidata_sitelinks', 999) < 20
            and c.get('total_mentions', 0) >= 10
            and not c.get('qid_flagged')
        ]
        low_sitelinks.sort(key=lambda x: x.get('wikidata_sitelinks', 0))
        for c in low_sitelinks[:100]:
            qid = c['wikidata_qid']
            v = validation.get(qid, {})
            f.write(f"  {c['cluster_id']:30s} {qid:12s} "
                    f"sitelinks={c.get('wikidata_sitelinks', '?'):4} "
                    f"inst={v.get('instance_of', [])!r:60s} "
                    f"mentions={c.get('total_mentions', 0)}\n")

    print(f"  Review file: {REVIEW_FILE}")

    # Summary
    print("\n=== SUMMARY ===")
    total_with_qid = sum(1 for c in clusters if c.get('wikidata_qid'))
    total_10plus = sum(1 for c in clusters if c.get('total_mentions', 0) >= 10)
    qid_10plus = sum(1 for c in clusters if c.get('wikidata_qid') and c.get('total_mentions', 0) >= 10)
    print(f"  Clusters with QID: {total_with_qid}")
    print(f"  Clusters (10+ mentions) with QID: {qid_10plus} / {total_10plus}")
    print(f"  Flagged for review: {len(flagged)}")
    print(f"  Low sitelinks (10+ mentions): {len(low_sitelinks)}")


if __name__ == '__main__':
    main()
