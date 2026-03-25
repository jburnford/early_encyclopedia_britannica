#!/usr/bin/env python3
"""
Systematic validation of Wikidata QIDs for toponym clusters.

Phase 1: Refresh sitelinks + P31 for ALL QIDs, flag suspicious entries
Phase 2: Auto-fix flagged entries via SPARQL label search
Phase 3: Re-validate fixes

Usage:
    python3 graphrag/validate_toponym_qids.py [--phase {1,2,3,all}] [--dry-run] [--min-mentions 10]
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
BATCH_SIZE = 150

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"
DEFAULT_INPUT = NER_DIR / "toponym_clusters_enriched.jsonl"
DEFAULT_REPORT = NER_DIR / "toponym_validation_report.txt"

# Keywords that indicate a geographic P31 value
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

# Description terms that indicate geographic entities (from wikidata_requery.py)
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
    'battle', 'siege',
}

NOT_GEO_PATTERNS = [
    'genus of', 'genus in', 'family name', 'given name', 'surname',
    'video game', 'tv series', 'film', 'novel', 'song', 'album',
    'protein', 'enzyme', 'chemical', 'species of', 'breed of',
    'taxon', 'taxonomic', 'disambiguation', 'duplicat',
    'language', 'dialect', 'ethnic group', 'human population',
    'mytholog', 'legendary', 'painting', 'ship', 'railway station',
    'lunar crater', 'moon of', 'asteroid',
]


def sparql_query(query, retries=3):
    """Execute SPARQL query against Wikidata endpoint with retries."""
    params = urllib.parse.urlencode({'query': query, 'format': 'json'})
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
                print(f"    SPARQL error: {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    SPARQL failed after {retries} attempts: {e}")
                return None


def is_geographic(instance_of_labels, description=''):
    """Check if P31 labels or description suggest a geographic entity."""
    all_text = ' '.join(instance_of_labels).lower() + ' ' + description.lower()
    return any(kw in all_text for kw in GEO_KEYWORDS)


def is_non_geographic(instance_of_labels, description=''):
    """Check if P31/description contains known non-geographic patterns."""
    all_text = ' '.join(instance_of_labels).lower() + ' ' + description.lower()
    return any(pat in all_text for pat in NOT_GEO_PATTERNS)


def pick_best_geo_result(results):
    """From SPARQL results, pick the most likely geographic entity."""
    for r in results:
        desc = (r.get('itemDescription', {}).get('value', '') or '').lower()
        desc_words = set(re.findall(r'\w+', desc))

        if any(pat in desc for pat in NOT_GEO_PATTERNS):
            continue

        if desc_words & GEO_DESC_TERMS:
            qid = r['item']['value'].split('/')[-1]
            label = r.get('itemLabel', {}).get('value', '')
            sitelinks = int(r.get('sitelinks', {}).get('value', 0))
            return qid, label, desc, sitelinks

        sitelinks = int(r.get('sitelinks', {}).get('value', 0))
        if sitelinks >= 10 and not any(pat in desc for pat in NOT_GEO_PATTERNS):
            qid = r['item']['value'].split('/')[-1]
            label = r.get('itemLabel', {}).get('value', '')
            return qid, label, desc, sitelinks

    return None, None, None, 0


# ── Phase 1: Validate ALL QIDs ──────────────────────────────────────────────

def batch_fetch_metadata(qids):
    """Fetch sitelinks, P31, coordinates for a list of QIDs."""
    results = {}
    batches = [qids[i:i+BATCH_SIZE] for i in range(0, len(qids), BATCH_SIZE)]

    for batch_num, batch in enumerate(batches):
        values = " ".join(f'wd:{qid}' for qid in batch)
        query = f"""
SELECT ?item ?itemLabel ?itemDescription ?instanceLabel ?sitelinks
       (BOUND(?coord) AS ?hasCoord) WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P31 ?instance . }}
  OPTIONAL {{ ?item wdt:P625 ?coord . }}
  ?item wikibase:sitelinks ?sitelinks .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
"""
        print(f"  Phase 1 batch {batch_num+1}/{len(batches)} ({len(batch)} QIDs)...")
        data = sparql_query(query)
        if data and 'results' in data:
            for row in data['results']['bindings']:
                qid = row['item']['value'].split('/')[-1]
                if qid not in results:
                    results[qid] = {
                        'label': row.get('itemLabel', {}).get('value', ''),
                        'description': row.get('itemDescription', {}).get('value', ''),
                        'sitelinks': int(row.get('sitelinks', {}).get('value', 0)),
                        'instance_of': [],
                        'has_coordinates': False,
                    }
                inst = row.get('instanceLabel', {}).get('value', '')
                if inst and inst not in results[qid]['instance_of']:
                    results[qid]['instance_of'].append(inst)
                if row.get('hasCoord', {}).get('value', '') == 'true':
                    results[qid]['has_coordinates'] = True

        time.sleep(1.5)

    return results


def classify_flags(cluster, qid_info, min_mentions):
    """Determine validation flags for a cluster based on its QID metadata."""
    flags = []
    mentions = cluster.get('total_mentions', 0)
    if mentions < min_mentions:
        return flags

    sitelinks = qid_info.get('sitelinks', 0)
    instance_of = qid_info.get('instance_of', [])
    description = qid_info.get('description', '')

    # Sitelinks-based flags
    if sitelinks < 5 and mentions >= 50:
        flags.append('very_low_sitelinks')
    elif sitelinks < 20 and mentions >= 20:
        flags.append('low_sitelinks')

    # P31-based flags
    geo = is_geographic(instance_of, description)
    non_geo = is_non_geographic(instance_of, description)

    if not instance_of:
        flags.append('no_p31')
    elif non_geo and not geo:
        # Clearly non-geographic (person, species, etc.) with no geographic signal
        flags.append('non_geographic')
    elif not geo and not non_geo:
        # Unknown type — not clearly geographic or non-geographic
        flags.append('non_geographic')

    return flags


def run_phase1(clusters, min_mentions, sitelinks_threshold):
    """Phase 1: Refresh and validate all QIDs."""
    print("\n=== PHASE 1: Validate ALL QIDs ===")

    # Collect unique QIDs
    qid_to_clusters = {}
    for c in clusters:
        qid = c.get('wikidata_qid')
        if not qid or c.get('not_a_place'):
            continue
        if c.get('total_mentions', 0) < min_mentions:
            continue
        if qid not in qid_to_clusters:
            qid_to_clusters[qid] = []
        qid_to_clusters[qid].append(c)

    unique_qids = list(qid_to_clusters.keys())
    print(f"  {len(unique_qids)} unique QIDs to validate")

    # Batch fetch metadata
    metadata = batch_fetch_metadata(unique_qids)
    print(f"  {len(metadata)} QIDs fetched")

    # Apply metadata and classify
    flagged = []
    for qid, info in metadata.items():
        for c in qid_to_clusters.get(qid, []):
            # Update with fresh data
            c['wikidata_sitelinks'] = info['sitelinks']
            c['wikidata_instance_of'] = info['instance_of']
            c['wikidata_description'] = info.get('description', '')

            # Classify
            flags = classify_flags(c, info, min_mentions)
            if flags:
                c['validation_flags'] = flags
                flagged.append(c)
            else:
                c.pop('validation_flags', None)
                c.pop('qid_flagged', None)

    flagged.sort(key=lambda x: -x.get('total_mentions', 0))
    print(f"  {len(flagged)} clusters flagged")

    # Breakdown
    flag_counts = Counter()
    for c in flagged:
        for f in c.get('validation_flags', []):
            flag_counts[f] += 1
    for flag, count in flag_counts.most_common():
        print(f"    {flag}: {count}")

    return flagged, metadata


# ── Phase 2: Auto-fix flagged entries ────────────────────────────────────────

def sparql_search_label(label):
    """Search Wikidata by rdfs:label, ordered by sitelinks."""
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
    data = sparql_query(query)
    if data and 'results' in data:
        return data['results']['bindings']
    return None


def sparql_search_alias(label):
    """Fallback: search by skos:altLabel."""
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
    data = sparql_query(query)
    if data and 'results' in data:
        return data['results']['bindings']
    return None


def run_phase2(clusters, flagged, dry_run=False):
    """Phase 2: Try to auto-fix flagged entries."""
    print(f"\n=== PHASE 2: Auto-fix {len(flagged)} flagged entries ===")

    fixed = 0
    not_fixed = 0
    errors = 0
    changes = []

    for i, c in enumerate(flagged):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(flagged)} ({fixed} fixed, {not_fixed} kept, {errors} errors)")

        label = c['label']
        old_qid = c.get('wikidata_qid', '')
        old_sitelinks = c.get('wikidata_sitelinks', 0)

        # Search by label
        results = sparql_search_label(label)
        if results is None:
            errors += 1
            time.sleep(2)
            continue

        new_qid, new_label, new_desc, new_sitelinks = pick_best_geo_result(results)

        # Fallback to alias
        if new_qid is None:
            time.sleep(0.5)
            results = sparql_search_alias(label)
            if results is None:
                errors += 1
                time.sleep(2)
                continue
            new_qid, new_label, new_desc, new_sitelinks = pick_best_geo_result(results)

        # Decide whether to replace
        # If current QID is flagged non-geographic/no_p31, accept any geographic match
        # Otherwise require strictly higher sitelinks
        current_flags = c.get('validation_flags', [])
        is_current_bad = any(f in current_flags for f in ('non_geographic', 'no_p31'))
        should_replace = (
            new_qid and new_qid != old_qid and (
                new_sitelinks > old_sitelinks or  # better match by sitelinks
                is_current_bad  # current QID is non-geographic, any geo match is better
            )
        )
        if should_replace:
            change = {
                'cluster_id': c['cluster_id'],
                'label': label,
                'mentions': c.get('total_mentions', 0),
                'old_qid': old_qid,
                'old_sitelinks': old_sitelinks,
                'new_qid': new_qid,
                'new_label': new_label,
                'new_description': new_desc,
                'new_sitelinks': new_sitelinks,
                'flags': c.get('validation_flags', []),
            }
            changes.append(change)

            if not dry_run:
                c['wikidata_qid'] = new_qid
                c['wikidata_label'] = new_label
                c['wikidata_description'] = new_desc
                c['wikidata_sitelinks'] = new_sitelinks
                c['qid_source'] = 'validation_fix'
                c.pop('validation_flags', None)
                c.pop('qid_flagged', None)

            fixed += 1
        else:
            not_fixed += 1

        time.sleep(1.0)

    print(f"\n  Phase 2 complete:")
    print(f"    Fixed: {fixed}")
    print(f"    Not fixed: {not_fixed}")
    print(f"    Errors: {errors}")

    return changes


# ── Phase 3: Re-validate fixes ──────────────────────────────────────────────

def run_phase3(clusters, changes, min_mentions):
    """Phase 3: Re-validate the QIDs that were changed in Phase 2."""
    if not changes:
        print("\n=== PHASE 3: No changes to re-validate ===")
        return []

    changed_qids = list(set(ch['new_qid'] for ch in changes))
    print(f"\n=== PHASE 3: Re-validate {len(changed_qids)} changed QIDs ===")

    metadata = batch_fetch_metadata(changed_qids)

    still_flagged = []
    for c in clusters:
        qid = c.get('wikidata_qid')
        if qid not in metadata:
            continue
        if c.get('total_mentions', 0) < min_mentions:
            continue

        info = metadata[qid]
        c['wikidata_sitelinks'] = info['sitelinks']
        c['wikidata_instance_of'] = info['instance_of']

        flags = classify_flags(c, info, min_mentions)
        if flags:
            c['validation_flags'] = flags
            still_flagged.append(c)

    print(f"  {len(still_flagged)} still flagged after fixes")
    return still_flagged


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(clusters, flagged_phase1, changes, still_flagged, report_path, min_mentions):
    """Generate human-readable validation report."""
    with open(report_path, 'w') as f:
        # Summary
        total_10 = sum(1 for c in clusters if c.get('total_mentions', 0) >= min_mentions)
        with_qid = sum(1 for c in clusters if c.get('wikidata_qid') and c.get('total_mentions', 0) >= min_mentions and not c.get('not_a_place'))
        not_place = sum(1 for c in clusters if c.get('not_a_place') and c.get('total_mentions', 0) >= min_mentions)

        f.write("=== TOPONYM QID VALIDATION REPORT ===\n\n")
        f.write(f"Clusters (>={min_mentions} mentions): {total_10}\n")
        f.write(f"With Wikidata QID:    {with_qid} ({with_qid/total_10*100:.1f}%)\n")
        f.write(f"Marked NOT_A_PLACE:   {not_place}\n")
        f.write(f"Unmatched:            {total_10 - with_qid - not_place}\n\n")

        f.write(f"Phase 1 flagged:      {len(flagged_phase1)}\n")
        f.write(f"Phase 2 auto-fixed:   {len(changes)}\n")
        f.write(f"Phase 3 still bad:    {len(still_flagged)}\n\n")

        # Sitelinks distribution
        sitelinks_vals = []
        for c in clusters:
            if c.get('wikidata_qid') and c.get('total_mentions', 0) >= min_mentions and not c.get('not_a_place'):
                sitelinks_vals.append(c.get('wikidata_sitelinks', 0))

        if sitelinks_vals:
            f.write("=== SITELINKS DISTRIBUTION ===\n")
            buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]
            for lo, hi in buckets:
                count = sum(1 for s in sitelinks_vals if lo <= s < hi)
                f.write(f"  {lo:4d}-{hi:4d}: {count:5d}\n")
            count = sum(1 for s in sitelinks_vals if s >= 500)
            f.write(f"   500+: {count:5d}\n\n")

        # Phase 2 changes
        if changes:
            f.write("=== PHASE 2 CORRECTIONS ===\n\n")
            for ch in sorted(changes, key=lambda x: -x['mentions']):
                f.write(f"  {ch['cluster_id']:30s} mentions={ch['mentions']:5d}  "
                        f"{ch['old_qid']:12s} (sl={ch['old_sitelinks']:4d}) → "
                        f"{ch['new_qid']:12s} (sl={ch['new_sitelinks']:4d}) "
                        f"{ch['new_label']}\n")
                f.write(f"    flags: {ch['flags']}  desc: {ch.get('new_description','')[:80]}\n")
            f.write(f"\n  Total corrections: {len(changes)}\n\n")

        # Still flagged after all phases
        remaining = [c for c in clusters
                     if c.get('validation_flags') and c.get('total_mentions', 0) >= min_mentions]
        remaining.sort(key=lambda x: -x.get('total_mentions', 0))
        if remaining:
            f.write("=== REMAINING FLAGGED (manual review needed) ===\n\n")
            for c in remaining:
                f.write(f"  {c['cluster_id']:30s} {c.get('wikidata_qid',''):12s} "
                        f"sl={c.get('wikidata_sitelinks', 0):4d}  "
                        f"flags={c.get('validation_flags', [])}  "
                        f"inst={c.get('wikidata_instance_of', [])[:3]}  "
                        f"mentions={c.get('total_mentions', 0)}\n")
            f.write(f"\n  Total remaining: {len(remaining)}\n")

        # Top-10 verification
        f.write("\n=== TOP-10 MOST MENTIONED (sanity check) ===\n\n")
        top = sorted(
            [c for c in clusters if c.get('wikidata_qid') and c.get('total_mentions', 0) >= min_mentions],
            key=lambda x: -x['total_mentions']
        )[:10]
        for c in top:
            f.write(f"  {c['cluster_id']:25s} {c.get('wikidata_qid',''):12s} "
                    f"sl={c.get('wikidata_sitelinks', 0):4d}  "
                    f"{c.get('wikidata_label',''):30s}  "
                    f"mentions={c['total_mentions']}\n")

    print(f"  Report written to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Validate toponym Wikidata QIDs')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Input enriched JSONL')
    parser.add_argument('--output', type=Path, default=None, help='Output JSONL (default: overwrite input)')
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT, help='Validation report path')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='all', help='Phase to run')
    parser.add_argument('--min-mentions', type=int, default=10, help='Minimum mentions threshold')
    parser.add_argument('--sitelinks-threshold', type=int, default=20, help='Flag QIDs below this sitelinks count')
    parser.add_argument('--dry-run', action='store_true', help='Report only, do not modify data')
    args = parser.parse_args()

    output_path = args.output or args.input

    # Load
    print(f"Loading {args.input}...")
    clusters = []
    with open(args.input) as f:
        for line in f:
            clusters.append(json.loads(line))
    print(f"  {len(clusters)} total clusters")

    run_phases = args.phase

    flagged_phase1 = []
    changes = []
    still_flagged = []

    # Phase 1
    if run_phases in ('1', 'all'):
        flagged_phase1, metadata = run_phase1(clusters, args.min_mentions, args.sitelinks_threshold)

    # Phase 2
    if run_phases in ('2', 'all'):
        if not flagged_phase1:
            # If phase 1 wasn't run, collect flagged from stored flags
            flagged_phase1 = [c for c in clusters if c.get('validation_flags') and c.get('total_mentions', 0) >= args.min_mentions]
            flagged_phase1.sort(key=lambda x: -x.get('total_mentions', 0))
        changes = run_phase2(clusters, flagged_phase1, dry_run=args.dry_run)

    # Phase 3
    if run_phases in ('3', 'all') and changes and not args.dry_run:
        still_flagged = run_phase3(clusters, changes, args.min_mentions)

    # Save
    if not args.dry_run:
        print(f"\nSaving to {output_path}...")
        with open(output_path, 'w') as f:
            for c in clusters:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f"  Wrote {len(clusters)} clusters")

    # Report
    generate_report(clusters, flagged_phase1, changes, still_flagged, args.report, args.min_mentions)

    # Final stats
    total_10 = sum(1 for c in clusters if c.get('total_mentions', 0) >= args.min_mentions)
    with_qid = sum(1 for c in clusters if c.get('wikidata_qid') and c.get('total_mentions', 0) >= args.min_mentions and not c.get('not_a_place'))
    not_place = sum(1 for c in clusters if c.get('not_a_place') and c.get('total_mentions', 0) >= args.min_mentions)
    still_bad = sum(1 for c in clusters if c.get('validation_flags') and c.get('total_mentions', 0) >= args.min_mentions)

    print(f"\n=== FINAL STATUS (>={args.min_mentions} mentions) ===")
    print(f"  Total:           {total_10}")
    print(f"  With QID:        {with_qid} ({with_qid/total_10*100:.1f}%)")
    print(f"  NOT_A_PLACE:     {not_place}")
    print(f"  Unmatched:       {total_10 - with_qid - not_place}")
    print(f"  Still flagged:   {still_bad}")
    print(f"  Phase 2 fixes:   {len(changes)}")


if __name__ == '__main__':
    main()
