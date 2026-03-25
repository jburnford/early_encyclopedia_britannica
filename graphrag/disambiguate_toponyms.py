#!/usr/bin/env python3
"""
Toponym disambiguation pipeline for Encyclopedia Britannica NER results.

Steps:
1. Extract all TOPONYM surface forms from NER output
2. Filter false positives (coordinates, compass directions, short garbage)
3. Normalize (case-fold, expand abbreviations, strip punctuation, collapse hyphens)
4. Cluster by normalized form
5. Match against GeoNames (Neo4j fulltext index)
6. Resolve ambiguous matches: primary + historically significant alternatives
7. Wikidata fallback for unmatched clusters
8. Report match rates by frequency tier
"""

import json
import re
import sys
import os
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import HTTPError, URLError

# --- Configuration ---

EDITIONS = [
    ("1st", 1771), ("2nd", 1778), ("3rd", 1797), ("4th", 1810),
    ("5th", 1815), ("6th", 1823), ("7th", 1842), ("8th", 1860),
]

REPO_DIR = Path(__file__).resolve().parent.parent
NER_DIR = REPO_DIR / "data" / "ner"
CONCEPT_INDEX_PATH = REPO_DIR / "graphrag" / "concept_index.json"

# Output paths
CLUSTERS_PATH = NER_DIR / "toponym_clusters.jsonl"
ENTITIES_PATH = NER_DIR / "toponym_entities.jsonl"
REPORT_PATH = NER_DIR / "toponym_match_report.txt"

# --- Significance thresholds for disambiguation ---

# Population threshold: places below this are dismissed as irrelevant colonial
# namesakes UNLESS they are admin capitals or have specific feature codes.
# Set conservatively — we want to keep Boston MA (675K), York/Toronto (2.9M),
# Quebec (531K), but dismiss England AR (2,765), Paris TX (25K tiny hamlet).
POP_THRESHOLD_SIGNIFICANT = 50_000

# Feature codes that indicate historical significance regardless of modern pop
# PPLC = national capital, PPLA = first-order admin capital (state/province),
# PPLA2 = second-order admin capital (county seat)
SIGNIFICANT_FEATURE_CODES = {"PPLC", "PPLA", "PPLA2"}

# --- False positive filters ---

FALSE_POSITIVE_EXACT = {
    # Compass / coordinate fragments
    "Lat.", "Long.", "N. Lat.", "W. Long.", "E. Long.", "S. Lat.",
    "N.W.", "N.E.", "S.W.", "S.E.", "N.", "S.", "E.", "W.",
    "Lat", "Long", "N. Lat", "W. Long", "E. Long", "S. Lat",
    # Known NER errors from previous analysis
    "Bleaching", "Ammonia", "Common", "Hat", "orange-peel",
}

FALSE_POSITIVE_PATTERNS = [
    re.compile(r"^\d"),           # starts with digit
    re.compile(r"^[A-Z]\.$"),     # single letter + period (N., S., etc.)
    re.compile(r"^[NSEW]\.\s"),   # compass prefix
]

# --- Abbreviation expansions ---

ABBREVIATIONS = {
    "lond": "London",
    "lond.": "London",
    "edinb": "Edinburgh",
    "edinb.": "Edinburgh",
    "edin": "Edinburgh",
    "edin.": "Edinburgh",
    "amst": "Amsterdam",
    "amst.": "Amsterdam",
    "oxon": "Oxford",
    "oxon.": "Oxford",
    "westm": "Westminster",
    "westm.": "Westminster",
    "cantab": "Cambridge",
    "cantab.": "Cambridge",
    "frankf": "Frankfurt",
    "frankf.": "Frankfurt",
    "lips": "Leipzig",
    "lips.": "Leipzig",
    "lugd. bat": "Leiden",
    "lugd. bat.": "Leiden",
    "lugd": "Lyon",
    "lugd.": "Lyon",
    "venet": "Venice",
    "venet.": "Venice",
    "genev": "Geneva",
    "genev.": "Geneva",
    "antw": "Antwerp",
    "antw.": "Antwerp",
    "basil": "Basel",
    "basil.": "Basel",
    "hanov": "Hanover",
    "hanov.": "Hanover",
    "berol": "Berlin",
    "berol.": "Berlin",
    "petropol": "St Petersburg",
    "petropol.": "St Petersburg",
    "paris.": "Paris",
    "lond. & edinb": "London",
    "st.": "St",  # normalize St. → St
    "mosc": "Moscow",
    "mosc.": "Moscow",
}


def is_false_positive(text: str) -> bool:
    """Check if a surface form is a known false positive."""
    if text in FALSE_POSITIVE_EXACT:
        return True
    if len(text) < 2:
        return True
    for pat in FALSE_POSITIVE_PATTERNS:
        if pat.search(text):
            return True
    return False


def normalize(text: str) -> str:
    """Normalize a toponym surface form to a canonical string."""
    t = text.strip()

    # Check abbreviation table (case-insensitive)
    t_lower = t.lower().rstrip(".")
    if t.lower() in ABBREVIATIONS:
        return ABBREVIATIONS[t.lower()]
    if t_lower in ABBREVIATIONS:
        return ABBREVIATIONS[t_lower]

    # Strip trailing period
    t = t.rstrip(".")

    # Normalize hyphens: replace with space (East-Indies → East Indies)
    t = t.replace("-", " ")

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Title case (but preserve "St" prefix)
    if t.isupper() or t.islower():
        t = t.title()

    # Normalize "St " prefix
    t = re.sub(r"^St\b\.?\s*", "St ", t)

    return t


def load_ner_toponyms():
    """Load all TOPONYM entities from NER files."""
    print("Loading NER toponym data...")
    form_by_edition = defaultdict(lambda: Counter())
    form_by_article = defaultdict(lambda: Counter())
    form_total = Counter()

    for ed, year in EDITIONS:
        path = NER_DIR / f"eb_{ed}_{year}.entities.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                for ent in rec["entities"]:
                    if ent["type"] == "TOPONYM":
                        text = ent["text"]
                        form_total[text] += 1
                        form_by_edition[text][year] += 1
                        form_by_article[text][rec["article_id"]] += 1

    print(f"  {len(form_total):,} unique surface forms, {sum(form_total.values()):,} total mentions")
    return form_total, form_by_edition, form_by_article


def build_clusters(form_total, form_by_edition, form_by_article):
    """Normalize and cluster surface forms."""
    print("Building clusters...")

    clusters = defaultdict(list)
    filtered_count = 0
    filtered_mentions = 0

    for form, count in form_total.items():
        if is_false_positive(form):
            filtered_count += 1
            filtered_mentions += count
            continue

        norm = normalize(form)
        if len(norm) < 2:
            filtered_count += 1
            filtered_mentions += count
            continue

        key = norm.lower()
        clusters[key].append((form, count))

    print(f"  Filtered {filtered_count:,} false-positive forms ({filtered_mentions:,} mentions)")
    print(f"  {len(clusters):,} clusters from {len(form_total) - filtered_count:,} valid forms")

    cluster_records = []
    for key, forms in clusters.items():
        forms.sort(key=lambda x: x[1], reverse=True)
        label = normalize(forms[0][0])
        variants = [f for f, _ in forms]
        total = sum(c for _, c in forms)

        by_edition = Counter()
        article_ids = set()
        for form, _ in forms:
            for year, cnt in form_by_edition[form].items():
                by_edition[year] += cnt
            article_ids.update(form_by_article[form].keys())

        cluster_records.append({
            "cluster_id": key,
            "label": label,
            "variants": variants,
            "total_mentions": total,
            "by_edition": dict(sorted(by_edition.items())),
            "article_count": len(article_ids),
            "edition_count": len(by_edition),
        })

    cluster_records.sort(key=lambda r: r["total_mentions"], reverse=True)
    for i, rec in enumerate(cluster_records):
        rec["frequency_rank"] = i + 1

    return cluster_records


def load_concept_index_places():
    """Load place-related headwords from the concept index as anchors."""
    print("Loading concept index for place anchors...")
    if not CONCEPT_INDEX_PATH.exists():
        print(f"  WARNING: {CONCEPT_INDEX_PATH} not found")
        return {}

    with open(CONCEPT_INDEX_PATH) as f:
        ci = json.load(f)

    anchors = {}
    for key, val in ci.items():
        label = val.get("label", key)
        anchors[label.lower()] = label

    print(f"  {len(anchors):,} concept anchors loaded")
    return anchors


def _is_significant_candidate(candidate):
    """Determine if a GeoNames candidate is historically significant enough to keep.

    We keep a candidate if:
    - It has population >= 50K (modern proxy for historical significance), OR
    - It is an admin capital (PPLC, PPLA, PPLA2), OR
    - It is a country (feature class A with PCLI/PCLD/etc.)
    """
    pop = candidate.get("population") or 0
    fc = candidate.get("feature_code") or ""
    fclass = candidate.get("feature_class") or ""

    if pop >= POP_THRESHOLD_SIGNIFICANT:
        return True
    if fc in SIGNIFICANT_FEATURE_CODES:
        return True
    if fclass == "A" and fc.startswith("PCL"):
        return True
    return False


def _make_candidate_dict(c):
    """Convert a Neo4j record to a candidate dict."""
    return {
        "geonames_id": c["geonameId"],
        "name": c["name"],
        "country": c["country"],
        "feature_class": c["featureClass"],
        "feature_code": c["featureCode"],
        "population": c["population"],
        "lat": c["lat"],
        "lon": c["lon"],
        "wikidata_qid": c.get("wikidataId"),
        "score": c.get("score"),
    }


def match_geonames(cluster_records, neo4j_uri, neo4j_password, min_mentions=10):
    """Match clusters against GeoNames via Neo4j fulltext index.

    Disambiguation strategy:
    - Pick highest-population exact-name match as primary
    - Keep any other candidates that are historically significant
      (pop >= 50K or admin capitals) as alternatives
    - Dismiss small towns/hamlets as colonial namesakes
    """
    from neo4j import GraphDatabase

    print(f"Connecting to Neo4j at {neo4j_uri}...")
    driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", neo4j_password))

    with driver.session() as session:
        result = session.run("MATCH (p:Place) RETURN count(p) as c")
        count = result.single()["c"]
        print(f"  {count:,} Place nodes available")

    to_match = [r for r in cluster_records if r["total_mentions"] >= min_mentions]
    skipped = [r for r in cluster_records if r["total_mentions"] < min_mentions]
    for r in skipped:
        r["match_type"] = "below_threshold"

    matched = 0
    ambiguous = 0
    unmatched = 0
    total = len(to_match)

    print(f"Matching {total:,} clusters (>= {min_mentions} mentions) against GeoNames...")

    query = """
    CALL db.index.fulltext.queryNodes('place_name_fulltext', $name)
    YIELD node, score
    WHERE score > 1.0
    RETURN node.geonameId AS geonameId,
           node.name AS name,
           node.countryCode AS country,
           node.featureClass AS featureClass,
           node.featureCode AS featureCode,
           node.population AS population,
           node.latitude AS lat,
           node.longitude AS lon,
           score
    ORDER BY score DESC, node.population DESC
    LIMIT 15
    """

    with driver.session() as session:
        for i, rec in enumerate(to_match):
            if (i + 1) % 1000 == 0:
                print(f"  {i+1:,}/{total:,} ({matched:,} matched, {ambiguous:,} ambiguous, {unmatched:,} unmatched)")

            label = rec["label"]

            try:
                result = session.run(query, name=label)
                candidates = list(result)
            except Exception as e:
                rec["match_type"] = "error"
                rec["match_error"] = str(e)
                unmatched += 1
                continue

            if not candidates:
                rec["match_type"] = "none"
                unmatched += 1
                continue

            # Filter to exact name matches
            exact = [c for c in candidates if c["name"] and c["name"].lower() == label.lower()]
            if not exact:
                # Fallback: high-score fuzzy
                exact = [candidates[0]] if candidates[0]["score"] > 3.0 else []

            if not exact:
                rec["match_type"] = "none"
                unmatched += 1
                continue

            # Convert to dicts
            exact_dicts = [_make_candidate_dict(c) for c in exact]

            # Pick primary: highest population among exact matches
            primary = max(exact_dicts, key=lambda c: c["population"] or 0)

            # Find significant alternatives (different country from primary, large enough)
            alternatives = []
            for c in exact_dicts:
                if c["geonames_id"] == primary["geonames_id"]:
                    continue
                if c["country"] == primary["country"]:
                    continue  # same country, probably admin subdivision duplicate
                if _is_significant_candidate(c):
                    alternatives.append(c)

            # Set primary match fields
            rec["match_type"] = "matched"
            rec["geonames_id"] = primary["geonames_id"]
            rec["geonames_name"] = primary["name"]
            rec["country"] = primary["country"]
            rec["feature_class"] = primary["feature_class"]
            rec["feature_code"] = primary["feature_code"]
            rec["population"] = primary["population"]
            rec["lat"] = primary["lat"]
            rec["lon"] = primary["lon"]

            if alternatives:
                rec["alternatives"] = alternatives
                ambiguous += 1
            else:
                matched += 1

    driver.close()

    print(f"\nGeoNames matching complete:")
    print(f"  Matched (unambiguous): {matched:,} ({100*matched/total:.1f}%)")
    print(f"  Matched (with alternatives): {ambiguous:,} ({100*ambiguous/total:.1f}%)")
    print(f"  Unmatched: {unmatched:,} ({100*unmatched/total:.1f}%)")

    return cluster_records


def match_wikidata(cluster_records, min_mentions=10):
    """Wikidata fallback for clusters not matched by GeoNames.

    Uses the Wikidata search API to find geographic entities.
    Respects rate limits (polite delay between requests).
    """
    unmatched = [r for r in cluster_records
                 if r.get("match_type") == "none"
                 and r["total_mentions"] >= min_mentions]

    if not unmatched:
        print("No unmatched clusters to search Wikidata for.")
        return cluster_records

    print(f"Searching Wikidata for {len(unmatched):,} unmatched clusters...")

    wd_matched = 0
    wd_failed = 0

    for i, rec in enumerate(unmatched):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unmatched)} ({wd_matched} matched)")

        label = rec["label"]
        try:
            # Wikidata search API
            url = (
                "https://www.wikidata.org/w/api.php?"
                "action=wbsearchentities&format=json&language=en&type=item"
                f"&search={quote(label)}&limit=5"
            )
            req = Request(url, headers={"User-Agent": "EncyclopediaBritannicaKG/1.0"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            results = data.get("search", [])
            if not results:
                wd_failed += 1
                time.sleep(0.5)
                continue

            # Look for geographic entities in the results
            # We take the first result and fetch its details to check if it's a place
            best = results[0]
            qid = best["id"]
            wd_label = best.get("label", "")
            description = best.get("description", "")

            # Quick heuristic: if description mentions geographic terms, it's likely a place
            geo_terms = [
                "island", "region", "sea", "ocean", "river", "mountain", "peninsula",
                "gulf", "strait", "lake", "bay", "cape", "kingdom", "empire",
                "province", "county", "state", "territory", "colony", "city",
                "town", "village", "country", "republic", "duchy", "department",
                "continent", "historical", "ancient", "former",
            ]
            desc_lower = description.lower()
            is_geo = any(term in desc_lower for term in geo_terms)

            if is_geo or wd_label.lower() == label.lower():
                rec["match_type"] = "wikidata"
                rec["wikidata_qid"] = qid
                rec["wikidata_label"] = wd_label
                rec["wikidata_description"] = description
                wd_matched += 1
            else:
                wd_failed += 1

        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            wd_failed += 1

        # Rate limit: ~2 requests/second
        time.sleep(0.5)

    print(f"\nWikidata matching complete:")
    print(f"  Matched: {wd_matched:,}")
    print(f"  Not found: {wd_failed:,}")

    return cluster_records


def apply_concept_anchors(cluster_records, concept_anchors):
    """Use concept index headwords to boost confidence for matched clusters."""
    boosted = 0
    for rec in cluster_records:
        key = rec["label"].lower()
        if key in concept_anchors:
            rec["is_concept_headword"] = True
            rec["concept_label"] = concept_anchors[key]
            boosted += 1
        else:
            rec["is_concept_headword"] = False

    print(f"  {boosted:,} clusters match concept index headwords")
    return cluster_records


def generate_report(cluster_records, report_path, min_mentions=10):
    """Generate match rate analysis by frequency tier."""
    lines = []
    lines.append("=" * 80)
    lines.append("TOPONYM DISAMBIGUATION REPORT")
    lines.append("=" * 80)

    total_clusters = len(cluster_records)
    total_mentions = sum(r["total_mentions"] for r in cluster_records)
    matched_types = {"matched", "wikidata"}

    lines.append(f"\nTotal clusters: {total_clusters:,}")
    lines.append(f"Total mentions: {total_mentions:,}")
    lines.append(f"Min mentions for matching: {min_mentions}")

    # Count by match type
    by_type = Counter(r.get("match_type", "none") for r in cluster_records)
    lines.append(f"\nMatch type breakdown:")
    for mt, cnt in by_type.most_common():
        lines.append(f"  {mt}: {cnt:,}")

    # Match rate by frequency tier
    tiers = [
        ("100+", lambda r: r["total_mentions"] >= 100),
        ("50-99", lambda r: 50 <= r["total_mentions"] < 100),
        ("10-49", lambda r: 10 <= r["total_mentions"] < 50),
        ("5-9", lambda r: 5 <= r["total_mentions"] < 10),
        ("2-4", lambda r: 2 <= r["total_mentions"] < 5),
        ("1", lambda r: r["total_mentions"] == 1),
    ]

    lines.append(f"\n{'Tier':<10} {'Clusters':>10} {'GeoNames':>10} {'Wikidata':>10} {'w/Alts':>10} {'None':>10} {'Grounded%':>10} {'Mentions':>12}")
    lines.append("-" * 95)

    for tier_name, pred in tiers:
        subset = [r for r in cluster_records if pred(r)]
        n = len(subset)
        geonames = sum(1 for r in subset if r.get("match_type") == "matched" and not r.get("alternatives"))
        wikidata = sum(1 for r in subset if r.get("match_type") == "wikidata")
        with_alts = sum(1 for r in subset if r.get("match_type") == "matched" and r.get("alternatives"))
        none_ct = n - geonames - wikidata - with_alts
        grounded = geonames + wikidata + with_alts
        mentions = sum(r["total_mentions"] for r in subset)
        grounded_pct = 100 * grounded / n if n else 0
        lines.append(f"{tier_name:<10} {n:>10,} {geonames:>10,} {wikidata:>10,} {with_alts:>10,} {none_ct:>10,} {grounded_pct:>9.1f}% {mentions:>12,}")

    # Grounded mentions total
    grounded_recs = [r for r in cluster_records if r.get("match_type") in matched_types]
    grounded_mentions = sum(r["total_mentions"] for r in grounded_recs)
    lines.append(f"\nTotal grounded mentions: {grounded_mentions:,} / {total_mentions:,} ({100*grounded_mentions/total_mentions:.1f}%)")

    # Top 30 with significant alternatives (the London ON / Boston MA cases)
    alt_recs = [r for r in cluster_records if r.get("alternatives")]
    alt_recs.sort(key=lambda r: r["total_mentions"], reverse=True)

    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP 30 TOPONYMS WITH SIGNIFICANT ALTERNATIVES ({len(alt_recs)} total)")
    lines.append(f"{'=' * 80}")
    for rec in alt_recs[:30]:
        primary_pop = rec.get("population") or 0
        lines.append(f"\n  {rec['label']} ({rec['total_mentions']:,} mentions, {rec['edition_count']} editions)")
        lines.append(f"    PRIMARY: {rec['country']} | pop={primary_pop:,} | {rec.get('feature_class','')}.{rec.get('feature_code','')} | geonames={rec.get('geonames_id')}")
        for alt in rec.get("alternatives", []):
            alt_pop = alt.get("population") or 0
            lines.append(f"    ALT:     {alt['country']} | pop={alt_pop:,} | {alt.get('feature_class','')}.{alt.get('feature_code','')} | geonames={alt.get('geonames_id')}")

    # Top 30 Wikidata matches
    wd_recs = [r for r in cluster_records if r.get("match_type") == "wikidata"]
    wd_recs.sort(key=lambda r: r["total_mentions"], reverse=True)

    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP 30 WIKIDATA MATCHES ({len(wd_recs)} total)")
    lines.append(f"{'=' * 80}")
    for rec in wd_recs[:30]:
        concept_flag = " [CONCEPT]" if rec.get("is_concept_headword") else ""
        lines.append(f"  {rec['label']}: {rec['total_mentions']:,} mentions | {rec.get('wikidata_qid')} | {rec.get('wikidata_description', '')}{concept_flag}")

    # Top 30 still unmatched
    unmatched_recs = [r for r in cluster_records
                      if r.get("match_type") == "none"
                      and r["total_mentions"] >= min_mentions]
    unmatched_recs.sort(key=lambda r: r["total_mentions"], reverse=True)

    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP 30 STILL UNMATCHED (>= {min_mentions} mentions, {len(unmatched_recs)} total)")
    lines.append(f"{'=' * 80}")
    for rec in unmatched_recs[:30]:
        concept_flag = " [CONCEPT]" if rec.get("is_concept_headword") else ""
        lines.append(f"  {rec['label']}: {rec['total_mentions']:,} mentions, {rec['edition_count']} editions{concept_flag}")
        if len(rec["variants"]) > 1:
            lines.append(f"    variants: {rec['variants'][:5]}")

    # Temporal analysis
    lines.append(f"\n{'=' * 80}")
    lines.append(f"TEMPORAL PATTERNS (grounded entities, >= {min_mentions} mentions)")
    lines.append(f"{'=' * 80}")

    grounded = [r for r in cluster_records
                if r.get("match_type") in matched_types
                and r["total_mentions"] >= min_mentions]

    late_arrivals = [r for r in grounded
                     if 1771 not in r["by_edition"] and 1778 not in r["by_edition"]
                     and (1842 in r["by_edition"] or 1860 in r["by_edition"])]
    late_arrivals.sort(key=lambda r: r["total_mentions"], reverse=True)
    lines.append(f"\nLate arrivals (not in 1771/1778, appear in 1842/1860): {len(late_arrivals)}")
    for rec in late_arrivals[:20]:
        eds = sorted(rec["by_edition"].keys())
        country = rec.get("country", "?")
        lines.append(f"  {rec['label']} ({country}): {rec['total_mentions']} mentions, editions {eds}")

    disappearances = [r for r in grounded
                      if (1771 in r["by_edition"] or 1778 in r["by_edition"])
                      and 1842 not in r["by_edition"] and 1860 not in r["by_edition"]]
    disappearances.sort(key=lambda r: r["total_mentions"], reverse=True)
    lines.append(f"\nDisappearances (in 1771/1778, not in 1842/1860): {len(disappearances)}")
    for rec in disappearances[:20]:
        eds = sorted(rec["by_edition"].keys())
        country = rec.get("country", "?")
        lines.append(f"  {rec['label']} ({country}): {rec['total_mentions']} mentions, editions {eds}")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")
    print(report)


def save_clusters(cluster_records, path):
    """Save cluster records as JSONL."""
    with open(path, "w") as f:
        for rec in cluster_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(cluster_records):,} clusters to {path}")


def save_csv(cluster_records, path):
    """Save cluster records as CSV for easy review."""
    import csv

    editions = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "cluster_id", "label", "total_mentions", "article_count",
            "edition_count", "is_concept_headword", "match_type",
            "country", "geonames_id", "population", "lat", "lon",
            "wikidata_qid", "wikidata_description",
            "alternatives_summary", "variants",
            *[str(y) for y in editions],
        ])
        for r in cluster_records:
            # Build alternatives summary
            alts = r.get("alternatives", [])
            alt_str = "; ".join(
                f"{a['country']}(pop={a.get('population') or 0:,})"
                for a in alts
            ) if alts else ""

            w.writerow([
                r["frequency_rank"],
                r["cluster_id"],
                r["label"],
                r["total_mentions"],
                r["article_count"],
                r["edition_count"],
                r.get("is_concept_headword", False),
                r.get("match_type", ""),
                r.get("country", ""),
                r.get("geonames_id", ""),
                r.get("population", ""),
                r.get("lat", ""),
                r.get("lon", ""),
                r.get("wikidata_qid", ""),
                r.get("wikidata_description", ""),
                alt_str,
                "; ".join(r["variants"][:10]),
                *[r["by_edition"].get(str(y), r["by_edition"].get(y, 0)) for y in editions],
            ])

    print(f"Saved CSV to {path}")


def main():
    parser = argparse.ArgumentParser(description="Toponym disambiguation pipeline")
    parser.add_argument("--neo4j-uri", default="bolt://206.12.90.118:7687",
                        help="Neo4j connection URI")
    parser.add_argument("--neo4j-password", default=None,
                        help="Neo4j password (or set NEO4J_PASSWORD env var)")
    parser.add_argument("--skip-geonames", action="store_true",
                        help="Skip GeoNames matching (just normalize and cluster)")
    parser.add_argument("--skip-wikidata", action="store_true",
                        help="Skip Wikidata fallback matching")
    parser.add_argument("--min-mentions", type=int, default=10,
                        help="Only match clusters with this many mentions (default: 10)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    neo4j_password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD")

    # Try reading from .env file if not set
    if not neo4j_password:
        env_path = Path.home() / "textasdatacolonialofficelist" / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("NEO4J_PASSWORD="):
                        neo4j_password = line.strip().split("=", 1)[1]
                        break

    if args.output_dir:
        global CLUSTERS_PATH, ENTITIES_PATH, REPORT_PATH
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        CLUSTERS_PATH = out / "toponym_clusters.jsonl"
        ENTITIES_PATH = out / "toponym_entities.jsonl"
        REPORT_PATH = out / "toponym_match_report.txt"

    # Step 1: Load NER data
    form_total, form_by_edition, form_by_article = load_ner_toponyms()

    # Step 2: Normalize and cluster
    cluster_records = build_clusters(form_total, form_by_edition, form_by_article)

    # Step 3: Load concept anchors
    concept_anchors = load_concept_index_places()
    cluster_records = apply_concept_anchors(cluster_records, concept_anchors)

    # Step 4: Match against GeoNames
    if not args.skip_geonames:
        if not neo4j_password:
            print("ERROR: Neo4j password required. Set NEO4J_PASSWORD env var or use --neo4j-password")
            sys.exit(1)
        cluster_records = match_geonames(cluster_records, args.neo4j_uri, neo4j_password, args.min_mentions)

    # Step 5: Wikidata fallback for unmatched
    if not args.skip_wikidata and not args.skip_geonames:
        cluster_records = match_wikidata(cluster_records, args.min_mentions)

    # Step 6: Save results
    save_clusters(cluster_records, CLUSTERS_PATH)

    # Step 7: Save CSV
    csv_path = CLUSTERS_PATH.with_suffix(".csv")
    save_csv(cluster_records, csv_path)

    # Step 8: Generate report
    if not args.skip_geonames:
        generate_report(cluster_records, REPORT_PATH, args.min_mentions)

    print("\nDone!")


if __name__ == "__main__":
    main()
