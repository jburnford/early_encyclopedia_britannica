#!/usr/bin/env python3
"""
Post-processing cleanup for person clusters.

Fixes:
1. Merge bare surnames into full-name clusters where co-occurrence confirms identity
2. Remove non-person Wikidata matches (places, concepts, etc.)
3. Spelling variant merges (known aliases)
4. Flag ambiguous clusters for manual review
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"
INPUT_PATH = NER_DIR / "person_clusters.jsonl"
OUTPUT_PATH = NER_DIR / "person_clusters_clean.jsonl"
CSV_PATH = NER_DIR / "person_clusters_clean.csv"

# --- Spelling variant merges ---
# target_cluster_id: [source_cluster_ids_to_merge_into_it]
SPELLING_MERGES = {
    # Same person with different cluster keys (title-based or name-order variants)
    "james i": ["james vi", "king james"],  # James VI of Scotland = James I of England
    "william iii": ["king william"],  # William III / King William (William of Orange)
    "louis xiv": ["lewis xiv"],  # Lewis = older English spelling of Louis
    "st augustine": ["st augustin"],  # Spelling variant
    "charles i": ["king charles"],  # "King Charles" in context = Charles I
    "william the conqueror": ["william i"],  # William I = William the Conqueror
    "robert walpole": ["mr walpole"],  # Same person
}

# --- Known bare surname → full name merges ---
# Only merge when co-occurrence data confirms the identity.
# These are NOT auto-applied — they must be validated first.
SURNAME_MERGES = {
    # Populated after Phase B review.
    # Example:
    # "newton": "isaac newton",  # confirmed: Newton in science articles = Isaac Newton
    # "aristotle": keep as is (already a full name)
}


def load_clusters():
    clusters = []
    with open(INPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                clusters.append(json.loads(line))
    return clusters


def fix_wikidata_matches(clusters):
    """Remove Wikidata matches that are clearly not persons.

    Strategy:
    - If description mentions "genus of", "species", "village", "city" → not a person
    - If it's a concept headword appearing in 5+ editions → keep with note
    """
    removed = 0
    kept_as_concept = 0

    NOT_PERSON_PATTERNS = [
        "genus of", "genus in", "family of", "species of",
        "village", "city", "town", "municipality", "commune",
        "river", "mountain", "island", "lake", "region",
        "province", "county", "district",
        "video game", "tv series", "television", "film",
        "album", "band", "song",
        "protein", "enzyme", "chemical compound",
        "asteroid", "crater",
    ]

    for rec in clusters:
        if rec.get("match_type") != "wikidata":
            continue

        desc = (rec.get("wikidata_description") or "").lower()

        is_not_person = any(pat in desc for pat in NOT_PERSON_PATTERNS)

        if is_not_person:
            if rec.get("is_concept_headword") and rec.get("edition_count", 0) >= 5:
                rec["wikidata_note"] = f"bad_match_kept_as_concept (was: {desc})"
                kept_as_concept += 1
            else:
                rec["match_type"] = "none"
                rec.pop("wikidata_qid", None)
                rec.pop("wikidata_label", None)
                rec.pop("wikidata_description", None)
                rec.pop("birth_year", None)
                rec.pop("death_year", None)
                rec.pop("occupations", None)
                removed += 1

    print(f"  Removed {removed} non-person Wikidata matches")
    print(f"  Kept {kept_as_concept} bad matches because they're concept headwords")
    return clusters


def merge_spelling_variants(clusters):
    """Merge known spelling variant clusters."""
    if not SPELLING_MERGES:
        print("  No spelling merges configured")
        return clusters

    by_id = {r["cluster_id"]: r for r in clusters}
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
            target["variants"].extend(src["variants"])
            target["total_mentions"] += src["total_mentions"]
            target["article_count"] += src["article_count"]

            for ed, cnt in src.get("by_edition", {}).items():
                ed_key = str(ed) if isinstance(ed, int) else ed
                target_ed = target.get("by_edition", {})
                if ed_key in target_ed:
                    target_ed[ed_key] += cnt
                else:
                    target_ed[ed_key] = cnt

            target["edition_count"] = len(target.get("by_edition", {}))

            if src.get("is_concept_headword") and not target.get("is_concept_headword"):
                target["is_concept_headword"] = True
                target["concept_label"] = src.get("concept_label")

            # Merge sample articles
            existing = set(target.get("sample_articles", []))
            for art in src.get("sample_articles", []):
                if art not in existing:
                    target.setdefault("sample_articles", []).append(art)

            to_remove.add(src_id)
            merged_count += 1

    clusters = [r for r in clusters if r["cluster_id"] not in to_remove]

    # Re-sort and re-rank
    clusters.sort(key=lambda r: r["total_mentions"], reverse=True)
    for i, rec in enumerate(clusters):
        rec["frequency_rank"] = i + 1

    print(f"  Merged {merged_count} spelling variant pairs, removed {len(to_remove)} duplicate clusters")
    return clusters


def merge_bare_surnames(clusters):
    """Merge bare surname clusters into full-name clusters where confirmed."""
    if not SURNAME_MERGES:
        print("  No surname merges configured")
        return clusters

    by_id = {r["cluster_id"]: r for r in clusters}
    merged_count = 0
    to_remove = set()

    for surname_id, fullname_id in SURNAME_MERGES.items():
        surname_rec = by_id.get(surname_id)
        fullname_rec = by_id.get(fullname_id)
        if not surname_rec or not fullname_rec:
            continue

        # Merge surname into full name
        fullname_rec["variants"].extend(surname_rec["variants"])
        fullname_rec["total_mentions"] += surname_rec["total_mentions"]
        fullname_rec["article_count"] += surname_rec["article_count"]

        for ed, cnt in surname_rec.get("by_edition", {}).items():
            ed_key = str(ed) if isinstance(ed, int) else ed
            target_ed = fullname_rec.get("by_edition", {})
            if ed_key in target_ed:
                target_ed[ed_key] += cnt
            else:
                target_ed[ed_key] = cnt

        fullname_rec["edition_count"] = len(fullname_rec.get("by_edition", {}))

        to_remove.add(surname_id)
        merged_count += 1

    clusters = [r for r in clusters if r["cluster_id"] not in to_remove]

    clusters.sort(key=lambda r: r["total_mentions"], reverse=True)
    for i, rec in enumerate(clusters):
        rec["frequency_rank"] = i + 1

    print(f"  Merged {merged_count} bare surnames into full-name clusters")
    return clusters


def save_clusters(clusters, path):
    with open(path, "w") as f:
        for rec in clusters:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"  Saved {len(clusters):,} clusters to {path}")


def save_csv(clusters, path):
    import csv
    editions = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "cluster_id", "label", "total_mentions", "article_count",
            "edition_count", "is_concept_headword", "match_type",
            "wikidata_qid", "wikidata_description",
            "birth_year", "death_year", "occupations",
            "alternatives_summary", "variants",
            *[str(y) for y in editions],
        ])
        for r in clusters:
            alts = r.get("alternatives", [])
            alt_str = "; ".join(
                f"{a.get('qid','?')}: {a.get('label','')} ({a.get('description','')})"
                for a in alts
            ) if alts else ""

            occs = "; ".join(r.get("occupations", []))

            w.writerow([
                r["frequency_rank"],
                r["cluster_id"],
                r["label"],
                r["total_mentions"],
                r["article_count"],
                r["edition_count"],
                r.get("is_concept_headword", False),
                r.get("match_type", ""),
                r.get("wikidata_qid", ""),
                r.get("wikidata_description", ""),
                r.get("birth_year", ""),
                r.get("death_year", ""),
                occs,
                alt_str,
                "; ".join(r["variants"][:10]),
                *[r["by_edition"].get(str(y), r["by_edition"].get(y, 0)) for y in editions],
            ])

    print(f"  Saved CSV to {path}")


def print_summary(clusters, min_mentions=5):
    """Print summary stats after cleanup."""
    above = [r for r in clusters if r["total_mentions"] >= min_mentions]
    total_mentions = sum(r["total_mentions"] for r in clusters)

    by_type = Counter(r.get("match_type", "none") for r in above)
    grounded = sum(1 for r in above if r.get("match_type") == "wikidata")
    grounded_mentions = sum(r["total_mentions"] for r in above if r.get("match_type") == "wikidata")

    print(f"\n=== CLEAN PERSON SUMMARY (>= {min_mentions} mentions) ===")
    print(f"  Clusters: {len(above):,} / {len(clusters):,} total")
    print(f"  Grounded: {grounded:,} ({100*grounded/len(above):.1f}%)" if above else "  Grounded: 0")
    print(f"  Grounded mentions: {grounded_mentions:,} / {total_mentions:,} ({100*grounded_mentions/total_mentions:.1f}%)" if total_mentions else "")
    print(f"  Match types: {dict(by_type)}")

    # Top 10 matched
    matched_recs = [r for r in above if r.get("match_type") == "wikidata"]
    matched_recs.sort(key=lambda r: r["total_mentions"], reverse=True)
    print(f"\n  Top 10 matched persons:")
    for r in matched_recs[:10]:
        birth = r.get("birth_year", "?")
        death = r.get("death_year", "?")
        print(f"    {r['label']} ({birth}-{death}): {r['total_mentions']:,} mentions | {r.get('wikidata_qid')}")

    # Top 10 still unmatched
    unmatched = [r for r in above if r.get("match_type") in ("none", None)]
    unmatched.sort(key=lambda r: r["total_mentions"], reverse=True)
    print(f"\n  Top 10 still unmatched:")
    for r in unmatched[:10]:
        concept = " [C]" if r.get("is_concept_headword") else ""
        print(f"    {r['label']}: {r['total_mentions']:,} mentions{concept}")


def main():
    print("Loading clusters...")
    clusters = load_clusters()
    print(f"  {len(clusters):,} clusters loaded")

    print("\nFix 1: Remove non-person Wikidata matches...")
    clusters = fix_wikidata_matches(clusters)

    print("\nFix 2: Merge spelling variants...")
    clusters = merge_spelling_variants(clusters)

    print("\nFix 3: Merge bare surnames...")
    clusters = merge_bare_surnames(clusters)

    print("\nSaving results...")
    save_clusters(clusters, OUTPUT_PATH)
    save_csv(clusters, CSV_PATH)

    print_summary(clusters)

    print("\nDone!")


if __name__ == "__main__":
    main()
