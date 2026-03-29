#!/usr/bin/env python3
"""
Headword disambiguation support script.

The actual disambiguation happens interactively in Claude Code via MCP
(mcp__wikidata__search_items + get_statements). This script handles the
bookkeeping: building the queue, verifying QIDs, and reporting status.

Commands:
  --prepare     Build/refresh the lookup queue from cross-edition index
  --show-batch  Show next N unmatched headwords for MCP disambiguation
  --verify      Batch-verify all QIDs against Wikidata API
  --status      Report current progress
"""

import json
import sys
import argparse
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import HTTPError, URLError

# --- Configuration ---

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data"
NER_DIR = DATA_DIR / "ner"

CROSS_EDITION_INDEX = DATA_DIR / "cross_edition_index.jsonl"
PERSON_MATCHES = NER_DIR / "person_matches.jsonl"
TOPONYM_CLUSTERS = NER_DIR / "toponym_clusters_clean.jsonl"
MATCHES_PATH = DATA_DIR / "headword_matches.jsonl"

SKIP_HEADWORDS = {
    "DISS", "ZYGOMATICUS", "MATERA", "GENUS_IX", "SCRIBONIUS",
}


def load_cross_edition_index():
    headwords = []
    with open(CROSS_EDITION_INDEX) as f:
        for line in f:
            obj = json.loads(line)
            hid = obj["id"]
            title = obj.get("canonical_title", hid)
            editions = obj.get("editions", {})
            twc = sum(e.get("word_count", 0) for e in editions.values())
            ec = obj.get("edition_count", len(editions))
            headwords.append({
                "headword_id": hid,
                "canonical_title": title,
                "total_word_count": twc,
                "edition_count": ec,
            })
    headwords.sort(key=lambda x: x["total_word_count"], reverse=True)
    return headwords


def load_existing_matches():
    matched = {}
    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    matched[rec["headword_id"]] = rec
    return matched


def prepare(min_words=1000):
    """Build the queue and auto-populate from cross-references."""
    headwords = load_cross_edition_index()
    existing = load_existing_matches()

    # Load cross-references
    person_xref = {}
    if PERSON_MATCHES.exists():
        with open(PERSON_MATCHES) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("wikidata_qid"):
                        cid = rec["cluster_id"].upper().replace(" ", "_")
                        person_xref[cid] = {
                            "qid": rec["wikidata_qid"],
                            "label": rec.get("wikidata_label", ""),
                            "desc": rec.get("wikidata_desc",
                                           rec.get("wikidata_description", "")),
                        }

    toponym_xref = {}
    if TOPONYM_CLUSTERS.exists():
        with open(TOPONYM_CLUSTERS) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("wikidata_qid"):
                        label = rec.get("label", "").upper().replace(" ", "_")
                        toponym_xref[label] = {
                            "qid": rec["wikidata_qid"],
                            "label": rec.get("wikidata_label", ""),
                            "desc": rec.get("wikidata_desc", ""),
                        }

    # Auto-populate
    auto_added = 0
    with open(MATCHES_PATH, "a") as f:
        for hw in headwords:
            hid = hw["headword_id"]
            if hid in existing or hid in SKIP_HEADWORDS:
                continue
            xref = person_xref.get(hid) or toponym_xref.get(hid)
            if xref:
                rec = {
                    "headword_id": hid,
                    "canonical_title": hw["canonical_title"],
                    "wikidata_qid": xref["qid"],
                    "wikidata_label": xref["label"],
                    "wikidata_desc": xref["desc"],
                    "match_type": "crossref",
                    "total_word_count": hw["total_word_count"],
                    "edition_count": hw["edition_count"],
                }
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
                existing[hid] = rec
                auto_added += 1

    # Count remaining
    remaining = [hw for hw in headwords
                 if hw["headword_id"] not in existing
                 and hw["headword_id"] not in SKIP_HEADWORDS
                 and hw["total_word_count"] >= min_words]

    print(f"Total headwords: {len(headwords):,}")
    print(f"Already matched: {len(existing):,}")
    print(f"Auto-populated from cross-refs: {auto_added:,}")
    print(f"Remaining (>= {min_words:,} words): {len(remaining):,}")


def show_batch(n=50, min_words=1000):
    """Show next N unmatched headwords for MCP disambiguation."""
    headwords = load_cross_edition_index()
    existing = load_existing_matches()

    remaining = [hw for hw in headwords
                 if hw["headword_id"] not in existing
                 and hw["headword_id"] not in SKIP_HEADWORDS
                 and hw["total_word_count"] >= min_words]

    batch = remaining[:n]
    print(f"Next {len(batch)} unmatched headwords "
          f"({len(remaining)} total remaining):\n")
    for i, hw in enumerate(batch):
        print(f"  {i+1:3d}. {hw['canonical_title']:<45s} "
              f"{hw['total_word_count']:>10,d} words  {hw['edition_count']} eds")


def verify_qids():
    """Batch-verify all QIDs against Wikidata API."""
    existing = load_existing_matches()
    entries = [e for e in existing.values() if e.get("wikidata_qid")]
    qids = list(set(e["wikidata_qid"] for e in entries))

    if not qids:
        print("No QIDs to verify.")
        return 0

    print(f"Verifying {len(qids)} unique QIDs across {len(entries)} matches...")

    wikidata_info = {}
    for i in range(0, len(qids), 50):
        batch = qids[i:i+50]
        ids = "|".join(batch)
        url = (f"https://www.wikidata.org/w/api.php?action=wbgetentities"
               f"&ids={ids}&props=labels|descriptions&languages=en&format=json")
        req = Request(url, headers={"User-Agent": "EB-NLS-verification/1.0"})
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            for qid, info in data.get("entities", {}).items():
                label = info.get("labels", {}).get("en", {}).get("value", "???")
                desc = info.get("descriptions", {}).get("en", {}).get("value", "")
                wikidata_info[qid] = {"label": label, "desc": desc}
        except Exception as e:
            print(f"  ERROR fetching batch {i}: {e}", file=sys.stderr)
        time.sleep(0.5)
        if (i // 50) % 5 == 0:
            print(f"  Fetched {min(i+50, len(qids))}/{len(qids)}...")

    ok = 0
    bad = []
    for e in entries:
        qid = e["wikidata_qid"]
        if qid not in wikidata_info:
            continue
        stored = e.get("wikidata_label", "").lower().strip()
        actual = wikidata_info[qid]["label"].lower().strip()
        if stored == actual:
            ok += 1
        else:
            # Check if it's just a label variant vs truly wrong
            hw = e["headword_id"].lower().replace("_", " ")
            actual_l = wikidata_info[qid]["label"].lower()
            desc_l = wikidata_info[qid]["desc"].lower()
            stored_l = stored.lower()
            # Match if headword or stored label appears in actual label/desc/aliases
            # or vice versa; also handle "???" labels (missing English label)
            related = (
                hw in actual_l or actual_l in hw or hw in desc_l
                or stored_l in actual_l or actual_l in stored_l
                or stored_l in desc_l or hw in stored_l
                or actual_l == "???"  # missing English label, can't compare
                # Check if stored label words appear in desc
                or any(w in desc_l for w in stored_l.split() if len(w) > 3)
                # Check if the headword is a known synonym/hypernym
                or (hw.rstrip("s") in actual_l or actual_l in hw.rstrip("s"))
                # Also check stored label against aliases via description keywords
                or (stored_l.rstrip("s") in actual_l or actual_l in stored_l.rstrip("s"))
            )
            bad.append({
                "headword_id": e["headword_id"],
                "qid": qid,
                "stored_label": e.get("wikidata_label", ""),
                "actual_label": wikidata_info[qid]["label"],
                "actual_desc": wikidata_info[qid]["desc"],
                "likely_ok": related,
            })

    truly_bad = [b for b in bad if not b["likely_ok"]]
    total = ok + len(bad)

    print(f"\n=== VERIFICATION ===")
    print(f"OK (exact label match): {ok}")
    print(f"Label variants (likely OK): {len(bad) - len(truly_bad)}")
    print(f"WRONG entity: {len(truly_bad)}")
    print(f"Error rate: {len(truly_bad)}/{total} = "
          f"{len(truly_bad)/total*100:.1f}%" if total else "N/A")

    if truly_bad:
        print(f"\n=== BAD MATCHES ===")
        for b in truly_bad:
            print(f"  {b['headword_id']:30s} {b['qid']:12s}")
            print(f"    Stored:  {b['stored_label']}")
            print(f"    Actual:  {b['actual_label']} — {b['actual_desc']}")

    return len(truly_bad)


def status():
    """Report current progress."""
    headwords = load_cross_edition_index()
    existing = load_existing_matches()

    from collections import Counter
    types = Counter(e.get("match_type", "?") for e in existing.values())

    print(f"Headword disambiguation status:")
    print(f"  Total headwords: {len(headwords):,}")
    print(f"  Matched: {len(existing):,}")
    print(f"  Remaining: {len(headwords) - len(existing):,}")
    print(f"\n  By match type:")
    for t, c in types.most_common():
        print(f"    {t:20s} {c:,}")

    # Tier breakdown
    tiers = [(100000, "100K+"), (50000, "50K-100K"), (20000, "20K-50K"),
             (10000, "10K-20K"), (5000, "5K-10K"), (1000, "1K-5K"), (0, "<1K")]
    print(f"\n  By word count tier:")
    for i, (threshold, label) in enumerate(tiers):
        upper = tiers[i-1][0] if i > 0 else float("inf")
        in_tier = [h for h in headwords
                   if threshold <= h["total_word_count"] < upper]
        matched = sum(1 for h in in_tier if h["headword_id"] in existing)
        print(f"    {label:12s}: {matched:,}/{len(in_tier):,} matched")


def main():
    parser = argparse.ArgumentParser(description="Headword disambiguation support")
    parser.add_argument("--prepare", action="store_true",
                        help="Build queue, auto-populate from cross-refs")
    parser.add_argument("--show-batch", type=int, nargs="?", const=50,
                        help="Show next N unmatched headwords (default 50)")
    parser.add_argument("--verify", action="store_true",
                        help="Batch-verify all QIDs against Wikidata API")
    parser.add_argument("--status", action="store_true",
                        help="Report current progress")
    parser.add_argument("--min-words", type=int, default=1000,
                        help="Min total word count (default: 1000)")
    args = parser.parse_args()

    if args.prepare:
        prepare(min_words=args.min_words)
    elif args.show_batch is not None:
        show_batch(n=args.show_batch, min_words=args.min_words)
    elif args.verify:
        bad_count = verify_qids()
        if bad_count > 0:
            print(f"\n⚠ {bad_count} bad QIDs found. Fix before committing.")
            sys.exit(1)
    elif args.status:
        status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
