#!/usr/bin/env python3
"""Verify headword_matches.jsonl QIDs against Wikidata API.

Fetches actual labels/descriptions for each QID and flags mismatches.
"""
import json
import time
import urllib.request
import urllib.parse
import sys

def fetch_wikidata_labels(qids, lang="en"):
    """Fetch labels and descriptions for a batch of QIDs (max 50)."""
    ids = "|".join(qids)
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids}&props=labels|descriptions|aliases&languages={lang}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "EB-NLS-verification/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def main():
    with open("data/headword_matches.jsonl") as f:
        entries = [json.loads(l) for l in f]

    print(f"Verifying {len(entries)} entries against Wikidata API...")

    # Batch fetch in groups of 50
    qid_to_entries = {}
    for e in entries:
        qid = e["wikidata_qid"]
        if qid not in qid_to_entries:
            qid_to_entries[qid] = []
        qid_to_entries[qid].append(e)

    all_qids = list(qid_to_entries.keys())
    wikidata_info = {}

    for i in range(0, len(all_qids), 50):
        batch = all_qids[i:i+50]
        try:
            result = fetch_wikidata_labels(batch)
            for qid, data in result.get("entities", {}).items():
                label = data.get("labels", {}).get("en", {}).get("value", "???")
                desc = data.get("descriptions", {}).get("en", {}).get("value", "")
                aliases = [a["value"] for a in data.get("aliases", {}).get("en", [])]
                wikidata_info[qid] = {"label": label, "desc": desc, "aliases": aliases}
        except Exception as ex:
            print(f"  ERROR fetching batch {i}: {ex}", file=sys.stderr)
        time.sleep(0.5)  # rate limit
        if (i // 50) % 5 == 0:
            print(f"  Fetched {min(i+50, len(all_qids))}/{len(all_qids)} QIDs...")

    # Compare
    mismatches = []
    missing = []
    ok = []

    for e in entries:
        qid = e["wikidata_qid"]
        headword = e["headword_id"]
        stored_label = e["wikidata_label"]

        if qid not in wikidata_info:
            missing.append(e)
            continue

        actual = wikidata_info[qid]
        actual_label = actual["label"]

        # Check if stored label matches actual label
        if stored_label.lower().strip() == actual_label.lower().strip():
            ok.append(e)
        else:
            mismatches.append({
                "headword": headword,
                "qid": qid,
                "stored_label": stored_label,
                "actual_label": actual_label,
                "actual_desc": actual["desc"],
                "actual_aliases": actual["aliases"][:5],
            })

    print(f"\n=== RESULTS ===")
    print(f"OK (label matches): {len(ok)}")
    print(f"Label mismatches: {len(mismatches)}")
    print(f"Missing from API: {len(missing)}")

    if mismatches:
        print(f"\n=== MISMATCHES (need review) ===")
        for m in mismatches:
            # Flag severity: is the headword even vaguely related?
            headword_lower = m["headword"].lower().replace("_", " ")
            actual_lower = m["actual_label"].lower()
            aliases_lower = [a.lower() for a in m["actual_aliases"]]

            # Check if headword appears in label, desc, or aliases
            related = (
                headword_lower in actual_lower or
                actual_lower in headword_lower or
                any(headword_lower in a for a in aliases_lower) or
                any(a in headword_lower for a in aliases_lower) or
                headword_lower in m["actual_desc"].lower()
            )

            severity = "LIKELY_OK" if related else "BAD"
            print(f"  [{severity}] {m['headword']:30s} {m['qid']:12s}")
            print(f"    Stored:  {m['stored_label']}")
            print(f"    Actual:  {m['actual_label']} — {m['actual_desc']}")
            if m['actual_aliases']:
                print(f"    Aliases: {', '.join(m['actual_aliases'])}")

    # Save results
    with open("data/qid_verification.json", "w") as f:
        json.dump({
            "ok_count": len(ok),
            "mismatch_count": len(mismatches),
            "missing_count": len(missing),
            "mismatches": mismatches,
        }, f, indent=2)
    print(f"\nDetailed results saved to data/qid_verification.json")

if __name__ == "__main__":
    main()
