#!/usr/bin/env python3
"""Verify toponym_clusters_clean.jsonl QIDs against Wikidata API."""
import json, time, urllib.request, sys

def fetch_wikidata_labels(qids, lang="en"):
    ids = "|".join(qids)
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids}&props=labels|descriptions|aliases&languages={lang}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "EB-NLS-verification/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def main():
    with open("data/ner/toponym_clusters_clean.jsonl") as f:
        entries = [json.loads(l) for l in f]

    # Filter to entries with QIDs
    with_qid = [e for e in entries if e.get("wikidata_qid")]
    print(f"Total entries: {len(entries)}, with QIDs: {len(with_qid)}")

    qid_to_entries = {}
    for e in with_qid:
        qid = e["wikidata_qid"]
        qid_to_entries.setdefault(qid, []).append(e)

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
        time.sleep(0.5)
        if (i // 50) % 5 == 0:
            print(f"  Fetched {min(i+50, len(all_qids))}/{len(all_qids)} QIDs...")

    mismatches = []
    ok = 0
    for e in with_qid:
        qid = e["wikidata_qid"]
        if qid not in wikidata_info: continue
        stored_label = e.get("wikidata_label", "")
        actual = wikidata_info[qid]
        actual_label = actual["label"]

        if stored_label.lower().strip() == actual_label.lower().strip():
            ok += 1
        else:
            cluster_label = e.get("label", "").lower()
            actual_lower = actual_label.lower()
            aliases_lower = [a.lower() for a in actual["aliases"]]
            desc_lower = actual["desc"].lower()

            # Check if it's a place entity
            is_place = any(w in desc_lower for w in ["city","town","country","region","province","state","island","river","mountain","village","county","district","territory","kingdom","empire","peninsula","sea","ocean","lake","port","colony","commune","municipality"])

            related = (
                stored_label.lower() in actual_lower or
                actual_lower in stored_label.lower() or
                any(stored_label.lower() in a for a in aliases_lower) or
                cluster_label in actual_lower or
                actual_lower in cluster_label
            )
            severity = "LIKELY_OK" if (related or is_place) else "BAD"
            mismatches.append({
                "label": e.get("label", ""),
                "cluster_id": e.get("cluster_id", ""),
                "qid": qid,
                "stored_label": stored_label,
                "actual_label": actual_label,
                "actual_desc": actual["desc"],
                "severity": severity,
            })

    bad_count = sum(1 for m in mismatches if m["severity"] == "BAD")
    total = ok + len(mismatches)
    print(f"\n=== RESULTS ===")
    print(f"OK (label matches): {ok}")
    print(f"Label mismatches: {len(mismatches)} (BAD: {bad_count}, LIKELY_OK: {len(mismatches)-bad_count})")
    print(f"Error rate: {bad_count}/{total} = {bad_count/total*100:.1f}%" if total else "N/A")

    if bad_count > 0:
        print(f"\n=== BAD MISMATCHES ===")
        for m in mismatches:
            if m["severity"] == "BAD":
                print(f"  {m['label']:30s} {m['qid']:12s}")
                print(f"    Stored:  {m['stored_label']}")
                print(f"    Actual:  {m['actual_label']} — {m['actual_desc']}")

    with open("data/toponym_qid_verification.json", "w") as f:
        json.dump({"ok": ok, "mismatches": len(mismatches), "bad": bad_count, "details": mismatches}, f, indent=2)
    print(f"\nSaved to data/toponym_qid_verification.json")

if __name__ == "__main__":
    main()
