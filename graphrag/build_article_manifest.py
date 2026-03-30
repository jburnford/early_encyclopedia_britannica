#!/usr/bin/env python3
"""Build a content-fingerprinted manifest of all exported articles.

Computes SHA-256 hashes for each article so downstream scripts (embeddings,
NER) can detect exactly which articles changed after a parsing fix cycle.

Usage:
    python graphrag/build_article_manifest.py          # build + diff
    python graphrag/build_article_manifest.py --stats   # show summary only
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
MANIFEST_PATH = REPO_ROOT / "data" / "article_manifest.jsonl"
DIFF_PATH = REPO_ROOT / "data" / "article_manifest.diff.json"


def compute_hash(article_id: str, text: str) -> str:
    return hashlib.sha256((article_id + text).encode()).hexdigest()


def load_existing_manifest() -> dict[str, dict]:
    """Load previous manifest as {article_id: record}."""
    if not MANIFEST_PATH.exists():
        return {}
    manifest = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                manifest[rec["article_id"]] = rec
    return manifest


def build_manifest() -> list[dict]:
    """Read all export files and build manifest records."""
    records = []
    export_files = sorted(EXPORT_DIR.glob("eb_*_*.jsonl"))
    if not export_files:
        print(f"ERROR: No export files found in {EXPORT_DIR}", file=sys.stderr)
        sys.exit(1)

    for fpath in export_files:
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                art = json.loads(line)
                records.append({
                    "article_id": art["article_id"],
                    "title": art["title"],
                    "edition_year": art["edition_year"],
                    "word_count": art.get("word_count", len(art.get("text", "").split())),
                    "content_hash": compute_hash(art["article_id"], art.get("text", "")),
                })
    # Sort for deterministic output
    records.sort(key=lambda r: r["article_id"])
    return records


def diff_manifests(old: dict[str, dict], new: list[dict]) -> dict:
    """Compare old and new manifests, return diff."""
    new_by_id = {r["article_id"]: r for r in new}
    old_ids = set(old.keys())
    new_ids = set(new_by_id.keys())

    added = sorted(new_ids - old_ids)
    deleted = sorted(old_ids - new_ids)
    changed = sorted(
        aid for aid in (old_ids & new_ids)
        if old[aid]["content_hash"] != new_by_id[aid]["content_hash"]
    )
    unchanged = len(old_ids & new_ids) - len(changed)

    return {
        "added": added,
        "changed": changed,
        "deleted": deleted,
        "summary": {
            "added": len(added),
            "changed": len(changed),
            "deleted": len(deleted),
            "unchanged": unchanged,
            "total_new": len(new),
            "total_old": len(old),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build article content manifest")
    parser.add_argument("--stats", action="store_true", help="Show summary only, don't write")
    args = parser.parse_args()

    print("Loading existing manifest...")
    old_manifest = load_existing_manifest()
    print(f"  Previous: {len(old_manifest):,} articles")

    print("Building new manifest from export files...")
    new_records = build_manifest()
    print(f"  Current:  {len(new_records):,} articles")

    diff = diff_manifests(old_manifest, new_records)
    s = diff["summary"]
    print(f"\nDiff: +{s['added']} added, ~{s['changed']} changed, "
          f"-{s['deleted']} deleted, ={s['unchanged']} unchanged")

    if args.stats:
        return

    # Write new manifest
    with open(MANIFEST_PATH, "w") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nManifest written: {MANIFEST_PATH}")

    # Write diff
    with open(DIFF_PATH, "w") as f:
        json.dump(diff, f, indent=2)
    print(f"Diff written:     {DIFF_PATH}")

    # Show changed articles if any
    if diff["changed"]:
        new_by_id = {r["article_id"]: r for r in new_records}
        print(f"\nChanged articles ({len(diff['changed'])}):")
        for aid in diff["changed"][:20]:
            old_wc = old_manifest[aid]["word_count"]
            new_wc = new_by_id[aid]["word_count"]
            old_title = old_manifest[aid]["title"]
            new_title = new_by_id[aid]["title"]
            title_note = f" (was '{old_title}')" if old_title != new_title else ""
            print(f"  {new_title}{title_note} [{new_by_id[aid]['edition_year']}]: "
                  f"{old_wc:,}w → {new_wc:,}w")
        if len(diff["changed"]) > 20:
            print(f"  ... and {len(diff['changed']) - 20} more")

    if diff["added"]:
        new_by_id = {r["article_id"]: r for r in new_records}
        print(f"\nAdded articles ({len(diff['added'])}):")
        for aid in diff["added"][:10]:
            rec = new_by_id[aid]
            print(f"  {rec['title']} [{rec['edition_year']}]: {rec['word_count']:,}w")
        if len(diff["added"]) > 10:
            print(f"  ... and {len(diff['added']) - 10} more")


if __name__ == "__main__":
    main()
