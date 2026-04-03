#!/usr/bin/env python3
"""Apply anomaly classifications: REMOVE or MERGE flagged articles.

REMOVE articles are deleted from the export.
MERGE articles are appended (text concatenated) to the preceding
non-flagged article in char_start order within the same source file.

Usage:
    python scripts/apply_anomaly_fixes.py --dry-run    # preview
    python scripts/apply_anomaly_fixes.py               # apply
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO / "data" / "export"


def load_classifications():
    """Load REMOVE/MERGE classifications from agent output files."""
    remove_ids = set()
    merge_ids = set()

    for i in [1, 2, 3]:
        fp = REPO / "data" / f"anomaly_classifications_{i}.txt"
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 2:
                    continue
                aid = parts[0]
                action = parts[1].upper()
                if action == "REMOVE":
                    remove_ids.add(aid)
                elif action == "MERGE":
                    merge_ids.add(aid)

    # ETCHLADE is a misplaced standalone article — REMOVE not MERGE
    remove_ids.add("eb_3rd_1797_012795")
    merge_ids.discard("eb_3rd_1797_012795")

    return remove_ids, merge_ids


# Cross-volume merges: orphan articles that continue from the previous volume
CROSS_VOLUME_MERGES = {
    # orphan_id: parent_id
    "eb_3rd_1797_017787": "eb_3rd_1797_017785",     # ELIZABETH → WHEN THE (Britain)
    "eb_5th_1815_016669": "eb_5th_1815_016666",     # HOLD → POETRY
    "eb_6th_1823_008701": "eb_6th_1823_008699",     # ABDC → ELECTRICITY
    "eb_8th_1860_007331": "eb_8th_1860_007335",     # CONRAD GESNER → ENTOMOLOGY
}


def process_edition(export_path, remove_ids, merge_ids, dry_run=False):
    """Process one edition's export file."""
    # Load all articles
    articles = []
    with open(export_path) as f:
        for line in f:
            articles.append(json.loads(line))

    # Index by article_id for cross-volume merges
    by_id = {a["article_id"]: a for a in articles}

    # Handle cross-volume merges first (append orphan text to parent in different source file)
    cross_merged = set()
    for orphan_id, parent_id in CROSS_VOLUME_MERGES.items():
        if orphan_id in by_id and parent_id in by_id:
            orphan = by_id[orphan_id]
            parent = by_id[parent_id]
            parent["text"] = parent["text"] + "\n\n" + orphan["text"]
            parent["word_count"] = len(parent["text"].split())
            cross_merged.add(orphan_id)
            print(f"  CROSS-VOL MERGE: {orphan['title']} → {parent['title']}")

    # Group by source_file
    by_sf = defaultdict(list)
    for a in articles:
        by_sf[a.get("source_file", "")].append(a)

    # For each source file, walk in char_start order and merge/remove
    merged_count = 0
    removed_count = 0
    merged_words = 0
    removed_ids_actual = set()
    merge_targets = {}  # merge_id -> parent_id (for reporting)

    for sf, sf_articles in by_sf.items():
        sf_articles.sort(key=lambda a: a["char_start"])

        current_parent = None  # last non-flagged article

        for a in sf_articles:
            aid = a["article_id"]

            if aid in remove_ids:
                removed_ids_actual.add(aid)
                removed_count += 1
                # Don't update current_parent — removals are invisible
                continue

            if aid in merge_ids:
                if current_parent is not None and current_parent["article_id"] not in remove_ids:
                    # Append text to parent
                    parent_aid = current_parent["article_id"]
                    current_parent["text"] = (
                        current_parent["text"] + "\n\n" + a["text"]
                    )
                    current_parent["word_count"] = len(
                        current_parent["text"].split()
                    )
                    current_parent["char_end"] = a["char_end"]
                    merge_targets[aid] = (
                        parent_aid,
                        current_parent["title"],
                        a["title"],
                    )
                    merged_count += 1
                    merged_words += a.get("word_count", 0)
                else:
                    # No valid parent — keep as-is (orphaned merge)
                    print(f"  WARNING: No parent for MERGE {a['title']} ({aid})")
                    current_parent = a
                # Don't update current_parent — next merge goes to same parent
                continue

            # Normal article — becomes the new current parent
            current_parent = a

    # Rebuild the article list: remove REMOVE, MERGE, and cross-vol merged articles
    flagged = removed_ids_actual | set(merge_targets.keys()) | cross_merged
    new_articles = [a for a in articles if a["article_id"] not in flagged]

    if not dry_run and (merged_count > 0 or removed_count > 0):
        with open(export_path, "w") as f:
            for a in new_articles:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    return merged_count, removed_count, merged_words, merge_targets


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply anomaly fixes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remove_ids, merge_ids = load_classifications()
    print(f"Classifications: {len(remove_ids)} REMOVE, {len(merge_ids)} MERGE")

    total_merged = 0
    total_removed = 0
    total_merged_words = 0

    for fp in sorted(EXPORT_DIR.glob("eb_*.jsonl")):
        year = fp.stem.split("_")[-1]
        merged, removed, mwords, targets = process_edition(
            fp, remove_ids, merge_ids, args.dry_run
        )

        if merged > 0 or removed > 0:
            print(f"\n{year}: {merged} merged, {removed} removed ({mwords:,} words merged)")
            for mid, (pid, ptitle, mtitle) in sorted(
                targets.items(), key=lambda x: x[1][1]
            )[:10]:
                print(f"  MERGE: {mtitle} → {ptitle}")
            if len(targets) > 10:
                print(f"  ... +{len(targets) - 10} more")

        total_merged += merged
        total_removed += removed
        total_merged_words += mwords

    prefix = "DRY RUN — " if args.dry_run else ""
    print(f"\n{prefix}Summary:")
    print(f"  Merged: {total_merged} articles ({total_merged_words:,} words)")
    print(f"  Removed: {total_removed} articles")
    print(f"  Net reduction: {total_merged + total_removed} articles")


if __name__ == "__main__":
    main()
