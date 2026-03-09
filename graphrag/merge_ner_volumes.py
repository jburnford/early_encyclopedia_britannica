#!/usr/bin/env python3
"""
Merge per-volume NER entity files into per-edition files.

After running NER with --volume, each volume produces a separate file like:
    eb_1st_1771_v2.entities.jsonl
    eb_1st_1771_v3.entities.jsonl

This script merges them into:
    eb_1st_1771.entities.jsonl

Usage:
    python graphrag/merge_ner_volumes.py           # merge all editions
    python graphrag/merge_ner_volumes.py --year 1771  # merge one edition
"""

import argparse
import json
import re
from pathlib import Path

NER_DIR = Path(__file__).resolve().parent.parent / "data" / "ner"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]


def merge_edition(year: int):
    """Merge per-volume files for one edition."""
    # Find volume files: eb_*_YEAR_vN.entities.jsonl
    pattern = re.compile(rf"eb_\w+_{year}_v(\d+)\.entities\.jsonl$")
    vol_files = []
    for f in sorted(NER_DIR.iterdir()):
        m = pattern.match(f.name)
        if m:
            vol_files.append((int(m.group(1)), f))

    if not vol_files:
        print(f"  {year}: no volume files found, skipping")
        return

    vol_files.sort()

    # Determine edition label from first file
    edition_label = vol_files[0][1].name.split("_")[1]
    output_file = NER_DIR / f"eb_{edition_label}_{year}.entities.jsonl"

    total_articles = 0
    total_entities = 0
    with open(output_file, "w") as out:
        for vol_num, vol_file in vol_files:
            count = 0
            with open(vol_file) as f:
                for line in f:
                    out.write(line)
                    record = json.loads(line)
                    count += 1
                    total_entities += sum(record.get("entity_counts", {}).values())
            total_articles += count
            print(f"    vol {vol_num}: {count:,} articles")

    print(f"  {year}: merged {len(vol_files)} volumes → {total_articles:,} articles, "
          f"{total_entities:,} entities → {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Merge per-volume NER results")
    parser.add_argument("--year", type=int, default=None, help="Merge only this edition year")
    args = parser.parse_args()

    years = [args.year] if args.year else EDITION_YEARS
    for year in years:
        merge_edition(year)


if __name__ == "__main__":
    main()
