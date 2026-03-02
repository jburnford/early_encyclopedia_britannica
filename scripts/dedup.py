"""Deduplicate near-identical OCR source files.

Multiple OCR scans of the same volume produce duplicate JSONL files with slightly
different text. This module identifies duplicate groups by comparing text content
and produces a manifest listing canonical files (to keep) and duplicates (to skip).

Algorithm:
1. Parse filenames to extract (edition, volume) tuples
2. Group files sharing the same (edition, volume)
3. For each group with >1 file, compare text similarity using head+tail sampling
4. Mark files with >90% similarity as duplicates
5. Select canonical file per group (prefer named ranges > base > lowest part)
6. Output dedup_manifest.json
"""

import json
import logging
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from config import INPUT_DIR, OUTPUT_DIR, DEDUP_MANIFEST, ensure_dirs

log = logging.getLogger(__name__)

# How many characters to sample from head and tail for similarity comparison
SAMPLE_SIZE = 10_000

# Similarity threshold — above this, files are considered duplicates
SIMILARITY_THRESHOLD = 0.90


def parse_filename(filename: str) -> dict:
    """Extract edition, volume, part, and range from a filename.

    Examples:
        britannica_1st_1771_vol01_unknown.jsonl
          → edition='1st', year='1771', volume='01', part=None, range='unknown'
        britannica_3rd_1797_vol04_CAA-CIC.jsonl
          → edition='3rd', year='1797', volume='04', part=None, range='CAA-CIC'
        britannica_4th_1810_vol00_part37_unknown.jsonl
          → edition='4th', year='1810', volume='00', part='37', range='unknown'
    """
    stem = filename.replace('.jsonl', '')
    # Pattern: britannica_{edition}_{year}_vol{NN}[_part{N}]_{range}
    m = re.match(
        r'britannica_(\w+)_(\d{4})_vol(\d+)(?:_part(\d+))?_(.*)',
        stem
    )
    if not m:
        return None
    return {
        'edition': m.group(1),
        'year': m.group(2),
        'volume': m.group(3),
        'part': m.group(4),
        'range': m.group(5),
        'filename': filename,
    }


def load_text_sample(path: Path) -> str:
    """Load head and tail sample from a JSONL file's text field."""
    with open(path) as f:
        data = json.loads(f.readline())
    text = data.get('text', '')
    if len(text) <= SAMPLE_SIZE * 2:
        return text
    return text[:SAMPLE_SIZE] + text[-SAMPLE_SIZE:]


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute character-level similarity between two text samples."""
    return SequenceMatcher(None, text_a, text_b).ratio()


def select_canonical(group: list[dict]) -> tuple[dict, list[dict]]:
    """Select the canonical file from a group of duplicates.

    Priority:
    1. Files with named letter ranges (e.g., A-ANG) over 'unknown'
    2. Base filename (no part number) over parts
    3. Lowest part number among parts
    """
    def sort_key(info):
        has_range = 0 if info['range'] != 'unknown' else 1
        is_part = 0 if info['part'] is None else 1
        part_num = int(info['part']) if info['part'] else 0
        return (has_range, is_part, part_num)

    sorted_group = sorted(group, key=sort_key)
    canonical = sorted_group[0]
    duplicates = sorted_group[1:]
    return canonical, duplicates


def run(files: list[Path] | None = None):
    """Identify duplicate source files and produce dedup manifest."""
    ensure_dirs()

    if files is None:
        files = sorted(INPUT_DIR.glob('*.jsonl'))

    # Step 1: Parse all filenames
    file_infos = []
    for path in files:
        info = parse_filename(path.name)
        if info:
            info['path'] = path
            file_infos.append(info)
        else:
            log.warning(f"Could not parse filename: {path.name}")

    # Step 2: Group by (edition, volume)
    groups = defaultdict(list)
    for info in file_infos:
        key = (info['edition'], info['volume'])
        groups[key].append(info)

    # Step 3: Find candidate groups (>1 file per edition+volume)
    candidate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    singleton_files = [v[0] for k, v in groups.items() if len(v) == 1]

    log.info(f"Total files: {len(file_infos)}")
    log.info(f"Singleton volumes: {len(singleton_files)}")
    log.info(f"Candidate duplicate groups: {len(candidate_groups)} "
             f"({sum(len(v) for v in candidate_groups.values())} files)")

    # Step 4: Compare text similarity within each group
    manifest_canonical = [info['filename'] for info in singleton_files]
    manifest_duplicates = {}
    manifest_groups = []

    for (edition, volume), group in sorted(candidate_groups.items()):
        log.info(f"Analyzing {edition} vol{volume}: {len(group)} files")

        # Load text samples
        samples = {}
        for info in group:
            try:
                samples[info['filename']] = load_text_sample(info['path'])
            except Exception as e:
                log.error(f"  Failed to load {info['filename']}: {e}")

        # Compute pairwise similarity
        filenames = list(samples.keys())
        similarity_matrix = {}
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                sim = compute_similarity(samples[filenames[i]], samples[filenames[j]])
                similarity_matrix[(filenames[i], filenames[j])] = sim
                log.debug(f"  {filenames[i]} vs {filenames[j]}: {sim:.3f}")

        # Build duplicate clusters using union-find approach
        # Files with >SIMILARITY_THRESHOLD similarity are in the same cluster
        parent = {f: f for f in filenames}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (fa, fb), sim in similarity_matrix.items():
            if sim >= SIMILARITY_THRESHOLD:
                union(fa, fb)

        # Group by cluster root
        clusters = defaultdict(list)
        for f in filenames:
            clusters[find(f)].append(f)

        # Process each cluster
        for cluster_files in clusters.values():
            cluster_infos = [info for info in group if info['filename'] in cluster_files]

            if len(cluster_infos) == 1:
                # No duplicates found for this file
                manifest_canonical.append(cluster_infos[0]['filename'])
                continue

            # Select canonical and mark duplicates
            canonical, duplicates = select_canonical(cluster_infos)
            manifest_canonical.append(canonical['filename'])

            # Compute representative similarity (canonical vs first duplicate)
            dup_filenames = [d['filename'] for d in duplicates]
            rep_sim = 0.0
            for d in dup_filenames:
                key = tuple(sorted([canonical['filename'], d]))
                rep_sim = max(rep_sim, similarity_matrix.get(key, 0.0))

            for dup in duplicates:
                manifest_duplicates[dup['filename']] = canonical['filename']

            manifest_groups.append({
                'edition': canonical['edition'],
                'volume': canonical['volume'],
                'canonical': canonical['filename'],
                'duplicates': dup_filenames,
                'similarity': round(rep_sim, 4),
                'file_count': len(cluster_infos),
            })

            log.info(f"  Cluster: canonical={canonical['filename']}, "
                     f"duplicates={dup_filenames}, similarity={rep_sim:.3f}")

    # Sort canonical list
    manifest_canonical.sort()

    # Build manifest
    manifest = {
        'canonical': manifest_canonical,
        'duplicates': manifest_duplicates,
        'groups': manifest_groups,
        'stats': {
            'total_files': len(file_infos),
            'canonical_files': len(manifest_canonical),
            'duplicate_files': len(manifest_duplicates),
            'duplicate_groups': len(manifest_groups),
        },
    }

    # Write manifest
    with open(DEDUP_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info(f"Dedup manifest written to {DEDUP_MANIFEST}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"DEDUP RESULTS")
    print(f"{'='*60}")
    print(f"Total source files:   {manifest['stats']['total_files']}")
    print(f"Canonical (keep):     {manifest['stats']['canonical_files']}")
    print(f"Duplicates (remove):  {manifest['stats']['duplicate_files']}")
    print(f"Duplicate groups:     {manifest['stats']['duplicate_groups']}")
    print()

    if manifest_groups:
        print("Duplicate groups:")
        for g in manifest_groups:
            print(f"  {g['edition']} vol{g['volume']}: "
                  f"keep {g['canonical']}, "
                  f"drop {g['duplicates']} "
                  f"(sim={g['similarity']:.3f})")
    print(f"{'='*60}\n")

    return manifest


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run()
