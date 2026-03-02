"""Audit alphabetical order flags from verification.

The verify phase flags ~18,000 out-of-order titles but only stores 5 examples
per file. This module performs a deep audit: re-running the full alphabetical
check and categorizing each flag to determine which are genuine parsing errors
versus expected artifacts.

Categories:
- subsection_artifact: Title from a merged treatise's internal subsection list
- cross_ref_interspersed: Cross-references like "ABANGA. See ADY" between articles
- multi_sense: Same title appearing multiple times (legitimate)
- front_back_matter: "ENCYCLOPAEDIA BRITANNICA" and similar structural titles
- roman_numeral_title: Short Roman numeral titles (I, II, III, etc.)
- short_garbage: OCR garbage or very short titles (1-2 chars)
- genuine_error: True out-of-order articles suggesting misclassification
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from config import (
    INPUT_DIR, ARTICLES_DIR, DEDUP_MANIFEST, ORDER_AUDIT_REPORT, ensure_dirs,
)

log = logging.getLogger(__name__)

# Known front/back matter title patterns
FRONT_BACK_PATTERNS = re.compile(
    r'^(ENCYCLOP[AÆ]DIA|BRITANNICA|PREFACE|INTRODUCTION|ADVERTISEMENT|'
    r'DIRECTIONS|INDEX|CONTENTS|ERRATA|FINIS|SUPPLEMENT|APPENDIX|'
    r'TO THE READER|THE END|SUBSCRIBERS)',
    re.IGNORECASE,
)

# Roman numeral pattern
ROMAN_NUMERAL = re.compile(
    r'^[IVXLCDM]+\.?$'
)

# Short garbage titles
SHORT_GARBAGE = re.compile(
    r'^[^A-Za-z]*$|^.{1,2}$'
)


def categorize_flag(prev_article: dict, curr_article: dict,
                    all_titles_in_file: set) -> str:
    """Categorize an alphabetical order flag.

    Args:
        prev_article: The article before the out-of-order one
        curr_article: The out-of-order article
        all_titles_in_file: Set of all titles in this file (for multi-sense check)

    Returns:
        Category string.
    """
    curr_title = curr_article['title']
    prev_title = prev_article['title']
    curr_type = curr_article['type']

    # Cross-reference interspersed with articles
    if curr_type == 'cross_reference':
        return 'cross_ref_interspersed'

    # Front/back matter leaking into article stream
    if FRONT_BACK_PATTERNS.match(curr_title):
        return 'front_back_matter'
    if FRONT_BACK_PATTERNS.match(prev_title):
        return 'front_back_matter'

    # Roman numeral titles (from numbered sections)
    if ROMAN_NUMERAL.match(curr_title):
        return 'roman_numeral_title'

    # Short garbage (1-2 chars or no letters)
    if SHORT_GARBAGE.match(curr_title) or len(curr_title.strip()) <= 2:
        return 'short_garbage'

    # Multi-sense check: if the same title appears elsewhere in the file,
    # it's a legitimate multi-sense entry (e.g., MERCURY as element and planet)
    curr_norm = curr_title.upper().strip()
    if curr_norm in all_titles_in_file:
        # Count occurrences
        count = sum(1 for t in all_titles_in_file if t == curr_norm)
        if count > 1:
            return 'multi_sense'

    # Subsection artifact: title that looks like it leaked from a subsection
    # These tend to be mixed-case or very short relative to surrounding articles
    if not curr_title.isupper() and len(curr_title) > 2:
        return 'subsection_artifact'

    # If the curr title sorts between predecessor's predecessor and successor,
    # it might just be a minor OCR-induced swap (not a systematic error)
    # We treat remaining ALL-CAPS out-of-order titles as genuine errors
    return 'genuine_error'


def audit_file(articles_path: Path) -> list[dict]:
    """Re-run alphabetical order check with full categorization.

    Returns list of flag dicts with category, context, etc.
    """
    articles = []
    with open(articles_path) as f:
        for line in f:
            articles.append(json.loads(line))

    # Only check article and cross_reference types (skip front/back matter)
    checkable = [a for a in articles if a['type'] in ('article', 'cross_reference')]

    # Build title set for multi-sense detection
    all_titles = defaultdict(int)
    for a in checkable:
        all_titles[a['title'].upper().strip()] += 1
    multi_sense_titles = {t for t, c in all_titles.items() if c > 1}

    flags = []
    for i in range(1, len(checkable)):
        prev = checkable[i - 1]
        curr = checkable[i]

        prev_upper = prev['title'].upper()
        curr_upper = curr['title'].upper()

        # Same tolerance as verify.py: allow if first 2 chars match
        if curr_upper < prev_upper and prev_upper[:2] != curr_upper[:2]:
            category = categorize_flag(
                prev, curr,
                multi_sense_titles,
            )
            flags.append({
                'prev_title': prev['title'],
                'curr_title': curr['title'],
                'prev_index': i - 1,
                'curr_index': i,
                'prev_type': prev['type'],
                'curr_type': curr['type'],
                'category': category,
                'source_file': articles_path.stem.replace('.articles', ''),
            })

    return flags


def run(files: list[Path] | None = None):
    """Audit all alphabetical order flags across article files."""
    ensure_dirs()

    # Determine which files to process
    canonical_files = None
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        canonical_files = set(manifest.get('canonical', []))
        log.info(f"Using dedup manifest: {len(canonical_files)} canonical files")

    if files is None:
        article_paths = sorted(ARTICLES_DIR.glob('*.articles.jsonl'))
    else:
        article_paths = []
        for f in files:
            apath = ARTICLES_DIR / f"{f.stem}.articles.jsonl"
            if apath.exists():
                article_paths.append(apath)

    # Filter to canonical files if manifest exists
    if canonical_files:
        article_paths = [
            p for p in article_paths
            if p.stem.replace('.articles', '') + '.jsonl' in canonical_files
        ]

    if not article_paths:
        log.warning("No article files found to audit")
        return None

    log.info(f"Auditing alphabetical order in {len(article_paths)} files")

    # Collect all flags
    all_flags = []
    files_with_flags = 0

    for apath in article_paths:
        flags = audit_file(apath)
        if flags:
            files_with_flags += 1
        all_flags.extend(flags)

    log.info(f"Total flags: {len(all_flags)} across {files_with_flags} files")

    # Categorize
    by_category = defaultdict(list)
    for flag in all_flags:
        by_category[flag['category']].append(flag)

    # Build report
    category_summary = {}
    category_samples = {}
    for cat, flags in sorted(by_category.items()):
        category_summary[cat] = {
            'count': len(flags),
            'pct': round(100 * len(flags) / len(all_flags), 1) if all_flags else 0,
        }
        # Sample 20 examples per category
        category_samples[cat] = flags[:20]

    # Per-file breakdown
    per_file = defaultdict(lambda: defaultdict(int))
    for flag in all_flags:
        per_file[flag['source_file']][flag['category']] += 1

    # Top files by genuine errors
    genuine_by_file = {
        f: cats.get('genuine_error', 0) for f, cats in per_file.items()
    }
    top_error_files = sorted(
        genuine_by_file.items(), key=lambda x: -x[1]
    )[:20]

    report = {
        'summary': {
            'total_flags': len(all_flags),
            'files_audited': len(article_paths),
            'files_with_flags': files_with_flags,
        },
        'categories': category_summary,
        'samples': category_samples,
        'top_error_files': [
            {'file': f, 'genuine_errors': c} for f, c in top_error_files if c > 0
        ],
        'per_file': {f: dict(cats) for f, cats in per_file.items()},
    }

    # Recommendations
    genuine_count = category_summary.get('genuine_error', {}).get('count', 0)
    genuine_pct = category_summary.get('genuine_error', {}).get('pct', 0)

    recommendations = []
    if genuine_pct < 5:
        recommendations.append(
            f"Genuine errors are {genuine_pct}% of flags ({genuine_count} total) — "
            f"within acceptable range. Most flags are expected artifacts."
        )
    else:
        recommendations.append(
            f"Genuine errors are {genuine_pct}% of flags ({genuine_count} total) — "
            f"review top error files for systematic issues."
        )

    xref_count = category_summary.get('cross_ref_interspersed', {}).get('count', 0)
    if xref_count > 0:
        recommendations.append(
            f"{xref_count} flags from cross-references interspersed with articles — "
            f"expected behavior, safe to ignore."
        )

    sub_count = category_summary.get('subsection_artifact', {}).get('count', 0)
    if sub_count > 0:
        recommendations.append(
            f"{sub_count} subsection artifacts — mixed-case titles that leaked "
            f"from treatise subsections. Review if merge phase needs adjustment."
        )

    report['recommendations'] = recommendations

    # Write report
    with open(ORDER_AUDIT_REPORT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Order audit report written to {ORDER_AUDIT_REPORT}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"ALPHABETICAL ORDER AUDIT")
    print(f"{'='*60}")
    print(f"Files audited:    {len(article_paths)}")
    print(f"Files with flags: {files_with_flags}")
    print(f"Total flags:      {len(all_flags)}")
    print()

    print("Category breakdown:")
    for cat, info in sorted(category_summary.items(), key=lambda x: -x[1]['count']):
        print(f"  {cat:30s} {info['count']:6d}  ({info['pct']}%)")
    print()

    # Sample from each category
    for cat, samples in sorted(category_samples.items()):
        print(f"\n--- {cat} (sample) ---")
        for s in samples[:5]:
            print(f"  '{s['prev_title']}' > '{s['curr_title']}' "
                  f"[{s['source_file']}]")

    print()
    print("Recommendations:")
    for r in recommendations:
        print(f"  - {r}")

    if top_error_files and top_error_files[0][1] > 0:
        print(f"\nTop files by genuine errors:")
        for f, c in top_error_files[:10]:
            if c > 0:
                print(f"  {f}: {c} errors")

    print(f"{'='*60}\n")

    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run()
