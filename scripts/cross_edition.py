"""
Cross-edition validation for the LIS-based Britannica parser (V2).

Pass 1: Build union headword index across all 8 editions.
Pass 2: Score articles by cross-edition confidence, recover false negatives
         from rejected candidates and mega-articles, flag false positives.
Pass 3: Generate confidence report with benchmark validation.
"""

import bisect
import json
import logging
from collections import defaultdict
from pathlib import Path

from config import (
    ARTICLES_DIR, OUTPUT_DIR, DEDUP_MANIFEST, OCR_MANIFEST, INPUT_DIR,
    CONFIDENCE_THRESHOLD, ensure_dirs,
)
from lis_parser import (
    normalize_sort_key, generate_candidates, strip_front_matter,
    strip_back_matter, HeadingCandidate,
)

log = logging.getLogger(__name__)


UNION_INDEX_PATH = OUTPUT_DIR / "union_index.json"
CROSS_EDITION_REPORT = OUTPUT_DIR / "cross_edition_report.json"
CONFIDENCE_REPORT = OUTPUT_DIR / "confidence_report.json"

# 20 benchmark terms expected in most editions
BENCHMARK_TERMS = [
    'AGRICULTURE', 'ANATOMY', 'ARCHITECTURE', 'ASTRONOMY', 'BOTANY',
    'CHEMISTRY', 'ELECTRICITY', 'GEOGRAPHY', 'HISTORY', 'LAW',
    'MATHEMATICS', 'MEDICINE', 'MUSIC', 'NAVIGATION', 'OPTICS',
    'PAINTING', 'PHILOSOPHY', 'SURGERY', 'TANNING', 'ZOOLOGY',
]

MEGA_ARTICLE_THRESHOLD = 50_000  # words


def get_canonical_files() -> list[str]:
    """Get canonical filenames from OCR manifest, dedup manifest, or all files."""
    # Prefer new OCR manifest (has correct volume assignments)
    if OCR_MANIFEST.exists():
        with open(OCR_MANIFEST) as f:
            manifest = json.load(f)
        canonical = sorted([
            e['filename'] for e in manifest.get('files', [])
            if e.get('is_canonical', True)
        ])
        if canonical:
            log.info(f"Using OCR manifest: {len(canonical)} canonical files")
            return canonical

    # Fall back to legacy dedup manifest
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        canonical = manifest.get('canonical', [])
        if canonical:
            log.info(f"Using dedup manifest: {len(canonical)} canonical files")
            return canonical

    log.warning("No manifest found — using all input files")
    return [p.name for p in sorted(INPUT_DIR.glob('*.jsonl'))]


# ---------------------------------------------------------------------------
# Pass 1: Build union index
# ---------------------------------------------------------------------------

def build_union_index(canonical_files: list[str]) -> dict[str, dict]:
    """Build a union headword index from all editions' LIS-parsed articles.

    Returns: {normalized_headword: {
        'headword': str,           # canonical form (first seen)
        'editions': {edition_name: [volume_numbers]},
        'count': int,              # how many editions have it
    }}
    """
    union: dict[str, dict] = {}

    for filename in canonical_files:
        stem = filename.replace('.jsonl', '')
        path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        if not path.exists():
            continue

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                article = json.loads(line)
                if article.get('type') not in ('article', 'cross_reference'):
                    continue

                norm = normalize_sort_key(article['title'])
                if norm not in union:
                    union[norm] = {
                        'headword': article['title'],
                        'editions': defaultdict(list),
                        'count': 0,
                    }

                entry = union[norm]
                edition = article['edition']
                vol = article['volume']
                if vol not in entry['editions'][edition]:
                    entry['editions'][edition].append(vol)

    # Count unique editions per headword
    for entry in union.values():
        entry['count'] = len(entry['editions'])
        entry['editions'] = dict(entry['editions'])

    return union


def flag_anomalies(
    union_index: dict[str, dict],
    threshold: int = 5,
) -> dict[str, list[str]]:
    """Flag headwords present in >=threshold editions but missing from others.

    Returns: {edition_name: [missing_headwords]}
    """
    all_editions: set[str] = set()
    for entry in union_index.values():
        all_editions.update(entry['editions'].keys())

    missing_by_edition: dict[str, list[str]] = defaultdict(list)
    for norm, entry in union_index.items():
        if entry['count'] >= threshold:
            present = set(entry['editions'].keys())
            missing = all_editions - present
            for edition in missing:
                missing_by_edition[edition].append(entry['headword'])

    return dict(missing_by_edition)


# ---------------------------------------------------------------------------
# Pass 2a: Cross-edition confidence scoring
# ---------------------------------------------------------------------------

def score_cross_edition_confidence(edition_count: int, total_editions: int = 8) -> float:
    """Score a headword based on how many editions contain it."""
    if edition_count >= 6:
        return 0.95
    elif edition_count >= 3:
        return 0.80
    elif edition_count == 2:
        return 0.70
    else:  # edition_count == 1
        return 0.50


def score_all_articles(canonical_files: list[str], union_index: dict[str, dict]):
    """Score every article by cross-edition confidence and write updated files.

    Adds 'cross_edition_confidence' and 'combined_confidence' fields.
    """
    updated_count = 0

    for filename in canonical_files:
        stem = filename.replace('.jsonl', '')
        path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        if not path.exists():
            continue

        articles = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                articles.append(json.loads(line))

        for article in articles:
            norm = normalize_sort_key(article['title'])
            if norm in union_index:
                ce_conf = score_cross_edition_confidence(union_index[norm]['count'])
            else:
                ce_conf = 0.5

            lis_conf = article.get('lis_confidence', 1.0)
            article['cross_edition_confidence'] = ce_conf
            article['combined_confidence'] = round(lis_conf * ce_conf, 3)
            updated_count += 1

        with open(path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

    log.info(f"  Scored {updated_count} articles with cross-edition confidence")


# ---------------------------------------------------------------------------
# Pass 2b: False-negative recovery
# ---------------------------------------------------------------------------

def recover_false_negatives(
    canonical_files: list[str],
    union_index: dict[str, dict],
    anomalies: dict[str, list[str]],
) -> dict[str, int]:
    """Recover headwords missing from specific editions.

    For each headword flagged as missing from an edition:
    1. Search that edition's rejected candidates for the headword.
    2. Search inside mega-articles for the headword as a missed boundary.

    Returns: {edition_name: count_recovered}
    """
    recovered_counts: dict[str, int] = defaultdict(int)

    # Group canonical files by edition
    files_by_edition: dict[str, list[str]] = defaultdict(list)
    for filename in canonical_files:
        stem = filename.replace('.jsonl', '')
        path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            first_line = f.readline().strip()
            if first_line:
                article = json.loads(first_line)
                files_by_edition[article['edition']].append(filename)

    for edition_name, missing_headwords in anomalies.items():
        if not missing_headwords:
            continue
        missing_norms = {normalize_sort_key(hw) for hw in missing_headwords}

        edition_files = files_by_edition.get(edition_name, [])
        if not edition_files:
            continue

        for filename in edition_files:
            stem = filename.replace('.jsonl', '')
            articles_path = ARTICLES_DIR / f"{stem}.articles.jsonl"
            if not articles_path.exists():
                continue

            # Load existing articles
            articles = []
            with open(articles_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    articles.append(json.loads(line))

            # Check for mega-articles that might contain missing headwords
            mega_articles = [a for a in articles if a['word_count'] >= MEGA_ARTICLE_THRESHOLD]
            if not mega_articles:
                continue

            # Re-read OCR text for this file to search for missed boundaries
            input_path = INPUT_DIR / filename
            if not input_path.exists():
                continue

            with open(input_path) as f:
                meta = json.loads(f.readline())
            text = meta['text']
            edition_year = meta['edition']

            # Generate fresh candidates from the text
            all_candidates = generate_candidates(text, edition_year)
            all_candidates = strip_front_matter(all_candidates, text)
            all_candidates = strip_back_matter(all_candidates, text)

            # Find candidates inside mega-articles that match missing headwords
            new_boundaries = []
            for mega in mega_articles:
                mega_start = mega['char_start']
                mega_end = mega['char_end']
                mega_key = normalize_sort_key(mega['title'])

                for c in all_candidates:
                    if c.char_start <= mega_start or c.char_start >= mega_end:
                        continue
                    if c.sort_key == mega_key:
                        continue  # Running header
                    if c.sort_key in missing_norms:
                        new_boundaries.append(c)

            if not new_boundaries:
                continue

            # Deduplicate
            existing_starts = {a['char_start'] for a in articles}
            unique_new = [nb for nb in new_boundaries if nb.char_start not in existing_starts]

            if not unique_new:
                continue

            # Re-slice the mega-articles with the new boundaries inserted
            # Build complete accepted list with new boundaries
            all_accepted = []
            for a in articles:
                all_accepted.append(HeadingCandidate(
                    headword=a['title'],
                    sort_key=normalize_sort_key(a['title']),
                    char_start=a['char_start'],
                    char_end=a['char_start'] + len(a['title']) + 2,  # approximate
                    pattern=a.get('heading_pattern', 'article'),
                    confidence=a.get('lis_confidence', 1.0),
                ))

            all_accepted.extend(unique_new)
            all_accepted.sort(key=lambda c: c.char_start)

            # Re-extract articles
            from lis_parser import extract_articles
            new_articles = extract_articles(
                all_accepted, text,
                edition_name, edition_year,
                meta['volume'], filename,
            )

            # Score new articles
            for article in new_articles:
                norm = normalize_sort_key(article['title'])
                if norm in union_index:
                    ce_conf = score_cross_edition_confidence(union_index[norm]['count'])
                else:
                    ce_conf = 0.5
                lis_conf = article.get('lis_confidence', 1.0)
                article['cross_edition_confidence'] = ce_conf
                article['combined_confidence'] = round(lis_conf * ce_conf, 3)
                article['recovered'] = True

            # Write updated articles
            with open(articles_path, 'w') as f:
                for article in new_articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')

            recovered_counts[edition_name] += len(unique_new)
            log.info(f"    {filename}: recovered {len(unique_new)} headwords from mega-articles")

    return dict(recovered_counts)


# ---------------------------------------------------------------------------
# Pass 2c: False-positive flagging
# ---------------------------------------------------------------------------

def flag_false_positives(canonical_files: list[str], union_index: dict[str, dict]) -> int:
    """Flag single-edition headwords with low word count as likely noise.

    Adds 'flagged_false_positive': true to suspicious articles.
    Returns count of flagged articles.
    """
    flagged_count = 0

    for filename in canonical_files:
        stem = filename.replace('.jsonl', '')
        path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        if not path.exists():
            continue

        articles = []
        modified = False
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                articles.append(json.loads(line))

        for article in articles:
            norm = normalize_sort_key(article['title'])
            entry = union_index.get(norm)

            if (entry and entry['count'] == 1
                    and article['word_count'] < 20
                    and article.get('type') != 'cross_reference'):
                article['flagged_false_positive'] = True
                flagged_count += 1
                modified = True

        if modified:
            with open(path, 'w') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')

    return flagged_count


# ---------------------------------------------------------------------------
# Pass 3: Confidence report
# ---------------------------------------------------------------------------

def generate_report(
    union_index: dict[str, dict],
    anomalies: dict[str, list[str]],
    canonical_files: list[str],
    recovered_counts: dict[str, int] | None = None,
    flagged_fp_count: int = 0,
) -> dict:
    """Generate comprehensive confidence report."""
    all_editions: set[str] = set()
    for entry in union_index.values():
        all_editions.update(entry['editions'].keys())

    # Per-edition article counts
    edition_counts: dict[str, int] = defaultdict(int)
    for entry in union_index.values():
        for ed in entry['editions']:
            edition_counts[ed] += 1

    # Coverage distribution
    coverage_dist: dict[int, int] = defaultdict(int)
    for entry in union_index.values():
        coverage_dist[entry['count']] += 1

    # Per-edition confidence distribution and mega-article audit
    edition_stats: dict[str, dict] = {}
    total_mega = 0
    total_articles_all = 0
    total_below_threshold = 0

    for edition in sorted(all_editions):
        stats = {
            'total_articles': 0,
            'mega_articles': 0,
            'mega_article_titles': [],
            'confidence_below_threshold': 0,
            'flagged_false_positives': 0,
            'recovered': recovered_counts.get(edition, 0) if recovered_counts else 0,
        }

        for filename in canonical_files:
            stem = filename.replace('.jsonl', '')
            path = ARTICLES_DIR / f"{stem}.articles.jsonl"
            if not path.exists():
                continue

            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    article = json.loads(line)
                    if article.get('edition') != edition:
                        continue
                    stats['total_articles'] += 1
                    total_articles_all += 1

                    if article['word_count'] >= MEGA_ARTICLE_THRESHOLD:
                        stats['mega_articles'] += 1
                        total_mega += 1
                        stats['mega_article_titles'].append(
                            f"{article['title']} ({article['word_count']:,} words)"
                        )

                    combined = article.get('combined_confidence', 1.0)
                    if combined < CONFIDENCE_THRESHOLD:
                        stats['confidence_below_threshold'] += 1
                        total_below_threshold += 1

                    if article.get('flagged_false_positive'):
                        stats['flagged_false_positives'] += 1

        edition_stats[edition] = stats

    # Benchmark terms coverage
    benchmark_coverage: dict[str, dict[str, bool]] = {}
    for term in BENCHMARK_TERMS:
        norm = normalize_sort_key(term)
        entry = union_index.get(norm)
        benchmark_coverage[term] = {}
        for edition in sorted(all_editions):
            benchmark_coverage[term][edition] = (
                edition in entry['editions'] if entry else False
            )

    benchmark_summary = {}
    for edition in sorted(all_editions):
        found = sum(1 for term in BENCHMARK_TERMS
                    if benchmark_coverage[term].get(edition, False))
        benchmark_summary[edition] = f"{found}/{len(BENCHMARK_TERMS)}"

    return {
        'total_unique_headwords': len(union_index),
        'total_articles': total_articles_all,
        'editions': sorted(all_editions),
        'headwords_per_edition': dict(sorted(edition_counts.items())),
        'coverage_distribution': {
            f'in_{k}_editions': v for k, v in sorted(coverage_dist.items())
        },
        'mega_articles': {
            'total': total_mega,
            'threshold_words': MEGA_ARTICLE_THRESHOLD,
        },
        'confidence': {
            'threshold': CONFIDENCE_THRESHOLD,
            'below_threshold': total_below_threshold,
            'pct_below': round(total_below_threshold / max(total_articles_all, 1) * 100, 2),
        },
        'false_positives_flagged': flagged_fp_count,
        'recovery': recovered_counts or {},
        'per_edition': edition_stats,
        'benchmark_terms': {
            'terms': BENCHMARK_TERMS,
            'coverage': benchmark_coverage,
            'summary': benchmark_summary,
        },
        'anomalies': {
            ed: {
                'missing_count': len(missing),
                'sample': sorted(missing)[:20],
            }
            for ed, missing in sorted(anomalies.items())
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(files: list[Path] | None = None):
    """Full 2-pass cross-edition validation pipeline."""
    ensure_dirs()

    canonical_files = get_canonical_files()
    log.info(f"Building union index from {len(canonical_files)} canonical files")

    # Pass 1: Build union index
    union_index = build_union_index(canonical_files)
    log.info(f"Union index: {len(union_index)} unique headwords")

    # Flag anomalies
    anomalies = flag_anomalies(union_index, threshold=5)
    total_anomalies = sum(len(v) for v in anomalies.values())
    log.info(f"Anomalies: {total_anomalies} missing headwords across {len(anomalies)} editions")

    # Pass 2a: Score all articles
    log.info("Pass 2a: Scoring articles by cross-edition confidence...")
    score_all_articles(canonical_files, union_index)

    # Pass 2b: Recover false negatives from mega-articles
    log.info("Pass 2b: Recovering false negatives from mega-articles...")
    recovered_counts = recover_false_negatives(canonical_files, union_index, anomalies)
    total_recovered = sum(recovered_counts.values())
    log.info(f"  Total recovered: {total_recovered}")

    # Pass 2c: Flag false positives
    log.info("Pass 2c: Flagging false positives...")
    flagged_fp = flag_false_positives(canonical_files, union_index)
    log.info(f"  Flagged {flagged_fp} potential false positives")

    # Pass 3: Generate report
    log.info("Pass 3: Generating confidence report...")
    report = generate_report(
        union_index, anomalies, canonical_files,
        recovered_counts, flagged_fp,
    )

    # Write outputs
    with open(UNION_INDEX_PATH, 'w') as f:
        json.dump(union_index, f, indent=2, ensure_ascii=False)
    log.info(f"Union index written to {UNION_INDEX_PATH}")

    with open(CROSS_EDITION_REPORT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Report written to {CROSS_EDITION_REPORT}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-Edition Validation Report (V2)")
    print(f"{'='*60}")
    print(f"Unique headwords: {report['total_unique_headwords']:,}")
    print(f"Total articles:   {report['total_articles']:,}")
    print(f"\nHeadwords per edition:")
    for ed, count in report['headwords_per_edition'].items():
        n_missing = len(anomalies.get(ed, []))
        n_recovered = recovered_counts.get(ed, 0)
        print(f"  {ed}: {count:,} headwords, {n_missing} missing, {n_recovered} recovered")

    print(f"\nCoverage distribution:")
    for k, v in report['coverage_distribution'].items():
        print(f"  {k}: {v:,}")

    print(f"\nMega-articles (>{MEGA_ARTICLE_THRESHOLD:,} words): {report['mega_articles']['total']}")
    print(f"Below confidence {CONFIDENCE_THRESHOLD}: "
          f"{report['confidence']['below_threshold']} ({report['confidence']['pct_below']}%)")
    print(f"Flagged false positives: {flagged_fp}")

    print(f"\nBenchmark terms ({len(BENCHMARK_TERMS)} terms):")
    for ed, score in report['benchmark_terms']['summary'].items():
        print(f"  {ed}: {score}")
