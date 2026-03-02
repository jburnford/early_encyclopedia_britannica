"""Compare new parser's 1st edition output against the earlier hybrid parser.

The earlier parser's output is available at:
  https://jburnford.github.io/early_encyclopedia_britannica/1771/

This module fetches article titles from the web index, loads the new parser's
1st edition articles, and produces a comparison report showing matched, new-only,
and old-only articles.
"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen, Request

from config import (
    INPUT_DIR, ARTICLES_DIR, DEDUP_MANIFEST, COMPARISON_REPORT, EDITIONS,
    ensure_dirs,
)

log = logging.getLogger(__name__)

# Base URL for the earlier hybrid parser
OLD_PARSER_BASE = "https://jburnford.github.io/early_encyclopedia_britannica/1771"
OLD_PARSER_VOLUMES = ["vol1.html", "vol2.html", "vol3.html"]


def fetch_old_parser_titles() -> list[dict]:
    """Fetch article titles from the earlier hybrid parser's web pages.

    Returns list of dicts with 'title', 'volume', 'is_treatise' keys.
    """
    articles = []

    for vol_page in OLD_PARSER_VOLUMES:
        url = f"{OLD_PARSER_BASE}/{vol_page}"
        vol_num = int(vol_page.replace("vol", "").replace(".html", ""))

        log.info(f"Fetching {url}")
        try:
            req = Request(url, headers={"User-Agent": "BritannicaParser/1.0"})
            with urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            log.error(f"Failed to fetch {url}: {e}")
            continue

        # Extract titles from <h3> tags within article-item elements
        # Pattern: <h3>TITLE</h3>
        titles = re.findall(r'<h3>(.*?)</h3>', html)

        # Check for treatise badges near each article-item
        # Find all article-item blocks and check for treatise badge
        article_blocks = re.findall(
            r'<li class="article-item">(.*?)</li>',
            html,
            re.DOTALL,
        )

        for block in article_blocks:
            title_match = re.search(r'<h3>(.*?)</h3>', block)
            if not title_match:
                continue
            title = title_match.group(1).strip()
            # Strip inner HTML tags (e.g., <span class="badge">Place</span>)
            title = re.sub(r'<[^>]+>', '', title).strip()
            # Clean HTML entities
            title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            is_treatise = 'badge treatise' in block
            articles.append({
                'title': title,
                'volume': vol_num,
                'is_treatise': is_treatise,
            })

        # Fallback: if no article-item blocks found, use raw h3 extraction
        if not article_blocks and titles:
            for title in titles:
                raw = title.strip()
                is_treatise = 'badge treatise' in raw
                # Remove entire <span> elements (including badge text like "Place")
                clean = re.sub(r'<span[^>]*>.*?</span>', '', raw, flags=re.DOTALL).strip()
                # Strip any remaining HTML tags
                clean = re.sub(r'<[^>]+>', '', clean).strip()
                clean = clean.replace("&amp;", "&")
                articles.append({
                    'title': clean,
                    'volume': vol_num,
                    'is_treatise': is_treatise,
                })

        log.info(f"  Found {len([a for a in articles if a['volume'] == vol_num])} "
                 f"titles in volume {vol_num}")

    return articles


def normalize_title(title: str) -> str:
    """Normalize a title for comparison.

    Uppercases, strips punctuation, collapses whitespace, removes accents.
    """
    # Remove accents
    nfkd = unicodedata.normalize('NFKD', title)
    ascii_only = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    # Uppercase
    upper = ascii_only.upper()
    # Strip punctuation (keep letters, digits, spaces)
    clean = re.sub(r'[^A-Z0-9\s]', '', upper)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', clean).strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # replace
            ))
        prev_row = curr_row

    return prev_row[-1]


def load_new_parser_articles(canonical_files: list[str] | None = None) -> list[dict]:
    """Load 1st edition articles from new parser output.

    If canonical_files is provided, only load from those files.
    Otherwise, load all 1st edition article files.
    """
    articles = []

    if canonical_files:
        # Filter to 1st edition files
        first_ed_files = [f for f in canonical_files if '_1st_' in f]
        article_paths = [ARTICLES_DIR / f"{f.replace('.jsonl', '')}.articles.jsonl"
                         for f in first_ed_files]
    else:
        article_paths = sorted(ARTICLES_DIR.glob("britannica_1st_*.articles.jsonl"))

    for path in article_paths:
        if not path.exists():
            log.warning(f"Article file not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                article = json.loads(line)
                if article['type'] in ('article', 'cross_reference'):
                    articles.append(article)

    return articles


def run(files: list[Path] | None = None):
    """Compare new parser 1st edition output against earlier hybrid parser."""
    ensure_dirs()

    # Load dedup manifest if available (to get canonical files)
    canonical_files = None
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        canonical_files = manifest.get('canonical')
        log.info(f"Using dedup manifest: {len(canonical_files)} canonical files")

    # Step 1: Fetch old parser titles
    log.info("Fetching titles from earlier hybrid parser...")
    old_articles = fetch_old_parser_titles()
    if not old_articles:
        log.error("No titles fetched from earlier parser. Check network access.")
        return None

    old_titles = [a['title'] for a in old_articles]
    old_treatises = [a['title'] for a in old_articles if a['is_treatise']]
    log.info(f"Old parser: {len(old_titles)} articles, {len(old_treatises)} treatises")

    # Step 2: Load new parser articles
    log.info("Loading new parser 1st edition articles...")
    new_articles = load_new_parser_articles(canonical_files)
    new_titles = [a['title'] for a in new_articles]
    new_real = [a for a in new_articles if a['type'] == 'article']
    new_xrefs = [a for a in new_articles if a['type'] == 'cross_reference']
    log.info(f"New parser: {len(new_titles)} entries "
             f"({len(new_real)} articles, {len(new_xrefs)} cross-refs)")

    # Step 3: Normalize titles and match
    old_normalized = {normalize_title(t): t for t in old_titles}
    new_normalized = {}
    for a in new_articles:
        norm = normalize_title(a['title'])
        if norm not in new_normalized:
            new_normalized[norm] = a

    # Exact matches
    exact_matches = []
    old_only_norm = {}
    for norm, orig in old_normalized.items():
        if norm in new_normalized:
            exact_matches.append({
                'old_title': orig,
                'new_title': new_normalized[norm]['title'],
                'match_type': 'exact',
            })
        else:
            old_only_norm[norm] = orig

    # Remove exact matches from new-only pool
    new_only_norm = {
        norm: a for norm, a in new_normalized.items()
        if norm not in old_normalized
    }

    # Fuzzy matches (Levenshtein ≤ 3 on normalized titles)
    fuzzy_matches = []
    matched_old = set()
    matched_new = set()

    # Only try fuzzy matching for remaining unmatched
    old_remaining = list(old_only_norm.items())
    new_remaining = list(new_only_norm.items())

    for old_norm, old_orig in old_remaining:
        best_dist = 999
        best_new = None
        for new_norm, new_art in new_remaining:
            if new_norm in matched_new:
                continue
            # Quick length filter — Levenshtein can't be < |len diff|
            if abs(len(old_norm) - len(new_norm)) > 3:
                continue
            dist = levenshtein_distance(old_norm, new_norm)
            # Require distance < 30% of shorter string to avoid false positives
            min_len = min(len(old_norm), len(new_norm))
            if dist <= 3 and dist < best_dist and (min_len == 0 or dist / min_len < 0.3):
                best_dist = dist
                best_new = (new_norm, new_art)

        if best_new:
            fuzzy_matches.append({
                'old_title': old_orig,
                'new_title': best_new[1]['title'],
                'match_type': 'fuzzy',
                'distance': best_dist,
            })
            matched_old.add(old_norm)
            matched_new.add(best_new[0])

    # Final old-only and new-only
    old_only = [
        old_only_norm[norm] for norm in old_only_norm
        if norm not in matched_old
    ]
    new_only = [
        new_only_norm[norm]['title'] for norm in new_only_norm
        if norm not in matched_new
    ]

    # Step 4: Build report
    report = {
        'summary': {
            'old_parser_total': len(old_titles),
            'old_parser_treatises': len(old_treatises),
            'new_parser_total': len(new_titles),
            'new_parser_articles': len(new_real),
            'new_parser_cross_refs': len(new_xrefs),
            'exact_matches': len(exact_matches),
            'fuzzy_matches': len(fuzzy_matches),
            'old_only': len(old_only),
            'new_only': len(new_only),
            'match_rate': round(
                100 * (len(exact_matches) + len(fuzzy_matches)) / len(old_titles), 1
            ) if old_titles else 0,
        },
        'exact_matches_sample': exact_matches[:20],
        'fuzzy_matches': fuzzy_matches[:50],
        'old_only': sorted(old_only)[:100],
        'new_only': sorted(new_only)[:100],
        'old_treatises': sorted(old_treatises),
    }

    # Write report
    with open(COMPARISON_REPORT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"Comparison report written to {COMPARISON_REPORT}")

    # Print summary
    s = report['summary']
    print(f"\n{'='*60}")
    print(f"1ST EDITION COMPARISON REPORT")
    print(f"{'='*60}")
    print(f"Old parser (hybrid):  {s['old_parser_total']} titles "
          f"({s['old_parser_treatises']} treatises)")
    print(f"New parser:           {s['new_parser_total']} entries "
          f"({s['new_parser_articles']} articles, "
          f"{s['new_parser_cross_refs']} cross-refs)")
    print()
    print(f"Exact matches:        {s['exact_matches']}")
    print(f"Fuzzy matches:        {s['fuzzy_matches']}")
    print(f"Match rate:           {s['match_rate']}% of old parser titles found")
    print()
    print(f"Old-only (missing):   {s['old_only']}")
    print(f"New-only (new finds): {s['new_only']}")

    if old_only:
        print(f"\nSample OLD-ONLY titles (potential gaps):")
        for t in sorted(old_only)[:20]:
            print(f"  {t}")

    if new_only:
        print(f"\nSample NEW-ONLY titles (new finds):")
        for t in sorted(new_only)[:20]:
            print(f"  {t}")

    if fuzzy_matches:
        print(f"\nFuzzy matches (OCR variation):")
        for m in fuzzy_matches[:10]:
            print(f"  '{m['old_title']}' ~ '{m['new_title']}' (dist={m['distance']})")

    print(f"{'='*60}\n")

    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run()
