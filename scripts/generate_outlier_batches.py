#!/usr/bin/env python3
"""
Generate LLM review batches for alphabetic outliers.

Takes the detected outliers from detect_alphabetic_outliers.py and creates
structured batches for LLM review. The LLM will determine:
1. MERGE - outlier should be merged into a specific article (provide merge target)
2. RENAME - outlier headword is OCR error (provide corrected headword)
3. KEEP - outlier is actually a valid standalone article

Uses page numbers to identify merge candidates.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "llm_corrections" / "outlier_batches"
OUTLIERS_FILE = PROJECT_ROOT / "llm_corrections" / "outliers" / "alphabetic_outliers.json"
BATCH_SIZE = 25  # Smaller batches for more complex decisions

def load_articles(edition_year: int) -> dict[str, dict]:
    """Load articles indexed by article_id."""
    filepath = PROJECT_ROOT / f"output_v2/articles_{edition_year}.jsonl"
    articles = {}
    with open(filepath) as f:
        for line in f:
            art = json.loads(line)
            articles[art['article_id']] = art
    return articles

def get_text_context(articles: dict, article_id: str, position: str = 'end', chars: int = 500) -> str:
    """Get text context from an article."""
    art = articles.get(article_id, {})
    text = art.get('text', '')
    if position == 'end':
        return text[-chars:].strip()
    else:  # start
        return text[:chars].strip()

def enrich_outlier(outlier: dict, articles: dict) -> dict:
    """Enrich outlier with full article text context."""
    article_id = outlier['article_id']
    art = articles.get(article_id, {})

    # Get full text preview (first 1000 chars for LLM to understand content)
    full_text = art.get('text', '')

    # Get merge candidate context
    enriched_candidates = []
    for candidate in outlier.get('merge_candidates', []):
        cand_id = candidate.get('article_id')
        if cand_id and cand_id in articles:
            cand_art = articles[cand_id]
            enriched_candidates.append({
                'article_id': cand_id,
                'headword': candidate['headword'],
                'start_page': candidate['start_page'],
                'end_page': candidate['end_page'],
                'word_count': candidate['word_count'],
                'text_end': get_text_context(articles, cand_id, 'end', 800)
            })

    return {
        'article_id': article_id,
        'headword': outlier['headword'],
        'edition_year': outlier['edition_year'],
        'volume_num': outlier['volume_num'],
        'start_page': outlier['start_page'],
        'end_page': outlier['end_page'],
        'word_count': outlier['word_count'],
        'first_letter': outlier['first_letter'],
        'expected_range': outlier['expected_range'],
        'reason': outlier['reason'],
        'text_preview': full_text[:1000],
        'text_end': full_text[-500:] if len(full_text) > 500 else '',
        'merge_candidates': enriched_candidates,
        'prev_articles': outlier.get('prev_articles', []),
        'next_articles': outlier.get('next_articles', [])
    }

def generate_batches(edition_year: int, outliers: list[dict]) -> list[dict]:
    """Generate batches for a single edition."""
    if not outliers:
        return []

    # Load articles for context
    articles = load_articles(edition_year)

    # Enrich outliers with full context
    enriched = [enrich_outlier(o, articles) for o in outliers]

    # Split into batches
    batches = []
    for i in range(0, len(enriched), BATCH_SIZE):
        batch_items = enriched[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        batches.append({
            'edition_year': edition_year,
            'batch_num': batch_num,
            'total_batches': (len(enriched) + BATCH_SIZE - 1) // BATCH_SIZE,
            'items': batch_items
        })

    return batches

def main():
    # Load detected outliers
    if not OUTLIERS_FILE.exists():
        print(f"Error: {OUTLIERS_FILE} not found. Run detect_alphabetic_outliers.py first.")
        sys.exit(1)

    with open(OUTLIERS_FILE) as f:
        all_results = json.load(f)

    # Filter to main files only (not backups)
    # Main files have specific article counts based on post-correction data
    main_file_counts = {
        1771: 12452, 1778: 17027, 1797: 20910, 1810: 14899,
        1815: 18335, 1823: 15753, 1842: 19482, 1860: 16259
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_batches = 0
    total_outliers = 0

    print("Generating LLM review batches for alphabetic outliers")
    print("=" * 60)

    for result in all_results:
        year = result['edition_year']
        count = result['total_articles']

        # Skip backup files
        if year in main_file_counts and count != main_file_counts[year]:
            continue

        outliers = result['outliers']
        if not outliers:
            continue

        batches = generate_batches(year, outliers)

        # Save batches
        for batch in batches:
            batch_file = OUTPUT_DIR / f"outlier_batch_{year}_{batch['batch_num']:03d}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch, f, indent=2)

        print(f"{year}: {len(outliers):3d} outliers -> {len(batches)} batches")
        total_batches += len(batches)
        total_outliers += len(outliers)

    print("=" * 60)
    print(f"Total: {total_outliers} outliers in {total_batches} batches")
    print(f"Output: {OUTPUT_DIR}")

    # Generate summary file
    summary = {
        'total_outliers': total_outliers,
        'total_batches': total_batches,
        'batch_size': BATCH_SIZE,
        'editions': {}
    }

    for result in all_results:
        year = result['edition_year']
        count = result['total_articles']
        if year in main_file_counts and count == main_file_counts[year]:
            summary['editions'][year] = {
                'outliers': len(result['outliers']),
                'batches': (len(result['outliers']) + BATCH_SIZE - 1) // BATCH_SIZE
            }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {OUTPUT_DIR / 'summary.json'}")

if __name__ == '__main__':
    main()
