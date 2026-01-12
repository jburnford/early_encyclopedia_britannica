#!/usr/bin/env python3
"""
Generate review batches from volume outliers.

Creates structured batches for LLM-assisted or human review of detected
volume outliers. Batches are grouped by edition/volume and include
auto-classification suggestions with confidence levels.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def load_outliers(input_file: Path) -> list[dict]:
    """Load all outliers from all editions."""
    with open(input_file) as f:
        results = json.load(f)

    all_outliers = []
    for edition in results:
        for outlier in edition['outliers']:
            all_outliers.append(outlier)

    return all_outliers


def load_articles_for_context(edition_year: int) -> dict[str, dict]:
    """Load articles for an edition to get full text context."""
    articles_file = Path(f'output_v2/articles_{edition_year}.jsonl')
    articles = {}

    if articles_file.exists():
        with open(articles_file) as f:
            for line in f:
                art = json.loads(line)
                articles[art.get('article_id', '')] = art

    return articles


def create_batch(outliers: list[dict], batch_num: int, edition_year: int, volume_num: int) -> dict:
    """Create a single review batch from outliers."""

    # Load full articles for context
    articles = load_articles_for_context(edition_year)

    items = []
    for i, outlier in enumerate(outliers):
        # Get full article text
        article_id = outlier.get('article_id', '')
        full_text = ''
        if article_id in articles:
            full_text = articles[article_id].get('text', '')

        # Get merge candidates with their full text
        merge_candidates = []
        for mc in outlier.get('merge_candidates', [])[:4]:
            mc_id = mc.get('article_id', '')
            mc_text = ''
            if mc_id in articles:
                mc_text = articles[mc_id].get('text', '')

            merge_candidates.append({
                'article_id': mc_id,
                'headword': mc.get('headword', ''),
                'direction': mc.get('direction', ''),
                'start_page': mc.get('start_page', 0),
                'text_preview': mc_text[:1000] if mc_text else '',
                'text_end': mc_text[-500:] if len(mc_text) > 500 else ''
            })

        classification = outlier.get('classification', {})

        items.append({
            'item_num': i + 1,
            'article_id': article_id,
            'headword': outlier.get('headword', ''),
            'edition_year': edition_year,
            'volume_num': volume_num,
            'start_page': outlier.get('start_page', 0),
            'end_page': outlier.get('end_page', 0),
            'word_count': outlier.get('word_count', 0),
            'volume_range': outlier.get('volume_range', ''),
            'effective_start': outlier.get('effective_start', ''),
            'effective_end': outlier.get('effective_end', ''),
            'reason': outlier.get('reason', ''),
            'text_preview': outlier.get('text_preview', ''),
            'text_end': outlier.get('text_end', ''),
            'full_text': full_text[:2000] if full_text else '',
            'auto_classification': {
                'decision': classification.get('decision', 'REVIEW'),
                'confidence': classification.get('confidence', 'low'),
                'reason': classification.get('reason', ''),
                'merge_target': classification.get('merge_target', '')
            },
            'prev_articles': outlier.get('prev_articles', []),
            'next_articles': outlier.get('next_articles', []),
            'merge_candidates': merge_candidates
        })

    return {
        'batch_id': f'vol_outliers_{edition_year}_v{volume_num:02d}_{batch_num:03d}',
        'created': datetime.now().isoformat(),
        'edition_year': edition_year,
        'volume_num': volume_num,
        'item_count': len(items),
        'items': items
    }


def generate_batches(
    outliers: list[dict],
    batch_size: int = 20,
    output_dir: Path = None
) -> list[dict]:
    """Generate review batches from outliers, grouped by edition/volume."""

    if output_dir is None:
        output_dir = Path('llm_corrections/volume_outlier_batches')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by edition and volume
    by_edition_vol = defaultdict(list)
    for outlier in outliers:
        key = (outlier['edition_year'], outlier['volume_num'])
        by_edition_vol[key].append(outlier)

    all_batches = []

    for (edition_year, volume_num), vol_outliers in sorted(by_edition_vol.items()):
        # Sort by page
        vol_outliers.sort(key=lambda o: o.get('start_page', 0))

        # Split into batches
        for batch_num, start_idx in enumerate(range(0, len(vol_outliers), batch_size)):
            batch_outliers = vol_outliers[start_idx:start_idx + batch_size]
            batch = create_batch(batch_outliers, batch_num + 1, edition_year, volume_num)

            # Save batch
            batch_file = output_dir / f"{batch['batch_id']}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch, f, indent=2)

            all_batches.append({
                'batch_id': batch['batch_id'],
                'file': str(batch_file),
                'edition_year': edition_year,
                'volume_num': volume_num,
                'item_count': batch['item_count']
            })

    return all_batches


def generate_summary_prompt(batch: dict) -> str:
    """Generate a prompt for LLM review of a batch."""

    prompt_lines = [
        "# Volume Outlier Review Batch",
        "",
        f"Edition: {batch['edition_year']}",
        f"Volume: {batch['volume_num']}",
        f"Items: {batch['item_count']}",
        "",
        "## Review Instructions",
        "",
        "Each item below is an article that appears OUTSIDE its volume's expected",
        "alphabetic range. For each item, provide a decision:",
        "",
        "- **MERGE <target>**: Merge this text into the specified target article",
        "- **DELETE**: Remove entirely (publisher notices, structural markers)",
        "- **RENAME <new_headword>**: Fix OCR error in the headword",
        "- **KEEP**: Valid article, leave as-is",
        "",
        "Auto-classifications are provided as suggestions. Override if needed.",
        "",
        "---",
        ""
    ]

    for item in batch['items']:
        cls = item['auto_classification']
        prompt_lines.extend([
            f"## Item {item['item_num']}: {item['headword']}",
            "",
            f"- **Page**: {item['start_page']}",
            f"- **Volume Range**: {item['volume_range']} ({item['effective_start']}-{item['effective_end']})",
            f"- **Reason**: {item['reason']}",
            f"- **Word Count**: {item['word_count']}",
            "",
            f"**Auto-Classification**: {cls['decision']} ({cls['confidence']})",
            f"- Reason: {cls['reason']}",
        ])

        if cls.get('merge_target'):
            prompt_lines.append(f"- Suggested merge target: {cls['merge_target']}")

        prompt_lines.extend([
            "",
            "**Text Preview**:",
            "```",
            item['text_preview'][:800] if item['text_preview'] else "(no text)",
            "```",
            "",
            "**Context (previous articles)**:",
        ])

        for prev in item['prev_articles'][:2]:
            prompt_lines.append(f"- p.{prev.get('start_page', '?')}: {prev.get('headword', '?')}")

        prompt_lines.append("")
        prompt_lines.append("**Context (next articles)**:")

        for nxt in item['next_articles'][:2]:
            prompt_lines.append(f"- p.{nxt.get('start_page', '?')}: {nxt.get('headword', '?')}")

        prompt_lines.extend([
            "",
            "**Decision**: _________________",
            "",
            "---",
            ""
        ])

    return "\n".join(prompt_lines)


def main():
    input_file = Path('llm_corrections/outliers/volume_outliers.json')

    if not input_file.exists():
        print(f"Error: Run detect_volume_outliers.py first to generate {input_file}")
        return

    print("Loading volume outliers...")
    outliers = load_outliers(input_file)
    print(f"  Found {len(outliers)} outliers")

    print("\nGenerating review batches...")
    batches = generate_batches(outliers, batch_size=20)

    print(f"\nGenerated {len(batches)} batches:")

    # Group batches by edition
    by_edition = defaultdict(list)
    for b in batches:
        by_edition[b['edition_year']].append(b)

    total_items = 0
    for year in sorted(by_edition.keys()):
        edition_batches = by_edition[year]
        items = sum(b['item_count'] for b in edition_batches)
        total_items += items
        print(f"  {year}: {len(edition_batches)} batches ({items} items)")

    print(f"\nTotal: {len(batches)} batches, {total_items} items")

    # Generate sample prompts
    prompt_dir = Path('llm_corrections/volume_outlier_prompts')
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating sample prompts to {prompt_dir}...")

    # Generate first batch prompt for each edition
    for year in sorted(by_edition.keys())[:3]:
        first_batch_file = Path(by_edition[year][0]['file'])
        with open(first_batch_file) as f:
            batch = json.load(f)

        prompt = generate_summary_prompt(batch)
        prompt_file = prompt_dir / f"sample_prompt_{year}.md"
        with open(prompt_file, 'w') as f:
            f.write(prompt)

        print(f"  Created {prompt_file}")

    print("\nBatch generation complete!")
    print(f"\nBatch files: llm_corrections/volume_outlier_batches/")
    print(f"Sample prompts: llm_corrections/volume_outlier_prompts/")


if __name__ == '__main__':
    main()
