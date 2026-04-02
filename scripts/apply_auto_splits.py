#!/usr/bin/env python3
"""Apply auto-detected article splits using paragraph character positions.

Operates on export files (data/export/eb_*.jsonl) which match the paragraph
embeddings. Uses exact character positions from the paragraph charmap — no
regex matching needed.

Usage:
    python scripts/apply_auto_splits.py --dry-run          # preview
    python scripts/apply_auto_splits.py --dry-run --stats   # just counts
    python scripts/apply_auto_splits.py                     # apply
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import REPO_DIR

EXPORT_DIR = REPO_DIR / "data" / "export"
EMBEDDINGS_DIR = REPO_DIR / "data" / "embeddings"
FIXES_PATH = REPO_DIR / "data" / "proposed_fixes.jsonl"

EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]


def load_paragraph_charmap(edition_year):
    """Load paragraph char_start positions for one edition.

    Returns: {article_id: {para_index: char_start}}
    char_start is relative to the article text.
    """
    fp = EMBEDDINGS_DIR / f"eb_{edition_year}.paragraphs.jsonl"
    if not fp.exists():
        return {}

    charmap = defaultdict(dict)
    with open(fp) as f:
        for line in f:
            rec = json.loads(line)
            charmap[rec["article_id"]][rec["para_index"]] = rec["char_start"]
    return charmap


def split_at_char(article, split_points):
    """Split an article at exact character positions.

    split_points: list of (char_pos, new_title) sorted by position.
    char_pos is relative to the article text.
    Returns list of new article dicts.
    """
    text = article["text"]
    result = []

    # Snap each split point to nearest paragraph boundary (\n\n)
    adjusted = []
    for char_pos, new_title in split_points:
        boundary = text.rfind("\n\n", 0, char_pos + 10)  # small forward tolerance
        if boundary >= 0 and abs(boundary - char_pos) < 200:
            pos = boundary + 2
        else:
            pos = char_pos
        if pos > 0 and (not adjusted or pos > adjusted[-1][0]):
            adjusted.append((pos, new_title))

    if not adjusted:
        return [article]

    prev_pos = 0
    prev_title = article["title"]

    for pos, new_title in adjusted:
        chunk_text = text[prev_pos:pos].strip()
        if chunk_text:
            art = dict(article)
            art["title"] = prev_title
            art["text"] = chunk_text
            art["word_count"] = len(chunk_text.split())
            art["char_start"] = article["char_start"] + prev_pos
            art["char_end"] = article["char_start"] + pos
            art["article_id"] = (
                f"{article['article_id']}_{len(result)}" if result
                else article["article_id"]
            )
            art["heading_pattern"] = "auto_split"
            result.append(art)
        prev_pos = pos
        prev_title = new_title

    # Final chunk
    chunk_text = text[prev_pos:].strip()
    if chunk_text:
        art = dict(article)
        art["title"] = prev_title
        art["text"] = chunk_text
        art["word_count"] = len(chunk_text.split())
        art["char_start"] = article["char_start"] + prev_pos
        art["char_end"] = article["char_end"]
        art["article_id"] = f"{article['article_id']}_{len(result)}"
        art["heading_pattern"] = "auto_split"
        result.append(art)

    return result


def load_fixes(min_alpha_score=4):
    """Load fix specs grouped by (edition_year, article_id)."""
    fixes = []
    with open(FIXES_PATH) as f:
        for line in f:
            spec = json.loads(line)
            if spec.get("alpha_score", 0) >= min_alpha_score:
                fixes.append(spec)

    grouped = defaultdict(list)
    for spec in fixes:
        key = (spec["edition_year"], spec["article_id"])
        grouped[key].append(spec)

    return grouped


def find_export_file(edition_year):
    matches = list(EXPORT_DIR.glob(f"eb_*_{edition_year}.jsonl"))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Apply auto-detected article splits")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--min-alpha", type=int, default=4)
    parser.add_argument("--edition-year", type=int)
    args = parser.parse_args()

    grouped = load_fixes(args.min_alpha)

    if args.edition_year:
        grouped = {k: v for k, v in grouped.items() if k[0] == args.edition_year}

    total_specs = sum(len(v) for v in grouped.values())
    print(f"Loaded {total_specs:,} fix specs across {len(grouped):,} parent articles")

    if args.stats:
        from collections import Counter
        by_year = Counter()
        for (year, _), specs in grouped.items():
            by_year[year] += len(specs)
        print("\nBy edition:")
        for y in sorted(by_year):
            print(f"  {y}: {by_year[y]}")
        return

    years = sorted(set(k[0] for k in grouped))
    total_splits = 0
    total_failed = 0

    for year in years:
        print(f"\n{'='*60}")
        print(f"Edition {year}")
        print(f"{'='*60}")

        export_file = find_export_file(year)
        if not export_file:
            print(f"  WARNING: No export file for {year}")
            continue

        charmap = load_paragraph_charmap(year)
        print(f"  Charmap: {len(charmap):,} articles")

        # Load all articles from export
        articles = []
        with open(export_file) as f:
            for line in f:
                articles.append(json.loads(line))
        print(f"  Articles: {len(articles):,}")

        # Index by article_id for fast lookup
        art_index = {a["article_id"]: i for i, a in enumerate(articles)}

        # Get fixes for this year, sorted by article position (reverse to
        # avoid index shifting when we insert)
        year_fixes = sorted(
            [(k, v) for k, v in grouped.items() if k[0] == year],
            key=lambda x: art_index.get(x[0][1], 0),
            reverse=True,
        )

        edition_splits = 0
        for (_, article_id), specs in year_fixes:
            if article_id not in art_index:
                total_failed += len(specs)
                continue

            idx = art_index[article_id]
            art = articles[idx]
            art_cm = charmap.get(article_id, {})

            # Build split points from paragraph char positions
            split_points = []
            for spec in sorted(specs, key=lambda s: int(s["para_break"].split("→")[1])):
                para_after = int(spec["para_break"].split("→")[1])
                new_title = spec["break_headword"]

                if para_after in art_cm:
                    split_points.append((art_cm[para_after], new_title))
                else:
                    # Fallback: estimate position
                    total = spec.get("total_paras", 1)
                    if total > 0:
                        est = int(len(art["text"]) * para_after / total)
                        split_points.append((est, new_title))
                    else:
                        total_failed += 1

            if not split_points:
                continue

            split_points.sort(key=lambda x: x[0])
            new_articles = split_at_char(art, split_points)

            if len(new_articles) <= 1:
                total_failed += len(specs)
                continue

            parent_wc = art["word_count"]
            pieces = [(a["title"], a["word_count"]) for a in new_articles]
            pieces_str = " + ".join(f"{t}({wc:,}w)" for t, wc in pieces)
            print(f"  SPLIT: {art['title']} ({parent_wc:,}w) → {pieces_str}")
            edition_splits += len(new_articles) - 1

            # Replace in-place
            articles[idx:idx + 1] = new_articles
            # Update index for subsequent lookups (shifted by insertion)
            art_index = {a["article_id"]: i for i, a in enumerate(articles)}

        total_splits += edition_splits
        print(f"  Edition splits: {edition_splits}")

        if edition_splits > 0 and not args.dry_run:
            with open(export_file, "w") as f:
                for a in articles:
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
            print(f"  Wrote {len(articles):,} articles to {export_file.name}")

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Successful splits: {total_splits}")
    print(f"  Failed: {total_failed}")


if __name__ == "__main__":
    main()
