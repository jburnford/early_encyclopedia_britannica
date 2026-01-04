#!/usr/bin/env python3
"""
Encyclopedia Britannica Article Cleanup Script

Identifies and removes problematic articles from the parsed corpus.

Usage:
    python3 cleanup_articles.py --audit     # Read-only audit, produces reports
    python3 cleanup_articles.py --fix       # Write cleaned JSONL files

Options:
    --editions YEARS    Comma-separated list of edition years (default: all)
    --output-dir DIR    Output directory for cleaned files (default: output_v2)
    --report-dir DIR    Output directory for audit reports (default: reports)
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from encyclopedia_parser.article_validator import (
    ArticleValidator,
    ValidationResult,
    IssueType,
    IssueSeverity,
)


# All edition years in the corpus
ALL_EDITIONS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]


@dataclass
class CleanupStats:
    """Statistics from cleaning an edition."""
    edition_year: int
    total_articles: int = 0
    kept: int = 0
    removed: int = 0
    flagged: int = 0
    by_issue_type: dict = field(default_factory=lambda: defaultdict(int))
    removed_examples: list = field(default_factory=list)
    flagged_examples: list = field(default_factory=list)


def load_articles(filepath: Path) -> Iterator[dict]:
    """Stream articles from a JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue


def write_articles(articles: list[dict], filepath: Path):
    """Write articles to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


def cleanup_edition(
    input_file: Path,
    output_file: Path | None,
    validator: ArticleValidator,
    edition_year: int,
    dry_run: bool = True
) -> CleanupStats:
    """
    Process one edition's articles.

    Args:
        input_file: Path to articles_YYYY.jsonl
        output_file: Path for cleaned output (or None for audit mode)
        validator: ArticleValidator instance
        edition_year: Year of this edition
        dry_run: If True, don't write output

    Returns:
        CleanupStats with results
    """
    stats = CleanupStats(edition_year=edition_year)
    cleaned_articles = []

    for article in load_articles(input_file):
        stats.total_articles += 1
        result = validator.validate(article)

        if result.action == "keep":
            stats.kept += 1
            cleaned_articles.append(article)

        elif result.action == "remove":
            stats.removed += 1
            for issue in result.issues:
                stats.by_issue_type[issue.issue_type.value] += 1

            # Save example (first 20 per type)
            if len(stats.removed_examples) < 50:
                stats.removed_examples.append({
                    "headword": article.get("headword", "")[:80],
                    "issues": [i.reason for i in result.issues],
                    "text_preview": article.get("text", "")[:100]
                })

        elif result.action == "flag":
            stats.flagged += 1
            for issue in result.issues:
                stats.by_issue_type[issue.issue_type.value] += 1

            # Add flag to article
            article["needs_review"] = True
            article["issues"] = [i.to_dict() for i in result.issues]
            cleaned_articles.append(article)

            # Save example
            if len(stats.flagged_examples) < 30:
                stats.flagged_examples.append({
                    "headword": article.get("headword", "")[:80],
                    "issues": [i.reason for i in result.issues]
                })

    # Write output if not dry run
    if not dry_run and output_file:
        write_articles(cleaned_articles, output_file)
        print(f"  Wrote {len(cleaned_articles)} articles to {output_file}")

    return stats


def generate_audit_report(stats: CleanupStats, report_file: Path):
    """Generate a markdown audit report for one edition."""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Cleanup Audit Report: {stats.edition_year} Edition\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"| Metric | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| Total Articles | {stats.total_articles:,} | 100% |\n")
        f.write(f"| Keep (clean) | {stats.kept:,} | {100*stats.kept/stats.total_articles:.1f}% |\n")
        f.write(f"| Remove | {stats.removed:,} | {100*stats.removed/stats.total_articles:.1f}% |\n")
        f.write(f"| Flag for Review | {stats.flagged:,} | {100*stats.flagged/stats.total_articles:.1f}% |\n")
        f.write("\n")

        # By issue type
        if stats.by_issue_type:
            f.write("## Issues by Type\n\n")
            f.write("| Issue Type | Count |\n")
            f.write("|------------|-------|\n")
            for issue_type, count in sorted(stats.by_issue_type.items(), key=lambda x: -x[1]):
                f.write(f"| {issue_type} | {count:,} |\n")
            f.write("\n")

        # Removed examples
        if stats.removed_examples:
            f.write("## Articles to Remove (Examples)\n\n")
            for i, example in enumerate(stats.removed_examples[:30], 1):
                f.write(f"### {i}. `{example['headword']}`\n\n")
                f.write(f"**Issues:** {', '.join(example['issues'])}\n\n")
                if example['text_preview']:
                    f.write(f"**Preview:** {example['text_preview']}...\n\n")

        # Flagged examples
        if stats.flagged_examples:
            f.write("## Articles Flagged for Review (Examples)\n\n")
            for i, example in enumerate(stats.flagged_examples[:20], 1):
                f.write(f"### {i}. `{example['headword']}`\n\n")
                f.write(f"**Issues:** {', '.join(example['issues'])}\n\n")

    print(f"  Report written to {report_file}")


def generate_summary_report(all_stats: list[CleanupStats], report_file: Path):
    """Generate a summary report across all editions."""
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Cleanup Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall stats
        total = sum(s.total_articles for s in all_stats)
        kept = sum(s.kept for s in all_stats)
        removed = sum(s.removed for s in all_stats)
        flagged = sum(s.flagged for s in all_stats)

        f.write("## Overall Summary\n\n")
        f.write(f"| Metric | Count | Percentage |\n")
        f.write(f"|--------|-------|------------|\n")
        f.write(f"| Total Articles | {total:,} | 100% |\n")
        f.write(f"| Keep (clean) | {kept:,} | {100*kept/total:.1f}% |\n")
        f.write(f"| Remove | {removed:,} | {100*removed/total:.1f}% |\n")
        f.write(f"| Flag for Review | {flagged:,} | {100*flagged/total:.1f}% |\n")
        f.write("\n")

        # Per edition
        f.write("## By Edition\n\n")
        f.write("| Edition | Total | Keep | Remove | Flag |\n")
        f.write("|---------|-------|------|--------|------|\n")
        for s in all_stats:
            f.write(f"| {s.edition_year} | {s.total_articles:,} | {s.kept:,} | {s.removed:,} | {s.flagged:,} |\n")
        f.write("\n")

        # Aggregate issues
        all_issues = defaultdict(int)
        for s in all_stats:
            for issue_type, count in s.by_issue_type.items():
                all_issues[issue_type] += count

        f.write("## Issues by Type (All Editions)\n\n")
        f.write("| Issue Type | Count |\n")
        f.write("|------------|-------|\n")
        for issue_type, count in sorted(all_issues.items(), key=lambda x: -x[1]):
            f.write(f"| {issue_type} | {count:,} |\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write(f"1. **{removed:,} articles** will be removed as they are structural markers or parsing errors\n")
        f.write(f"2. **{flagged:,} articles** are flagged for manual review (very long, short, or out-of-range)\n")
        f.write(f"3. Run with `--fix` to generate cleaned JSONL files\n")
        f.write(f"4. After fixing, regenerate the site with `python3 generate_site_optimized.py`\n")

    print(f"Summary report written to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean problematic articles from Encyclopedia Britannica corpus"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Audit mode: produce reports without modifying files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix mode: write cleaned JSONL files"
    )
    parser.add_argument(
        "--editions",
        type=str,
        default=None,
        help="Comma-separated list of edition years (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_v2",
        help="Output directory for cleaned files"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports",
        help="Output directory for audit reports"
    )

    args = parser.parse_args()

    if not args.audit and not args.fix:
        print("Error: Must specify --audit or --fix")
        parser.print_help()
        sys.exit(1)

    # Determine editions to process
    if args.editions:
        editions = [int(y.strip()) for y in args.editions.split(",")]
    else:
        editions = ALL_EDITIONS

    # Setup paths
    base_dir = Path(__file__).parent
    input_dir = base_dir / args.output_dir
    output_dir = base_dir / args.output_dir
    report_dir = base_dir / args.report_dir
    report_dir.mkdir(exist_ok=True)

    # Create validator
    validator = ArticleValidator(
        min_text_length=20,
        max_headword_length=60,
        max_text_length=500_000,
        removal_confidence_threshold=0.95,
        flag_confidence_threshold=0.70
    )

    print(f"{'Audit' if args.audit else 'Fix'} mode: Processing {len(editions)} editions")
    print()

    all_stats = []

    for edition_year in editions:
        input_file = input_dir / f"articles_{edition_year}.jsonl"

        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        print(f"Processing {edition_year} edition...")

        # Determine output file
        if args.fix:
            output_file = output_dir / f"articles_{edition_year}_cleaned.jsonl"
        else:
            output_file = None

        # Process
        stats = cleanup_edition(
            input_file=input_file,
            output_file=output_file,
            validator=validator,
            edition_year=edition_year,
            dry_run=args.audit
        )

        all_stats.append(stats)

        # Generate per-edition report
        report_file = report_dir / f"cleanup_audit_{edition_year}.md"
        generate_audit_report(stats, report_file)

        print(f"  Total: {stats.total_articles:,} | Keep: {stats.kept:,} | Remove: {stats.removed:,} | Flag: {stats.flagged:,}")
        print()

    # Generate summary report
    summary_file = report_dir / "CLEANUP_SUMMARY.md"
    generate_summary_report(all_stats, summary_file)

    print("Done!")


if __name__ == "__main__":
    main()
