#!/usr/bin/env python3
"""
Analyze text coverage within each volume to detect gaps and overlaps.

Uses the article_index.jsonl to:
- Build page-by-page coverage maps
- Identify pages with no articles (gaps)
- Identify pages with multiple articles (expected for multi-column)
- Flag unusual patterns that might indicate parsing issues

Since multi-column layouts mean multiple articles per page is normal,
we focus on:
1. Page sequence continuity (missing page numbers)
2. Character density anomalies (unusually sparse/dense pages)
3. Alphabetical sequence breaks (articles out of order)
"""

import json
from collections import defaultdict
from pathlib import Path


def load_article_index(index_path: Path) -> list[dict]:
    """Load article index from JSONL file."""
    articles = []
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def analyze_volume(articles: list[dict]) -> dict:
    """
    Analyze coverage for a single volume.

    Returns dict with coverage statistics and anomalies.
    """
    if not articles:
        return {"error": "no_articles"}

    # Sort by page order (char_start)
    sorted_articles = sorted(articles, key=lambda a: a["boundaries"]["char_start"])

    # Basic stats
    total_chars = articles[0]["source"]["volume_total_chars"]
    page_numbers = [a["boundaries"]["start_page"] for a in articles if a["boundaries"]["start_page"]]
    min_page = min(page_numbers) if page_numbers else None
    max_page = max(page_numbers) if page_numbers else None

    # Page coverage map: page -> list of articles
    page_coverage = defaultdict(list)
    for a in articles:
        sp = a["boundaries"]["start_page"]
        ep = a["boundaries"]["end_page"]
        if sp is not None:
            for p in range(sp, (ep or sp) + 1):
                page_coverage[p].append(a["article_id"])

    # Find page gaps (missing page numbers in sequence)
    if min_page and max_page:
        all_pages = set(range(min_page, max_page + 1))
        covered_pages = set(page_coverage.keys())
        missing_pages = sorted(all_pages - covered_pages)
    else:
        missing_pages = []

    # Articles per page distribution
    articles_per_page = {p: len(arts) for p, arts in page_coverage.items()}
    if articles_per_page:
        avg_per_page = sum(articles_per_page.values()) / len(articles_per_page)
        max_per_page = max(articles_per_page.values())
        pages_with_many = [p for p, c in articles_per_page.items() if c > 10]
    else:
        avg_per_page = 0
        max_per_page = 0
        pages_with_many = []

    # Check alphabetical order within volume
    # Articles should generally be in alphabetical order
    alpha_breaks = []
    for i in range(1, len(sorted_articles)):
        prev = sorted_articles[i-1]["headword_normalized"]
        curr = sorted_articles[i]["headword_normalized"]
        # Only flag if both are green quality and significantly out of order
        if (sorted_articles[i-1]["quality_flag"] == "green" and
            sorted_articles[i]["quality_flag"] == "green"):
            if prev > curr and prev[:3] > curr[:3]:  # First 3 chars differ
                alpha_breaks.append({
                    "position": i,
                    "prev": sorted_articles[i-1]["headword"],
                    "curr": sorted_articles[i]["headword"],
                    "prev_page": sorted_articles[i-1]["boundaries"]["start_page"],
                    "curr_page": sorted_articles[i]["boundaries"]["start_page"]
                })

    # Quality distribution
    quality_counts = defaultdict(int)
    for a in articles:
        quality_counts[a["quality_flag"]] += 1

    # Red flag articles (parsing errors)
    red_articles = [
        {
            "article_id": a["article_id"],
            "headword": a["headword"],
            "issues": a["issues"],
            "page": a["boundaries"]["start_page"]
        }
        for a in sorted_articles
        if a["quality_flag"] == "red"
    ]

    return {
        "article_count": len(articles),
        "total_chars": total_chars,
        "page_range": {"min": min_page, "max": max_page},
        "pages_covered": len(page_coverage),
        "missing_pages": missing_pages[:20] if len(missing_pages) > 20 else missing_pages,
        "missing_pages_count": len(missing_pages),
        "avg_articles_per_page": round(avg_per_page, 2),
        "max_articles_per_page": max_per_page,
        "pages_with_many_articles": pages_with_many[:10],
        "alphabetical_breaks": alpha_breaks[:10],
        "alphabetical_breaks_count": len(alpha_breaks),
        "quality_distribution": dict(quality_counts),
        "red_articles": red_articles[:20],
        "red_articles_count": len(red_articles)
    }


def main():
    index_path = Path(__file__).parent.parent / "article_index.jsonl"
    output_path = Path(__file__).parent.parent / "coverage_report.json"

    print("Loading article index...")
    articles = load_article_index(index_path)
    print(f"  Loaded {len(articles):,} articles")

    # Group by edition and volume
    by_edition_volume = defaultdict(list)
    for a in articles:
        key = (a["source"]["edition"], a["source"]["volume"])
        by_edition_volume[key].append(a)

    print(f"\nAnalyzing {len(by_edition_volume)} volumes...")
    print("=" * 70)

    report = {
        "total_articles": len(articles),
        "editions": {},
        "issues_summary": {
            "total_missing_pages": 0,
            "total_alpha_breaks": 0,
            "total_red_articles": 0
        }
    }

    for (edition, volume), vol_articles in sorted(by_edition_volume.items()):
        if edition not in report["editions"]:
            report["editions"][edition] = {"volumes": {}, "totals": {
                "articles": 0, "chars": 0, "missing_pages": 0,
                "alpha_breaks": 0, "red_articles": 0
            }}

        analysis = analyze_volume(vol_articles)
        report["editions"][edition]["volumes"][volume] = analysis

        # Update totals
        totals = report["editions"][edition]["totals"]
        totals["articles"] += analysis["article_count"]
        totals["chars"] += analysis["total_chars"]
        totals["missing_pages"] += analysis["missing_pages_count"]
        totals["alpha_breaks"] += analysis["alphabetical_breaks_count"]
        totals["red_articles"] += analysis["red_articles_count"]

        report["issues_summary"]["total_missing_pages"] += analysis["missing_pages_count"]
        report["issues_summary"]["total_alpha_breaks"] += analysis["alphabetical_breaks_count"]
        report["issues_summary"]["total_red_articles"] += analysis["red_articles_count"]

        # Print summary for volume
        issues = []
        if analysis["missing_pages_count"] > 0:
            issues.append(f"{analysis['missing_pages_count']} missing pages")
        if analysis["alphabetical_breaks_count"] > 0:
            issues.append(f"{analysis['alphabetical_breaks_count']} alpha breaks")
        if analysis["red_articles_count"] > 0:
            issues.append(f"{analysis['red_articles_count']} red articles")

        issue_str = ", ".join(issues) if issues else "OK"
        print(f"  {edition} {volume}: {analysis['article_count']:,} articles, "
              f"pages {analysis['page_range']['min']}-{analysis['page_range']['max']} "
              f"[{issue_str}]")

    # Print summary
    print("\n" + "=" * 70)
    print("COVERAGE ANALYSIS SUMMARY")
    print("=" * 70)

    for edition in sorted(report["editions"].keys()):
        totals = report["editions"][edition]["totals"]
        print(f"\n{edition} Edition:")
        print(f"  Articles: {totals['articles']:,}")
        print(f"  Characters: {totals['chars']:,}")
        print(f"  Missing pages: {totals['missing_pages']}")
        print(f"  Alphabetical breaks: {totals['alpha_breaks']}")
        print(f"  Red (parsing error) articles: {totals['red_articles']}")

    print(f"\nGlobal Issues:")
    print(f"  Total missing pages: {report['issues_summary']['total_missing_pages']}")
    print(f"  Total alphabetical breaks: {report['issues_summary']['total_alpha_breaks']}")
    print(f"  Total red articles: {report['issues_summary']['total_red_articles']}")

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {output_path}")


if __name__ == "__main__":
    main()
