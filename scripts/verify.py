"""Verification and quality checks for parsed Britannica articles."""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from config import INPUT_DIR, ARTICLES_DIR, PARAGRAPHS_DIR, CLASSIFICATIONS_DIR, ensure_dirs

log = logging.getLogger(__name__)


def load_articles(path: Path) -> list[dict]:
    articles = []
    with open(path) as f:
        for line in f:
            articles.append(json.loads(line))
    return articles


def check_article_counts(articles_dir: Path) -> dict:
    """Check article counts per file."""
    results = {}
    for path in sorted(articles_dir.glob("*.articles.jsonl")):
        articles = load_articles(path)
        real_articles = [a for a in articles if a["type"] == "article"]
        cross_refs = [a for a in articles if a["type"] == "cross_reference"]
        results[path.stem.replace(".articles", "")] = {
            "total": len(articles),
            "articles": len(real_articles),
            "cross_references": len(cross_refs),
            "front_matter": sum(1 for a in articles if a["type"] == "front_matter"),
            "back_matter": sum(1 for a in articles if a["type"] == "back_matter"),
        }
    return results


def check_alphabetical_order(articles_path: Path) -> list[str]:
    """Check if article titles are roughly alphabetical."""
    articles = load_articles(articles_path)
    real_articles = [a for a in articles if a["type"] in ("article", "cross_reference")]

    issues = []
    for i in range(1, len(real_articles)):
        prev = real_articles[i - 1]["title"].upper()
        curr = real_articles[i]["title"].upper()
        # Allow some tolerance — OCR and parsing imperfections
        if curr < prev and prev[:2] != curr[:2]:
            issues.append(
                f"  Out of order: '{real_articles[i-1]['title']}' (#{i-1}) "
                f"> '{real_articles[i]['title']}' (#{i})"
            )
    return issues


def check_text_coverage(input_path: Path, articles_path: Path) -> dict:
    """Check what fraction of original text is covered by articles."""
    with open(input_path) as f:
        meta = json.loads(f.readline())
    total_chars = len(meta["text"])

    articles = load_articles(articles_path)
    article_chars = sum(a["char_end"] - a["char_start"] for a in articles)
    body_chars = sum(
        a["char_end"] - a["char_start"]
        for a in articles if a["type"] in ("article", "cross_reference")
    )

    return {
        "total_chars": total_chars,
        "article_chars": article_chars,
        "body_chars": body_chars,
        "coverage_pct": round(100 * article_chars / total_chars, 1) if total_chars else 0,
        "body_coverage_pct": round(100 * body_chars / total_chars, 1) if total_chars else 0,
    }


def check_classification_distribution(cls_path: Path) -> dict:
    """Summarize classification type distribution."""
    counts = defaultdict(int)
    with open(cls_path) as f:
        for line in f:
            cls = json.loads(line)
            counts[cls["type"]] += 1
    return dict(counts)


def check_running_headers(articles_path: Path, cls_path: Path) -> dict:
    """Check running header removal stats."""
    articles = load_articles(articles_path)

    # Count running headers from classifications
    header_count = 0
    with open(cls_path) as f:
        for line in f:
            cls = json.loads(line)
            if cls["type"] == "running_header":
                header_count += 1

    # Check for potential remaining headers in article text
    # (short ALL-CAPS paragraphs that might have been missed)
    suspicious = 0
    for a in articles:
        if a["type"] != "article":
            continue
        for para in a["text"].split("\n\n"):
            stripped = para.strip()
            if (len(stripped) < 30
                    and stripped == stripped.upper()
                    and re.match(r'^[A-Z][A-Z\s]+$', stripped)):
                suspicious += 1

    return {
        "headers_removed": header_count,
        "suspicious_remaining": suspicious,
    }


def check_cross_references(articles_dir: Path) -> dict:
    """Check if cross-reference targets exist within the same edition."""
    # Group articles by edition
    editions = defaultdict(set)  # edition -> set of titles
    cross_refs = defaultdict(list)  # edition -> list of (title, target)

    for path in sorted(articles_dir.glob("*.articles.jsonl")):
        articles = load_articles(path)
        for a in articles:
            edition = a["edition"]
            if a["type"] == "article":
                editions[edition].add(a["title"].upper())
            elif a["type"] == "cross_reference" and a.get("target"):
                cross_refs[edition].append((a["title"], a["target"]))

    results = {}
    for edition in sorted(editions.keys()):
        titles = editions[edition]
        refs = cross_refs.get(edition, [])
        found = sum(1 for _, target in refs if target.upper() in titles)
        results[edition] = {
            "cross_references": len(refs),
            "targets_found": found,
            "targets_missing": len(refs) - found,
        }
    return results


def run_all_checks(files: list[Path] | None = None) -> dict:
    """Run all verification checks."""
    ensure_dirs()

    if files is None:
        files = sorted(INPUT_DIR.glob("*.jsonl"))

    report = {
        "summary": {},
        "per_file": {},
    }

    total_articles = 0
    total_words = 0
    edition_counts = defaultdict(int)

    for input_path in files:
        stem = input_path.stem
        articles_path = ARTICLES_DIR / f"{stem}.articles.jsonl"
        cls_path = CLASSIFICATIONS_DIR / f"{stem}.classifications.jsonl"

        if not articles_path.exists():
            log.warning(f"No articles for {stem}")
            continue

        articles = load_articles(articles_path)
        n_articles = len([a for a in articles if a["type"] == "article"])
        n_words = sum(a["word_count"] for a in articles)
        total_articles += n_articles
        total_words += n_words

        # Extract edition from metadata
        with open(input_path) as f:
            meta = json.loads(f.readline())
        edition_counts[meta["edition_name"]] += n_articles

        file_report = {
            "articles": n_articles,
            "words": n_words,
        }

        # Coverage
        coverage = check_text_coverage(input_path, articles_path)
        file_report["coverage"] = coverage

        # Alphabetical order
        order_issues = check_alphabetical_order(articles_path)
        file_report["order_issues"] = len(order_issues)
        if order_issues:
            file_report["order_examples"] = order_issues[:5]

        # Classification distribution
        if cls_path.exists():
            file_report["classifications"] = check_classification_distribution(cls_path)

            # Running headers
            file_report["headers"] = check_running_headers(articles_path, cls_path)

        report["per_file"][stem] = file_report

    # Cross-reference check (edition-wide)
    report["cross_references"] = check_cross_references(ARTICLES_DIR)

    # Summary
    report["summary"] = {
        "files_processed": len(report["per_file"]),
        "total_articles": total_articles,
        "total_words": total_words,
        "articles_per_edition": dict(edition_counts),
    }

    return report


def print_report(report: dict):
    """Print verification report."""
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"BRITANNICA PARSER VERIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Files processed: {s['files_processed']}")
    print(f"Total articles:  {s['total_articles']:,}")
    print(f"Total words:     {s['total_words']:,}")
    print(f"\nArticles per edition:")
    for ed, count in sorted(s.get("articles_per_edition", {}).items()):
        print(f"  {ed}: {count:,}")

    # Coverage summary
    coverages = [
        f["coverage"]["coverage_pct"]
        for f in report["per_file"].values()
        if "coverage" in f
    ]
    if coverages:
        avg = sum(coverages) / len(coverages)
        print(f"\nAverage text coverage: {avg:.1f}%")

    # Order issues
    total_order_issues = sum(
        f.get("order_issues", 0) for f in report["per_file"].values()
    )
    print(f"Alphabetical order issues: {total_order_issues}")

    # Running headers
    total_removed = sum(
        f.get("headers", {}).get("headers_removed", 0)
        for f in report["per_file"].values()
    )
    total_suspicious = sum(
        f.get("headers", {}).get("suspicious_remaining", 0)
        for f in report["per_file"].values()
    )
    print(f"Running headers removed: {total_removed:,}")
    print(f"Suspicious remaining:    {total_suspicious:,}")

    # Cross-references
    xrefs = report.get("cross_references", {})
    if xrefs:
        print(f"\nCross-reference targets:")
        for ed, info in sorted(xrefs.items()):
            print(f"  {ed}: {info['targets_found']}/{info['cross_references']} found")

    print(f"{'='*60}\n")

    # Per-file issues
    problem_files = [
        (stem, f) for stem, f in report["per_file"].items()
        if f.get("order_issues", 0) > 5
        or f.get("coverage", {}).get("coverage_pct", 100) < 80
    ]
    if problem_files:
        print("FILES WITH POTENTIAL ISSUES:")
        for stem, f in problem_files:
            issues = []
            if f.get("order_issues", 0) > 5:
                issues.append(f"{f['order_issues']} order issues")
            cov = f.get("coverage", {}).get("coverage_pct", 100)
            if cov < 80:
                issues.append(f"{cov}% coverage")
            print(f"  {stem}: {', '.join(issues)}")
        print()


def run(files: list[Path] | None = None):
    """Run verification and print report."""
    report = run_all_checks(files)
    print_report(report)

    # Save JSON report
    report_path = ARTICLES_DIR / "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {report_path}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
