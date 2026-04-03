#!/usr/bin/env python3
"""Flag anomalous articles that violate alphabetical, volume, or naming logic.

Reads per-letter HTML index pages to extract (title, volume, word_count),
then applies three checks:
  1. Volume consistency — articles in wrong volume for their letter
  2. Alphabetical order — articles out of sort order
  3. Known bad patterns — section headings, back matter, publisher artifacts

Usage:
    python scripts/flag_anomalies.py
    python scripts/flag_anomalies.py --edition-year 1815
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SITE_DIR = REPO / "docs" / "articles"
EXPORT_DIR = REPO / "data" / "export"
OUTPUT_JSONL = REPO / "data" / "anomalous_articles.jsonl"
OUTPUT_MD = REPO / "data" / "anomalous_articles.md"

YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# Known bad title patterns
BAD_PATTERNS = [
    (re.compile(r'^AND SONS', re.I), "publisher_artifact"),
    (re.compile(r'^END OF THE', re.I), "volume_boundary"),
    (re.compile(r'^SECT\.\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^CHAP\.\s', re.I), "section_heading"),
    (re.compile(r'^CHAP\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^ORDER\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^PART\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^GENUS\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^DIRECTIONS\s+FOR', re.I), "back_matter"),
    (re.compile(r'^EXPLANATION\s+OF\s+PLATE', re.I), "back_matter"),
    (re.compile(r'^ERRATA', re.I), "back_matter"),
    (re.compile(r'^FINIS$', re.I), "back_matter"),
    (re.compile(r'^ANTISTROPHE\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^EPISODE\s+[IVX\d]', re.I), "section_heading"),
    (re.compile(r'^STROPHE\s+[IVX\d]', re.I), "section_heading"),
]


def parse_letter_page(html_path):
    """Extract (article_id, title, volume, word_count) from a letter HTML page."""
    text = html_path.read_text()
    articles = []
    # Pattern: <tr><td><a href="ARTICLE_ID.html">TITLE</a></td><td>VOL</td><td>WC</td></tr>
    for m in re.finditer(
        r'<tr><td><a href="([^"]+)\.html">([^<]+)</a></td><td>([^<]+)</td><td>([^<]+)</td></tr>',
        text
    ):
        aid = m.group(1)
        title = m.group(2).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        try:
            vol = int(m.group(3))
        except ValueError:
            vol = 0
        wc_str = m.group(4).replace(",", "")
        try:
            wc = int(wc_str)
        except ValueError:
            wc = 0
        articles.append({"article_id": aid, "title": title, "volume": vol, "word_count": wc})
    return articles


def check_volume_consistency(articles, letter, year):
    """Flag articles in wrong volume for their letter."""
    if len(articles) < 3:
        return []

    # Find dominant volume(s) — the volume(s) that contain most articles
    # 1810 4th edition is a supplement — volumes are NOT alphabetically ordered
    # Each supplement volume has mixed topics, so volume checks don't apply
    if year == 1810:
        return []

    # Exclude supplement volumes (500+) from dominance calculation — they
    # legitimately contain articles across all letters
    regular_vols = [a["volume"] for a in articles if a["volume"] < 500]
    if not regular_vols:
        return []  # all supplement articles — nothing to flag

    vol_counts = Counter(regular_vols)
    total = len(regular_vols)

    # Dominant = regular volumes that collectively hold >= 80% of regular articles
    dominant = set()
    cumulative = 0
    for vol, count in vol_counts.most_common():
        dominant.add(vol)
        cumulative += count
        if cumulative >= total * 0.8:
            break

    # Also include volumes adjacent to dominant (for letter boundaries)
    adjacent = set()
    for v in dominant:
        adjacent.add(v - 1)
        adjacent.add(v + 1)
    allowed = dominant | adjacent
    # Supplement volumes (500+) are always allowed
    supplement_vols = {a["volume"] for a in articles if a["volume"] >= 500}
    allowed |= supplement_vols

    flags = []
    for a in articles:
        if a["volume"] not in allowed and a["volume"] != 0:
            flags.append({
                **a,
                "edition_year": year,
                "letter": letter,
                "flag": "wrong_volume",
                "detail": f"vol {a['volume']} not in expected {sorted(dominant)}",
            })
    return flags


def check_alphabetical_order(articles, letter, year):
    """Flag articles that are clearly in the wrong alphabetical position.

    Only flags articles where the first letter doesn't match the page letter,
    or the first TWO chars are completely different from neighbors (not just
    minor sort variations like AB-INTESTATE after ABINGDON).
    """
    flags = []
    for i in range(len(articles)):
        title = articles[i]["title"].upper()
        if not title:
            continue

        # Check: does the first letter match the page letter?
        first_char = title[0]
        if first_char != letter.upper() and letter != "#":
            flags.append({
                **articles[i],
                "edition_year": year,
                "letter": letter,
                "flag": "wrong_letter",
                "detail": f"'{articles[i]['title']}' starts with {first_char}, expected {letter}",
            })
    return flags


def check_bad_patterns(articles, letter, year):
    """Flag articles matching known problematic title patterns."""
    flags = []
    for a in articles:
        title = a["title"]
        for pattern, flag_type in BAD_PATTERNS:
            if pattern.search(title):
                flags.append({
                    **a,
                    "edition_year": year,
                    "letter": letter,
                    "flag": flag_type,
                    "detail": title,
                })
                break  # one flag per article

        # Very short articles (likely OCR noise or truncated cross-refs)
        if a["word_count"] <= 2 and not any(p.search(title) for p, _ in BAD_PATTERNS):
            flags.append({
                **a,
                "edition_year": year,
                "letter": letter,
                "flag": "tiny_article",
                "detail": f"{a['word_count']}w",
            })
    return flags


def write_report(all_flags):
    """Write markdown report."""
    by_flag = defaultdict(list)
    for f in all_flags:
        by_flag[f["flag"]].append(f)

    by_year = Counter(f["edition_year"] for f in all_flags)

    lines = [
        "# Anomalous Articles Report",
        "",
        f"**Total flagged:** {len(all_flags)}",
        "",
        "## By Flag Type",
        "",
        "| Flag | Count | Description |",
        "|------|-------|-------------|",
    ]
    descriptions = {
        "wrong_volume": "Article in unexpected volume for its letter",
        "out_of_order": "Article breaks alphabetical sort order",
        "publisher_artifact": "'AND SONS' or similar publisher text",
        "volume_boundary": "'END OF THE ... VOLUME' marker",
        "section_heading": "SECT./CHAP./ORDER/PART parsed as article",
        "back_matter": "Plate directions, errata, etc.",
        "tiny_article": "Article with <= 2 words",
    }
    for flag in sorted(by_flag, key=lambda f: -len(by_flag[f])):
        count = len(by_flag[flag])
        desc = descriptions.get(flag, flag)
        lines.append(f"| {flag} | {count} | {desc} |")

    lines += ["", "## By Edition", ""]
    for y in sorted(by_year):
        lines.append(f"- **{y}**: {by_year[y]} flagged")

    for flag in ["wrong_volume", "section_heading", "publisher_artifact",
                  "volume_boundary", "back_matter", "out_of_order", "tiny_article"]:
        items = by_flag.get(flag, [])
        if not items:
            continue
        lines += ["", f"## {flag} ({len(items)})", ""]
        for f in sorted(items, key=lambda x: (x["edition_year"], x["letter"], x["title"])):
            lines.append(
                f"- {f['edition_year']} vol {f['volume']} **{f['title']}** "
                f"({f['word_count']:,}w) — {f['detail']}"
            )

    with open(OUTPUT_MD, "w") as fout:
        fout.write("\n".join(lines))
    print(f"Report: {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser(description="Flag anomalous articles")
    parser.add_argument("--edition-year", type=int)
    args = parser.parse_args()

    years = [args.edition_year] if args.edition_year else YEARS
    all_flags = []

    for year in years:
        ed_dir = SITE_DIR / str(year)
        if not ed_dir.exists():
            continue

        letter_pages = sorted(ed_dir.glob("letter_*.html"))
        print(f"{year}: {len(letter_pages)} letter pages")

        for lp in letter_pages:
            letter = lp.stem.replace("letter_", "")
            articles = parse_letter_page(lp)
            if not articles:
                continue

            flags = []
            flags.extend(check_volume_consistency(articles, letter, year))
            flags.extend(check_alphabetical_order(articles, letter, year))
            flags.extend(check_bad_patterns(articles, letter, year))
            all_flags.extend(flags)

    # Deduplicate (article can be flagged by multiple checks)
    seen = {}
    deduped = []
    for f in all_flags:
        key = (f["article_id"], f["flag"])
        if key not in seen:
            seen[key] = f
            deduped.append(f)
    all_flags = deduped

    print(f"\nTotal flagged: {len(all_flags)}")
    by_flag = Counter(f["flag"] for f in all_flags)
    for flag, count in by_flag.most_common():
        print(f"  {flag}: {count}")

    # Write outputs
    with open(OUTPUT_JSONL, "w") as fout:
        for f in all_flags:
            fout.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"JSONL: {OUTPUT_JSONL}")

    write_report(all_flags)


if __name__ == "__main__":
    main()
