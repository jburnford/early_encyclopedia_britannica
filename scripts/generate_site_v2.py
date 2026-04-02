#!/usr/bin/env python3
"""Generate static site with one HTML file per article.

Creates ~146K lightweight HTML files for instant loading.
Structure: docs/articles/{year}/{article_id}.html
Plus edition index pages and a main landing page.

Usage:
    python scripts/generate_site_v2.py
    python scripts/generate_site_v2.py --edition-year 1810
"""

import html
import json
import logging
import re
import sys
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO / "data" / "export"
SITE_DIR = REPO / "docs"

EDITIONS = [
    (1771, "1st", "First Edition", 3),
    (1778, "2nd", "Second Edition", 10),
    (1797, "3rd", "Third Edition", 18),
    (1810, "4th", "Fourth Edition (Supplement)", 20),
    (1815, "5th", "Fifth Edition", 20),
    (1823, "6th", "Sixth Edition", 20),
    (1842, "7th", "Seventh Edition", 21),
    (1860, "8th", "Eighth Edition", 22),
]

CSS = """\
:root{--bg:#faf8f5;--bg2:#fff;--fg:#2c2c2c;--fg2:#666;--accent:#8b4513;--accent2:#d4a574;--border:#e0d8d0}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Georgia,'Times New Roman',serif;background:var(--bg);color:var(--fg);line-height:1.7}
.c{max-width:900px;margin:0 auto;padding:1.5rem}
header{text-align:center;padding:2rem 0;border-bottom:2px solid var(--border);margin-bottom:1.5rem}
header h1{font-size:2rem;color:var(--accent);font-weight:normal;letter-spacing:.05em}
header .sub{color:var(--fg2);font-style:italic}
nav{background:var(--bg2);padding:.8rem 1rem;border-radius:4px;margin-bottom:1.5rem;border:1px solid var(--border);font-size:.9rem}
nav a{color:var(--accent);text-decoration:none;margin-right:1.2rem}
nav a:hover{text-decoration:underline}
.bc{color:var(--fg2);margin-bottom:1rem;font-size:.85rem}
.bc a{color:var(--accent);text-decoration:none}
.bc a:hover{text-decoration:underline}
.article{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:2rem;margin-bottom:1.5rem}
.article h2{color:var(--accent);font-size:1.6rem;margin-bottom:.5rem;font-weight:normal}
.article .meta{color:var(--fg2);font-size:.85rem;margin-bottom:1.5rem;border-bottom:1px solid var(--border);padding-bottom:1rem}
.article .text{white-space:pre-wrap;font-size:.95rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1.2rem;margin:1.5rem 0}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:1.2rem;transition:box-shadow .2s}
.card:hover{box-shadow:0 3px 10px rgba(0,0,0,.08)}
.card h3{color:var(--accent);font-size:1.1rem;font-weight:normal;margin-bottom:.3rem}
.card .yr{font-size:1.8rem;color:var(--fg2);margin-bottom:.5rem}
.card .stats{font-size:.85rem;color:var(--fg2)}
.card a.btn{display:inline-block;margin-top:.8rem;color:var(--bg2);background:var(--accent);padding:.4rem .8rem;border-radius:4px;text-decoration:none;font-size:.85rem}
.card a.btn:hover{background:var(--accent2)}
table{width:100%;border-collapse:collapse;margin:1rem 0}
th,td{text-align:left;padding:.5rem .8rem;border-bottom:1px solid var(--border);font-size:.9rem}
th{background:var(--bg);color:var(--fg2)}
td a{color:var(--accent);text-decoration:none}
td a:hover{text-decoration:underline}
.search{width:100%;padding:.6rem;font-size:1rem;border:1px solid var(--border);border-radius:4px;margin-bottom:1rem;font-family:inherit}
footer{text-align:center;color:var(--fg2);font-size:.8rem;padding:2rem 0;border-top:1px solid var(--border);margin-top:2rem}
"""


def esc(text):
    return html.escape(text) if text else ""


def page(title, body, breadcrumbs=None, depth=0):
    """Wrap body HTML in a full page."""
    prefix = "../" * depth
    bc_html = ""
    if breadcrumbs:
        parts = [f'<a href="{prefix}{href}">{label}</a>' for label, href in breadcrumbs]
        bc_html = f'<div class="bc">{" &rsaquo; ".join(parts)}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} — Encyclopaedia Britannica Historical Editions</title>
<style>{CSS}</style>
</head>
<body>
<div class="c">
{bc_html}
{body}
<footer>Encyclopaedia Britannica Historical Editions (1771–1860) &middot; National Library of Scotland Digitisation</footer>
</div>
</body>
</html>"""


def format_text(text):
    """Convert article text to HTML paragraphs."""
    if not text:
        return ""
    paragraphs = text.split("\n\n")
    parts = []
    for p in paragraphs:
        p = p.strip()
        if p:
            parts.append(f"<p>{esc(p)}</p>")
    return "\n".join(parts)


def generate_article_page(article, prev_art, next_art):
    """Generate HTML for a single article."""
    year = article["edition_year"]
    title = article["title"]
    aid = article["article_id"]
    wc = article.get("word_count", 0)
    vol = article.get("volume", "?")

    breadcrumbs = [
        ("Home", "index.html"),
        (f"{year} Edition", f"articles/{year}/index.html"),
    ]

    # Prev/next navigation
    nav_parts = []
    if prev_art:
        nav_parts.append(f'<a href="{prev_art["article_id"]}.html">&larr; {esc(prev_art["title"])}</a>')
    nav_parts.append(f'<a href="index.html">Index</a>')
    if next_art:
        nav_parts.append(f'<a href="{next_art["article_id"]}.html">{esc(next_art["title"])} &rarr;</a>')
    nav_html = " &middot; ".join(nav_parts)

    text_html = format_text(article.get("text", ""))

    body = f"""
<div class="article">
<h2>{esc(title)}</h2>
<div class="meta">Volume {vol} &middot; {wc:,} words &middot; {year} Edition</div>
<nav style="margin-bottom:1.5rem">{nav_html}</nav>
<div class="text">{text_html}</div>
</div>
<nav>{nav_html}</nav>
"""
    return page(f"{title} ({year})", body, breadcrumbs, depth=2)


def generate_edition_index(year, articles):
    """Generate index page listing all articles for one edition."""
    breadcrumbs = [("Home", "index.html")]

    ed_name = next((n for y, _, n, _ in EDITIONS if y == year), f"{year}")
    rows = []
    for a in articles:
        aid = a["article_id"]
        title = a["title"]
        wc = a.get("word_count", 0)
        vol = a.get("volume", "?")
        rows.append(
            f'<tr><td><a href="{aid}.html">{esc(title)}</a></td>'
            f'<td>{vol}</td><td>{wc:,}</td></tr>'
        )

    body = f"""
<header>
<h1>{ed_name} ({year})</h1>
<div class="sub">{len(articles):,} articles</div>
</header>
<input type="text" class="search" id="search" placeholder="Filter articles..." oninput="filterTable()">
<table id="articles">
<thead><tr><th>Article</th><th>Vol</th><th>Words</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<script>
function filterTable(){{
  const q=document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('#articles tbody tr').forEach(r=>{{
    r.style.display=r.cells[0].textContent.toLowerCase().includes(q)?'':'none';
  }});
}}
</script>
"""
    return page(f"{ed_name} ({year})", body, breadcrumbs, depth=2)


def generate_home(edition_stats):
    """Generate the main landing page."""
    cards = []
    total_articles = sum(s["count"] for s in edition_stats.values())
    total_words = sum(s["words"] for s in edition_stats.values())

    for year, ed_short, ed_name, vols in EDITIONS:
        s = edition_stats.get(year, {"count": 0, "words": 0})
        cards.append(f"""
<div class="card">
<div class="yr">{year}</div>
<h3>{ed_name}</h3>
<div class="stats">{s['count']:,} articles &middot; {s['words']:,} words &middot; {vols} vols</div>
<a class="btn" href="articles/{year}/index.html">Browse &rarr;</a>
</div>""")

    body = f"""
<header>
<h1>Encyclopaedia Britannica</h1>
<div class="sub">Historical Editions (1771–1860) &middot; National Library of Scotland</div>
</header>
<p style="text-align:center;margin-bottom:2rem;color:var(--fg2)">
{total_articles:,} articles &middot; {total_words:,} words across 8 editions
</p>
<div class="grid">
{"".join(cards)}
</div>
"""
    return page("Encyclopaedia Britannica Historical Editions", body, depth=0)


def main():
    parser = ArgumentParser(description="Generate per-article static site")
    parser.add_argument("--edition-year", type=int)
    args = parser.parse_args()

    years = [args.edition_year] if args.edition_year else [y for y, _, _, _ in EDITIONS]

    edition_stats = {}

    for year in years:
        export_file = list(EXPORT_DIR.glob(f"eb_*_{year}.jsonl"))
        if not export_file:
            log.warning(f"No export file for {year}")
            continue
        export_file = export_file[0]

        log.info(f"Loading {year}...")
        articles = []
        with open(export_file) as f:
            for line in f:
                a = json.loads(line)
                if a.get("word_count", 0) >= 5:
                    articles.append(a)

        log.info(f"  {len(articles):,} articles")
        edition_stats[year] = {
            "count": len(articles),
            "words": sum(a.get("word_count", 0) for a in articles),
        }

        # Create edition directory
        ed_dir = SITE_DIR / "articles" / str(year)
        ed_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual article pages
        for i, art in enumerate(articles):
            prev_art = articles[i - 1] if i > 0 else None
            next_art = articles[i + 1] if i < len(articles) - 1 else None
            html_content = generate_article_page(art, prev_art, next_art)
            (ed_dir / f"{art['article_id']}.html").write_text(html_content)

        # Generate edition index
        index_html = generate_edition_index(year, articles)
        (ed_dir / "index.html").write_text(index_html)
        log.info(f"  Wrote {len(articles):,} article pages + index")

    # Generate home page
    # Load stats for editions we didn't process this run
    for year, _, _, _ in EDITIONS:
        if year not in edition_stats:
            export_file = list(EXPORT_DIR.glob(f"eb_*_{year}.jsonl"))
            if export_file:
                count = 0
                words = 0
                with open(export_file[0]) as f:
                    for line in f:
                        a = json.loads(line)
                        if a.get("word_count", 0) >= 5:
                            count += 1
                            words += a.get("word_count", 0)
                edition_stats[year] = {"count": count, "words": words}

    home_html = generate_home(edition_stats)
    (SITE_DIR / "index.html").write_text(home_html)
    log.info(f"\nDone! {sum(s['count'] for s in edition_stats.values()):,} article pages generated")
    log.info(f"Output: {SITE_DIR}")


if __name__ == "__main__":
    main()
