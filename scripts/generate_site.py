#!/usr/bin/env python3
"""
Generate static GitHub Pages site from Britannica parser export.

Reads per-edition JSONL files from EXPORT_DIR and generates a complete
static site with lazy-loaded article text, cross-edition search, and
a statistics dashboard.

Usage:
    python generate_site.py                    # Default: read from EXPORT_DIR
    python generate_site.py --export-dir /path # Override export directory
    python generate_site.py --site-dir /path   # Override output directory
"""

import json
import html
import logging
import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from config import EXPORT_DIR, SITE_DIR, EDITIONS
except ImportError:
    # Standalone mode: run from the repository root without config.py
    _REPO = Path(__file__).resolve().parent
    if _REPO.name == "scripts":
        _REPO = _REPO.parent
    EXPORT_DIR = _REPO / "data" / "export"
    SITE_DIR = _REPO / "docs"
    EDITIONS = {
        "1st": {"year": 1771, "name": "1st", "full_name": "First Edition"},
        "2nd": {"year": 1778, "name": "2nd", "full_name": "Second Edition"},
        "3rd": {"year": 1797, "name": "3rd", "full_name": "Third Edition"},
        "4th": {"year": 1810, "name": "4th", "full_name": "Fourth Edition"},
        "5th": {"year": 1815, "name": "5th", "full_name": "Fifth Edition"},
        "6th": {"year": 1823, "name": "6th", "full_name": "Sixth Edition"},
        "7th": {"year": 1842, "name": "7th", "full_name": "Seventh Edition"},
        "8th": {"year": 1860, "name": "8th", "full_name": "Eighth Edition"},
    }

log = logging.getLogger(__name__)

EDITION_YEARS = {v['year']: k for k, v in EDITIONS.items()}  # year -> edition name

EDITION_INFO = {
    1771: ("1st Edition", "First", 3),
    1778: ("2nd Edition", "Second", 10),
    1797: ("3rd Edition", "Third", 18),
    1810: ("4th Edition", "Fourth", 20),
    1817: ("5th Edition", "Fifth", 20),
    1823: ("6th Edition", "Sixth", 20),
    1842: ("7th Edition", "Seventh", 21),
    1860: ("8th Edition", "Eighth", 22),
}


# ──────────────────────────────────────────────────────────────────
# CSS (Georgian serif theme — cream/brown palette)
# ──────────────────────────────────────────────────────────────────

BASE_CSS = """
:root {
    --bg-primary: #faf8f5;
    --bg-secondary: #fff;
    --text-primary: #2c2c2c;
    --text-secondary: #666;
    --accent: #8b4513;
    --accent-light: #d4a574;
    --border: #e0d8d0;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: 'Georgia', 'Times New Roman', serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.7;
    min-height: 100vh;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    text-align: center;
    padding: 3rem 0;
    border-bottom: 2px solid var(--border);
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    color: var(--accent);
    margin-bottom: 0.5rem;
    font-weight: normal;
    letter-spacing: 0.05em;
}

header .subtitle {
    color: var(--text-secondary);
    font-style: italic;
}

nav {
    background: var(--bg-secondary);
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 2rem;
    border: 1px solid var(--border);
}

nav a {
    color: var(--accent);
    text-decoration: none;
    margin-right: 1.5rem;
}

nav a:hover { text-decoration: underline; }

.breadcrumb {
    color: var(--text-secondary);
    margin-bottom: 1rem;
    font-size: 0.9rem;
}

.breadcrumb a { color: var(--accent); text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }

.edition-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.edition-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    transition: box-shadow 0.2s, transform 0.2s;
}

.edition-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.edition-card h2 {
    color: var(--accent);
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
    font-weight: normal;
}

.edition-card .year {
    font-size: 2rem;
    color: var(--text-secondary);
    margin-bottom: 1rem;
}

.edition-card .stats {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.edition-card a.btn {
    display: inline-block;
    margin-top: 1rem;
    color: var(--bg-secondary);
    background: var(--accent);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    text-decoration: none;
}

.edition-card a.btn:hover { background: var(--accent-light); }

.volume-list { list-style: none; }

.volume-list li {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    margin-bottom: 0.5rem;
    border-radius: 4px;
}

.volume-list a {
    display: block;
    padding: 1rem;
    color: var(--text-primary);
    text-decoration: none;
}

.volume-list a:hover { background: var(--bg-primary); }

.volume-list .meta {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.article-list { list-style: none; }

.article-item {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}

.article-header {
    padding: 0.8rem 1rem;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: background 0.2s;
}

.article-header:hover { background: var(--bg-primary); }

.article-header h3 {
    font-size: 1rem;
    font-weight: normal;
    color: var(--accent);
}

.article-header .meta {
    font-size: 0.8rem;
    color: var(--text-secondary);
    white-space: nowrap;
    margin-left: 1rem;
}

.article-header .badge {
    background: var(--accent-light);
    color: white;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
    margin-left: 0.5rem;
}

.article-header .badge.treatise { background: #8b4513; }
.article-header .badge.xref { background: #6a5acd; }

.article-content {
    padding: 1rem;
    border-top: 1px solid var(--border);
    display: none;
    background: #fffef8;
}

.article-content.show { display: block; }

.article-text {
    font-size: 0.95rem;
    line-height: 1.8;
    margin-bottom: 1rem;
}

.article-actions {
    padding-top: 0.5rem;
    border-top: 1px dashed var(--border);
}

.article-actions button {
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    margin-right: 0.5rem;
}

.article-actions button:hover { background: var(--accent-light); }

.loading {
    color: var(--text-secondary);
    font-style: italic;
}

.search-box {
    width: 100%;
    padding: 1rem;
    font-size: 1.1rem;
    border: 2px solid var(--border);
    border-radius: 8px;
    margin-bottom: 1rem;
    font-family: inherit;
}

.search-box:focus {
    outline: none;
    border-color: var(--accent);
}

.search-results { margin-top: 1rem; }

.search-result {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    padding: 1rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
}

.search-result h4 { color: var(--accent); margin-bottom: 0.3rem; }
.search-result .edition { color: var(--text-secondary); font-size: 0.85rem; }
.search-result a { color: var(--accent); }

.stats-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}

.stats-table th, .stats-table td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

.stats-table th { background: var(--bg-primary); color: var(--accent); }
.stats-table tr:hover { background: var(--bg-primary); }
.stats-table td.num { text-align: right; font-variant-numeric: tabular-nums; }

.filter-bar {
    margin-bottom: 1rem;
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
}

.filter-bar input {
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: inherit;
}

.filter-bar select {
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: inherit;
}

/* Bar charts for stats page */
.bar-chart { margin: 1.5rem 0; }

.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}

.bar-label {
    width: 120px;
    font-size: 0.9rem;
    text-align: right;
    padding-right: 1rem;
    color: var(--text-secondary);
}

.bar-track {
    flex: 1;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 4px;
    height: 28px;
    position: relative;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 3px;
    transition: width 0.3s;
    min-width: 2px;
}

.bar-value {
    width: 100px;
    font-size: 0.85rem;
    padding-left: 0.75rem;
    color: var(--text-primary);
    font-variant-numeric: tabular-nums;
}

.stat-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
}

.stat-card .value {
    font-size: 2rem;
    color: var(--accent);
    font-weight: bold;
}

.stat-card .label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* Letter-range navigation for split volumes */
.letter-nav {
    background: var(--bg-secondary);
    padding: 0.8rem 1rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 1rem;
    text-align: center;
    font-size: 0.95rem;
}

.letter-nav a {
    color: var(--accent);
    text-decoration: none;
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
}

.letter-nav a:hover { background: var(--bg-primary); text-decoration: underline; }
.letter-nav a.current { background: var(--accent); color: white; }
.letter-nav .sep { color: var(--text-secondary); margin: 0 0.2rem; }

/* Split volume sub-links on edition index */
.volume-splits {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.3rem;
}

.volume-splits a {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    font-size: 0.8rem;
    color: var(--accent);
    border: 1px solid var(--border);
    border-radius: 3px;
    text-decoration: none;
}

.volume-splits a:hover { background: var(--bg-primary); }

/* Markdown-rendered article text */
.article-text p { margin-bottom: 0.8em; }
.article-text table { border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 0.9rem; }
.article-text th, .article-text td { border: 1px solid var(--border); padding: 0.4rem 0.6rem; text-align: left; }
.article-text th { background: var(--bg-primary); }

@media (max-width: 600px) {
    .container { padding: 1rem; }
    header h1 { font-size: 1.8rem; }
    .edition-grid { grid-template-columns: 1fr; }
    .stat-cards { grid-template-columns: 1fr 1fr; }
    .bar-label { width: 80px; font-size: 0.8rem; }
    .bar-value { width: 80px; }
    .letter-nav { font-size: 0.85rem; }
    .letter-nav a { padding: 0.2rem 0.4rem; }
}
"""


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def esc(text):
    """Escape HTML entities."""
    return html.escape(text) if text else ""


def title_to_id(title: str) -> str:
    """Convert article title to valid HTML ID."""
    clean = re.sub(r'[^A-Za-z0-9]+', '_', title.upper()).strip('_')
    return f"article-{clean}"


def is_treatise(article: dict) -> bool:
    """Detect treatise-length articles."""
    if article.get('type') == 'cross_reference':
        return False
    wc = article.get('word_count', 0)
    subs = article.get('subsections') or []
    return wc > 5000 or len(subs) > 3


def fmt_words(n: int) -> str:
    """Format word count compactly."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_edition(export_dir: Path, edition_key: str, year: int) -> list[dict]:
    """Load articles from an exported JSONL file."""
    path = export_dir / f"eb_{edition_key}_{year}.jsonl"
    if not path.exists():
        log.warning(f"Export file not found: {path}")
        return []

    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def load_statistics(export_dir: Path) -> dict | None:
    """Load pre-computed statistics if available."""
    path = export_dir / "statistics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_all_editions(export_dir: Path) -> dict[int, list[dict]]:
    """Load all editions, keyed by year."""
    all_editions = {}
    for edition_key, info in sorted(EDITIONS.items(), key=lambda x: x[1]['year']):
        year = info['year']
        articles = load_edition(export_dir, edition_key, year)
        if articles:
            all_editions[year] = articles
            log.info(f"  {year} ({edition_key}): {len(articles):,} entries")
    return all_editions


# ──────────────────────────────────────────────────────────────────
# Page template
# ──────────────────────────────────────────────────────────────────

def generate_html_page(title: str, content: str, breadcrumbs=None, extra_js=""):
    """Wrap content in a complete HTML page."""
    bc_html = ""
    if breadcrumbs:
        bc_items = ' &raquo; '.join(
            f'<a href="{url}">{name}</a>' if url else name
            for name, url in breadcrumbs
        )
        bc_html = f'<div class="breadcrumb">{bc_items}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)} - Encyclopaedia Britannica Historical Corpus</title>
    <style>{BASE_CSS}</style>
</head>
<body>
    <div class="container">
        {bc_html}
        {content}
        <footer>
            <p>Encyclopaedia Britannica Historical Corpus</p>
            <p>Parsed with LLM-based article boundary detection | Generated {datetime.now().strftime('%Y-%m-%d')}</p>
        </footer>
    </div>
    {extra_js}
</body>
</html>"""


def nav_html(root=""):
    """Generate navigation bar. root="" for top-level, root=".." for nested."""
    prefix = f"{root}/" if root else ""
    return f"""
    <nav>
        <a href="{prefix}index.html">Home</a>
        <a href="{prefix}search.html">Search</a>
        <a href="{prefix}stats.html">Statistics</a>
        <a href="{prefix}about.html">About</a>
    </nav>"""


# ──────────────────────────────────────────────────────────────────
# Compute per-edition stats
# ──────────────────────────────────────────────────────────────────

def compute_stats(all_editions: dict[int, list[dict]]) -> dict:
    """Compute statistics for all editions."""
    stats = {}
    for year, articles in all_editions.items():
        real_articles = [a for a in articles if a.get('type') == 'article']
        xrefs = [a for a in articles if a.get('type') == 'cross_reference']
        treatises = [a for a in real_articles if is_treatise(a)]
        volumes = set(a.get('volume', 0) for a in articles)
        total_words = sum(a.get('word_count', 0) for a in articles)

        stats[year] = {
            'articles': len(real_articles),
            'cross_refs': len(xrefs),
            'total_entries': len(articles),
            'treatises': len(treatises),
            'volumes': len(volumes),
            'word_count': total_words,
        }
    return stats


# ──────────────────────────────────────────────────────────────────
# Page generators
# ──────────────────────────────────────────────────────────────────

def generate_index_page(stats: dict) -> str:
    """Generate home page with edition cards and corpus summary."""
    cards = []
    for year in sorted(stats.keys()):
        s = stats[year]
        info = EDITION_INFO.get(year)
        ordinal = info[1] if info else ""
        cards.append(f"""
        <div class="edition-card">
            <div class="year">{year}</div>
            <h2>{ordinal} Edition</h2>
            <div class="stats">
                <div>{s['volumes']} volumes</div>
                <div>{s['articles']:,} articles</div>
                <div>{s['cross_refs']:,} cross-references</div>
                <div>{s['treatises']:,} treatises</div>
                <div>{fmt_words(s['word_count'])} words</div>
            </div>
            <a class="btn" href="{year}/index.html">Browse Edition</a>
        </div>
        """)

    total_articles = sum(s['articles'] for s in stats.values())
    total_xrefs = sum(s['cross_refs'] for s in stats.values())
    total_words = sum(s['word_count'] for s in stats.values())
    total_entries = sum(s['total_entries'] for s in stats.values())

    # Stats table
    rows = []
    for year in sorted(stats.keys()):
        s = stats[year]
        info = EDITION_INFO.get(year)
        name = info[0] if info else f"{year}"
        rows.append(f"""
            <tr>
                <td>{name}</td>
                <td class="num">{s['volumes']}</td>
                <td class="num">{s['articles']:,}</td>
                <td class="num">{s['cross_refs']:,}</td>
                <td class="num">{s['treatises']:,}</td>
                <td class="num">{s['word_count']:,}</td>
            </tr>""")

    content = f"""
    <header>
        <h1>Encyclopaedia Britannica</h1>
        <p class="subtitle">Historical Corpus (1771&ndash;1860) &mdash; {total_entries:,} entries, {fmt_words(total_words)} words</p>
    </header>

    {nav_html()}

    <p>Browse the complete text of eight editions of the Encyclopaedia Britannica,
    spanning nearly a century of knowledge from 1771 to 1860. This corpus contains
    <strong>{total_articles:,} articles</strong> and <strong>{total_xrefs:,} cross-references</strong>,
    totalling <strong>{total_words:,} words</strong>.</p>

    <div class="edition-grid">
        {''.join(cards)}
    </div>

    <h2>Corpus Summary</h2>
    <table class="stats-table">
        <thead>
            <tr>
                <th>Edition</th>
                <th>Volumes</th>
                <th>Articles</th>
                <th>Cross-refs</th>
                <th>Treatises</th>
                <th>Words</th>
            </tr>
        </thead>
        <tbody>
        {''.join(rows)}
        </tbody>
        <tfoot>
            <tr style="font-weight:bold; border-top:2px solid var(--border);">
                <td>Total</td>
                <td class="num">{sum(s['volumes'] for s in stats.values())}</td>
                <td class="num">{total_articles:,}</td>
                <td class="num">{total_xrefs:,}</td>
                <td class="num">{sum(s['treatises'] for s in stats.values()):,}</td>
                <td class="num">{total_words:,}</td>
            </tr>
        </tfoot>
    </table>
    """
    return generate_html_page("Home", content)


def generate_edition_page(year: int, articles: list[dict],
                          vol_splits: dict[int, list[tuple[str, list[dict]]]] | None = None) -> str:
    """Generate edition index page with volume listing.

    Args:
        vol_splits: Pre-computed split info per volume from split_articles_by_letter().
                    {vol_num: [(range_label, articles), ...]}
    """
    info = EDITION_INFO.get(year)
    ordinal = info[1] if info else ""

    # Group by volume
    vol_articles = defaultdict(list)
    for a in articles:
        vol_articles[a.get('volume', 0)].append(a)

    sorted_vols = sorted(vol_articles.keys())

    vol_items = []
    for vol_num in sorted_vols:
        vol_arts = vol_articles[vol_num]
        real = [a for a in vol_arts if a.get('type') == 'article']
        xrefs = [a for a in vol_arts if a.get('type') == 'cross_reference']
        treatise_count = sum(1 for a in real if is_treatise(a))
        words = sum(a.get('word_count', 0) for a in vol_arts)

        # Determine letter range using percentile-based approach.
        # Sort titles alphabetically and use 2nd/98th percentile to skip
        # stray misplaced articles at the edges of volumes.
        content_arts = [a for a in vol_arts
                        if a.get('type') in ('article', 'cross_reference')
                        and len(a.get('title', '')) > 1]
        sorted_alpha = sorted(content_arts, key=lambda a: a.get('title', '').upper())
        letter_range = ""
        if len(sorted_alpha) >= 5:
            idx_lo = max(0, len(sorted_alpha) * 2 // 100)
            idx_hi = min(len(sorted_alpha) - 1, len(sorted_alpha) * 98 // 100)
            first_title = sorted_alpha[idx_lo].get('title', '')
            last_title = sorted_alpha[idx_hi].get('title', '')
            letter_range = f"{first_title[:20]}&ndash;{last_title[:20]}"
        elif sorted_alpha:
            first_title = sorted_alpha[0].get('title', '')
            last_title = sorted_alpha[-1].get('title', '')
            letter_range = f"{first_title[:20]}&ndash;{last_title[:20]}"

        # Check if this volume is split
        groups = vol_splits.get(vol_num, []) if vol_splits else []
        is_split = len(groups) > 1

        if is_split:
            # Show volume header + letter-range sub-links
            split_links = []
            for rl, _ in groups:
                safe_range = rl.replace(' ', '')
                split_links.append(
                    f'<a href="vol{vol_num}_{safe_range}.html">{rl}</a>'
                )
            splits_html = f'<div class="volume-splits">{" ".join(split_links)}</div>'

            vol_items.append(f"""
        <li>
            <a href="vol{vol_num}.html">
                <strong>Volume {vol_num}</strong>{f': {letter_range}' if letter_range else ''}
                <div class="meta">
                    {len(real):,} articles, {len(xrefs):,} cross-refs
                    {f', {treatise_count:,} treatises' if treatise_count else ''}
                    &mdash; {fmt_words(words)} words
                </div>
            </a>
            {splits_html}
        </li>
        """)
        else:
            vol_items.append(f"""
        <li>
            <a href="vol{vol_num}.html">
                <strong>Volume {vol_num}</strong>{f': {letter_range}' if letter_range else ''}
                <div class="meta">
                    {len(real):,} articles, {len(xrefs):,} cross-refs
                    {f', {treatise_count:,} treatises' if treatise_count else ''}
                    &mdash; {fmt_words(words)} words
                </div>
            </a>
        </li>
        """)

    total_real = sum(1 for a in articles if a.get('type') == 'article')
    total_xref = sum(1 for a in articles if a.get('type') == 'cross_reference')
    total_words = sum(a.get('word_count', 0) for a in articles)

    content = f"""
    <header>
        <h1>Encyclopaedia Britannica</h1>
        <p class="subtitle">{ordinal} Edition ({year})</p>
    </header>

    {nav_html('..')}

    <p>The {ordinal} Edition contains <strong>{total_real:,} articles</strong>
    and <strong>{total_xref:,} cross-references</strong> across
    <strong>{len(sorted_vols)} volumes</strong>
    ({total_words:,} words total).</p>

    <h2>Volumes</h2>
    <ul class="volume-list">
        {''.join(vol_items)}
    </ul>
    """

    breadcrumbs = [("Home", "../index.html"), (f"{year} Edition", None)]
    return generate_html_page(f"{year} Edition", content, breadcrumbs)


def generate_volume_page(year: int, vol_num: int, articles: list[dict],
                         letter_range: str = "", data_file: str = "",
                         sibling_pages: list[tuple[str, str]] | None = None) -> str:
    """Generate volume page with article headers (text lazy-loaded).

    Args:
        letter_range: e.g. "A-D" if this is a split sub-page, "" if unsplit.
        data_file: JSON data filename (e.g. "data/vol3_A-D.json").
        sibling_pages: [(range_label, filename), ...] for letter nav bar.
    """
    info = EDITION_INFO.get(year)
    ordinal = info[1] if info else ""
    data_file = data_file or f"data/vol{vol_num}.json"

    # Sort by char_start (original document order)
    sorted_articles = sorted(articles, key=lambda a: a.get('char_start', 0))

    article_items = []
    for i, a in enumerate(sorted_articles):
        title = a.get('title', 'Unknown')
        art_type = a.get('type', 'article')
        word_count = a.get('word_count', 0)
        target = a.get('target', '')

        # Badge
        if art_type == 'cross_reference':
            badge_text = f"Cross-ref &rarr; {esc(target)}" if target else "Cross-ref"
            badge = f'<span class="badge xref">{badge_text}</span>'
        elif is_treatise(a):
            badge = '<span class="badge treatise">Treatise</span>'
        else:
            badge = ''

        article_id = title_to_id(title)
        article_items.append(f"""
        <li class="article-item" id="{article_id}" data-idx="{i}" data-type="{art_type}">
            <div class="article-header" onclick="toggleArticle({i})">
                <h3>{esc(title)}{badge}</h3>
                <span class="meta">{word_count:,} words</span>
            </div>
            <div class="article-content" id="content-{i}">
                <div class="loading">Loading...</div>
            </div>
        </li>
        """)

    # Letter-range navigation bar (for split volumes)
    letter_nav_html = ""
    if sibling_pages:
        nav_links = []
        for sib_range, sib_file in sibling_pages:
            css_class = ' class="current"' if sib_range == letter_range else ''
            nav_links.append(f'<a href="{sib_file}"{css_class}>{sib_range}</a>')
        sep = '<span class="sep"> | </span>'
        letter_nav_html = f'<div class="letter-nav">{sep.join(nav_links)}</div>'

    title_suffix = f": {letter_range}" if letter_range else ""
    subtitle_range = f" ({letter_range})" if letter_range else ""

    content = f"""
    <header>
        <h1>Volume {vol_num}{title_suffix}</h1>
        <p class="subtitle">{ordinal} Edition ({year})</p>
    </header>

    {nav_html('..')}

    {letter_nav_html}

    <p>This {'section' if letter_range else 'volume'} contains <strong>{len(sorted_articles):,} entries</strong>.
    Click on an article to view its full text.</p>

    <div class="filter-bar">
        <input type="text" id="filterInput" placeholder="Filter articles..." onkeyup="filterArticles()">
        <select id="typeFilter" onchange="filterArticles()">
            <option value="all">All Types</option>
            <option value="article">Articles</option>
            <option value="cross_reference">Cross-references</option>
            <option value="treatise">Treatises</option>
        </select>
    </div>

    <ul class="article-list" id="articleList">
        {''.join(article_items)}
    </ul>
    """

    # JavaScript for lazy loading + markdown rendering
    data_file_js = data_file  # already safe string
    extra_js = f"""
    <script>
    let articlesData = null;
    let loadedArticles = new Set();

    async function loadArticleData() {{
        if (articlesData) return;
        try {{
            const response = await fetch('{data_file_js}');
            articlesData = await response.json();
        }} catch (err) {{
            console.error('Failed to load article data:', err);
        }}
    }}

    function renderMarkdown(text) {{
        // Escape HTML first
        const div = document.createElement('div');
        div.textContent = text;
        let s = div.innerHTML;

        // Markdown tables: detect lines with | delimiters
        s = s.replace(/((?:^|\\n)\\|.+\\|(?:\\n\\|.+\\|)+)/g, function(table) {{
            const rows = table.trim().split('\\n').filter(r => r.trim());
            // Skip separator rows (e.g. |---|---|)
            const dataRows = rows.filter(r => !/^\\|[\\s\\-:|]+\\|$/.test(r.trim()));
            if (dataRows.length === 0) return table;
            let html = '<table>';
            dataRows.forEach(function(row, i) {{
                const cells = row.split('|').slice(1, -1).map(c => c.trim());
                const tag = i === 0 ? 'th' : 'td';
                html += '<tr>' + cells.map(c => '<' + tag + '>' + c + '</' + tag + '>').join('') + '</tr>';
            }});
            html += '</table>';
            return html;
        }});

        // Bold: **text**
        s = s.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
        // Italic: *text* (but not inside words like "don*t")
        s = s.replace(/(^|[\\s(])\\*([^*]+?)\\*([\\s).,;:!?]|$)/g, '$1<em>$2</em>$3');
        // Strip LaTeX markers \\( ... \\) — leave content visible
        s = s.replace(/\\\\\\((.+?)\\\\\\)/g, '$1');

        // Paragraphs: double newline
        s = s.replace(/\\n\\n+/g, '</p><p>');
        // Single newlines to <br> (but not inside tables)
        s = s.replace(/\\n/g, '<br>');

        return '<p>' + s + '</p>';
    }}

    async function toggleArticle(idx) {{
        const content = document.getElementById('content-' + idx);
        const isShown = content.classList.contains('show');

        if (isShown) {{
            content.classList.remove('show');
            return;
        }}

        content.classList.add('show');

        if (loadedArticles.has(idx)) return;

        await loadArticleData();
        if (!articlesData || !articlesData[idx]) {{
            content.innerHTML = '<div class="article-text">Error loading article.</div>';
            return;
        }}

        const article = articlesData[idx];

        content.innerHTML = `
            <div class="article-text">${{renderMarkdown(article.t)}}</div>
            <div class="article-actions">
                <button onclick="downloadMd(${{idx}})">Download .md</button>
                <button onclick="copyText(${{idx}})">Copy Text</button>
            </div>
        `;
        loadedArticles.add(idx);
    }}

    function downloadMd(idx) {{
        const article = articlesData[idx];
        const header = `# ${{article.h}}\\n\\n**Edition:** {year} {ordinal} Edition\\n**Volume:** {vol_num}\\n**Words:** ${{article.wc}}\\n\\n---\\n\\n`;
        const blob = new Blob([header + article.t], {{type: 'text/markdown'}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = article.h.replace(/[^a-zA-Z0-9]/g, '_') + '.md';
        a.click();
        URL.revokeObjectURL(url);
    }}

    function copyText(idx) {{
        const article = articlesData[idx];
        navigator.clipboard.writeText(article.t).then(() => {{
            alert('Copied to clipboard!');
        }});
    }}

    function filterArticles() {{
        const query = document.getElementById('filterInput').value.toLowerCase();
        const typeFilter = document.getElementById('typeFilter').value;
        const items = document.querySelectorAll('.article-item');

        items.forEach(item => {{
            const header = item.querySelector('h3').textContent.toLowerCase();
            const articleType = item.dataset.type;
            const isTreatise = item.querySelector('.badge.treatise') !== null;

            let show = header.includes(query);
            if (typeFilter === 'treatise') {{
                show = show && isTreatise;
            }} else if (typeFilter !== 'all') {{
                show = show && (articleType === typeFilter);
            }}

            item.style.display = show ? '' : 'none';
        }});
    }}

    // Hash navigation: auto-expand and scroll to article on page load
    window.addEventListener('load', function() {{
        const hash = window.location.hash;
        if (hash && hash.startsWith('#article-')) {{
            const article = document.querySelector(hash);
            if (article) {{
                const idx = parseInt(article.dataset.idx);
                toggleArticle(idx);
                setTimeout(() => {{
                    article.scrollIntoView({{behavior: 'smooth', block: 'start'}});
                }}, 100);
            }}
        }}
    }});

    window.addEventListener('hashchange', function() {{
        const hash = window.location.hash;
        if (hash && hash.startsWith('#article-')) {{
            const article = document.querySelector(hash);
            if (article) {{
                const idx = parseInt(article.dataset.idx);
                toggleArticle(idx);
                article.scrollIntoView({{behavior: 'smooth', block: 'start'}});
            }}
        }}
    }});
    </script>
    """

    bc_vol_label = f"Volume {vol_num}"
    if letter_range:
        breadcrumbs = [
            ("Home", "../index.html"),
            (f"{year} Edition", "index.html"),
            (bc_vol_label, f"vol{vol_num}.html"),
            (letter_range, None),
        ]
    else:
        breadcrumbs = [
            ("Home", "../index.html"),
            (f"{year} Edition", "index.html"),
            (bc_vol_label, None),
        ]
    return generate_html_page(f"Volume {vol_num}{subtitle_range} - {year}", content, breadcrumbs, extra_js)


VOLUME_SPLIT_THRESHOLD = 1500


def split_articles_by_letter(articles: list[dict], max_per_page: int = VOLUME_SPLIT_THRESHOLD) -> list[tuple[str, list[dict]]]:
    """Split articles into letter-range groups if volume exceeds threshold.

    Returns [(range_label, articles), ...] where range_label is e.g. "A-D"
    or "" if no split needed.
    """
    if len(articles) <= max_per_page:
        return [("", articles)]

    # Sort by title alphabetically for grouping
    sorted_arts = sorted(articles, key=lambda a: a.get('title', '').upper())

    # Group by uppercase first letter (dict preserves insertion order in Python 3.7+)
    letter_groups = {}
    for a in sorted_arts:
        title = a.get('title', '')
        first = title[0].upper() if title and title[0].isalpha() else '#'
        letter_groups.setdefault(first, []).append(a)

    # Greedily merge consecutive letter groups
    result = []
    current_letters = []
    current_articles = []

    for letter, arts in letter_groups.items():
        # If adding this letter would exceed the limit and we already have content,
        # flush current group first
        if current_articles and len(current_articles) + len(arts) > max_per_page:
            label = current_letters[0] if len(current_letters) == 1 else f"{current_letters[0]}-{current_letters[-1]}"
            result.append((label, current_articles))
            current_letters = []
            current_articles = []

        current_letters.append(letter)
        current_articles.extend(arts)

    # Flush remaining
    if current_articles:
        label = current_letters[0] if len(current_letters) == 1 else f"{current_letters[0]}-{current_letters[-1]}"
        result.append((label, current_articles))

    return result


def generate_volume_data(articles: list[dict]) -> list[dict]:
    """Generate compact JSON data for a volume's articles.

    Format: [{"h": title, "t": text, "wc": word_count, "tp": type, "tgt": target}, ...]
    Sorted by char_start (document order), matching the HTML listing.
    """
    sorted_articles = sorted(articles, key=lambda a: a.get('char_start', 0))
    data = []
    for a in sorted_articles:
        entry = {
            "h": a.get('title', ''),
            "t": a.get('text', ''),
            "wc": a.get('word_count', 0),
            "tp": a.get('type', 'article'),
        }
        if a.get('target'):
            entry["tgt"] = a['target']
        data.append(entry)
    return data


def generate_volume_redirect(year: int, vol_num: int,
                             sub_pages: list[tuple[str, str]]) -> str:
    """Generate a redirect stub for split volumes.

    When a user visits vol{N}.html (e.g. from a search result with #article-PARIS),
    this page loads a tiny index JSON mapping article IDs to sub-page filenames,
    checks the URL hash, and redirects to the correct sub-page.

    Args:
        sub_pages: [(range_label, filename), ...] e.g. [("A-D", "vol3_A-D.html"), ...]
    """
    info = EDITION_INFO.get(year)
    ordinal = info[1] if info else ""

    links_html = "\n".join(
        f'        <li><a href="{fname}">Volume {vol_num}: {rlabel}</a></li>'
        for rlabel, fname in sub_pages
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Volume {vol_num} - {year} - Encyclopaedia Britannica Historical Corpus</title>
    <style>{BASE_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="breadcrumb">
            <a href="../index.html">Home</a> &raquo; <a href="index.html">{year} Edition</a> &raquo; Volume {vol_num}
        </div>
        <header>
            <h1>Volume {vol_num}</h1>
            <p class="subtitle">{ordinal} Edition ({year})</p>
        </header>
        {nav_html('..')}
        <p>This volume has been split into sections for faster loading.
        Select a letter range below, or wait to be redirected if you followed a link to a specific article.</p>
        <ul class="volume-list" id="subPageList">
{links_html}
        </ul>
        <p id="redirectMsg" style="display:none; color:var(--text-secondary); font-style:italic;">
            Looking up article location...</p>
        <footer>
            <p>Encyclopaedia Britannica Historical Corpus</p>
        </footer>
    </div>
    <script>
    (function() {{
        const hash = window.location.hash;
        if (!hash || !hash.startsWith('#article-')) return;

        document.getElementById('redirectMsg').style.display = 'block';

        fetch('data/vol{vol_num}_index.json')
            .then(r => r.json())
            .then(index => {{
                const id = hash.slice(1);  // remove #
                const target = index[id];
                if (target) {{
                    window.location.replace(target + hash);
                }} else {{
                    document.getElementById('redirectMsg').textContent =
                        'Article not found in index. Please select a section above.';
                }}
            }})
            .catch(() => {{
                document.getElementById('redirectMsg').textContent =
                    'Could not load article index. Please select a section above.';
            }});
    }})();
    </script>
</body>
</html>"""


def generate_volume_index_json(vol_num: int, split_groups: list[tuple[str, list[dict]]]) -> dict:
    """Generate article-ID-to-subpage mapping for redirect stub.

    Returns dict like {"article-PARIS": "vol3_E-J.html", ...}
    """
    index = {}
    for range_label, articles in split_groups:
        safe_range = range_label.replace(' ', '')
        filename = f"vol{vol_num}_{safe_range}.html"
        for a in articles:
            art_id = title_to_id(a.get('title', ''))
            index[art_id] = filename
    return index


def generate_search_page() -> str:
    """Generate cross-edition search page."""
    content = f"""
    <header>
        <h1>Search the Corpus</h1>
        <p class="subtitle">Find articles across all eight editions</p>
    </header>

    {nav_html()}

    <input type="text" class="search-box" id="searchInput"
           placeholder="Enter an article title to search..."
           onkeyup="performSearch()">

    <div id="searchResults" class="search-results"></div>

    <script>
    let searchIndex = null;

    fetch('api/index.json')
        .then(r => r.json())
        .then(data => {{ searchIndex = data; }})
        .catch(err => console.error('Failed to load search index:', err));

    function performSearch() {{
        const query = document.getElementById('searchInput').value.toLowerCase().trim();
        const results = document.getElementById('searchResults');

        if (!searchIndex || query.length < 2) {{
            results.innerHTML = query.length > 0 ? '<p>Type at least 2 characters...</p>' : '';
            return;
        }}

        const matches = searchIndex.filter(item =>
            item[0].toLowerCase().includes(query)
        ).slice(0, 100);

        if (matches.length === 0) {{
            results.innerHTML = '<p>No results found.</p>';
            return;
        }}

        results.innerHTML = matches.map(m => `
            <div class="search-result">
                <h4><a href="${{m[1]}}/vol${{m[2]}}.html#article-${{m[0].toUpperCase().replace(/[^A-Z0-9]+/g, '_').replace(/^_|_$/g, '')}}">${{m[0]}}</a></h4>
                <span class="edition">${{m[1]}} Edition, Volume ${{m[2]}} &mdash; ${{m[3].toLocaleString()}} words</span>
            </div>
        `).join('');
    }}
    </script>
    """
    return generate_html_page("Search", content)


def generate_stats_page(stats: dict, all_editions: dict[int, list[dict]]) -> str:
    """Generate statistics dashboard with CSS bar charts."""
    total_articles = sum(s['articles'] for s in stats.values())
    total_xrefs = sum(s['cross_refs'] for s in stats.values())
    total_words = sum(s['word_count'] for s in stats.values())
    total_entries = sum(s['total_entries'] for s in stats.values())
    total_treatises = sum(s['treatises'] for s in stats.values())

    # Summary cards
    summary_cards = f"""
    <div class="stat-cards">
        <div class="stat-card">
            <div class="value">{len(stats)}</div>
            <div class="label">Editions</div>
        </div>
        <div class="stat-card">
            <div class="value">{total_entries:,}</div>
            <div class="label">Total Entries</div>
        </div>
        <div class="stat-card">
            <div class="value">{total_articles:,}</div>
            <div class="label">Articles</div>
        </div>
        <div class="stat-card">
            <div class="value">{total_xrefs:,}</div>
            <div class="label">Cross-references</div>
        </div>
        <div class="stat-card">
            <div class="value">{total_treatises:,}</div>
            <div class="label">Treatises</div>
        </div>
        <div class="stat-card">
            <div class="value">{fmt_words(total_words)}</div>
            <div class="label">Total Words</div>
        </div>
    </div>
    """

    # Articles per edition bar chart
    max_articles = max(s['articles'] for s in stats.values())
    article_bars = []
    for year in sorted(stats.keys()):
        s = stats[year]
        pct = (s['articles'] / max_articles * 100) if max_articles else 0
        article_bars.append(f"""
        <div class="bar-row">
            <div class="bar-label">{year}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="bar-value">{s['articles']:,}</div>
        </div>""")

    # Words per edition bar chart
    max_words = max(s['word_count'] for s in stats.values())
    word_bars = []
    for year in sorted(stats.keys()):
        s = stats[year]
        pct = (s['word_count'] / max_words * 100) if max_words else 0
        word_bars.append(f"""
        <div class="bar-row">
            <div class="bar-label">{year}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="bar-value">{s['word_count']:,}</div>
        </div>""")

    # Detailed table
    rows = []
    for year in sorted(stats.keys()):
        s = stats[year]
        info = EDITION_INFO.get(year)
        name = info[0] if info else str(year)
        avg_wc = s['word_count'] // s['articles'] if s['articles'] else 0
        rows.append(f"""
            <tr>
                <td><a href="{year}/index.html">{name}</a></td>
                <td class="num">{year}</td>
                <td class="num">{s['volumes']}</td>
                <td class="num">{s['articles']:,}</td>
                <td class="num">{s['cross_refs']:,}</td>
                <td class="num">{s['total_entries']:,}</td>
                <td class="num">{s['treatises']:,}</td>
                <td class="num">{s['word_count']:,}</td>
                <td class="num">{avg_wc:,}</td>
            </tr>""")

    # Totals row
    avg_total = total_words // total_articles if total_articles else 0

    content = f"""
    <header>
        <h1>Corpus Statistics</h1>
        <p class="subtitle">Encyclopaedia Britannica (1771&ndash;1860)</p>
    </header>

    {nav_html()}

    <h2>Overview</h2>
    {summary_cards}

    <h2>Articles by Edition</h2>
    <div class="bar-chart">
        {''.join(article_bars)}
    </div>

    <h2>Word Count by Edition</h2>
    <div class="bar-chart">
        {''.join(word_bars)}
    </div>

    <h2>Detailed Breakdown</h2>
    <table class="stats-table">
        <thead>
            <tr>
                <th>Edition</th>
                <th>Year</th>
                <th>Vols</th>
                <th>Articles</th>
                <th>Cross-refs</th>
                <th>Total</th>
                <th>Treatises</th>
                <th>Words</th>
                <th>Avg Words</th>
            </tr>
        </thead>
        <tbody>
        {''.join(rows)}
        </tbody>
        <tfoot>
            <tr style="font-weight:bold; border-top:2px solid var(--border);">
                <td>Total</td>
                <td></td>
                <td class="num">{sum(s['volumes'] for s in stats.values())}</td>
                <td class="num">{total_articles:,}</td>
                <td class="num">{total_xrefs:,}</td>
                <td class="num">{total_entries:,}</td>
                <td class="num">{total_treatises:,}</td>
                <td class="num">{total_words:,}</td>
                <td class="num">{avg_total:,}</td>
            </tr>
        </tfoot>
    </table>
    """
    return generate_html_page("Statistics", content)


def generate_about_page() -> str:
    """Generate about page describing the project and methodology."""
    content = f"""
    <header>
        <h1>About This Corpus</h1>
        <p class="subtitle">Historical Encyclopaedia Britannica Digital Archive</p>
    </header>

    {nav_html()}

    <h2>The Corpus</h2>
    <p>This digital corpus contains the complete text of eight editions of the
    Encyclopaedia Britannica published between 1771 and 1860, covering nearly a century
    of knowledge from the Scottish Enlightenment through the early Victorian era.
    The corpus includes over 80,000 articles and 23,000 cross-references totalling
    approximately 120 million words.</p>

    <h2>Editions Included</h2>
    <ul>
        <li><strong>1st Edition (1771)</strong> &mdash; The original three-volume work published in Edinburgh by Andrew Bell and Colin Macfarquhar</li>
        <li><strong>2nd Edition (1778&ndash;1783)</strong> &mdash; Expanded to ten volumes</li>
        <li><strong>3rd Edition (1797)</strong> &mdash; Eighteen volumes plus supplement</li>
        <li><strong>4th Edition (1810)</strong> &mdash; Twenty volumes</li>
        <li><strong>5th Edition (1817)</strong> &mdash; Twenty volumes with corrections</li>
        <li><strong>6th Edition (1823)</strong> &mdash; Twenty volumes, edited by Charles Maclaren</li>
        <li><strong>7th Edition (1842)</strong> &mdash; Twenty-one volumes plus index</li>
        <li><strong>8th Edition (1860)</strong> &mdash; Twenty-two volumes</li>
    </ul>

    <h2>Sources</h2>
    <p>The source PDF documents come from two collections:</p>
    <ul>
        <li><strong><a href="https://data.nls.uk/data/digitised-collections/encyclopaedia-britannica/">National Library of Scotland</a></strong> &mdash;
        Digitised Collections: Encyclopaedia Britannica</li>
        <li><strong><a href="https://archive.org/">Internet Archive</a></strong> &mdash;
        Historical book digitisation project</li>
    </ul>

    <h2>Technical Pipeline</h2>
    <p>The text extraction and article segmentation pipeline consists of several stages:</p>
    <ol>
        <li><strong>OCR</strong>: Raw text was extracted from scanned PDFs using
        <a href="https://github.com/allenai/olmocr">OLMoCR</a> (Allen Institute for AI),
        a state-of-the-art vision-language model that preserves document layout structure.</li>
        <li><strong>Paragraph splitting</strong>: OCR output was segmented into paragraphs,
        preserving page boundaries and character offsets.</li>
        <li><strong>LLM classification</strong>: A large language model
        (DeepSeek-R1-Distill-Llama-70B) classified each paragraph boundary as an
        article start, continuation, cross-reference, or section header.</li>
        <li><strong>Article assembly</strong>: Paragraphs were assembled into complete
        articles based on LLM boundary classifications, with fragment merging and
        deduplication across overlapping source files.</li>
        <li><strong>Export</strong>: Final deduplicated articles were exported as JSONL
        with full provenance (source file, character offsets, edition, volume).</li>
    </ol>

    <h2>Article Types</h2>
    <ul>
        <li><strong>Articles</strong>: Full encyclopaedia entries, from short definitions
        to multi-thousand-word treatises on subjects like Chemistry, Agriculture, or Medicine</li>
        <li><strong>Cross-references</strong>: Redirects from one headword to another
        (e.g., "COLOUR. See OPTICS.")</li>
        <li><strong>Treatises</strong>: Long-form articles (over 5,000 words or with multiple
        sub-sections), identified heuristically</li>
    </ul>

    <h2>Data Format</h2>
    <p>The underlying data is available in JSONL format with fields including:</p>
    <ul>
        <li>Article title and unique ID</li>
        <li>Edition and volume</li>
        <li>Article type (article or cross-reference)</li>
        <li>Character offsets within source OCR text</li>
        <li>Word count, paragraph count</li>
        <li>Sub-sections, keywords, and author attribution (where detected)</li>
    </ul>

    <h2>Acknowledgements</h2>
    <p>This project was made possible by the open data policies of the National Library
    of Scotland and the Internet Archive. OCR processing was performed on Compute Canada
    HPC infrastructure (Nibi cluster, H100 GPUs). Article boundary detection used the
    DeepSeek-R1-Distill-Llama-70B model served via vLLM on USask's Plato cluster
    (A100 80GB GPU).</p>
    """
    return generate_html_page("About", content)


# ──────────────────────────────────────────────────────────────────
# Search index
# ──────────────────────────────────────────────────────────────────

def generate_search_index(all_editions: dict[int, list[dict]]) -> list:
    """Generate compact search index: [title, year, volume, word_count]."""
    index = []
    seen = set()

    for year, articles in sorted(all_editions.items()):
        for a in articles:
            title = a.get('title', '')[:100]
            vol = a.get('volume', 0)
            wc = a.get('word_count', 0)

            key = (title.lower(), year, vol)
            if key in seen:
                continue
            seen.add(key)

            index.append([title, year, vol, wc])

    return index


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def run(export_dir: Path | None = None, site_dir: Path | None = None):
    """Generate the complete static site.

    Args:
        export_dir: Directory containing per-edition JSONL files.
        site_dir: Output directory for the generated site.
    """
    export_dir = export_dir or EXPORT_DIR
    site_dir = site_dir or SITE_DIR

    log.info(f"Generating site from {export_dir} -> {site_dir}")

    # Create output directories
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "api").mkdir(exist_ok=True)

    # Load all editions
    log.info("Loading editions...")
    all_editions = load_all_editions(export_dir)

    if not all_editions:
        log.error(f"No edition data found in {export_dir}")
        return

    # Compute stats
    stats = compute_stats(all_editions)

    # Generate top-level pages
    log.info("Generating index.html...")
    with open(site_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(generate_index_page(stats))

    log.info("Generating search.html...")
    with open(site_dir / "search.html", 'w', encoding='utf-8') as f:
        f.write(generate_search_page())

    log.info("Generating stats.html...")
    with open(site_dir / "stats.html", 'w', encoding='utf-8') as f:
        f.write(generate_stats_page(stats, all_editions))

    log.info("Generating about.html...")
    with open(site_dir / "about.html", 'w', encoding='utf-8') as f:
        f.write(generate_about_page())

    # Generate search index
    log.info("Generating search index...")
    search_index = generate_search_index(all_editions)
    with open(site_dir / "api" / "index.json", 'w', encoding='utf-8') as f:
        json.dump(search_index, f, separators=(',', ':'))
    log.info(f"  {len(search_index):,} entries in search index")

    # Generate edition pages
    for year, articles in sorted(all_editions.items()):
        log.info(f"Generating {year} edition pages...")
        edition_dir = site_dir / str(year)
        edition_dir.mkdir(exist_ok=True)
        data_dir = edition_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Clean stale volume files from previous runs (e.g. unsplit vol3.json
        # lingering after vol3 gets split into vol3_A-D.json etc.)
        for old_file in list(edition_dir.glob("vol*.html")) + list(data_dir.glob("vol*.json")):
            old_file.unlink()

        # Group articles by volume
        vol_articles = defaultdict(list)
        for a in articles:
            vol_articles[a.get('volume', 0)].append(a)

        # Pre-compute splits for each volume so edition index can reference them
        # vol_splits: {vol_num: [(range_label, articles), ...]}
        vol_splits = {}
        for vol_num, vol_arts in sorted(vol_articles.items()):
            vol_splits[vol_num] = split_articles_by_letter(vol_arts)

        # Edition index page (needs split info for sub-page links)
        with open(edition_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(generate_edition_page(year, articles, vol_splits))

        # Volume pages and data files
        split_count = 0
        for vol_num, vol_arts in sorted(vol_articles.items()):
            groups = vol_splits[vol_num]

            if len(groups) == 1 and groups[0][0] == "":
                # Unsplit volume — generate as before
                with open(edition_dir / f"vol{vol_num}.html", 'w', encoding='utf-8') as f:
                    f.write(generate_volume_page(year, vol_num, vol_arts))
                with open(data_dir / f"vol{vol_num}.json", 'w', encoding='utf-8') as f:
                    json.dump(generate_volume_data(vol_arts), f, separators=(',', ':'))
            else:
                # Split volume — generate sub-pages + redirect stub
                split_count += 1
                sibling_pages = [
                    (rl, f"vol{vol_num}_{rl.replace(' ', '')}.html")
                    for rl, _ in groups
                ]

                for range_label, group_arts in groups:
                    safe_range = range_label.replace(' ', '')
                    sub_filename = f"vol{vol_num}_{safe_range}.html"
                    sub_data = f"data/vol{vol_num}_{safe_range}.json"

                    # Sub-page HTML
                    with open(edition_dir / sub_filename, 'w', encoding='utf-8') as f:
                        f.write(generate_volume_page(
                            year, vol_num, group_arts,
                            letter_range=range_label,
                            data_file=sub_data,
                            sibling_pages=sibling_pages,
                        ))

                    # Sub-page JSON data
                    with open(data_dir / f"vol{vol_num}_{safe_range}.json", 'w', encoding='utf-8') as f:
                        json.dump(generate_volume_data(group_arts), f, separators=(',', ':'))

                # Redirect stub
                with open(edition_dir / f"vol{vol_num}.html", 'w', encoding='utf-8') as f:
                    f.write(generate_volume_redirect(year, vol_num, sibling_pages))

                # Article-to-subpage index JSON
                with open(data_dir / f"vol{vol_num}_index.json", 'w', encoding='utf-8') as f:
                    json.dump(generate_volume_index_json(vol_num, groups), f, separators=(',', ':'))

                log.info(f"    Volume {vol_num}: split into {len(groups)} sub-pages "
                         f"({', '.join(rl for rl, _ in groups)})")

        log.info(f"  {len(vol_articles)} volumes ({split_count} split)")

    # Summary
    total_entries = sum(len(a) for a in all_editions.values())
    total_words = sum(s['word_count'] for s in stats.values())
    log.info(f"Done! Generated site with {total_entries:,} entries ({total_words:,} words)")
    log.info(f"Output: {site_dir}")


def main():
    parser = ArgumentParser(description="Generate Britannica static site")
    parser.add_argument("--export-dir", type=Path,
                        help=f"Directory with JSONL exports (default: {EXPORT_DIR})")
    parser.add_argument("--site-dir", type=Path,
                        help=f"Output directory (default: {SITE_DIR})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run(export_dir=args.export_dir, site_dir=args.site_dir)


if __name__ == '__main__':
    main()
