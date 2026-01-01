#!/usr/bin/env python3
"""
Generate static GitHub Pages site for Encyclopaedia Britannica corpus.

Optimized version:
- Volume pages contain headword listings only (lightweight)
- Article text stored in separate JSON files, loaded on-demand
- Compressed search index
"""

import json
import html
import os
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
INPUT_DIR = Path("output_v2")
OUTPUT_DIR = Path("docs")
EDITIONS_TO_INCLUDE = [1771, 1778, 1810, 1815, 1823, 1842, 1860]  # Excluding 1797

EDITION_NAMES = {
    1771: ("1st Edition", "First"),
    1778: ("2nd Edition", "Second"),
    1797: ("3rd Edition", "Third"),
    1810: ("4th Edition", "Fourth"),
    1815: ("5th Edition", "Fifth"),
    1823: ("6th Edition", "Sixth"),
    1842: ("7th Edition", "Seventh"),
    1860: ("8th Edition", "Eighth"),
}

# HTML Templates
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

.edition-card a {
    display: inline-block;
    margin-top: 1rem;
    color: var(--bg-secondary);
    background: var(--accent);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    text-decoration: none;
}

.edition-card a:hover { background: var(--accent-light); }

.volume-list {
    list-style: none;
}

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

.article-list {
    list-style: none;
}

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
}

.article-header .badge {
    background: var(--accent-light);
    color: white;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
    margin-left: 0.5rem;
}

.article-content {
    padding: 1rem;
    border-top: 1px solid var(--border);
    display: none;
    background: #fffef8;
}

.article-content.show { display: block; }

.article-text {
    white-space: pre-wrap;
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

.search-results {
    margin-top: 1rem;
}

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

footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.9rem;
}

@media (max-width: 600px) {
    .container { padding: 1rem; }
    header h1 { font-size: 1.8rem; }
    .edition-grid { grid-template-columns: 1fr; }
}
"""


def escape_html(text):
    """Escape HTML entities."""
    return html.escape(text) if text else ""


def load_articles(year):
    """Load articles for a given edition year."""
    path = INPUT_DIR / f"articles_{year}.jsonl"
    if not path.exists():
        return []

    articles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def load_volumes(year):
    """Load volume metadata for a given edition year."""
    path = INPUT_DIR / f"volumes_{year}.jsonl"
    if not path.exists():
        return []

    volumes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                volumes.append(json.loads(line))
    return volumes


def generate_html_page(title, content, breadcrumbs=None, extra_js=""):
    """Generate a complete HTML page."""
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
    <title>{escape_html(title)} - Encyclopaedia Britannica Historical Corpus</title>
    <style>{BASE_CSS}</style>
</head>
<body>
    <div class="container">
        {bc_html}
        {content}
        <footer>
            <p>Encyclopaedia Britannica Historical Corpus</p>
            <p>OCR processed with OLMoCR | Generated {datetime.now().strftime('%Y-%m-%d')}</p>
        </footer>
    </div>
    {extra_js}
</body>
</html>"""


def generate_index_page(stats):
    """Generate main index page."""
    cards = []
    for year in EDITIONS_TO_INCLUDE:
        if year not in stats:
            continue
        s = stats[year]
        name, ordinal = EDITION_NAMES.get(year, (f"{year} Edition", ""))
        cards.append(f"""
        <div class="edition-card">
            <div class="year">{year}</div>
            <h2>{ordinal} Edition</h2>
            <div class="stats">
                <div>{s['volumes']:,} volumes</div>
                <div>{s['articles']:,} articles</div>
                <div>{s['treatises']:,} treatises</div>
            </div>
            <a href="{year}/index.html">Browse Edition</a>
        </div>
        """)

    content = f"""
    <header>
        <h1>Encyclopaedia Britannica</h1>
        <p class="subtitle">Historical Corpus (1771-1860)</p>
    </header>

    <nav>
        <a href="index.html">Home</a>
        <a href="search.html">Search</a>
        <a href="about.html">About</a>
    </nav>

    <p>Browse the complete text of seven editions of the Encyclopaedia Britannica,
    spanning nearly a century of knowledge from 1771 to 1860. This corpus contains
    over 174,000 articles extracted from OCR-processed historical volumes.</p>

    <div class="edition-grid">
        {''.join(cards)}
    </div>

    <h2>Corpus Statistics</h2>
    <table class="stats-table">
        <thead>
            <tr>
                <th>Edition</th>
                <th>Year</th>
                <th>Volumes</th>
                <th>Articles</th>
                <th>Treatises</th>
            </tr>
        </thead>
        <tbody>
        {''.join(f"""
            <tr>
                <td>{EDITION_NAMES.get(y, (f"{y}",))[0]}</td>
                <td>{y}</td>
                <td>{stats[y]['volumes']:,}</td>
                <td>{stats[y]['articles']:,}</td>
                <td>{stats[y]['treatises']:,}</td>
            </tr>
        """ for y in EDITIONS_TO_INCLUDE if y in stats)}
        </tbody>
    </table>
    """

    return generate_html_page("Home", content)


def generate_edition_page(year, volumes, articles):
    """Generate edition index page."""
    name, ordinal = EDITION_NAMES.get(year, (f"{year} Edition", ""))

    # Group articles by volume
    vol_articles = defaultdict(list)
    for a in articles:
        vol_articles[a.get('volume_num', 0)].append(a)

    # Sort volumes
    sorted_vols = sorted(set(v.get('volume_num', 0) for v in volumes))

    vol_items = []
    for vol_num in sorted_vols:
        vol_info = next((v for v in volumes if v.get('volume_num') == vol_num), {})
        letter_range = vol_info.get('letter_range', '')
        article_count = len(vol_articles.get(vol_num, []))
        treatise_count = sum(1 for a in vol_articles.get(vol_num, []) if a.get('article_type') == 'treatise')

        vol_items.append(f"""
        <li>
            <a href="vol{vol_num}.html">
                <strong>Volume {vol_num}</strong>
                {f': {letter_range}' if letter_range else ''}
                <div class="meta">{article_count:,} articles, {treatise_count:,} treatises</div>
            </a>
        </li>
        """)

    total_articles = len(articles)
    total_treatises = sum(1 for a in articles if a.get('article_type') == 'treatise')

    content = f"""
    <header>
        <h1>Encyclopaedia Britannica</h1>
        <p class="subtitle">{ordinal} Edition ({year})</p>
    </header>

    <nav>
        <a href="../index.html">Home</a>
        <a href="../search.html">Search</a>
    </nav>

    <p>The {ordinal} Edition contains <strong>{total_articles:,} articles</strong>
    including <strong>{total_treatises:,} treatises</strong> across
    <strong>{len(sorted_vols)} volumes</strong>.</p>

    <h2>Volumes</h2>
    <ul class="volume-list">
        {''.join(vol_items)}
    </ul>
    """

    breadcrumbs = [("Home", "../index.html"), (f"{year} Edition", None)]
    return generate_html_page(f"{year} Edition", content, breadcrumbs)


def generate_volume_page(year, vol_num, articles, vol_info):
    """Generate volume page with article listings (text loaded on-demand)."""
    name, ordinal = EDITION_NAMES.get(year, (f"{year} Edition", ""))
    letter_range = vol_info.get('letter_range', '')

    # Sort articles by headword
    sorted_articles = sorted(articles, key=lambda a: a.get('headword', '').upper())

    # Generate article list HTML (headers only, no text)
    article_items = []
    for i, a in enumerate(sorted_articles):
        headword = a.get('headword', 'Unknown')
        article_type = a.get('article_type', 'dictionary')
        start_page = a.get('start_page', '?')
        end_page = a.get('end_page', '?')
        word_count = a.get('word_count', 0)

        pages = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
        badge = '<span class="badge">Treatise</span>' if article_type == 'treatise' else ''

        article_items.append(f"""
        <li class="article-item" data-idx="{i}">
            <div class="article-header" onclick="toggleArticle({i})">
                <h3>{escape_html(headword)}{badge}</h3>
                <span class="meta">pp. {pages} | {word_count:,} words</span>
            </div>
            <div class="article-content" id="content-{i}">
                <div class="loading">Loading...</div>
            </div>
        </li>
        """)

    content = f"""
    <header>
        <h1>Volume {vol_num}</h1>
        <p class="subtitle">{ordinal} Edition ({year}) {f'| {letter_range}' if letter_range else ''}</p>
    </header>

    <nav>
        <a href="../index.html">Home</a>
        <a href="index.html">{year} Edition</a>
        <a href="../search.html">Search</a>
    </nav>

    <p>This volume contains <strong>{len(sorted_articles):,} articles</strong>.
    Click on an article to view its full text.</p>

    <div class="filter-bar">
        <input type="text" id="filterInput" placeholder="Filter articles..." onkeyup="filterArticles()">
        <select id="typeFilter" onchange="filterArticles()">
            <option value="all">All Types</option>
            <option value="treatise">Treatises Only</option>
            <option value="dictionary">Dictionary Only</option>
        </select>
    </div>

    <ul class="article-list" id="articleList">
        {''.join(article_items)}
    </ul>
    """

    # JavaScript for lazy loading article text
    extra_js = f"""
    <script>
    const YEAR = {year};
    const VOL = {vol_num};
    let articlesData = null;
    let loadedArticles = new Set();

    async function loadArticleData() {{
        if (articlesData) return;
        try {{
            const response = await fetch('data/vol{vol_num}.json');
            articlesData = await response.json();
        }} catch (err) {{
            console.error('Failed to load article data:', err);
        }}
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
        const pages = article.sp === article.ep ? article.sp : article.sp + '-' + article.ep;

        content.innerHTML = `
            <div class="article-text">${{escapeHtml(article.t)}}</div>
            <div class="article-actions">
                <button onclick="downloadMd(${{idx}})">Download .md</button>
                <button onclick="copyText(${{idx}})">Copy Text</button>
            </div>
        `;
        loadedArticles.add(idx);
    }}

    function escapeHtml(text) {{
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }}

    function downloadMd(idx) {{
        const article = articlesData[idx];
        const pages = article.sp === article.ep ? article.sp : article.sp + '-' + article.ep;
        const header = `# ${{article.h}}\\n\\n**Edition:** {year} {ordinal} Edition\\n**Volume:** {vol_num}\\n**Pages:** ${{pages}}\\n\\n---\\n\\n`;
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
            const isTreatise = item.querySelector('.badge') !== null;

            let show = header.includes(query);
            if (typeFilter === 'treatise') show = show && isTreatise;
            if (typeFilter === 'dictionary') show = show && !isTreatise;

            item.style.display = show ? '' : 'none';
        }});
    }}
    </script>
    """

    breadcrumbs = [
        ("Home", "../index.html"),
        (f"{year} Edition", "index.html"),
        (f"Volume {vol_num}", None)
    ]
    return generate_html_page(f"Volume {vol_num} - {year}", content, breadcrumbs, extra_js)


def generate_volume_data(articles):
    """Generate compact JSON data for a volume's articles."""
    data = []
    sorted_articles = sorted(articles, key=lambda a: a.get('headword', '').upper())

    for a in sorted_articles:
        data.append({
            "h": a.get('headword', ''),      # headword
            "t": a.get('text', ''),           # text
            "sp": a.get('start_page'),        # start page
            "ep": a.get('end_page'),          # end page
        })
    return data


def generate_search_page():
    """Generate search page."""
    content = """
    <header>
        <h1>Search the Corpus</h1>
        <p class="subtitle">Find articles across all editions</p>
    </header>

    <nav>
        <a href="index.html">Home</a>
        <a href="search.html">Search</a>
    </nav>

    <input type="text" class="search-box" id="searchInput"
           placeholder="Enter a headword to search..."
           onkeyup="performSearch()">

    <div id="searchResults" class="search-results"></div>

    <script>
    let searchIndex = null;

    fetch('api/index.json')
        .then(r => r.json())
        .then(data => { searchIndex = data; })
        .catch(err => console.error('Failed to load search index:', err));

    function performSearch() {
        const query = document.getElementById('searchInput').value.toLowerCase().trim();
        const results = document.getElementById('searchResults');

        if (!searchIndex || query.length < 2) {
            results.innerHTML = query.length > 0 ? '<p>Type at least 2 characters...</p>' : '';
            return;
        }

        const matches = searchIndex.filter(item =>
            item[0].toLowerCase().includes(query)
        ).slice(0, 100);

        if (matches.length === 0) {
            results.innerHTML = '<p>No results found.</p>';
            return;
        }

        results.innerHTML = matches.map(m => `
            <div class="search-result">
                <h4><a href="${m[1]}/vol${m[2]}.html">${m[0]}</a></h4>
                <span class="edition">${m[1]} Edition, Volume ${m[2]}, pp. ${m[3]}</span>
            </div>
        `).join('');
    }
    </script>
    """

    return generate_html_page("Search", content)


def generate_about_page():
    """Generate about page."""
    content = """
    <header>
        <h1>About This Corpus</h1>
        <p class="subtitle">Historical Encyclopaedia Britannica Digital Archive</p>
    </header>

    <nav>
        <a href="index.html">Home</a>
        <a href="search.html">Search</a>
        <a href="about.html">About</a>
    </nav>

    <h2>The Corpus</h2>
    <p>This digital corpus contains OCR-processed text from seven editions of the
    Encyclopaedia Britannica published between 1771 and 1860. The editions represent
    a remarkable evolution of human knowledge during the Enlightenment and early
    Industrial Revolution.</p>

    <h2>Editions Included</h2>
    <ul>
        <li><strong>1st Edition (1771)</strong> - The original three-volume work published in Edinburgh</li>
        <li><strong>2nd Edition (1778-1783)</strong> - Expanded to ten volumes</li>
        <li><strong>4th Edition (1810)</strong> - Twenty volumes with supplement</li>
        <li><strong>5th Edition (1815)</strong> - Reprint with corrections</li>
        <li><strong>6th Edition (1823)</strong> - Twenty volumes</li>
        <li><strong>7th Edition (1842)</strong> - Twenty-one volumes plus index</li>
        <li><strong>8th Edition (1860)</strong> - Twenty-two volumes</li>
    </ul>

    <h2>Sources</h2>
    <p>The source PDF documents come from two collections:</p>
    <ul>
        <li><strong><a href="https://data.nls.uk/data/digitised-collections/encyclopaedia-britannica/">National Library of Scotland</a></strong> -
        Digitised Collections: Encyclopaedia Britannica</li>
        <li><strong><a href="https://archive.org/">Internet Archive</a></strong> -
        Historical book digitization project</li>
    </ul>

    <h2>Technical Details</h2>
    <p>Text extraction was performed using <strong><a href="https://github.com/allenai/olmocr">OLMoCR</a></strong>
    (Optical Layout Model OCR), a state-of-the-art vision-language model developed by the Allen Institute for AI.
    OLMoCR preserves document structure and provides character-level page number mapping, enabling precise
    provenance tracking for each article.</p>

    <h2>Usage</h2>
    <p>This corpus is provided for research and educational purposes. Individual
    articles can be downloaded in Markdown format for offline use.</p>

    <h2>Data Format</h2>
    <p>The underlying data is available in JSONL format with full provenance including:</p>
    <ul>
        <li>Edition year and name</li>
        <li>Volume number</li>
        <li>Page numbers (start and end)</li>
        <li>Article type (dictionary entry vs treatise)</li>
        <li>Word count</li>
    </ul>

    <h2>Acknowledgments</h2>
    <p>This project was made possible by the open data policies of the National Library of Scotland
    and the Internet Archive's commitment to universal access to knowledge.</p>
    """

    return generate_html_page("About", content)


def generate_search_index(all_articles):
    """Generate compact search index: [headword, year, volume, pages]."""
    index = []
    seen = set()  # Avoid exact duplicates

    for year, articles in all_articles.items():
        for a in articles:
            headword = a.get('headword', '')[:100]  # Truncate long headwords
            vol = a.get('volume_num', 0)
            start_page = a.get('start_page', '?')
            end_page = a.get('end_page', '?')
            pages = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"

            key = (headword.lower(), year, vol)
            if key in seen:
                continue
            seen.add(key)

            # Compact array format: [headword, year, volume, pages]
            index.append([headword, year, vol, pages])

    return index


def main():
    print("Generating Encyclopaedia Britannica GitHub Pages site (optimized)...")

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "api").mkdir(exist_ok=True)

    # Load all data
    stats = {}
    all_articles = {}
    all_volumes = {}

    for year in EDITIONS_TO_INCLUDE:
        print(f"  Loading {year}...")
        articles = load_articles(year)
        volumes = load_volumes(year)

        if not articles:
            print(f"    No articles found for {year}, skipping")
            continue

        all_articles[year] = articles
        all_volumes[year] = volumes

        treatises = sum(1 for a in articles if a.get('article_type') == 'treatise')
        stats[year] = {
            'volumes': len(set(v.get('volume_num', 0) for v in volumes)),
            'articles': len(articles),
            'treatises': treatises,
        }
        print(f"    {len(articles):,} articles, {treatises:,} treatises")

    # Generate index page
    print("  Generating index.html...")
    with open(OUTPUT_DIR / "index.html", 'w', encoding='utf-8') as f:
        f.write(generate_index_page(stats))

    # Generate search page
    print("  Generating search.html...")
    with open(OUTPUT_DIR / "search.html", 'w', encoding='utf-8') as f:
        f.write(generate_search_page())

    # Generate about page
    print("  Generating about.html...")
    with open(OUTPUT_DIR / "about.html", 'w', encoding='utf-8') as f:
        f.write(generate_about_page())

    # Generate search index (compact format)
    print("  Generating search index...")
    search_index = generate_search_index(all_articles)
    with open(OUTPUT_DIR / "api" / "index.json", 'w', encoding='utf-8') as f:
        json.dump(search_index, f, separators=(',', ':'))  # Compact JSON
    print(f"    {len(search_index):,} entries")

    # Generate edition pages
    for year in EDITIONS_TO_INCLUDE:
        if year not in all_articles:
            continue

        print(f"  Generating {year} edition pages...")
        edition_dir = OUTPUT_DIR / str(year)
        edition_dir.mkdir(exist_ok=True)
        data_dir = edition_dir / "data"
        data_dir.mkdir(exist_ok=True)

        articles = all_articles[year]
        volumes = all_volumes[year]

        # Edition index
        with open(edition_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(generate_edition_page(year, volumes, articles))

        # Group articles by volume
        vol_articles = defaultdict(list)
        for a in articles:
            vol_articles[a.get('volume_num', 0)].append(a)

        # Volume pages and data files
        for vol_num, vol_arts in vol_articles.items():
            vol_info = next((v for v in volumes if v.get('volume_num') == vol_num), {})

            # HTML page (lightweight)
            with open(edition_dir / f"vol{vol_num}.html", 'w', encoding='utf-8') as f:
                f.write(generate_volume_page(year, vol_num, vol_arts, vol_info))

            # JSON data file (article text)
            with open(data_dir / f"vol{vol_num}.json", 'w', encoding='utf-8') as f:
                json.dump(generate_volume_data(vol_arts), f, separators=(',', ':'))

        print(f"    {len(vol_articles)} volumes")

    total_articles = sum(len(a) for a in all_articles.values())
    print(f"\nDone! Generated site with {total_articles:,} articles")
    print(f"Output: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
