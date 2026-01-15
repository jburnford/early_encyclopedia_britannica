#!/usr/bin/env python3
"""
Regenerate HTML files from corrected JSON data with proper lazy loading.

Reads from docs/{edition}/data/vol*.json (which already has hyperlinks)
and generates proper HTML with:
- Lazy loading (article text fetched on-demand)
- Filter bar
- Type badges (estimated from text length)
- Proper breadcrumb navigation

Usage:
    python3 regenerate_html_v2.py 1771
    python3 regenerate_html_v2.py --all
"""

import json
import html
import re
import argparse
from pathlib import Path
from datetime import datetime

EDITION_NAMES = {
    '1771': ("1st Edition", "First"),
    '1778': ("2nd Edition", "Second"),
    '1797': ("3rd Edition", "Third"),
    '1810': ("4th Edition", "Fourth"),
    '1815': ("5th Edition", "Fifth"),
    '1823': ("6th Edition", "Sixth"),
    '1842': ("7th Edition", "Seventh"),
    '1853': ("8th Edition", "Eighth"),
    '1860': ("8th Edition Alt", "Eighth Alt"),
}

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

.article-header .badge.treatise { background: #8b4513; }
.article-header .badge.biographical { background: #2e7d32; }
.article-header .badge.geographical { background: #1565c0; }

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
    flex-grow: 1;
    min-width: 200px;
}

.filter-bar select {
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: inherit;
}

/* Cross-reference links within article text */
.xref {
    color: #8b4513;
    text-decoration: none;
    border-bottom: 1px dotted #8b4513;
    transition: background 0.2s;
}

.xref:hover {
    background: #fff3cd;
    border-bottom-style: solid;
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
}
"""


def escape_html(text):
    """Escape HTML entities."""
    return html.escape(text) if text else ""


def headword_to_id(headword: str) -> str:
    """Convert headword to valid HTML ID for deep-linking."""
    clean = re.sub(r'[^A-Za-z0-9]+', '_', headword.upper())
    clean = clean.strip('_')
    return f"article-{clean}"


def estimate_article_type(article: dict) -> str:
    """Estimate article type from content characteristics."""
    text = article.get('t', '')
    headword = article.get('h', '').upper()
    sp = article.get('sp', 0)
    ep = article.get('ep', sp)

    # Calculate text length (strip HTML tags for better estimate)
    clean_text = re.sub(r'<[^>]+>', '', text)
    text_len = len(clean_text)
    page_span = (ep or sp) - (sp or 0) + 1 if sp else 1

    # Treatise: Long articles spanning multiple pages
    if page_span >= 3 or text_len > 5000:
        return 'treatise'

    # Geographical indicators
    geo_patterns = [
        r'\ba (town|city|village|island|river|mountain|lake|country|kingdom|province|county|district|region) ',
        r'\bsituated (in|on|near|at)\b',
        r'\bthe capital of\b',
        r'\bport|harbour|coast\b',
    ]
    for pattern in geo_patterns:
        if re.search(pattern, text.lower()):
            return 'geographical'

    # Biographical indicators
    bio_patterns = [
        r'\bborn (in|at|about)\b',
        r'\bdied (in|at|about)\b',
        r'\b(king|queen|emperor|pope|bishop|professor|author|philosopher|scientist) of\b',
        r'\bwas a (famous|celebrated|eminent|distinguished)\b',
    ]
    for pattern in bio_patterns:
        if re.search(pattern, text.lower()):
            return 'biographical'

    return 'dictionary'


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


def generate_edition_index(edition: str, volumes: dict) -> str:
    """Generate edition index.html."""
    name, ordinal = EDITION_NAMES.get(edition, (f"{edition} Edition", ""))

    vol_items = []
    total_articles = 0
    total_treatises = 0

    for vol_name in sorted(volumes.keys()):
        articles = volumes[vol_name]['articles']
        count = len(articles)
        total_articles += count

        # Count treatises
        treatise_count = sum(1 for a in articles if estimate_article_type(a) == 'treatise')
        total_treatises += treatise_count

        vol_items.append(f"""
        <li>
            <a href="{vol_name}.html">
                <strong>Volume {vol_name.replace('vol', '')}</strong>
                <div class="meta">{count:,} articles, {treatise_count:,} treatises</div>
            </a>
        </li>
        """)

    content = f"""
    <header>
        <h1>Encyclopaedia Britannica</h1>
        <p class="subtitle">{ordinal} Edition ({edition})</p>
    </header>

    <nav>
        <a href="../index.html">Home</a>
        <a href="../search.html">Search</a>
    </nav>

    <p>The {ordinal} Edition contains <strong>{total_articles:,} articles</strong>
    including <strong>{total_treatises:,} treatises</strong> across
    <strong>{len(volumes)} volumes</strong>.</p>

    <h2>Volumes</h2>
    <ul class="volume-list">
        {''.join(vol_items)}
    </ul>
    """

    breadcrumbs = [("Home", "../index.html"), (f"{edition} Edition", None)]
    return generate_html_page(f"{edition} Edition", content, breadcrumbs)


def generate_volume_page(edition: str, vol_name: str, articles: list) -> str:
    """Generate volume page with lazy loading."""
    name, ordinal = EDITION_NAMES.get(edition, (f"{edition} Edition", ""))
    vol_num = vol_name.replace('vol', '')

    # Sort articles alphabetically
    sorted_articles = sorted(articles, key=lambda a: a.get('h', '').upper())

    article_items = []
    for i, a in enumerate(sorted_articles):
        headword = a.get('h', 'Unknown')
        sp = a.get('sp', '?')
        ep = a.get('ep', sp)
        article_type = estimate_article_type(a)

        # Calculate word count
        clean_text = re.sub(r'<[^>]+>', '', a.get('t', ''))
        word_count = len(clean_text.split())

        pages = f"{sp}" if sp == ep else f"{sp}-{ep}"

        if article_type == 'treatise':
            badge = '<span class="badge treatise">Treatise</span>'
        elif article_type == 'biographical':
            badge = '<span class="badge biographical">Biography</span>'
        elif article_type == 'geographical':
            badge = '<span class="badge geographical">Place</span>'
        else:
            badge = ''

        article_id = headword_to_id(headword)
        article_items.append(f"""
        <li class="article-item" id="{article_id}" data-idx="{i}" data-type="{article_type}">
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
        <p class="subtitle">{ordinal} Edition ({edition})</p>
    </header>

    <nav>
        <a href="../index.html">Home</a>
        <a href="index.html">{edition} Edition</a>
        <a href="../search.html">Search</a>
    </nav>

    <p>This volume contains <strong>{len(sorted_articles):,} articles</strong>.
    Click on an article to view its full text.</p>

    <div class="filter-bar">
        <input type="text" id="filterInput" placeholder="Filter articles..." onkeyup="filterArticles()">
        <select id="typeFilter" onchange="filterArticles()">
            <option value="all">All Types</option>
            <option value="treatise">Treatises</option>
            <option value="biographical">Biographical</option>
            <option value="geographical">Geographical</option>
            <option value="dictionary">Dictionary</option>
        </select>
    </div>

    <ul class="article-list" id="articleList">
        {''.join(article_items)}
    </ul>
    """

    extra_js = f"""
    <script>
    const YEAR = {edition};
    const VOL = {vol_num};
    let articlesData = null;
    let loadedArticles = new Set();

    async function loadArticleData() {{
        if (articlesData) return;
        try {{
            const response = await fetch('data/{vol_name}.json');
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
            <div class="article-text">${{renderArticleHtml(article.t)}}</div>
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

    function renderArticleHtml(text) {{
        // Text already contains hyperlinks as HTML, just need to escape non-link content
        // The hyperlinks are stored as actual HTML in the JSON
        return text;
    }}

    function downloadMd(idx) {{
        const article = articlesData[idx];
        const pages = article.sp === article.ep ? article.sp : article.sp + '-' + article.ep;
        const header = `# ${{article.h}}\\n\\n**Edition:** {edition} {ordinal} Edition\\n**Volume:** {vol_num}\\n**Pages:** ${{pages}}\\n\\n---\\n\\n`;
        // Strip HTML for markdown
        const cleanText = article.t.replace(/<a[^>]*>([^<]*)<\\/a>/g, '$1');
        const blob = new Blob([header + cleanText], {{type: 'text/markdown'}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = article.h.replace(/[^a-zA-Z0-9]/g, '_') + '.md';
        a.click();
        URL.revokeObjectURL(url);
    }}

    function copyText(idx) {{
        const article = articlesData[idx];
        // Strip HTML for clipboard
        const cleanText = article.t.replace(/<a[^>]*>([^<]*)<\\/a>/g, '$1');
        navigator.clipboard.writeText(cleanText).then(() => {{
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

            let show = header.includes(query);
            if (typeFilter !== 'all') {{
                show = show && (articleType === typeFilter);
            }}

            item.style.display = show ? '' : 'none';
        }});
    }}

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

    breadcrumbs = [
        ("Home", "../index.html"),
        (f"{edition} Edition", "index.html"),
        (f"Volume {vol_num}", None)
    ]
    return generate_html_page(f"Volume {vol_num} - {edition}", content, breadcrumbs, extra_js)


def regenerate_edition(edition: str):
    """Regenerate all HTML for an edition from its JSON data."""
    docs_dir = Path('docs') / edition
    data_dir = docs_dir / 'data'

    if not data_dir.exists():
        print(f"No data directory for {edition}")
        return

    name, ordinal = EDITION_NAMES.get(edition, (f"{edition} Edition", ""))
    print(f"Regenerating {edition} ({name})...")

    volumes = {}

    # Load each volume's JSON
    for json_file in sorted(data_dir.glob('vol*.json')):
        # Skip backup/original files
        if '_original' in json_file.name or '_corrected' in json_file.name:
            continue

        vol_name = json_file.stem  # e.g., 'vol1'

        # Skip vol0 for pre-1842 editions
        if vol_name == 'vol0' and edition < '1842':
            print(f"  Skipping {vol_name} (not a real volume)")
            continue

        with open(json_file) as f:
            articles = json.load(f)

        if not articles:
            print(f"  Skipping {vol_name} (empty)")
            continue

        print(f"  {vol_name}: {len(articles)} articles")
        volumes[vol_name] = {'articles': articles}

        # Generate volume HTML
        vol_html = generate_volume_page(edition, vol_name, articles)
        output_path = docs_dir / f'{vol_name}.html'
        with open(output_path, 'w') as f:
            f.write(vol_html)

    # Generate index.html
    index_html = generate_edition_index(edition, volumes)
    with open(docs_dir / 'index.html', 'w') as f:
        f.write(index_html)

    print(f"  Generated index.html and {len(volumes)} volume files")

    # Clean up spurious vol0 files
    vol0_html = docs_dir / 'vol0.html'
    vol0_json = data_dir / 'vol0.json'
    if edition < '1842':
        if vol0_html.exists():
            vol0_html.unlink()
            print(f"  Removed spurious vol0.html")
        if vol0_json.exists():
            vol0_json.unlink()
            print(f"  Removed spurious vol0.json")


def main():
    parser = argparse.ArgumentParser(description="Regenerate HTML with lazy loading")
    parser.add_argument("edition", nargs="?", help="Edition year (e.g., 1771)")
    parser.add_argument("--all", action="store_true", help="Regenerate all editions")
    args = parser.parse_args()

    if args.all:
        for edition in sorted(EDITION_NAMES.keys()):
            docs_dir = Path('docs') / edition / 'data'
            if docs_dir.exists():
                regenerate_edition(edition)
    elif args.edition:
        regenerate_edition(args.edition)
    else:
        print("Specify an edition year or use --all")


if __name__ == "__main__":
    main()
