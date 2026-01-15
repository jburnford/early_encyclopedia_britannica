#!/usr/bin/env python3
"""
Regenerate HTML files for a specific edition from its JSON data.

Usage:
    python3 regenerate_html.py 1771
    python3 regenerate_html.py --all
"""

import json
import html
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
}
.container { max-width: 1000px; margin: 0 auto; padding: 2rem; }
header { background: var(--accent); color: white; padding: 1rem 2rem; }
header h1 { font-size: 1.5rem; }
header a { color: white; text-decoration: none; }
.article { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
.article h2 { color: var(--accent); margin-bottom: 0.5rem; font-size: 1.3rem; }
.article-meta { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem; }
.article-content { text-align: justify; }
.toc { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; margin: 1rem 0; }
.toc h3 { margin-bottom: 0.5rem; }
.toc-list { column-count: 3; column-gap: 1rem; }
.toc-list a { display: block; padding: 0.2rem 0; color: var(--accent); text-decoration: none; }
.toc-list a:hover { text-decoration: underline; }
nav { background: var(--bg-secondary); border-bottom: 1px solid var(--border); padding: 0.5rem 2rem; }
nav a { color: var(--accent); margin-right: 1rem; text-decoration: none; }
.stats { background: #f0f0f0; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
"""


def generate_volume_html(edition: str, vol_name: str, articles: list, edition_name: str) -> str:
    """Generate HTML for a volume."""

    # Sort articles alphabetically
    sorted_articles = sorted(articles, key=lambda a: a.get('h', '').upper())

    # Build TOC
    toc_items = []
    for a in sorted_articles:
        title = a.get('h', 'Untitled')
        anchor = html.escape(title.replace(' ', '_').replace('"', ''))
        toc_items.append(f'<a href="#{anchor}">{html.escape(title)}</a>')

    toc_html = '\n'.join(toc_items)

    # Build articles
    articles_html = []
    for a in sorted_articles:
        title = a.get('h', 'Untitled')
        anchor = html.escape(title.replace(' ', '_').replace('"', ''))
        sp = a.get('sp', '?')
        ep = a.get('ep', sp)
        content = a.get('t', '')

        # Clean content (basic HTML allowed)
        if not content.startswith('<'):
            content = f'<p>{html.escape(content)}</p>'

        articles_html.append(f'''
<div class="article" id="{anchor}">
    <h2>{html.escape(title)}</h2>
    <div class="article-meta">Pages {sp}-{ep}</div>
    <div class="article-content">{content}</div>
</div>
''')

    articles_str = '\n'.join(articles_html)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{vol_name.upper()} - {edition_name} ({edition}) - Encyclopaedia Britannica</title>
    <style>{BASE_CSS}</style>
</head>
<body>
    <header>
        <h1><a href="../index.html">Encyclopaedia Britannica</a> - {edition_name} ({edition})</h1>
    </header>
    <nav>
        <a href="index.html">← Edition Index</a>
    </nav>
    <div class="container">
        <h1>{vol_name.upper()} - {len(sorted_articles)} Articles</h1>

        <div class="toc">
            <h3>Table of Contents</h3>
            <div class="toc-list">
                {toc_html}
            </div>
        </div>

        {articles_str}
    </div>
    <footer style="text-align: center; padding: 2rem; color: var(--text-secondary);">
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} |
        <a href="https://github.com/TBD">View on GitHub</a>
    </footer>
</body>
</html>
'''


def generate_index_html(edition: str, volumes: dict, edition_name: str) -> str:
    """Generate index.html for an edition."""

    vol_links = []
    total_articles = 0

    for vol_name in sorted(volumes.keys()):
        count = volumes[vol_name]
        total_articles += count
        vol_links.append(f'<a href="{vol_name}.html">{vol_name.upper()}: {count} articles</a>')

    vol_links_html = '\n'.join(vol_links)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{edition_name} ({edition}) - Encyclopaedia Britannica</title>
    <style>{BASE_CSS}</style>
</head>
<body>
    <header>
        <h1><a href="../index.html">Encyclopaedia Britannica</a></h1>
    </header>
    <div class="container">
        <h1>{edition_name} ({edition})</h1>

        <div class="stats">
            <strong>Total Articles:</strong> {total_articles:,}<br>
            <strong>Volumes:</strong> {len(volumes)}
        </div>

        <div class="toc">
            <h3>Volumes</h3>
            <div class="toc-list" style="column-count: 1;">
                {vol_links_html}
            </div>
        </div>
    </div>
</body>
</html>
'''


def regenerate_edition(edition: str):
    """Regenerate all HTML for an edition."""

    docs_dir = Path('docs') / edition
    data_dir = docs_dir / 'data'

    if not data_dir.exists():
        print(f"No data directory for {edition}")
        return

    edition_name = EDITION_NAMES.get(edition, (f"{edition} Edition", ""))[0]

    print(f"Regenerating {edition} ({edition_name})...")

    volumes = {}

    # Process each volume JSON
    for json_file in sorted(data_dir.glob('vol*.json')):
        # Skip backup files
        if '_original' in json_file.name or '_corrected' in json_file.name:
            continue

        vol_name = json_file.stem  # e.g., 'vol1'

        # Skip vol0 for pre-1842 editions (it's a generated index, not real content)
        if vol_name == 'vol0' and edition < '1842':
            print(f"  Skipping {vol_name} (generated index)")
            continue

        with open(json_file) as f:
            articles = json.load(f)

        print(f"  {vol_name}: {len(articles)} articles")
        volumes[vol_name] = len(articles)

        # Generate volume HTML
        vol_html = generate_volume_html(edition, vol_name, articles, edition_name)

        output_path = docs_dir / f'{vol_name}.html'
        with open(output_path, 'w') as f:
            f.write(vol_html)

    # Generate index.html
    index_html = generate_index_html(edition, volumes, edition_name)
    with open(docs_dir / 'index.html', 'w') as f:
        f.write(index_html)

    print(f"  Generated index.html and {len(volumes)} volume files")


def main():
    parser = argparse.ArgumentParser(description="Regenerate HTML from JSON data")
    parser.add_argument("edition", nargs="?", help="Edition year (e.g., 1771)")
    parser.add_argument("--all", action="store_true", help="Regenerate all editions")
    args = parser.parse_args()

    if args.all:
        for edition in sorted(EDITION_NAMES.keys()):
            regenerate_edition(edition)
    elif args.edition:
        regenerate_edition(args.edition)
    else:
        print("Specify an edition year or use --all")


if __name__ == "__main__":
    main()
