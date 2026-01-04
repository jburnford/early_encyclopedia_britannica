#!/usr/bin/env python3
"""
Generate static GitHub Pages site for Encyclopaedia Britannica corpus.

OPTIMIZED VERSION - Handles 125K+ articles and 500K+ hyperlinks without memory exhaustion.

Key optimizations:
1. Aho-Corasick algorithm for O(n+m) multi-pattern matching (vs exponential regex)
2. Per-edition processing with garbage collection between editions
3. Streaming article lookup - doesn't hold all article text in memory
4. Pre-compiled pattern automaton (built once, used 125K times)
"""

import json
import html
import os
import re
import gc
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Set, Dict, List, Tuple, Optional

# Try to import ahocorasick, fall back to regex if not available
try:
    import ahocorasick
    HAVE_AHOCORASICK = True
except ImportError:
    HAVE_AHOCORASICK = False
    print("WARNING: pyahocorasick not installed. Using slower regex fallback.")
    print("Install with: pip install pyahocorasick")

# Configuration
INPUT_DIR = Path("output_v2")
OUTPUT_DIR = Path("docs")
EDITIONS_TO_INCLUDE = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

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

# Hyperlink filtering - only link meaningful article types
# Types that should be hyperlinked (places, people, major topics)
LINKABLE_ARTICLE_TYPES = {"geographical", "biographical", "treatise"}

# Minimum text length for "unknown" type articles to be linkable
MIN_UNKNOWN_LENGTH = 2000  # ~1.5 pages of text

# HTML Templates - identical to original
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


def headword_to_id(headword: str) -> str:
    """Convert headword to valid HTML ID for deep-linking."""
    clean = re.sub(r'[^A-Za-z0-9]+', '_', headword.upper())
    clean = clean.strip('_')
    return f"article-{clean}"


class HyperlinkInjector:
    """
    Efficient hyperlink injection using Aho-Corasick algorithm.

    Instead of building a regex from 5000+ terms for EACH article (O(n*m) complexity),
    this builds an automaton ONCE and finds all matches in O(n+m) time.

    Memory: ~50MB for the automaton vs ~100KB per regex pattern * 125K articles
    Speed: Linear scan vs potential catastrophic backtracking
    """

    def __init__(self, index_headwords: Set[str], article_lookup: Dict[str, Dict[int, int]],
                 linkable_headwords: Set[str]):
        """
        Initialize the hyperlink injector.

        Args:
            index_headwords: Set of valid link targets from 1842 index (uppercase)
            article_lookup: Mapping of headword -> {year: volume}
            linkable_headwords: Set of headwords that are linkable (based on article type)
        """
        self.article_lookup = article_lookup

        # Only link headwords that:
        # 1. Are in the index
        # 2. Exist as articles
        # 3. Are marked as linkable (geographical, biographical, treatise, or long unknown)
        self.valid_headwords = {
            hw for hw in index_headwords
            if hw in article_lookup and hw in linkable_headwords
        }
        print(f"    Building hyperlink automaton for {len(self.valid_headwords):,} terms...")

        if HAVE_AHOCORASICK:
            self._build_automaton()
        else:
            self._build_regex_fallback()

    def _build_automaton(self):
        """Build Aho-Corasick automaton for fast multi-pattern matching."""
        self.automaton = ahocorasick.Automaton()

        # Add all valid headwords (store lowercase for case-insensitive matching)
        for hw in self.valid_headwords:
            # Store original uppercase version as value
            self.automaton.add_word(hw.lower(), hw)

        self.automaton.make_automaton()
        print(f"    Automaton built with {len(self.automaton)} patterns")

    def _build_regex_fallback(self):
        """Build compiled regex as fallback (slower but works without pyahocorasick)."""
        # Sort by length descending to match longer terms first
        sorted_terms = sorted(self.valid_headwords, key=len, reverse=True)

        # Chunk into batches to avoid regex complexity limits
        self.regex_patterns = []
        chunk_size = 1000

        for i in range(0, len(sorted_terms), chunk_size):
            chunk = sorted_terms[i:i + chunk_size]
            escaped = [re.escape(t) for t in chunk]
            pattern = r'\b(' + '|'.join(escaped) + r')\b'
            self.regex_patterns.append(re.compile(pattern, re.IGNORECASE))

        print(f"    Built {len(self.regex_patterns)} regex patterns (fallback mode)")

    def inject(self, text: str, current_headword: str, edition_year: int) -> str:
        """
        Inject hyperlinks into article text.

        Args:
            text: Article body text
            current_headword: Headword of current article (to avoid self-linking)
            edition_year: Current edition year

        Returns:
            Text with <a class="xref"> tags for first occurrence of each linked term.
        """
        if not self.valid_headwords:
            return text

        current_upper = current_headword.upper().strip()

        if HAVE_AHOCORASICK:
            return self._inject_ahocorasick(text, current_upper, edition_year)
        else:
            return self._inject_regex_fallback(text, current_upper, edition_year)

    def _inject_ahocorasick(self, text: str, current_upper: str, edition_year: int) -> str:
        """Inject hyperlinks using Aho-Corasick (fast path)."""
        text_lower = text.lower()
        linked_terms = set()

        # Find all matches with their positions
        matches = []
        for end_idx, original_hw in self.automaton.iter(text_lower):
            start_idx = end_idx - len(original_hw) + 1

            # Check word boundaries
            if start_idx > 0 and text_lower[start_idx - 1].isalnum():
                continue
            if end_idx + 1 < len(text_lower) and text_lower[end_idx + 1].isalnum():
                continue

            # Skip if already linked this term or it's the current article
            if original_hw in linked_terms:
                continue
            if original_hw == current_upper:
                continue

            # Find target location
            locations = self.article_lookup.get(original_hw, {})
            if not locations:
                continue

            if edition_year in locations:
                target_year = edition_year
                target_vol = locations[edition_year]
            elif 1842 in locations:
                target_year = 1842
                target_vol = locations[1842]
            else:
                target_year = min(locations.keys())
                target_vol = locations[target_year]

            linked_terms.add(original_hw)
            matches.append((start_idx, end_idx + 1, original_hw, target_year, target_vol))

        # Sort by position (descending) to replace from end to start
        matches.sort(key=lambda x: x[0], reverse=True)

        # Apply replacements
        result = text
        for start, end, hw, target_year, target_vol in matches:
            original_text = result[start:end]
            article_id = headword_to_id(hw)

            if target_year == edition_year:
                url = f"vol{target_vol}.html#{article_id}"
            else:
                url = f"../{target_year}/vol{target_vol}.html#{article_id}"

            replacement = f'<a class="xref" href="{url}">{original_text}</a>'
            result = result[:start] + replacement + result[end:]

        return result

    def _inject_regex_fallback(self, text: str, current_upper: str, edition_year: int) -> str:
        """Inject hyperlinks using regex (slow fallback)."""
        linked_terms = set()

        def replace_match(match):
            term = match.group(0)
            term_upper = term.upper()

            if term_upper in linked_terms:
                return term
            if term_upper == current_upper:
                return term
            if term_upper not in self.article_lookup:
                return term

            locations = self.article_lookup[term_upper]
            if edition_year in locations:
                target_year = edition_year
                target_vol = locations[edition_year]
            elif 1842 in locations:
                target_year = 1842
                target_vol = locations[1842]
            else:
                target_year = min(locations.keys())
                target_vol = locations[target_year]

            linked_terms.add(term_upper)
            article_id = headword_to_id(term_upper)

            if target_year == edition_year:
                url = f"vol{target_vol}.html#{article_id}"
            else:
                url = f"../{target_year}/vol{target_vol}.html#{article_id}"

            return f'<a class="xref" href="{url}">{term}</a>'

        result = text
        for pattern in self.regex_patterns:
            try:
                result = pattern.sub(replace_match, result)
            except re.error:
                continue

        return result


def load_index_headwords() -> Set[str]:
    """Load main entry headwords from 1842 index as link targets."""
    index_path = INPUT_DIR / "index_1842.jsonl"
    if not index_path.exists():
        print(f"Warning: Index file not found: {index_path}")
        return set()

    headwords = set()
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get('entry_type') == 'main' and entry.get('references'):
                    term = entry.get('term', '').upper().strip()
                    if len(term) > 1:
                        headwords.add(term)

    return headwords


def build_article_lookup_streaming() -> Tuple[Dict[str, Dict[int, int]], Set[str]]:
    """
    Build article lookup by streaming through JSONL files.

    This only extracts headword -> {year: volume} mapping without loading article text,
    keeping memory usage low (~20MB vs ~600MB).

    Returns:
        Tuple of (lookup dict, set of linkable headwords based on article type)
    """
    lookup = defaultdict(dict)
    linkable = set()

    for year in EDITIONS_TO_INCLUDE:
        path = INPUT_DIR / f"articles_{year}.jsonl"
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # Parse only the fields we need
                    article = json.loads(line)
                    headword = article.get('headword', '').upper().strip()
                    if headword:
                        vol_num = article.get('volume_num', 0)
                        if year not in lookup[headword]:
                            lookup[headword][year] = vol_num

                        # Check if this article type is linkable
                        article_type = article.get('article_type', 'unknown')
                        if article_type in LINKABLE_ARTICLE_TYPES:
                            linkable.add(headword)
                        elif article_type == 'unknown':
                            # For unknown types, only link if substantial length
                            text_len = len(article.get('text', ''))
                            if text_len >= MIN_UNKNOWN_LENGTH:
                                linkable.add(headword)

    return dict(lookup), linkable


def load_articles_streaming(year: int):
    """Generator to stream articles for a given edition year."""
    path = INPUT_DIR / f"articles_{year}.jsonl"
    if not path.exists():
        return

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_articles(year: int) -> List[dict]:
    """Load all articles for a given edition year (for stats/search index)."""
    return list(load_articles_streaming(year))


def load_volumes(year: int) -> List[dict]:
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
                <th>Biographical</th>
                <th>Geographical</th>
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
                <td>{stats[y]['biographical']:,}</td>
                <td>{stats[y]['geographical']:,}</td>
            </tr>
        """ for y in EDITIONS_TO_INCLUDE if y in stats)}
        </tbody>
    </table>
    """

    return generate_html_page("Home", content)


def generate_edition_page(year, volumes, articles):
    """Generate edition index page."""
    name, ordinal = EDITION_NAMES.get(year, (f"{year} Edition", ""))

    vol_articles = defaultdict(list)
    for a in articles:
        vol_articles[a.get('volume_num', 0)].append(a)

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

    sorted_articles = sorted(articles, key=lambda a: a.get('headword', '').upper())

    article_items = []
    for i, a in enumerate(sorted_articles):
        headword = a.get('headword', 'Unknown')
        article_type = a.get('article_type', 'dictionary')
        start_page = a.get('start_page', '?')
        end_page = a.get('end_page', '?')
        word_count = a.get('word_count', 0)

        pages = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"

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
        let safe = escapeHtml(text);
        // Match escaped HTML anchor tags (quotes stay as quotes, only < and > are escaped)
        safe = safe.replace(
            /&lt;a class="xref" href="([^"]+)"&gt;([^&]+)&lt;\\/a&gt;/g,
            '<a class="xref" href="$1">$2</a>'
        );
        return safe;
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
        (f"{year} Edition", "index.html"),
        (f"Volume {vol_num}", None)
    ]
    return generate_html_page(f"Volume {vol_num} - {year}", content, breadcrumbs, extra_js)


def generate_volume_data(articles: List[dict], edition_year: int,
                         injector: Optional[HyperlinkInjector] = None) -> List[dict]:
    """
    Generate compact JSON data for a volume's articles.

    Args:
        articles: List of article dicts
        edition_year: Year of this edition
        injector: Pre-built HyperlinkInjector (or None to skip hyperlinks)
    """
    data = []
    sorted_articles = sorted(articles, key=lambda a: a.get('headword', '').upper())

    for a in sorted_articles:
        text = a.get('text', '')
        headword = a.get('headword', '')

        # Inject hyperlinks if injector available
        if injector:
            text = injector.inject(text, headword, edition_year)

        data.append({
            "h": headword,
            "t": text,
            "sp": a.get('start_page'),
            "ep": a.get('end_page'),
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


def generate_search_index_streaming() -> List[list]:
    """Generate compact search index by streaming through files."""
    index = []
    seen = set()

    for year in EDITIONS_TO_INCLUDE:
        for article in load_articles_streaming(year):
            headword = article.get('headword', '')[:100]
            vol = article.get('volume_num', 0)
            start_page = article.get('start_page', '?')
            end_page = article.get('end_page', '?')
            pages = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"

            key = (headword.lower(), year, vol)
            if key in seen:
                continue
            seen.add(key)

            index.append([headword, year, vol, pages])

    return index


def process_edition(year: int, volumes: List[dict], injector: Optional[HyperlinkInjector],
                    edition_dir: Path, stats: dict) -> dict:
    """
    Process a single edition - load articles, generate pages, write output.

    This is the memory-critical function - we process one edition at a time
    and release memory before moving to the next.
    """
    print(f"  Processing {year} edition...")

    # Load articles for this edition
    articles = load_articles(year)
    if not articles:
        print(f"    No articles found for {year}, skipping")
        return None

    # Calculate stats
    treatises = sum(1 for a in articles if a.get('article_type') == 'treatise')
    biographical = sum(1 for a in articles if a.get('article_type') == 'biographical')
    geographical = sum(1 for a in articles if a.get('article_type') == 'geographical')

    stats[year] = {
        'volumes': len(set(v.get('volume_num', 0) for v in volumes)),
        'articles': len(articles),
        'treatises': treatises,
        'biographical': biographical,
        'geographical': geographical,
    }
    print(f"    {len(articles):,} articles, {treatises:,} treatises")

    # Create directories
    edition_dir.mkdir(exist_ok=True)
    data_dir = edition_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Edition index page
    with open(edition_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(generate_edition_page(year, volumes, articles))

    # Group articles by volume
    vol_articles = defaultdict(list)
    for a in articles:
        vol_articles[a.get('volume_num', 0)].append(a)

    # Process each volume
    link_count = 0
    for vol_num, vol_arts in vol_articles.items():
        vol_info = next((v for v in volumes if v.get('volume_num') == vol_num), {})

        # HTML page
        with open(edition_dir / f"vol{vol_num}.html", 'w', encoding='utf-8') as f:
            f.write(generate_volume_page(year, vol_num, vol_arts, vol_info))

        # JSON data file with hyperlinks
        vol_data = generate_volume_data(vol_arts, year, injector)

        # Count links for stats
        for item in vol_data:
            link_count += item['t'].count('class="xref"')

        with open(data_dir / f"vol{vol_num}.json", 'w', encoding='utf-8') as f:
            json.dump(vol_data, f, separators=(',', ':'))

    print(f"    {len(vol_articles)} volumes, {link_count:,} hyperlinks")

    return stats[year]


def main():
    print("=" * 60)
    print("Encyclopaedia Britannica Site Generator (OPTIMIZED)")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "api").mkdir(exist_ok=True)

    # Phase 1: Build article lookup (streaming, low memory)
    print("\nPhase 1: Building article lookup (streaming)...")
    article_lookup, linkable_headwords = build_article_lookup_streaming()
    print(f"  {len(article_lookup):,} unique headwords")
    print(f"  {len(linkable_headwords):,} linkable (geo/bio/treatise)")

    # Phase 2: Load index headwords
    print("\nPhase 2: Loading index headwords...")
    index_headwords = load_index_headwords()
    print(f"  {len(index_headwords):,} index terms")

    # Phase 3: Build hyperlink injector (Aho-Corasick automaton)
    print("\nPhase 3: Building hyperlink injector...")
    injector = HyperlinkInjector(index_headwords, article_lookup, linkable_headwords)

    # Phase 4: Load volume metadata (lightweight)
    print("\nPhase 4: Loading volume metadata...")
    all_volumes = {}
    for year in EDITIONS_TO_INCLUDE:
        all_volumes[year] = load_volumes(year)

    # Phase 5: Generate search index (streaming)
    print("\nPhase 5: Generating search index...")
    search_index = generate_search_index_streaming()
    with open(OUTPUT_DIR / "api" / "index.json", 'w', encoding='utf-8') as f:
        json.dump(search_index, f, separators=(',', ':'))
    print(f"  {len(search_index):,} entries")

    # Force garbage collection before heavy processing
    gc.collect()

    # Phase 6: Process editions one at a time
    print("\nPhase 6: Processing editions...")
    stats = {}

    for year in EDITIONS_TO_INCLUDE:
        if year not in all_volumes:
            continue

        volumes = all_volumes[year]
        edition_dir = OUTPUT_DIR / str(year)

        process_edition(year, volumes, injector, edition_dir, stats)

        # Force garbage collection after each edition
        gc.collect()

    # Phase 7: Generate static pages
    print("\nPhase 7: Generating static pages...")

    with open(OUTPUT_DIR / "index.html", 'w', encoding='utf-8') as f:
        f.write(generate_index_page(stats))
    print("  index.html")

    with open(OUTPUT_DIR / "search.html", 'w', encoding='utf-8') as f:
        f.write(generate_search_page())
    print("  search.html")

    with open(OUTPUT_DIR / "about.html", 'w', encoding='utf-8') as f:
        f.write(generate_about_page())
    print("  about.html")

    # Summary
    total_articles = sum(s['articles'] for s in stats.values())
    print("\n" + "=" * 60)
    print(f"COMPLETE: {total_articles:,} articles across {len(stats)} editions")
    print(f"Output: {OUTPUT_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
