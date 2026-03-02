"""Export final deduplicated dataset as per-edition JSONL and SQLite.

Consolidates articles from canonical files (post-dedup) into clean export formats:
- Per-edition JSONL files (one line per article, sorted by volume/position)
- SQLite database with FTS5 full-text search
- Statistics summary JSON
"""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

from config import (
    INPUT_DIR, ARTICLES_DIR, DEDUP_MANIFEST, OCR_MANIFEST, EXPORT_DIR, EDITIONS, ensure_dirs,
)

log = logging.getLogger(__name__)


def get_canonical_files() -> list[str]:
    """Get canonical filenames from OCR manifest, dedup manifest, or all files."""
    # Prefer new OCR manifest (has correct volume assignments)
    if OCR_MANIFEST.exists():
        with open(OCR_MANIFEST) as f:
            manifest = json.load(f)
        canonical = sorted([
            e['filename'] for e in manifest.get('files', [])
            if e.get('is_canonical', True)
        ])
        if canonical:
            log.info(f"Using OCR manifest: {len(canonical)} canonical files")
            return canonical

    # Fall back to legacy dedup manifest
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        canonical = manifest.get('canonical', [])
        if canonical:
            log.info(f"Using dedup manifest: {len(canonical)} canonical files")
            return canonical

    log.warning("No manifest found — using all article files")
    return [p.name for p in sorted(INPUT_DIR.glob('*.jsonl'))]


def load_all_articles(canonical_files: list[str]) -> dict[str, list[dict]]:
    """Load all articles from canonical files, grouped by edition.

    Returns dict mapping edition name to list of articles.
    """
    by_edition = defaultdict(list)

    for filename in sorted(canonical_files):
        stem = filename.replace('.jsonl', '')
        articles_path = ARTICLES_DIR / f"{stem}.articles.jsonl"

        if not articles_path.exists():
            log.warning(f"Article file not found: {articles_path}")
            continue

        with open(articles_path) as f:
            for line in f:
                article = json.loads(line)
                by_edition[article['edition']].append(article)

    # Sort each edition by volume, then by char_start within volume
    for edition in by_edition:
        by_edition[edition].sort(key=lambda a: (a['volume'], a['char_start']))

    return dict(by_edition)


def renumber_articles(articles: list[dict], edition: str, year: int) -> list[dict]:
    """Re-number article IDs sequentially within an edition.

    Format: eb_{edition}_{year}_{sequential_number:06d}
    """
    counter = 0
    for article in articles:
        counter += 1
        article['article_id'] = f"eb_{edition}_{year}_{counter:06d}"
    return articles


def export_jsonl(by_edition: dict[str, list[dict]]):
    """Export per-edition JSONL files."""
    for edition, articles in sorted(by_edition.items()):
        year = EDITIONS.get(edition, {}).get('year', 0)
        filename = f"eb_{edition}_{year}.jsonl"
        output_path = EXPORT_DIR / filename

        with open(output_path, 'w') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        log.info(f"Exported {filename}: {len(articles):,} entries")


def export_sqlite(by_edition: dict[str, list[dict]]):
    """Export SQLite database with FTS5 full-text search."""
    db_path = EXPORT_DIR / 'britannica.db'

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Enable WAL mode for better write performance
    cur.execute("PRAGMA journal_mode=WAL")

    # Create tables
    cur.execute("""
        CREATE TABLE editions (
            edition TEXT PRIMARY KEY,
            year INTEGER NOT NULL,
            full_name TEXT,
            article_count INTEGER,
            cross_ref_count INTEGER,
            word_count INTEGER,
            volume_count INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE articles (
            article_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            edition TEXT NOT NULL,
            edition_year INTEGER NOT NULL,
            volume INTEGER NOT NULL,
            source_file TEXT,
            type TEXT NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            text TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            paragraph_count INTEGER,
            keywords TEXT,
            author_attribution TEXT,
            target TEXT,
            subsections TEXT,
            FOREIGN KEY (edition) REFERENCES editions(edition)
        )
    """)

    cur.execute("""
        CREATE TABLE cross_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            target TEXT,
            edition TEXT NOT NULL,
            source_article_id TEXT,
            FOREIGN KEY (edition) REFERENCES editions(edition),
            FOREIGN KEY (source_article_id) REFERENCES articles(article_id)
        )
    """)

    # Create indexes
    cur.execute("CREATE INDEX idx_articles_edition ON articles(edition)")
    cur.execute("CREATE INDEX idx_articles_volume ON articles(edition, volume)")
    cur.execute("CREATE INDEX idx_articles_title ON articles(title)")
    cur.execute("CREATE INDEX idx_articles_type ON articles(type)")
    cur.execute("CREATE INDEX idx_xref_edition ON cross_references(edition)")
    cur.execute("CREATE INDEX idx_xref_target ON cross_references(target)")

    # Create FTS5 virtual table
    cur.execute("""
        CREATE VIRTUAL TABLE articles_fts USING fts5(
            title, text, edition,
            content='articles',
            content_rowid='rowid'
        )
    """)

    # Insert data
    total_inserted = 0
    for edition, articles in sorted(by_edition.items()):
        year = EDITIONS.get(edition, {}).get('year', 0)
        full_name = EDITIONS.get(edition, {}).get('full_name', '')

        real_articles = [a for a in articles if a['type'] == 'article']
        cross_refs = [a for a in articles if a['type'] == 'cross_reference']
        total_words = sum(a['word_count'] for a in articles)
        volumes = set(a['volume'] for a in articles)

        # Insert edition metadata
        cur.execute("""
            INSERT INTO editions VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            edition, year, full_name,
            len(real_articles), len(cross_refs),
            total_words, len(volumes),
        ))

        # Insert articles
        for article in articles:
            keywords_json = json.dumps(article.get('keywords')) if article.get('keywords') else None
            subsections_json = json.dumps(article.get('subsections')) if article.get('subsections') else None

            cur.execute("""
                INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article['article_id'],
                article['title'],
                article['edition'],
                article['edition_year'],
                article['volume'],
                article.get('source_file'),
                article['type'],
                article.get('char_start'),
                article.get('char_end'),
                article['text'],
                article['word_count'],
                article.get('paragraph_count'),
                keywords_json,
                article.get('author_attribution'),
                article.get('target'),
                subsections_json,
            ))
            total_inserted += 1

            # Insert cross-references
            if article['type'] == 'cross_reference':
                cur.execute("""
                    INSERT INTO cross_references (title, target, edition, source_article_id)
                    VALUES (?, ?, ?, ?)
                """, (
                    article['title'],
                    article.get('target'),
                    article['edition'],
                    article['article_id'],
                ))

    # Populate FTS index
    cur.execute("""
        INSERT INTO articles_fts(articles_fts) VALUES('rebuild')
    """)

    conn.commit()

    # Report size
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    log.info(f"SQLite database: {db_path} ({db_size_mb:.1f} MB, "
             f"{total_inserted:,} articles)")

    # Verify
    cur.execute("SELECT count(*) FROM articles")
    count = cur.fetchone()[0]
    cur.execute("SELECT edition, count(*) FROM articles GROUP BY edition ORDER BY edition")
    edition_counts = cur.fetchall()

    log.info(f"Verification: {count:,} articles in database")
    for ed, c in edition_counts:
        log.info(f"  {ed}: {c:,}")

    # Test FTS
    cur.execute("""
        SELECT title, edition FROM articles_fts
        WHERE articles_fts MATCH 'ABACUS' LIMIT 5
    """)
    fts_results = cur.fetchall()
    if fts_results:
        log.info(f"FTS test 'ABACUS': {fts_results}")
    else:
        log.info("FTS test 'ABACUS': no results (may not exist in dataset)")

    conn.close()


def export_statistics(by_edition: dict[str, list[dict]], dedup_stats: dict | None):
    """Export statistics summary."""
    stats = {
        'overall': {},
        'per_edition': {},
        'per_volume': {},
    }

    total_articles = 0
    total_xrefs = 0
    total_words = 0
    total_volumes = set()

    for edition, articles in sorted(by_edition.items()):
        year = EDITIONS.get(edition, {}).get('year', 0)
        real = [a for a in articles if a['type'] == 'article']
        xrefs = [a for a in articles if a['type'] == 'cross_reference']
        words = sum(a['word_count'] for a in articles)
        volumes = set(a['volume'] for a in articles)

        total_articles += len(real)
        total_xrefs += len(xrefs)
        total_words += words
        total_volumes.update(f"{edition}_v{v}" for v in volumes)

        stats['per_edition'][edition] = {
            'year': year,
            'articles': len(real),
            'cross_references': len(xrefs),
            'total_entries': len(articles),
            'word_count': words,
            'volume_count': len(volumes),
        }

        # Per-volume breakdown
        vol_articles = defaultdict(list)
        for a in articles:
            vol_articles[a['volume']].append(a)

        for vol, vol_arts in sorted(vol_articles.items()):
            key = f"{edition}_vol{vol:02d}"
            stats['per_volume'][key] = {
                'edition': edition,
                'volume': vol,
                'articles': len([a for a in vol_arts if a['type'] == 'article']),
                'cross_references': len([a for a in vol_arts if a['type'] == 'cross_reference']),
                'word_count': sum(a['word_count'] for a in vol_arts),
            }

    stats['overall'] = {
        'total_articles': total_articles,
        'total_cross_references': total_xrefs,
        'total_entries': total_articles + total_xrefs,
        'total_words': total_words,
        'total_volumes': len(total_volumes),
        'editions': len(by_edition),
    }

    if dedup_stats:
        stats['dedup'] = dedup_stats

    stats_path = EXPORT_DIR / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Statistics written to {stats_path}")

    return stats


def run(files: list[Path] | None = None):
    """Export final deduplicated dataset."""
    ensure_dirs()

    # Get canonical files
    canonical_files = get_canonical_files()

    # If specific files were passed, filter canonical to those
    if files is not None:
        file_stems = {f.stem for f in files}
        canonical_files = [
            cf for cf in canonical_files
            if cf.replace('.jsonl', '') in file_stems
        ]

    log.info(f"Loading articles from {len(canonical_files)} canonical files...")

    # Load dedup stats
    dedup_stats = None
    if DEDUP_MANIFEST.exists():
        with open(DEDUP_MANIFEST) as f:
            manifest = json.load(f)
        dedup_stats = manifest.get('stats')

    # Load all articles
    by_edition = load_all_articles(canonical_files)

    if not by_edition:
        log.error("No articles loaded — check that article files exist in "
                   f"{ARTICLES_DIR}")
        return None

    # Re-number articles sequentially per edition
    for edition, articles in by_edition.items():
        year = EDITIONS.get(edition, {}).get('year', 0)
        renumber_articles(articles, edition, year)

    total_entries = sum(len(arts) for arts in by_edition.values())
    log.info(f"Loaded {total_entries:,} entries across {len(by_edition)} editions")

    # Export JSONL
    log.info("Exporting per-edition JSONL files...")
    export_jsonl(by_edition)

    # Export SQLite
    log.info("Exporting SQLite database...")
    export_sqlite(by_edition)

    # Export statistics
    log.info("Exporting statistics...")
    stats = export_statistics(by_edition, dedup_stats)

    # Print summary
    s = stats['overall']
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Editions:          {s['editions']}")
    print(f"Total articles:    {s['total_articles']:,}")
    print(f"Total cross-refs:  {s['total_cross_references']:,}")
    print(f"Total entries:     {s['total_entries']:,}")
    print(f"Total words:       {s['total_words']:,}")
    print()

    print("Per-edition breakdown:")
    for edition, info in sorted(stats['per_edition'].items()):
        print(f"  {edition} ({info['year']}): "
              f"{info['articles']:,} articles, "
              f"{info['cross_references']:,} xrefs, "
              f"{info['word_count']:,} words, "
              f"{info['volume_count']} volumes")

    print(f"\nExport directory: {EXPORT_DIR}")
    print(f"Files:")
    for p in sorted(EXPORT_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name} ({size_mb:.1f} MB)")

    if dedup_stats:
        print(f"\nDedup: {dedup_stats['total_files']} source files → "
              f"{dedup_stats['canonical_files']} canonical "
              f"({dedup_stats['duplicate_files']} duplicates removed)")

    print(f"{'='*60}\n")

    return stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    run()
