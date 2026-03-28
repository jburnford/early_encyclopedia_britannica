#!/usr/bin/env python3
"""Search Encyclopedia Britannica articles for timber, wood rot, naval shipbuilding,
and related topics across all 8 editions (1771-1860).

Two kinds of output:
  1. "relevant" — articles whose TOPIC is timber/rot/shipbuilding (keep whole)
  2. "chunks"  — passages extracted from large unrelated articles that happen
                  to mention these topics

Usage:
    python scripts/search_timber.py search [--output FILE]
    python scripts/search_timber.py report [--input FILE] [--output FILE]
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from config import SITE_DIR, EDITIONS, REPO_DIR

# ---------------------------------------------------------------------------
# Edition lookup
# ---------------------------------------------------------------------------
YEAR_TO_EDITION = {v["year"]: v["name"] for v in EDITIONS.values()}
EDITION_ORDER = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# ---------------------------------------------------------------------------
# Keyword definitions
# ---------------------------------------------------------------------------

# "Core" patterns — if the article is ABOUT one of these, it's relevant.
# Used to identify topical articles AND to extract chunks from other articles.
CORE_PATTERNS = [
    (r"dry[- ]rot", "dry rot"),
    (r"wet[- ]rot", "wet rot"),
    (r"(?:wood|timber)[- ]rot", "wood rot"),
    (r"ship[- ]?building", "shipbuilding"),
    (r"naval\s+architecture", "naval architecture"),
    (r"kyaniz", "kyanizing"),
    (r"creosot", "creosote"),
    (r"wood[- ]?preservation", "wood preservation"),
    (r"seasoning\s+(?:of\s+)?(?:wood|timber)", "seasoning of timber"),
    (r"timber\s+(?:shortage|scarcity|supply|famine)", "timber shortage"),
    (r"oak\s+(?:shortage|scarcity|supply)", "oak shortage"),
    (r"(?:rot|decay)\s+(?:of|in)\s+(?:wood|timber|ship)", "rot in timber"),
    (r"(?:wood|timber)\s+(?:rot|decay|preserv)", "timber decay"),
    (r"merulius", "merulius"),
    (r"serpula\s+(?:lacr[iy]mans|destruens)", "serpula lacrymans"),
    (r"(?:british|english|european|american|canadian|african)\s+oak", "oak variety"),
    (r"(?:white|red|baltic|scotch|scots)\s+pine", "pine variety"),
    (r"(?:quercus|pinus)\b", "botanical name"),
    (r"dock[- ]?yard", "dockyard"),
    (r"heart[- ]?wood", "heartwood"),
    (r"sap[- ]?wood", "sapwood"),
    (r"naval\s+timber", "naval timber"),
    (r"timber\s+trade", "timber trade"),
]
CORE_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in CORE_PATTERNS]

# Combined fast-match regex for all core patterns
CORE_FAST_RE = re.compile(
    "|".join(f"(?:{p})" for p, _ in CORE_PATTERNS), re.IGNORECASE
)

# Headwords that identify a TOPICAL article (article is *about* this subject).
# These get included whole regardless of body content.
TOPICAL_HEADWORDS = {
    "TIMBER", "OAK", "TEAK", "SHIP-BUILDING", "SHIPBUILDING", "DRY-ROT",
    "DRY ROT", "WOOD", "FORESTS", "FOREST", "LARCH", "PINE", "PINES",
    "ELM", "FIR", "CEDAR", "BEECH", "MAHOGANY", "NAVAL ARCHITECTURE",
    "DOCK-YARD", "DOCK", "FUNGUS", "FUNGI", "PUTREFACTION", "QUERCUS",
    "PINUS", "MAST", "FLOOR TIMBERS", "PRESERVATION", "PLANTING",
    "SEASONING", "NAVY", "SHIP",
}

# Secondary keywords — used to boost relevance when they co-occur with core patterns
SECONDARY_PATTERNS = [
    (r"\boak\b", "oak"),
    (r"\bteak\b", "teak"),
    (r"\bpine\b", "pine"),
    (r"\blarch\b", "larch"),
    (r"\belm\b", "elm"),
    (r"\bcedar\b", "cedar"),
    (r"\bmahogany\b", "mahogany"),
    (r"\bbeech\b", "beech"),
    (r"\btimber\b", "timber"),
    (r"\bfungus\b", "fungus"),
    (r"\bfungi\b", "fungi"),
    (r"\bnavy\b", "navy"),
    (r"\bnaval\b", "naval"),
    (r"\bseasoning\b", "seasoning"),
    (r"\bdecay(?:ed|ing)?\b", "decay"),
    (r"\brotten\b", "rotten"),
    (r"\bplank(?:s|ing)?\b", "plank"),
    (r"\bkeel\b", "keel"),
    (r"mycelium", "mycelium"),
    (r"sheathing", "sheathing"),
]
SECONDARY_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in SECONDARY_PATTERNS]


def title_to_id(title: str) -> str:
    """Convert article title to valid HTML ID (matches generate_site.py)."""
    clean = re.sub(r'[^A-Za-z0-9]+', '_', title.upper()).strip('_')
    return f"article-{clean}"


def make_link(year, vol_filename, headword):
    vol_stem = vol_filename.replace('.json', '')
    aid = title_to_id(headword)
    return f"docs/{year}/{vol_stem}.html#{aid}"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_article_files():
    """Yield (filepath, year, volume_number, vol_filename) for all editions."""
    for ed_info in EDITIONS.values():
        year = ed_info["year"]
        data_dir = SITE_DIR / str(year) / "data"
        if not data_dir.exists():
            continue
        for f in sorted(data_dir.glob("vol*.json")):
            if "index" in f.name:
                continue
            m = re.match(r'vol(\d+)', f.name)
            if m:
                yield f, year, int(m.group(1)), f.name


# ---------------------------------------------------------------------------
# Chunk extraction — get paragraphs around keyword matches
# ---------------------------------------------------------------------------

def extract_chunks(text: str, window_chars=500):
    """Find all core-pattern matches and extract surrounding text chunks.

    Returns list of {"terms": [...], "text": "...", "char_start": N, "char_end": N}
    Merges overlapping windows.
    """
    # Find all match positions
    positions = []  # (start, end, label)
    for regex, label in CORE_RE:
        for m in regex.finditer(text):
            positions.append((m.start(), m.end(), label))
    # Also check secondary patterns but only include if near a core match
    for regex, label in SECONDARY_RE:
        for m in regex.finditer(text):
            positions.append((m.start(), m.end(), label))

    if not positions:
        return []

    positions.sort(key=lambda x: x[0])

    # Build windows around each match
    windows = []
    for start, end, label in positions:
        win_start = max(0, start - window_chars)
        win_end = min(len(text), end + window_chars)
        # Snap to paragraph boundaries (\n\n) if possible
        para_start = text.rfind('\n\n', win_start, start)
        if para_start != -1 and para_start >= win_start:
            win_start = para_start + 2
        para_end = text.find('\n\n', end, win_end)
        if para_end != -1:
            win_end = para_end
        windows.append((win_start, win_end, label))

    # Merge overlapping windows
    merged = []
    for ws, we, label in windows:
        if merged and ws <= merged[-1][1] + 100:
            prev_s, prev_e, prev_labels = merged[-1]
            merged[-1] = (prev_s, max(prev_e, we), prev_labels | {label})
        else:
            merged.append((ws, we, {label}))

    # Build chunks
    chunks = []
    for ws, we, labels in merged:
        chunk_text = text[ws:we].strip()
        chunk_text = re.sub(r'\s+', ' ', chunk_text)
        if len(chunk_text) < 20:
            continue
        # Only keep chunks that contain at least one core pattern match
        if not CORE_FAST_RE.search(chunk_text):
            continue
        chunks.append({
            "terms": sorted(labels),
            "text": chunk_text,
            "char_start": ws,
            "char_end": we,
        })

    return chunks


# ---------------------------------------------------------------------------
# Classify each article
# ---------------------------------------------------------------------------

def classify_article(headword, text, word_count):
    """Classify an article as 'relevant', 'has_chunks', or None (skip).

    - 'relevant': article is topically about timber/rot/ships — keep whole
    - 'has_chunks': article is about something else but mentions our keywords
    - None: no matches at all
    """
    hw_upper = headword.upper().strip()
    is_topical_hw = hw_upper in TOPICAL_HEADWORDS

    # Fast reject: no topical headword and no core pattern anywhere in text
    if not is_topical_hw and not CORE_FAST_RE.search(text):
        return None, [], []

    # Count core pattern matches
    core_terms = set()
    for regex, label in CORE_RE:
        if regex.search(text):
            core_terms.add(label)

    secondary_terms = set()
    for regex, label in SECONDARY_RE:
        if regex.search(text):
            secondary_terms.add(label)

    all_terms = core_terms | secondary_terms

    # Decision: is this article ABOUT our topics?
    # Criteria for "relevant" (topical article):
    #   1. Headword is in TOPICAL_HEADWORDS AND article < 20K words
    #      (longer articles with topical headwords like NAVY are kept but
    #       huge ones like ASIA or BOSWORTH-MARKET are demoted to chunks)
    #   2. Short article (<5000 words) with 2+ core pattern matches
    #   3. Medium article (<15000 words) with 3+ core pattern matches
    if is_topical_hw and word_count < 20000:
        return "relevant", sorted(core_terms), sorted(secondary_terms)
    if word_count < 5000 and len(core_terms) >= 2:
        return "relevant", sorted(core_terms), sorted(secondary_terms)
    if word_count < 15000 and len(core_terms) >= 3:
        return "relevant", sorted(core_terms), sorted(secondary_terms)

    # Otherwise: extract chunks if there are core matches
    if core_terms:
        return "has_chunks", sorted(core_terms), sorted(secondary_terms)

    # Secondary-only matches in short articles
    if len(secondary_terms) >= 3 and word_count < 3000:
        return "has_chunks", sorted(core_terms), sorted(secondary_terms)

    return None, [], []


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

def cmd_search(args):
    output_path = Path(args.output)
    relevant = []
    chunk_entries = []
    articles_scanned = 0

    print(f"Scanning articles across {len(EDITIONS)} editions...")
    t0 = time.monotonic()

    for filepath, year, vol_num, vol_filename in discover_article_files():
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        for art in articles:
            headword = art.get("h", "")
            text = art.get("t", "")
            wc = art.get("wc", 0)
            tp = art.get("tp", "article")
            articles_scanned += 1

            category, core_terms, secondary_terms = classify_article(headword, text, wc)
            if category is None:
                continue

            link = make_link(year, vol_filename, headword)

            if category == "relevant":
                # Flags for parsing issues
                flags = []
                if wc > 50000:
                    flags.append("mega_article")
                elif wc > 15000:
                    caps_lines = re.findall(r'\n([A-Z][A-Z ,.;:\'-]{5,})\n', text)
                    if len(caps_lines) >= 3:
                        flags.append("possible_missed_boundaries")

                relevant.append({
                    "category": "relevant",
                    "edition_year": year,
                    "edition": YEAR_TO_EDITION.get(year, "?"),
                    "volume": vol_num,
                    "vol_file": vol_filename,
                    "headword": headword,
                    "word_count": wc,
                    "article_type": tp,
                    "link": link,
                    "core_terms": core_terms,
                    "secondary_terms": secondary_terms,
                    "flags": flags,
                })
            else:
                # Extract chunks from non-topical articles
                chunks = extract_chunks(text)
                if not chunks:
                    continue
                chunk_entries.append({
                    "category": "chunk",
                    "edition_year": year,
                    "edition": YEAR_TO_EDITION.get(year, "?"),
                    "volume": vol_num,
                    "vol_file": vol_filename,
                    "headword": headword,
                    "word_count": wc,
                    "link": link,
                    "core_terms": core_terms,
                    "secondary_terms": secondary_terms,
                    "chunks": chunks,
                })

    elapsed = time.monotonic() - t0

    # Combine and write
    all_results = relevant + chunk_entries
    all_results.sort(key=lambda r: (r["edition_year"], r["category"], r.get("headword", "")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nScanned {articles_scanned:,} articles in {elapsed:.1f}s")
    print(f"Found {len(relevant)} relevant articles + {len(chunk_entries)} articles with extracted chunks")
    print(f"Output: {output_path}")

    # Summary by edition
    print(f"\n{'Edition':<12} {'Relevant':>8} {'Chunks':>8}  Top relevant articles")
    print("-" * 75)
    for year in EDITION_ORDER:
        rel = [r for r in relevant if r["edition_year"] == year]
        chk = [r for r in chunk_entries if r["edition_year"] == year]
        top = ", ".join(r["headword"] for r in sorted(rel, key=lambda x: -x["word_count"])[:4])
        print(f"{year} {YEAR_TO_EDITION.get(year, '?'):>4}  {len(rel):>8} {len(chk):>8}  {top}")


# ---------------------------------------------------------------------------
# Report command
# ---------------------------------------------------------------------------

THEMES = {
    "Wood Rot and Preservation": {
        "description": "Dry rot, wet rot, wood preservation, seasoning, kyanizing, creosote",
        "terms": {"dry rot", "wet rot", "wood rot", "timber decay", "rot in timber",
                  "kyanizing", "creosote", "wood preservation", "seasoning of timber",
                  "merulius", "serpula lacrymans", "decay", "rotten"},
    },
    "Naval Shipbuilding and Timber Supply": {
        "description": "Ship-building, naval architecture, dockyards, timber supply, the navy",
        "terms": {"shipbuilding", "naval architecture", "dockyard", "naval timber",
                  "timber shortage", "oak shortage", "navy", "naval", "keel",
                  "plank", "sheathing"},
    },
    "Timber Species and Properties": {
        "description": "Oak (British, European, American), teak, pine, larch, elm, fir, cedar",
        "terms": {"oak", "teak", "pine", "larch", "elm", "cedar", "mahogany", "beech",
                  "oak variety", "pine variety", "botanical name", "heartwood", "sapwood",
                  "timber"},
    },
    "Forestry and Timber Trade": {
        "description": "Forest management, plantations, timber trade and supply chains",
        "terms": {"timber trade", "timber shortage", "timber"},
    },
    "Wood Science — Fungi and Decay": {
        "description": "Fungi, mycology, decomposition of wood by organisms",
        "terms": {"fungus", "fungi", "mycelium", "merulius", "serpula lacrymans",
                  "decay", "rotten"},
    },
}


def cmd_report(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]

    relevant = [r for r in results if r["category"] == "relevant"]
    chunks = [r for r in results if r["category"] == "chunk"]

    lines = []
    w = lines.append

    w("# Timber, Wood Rot & Naval Shipbuilding in the Encyclopedia Britannica (1771-1860)\n")
    w("*Auto-generated search report. Article links point to the project site.*\n")

    # --- Summary ---
    w("## Summary\n")
    w(f"- **{len(relevant)} topical articles** (directly about timber, rot, shipbuilding, etc.)")
    w(f"- **{len(chunks)} other articles** with relevant passages extracted as chunks\n")

    w("| Edition | Year | Topical | Chunks | Key Articles |")
    w("|---------|------|---------|--------|--------------|")
    for year in EDITION_ORDER:
        rel = sorted([r for r in relevant if r["edition_year"] == year],
                     key=lambda x: -x["word_count"])
        chk = [r for r in chunks if r["edition_year"] == year]
        top = ", ".join(f"[{r['headword']}]({r['link']})" for r in rel[:4])
        w(f"| {YEAR_TO_EDITION.get(year, '?')} | {year} | {len(rel)} | {len(chk)} | {top} |")
    w("")

    # --- Core article tracker ---
    w("## Core Articles Across Editions\n")
    w("Word counts for key articles in each edition (- = not found):\n")
    core_hws = ["TIMBER", "WOOD", "OAK", "DRY-ROT", "SHIP-BUILDING", "SHIP",
                "FUNGI", "PINE", "TEAK", "LARCH", "NAVY", "DOCK", "FOREST", "MAST"]

    by_year_hw = {}
    for r in relevant:
        key = (r["edition_year"], r["headword"].upper().strip())
        if key not in by_year_hw or r["word_count"] > by_year_hw[key]["word_count"]:
            by_year_hw[key] = r

    header = "| Headword | " + " | ".join(str(y) for y in EDITION_ORDER) + " |"
    sep = "|----------|" + "|".join(["------"] * len(EDITION_ORDER)) + "|"
    w(header)
    w(sep)
    for hw in core_hws:
        cells = []
        for year in EDITION_ORDER:
            r = by_year_hw.get((year, hw))
            if r:
                cells.append(f"[{r['word_count']:,}]({r['link']})")
            else:
                cells.append("-")
        w(f"| **{hw}** | " + " | ".join(cells) + " |")
    w("")

    # --- Topical articles by edition ---
    w("## All Topical Articles\n")
    for year in EDITION_ORDER:
        rel = sorted([r for r in relevant if r["edition_year"] == year],
                     key=lambda x: x["headword"])
        if not rel:
            continue
        w(f"### {year} ({YEAR_TO_EDITION.get(year, '?')} edition) — {len(rel)} articles\n")
        for r in rel:
            terms = ", ".join(r["core_terms"][:5])
            flags = f" **[{', '.join(r['flags'])}]**" if r.get("flags") else ""
            w(f"- [{r['headword']}]({r['link']}) ({r['word_count']:,}w) — {terms}{flags}")
        w("")

    # --- Thematic sections with chunks ---
    w("## Thematic Keyword Chunks\n")
    w("Relevant passages extracted from articles not primarily about these topics.\n")

    for theme_name, theme_info in THEMES.items():
        w(f"### {theme_name}\n")
        w(f"*{theme_info['description']}*\n")

        theme_terms = theme_info["terms"]
        theme_chunks = []
        for r in chunks:
            all_terms = set(r["core_terms"] + r["secondary_terms"])
            if all_terms & theme_terms:
                for chunk in r.get("chunks", []):
                    chunk_terms = set(chunk["terms"])
                    if chunk_terms & theme_terms:
                        theme_chunks.append((r, chunk))

        if not theme_chunks:
            w("*No chunks found.*\n")
            continue

        # Group by edition
        by_ed = defaultdict(list)
        for r, chunk in theme_chunks:
            by_ed[r["edition_year"]].append((r, chunk))

        for year in EDITION_ORDER:
            items = by_ed.get(year)
            if not items:
                continue
            w(f"**{year} ({YEAR_TO_EDITION.get(year, '?')})**\n")
            # Deduplicate by article
            seen_articles = set()
            for r, chunk in items:
                key = r["headword"]
                if key in seen_articles:
                    continue
                seen_articles.add(key)
                text = chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else "")
                w(f"- **[{r['headword']}]({r['link']})** ({r['word_count']:,}w)")
                w(f'  > "{text}"')
            w("")

    # --- Parsing issues ---
    flagged = [r for r in relevant if r.get("flags")]
    if flagged:
        w("## Parsing Issues\n")
        w("Articles that may have parsing errors:\n")
        for r in sorted(flagged, key=lambda a: -a["word_count"]):
            w(f"- **[{r['headword']}]({r['link']})** ({r['edition_year']}, "
              f"{r['word_count']:,}w) — {', '.join(r['flags'])}")
        w("")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Report written to {output_path}")
    print(f"  {len(relevant)} topical articles, {len(chunks)} chunk articles")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Search Britannica for timber/rot topics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Scan all articles, classify, extract chunks")
    p_search.add_argument("--output", default=str(REPO_DIR / "data" / "timber_search_results.jsonl"))

    p_report = sub.add_parser("report", help="Generate thematic Markdown report")
    p_report.add_argument("--input", default=str(REPO_DIR / "data" / "timber_search_results.jsonl"))
    p_report.add_argument("--output", default=str(REPO_DIR / "data" / "timber_report.md"))

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
