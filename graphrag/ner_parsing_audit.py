#!/usr/bin/env python3
"""NER-guided parsing error detection and repair.

Uses NER entity data to detect bloated articles (cross-edition anomalies),
diagnoses what headwords they absorbed via the concept index, localizes
missing headwords in OCR text, and outputs candidates compatible with the
parser's supplementary_injection() mechanism.

Usage:
    python3 graphrag/ner_parsing_audit.py [--min-bloat-ratio 10] [--min-entities 50]
"""

import argparse
import json
import glob
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median as stat_median


# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
NER_DIR = REPO_ROOT / "data" / "ner"
CONCEPT_INDEX_PATH = REPO_ROOT / "graphrag" / "concept_index.json"
EXPORT_DIR = REPO_ROOT / "data" / "export"
OCR_DIR = REPO_ROOT / "data" / "ocr" / "organized"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "ner_audit_candidates.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "ner_parsing_audit_report.txt"


# ---------------------------------------------------------------------------
# Sort key normalization (mirrors lis_parser.py)
# ---------------------------------------------------------------------------

def normalize_sort_key(headword: str) -> str:
    key = headword.upper()
    key = key.replace("U", "V").replace("I", "J")
    key = unicodedata.normalize("NFKD", key)
    key = key.encode("ASCII", "ignore").decode("ASCII")
    key = re.sub(r"['\-]", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BloatedArticle:
    title: str
    edition: str
    edition_year: int
    volume: int
    entity_count: int
    median: float
    ratio: float
    article_id: str = ""


@dataclass
class AbsorbedHeadword:
    headword: str
    expected_wc: int  # word count from concept index
    edition_count: int  # how many editions it appears in
    in_this_edition: bool


@dataclass
class Candidate:
    file: str
    edition: int
    vol: str
    candidate: str
    position: int
    pct: float
    before: str
    match: str
    after: str
    alpha_check: str
    prev_article: str
    next_article: str
    containing_article: str
    source: str = "ner_audit"
    bloat_ratio: float = 0.0
    confidence: str = "HIGH"
    expected_wc: int = 0


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def load_ner_data(ner_dir: Path) -> dict[str, list[dict]]:
    """Load all NER entity files, grouped by uppercase title."""
    ner_by_title = defaultdict(list)
    files = sorted(ner_dir.glob("eb_*_v*.entities.jsonl"))
    total = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["_total_entities"] = sum(rec["entity_counts"].values())
                ner_by_title[rec["title"].upper()].append(rec)
                total += 1
    print(f"  Loaded {total:,} NER articles from {len(files)} files")
    return ner_by_title


def load_concept_index(path: Path) -> dict:
    with open(path) as f:
        ci = json.load(f)
    print(f"  Loaded concept index: {len(ci):,} headwords")
    return ci


def load_export_data(export_dir: Path) -> tuple[dict, dict]:
    """Load export articles. Returns:
    - export_by_key: {(TITLE, edition_year): record}
    - export_by_vol: {(edition_year, volume): [records sorted by char_start]}
    """
    export_by_key = {}
    export_by_vol = defaultdict(list)
    files = sorted(export_dir.glob("eb_*.jsonl"))
    total = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                # Don't load full text — too much memory
                text_preview = rec.get("text", "")[:200]
                rec_slim = {
                    "article_id": rec["article_id"],
                    "title": rec["title"],
                    "edition": rec["edition"],
                    "edition_year": rec["edition_year"],
                    "volume": rec["volume"],
                    "source_file": rec["source_file"],
                    "char_start": rec["char_start"],
                    "char_end": rec["char_end"],
                    "word_count": rec["word_count"],
                    "type": rec["type"],
                    "text_preview": text_preview,
                }
                key = (rec["title"].upper(), rec["edition_year"])
                export_by_key[key] = rec_slim
                export_by_vol[(rec["edition_year"], rec["volume"])].append(rec_slim)
                total += 1
    # Sort each volume by char_start
    for vol_key in export_by_vol:
        export_by_vol[vol_key].sort(key=lambda r: r["char_start"])
    print(f"  Loaded {total:,} export articles from {len(files)} files")
    return export_by_key, export_by_vol


def load_ocr_text(ocr_dir: Path, source_file: str, _cache: dict = {}) -> str:
    """Load OCR text for a given source file (cached)."""
    if source_file in _cache:
        return _cache[source_file]
    path = ocr_dir / source_file
    if not path.exists():
        return ""
    with open(path) as f:
        data = json.loads(f.readline())
    text = data.get("text", "")
    _cache[source_file] = text
    return text


# ---------------------------------------------------------------------------
# Step 2: Detect bloated articles
# ---------------------------------------------------------------------------

def detect_bloated(
    ner_by_title: dict[str, list[dict]],
    export_by_key: dict,
    concept_index: dict,
    min_ratio: float = 10.0,
    min_entities: int = 50,
    min_editions: int = 2,
) -> list[BloatedArticle]:
    """Find articles anomalously large vs their cross-edition median.

    Two detection paths:
    1. Cross-edition NER: entity count >= 10x median across editions
    2. Word count vs concept index: export word count >= 20x concept index max
       (catches single-edition articles like IVAHAI, TZETZES, EGREMONT)
    """
    bloated = []
    seen = set()  # (title, edition_year)

    # Path 1: Cross-edition NER anomaly
    for title, articles in ner_by_title.items():
        if len(articles) < min_editions:
            continue
        counts = [a["_total_entities"] for a in articles]
        med = stat_median(counts)
        effective_med = max(med, 1)
        for a in articles:
            ratio = a["_total_entities"] / effective_med
            if ratio >= min_ratio and a["_total_entities"] >= min_entities:
                bloated.append(BloatedArticle(
                    title=title,
                    edition=a["edition"],
                    edition_year=a["edition_year"],
                    volume=a["volume"],
                    entity_count=a["_total_entities"],
                    median=med,
                    ratio=ratio,
                    article_id=a.get("article_id", ""),
                ))
                seen.add((title, a["edition_year"]))

    # Path 2: Word count vs concept index OTHER editions
    # (for single/few-edition articles like IVAHAI, TZETZES, EGREMONT)
    # Compare this edition's word count against median of OTHER editions.
    # If this edition is 20x+ bigger, it's bloated.
    for title, articles in ner_by_title.items():
        for a in articles:
            if (title, a["edition_year"]) in seen:
                continue
            if a["_total_entities"] < 200:
                continue
            ci_key = normalize_sort_key(title)
            ci_entry = concept_index.get(title) or concept_index.get(ci_key)
            if not ci_entry:
                continue
            ed_str = str(a["edition_year"])
            editions = ci_entry.get("editions", {})
            # Get word counts from OTHER editions
            other_wcs = [
                ed.get("word_count", 0)
                for yr, ed in editions.items()
                if yr != ed_str and ed.get("word_count", 0) > 0
            ]
            if not other_wcs:
                continue
            other_median = stat_median(other_wcs)
            if other_median > 5000:
                continue  # article is legitimately large in other editions too
            # Get this edition's word count
            this_ed = editions.get(ed_str, {})
            this_wc = this_ed.get("word_count", 0)
            if this_wc > 20000 and this_wc > max(other_median, 1) * 20:
                ratio = this_wc / max(other_median, 1)
                bloated.append(BloatedArticle(
                    title=title,
                    edition=a["edition"],
                    edition_year=a["edition_year"],
                    volume=a["volume"],
                    entity_count=a["_total_entities"],
                    median=other_median,
                    ratio=ratio,
                    article_id=a.get("article_id", ""),
                ))
                seen.add((title, a["edition_year"]))

    bloated.sort(key=lambda b: -b.ratio)
    return bloated


# ---------------------------------------------------------------------------
# Step 3: Diagnose absorbed headwords
# ---------------------------------------------------------------------------

def get_next_article(
    export_by_vol: dict, edition_year: int, volume: int, title: str
) -> dict | None:
    """Get the article immediately after the given title in the same volume."""
    articles = export_by_vol.get((edition_year, volume), [])
    for i, a in enumerate(articles):
        if a["title"].upper() == title.upper():
            if i + 1 < len(articles):
                return articles[i + 1]
    return None


def diagnose_absorbed(
    bloated: list[BloatedArticle],
    concept_index: dict,
    export_by_key: dict,
    export_by_vol: dict,
) -> dict[tuple, list[AbsorbedHeadword]]:
    """For each bloated article, find headwords it likely absorbed.

    Returns: {(title, edition_year): [AbsorbedHeadword, ...]}
    """
    result = {}
    for b in bloated:
        key = (b.title, b.edition_year)
        bloat_sort = normalize_sort_key(b.title)

        # Find next article's sort key
        next_art = get_next_article(export_by_vol, b.edition_year, b.volume, b.title)
        if next_art is None:
            # Last article in volume — use end of alphabet for this volume
            next_sort = "ZZZZZ"
        else:
            next_sort = normalize_sort_key(next_art["title"])

        # Find concept index headwords in the alphabetical gap
        absorbed = []
        ed_year_str = str(b.edition_year)
        for hw_key, ci_entry in concept_index.items():
            hw_sort = normalize_sort_key(hw_key)
            # Must fall strictly between bloated article and next article
            if hw_sort <= bloat_sort or hw_sort >= next_sort:
                continue
            # Check if expected in this edition
            in_this = ed_year_str in ci_entry.get("editions", {})
            if not in_this:
                # Still useful if it appears in many other editions
                edition_count = len(ci_entry.get("editions", {}))
                if edition_count < 4:
                    continue
            else:
                edition_count = len(ci_entry.get("editions", {}))

            # Get expected word count (max across editions)
            max_wc = 0
            for ed_data in ci_entry.get("editions", {}).values():
                max_wc = max(max_wc, ed_data.get("word_count", 0))

            absorbed.append(AbsorbedHeadword(
                headword=ci_entry.get("label", hw_key),
                expected_wc=max_wc,
                edition_count=edition_count,
                in_this_edition=in_this,
            ))

        # --- Late boundary detection ---
        # If the NEXT article exists and is suspiciously small compared to its
        # cross-edition word count, the parser found its heading too late.
        # The real heading is inside the bloated article's text range.
        if next_art is not None:
            next_title = next_art["title"].upper()
            next_wc = next_art["word_count"]
            # Check cross-edition word count for next article
            # Concept index uses normalized keys (I->J, U->V)
            next_ci_key = normalize_sort_key(next_title)
            ci_next = concept_index.get(next_title) or concept_index.get(next_ci_key)
            if ci_next:
                max_wc_next = max(
                    (ed.get("word_count", 0) for ed in ci_next["editions"].values()),
                    default=0,
                )
                # If the next article is <30% of its typical size AND the bloated
                # article is big, then the boundary is probably too late
                if max_wc_next > 1000 and next_wc < max_wc_next * 0.3:
                    edition_count = len(ci_next.get("editions", {}))
                    in_this = ed_year_str in ci_next.get("editions", {})
                    absorbed.append(AbsorbedHeadword(
                        headword=ci_next.get("label", next_title),
                        expected_wc=max_wc_next,
                        edition_count=edition_count,
                        in_this_edition=in_this or True,  # it exists, just at wrong position
                    ))

        # Sort by expected word count (most important first)
        absorbed.sort(key=lambda a: -a.expected_wc)
        result[key] = absorbed

    return result


# ---------------------------------------------------------------------------
# Step 4: Localize in OCR text
# ---------------------------------------------------------------------------

def make_titlecase(hw: str) -> str:
    """Convert headword to titlecase for search. FRANCE -> France."""
    return hw.capitalize() if " " not in hw else hw.title()


def localize_heading(
    ocr_text: str,
    headword: str,
    search_start: int,
    search_end: int,
) -> tuple[int, str, str] | None:
    """Search for a headword in the OCR text within the given range.

    Returns (position, match_text, pattern_name) or None.
    """
    region = ocr_text[search_start:search_end]
    tc = make_titlecase(headword)
    uc = headword.upper()

    # Pattern 1: Titlecase treatise — "France.\n\nFrance,"
    pat1 = re.compile(
        r'\n\n' + re.escape(tc) + r'\.\s*\n\n' + re.escape(tc) + r'[,.]',
    )
    # Pattern 2: Titlecase with article — "France, a large"
    pat2 = re.compile(
        r'\n\n' + re.escape(tc) + r',\s+(?:a|an|the|in|one|or)\s',
    )
    # Pattern 3: ALL-CAPS comma — "\n\nFRANCE," or "\nFRANCE,"
    pat3 = re.compile(
        r'\n\n' + re.escape(uc) + r',\s+',
    )
    # Pattern 4: ALL-CAPS period treatise — "\n\nFRANCE.\n"
    pat4 = re.compile(
        r'\n\n' + re.escape(uc) + r'\.\s*\n',
    )
    # Pattern 5: ALL-CAPS period inline — "\n\nFRANCE. Text"
    pat5 = re.compile(
        r'\n\n' + re.escape(uc) + r'\.\s+[A-Z]',
    )
    # Pattern 6: Single newline ALL-CAPS — "\nFRANCE,"
    pat6 = re.compile(
        r'\n' + re.escape(uc) + r'[,.]\s',
    )
    # Pattern 7: Titlecase period — "\n\nFrance. The"
    pat7 = re.compile(
        r'\n\n' + re.escape(tc) + r'\.\s+[A-Z]',
    )
    # Pattern 8: ALL-CAPS standalone heading — "\nCHEMISTRY\n\n" (no punctuation)
    pat8 = re.compile(
        r'\n' + re.escape(uc) + r'\s*\n\n',
    )
    # Pattern 9: Titlecase standalone heading — "\nChemistry\n\n"
    pat9 = re.compile(
        r'\n' + re.escape(tc) + r'\s*\n\n',
    )
    # Pattern 10: Bold markers — "\n\n**FARRIERY**\n"
    pat10 = re.compile(
        r'\n\n\*\*' + re.escape(uc) + r'\*\*\s*\n',
    )
    # Pattern 11: ALL-CAPS with semicolon — "\n\nOPTICS;\n"
    pat11 = re.compile(
        r'\n\n' + re.escape(uc) + r'[;:]\s*\n',
    )
    # Pattern 12: Bold titlecase — "\n\n**Farriery**\n"
    pat12 = re.compile(
        r'\n\n\*\*' + re.escape(tc) + r'\*\*\s*\n',
    )

    patterns = [
        (pat1, "titlecase_treatise"),
        (pat2, "titlecase_comma"),
        (pat3, "allcaps_comma"),
        (pat4, "allcaps_treatise"),
        (pat5, "allcaps_period"),
        (pat7, "titlecase_period"),
        (pat10, "bold_allcaps"),
        (pat11, "allcaps_semicolon"),
        (pat12, "bold_titlecase"),
        (pat8, "allcaps_standalone"),
        (pat9, "titlecase_standalone"),
        (pat6, "single_newline"),
    ]

    for pat, name in patterns:
        m = pat.search(region)
        if m:
            abs_pos = search_start + m.start()
            match_text = m.group()
            # Running header check: if this headword appears 5+ times
            # in the region at regular intervals, it's a running header.
            # Take only the first occurrence with >200 chars before next.
            all_matches = list(pat.finditer(region))
            if len(all_matches) >= 5 and name == "single_newline":
                # Likely running headers — skip
                continue
            return abs_pos, match_text, name

    return None


# ---------------------------------------------------------------------------
# Step 5: Generate candidates
# ---------------------------------------------------------------------------

def generate_candidates(
    bloated: list[BloatedArticle],
    absorbed_map: dict[tuple, list[AbsorbedHeadword]],
    export_by_key: dict,
    export_by_vol: dict,
    ocr_dir: Path,
) -> list[Candidate]:
    """For each bloated article + absorbed headword, try to find it in OCR text."""
    candidates = []

    for b in bloated:
        key = (b.title, b.edition_year)
        absorbed_list = absorbed_map.get(key, [])
        if not absorbed_list:
            continue

        # Get the bloated article's export data
        export_rec = export_by_key.get(key)
        if export_rec is None:
            continue

        source_file = export_rec["source_file"]
        search_start = export_rec["char_start"]
        search_end = export_rec["char_end"]

        # Get next article for alpha check
        next_art = get_next_article(
            export_by_vol, b.edition_year, b.volume, b.title
        )

        # Load OCR text
        ocr_text = load_ocr_text(ocr_dir, source_file)
        if not ocr_text:
            continue
        text_len = len(ocr_text)

        # Try to locate each absorbed headword (top 20 by expected wc)
        for ah in absorbed_list[:20]:
            result = localize_heading(
                ocr_text, ah.headword, search_start, search_end,
            )
            if result is None:
                continue

            abs_pos, match_text, pattern_name = result

            # Extract context
            before = ocr_text[max(0, abs_pos - 60):abs_pos].replace("\n", "\\n")
            after_start = abs_pos + len(match_text)
            after = ocr_text[after_start:after_start + 60].replace("\n", "\\n")
            match_escaped = match_text.replace("\n", "\\n")

            pct = round(abs_pos / text_len * 100, 1) if text_len > 0 else 0

            # Determine volume string
            vol_str = f"v{b.volume:02d}" if b.volume < 100 else f"v{b.volume}"

            # Determine confidence
            if ah.in_this_edition and pattern_name != "single_newline":
                confidence = "HIGH"
            elif ah.in_this_edition:
                confidence = "MEDIUM"
            elif ah.edition_count >= 4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            # Determine prev/next article names
            prev_article = b.title
            next_article = next_art["title"] if next_art else ""

            candidates.append(Candidate(
                file=source_file,
                edition=b.edition_year,
                vol=vol_str,
                candidate=ah.headword.upper(),
                position=abs_pos,
                pct=pct,
                before=before,
                match=match_escaped,
                after=after,
                alpha_check="in_order",
                prev_article=prev_article,
                next_article=next_article,
                containing_article=b.title,
                source="ner_audit",
                bloat_ratio=round(b.ratio, 1),
                confidence=confidence,
                expected_wc=ah.expected_wc,
            ))

    # Deduplicate: same (file, candidate, position within 100 chars)
    seen = set()
    deduped = []
    for c in candidates:
        dedup_key = (c.file, c.candidate, c.position // 100)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(c)

    # Sort by impact: bloat_ratio * expected_wc
    deduped.sort(key=lambda c: -(c.bloat_ratio * c.expected_wc))
    return deduped


# ---------------------------------------------------------------------------
# Step 6: Output
# ---------------------------------------------------------------------------

def write_candidates(candidates: list[Candidate], output_path: Path):
    """Write candidates in supplementary injection JSONL format."""
    with open(output_path, "w") as f:
        for c in candidates:
            entry = {
                "file": c.file,
                "edition": c.edition,
                "vol": c.vol,
                "candidate": c.candidate,
                "position": c.position,
                "pct": c.pct,
                "before": c.before,
                "match": c.match,
                "after": c.after,
                "alpha_check": c.alpha_check,
                "prev_article": c.prev_article,
                "next_article": c.next_article,
                "containing_article": c.containing_article,
                "source": c.source,
                "bloat_ratio": c.bloat_ratio,
                "confidence": c.confidence,
                "expected_wc": c.expected_wc,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(candidates)} candidates to {output_path}")


def write_report(
    bloated: list[BloatedArticle],
    absorbed_map: dict[tuple, list[AbsorbedHeadword]],
    candidates: list[Candidate],
    report_path: Path,
):
    """Write human-readable audit report."""
    lines = []
    lines.append("=" * 90)
    lines.append("NER-GUIDED PARSING AUDIT REPORT")
    lines.append("=" * 90)

    # Summary
    lines.append(f"\nBloated articles detected:     {len(bloated)}")
    diagnosed = sum(1 for v in absorbed_map.values() if v)
    lines.append(f"With absorbed headwords found:  {diagnosed}")
    lines.append(f"OCR candidates generated:       {len(candidates)}")

    # Confidence breakdown
    high = sum(1 for c in candidates if c.confidence == "HIGH")
    med = sum(1 for c in candidates if c.confidence == "MEDIUM")
    low = sum(1 for c in candidates if c.confidence == "LOW")
    lines.append(f"  HIGH confidence:   {high}")
    lines.append(f"  MEDIUM confidence: {med}")
    lines.append(f"  LOW confidence:    {low}")

    # Per-edition summary
    lines.append(f"\n{'Edition':>12} | {'Bloated':>7} | {'Candidates':>10}")
    lines.append("-" * 40)
    by_ed_bloat = defaultdict(int)
    by_ed_cand = defaultdict(int)
    for b in bloated:
        by_ed_bloat[b.edition_year] += 1
    for c in candidates:
        by_ed_cand[c.edition] += 1
    for yr in sorted(set(list(by_ed_bloat.keys()) + list(by_ed_cand.keys()))):
        lines.append(f"  {yr:>10} | {by_ed_bloat[yr]:>7} | {by_ed_cand[yr]:>10}")

    # Top candidates by impact
    lines.append(f"\n\n{'=' * 90}")
    lines.append("TOP CANDIDATES (ranked by bloat_ratio * expected_word_count)")
    lines.append("=" * 90)

    for i, c in enumerate(candidates[:80]):
        impact = c.bloat_ratio * c.expected_wc
        lines.append(f"\n--- #{i+1} [{c.confidence}] impact={impact:,.0f} ---")
        lines.append(f"  Absorbed headword: {c.candidate}")
        lines.append(f"  Containing article: {c.containing_article} ({c.edition})")
        lines.append(f"  Bloat ratio: {c.bloat_ratio}x | Expected word count: {c.expected_wc:,}")
        lines.append(f"  File: {c.file} @ position {c.position} ({c.pct}%)")
        lines.append(f"  Context: ...{c.before}[{c.match}]{c.after}...")

    # All bloated articles without candidates (for manual review)
    no_candidate_titles = set()
    candidate_keys = {(c.containing_article, c.edition) for c in candidates}
    unresolved = [b for b in bloated if (b.title, b.edition_year) not in candidate_keys]

    if unresolved:
        lines.append(f"\n\n{'=' * 90}")
        lines.append(f"UNRESOLVED BLOATED ARTICLES ({len(unresolved)} — no OCR match found)")
        lines.append("=" * 90)
        for b in unresolved[:60]:
            absorbed = absorbed_map.get((b.title, b.edition_year), [])
            top_expected = absorbed[:3] if absorbed else []
            expected_str = ", ".join(
                f"{a.headword}({a.expected_wc:,}wc)" for a in top_expected
            )
            lines.append(
                f"  {b.ratio:6.1f}x | {b.edition} v{b.volume:>2} | "
                f"{b.entity_count:>4} ents (med {b.median:.0f}) | "
                f"{b.title[:35]:35s} | expected: {expected_str[:50]}"
            )

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Wrote report to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NER-guided parsing audit")
    parser.add_argument("--min-bloat-ratio", type=float, default=10.0,
                        help="Minimum entity_count/median ratio to flag (default: 10)")
    parser.add_argument("--min-entities", type=int, default=50,
                        help="Minimum entity count to flag (default: 50)")
    parser.add_argument("--min-editions", type=int, default=2,
                        help="Minimum editions for cross-comparison (default: 2)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path for candidates")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="Output path for human-readable report")
    args = parser.parse_args()

    print("Step 1: Loading data...")
    ner_by_title = load_ner_data(NER_DIR)
    concept_index = load_concept_index(CONCEPT_INDEX_PATH)
    export_by_key, export_by_vol = load_export_data(EXPORT_DIR)

    print("\nStep 2: Detecting bloated articles...")
    bloated = detect_bloated(
        ner_by_title,
        export_by_key,
        concept_index,
        min_ratio=args.min_bloat_ratio,
        min_entities=args.min_entities,
        min_editions=args.min_editions,
    )
    print(f"  Found {len(bloated)} bloated articles")

    print("\nStep 3: Diagnosing absorbed headwords...")
    absorbed_map = diagnose_absorbed(bloated, concept_index, export_by_key, export_by_vol)
    has_absorbed = sum(1 for v in absorbed_map.values() if v)
    print(f"  {has_absorbed} bloated articles have candidate absorbed headwords")

    print("\nStep 4-5: Localizing headwords in OCR text and generating candidates...")
    candidates = generate_candidates(
        bloated, absorbed_map, export_by_key, export_by_vol, OCR_DIR,
    )
    print(f"  Generated {len(candidates)} candidates")

    # Check the FRANCE case specifically
    france_cands = [c for c in candidates if c.candidate == "FRANCE"]
    if france_cands:
        print(f"\n  FRANCE candidates found:")
        for c in france_cands:
            print(f"    {c.edition} {c.file} @ pos {c.position} [{c.confidence}] "
                  f"(containing: {c.containing_article})")

    print("\nStep 6: Writing output...")
    write_candidates(candidates, args.output)
    write_report(bloated, absorbed_map, candidates, args.report)

    print("\nDone!")


if __name__ == "__main__":
    main()
