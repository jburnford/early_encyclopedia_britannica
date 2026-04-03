#!/usr/bin/env python3
"""Generate fix specs from swallowed article detections.

Cross-references paragraph-level topic breaks with the gap classification
list and existing fixes. Outputs proposed splits for human review.

Confidence levels:
  - HIGH: break_headword matches a known gap AND appears in 2+ editions
  - MEDIUM: break_headword matches a gap OR appears in 2+ editions
  - LOW: single-edition detection with clear headword, no gap match

Usage:
    python graphrag/generate_fixes.py                    # all detections
    python graphrag/generate_fixes.py --min-confidence medium
    python graphrag/generate_fixes.py --edition-year 1810
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTIONS_PATH = REPO_ROOT / "data" / "swallowed_detections.jsonl"
GAPS_PATH = REPO_ROOT / "data" / "gap_classifications.jsonl"
EXPORT_DIR = REPO_ROOT / "data" / "export"
OUTPUT_FIXES = REPO_ROOT / "data" / "proposed_fixes.jsonl"
OUTPUT_REPORT = REPO_ROOT / "data" / "proposed_fixes.md"

# Already-fixed articles (from fix_mega_articles.py) — skip these
ALREADY_FIXED = set()


def load_existing_fixes():
    """Parse fix_mega_articles.py to find already-handled articles."""
    fix_path = REPO_ROOT / "scripts" / "fix_mega_articles.py"
    if not fix_path.exists():
        return set()
    text = fix_path.read_text()
    # Extract (year, title) from FIXES list entries like (1860, 'BOSWORTH-MARKET', ...
    fixed = set()
    for m in re.finditer(r"\((\d{4}),\s*'([^']+)'", text):
        fixed.add((int(m.group(1)), m.group(2).upper()))
    return fixed


def load_gaps():
    """Load gap classifications as lookup dicts."""
    if not GAPS_PATH.exists():
        print(f"Warning: {GAPS_PATH} not found")
        return {}, {}

    # (headword, year) -> gap record
    by_hw_year = {}
    # headword -> [gap records across years]
    by_hw = defaultdict(list)
    with open(GAPS_PATH) as f:
        for line in f:
            g = json.loads(line)
            hw = g["canonical_title"].upper()
            by_hw_year[(hw, g["missing_year"])] = g
            by_hw[hw].append(g)
    return by_hw_year, by_hw


def load_article_index():
    """Build (title, year) -> source_file lookup from export files."""
    index = {}
    for fp in sorted(EXPORT_DIR.glob("eb_*.jsonl")):
        with open(fp) as f:
            for line in f:
                art = json.loads(line)
                key = (art["title"].upper(), art["edition_year"])
                index[key] = {
                    "article_id": art["article_id"],
                    "source_file": art.get("source_file", ""),
                    "word_count": art.get("word_count", 0),
                    "char_start": art.get("char_start", 0),
                }
    return index


# Headwords that are clearly NOT real articles — volume markers, common words, etc.
NOISE_HEADWORDS = {
    "THE", "THIS", "IN", "OF", "A", "WE", "ON", "AT", "BY", "TO", "IT",
    "HE", "HIS", "AS", "OR", "AN", "BUT", "SO", "NO", "IF", "ITS",
    "SINCE", "DURING", "AFTER", "BEFORE", "BETWEEN", "UNDER", "ABOUT",
    "PARLIAMENT", "DOMESTIC",
}
NOISE_PREFIXES = (
    "VOL", "CHAP", "PART ", "SECT.", "SECTION", "INDEX", "DIRECTIONS",
    "ERRATA", "END OF", "EXPLANATION OF PLATES", "TABLE", "APPENDIX",
    "FINIS", "PLATE", "FIG",
)


def is_noise_headword(headword):
    """Check if a break headword is noise (not a real article title)."""
    hw = headword.upper().strip()
    if hw in NOISE_HEADWORDS:
        return True
    if any(hw.startswith(p) for p in NOISE_PREFIXES):
        return True
    # Single character or very short
    if len(hw) <= 1:
        return True
    return False


def match_gap(break_headword, edition_year, gaps_by_hw_year, gaps_by_hw):
    """Try to match a break headword to a known gap. Returns (gap, match_type)."""
    bh = break_headword.upper().strip()
    if not bh:
        return None, None

    # Exact match
    key = (bh, edition_year)
    if key in gaps_by_hw_year:
        return gaps_by_hw_year[key], "exact"

    # Try prefix match (break headword might be truncated)
    if len(bh) >= 3:
        for (hw, yr), g in gaps_by_hw_year.items():
            if yr == edition_year and len(hw) >= 3:
                if hw.startswith(bh[:4]) or bh.startswith(hw[:4]):
                    return g, "prefix"

    return None, None


def build_regex_pattern(after_text):
    """Build a regex pattern from the first distinctive words after the break."""
    text = after_text.strip().replace("\n", " ")
    if not text:
        return None

    # Take first 3-5 distinctive words
    words = text.split()[:5]
    if not words:
        return None

    # Escape for regex, join with flexible whitespace
    escaped = [re.escape(w) for w in words]
    pattern = r"\s+".join(escaped[:3])  # use first 3 words
    return pattern


def estimate_after_pct(para_before, total_paras):
    """Estimate the percentage through the article where the break occurs."""
    if total_paras <= 0:
        return 0
    return max(0, int(100 * para_before / total_paras) - 5)  # 5% buffer


def generate_fix_spec(detection, article_info, gap, match_type):
    """Generate a fix spec dict from a detection."""
    source_stem = Path(article_info["source_file"]).stem if article_info else ""
    after_pct = estimate_after_pct(
        detection["para_before"], detection["total_paras"]
    )
    regex = build_regex_pattern(detection["after_start"])

    a_score, a_reason = alpha_score(detection["title"], detection["after_start"])

    spec = {
        "parent_title": detection["title"],
        "break_headword": detection["break_headword"],
        "edition_year": detection["edition_year"],
        "alpha_score": a_score,
        "alpha_reason": a_reason,
        "source_file_pattern": source_stem,
        "article_id": detection["article_id"],
        "similarity": detection["similarity"],
        "classification": detection["classification"],
        "after_pct": after_pct,
        "regex_pattern": regex,
        "para_break": f"{detection['para_before']}→{detection['para_after']}",
        "total_paras": detection["total_paras"],
        "cross_edition_count": detection.get("cross_edition_count", 1),
        "cross_edition_years": detection.get("cross_edition_years", []),
        "gap_match": match_type,
        "gap_classification": gap["classification"] if gap else None,
        "before_end": detection["before_end"][-60:],
        "after_start": detection["after_start"][:100],
    }

    # Confidence level
    has_gap = match_type is not None
    multi_ed = detection.get("cross_edition_count", 1) >= 2
    structural = detection["classification"] in ("mid_word", "mid_sentence", "new_headword")

    if has_gap and multi_ed:
        spec["confidence"] = "HIGH"
    elif has_gap or (multi_ed and structural):
        spec["confidence"] = "MEDIUM"
    elif multi_ed or structural:
        spec["confidence"] = "MEDIUM-LOW"
    else:
        spec["confidence"] = "LOW"

    return spec


def get_full_headword(after_text):
    """Extract the fullest headword possible from the text after a break."""
    text = after_text.strip().replace("\n", " ")
    if not text:
        return ""
    m = re.match(r'([A-Z][A-ZÆŒæœ\s\-\'\.]+)', text)
    if m:
        return m.group(1).strip().rstrip(".,;:")
    m = re.match(r'\*\*([^*]+)\*\*', text)
    if m:
        return m.group(1).strip().upper()
    m = re.match(r'([A-Z][a-zæœ]+(?:[\-\s][A-Z][a-z]+)*)', text)
    if m:
        return m.group(1).strip().upper()
    return ""


def alpha_score(parent_title, after_text):
    """Score 0-5 for how alphabetically adjacent the break headword is.

    5 = exact neighbor (shared 3+ prefix chars, or shared 2 with adjacent 3rd char)
    4 = close neighbor (shared 2+ prefix, or backward with 4+ shared prefix)
    3 = same first letter, plausible
    2 = next letter or no headword extractable
    1 = backward but close (shared 2-3 prefix)
    0 = clearly wrong (different letter backward, or far forward)
    """
    p = parent_title.upper()
    c = get_full_headword(after_text).upper()
    if not p or not c:
        return 2, "no_headword"

    # Shared prefix length
    shared = 0
    for i in range(min(len(p), len(c), 6)):
        if p[i] == c[i]:
            shared += 1
        else:
            break

    forward = c >= p

    if forward:
        if shared >= 3:
            return 5, "exact_neighbor"
        elif shared == 2:
            d = abs(ord(c[2]) - ord(p[2])) if len(c) > 2 and len(p) > 2 else 99
            if d <= 3:
                return 5, "exact_neighbor"
            else:
                return 4, "close_neighbor"
        elif shared == 1:
            d = ord(c[1]) - ord(p[1]) if len(c) > 1 and len(p) > 1 else 99
            if d <= 2:
                return 4, "close_neighbor"
            else:
                return 3, "same_letter"
        else:
            d = ord(c[0]) - ord(p[0])
            if d == 1:
                return 2, "next_letter"
            else:
                return 0, "far_forward"
    else:  # backward
        # Backward with long shared prefix = real adjacent entry
        # (accented chars, hyphens cause minor sort differences)
        if shared >= 4:
            return 5, "adjacent_back"
        elif shared >= 3:
            return 4, "close_back"
        elif shared >= 2:
            return 1, "near_back"
        else:
            return 0, "far_back"


def deduplicate_specs(specs):
    """Keep only the best detection per (parent_title, break_headword, year)."""
    best = {}
    for s in specs:
        key = (s["parent_title"].upper(), s["break_headword"], s["edition_year"])
        if key not in best or s["similarity"] < best[key]["similarity"]:
            best[key] = s
    return sorted(best.values(), key=lambda s: (
        {"HIGH": 0, "MEDIUM": 1, "MEDIUM-LOW": 2, "LOW": 3}[s["confidence"]],
        -s.get("alpha_score", 2),
        s["similarity"],
    ))


def write_report(specs):
    """Write markdown report of proposed fixes."""
    from collections import Counter

    by_conf = defaultdict(list)
    for s in specs:
        by_conf[s["confidence"]].append(s)

    lines = [
        "# Proposed Fixes from Swallowed Article Detection",
        "",
        f"**Generated:** auto",
        f"**Total proposed fixes:** {len(specs)}",
        "",
        "## Summary",
        "",
        "| Confidence | Count | Description |",
        "|-----------|-------|-------------|",
        f"| HIGH | {len(by_conf['HIGH'])} | Gap match + multi-edition |",
        f"| MEDIUM | {len(by_conf['MEDIUM'])} | Gap match OR multi-edition + structural |",
        f"| MEDIUM-LOW | {len(by_conf['MEDIUM-LOW'])} | Multi-edition OR structural signal |",
        f"| LOW | {len(by_conf['LOW'])} | Single-edition, no gap match |",
        "",
        "### Alphabetical Adjacency Score",
        "",
        "| Score | Count | Meaning |",
        "|-------|-------|---------|",
        f"| 5 | {sum(1 for s in specs if s.get('alpha_score')==5)} | "
        f"Exact neighbor — shared 3+ prefix, safe to auto-apply |",
        f"| 4 | {sum(1 for s in specs if s.get('alpha_score')==4)} | "
        f"Close neighbor — shared 2+ prefix, high confidence |",
        f"| 3 | {sum(1 for s in specs if s.get('alpha_score')==3)} | "
        f"Same letter — plausible |",
        f"| 2 | {sum(1 for s in specs if s.get('alpha_score')==2)} | "
        f"Next letter or no headword — review |",
        f"| 1 | {sum(1 for s in specs if s.get('alpha_score')==1)} | "
        f"Backward but close — caution |",
        f"| 0 | {sum(1 for s in specs if s.get('alpha_score')==0)} | "
        f"Far away or wrong direction — likely false positive |",
        "",
        "## How to Use",
        "",
        "Review each proposed fix, then add confirmed ones to `scripts/fix_mega_articles.py`:",
        "```python",
        "# Example fix spec:",
        "# (year, 'PARENT_TITLE', 'source_file_pattern', [",
        "#     ('BREAK_HEADWORD', r'regex_pattern', after_pct),",
        "# ]),",
        "```",
        "",
    ]

    for conf in ["HIGH", "MEDIUM", "MEDIUM-LOW", "LOW"]:
        items = by_conf.get(conf, [])
        if not items:
            continue
        lines.append(f"## {conf} Confidence ({len(items)} fixes)")
        lines.append("")

        for s in items:
            xed = f" ({len(s['cross_edition_years'])} eds: {', '.join(str(y) for y in s['cross_edition_years'])})" if s["cross_edition_count"] > 1 else ""
            gap = f" [gap: {s['gap_classification']}]" if s["gap_match"] else ""
            score = s.get("alpha_score", 2)
            flag = {5: "🟢", 4: "🟢", 3: "🟡", 2: "🟡", 1: "🟠", 0: "🔴"}.get(score, "")
            lines.append(
                f"- {flag} **{s['parent_title']}** → **{s['break_headword']}** "
                f"({s['edition_year']}) sim={s['similarity']:.3f} "
                f"[{s['classification']}]{gap}{xed}"
            )
            lines.append(f"  - `...{s['before_end']}`")
            lines.append(f"  - `{s['after_start']}`")
            if s["regex_pattern"]:
                lines.append(
                    f"  - Fix: `({s['edition_year']}, '{s['parent_title']}', "
                    f"'{s['source_file_pattern']}', "
                    f"[('{s['break_headword']}', r'{s['regex_pattern']}', {s['after_pct']})])`"
                )
            lines.append("")

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines))
    print(f"Report: {OUTPUT_REPORT}")


def main():
    parser = argparse.ArgumentParser(description="Generate fix specs from detections")
    parser.add_argument("--edition-year", type=int)
    parser.add_argument("--min-confidence", default="low",
                        choices=["high", "medium", "medium-low", "low"])
    args = parser.parse_args()

    conf_order = {"high": 0, "medium": 1, "medium-low": 2, "low": 3}
    min_conf = conf_order[args.min_confidence]

    # Load data
    print("Loading detections...", end=" ", flush=True)
    detections = []
    with open(DETECTIONS_PATH) as f:
        for line in f:
            detections.append(json.loads(line))
    print(f"{len(detections):,}")

    print("Loading gaps...", end=" ", flush=True)
    gaps_by_hw_year, gaps_by_hw = load_gaps()
    print(f"{len(gaps_by_hw_year):,}")

    print("Loading article index...", end=" ", flush=True)
    article_index = load_article_index()
    print(f"{len(article_index):,}")

    print("Loading existing fixes...", end=" ", flush=True)
    already_fixed = load_existing_fixes()
    print(f"{len(already_fixed)}")

    # Filter detections
    if args.edition_year:
        detections = [d for d in detections if d["edition_year"] == args.edition_year]

    # Only process detections with a break headword
    detections = [d for d in detections if d.get("break_headword")]
    print(f"\nDetections with headword: {len(detections):,}")

    # Generate fix specs
    specs = []
    skipped_fixed = 0
    skipped_noise = 0
    for d in detections:
        # Skip already-fixed articles
        if (d["edition_year"], d["title"].upper()) in already_fixed:
            skipped_fixed += 1
            continue

        # Skip noise headwords (volume markers, common words)
        if is_noise_headword(d["break_headword"]):
            skipped_noise += 1
            continue

        # Look up article info
        art_key = (d["title"].upper(), d["edition_year"])
        article_info = article_index.get(art_key)

        # Try to match break headword to a gap
        gap, match_type = match_gap(
            d["break_headword"], d["edition_year"],
            gaps_by_hw_year, gaps_by_hw
        )

        spec = generate_fix_spec(d, article_info, gap, match_type)
        specs.append(spec)

    print(f"Skipped (already fixed): {skipped_fixed}")
    print(f"Skipped (noise headwords): {skipped_noise}")

    # Deduplicate: keep best detection per (parent, headword, year)
    specs = deduplicate_specs(specs)
    print(f"Unique fix specs: {len(specs):,}")

    # Filter by confidence
    conf_labels = ["HIGH", "MEDIUM", "MEDIUM-LOW", "LOW"]
    specs = [s for s in specs if conf_order[s["confidence"].lower()] <= min_conf]
    print(f"After min-confidence={args.min_confidence}: {len(specs):,}")

    # Summary
    from collections import Counter
    by_conf = Counter(s["confidence"] for s in specs)
    print(f"\nBy confidence:")
    for c in conf_labels:
        if by_conf[c]:
            print(f"  {c}: {by_conf[c]:,}")

    by_gap = Counter(s["gap_match"] for s in specs if s["gap_match"])
    if by_gap:
        print(f"\nGap matches: {sum(by_gap.values())} ({dict(by_gap)})")

    # Write outputs
    with open(OUTPUT_FIXES, "w") as f:
        for s in specs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nJSONL: {OUTPUT_FIXES}")

    write_report(specs)


if __name__ == "__main__":
    main()
