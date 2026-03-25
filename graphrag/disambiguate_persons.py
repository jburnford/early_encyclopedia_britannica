#!/usr/bin/env python3
"""
Person disambiguation pipeline for Encyclopedia Britannica NER results.

Three-phase architecture:
  --prepare   (Phase A): Extract, filter, normalize, cluster → lookup queue
  --finalize  (Phase C): Merge MCP matches into final clusters + report

Phase B (Wikidata MCP lookups) runs interactively through Claude Code.

Modeled after disambiguate_toponyms.py but with person-specific logic:
- Title/honorific stripping for clustering keys
- Name abbreviation expansion (Hen. → Henry, Wm. → William)
- Ordinal normalization (the Third → III)
- Bare surname handling (conservative — don't auto-merge)
- Temporal filtering (person must be plausibly known by 1860)
"""

import json
import re
import sys
import os
import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import HTTPError, URLError

# --- Configuration ---

EDITIONS = [
    ("1st", 1771), ("2nd", 1778), ("3rd", 1797), ("4th", 1810),
    ("5th", 1815), ("6th", 1823), ("7th", 1842), ("8th", 1860),
]

REPO_DIR = Path(__file__).resolve().parent.parent
NER_DIR = REPO_DIR / "data" / "ner"
CONCEPT_INDEX_PATH = REPO_DIR / "graphrag" / "concept_index.json"

# Output paths
CLUSTERS_PATH = NER_DIR / "person_clusters.jsonl"
QUEUE_PATH = NER_DIR / "person_lookup_queue.jsonl"
MATCHES_PATH = NER_DIR / "person_matches.jsonl"
CANDIDATES_PATH = NER_DIR / "person_candidates.jsonl"
REPORT_PATH = NER_DIR / "person_match_report.txt"

# --- False positive filters ---

FALSE_POSITIVE_EXACT = {
    # Longitude abbreviations misclassified as persons
    "W. Long.", "E. Long.", "W. Long", "E. Long",
    "N. Lat.", "S. Lat.", "N. Lat", "S. Lat",
    # Generic titles used alone (not a specific person)
    "King", "Queen", "Prince", "Princess", "Bishop", "Pope", "Emperor",
    "Empress", "Sultan", "Caliph", "Czar", "Tsar",
    "Captain", "General", "Colonel", "Major", "Admiral",
    "Lord", "Lady", "Duke", "Earl", "Baron", "Count", "Marquis",
    "Archbishop", "Cardinal",
    # Nationalities/demonyms misclassified as persons
    "English", "French", "Roman", "Greek", "Roman Catholic",
    "German", "Spanish", "Italian", "Dutch", "Portuguese",
    "Scottish", "Irish", "British", "American", "Persian",
    "Turkish", "Arabian", "Egyptian", "Chinese", "Indian",
    "Mohammedan", "Mahometan", "Christian", "Protestant",
    "Catholic", "Lutheran", "Calvinist", "Presbyterian",
    # Mythological / collective / generic
    "Christ", "Jesus Christ", "Jesus",
    "Castor and Pollux",
    "God", "Allah",
    # Known OCR/NER errors
    "Ibid", "Ibid.", "Fig", "Fig.",
}

FALSE_POSITIVE_PATTERNS = [
    re.compile(r"^\d"),                # starts with digit
    re.compile(r"^[A-Z]\.$"),          # single letter + period only ("N.", "S.")
    re.compile(r"^[A-Z][A-Z]$"),       # two uppercase letters ("II", "IV" — ordinals alone)
    re.compile(r"^[NSEW]\.\s"),        # compass prefix
    re.compile(r"^[a-z]"),             # starts with lowercase (not a proper name)
]

# --- Title prefixes to strip for clustering ---

TITLE_PREFIXES = {
    "sir", "dr", "dr.", "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
    "lord", "lady", "king", "queen", "prince", "princess",
    "captain", "capt", "capt.", "general", "gen", "gen.",
    "colonel", "col", "col.", "major", "maj", "maj.",
    "lieutenant", "lieut", "lieut.", "lt", "lt.",
    "bishop", "archbishop", "rev", "rev.", "reverend",
    "st", "st.", "saint",
    "count", "duke", "earl", "baron", "baroness", "marquis", "viscount",
    "professor", "prof", "prof.",
    "emperor", "empress",
    "cardinal",
}

# --- Name abbreviation expansions ---

NAME_ABBREVIATIONS = {
    # First name abbreviations common in 18th-19th century texts
    "hen.": "Henry",
    "hen": "Henry",
    "jas.": "James",
    "jas": "James",
    "jno.": "John",
    "jno": "John",
    "chas.": "Charles",
    "chas": "Charles",
    "car.": "Charles",  # Carolus (Latin for Charles)
    "car": "Charles",
    "thos.": "Thomas",
    "thos": "Thomas",
    "wm.": "William",
    "wm": "William",
    "robt.": "Robert",
    "robt": "Robert",
    "richd.": "Richard",
    "richd": "Richard",
    "edw.": "Edward",
    "edw": "Edward",
    "geo.": "George",
    "geo": "George",
    "benj.": "Benjamin",
    "benj": "Benjamin",
    "saml.": "Samuel",
    "saml": "Samuel",
    "danl.": "Daniel",
    "danl": "Daniel",
    "nathl.": "Nathaniel",
    "nathl": "Nathaniel",
    "alexr.": "Alexander",
    "alexr": "Alexander",
    "alex.": "Alexander",
    "fras.": "Francis",
    "fras": "Francis",
    "fredk.": "Frederick",
    "fredk": "Frederick",
    "andw.": "Andrew",
    "andw": "Andrew",
}

# --- Ordinal normalization ---

WORD_TO_ROMAN = {
    "the first": "I", "the second": "II", "the third": "III",
    "the fourth": "IV", "the fifth": "V", "the sixth": "VI",
    "the seventh": "VII", "the eighth": "VIII", "the ninth": "IX",
    "the tenth": "X", "the eleventh": "XI", "the twelfth": "XII",
    "the thirteenth": "XIII", "the fourteenth": "XIV",
    "the fifteenth": "XV", "the sixteenth": "XVI",
}

ARABIC_TO_ROMAN = {
    "1st": "I", "2nd": "II", "3rd": "III", "4th": "IV",
    "5th": "V", "6th": "VI", "7th": "VII", "8th": "VIII",
    "9th": "IX", "10th": "X", "11th": "XI", "12th": "XII",
    "13th": "XIII", "14th": "XIV", "15th": "XV", "16th": "XVI",
}


def is_false_positive(text: str) -> bool:
    """Check if a surface form is a known false positive."""
    if text in FALSE_POSITIVE_EXACT:
        return True
    if len(text) < 2:
        return True
    # All-caps abbreviations < 3 chars
    if text.isupper() and len(text) < 3:
        return True
    for pat in FALSE_POSITIVE_PATTERNS:
        if pat.search(text):
            return True
    return False


def normalize_person(text: str) -> str:
    """Normalize a person surface form to a canonical display string.

    Returns the cleaned display form (keeps titles in display but strips
    for clustering key). The clustering key is derived separately via
    make_cluster_key().
    """
    t = text.strip()

    # Strip trailing period(s)
    t = t.rstrip(".")

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Expand name abbreviations (first token only)
    parts = t.split()
    if parts:
        first_lower = parts[0].lower().rstrip(".")
        # Check with and without period
        for key in [parts[0].lower(), first_lower]:
            if key in NAME_ABBREVIATIONS:
                parts[0] = NAME_ABBREVIATIONS[key]
                break
        t = " ".join(parts)

    # Normalize ordinals: "the Third" → "III", "1st" → "I"
    t_lower = t.lower()
    for word_form, roman in WORD_TO_ROMAN.items():
        if t_lower.endswith(word_form):
            prefix = t[:len(t) - len(word_form)].rstrip()
            t = f"{prefix} {roman}"
            break

    for arabic, roman in ARABIC_TO_ROMAN.items():
        if t.endswith(arabic):
            prefix = t[:len(t) - len(arabic)].rstrip()
            t = f"{prefix} {roman}"
            break

    # Normalize "St " / "St." / "Saint" prefix (for saints)
    t = re.sub(r"^Saint\s+", "St ", t)
    t = re.sub(r"^St\.\s*", "St ", t)

    return t


def make_cluster_key(display_name: str) -> str:
    """Create a clustering key by stripping titles and lowercasing.

    "Sir Isaac Newton" → "isaac newton"
    "Dr Johnson" → "dr johnson"  (bare surname — keep title as discriminator)
    "Henry VIII" → "henry viii"
    "King John" → "king john"  (single first name — keep title)
    "Queen Elizabeth" → "queen elizabeth"  (single first name — keep)
    "St John" → "st john"  (saint — keep prefix)

    Key insight: only strip titles when the remainder has ≥2 tokens
    (i.e., at least first + last name). Otherwise "King John", "Sir Walter",
    "Dr Johnson" are kept as-is to avoid merging different people who share
    a first or last name.
    """
    t = display_name.lower().strip()

    # Strip title prefixes ONLY when remainder has 2+ words
    changed = True
    while changed:
        changed = False
        parts = t.split(None, 1)
        if len(parts) >= 2 and parts[0].rstrip(".") in TITLE_PREFIXES:
            remainder = parts[1]
            # Only strip if remainder has 2+ tokens (first + last name)
            if len(remainder.split()) >= 2:
                t = remainder
                changed = True
            else:
                break  # "Sir Walter" → keep as "sir walter"

    return t.strip()


def load_ner_persons():
    """Load all PERSON entities from NER files."""
    print("Loading NER person data...")
    form_by_edition = defaultdict(lambda: Counter())
    form_by_article = defaultdict(lambda: Counter())
    form_total = Counter()
    # Track co-occurring entities for context
    article_titles = {}  # article_id → title

    for ed, year in EDITIONS:
        path = NER_DIR / f"eb_{ed}_{year}.entities.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                article_titles[rec["article_id"]] = rec["title"]
                for ent in rec["entities"]:
                    if ent["type"] == "PERSON":
                        text = ent["text"]
                        form_total[text] += 1
                        form_by_edition[text][year] += 1
                        form_by_article[text][rec["article_id"]] += 1

    print(f"  {len(form_total):,} unique surface forms, {sum(form_total.values()):,} total mentions")
    return form_total, form_by_edition, form_by_article, article_titles


def build_clusters(form_total, form_by_edition, form_by_article, article_titles):
    """Normalize and cluster surface forms."""
    print("Building clusters...")

    clusters = defaultdict(list)
    filtered_count = 0
    filtered_mentions = 0

    for form, count in form_total.items():
        if is_false_positive(form):
            filtered_count += 1
            filtered_mentions += count
            continue

        display = normalize_person(form)
        if len(display) < 2:
            filtered_count += 1
            filtered_mentions += count
            continue

        key = make_cluster_key(display)
        if not key or len(key) < 2:
            filtered_count += 1
            filtered_mentions += count
            continue

        clusters[key].append((form, count))

    print(f"  Filtered {filtered_count:,} false-positive forms ({filtered_mentions:,} mentions)")
    print(f"  {len(clusters):,} clusters from {len(form_total) - filtered_count:,} valid forms")

    cluster_records = []
    for key, forms in clusters.items():
        forms.sort(key=lambda x: x[1], reverse=True)
        label = normalize_person(forms[0][0])
        variants = list(dict.fromkeys(f for f, _ in forms))  # deduplicate, preserve order

        total = sum(c for _, c in forms)

        by_edition = Counter()
        article_ids = set()
        for form, _ in forms:
            for year, cnt in form_by_edition[form].items():
                by_edition[year] += cnt
            article_ids.update(form_by_article[form].keys())

        # Gather sample article titles for context
        sample_articles = []
        for aid in sorted(article_ids)[:10]:
            if aid in article_titles:
                sample_articles.append(article_titles[aid])

        cluster_records.append({
            "cluster_id": key,
            "label": label,
            "variants": variants,
            "total_mentions": total,
            "by_edition": dict(sorted(by_edition.items())),
            "article_count": len(article_ids),
            "edition_count": len(by_edition),
            "sample_articles": sample_articles,
        })

    cluster_records.sort(key=lambda r: r["total_mentions"], reverse=True)
    for i, rec in enumerate(cluster_records):
        rec["frequency_rank"] = i + 1

    return cluster_records


def load_concept_index_persons():
    """Load person-related headwords from the concept index as anchors.

    Biography articles are typically headworded by the person's surname
    or full name (e.g., "NEWTON, Sir Isaac" or "ARISTOTLE").
    """
    print("Loading concept index for person anchors...")
    if not CONCEPT_INDEX_PATH.exists():
        print(f"  WARNING: {CONCEPT_INDEX_PATH} not found")
        return {}

    with open(CONCEPT_INDEX_PATH) as f:
        ci = json.load(f)

    anchors = {}
    for key, val in ci.items():
        label = val.get("label", key)
        anchors[label.lower()] = label

    print(f"  {len(anchors):,} concept anchors loaded")
    return anchors


def apply_concept_anchors(cluster_records, concept_anchors):
    """Mark clusters that match concept index headwords (likely biography articles)."""
    boosted = 0
    for rec in cluster_records:
        key = rec["label"].lower()
        # Also check cluster_id (title-stripped)
        if key in concept_anchors or rec["cluster_id"] in concept_anchors:
            rec["is_concept_headword"] = True
            rec["concept_label"] = concept_anchors.get(key) or concept_anchors.get(rec["cluster_id"])
            boosted += 1
        else:
            rec["is_concept_headword"] = False

    print(f"  {boosted:,} clusters match concept index headwords")
    return cluster_records


def generate_lookup_queue(cluster_records, min_mentions=5):
    """Generate the Wikidata lookup queue for Phase B (Claude MCP loop)."""
    # Already-matched entries from a previous run
    existing_matches = set()
    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            for line in f:
                rec = json.loads(line)
                existing_matches.add(rec["cluster_id"])
        print(f"  {len(existing_matches):,} clusters already matched (will skip)")

    queue = []
    for rec in cluster_records:
        if rec["total_mentions"] < min_mentions:
            continue
        if rec["cluster_id"] in existing_matches:
            continue

        queue.append({
            "cluster_id": rec["cluster_id"],
            "label": rec["label"],
            "variants": rec["variants"][:10],
            "total_mentions": rec["total_mentions"],
            "edition_range": [min(rec["by_edition"].keys()), max(rec["by_edition"].keys())],
            "edition_count": rec["edition_count"],
            "is_concept_headword": rec.get("is_concept_headword", False),
            "sample_articles": rec.get("sample_articles", [])[:5],
            "frequency_rank": rec["frequency_rank"],
        })

    queue.sort(key=lambda r: r["total_mentions"], reverse=True)

    with open(QUEUE_PATH, "w") as f:
        for rec in queue:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"  Wrote {len(queue):,} clusters to lookup queue: {QUEUE_PATH}")
    return queue


# ============================================================
# Phase B1: Bulk Wikidata API search (fetch candidates)
# ============================================================

def fetch_wikidata_candidates(cluster_records, min_mentions=5, delay=0.5):
    """Hit Wikidata search API for each cluster, save raw candidates.

    This is cheap and fast — just search results, no judgment.
    Skips clusters that already have candidates on disk (resume-safe).
    """
    to_search = [r for r in cluster_records if r["total_mentions"] >= min_mentions]

    # Load existing candidates for resume
    existing = set()
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["cluster_id"])
        print(f"  {len(existing):,} existing candidates loaded (will skip)")

    # Also skip clusters already in matches file
    matched_ids = set()
    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    matched_ids.add(json.loads(line)["cluster_id"])

    remaining = [r for r in to_search
                 if r["cluster_id"] not in existing
                 and r["cluster_id"] not in matched_ids]

    if not remaining:
        print("  All clusters already have candidates or matches. Nothing to fetch.")
        return

    print(f"Fetching Wikidata candidates for {len(remaining):,} clusters "
          f"(skipped {len(to_search) - len(remaining):,})...")

    fetched = 0
    errors = 0

    # Open in append mode for resume safety
    with open(CANDIDATES_PATH, "a") as f:
        for i, rec in enumerate(remaining):
            if (i + 1) % 200 == 0:
                print(f"  {i+1:,}/{len(remaining):,} fetched ({fetched:,} with results, {errors:,} errors)")

            label = rec["label"]
            cluster_id = rec["cluster_id"]

            try:
                url = (
                    "https://www.wikidata.org/w/api.php?"
                    "action=wbsearchentities&format=json&language=en&type=item"
                    f"&search={quote(label)}&limit=5"
                )
                req = Request(url, headers={"User-Agent": "EncyclopediaBritannicaKG/1.0"})
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())

                results = data.get("search", [])
                candidates = []
                for r in results:
                    candidates.append({
                        "qid": r["id"],
                        "label": r.get("label", ""),
                        "description": r.get("description", ""),
                    })

                out_rec = {
                    "cluster_id": cluster_id,
                    "search_label": label,
                    "total_mentions": rec["total_mentions"],
                    "edition_count": rec["edition_count"],
                    "is_concept_headword": rec.get("is_concept_headword", False),
                    "sample_articles": rec.get("sample_articles", [])[:5],
                    "candidates": candidates,
                }
                json.dump(out_rec, f, ensure_ascii=False)
                f.write("\n")

                if candidates:
                    fetched += 1

            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
                errors += 1
                # Write empty candidates so we don't retry on resume
                out_rec = {
                    "cluster_id": cluster_id,
                    "search_label": label,
                    "total_mentions": rec["total_mentions"],
                    "candidates": [],
                    "error": str(e),
                }
                json.dump(out_rec, f, ensure_ascii=False)
                f.write("\n")

            time.sleep(delay)

    print(f"\nWikidata candidate fetch complete:")
    print(f"  Fetched: {fetched:,} with results")
    print(f"  Errors: {errors:,}")
    print(f"  Output: {CANDIDATES_PATH}")


# ============================================================
# Phase B2: LLM judge (Sonnet evaluates candidates)
# ============================================================

JUDGE_SYSTEM_PROMPT = """\
You are a historical knowledge expert evaluating Wikidata matches for person \
names found in the Encyclopedia Britannica (editions 1771-1860).

For each person cluster, you receive:
- The name as it appears in the encyclopedia
- The number of mentions and which editions it appears in
- Sample article titles where this person is mentioned
- Up to 5 Wikidata candidate matches (QID, label, description)

Your task: decide which candidate (if any) is the correct match.

Rules:
1. The person MUST be someone who could plausibly be discussed in an \
encyclopedia published between 1771-1860.
2. Reject candidates born after 1870 (too modern).
3. For ambiguous names like "Dr Smith" or "Mr Brown", only match if one \
candidate is overwhelmingly the most famous person of that name in the \
historical period. If genuinely ambiguous, output "none".
4. If no candidate fits, output "none".
5. If the cluster is actually a false positive (not a real person — \
e.g. a nationality, a place, a generic noun), output "false_positive".
6. Descriptions with birth-death years like "(1632-1704)" are strong \
signals — check temporal plausibility.
7. Concept headwords (is_concept_headword=true) mean the encyclopedia \
has a dedicated article about this person — weight toward matching.

Respond with a JSON array. Each element:
{"cluster_id": "...", "match": "Q12345" or "none" or "false_positive", \
"confidence": 0.0-1.0, "reason": "brief explanation"}

Respond ONLY with the JSON array, no other text."""


def judge_candidates(batch_size=20, model="claude-sonnet-4-20250514",
                     api_key=None, max_batches=None):
    """Use Sonnet to evaluate Wikidata candidates and produce matches.

    Reads person_candidates.jsonl, sends batches to Sonnet for judgment,
    writes confirmed matches to person_matches.jsonl.
    Resume-safe: skips clusters already in matches file.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic  (needed for --judge)")
        return

    if not CANDIDATES_PATH.exists():
        print("ERROR: No candidates file. Run --match first.")
        return

    # Load candidates
    candidates = []
    with open(CANDIDATES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    # Filter to those with actual candidates and not already matched
    matched_ids = set()
    if MATCHES_PATH.exists():
        with open(MATCHES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    matched_ids.add(json.loads(line)["cluster_id"])

    to_judge = [c for c in candidates
                if c.get("candidates") and c["cluster_id"] not in matched_ids]

    if not to_judge:
        print("  All candidates already judged. Nothing to do.")
        return

    print(f"Judging {len(to_judge):,} candidates with {model} "
          f"(batch size {batch_size}, skipped {len(matched_ids):,} already matched)...")

    # Sort by mentions descending (high-value first)
    to_judge.sort(key=lambda c: c["total_mentions"], reverse=True)

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    total_matched = 0
    total_none = 0
    total_fp = 0
    batches_done = 0

    for batch_start in range(0, len(to_judge), batch_size):
        if max_batches and batches_done >= max_batches:
            print(f"  Reached max_batches={max_batches}, stopping.")
            break

        batch = to_judge[batch_start:batch_start + batch_size]

        # Build prompt
        entries = []
        for c in batch:
            cand_lines = []
            for idx, cand in enumerate(c["candidates"][:5]):
                cand_lines.append(
                    f"  {idx+1}. {cand['qid']}: {cand['label']} — {cand['description']}"
                )

            entries.append(
                f"CLUSTER: \"{c['search_label']}\" "
                f"(cluster_id=\"{c['cluster_id']}\", {c['total_mentions']} mentions, "
                f"{c.get('edition_count', '?')} editions, "
                f"concept_headword={c.get('is_concept_headword', False)})\n"
                f"  Sample articles: {c.get('sample_articles', [])}\n"
                f"  Candidates:\n" + "\n".join(cand_lines)
            )

        user_msg = "\n\n".join(entries)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            response_text = response.content[0].text.strip()

            # Parse JSON — handle markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r"^```(?:json)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

            judgments = json.loads(response_text)

            # Write matches
            with open(MATCHES_PATH, "a") as f:
                for j in judgments:
                    cid = j["cluster_id"]
                    match_qid = j.get("match")
                    confidence = j.get("confidence", 0.5)
                    reason = j.get("reason", "")

                    # Find the candidate details
                    cand_rec = next(
                        (c for c in batch if c["cluster_id"] == cid), None
                    )
                    if not cand_rec:
                        continue

                    if match_qid and match_qid not in ("none", "false_positive"):
                        matched_cand = next(
                            (c for c in cand_rec["candidates"]
                             if c["qid"] == match_qid), None
                        )
                        match_rec = {
                            "cluster_id": cid,
                            "wikidata_qid": match_qid,
                            "wikidata_label": (
                                matched_cand["label"] if matched_cand else ""
                            ),
                            "wikidata_description": (
                                matched_cand["description"] if matched_cand else ""
                            ),
                            "match_type": "wikidata",
                            "confidence": confidence,
                            "reason": reason,
                            "judge_model": model,
                        }
                        total_matched += 1
                    elif match_qid == "false_positive":
                        match_rec = {
                            "cluster_id": cid,
                            "match_type": "false_positive",
                            "reason": reason,
                            "judge_model": model,
                        }
                        total_fp += 1
                    else:
                        match_rec = {
                            "cluster_id": cid,
                            "match_type": "none",
                            "reason": reason,
                            "judge_model": model,
                        }
                        total_none += 1

                    json.dump(match_rec, f, ensure_ascii=False)
                    f.write("\n")

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  WARNING: Failed to parse batch {batches_done + 1}: {e}")
            with open(MATCHES_PATH, "a") as f:
                for c in batch:
                    if c["cluster_id"] not in matched_ids:
                        json.dump({
                            "cluster_id": c["cluster_id"],
                            "match_type": "none",
                            "reason": f"judge_parse_error: {e}",
                        }, f, ensure_ascii=False)
                        f.write("\n")
                        total_none += 1

        except Exception as e:
            print(f"  ERROR: API call failed for batch {batches_done + 1}: {e}")
            break

        batches_done += 1
        if batches_done % 10 == 0:
            print(f"  Batch {batches_done}: {total_matched:,} matched, "
                  f"{total_none:,} none, {total_fp:,} false positives")

    print(f"\nJudging complete ({batches_done} batches):")
    print(f"  Matched: {total_matched:,}")
    print(f"  No match: {total_none:,}")
    print(f"  False positives: {total_fp:,}")


def finalize_clusters(cluster_records):
    """Phase C: Merge MCP match results into cluster records."""
    if not MATCHES_PATH.exists():
        print(f"  No matches file found at {MATCHES_PATH}")
        return cluster_records

    # Load matches
    matches = {}
    with open(MATCHES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            matches[rec["cluster_id"]] = rec

    print(f"  Loaded {len(matches):,} Wikidata matches")

    matched = 0
    for rec in cluster_records:
        match = matches.get(rec["cluster_id"])
        if not match:
            if rec.get("match_type") is None:
                if rec["total_mentions"] >= 5:
                    rec["match_type"] = "none"
                else:
                    rec["match_type"] = "below_threshold"
            continue

        if match.get("match_type") == "false_positive":
            rec["match_type"] = "false_positive"
            continue

        if match.get("wikidata_qid"):
            rec["match_type"] = "wikidata"
            rec["wikidata_qid"] = match["wikidata_qid"]
            rec["wikidata_label"] = match.get("wikidata_label", "")
            rec["wikidata_description"] = match.get("wikidata_description", "")
            rec["birth_year"] = match.get("birth_year")
            rec["death_year"] = match.get("death_year")
            rec["occupations"] = match.get("occupations", [])
            if match.get("alternatives"):
                rec["alternatives"] = match["alternatives"]
            matched += 1
        else:
            rec["match_type"] = match.get("match_type", "none")

    print(f"  Applied {matched:,} Wikidata matches to clusters")

    # Re-sort and re-rank
    cluster_records.sort(key=lambda r: r["total_mentions"], reverse=True)
    for i, rec in enumerate(cluster_records):
        rec["frequency_rank"] = i + 1

    return cluster_records


def generate_report(cluster_records, report_path, min_mentions=5):
    """Generate match rate analysis by frequency tier."""
    lines = []
    lines.append("=" * 80)
    lines.append("PERSON DISAMBIGUATION REPORT")
    lines.append("=" * 80)

    total_clusters = len(cluster_records)
    total_mentions = sum(r["total_mentions"] for r in cluster_records)
    matched_types = {"wikidata"}

    lines.append(f"\nTotal clusters: {total_clusters:,}")
    lines.append(f"Total mentions: {total_mentions:,}")
    lines.append(f"Min mentions for matching: {min_mentions}")

    # Count by match type
    by_type = Counter(r.get("match_type", "none") for r in cluster_records)
    lines.append(f"\nMatch type breakdown:")
    for mt, cnt in by_type.most_common():
        lines.append(f"  {mt}: {cnt:,}")

    # Match rate by frequency tier
    tiers = [
        ("100+", lambda r: r["total_mentions"] >= 100),
        ("50-99", lambda r: 50 <= r["total_mentions"] < 100),
        ("20-49", lambda r: 20 <= r["total_mentions"] < 50),
        ("10-19", lambda r: 10 <= r["total_mentions"] < 20),
        ("5-9", lambda r: 5 <= r["total_mentions"] < 10),
        ("2-4", lambda r: 2 <= r["total_mentions"] < 5),
        ("1", lambda r: r["total_mentions"] == 1),
    ]

    lines.append(f"\n{'Tier':<10} {'Clusters':>10} {'Wikidata':>10} {'FalsePos':>10} {'None':>10} {'Below':>10} {'Grounded%':>10} {'Mentions':>12}")
    lines.append("-" * 95)

    for tier_name, pred in tiers:
        subset = [r for r in cluster_records if pred(r)]
        n = len(subset)
        wikidata = sum(1 for r in subset if r.get("match_type") == "wikidata")
        false_pos = sum(1 for r in subset if r.get("match_type") == "false_positive")
        below = sum(1 for r in subset if r.get("match_type") == "below_threshold")
        none_ct = n - wikidata - false_pos - below
        grounded = wikidata
        mentions = sum(r["total_mentions"] for r in subset)
        grounded_pct = 100 * grounded / n if n else 0
        lines.append(f"{tier_name:<10} {n:>10,} {wikidata:>10,} {false_pos:>10,} {none_ct:>10,} {below:>10,} {grounded_pct:>9.1f}% {mentions:>12,}")

    # Grounded mentions total
    grounded_recs = [r for r in cluster_records if r.get("match_type") in matched_types]
    grounded_mentions = sum(r["total_mentions"] for r in grounded_recs)
    lines.append(f"\nTotal grounded mentions: {grounded_mentions:,} / {total_mentions:,} ({100*grounded_mentions/total_mentions:.1f}%)")

    # Top 30 matched persons
    matched_recs = [r for r in cluster_records if r.get("match_type") == "wikidata"]
    matched_recs.sort(key=lambda r: r["total_mentions"], reverse=True)

    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP 50 MATCHED PERSONS ({len(matched_recs)} total)")
    lines.append(f"{'=' * 80}")
    for rec in matched_recs[:50]:
        concept_flag = " [CONCEPT]" if rec.get("is_concept_headword") else ""
        birth = rec.get("birth_year", "?")
        death = rec.get("death_year", "?")
        occs = ", ".join(rec.get("occupations", [])[:3])
        lines.append(
            f"  {rec['label']}: {rec['total_mentions']:,} mentions | "
            f"{rec.get('wikidata_qid')} | {birth}-{death} | {occs}{concept_flag}"
        )

    # Top 30 unmatched
    unmatched_recs = [r for r in cluster_records
                      if r.get("match_type") == "none"
                      and r["total_mentions"] >= min_mentions]
    unmatched_recs.sort(key=lambda r: r["total_mentions"], reverse=True)

    lines.append(f"\n{'=' * 80}")
    lines.append(f"TOP 30 STILL UNMATCHED (>= {min_mentions} mentions, {len(unmatched_recs)} total)")
    lines.append(f"{'=' * 80}")
    for rec in unmatched_recs[:30]:
        concept_flag = " [CONCEPT]" if rec.get("is_concept_headword") else ""
        lines.append(f"  {rec['label']}: {rec['total_mentions']:,} mentions, {rec['edition_count']} editions{concept_flag}")
        if len(rec["variants"]) > 1:
            lines.append(f"    variants: {rec['variants'][:5]}")

    # Top false positives (sanity check)
    fp_recs = [r for r in cluster_records if r.get("match_type") == "false_positive"]
    fp_recs.sort(key=lambda r: r["total_mentions"], reverse=True)
    if fp_recs:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"TOP 20 FALSE POSITIVES ({len(fp_recs)} total)")
        lines.append(f"{'=' * 80}")
        for rec in fp_recs[:20]:
            lines.append(f"  {rec['label']}: {rec['total_mentions']:,} mentions")

    # Temporal analysis
    lines.append(f"\n{'=' * 80}")
    lines.append(f"TEMPORAL PATTERNS (grounded persons, >= {min_mentions} mentions)")
    lines.append(f"{'=' * 80}")

    grounded = [r for r in cluster_records
                if r.get("match_type") in matched_types
                and r["total_mentions"] >= min_mentions]

    late_arrivals = [r for r in grounded
                     if 1771 not in r["by_edition"] and 1778 not in r["by_edition"]
                     and (1842 in r["by_edition"] or 1860 in r["by_edition"])]
    late_arrivals.sort(key=lambda r: r["total_mentions"], reverse=True)
    lines.append(f"\nLate arrivals (not in 1771/1778, appear in 1842/1860): {len(late_arrivals)}")
    for rec in late_arrivals[:20]:
        eds = sorted(rec["by_edition"].keys())
        birth = rec.get("birth_year", "?")
        death = rec.get("death_year", "?")
        lines.append(f"  {rec['label']} ({birth}-{death}): {rec['total_mentions']} mentions, editions {eds}")

    disappearances = [r for r in grounded
                      if (1771 in r["by_edition"] or 1778 in r["by_edition"])
                      and 1842 not in r["by_edition"] and 1860 not in r["by_edition"]]
    disappearances.sort(key=lambda r: r["total_mentions"], reverse=True)
    lines.append(f"\nDisappearances (in 1771/1778, not in 1842/1860): {len(disappearances)}")
    for rec in disappearances[:20]:
        eds = sorted(rec["by_edition"].keys())
        birth = rec.get("birth_year", "?")
        death = rec.get("death_year", "?")
        lines.append(f"  {rec['label']} ({birth}-{death}): {rec['total_mentions']} mentions, editions {eds}")

    report = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")
    print(report)


def save_clusters(cluster_records, path):
    """Save cluster records as JSONL."""
    with open(path, "w") as f:
        for rec in cluster_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved {len(cluster_records):,} clusters to {path}")


def save_csv(cluster_records, path):
    """Save cluster records as CSV for easy review."""
    import csv

    editions = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "cluster_id", "label", "total_mentions", "article_count",
            "edition_count", "is_concept_headword", "match_type",
            "wikidata_qid", "wikidata_label", "wikidata_description",
            "birth_year", "death_year", "occupations",
            "alternatives_summary", "variants",
            *[str(y) for y in editions],
        ])
        for r in cluster_records:
            alts = r.get("alternatives", [])
            alt_str = "; ".join(
                f"{a.get('qid','?')}: {a.get('label','')} ({a.get('description','')})"
                for a in alts
            ) if alts else ""

            occs = "; ".join(r.get("occupations", []))

            w.writerow([
                r["frequency_rank"],
                r["cluster_id"],
                r["label"],
                r["total_mentions"],
                r["article_count"],
                r["edition_count"],
                r.get("is_concept_headword", False),
                r.get("match_type", ""),
                r.get("wikidata_qid", ""),
                r.get("wikidata_label", ""),
                r.get("wikidata_description", ""),
                r.get("birth_year", ""),
                r.get("death_year", ""),
                occs,
                alt_str,
                "; ".join(r["variants"][:10]),
                *[r["by_edition"].get(str(y), r["by_edition"].get(y, 0)) for y in editions],
            ])

    print(f"Saved CSV to {path}")


def main():
    parser = argparse.ArgumentParser(description="Person disambiguation pipeline")
    parser.add_argument("--prepare", action="store_true",
                        help="Phase A: Extract, filter, normalize, cluster → lookup queue")
    parser.add_argument("--match", action="store_true",
                        help="Phase B1: Bulk Wikidata API search for candidates")
    parser.add_argument("--judge", action="store_true",
                        help="Phase B2: Sonnet evaluates candidates → matches")
    parser.add_argument("--finalize", action="store_true",
                        help="Phase C: Merge matches into final clusters + report")
    parser.add_argument("--min-mentions", type=int, default=5,
                        help="Min mentions for matching (default: 5)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Clusters per Sonnet API call (default: 20)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Max Sonnet batches (for testing; default: unlimited)")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514",
                        help="Model for --judge (default: claude-sonnet-4-20250514)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between Wikidata API calls in seconds (default: 0.5)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    if not args.prepare and not args.match and not args.judge and not args.finalize:
        print("Usage:")
        print("  Phase A:  python disambiguate_persons.py --prepare")
        print("  Phase B1: python disambiguate_persons.py --match     (bulk Wikidata search)")
        print("  Phase B2: python disambiguate_persons.py --judge     (Sonnet evaluates)")
        print("  Phase C:  python disambiguate_persons.py --finalize  (merge + report)")
        print()
        print("Typical workflow:")
        print("  python disambiguate_persons.py --prepare")
        print("  python disambiguate_persons.py --match              # ~90 min")
        print("  python disambiguate_persons.py --judge              # ~$3, ~30 min")
        print("  python disambiguate_persons.py --judge --max-batches 5  # test first")
        print("  python disambiguate_persons.py --finalize")
        sys.exit(1)

    if args.output_dir:
        global CLUSTERS_PATH, QUEUE_PATH, MATCHES_PATH, CANDIDATES_PATH, REPORT_PATH
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        CLUSTERS_PATH = out / "person_clusters.jsonl"
        QUEUE_PATH = out / "person_lookup_queue.jsonl"
        MATCHES_PATH = out / "person_matches.jsonl"
        CANDIDATES_PATH = out / "person_candidates.jsonl"
        REPORT_PATH = out / "person_match_report.txt"

    # --judge doesn't need NER loading — it just reads candidates file
    if args.judge:
        judge_candidates(
            batch_size=args.batch_size,
            model=args.judge_model,
            max_batches=args.max_batches,
        )
        print("\nDone!")
        return

    # All other phases need clusters
    # Step 1: Load NER data
    form_total, form_by_edition, form_by_article, article_titles = load_ner_persons()

    # Step 2: Normalize and cluster
    cluster_records = build_clusters(form_total, form_by_edition, form_by_article, article_titles)

    # Step 3: Load concept anchors
    concept_anchors = load_concept_index_persons()
    cluster_records = apply_concept_anchors(cluster_records, concept_anchors)

    if args.prepare:
        # Phase A: Save clusters and generate lookup queue
        save_clusters(cluster_records, CLUSTERS_PATH)
        csv_path = CLUSTERS_PATH.with_suffix(".csv")
        save_csv(cluster_records, csv_path)

        # Generate queue for Phase B
        queue = generate_lookup_queue(cluster_records, args.min_mentions)

        # Quick stats
        total = len(cluster_records)
        total_mentions = sum(r["total_mentions"] for r in cluster_records)
        above = [r for r in cluster_records if r["total_mentions"] >= args.min_mentions]
        above_mentions = sum(r["total_mentions"] for r in above)
        concept_count = sum(1 for r in cluster_records if r.get("is_concept_headword"))

        print(f"\n=== PHASE A SUMMARY ===")
        print(f"  Total clusters: {total:,}")
        print(f"  Total mentions: {total_mentions:,}")
        print(f"  Clusters >= {args.min_mentions} mentions: {len(above):,} ({100*len(above)/total:.1f}%)")
        print(f"  Mentions covered by those: {above_mentions:,} ({100*above_mentions/total_mentions:.1f}%)")
        print(f"  Concept headword matches: {concept_count:,}")
        print(f"  Lookup queue size: {len(queue):,}")

        # Show top 20
        print(f"\n  Top 20 clusters:")
        for rec in cluster_records[:20]:
            concept = " [C]" if rec.get("is_concept_headword") else ""
            print(f"    {rec['frequency_rank']:>4}. {rec['label']}: {rec['total_mentions']:,} mentions, {rec['edition_count']} editions{concept}")
            if len(rec["variants"]) > 1:
                print(f"          variants: {rec['variants'][:5]}")

    elif args.match:
        # Phase B1: Bulk Wikidata API search
        fetch_wikidata_candidates(cluster_records, args.min_mentions, args.delay)

    elif args.finalize:
        # Phase C: Merge matches and generate report
        cluster_records = finalize_clusters(cluster_records)

        save_clusters(cluster_records, CLUSTERS_PATH)
        csv_path = CLUSTERS_PATH.with_suffix(".csv")
        save_csv(cluster_records, csv_path)
        generate_report(cluster_records, REPORT_PATH, args.min_mentions)

    print("\nDone!")


if __name__ == "__main__":
    main()
