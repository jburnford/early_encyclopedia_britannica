"""Phase 3.5: Merge fragment articles back into parent treatises.

The LLM classifier sometimes promotes internal section headings to article_start,
fragmenting long treatises (ANATOMY, AGRICULTURE, ALGEBRA, etc.) into dozens of
small articles. This module detects and merges those fragments.

Merge heuristics:
1. Consecutive identical titles (e.g., "HEAT" x80) → always merge
2. Chapter/section heading patterns ("Chap. XIII", "Part II", "Of the...") → always merge
3. ALL-CAPS title > 2 chars with no chapter pattern → hard boundary (real article)
4. Mixed-case title + contiguous + predecessor is large (>1000w) or already absorbed
   fragments → merge as subsection
5. Mixed-case title after short predecessor → keep separate (dictionary entry sequence)
"""

import json
import logging
import re
from pathlib import Path

from config import ARTICLES_DIR, ensure_dirs

log = logging.getLogger(__name__)

# Patterns that indicate a subsection heading, not a real article title
CHAPTER_PATTERNS = re.compile(
    r'^('
    r'Chap\.?\s|Chapter\s|Part\s[IVX\d]|Sect\.?\s|Section\s|'
    r'Book\s[IVX\d]|Division\s|Class\s[IVX\d]|Order\s[IVX\d]|'
    r'Of the\s|Of a\s|Of\s[A-Z]|'
    r'The\s[a-z]|An\s[a-z]|A\s[a-z]'
    r')',
    re.IGNORECASE
)

# ALL-CAPS versions of chapter patterns (these are still subsections, not articles)
ALLCAPS_CHAPTER = re.compile(
    r'^(CHAP\.?\s|CHAPTER\s|PART\s[IVX\d]|SECT\.?\s|SECTION\s|'
    r'BOOK\s[IVX\d]|PLATE\s|PLATES?\s)',
    re.IGNORECASE
)

# Word count threshold — predecessor must have this many words to absorb fragments
TREATISE_THRESHOLD = 1000

# Maximum char gap between articles to still consider them contiguous.
# Running headers (~50 chars each) create gaps when removed, so allow generous margin.
MAX_CONTIGUITY_GAP = 2000


def normalize_title(title: str) -> str:
    """Normalize title for duplicate comparison."""
    return re.sub(r'[^a-zA-Z0-9]', '', title).upper()


def is_chapter_pattern(title: str) -> bool:
    """Check if title looks like a chapter/section heading."""
    if CHAPTER_PATTERNS.match(title):
        return True
    if ALLCAPS_CHAPTER.match(title):
        return True
    return False


def is_garbage_title(title: str) -> bool:
    """Check if title is OCR garbage or clearly not an article headword."""
    # Very long titles are usually sentence fragments, not headwords
    if len(title) > 80:
        return True
    # Titles with digits mixed with letters in odd ways
    if re.search(r'\d{2,}[A-Z]{2,}', title):
        return True
    return False


def should_merge(current: dict, prev: dict) -> bool:
    """Decide if current article should be merged into prev as a subsection.

    Returns True if current looks like a fragment of prev (subsection heading
    misclassified as article_start).
    """
    if prev is None:
        return False

    # Never merge into or from special types
    if prev["type"] in ("front_matter", "back_matter", "cross_reference"):
        return False
    if current["type"] in ("front_matter", "back_matter", "cross_reference"):
        return False

    # Contiguity check — must be nearby in the original text
    gap = current["char_start"] - prev["char_end"]
    if gap > MAX_CONTIGUITY_GAP:
        return False

    title = current["title"]

    # UNTITLED articles are always fragments
    if title == "UNTITLED":
        return True

    # Garbage titles → merge
    if is_garbage_title(title):
        return True

    # Consecutive identical titles (normalized) → always merge
    # Handles "HEAT" x80 and similar repeated running-header-as-article cases
    if normalize_title(title) == normalize_title(prev["title"]):
        return True

    # Chapter/section heading patterns → always merge regardless of case
    if is_chapter_pattern(title):
        return True

    # ALL-CAPS titles are normally hard boundaries (real article headwords)
    if title.isupper() and len(title.strip()) > 2:
        # But still merge ALL-CAPS into a very large predecessor that has been
        # absorbing fragments — likely a running header the LLM mislabeled
        merged_count = prev.get("_merged_count", 0)
        if merged_count >= 3 and prev["word_count"] >= 5000:
            # Only if this "article" is small (likely a header, not a real article)
            if current["word_count"] < 500:
                return True
        return False

    # Mixed-case title: merge if predecessor is substantial (treatise)
    # or has already absorbed other fragments (snowball)
    merged_count = prev.get("_merged_count", 0)
    if prev["word_count"] >= TREATISE_THRESHOLD or merged_count > 0:
        return True

    # Mixed-case after short predecessor → keep separate
    # (legitimate sequence of short dictionary entries like Aabam, Aacch, ...)
    return False


def merge_article(parent: dict, child: dict):
    """Merge child article into parent as a subsection."""
    # Record child's title as a subsection of parent
    sub_start = parent["paragraph_count"]
    parent["subsections"].append({
        "title": child["title"],
        "paragraph_start": sub_start,
        "paragraph_end": sub_start + child["paragraph_count"],
    })

    # Also absorb child's own subsections with shifted paragraph offsets
    for sub in child.get("subsections", []):
        parent["subsections"].append({
            "title": sub["title"],
            "paragraph_start": sub_start + sub["paragraph_start"],
            "paragraph_end": sub_start + sub["paragraph_end"],
        })

    # Concatenate text
    parent["text"] += "\n\n" + child["text"]
    parent["char_end"] = child["char_end"]
    parent["word_count"] = len(parent["text"].split())
    parent["paragraph_count"] += child["paragraph_count"]

    # Track merge count for snowball detection
    parent["_merged_count"] = parent.get("_merged_count", 0) + 1

    # Preserve author attribution from child if parent doesn't have one
    if child.get("author_attribution") and not parent.get("author_attribution"):
        parent["author_attribution"] = child["author_attribution"]

    # Merge keywords
    if child.get("keywords"):
        existing = set(parent.get("keywords") or [])
        existing.update(child["keywords"])
        parent["keywords"] = sorted(existing)


def renumber_articles(articles: list[dict]):
    """Re-number article IDs to be sequential after merging."""
    counter = 0
    for article in articles:
        aid = article["article_id"]
        # Preserve special FM/BM suffixes
        if aid.endswith("_FM") or aid.endswith("_BM"):
            continue
        # Rebuild: eb_{edition}_{year}_v{vol}_{counter}
        counter += 1
        prefix = aid.rsplit("_", 1)[0]
        article["article_id"] = f"{prefix}_{counter:04d}"


def merge_file(articles_path: Path) -> tuple[int, int]:
    """Merge fragment articles in one file. Returns (before_count, after_count)."""
    articles = []
    with open(articles_path) as f:
        for line in f:
            articles.append(json.loads(line))

    if not articles:
        return 0, 0

    before_count = len(articles)
    merged = []
    merge_count = 0

    for article in articles:
        if not merged:
            merged.append(article)
            continue

        prev = merged[-1]

        if should_merge(article, prev):
            merge_article(prev, article)
            merge_count += 1
        else:
            merged.append(article)

    # Clean up internal tracking field
    for article in merged:
        article.pop("_merged_count", None)

    # Re-number article IDs
    renumber_articles(merged)

    # Write output (overwrite — assembly can regenerate from classifications)
    with open(articles_path, "w") as f:
        for article in merged:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    after_count = len(merged)
    stem = articles_path.stem.replace(".articles", "")

    # Stats
    type_counts = {}
    for a in merged:
        t = a["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    total_words = sum(a["word_count"] for a in merged)

    log.info(f"{stem}: {before_count} → {after_count} articles "
             f"({merge_count} fragments absorbed), "
             f"{total_words:,} words, types: {type_counts}")

    return before_count, after_count


def run(files: list[Path] | None = None):
    """Run merger on all article files."""
    ensure_dirs()

    if files is None:
        article_files = sorted(ARTICLES_DIR.glob("*.articles.jsonl"))
    else:
        article_files = []
        for f in files:
            stem = f.stem
            apath = ARTICLES_DIR / f"{stem}.articles.jsonl"
            if apath.exists():
                article_files.append(apath)

    if not article_files:
        log.warning("No article files found to merge")
        return 0

    total_before = 0
    total_after = 0

    for apath in article_files:
        before, after = merge_file(apath)
        total_before += before
        total_after += after

    merged_total = total_before - total_after
    log.info(f"Merge complete: {len(article_files)} files, "
             f"{total_before:,} → {total_after:,} articles "
             f"({merged_total:,} fragments absorbed)")
    return total_after


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
