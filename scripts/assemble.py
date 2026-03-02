"""Phase 3: Assemble structured articles from classified paragraphs."""

import json
import logging
from pathlib import Path

from config import (
    INPUT_DIR, PARAGRAPHS_DIR, CLASSIFICATIONS_DIR, ARTICLES_DIR, ensure_dirs,
)

log = logging.getLogger(__name__)


def load_paragraphs(para_path: Path) -> list[dict]:
    """Load paragraph data from JSONL."""
    paragraphs = []
    with open(para_path) as f:
        for line in f:
            paragraphs.append(json.loads(line))
    return paragraphs


def load_classifications(cls_path: Path) -> list[dict]:
    """Load classification data from JSONL."""
    classifications = []
    with open(cls_path) as f:
        for line in f:
            classifications.append(json.loads(line))
    return classifications


def extract_text(paragraphs: list[dict], original_text: str) -> str:
    """Extract full text from original using char offsets, excluding running headers."""
    parts = []
    for p in paragraphs:
        parts.append(original_text[p["char_start"]:p["char_end"]])
    return "\n\n".join(parts)


def assemble_file(
    input_path: Path,
    para_path: Path,
    cls_path: Path,
) -> Path:
    """Assemble articles from one file's paragraphs and classifications."""
    stem = input_path.stem

    # Load source metadata and original text
    with open(input_path) as f:
        meta = json.loads(f.readline())
    original_text = meta["text"]
    edition_name = meta["edition_name"]
    edition_year = meta["edition"]
    volume = meta["volume"]
    source_file = input_path.name

    # Load paragraphs and classifications
    paragraphs = load_paragraphs(para_path)
    classifications = load_classifications(cls_path)

    # Build index: para_index -> classification
    cls_map = {c["index"]: c for c in classifications}

    # Walk paragraphs and build articles
    articles = []
    current_article = None
    current_paras = []  # paragraphs belonging to current article (excluding headers)
    current_subsections = []
    current_subsection = None
    current_sub_start = 0
    front_matter_paras = []
    back_matter_paras = []
    in_back_matter = False
    article_counter = 0
    author_attrib = None

    def finalize_article():
        nonlocal current_article, current_paras, current_subsections
        nonlocal current_subsection, current_sub_start, article_counter, author_attrib

        if current_article is None:
            return

        # Close any open subsection
        if current_subsection is not None:
            current_subsections.append({
                "title": current_subsection,
                "paragraph_start": current_sub_start,
                "paragraph_end": len(current_paras),
            })

        if not current_paras:
            current_article = None
            current_subsections = []
            current_subsection = None
            current_sub_start = 0
            author_attrib = None
            return

        text = extract_text(current_paras, original_text)
        char_start = current_paras[0]["char_start"]
        char_end = current_paras[-1]["char_end"]

        article_counter += 1
        article_id = f"eb_{edition_name}_{edition_year}_v{volume:02d}_{article_counter:04d}"

        article = {
            "article_id": article_id,
            "title": current_article.get("title", "UNTITLED"),
            "edition": edition_name,
            "edition_year": edition_year,
            "volume": volume,
            "source_file": source_file,
            "type": current_article.get("article_type", "article"),
            "char_start": char_start,
            "char_end": char_end,
            "text": text,
            "word_count": len(text.split()),
            "paragraph_count": len(current_paras),
            "keywords": current_article.get("keywords"),
            "author_attribution": author_attrib,
            "subsections": current_subsections if current_subsections else [],
        }

        if current_article.get("target"):
            article["target"] = current_article["target"]

        articles.append(article)

        # Reset
        current_article = None
        current_paras = []
        current_subsections = []
        current_subsection = None
        current_sub_start = 0
        author_attrib = None

    for i, para in enumerate(paragraphs):
        cls = cls_map.get(para["index"], {"type": "body_text"})
        cls_type = cls.get("type", "body_text")

        # Back matter detection — only trigger near end of volume
        # (LLM sometimes misclassifies tables/errata mid-volume as back_matter)
        if cls_type == "back_matter":
            if i >= len(paragraphs) * 0.90:
                in_back_matter = True
                finalize_article()
            back_matter_paras.append(para)
            continue

        if in_back_matter:
            back_matter_paras.append(para)
            continue

        # Front matter — only before any article has been seen
        if cls_type == "front_matter":
            if not articles and current_article is None:
                front_matter_paras.append(para)
            else:
                # Misclassified front_matter mid-volume; treat as body text
                if current_article is not None:
                    current_paras.append(para)
            continue

        # Skip running headers and footnote separators
        if cls_type in ("running_header", "footnote_sep"):
            continue

        # Author attribution — record but don't include in text
        if cls_type == "author_attribution":
            author_attrib = para["text"].strip()
            continue

        # New article
        if cls_type == "article_start":
            finalize_article()
            current_article = {
                "title": cls.get("title", "UNTITLED"),
                "keywords": cls.get("keywords"),
                "article_type": "article",
            }
            current_paras = [para]
            continue

        # Cross-reference
        if cls_type == "cross_reference":
            finalize_article()
            current_article = {
                "title": cls.get("title", "UNTITLED"),
                "target": cls.get("target"),
                "article_type": "cross_reference",
            }
            current_paras = [para]
            # Cross-references are single-paragraph; finalize immediately
            finalize_article()
            continue

        # Subsection start
        if cls_type == "subsection_start":
            if current_article is not None:
                # Close previous subsection
                if current_subsection is not None:
                    current_subsections.append({
                        "title": current_subsection,
                        "paragraph_start": current_sub_start,
                        "paragraph_end": len(current_paras),
                    })
                current_subsection = cls.get("title", "")
                current_sub_start = len(current_paras)
            current_paras.append(para)
            continue

        # Body text
        if current_article is not None:
            current_paras.append(para)
        elif not front_matter_paras and not articles:
            # Before any article starts, treat as front matter
            front_matter_paras.append(para)
        elif front_matter_paras and not articles:
            front_matter_paras.append(para)
        else:
            # Orphan body text after articles started but no current article
            # Attach to previous article if possible
            if articles:
                prev_text = articles[-1]["text"]
                prev_text += "\n\n" + original_text[para["char_start"]:para["char_end"]]
                articles[-1]["text"] = prev_text
                articles[-1]["char_end"] = para["char_end"]
                articles[-1]["word_count"] = len(prev_text.split())
                articles[-1]["paragraph_count"] += 1

    # Finalize last article
    finalize_article()

    # Create front_matter article if present
    if front_matter_paras:
        text = extract_text(front_matter_paras, original_text)
        articles.insert(0, {
            "article_id": f"eb_{edition_name}_{edition_year}_v{volume:02d}_FM",
            "title": "FRONT MATTER",
            "edition": edition_name,
            "edition_year": edition_year,
            "volume": volume,
            "source_file": source_file,
            "type": "front_matter",
            "char_start": front_matter_paras[0]["char_start"],
            "char_end": front_matter_paras[-1]["char_end"],
            "text": text,
            "word_count": len(text.split()),
            "paragraph_count": len(front_matter_paras),
            "keywords": None,
            "author_attribution": None,
            "subsections": [],
        })

    # Create back_matter article if present
    if back_matter_paras:
        text = extract_text(back_matter_paras, original_text)
        articles.append({
            "article_id": f"eb_{edition_name}_{edition_year}_v{volume:02d}_BM",
            "title": "BACK MATTER",
            "edition": edition_name,
            "edition_year": edition_year,
            "volume": volume,
            "source_file": source_file,
            "type": "back_matter",
            "char_start": back_matter_paras[0]["char_start"],
            "char_end": back_matter_paras[-1]["char_end"],
            "text": text,
            "word_count": len(text.split()),
            "paragraph_count": len(back_matter_paras),
            "keywords": None,
            "author_attribution": None,
            "subsections": [],
        })

    # Write output grouped by edition
    output_path = ARTICLES_DIR / f"{stem}.articles.jsonl"
    with open(output_path, "w") as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    # Stats
    type_counts = {}
    for a in articles:
        t = a["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    total_words = sum(a["word_count"] for a in articles)
    log.info(f"{stem}: {len(articles)} articles, {total_words:,} words, types: {type_counts}")

    return output_path


def run(files: list[Path] | None = None):
    """Run Phase 3 on all or specified files."""
    ensure_dirs()

    if files is None:
        files = sorted(INPUT_DIR.glob("*.jsonl"))

    total_articles = 0
    for input_path in files:
        stem = input_path.stem
        para_path = PARAGRAPHS_DIR / f"{stem}.paragraphs.jsonl"
        cls_path = CLASSIFICATIONS_DIR / f"{stem}.classifications.jsonl"

        if not para_path.exists():
            log.error(f"No paragraphs file for {stem} — run Phase 1 first")
            continue
        if not cls_path.exists():
            log.error(f"No classifications file for {stem} — run Phase 2 first")
            continue

        output = assemble_file(input_path, para_path, cls_path)
        with open(output) as f:
            count = sum(1 for _ in f)
        total_articles += count

    log.info(f"Phase 3 complete: {len(files)} files, {total_articles:,} total articles")
    return total_articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
