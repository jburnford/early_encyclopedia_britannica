#!/usr/bin/env python3
"""Embed all articles for GraphRAG retrieval using Voyage AI.

Splits articles into 1500-word chunks with 200-word overlap, embeds each
chunk with voyage-4-large, and saves per-edition output files.

Supports incremental mode: only re-embeds articles that changed (via
article_manifest.diff.json).

Requires: VOYAGE_API_KEY environment variable (or in .env file)

Usage:
    # Full corpus (all editions):
    python graphrag/embed_articles.py

    # Single edition:
    python graphrag/embed_articles.py --edition-year 1771

    # Incremental (only changed articles):
    python graphrag/embed_articles.py --incremental

    # Quick test:
    python graphrag/embed_articles.py --edition-year 1771 --max-articles 50

    # Use context model for long chunks:
    python graphrag/embed_articles.py --model voyage-context-3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
MANIFEST_DIFF_PATH = REPO_ROOT / "data" / "article_manifest.diff.json"

MODEL_NAME = "voyage-4-large"
CHUNK_WORDS = 1500
CHUNK_OVERLAP = 200
MIN_WORDS = 10
BATCH_SIZE = 32  # Conservative: handles long chunks safely under 120K token limit
EMBED_DIM = 1024
EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# Custom query instruction for search time
QUERY_INPUT_TYPE = "query"
DOCUMENT_INPUT_TYPE = "document"


def find_export_file(edition_year: int) -> Path:
    matches = list(EXPORT_DIR.glob(f"eb_*_{edition_year}.jsonl"))
    if len(matches) == 1:
        return matches[0]
    sys.exit(f"Error: {'No' if not matches else 'Multiple'} export file(s) for {edition_year}")


def paragraphs_from_text(text: str, min_words: int = 10) -> list[dict]:
    """Split text on paragraph boundaries. Groups short paragraphs with the next one."""
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    result = []
    char_pos = 0
    buffer = ""

    for p in raw_paras:
        # Track char position in original text
        idx = text.find(p, char_pos)
        if idx >= 0:
            char_pos = idx

        wc = len(p.split())
        if wc < min_words and not buffer:
            # Too short on its own — buffer it to merge with next
            buffer = p
            continue

        if buffer:
            p = buffer + "\n\n" + p
            buffer = ""

        words = p.split()
        result.append({
            "text": p,
            "char_start": char_pos,
            "char_end": char_pos + len(p),
            "word_count": len(words),
        })
        char_pos += len(p)

    # Flush remaining buffer
    if buffer:
        words = buffer.split()
        if words:
            result.append({
                "text": buffer,
                "char_start": char_pos,
                "char_end": char_pos + len(buffer),
                "word_count": len(words),
            })

    return result


def chunk_text(text: str) -> list[dict]:
    """Split text into overlapping chunks."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [{"text": text, "char_start": 0, "char_end": len(text), "word_count": len(words)}]

    chunks = []
    word_idx = 0
    while word_idx < len(words):
        end_idx = min(word_idx + CHUNK_WORDS, len(words))
        chunk_words = words[word_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        if word_idx == 0:
            char_start = 0
        else:
            char_start = len(" ".join(words[:word_idx])) + 1
        char_end = char_start + len(chunk_text)

        chunks.append({
            "text": chunk_text,
            "char_start": char_start,
            "char_end": min(char_end, len(text)),
            "word_count": len(chunk_words),
        })

        if end_idx >= len(words):
            break
        word_idx += CHUNK_WORDS - CHUNK_OVERLAP

    return chunks


def load_changed_ids() -> set[str] | None:
    if not MANIFEST_DIFF_PATH.exists():
        return None
    with open(MANIFEST_DIFF_PATH) as f:
        diff = json.load(f)
    changed = set(diff.get("added", [])) | set(diff.get("changed", []))
    return changed if changed else None


def init_voyage(model_name: str):
    """Initialize Voyage AI client."""
    import voyageai
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        sys.exit("Error: VOYAGE_API_KEY not set. Set it in .env or environment.")

    client = voyageai.Client(api_key=api_key)
    print(f"Voyage AI client initialized (model: {model_name})")
    return client


def embed_batch(client, texts: list[str], model_name: str) -> np.ndarray:
    """Embed a batch of texts via Voyage API."""
    result = client.embed(
        texts,
        model=model_name,
        input_type=DOCUMENT_INPUT_TYPE,
        truncation=True,
    )
    return np.array(result.embeddings, dtype=np.float32)


def process_edition(edition_year: int, client, model_name: str,
                    max_articles: int = None,
                    changed_ids: set[str] | None = None,
                    paragraph_mode: bool = False) -> tuple[int, int]:
    """Process one edition. Returns (units_embedded, tokens_used)."""
    export_file = find_export_file(edition_year)
    unit = "para" if paragraph_mode else "chunk"
    suffix = "paragraphs" if paragraph_mode else "chunks"
    output_path = EMBEDDINGS_DIR / f"eb_{edition_year}.{suffix}.jsonl"

    # Load existing embeddings for incremental mode
    existing = {}
    if changed_ids is not None and output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec[f"{unit}_id"]] = rec
        print(f"  Loaded {len(existing):,} existing {suffix}")

    # Load articles
    articles = []
    with open(export_file) as f:
        for line in f:
            art = json.loads(line)
            if art.get("word_count", 0) < MIN_WORDS:
                continue
            if changed_ids is not None and art["article_id"] not in changed_ids:
                continue
            articles.append(art)
            if max_articles and len(articles) >= max_articles:
                break

    if not articles and changed_ids is not None:
        print(f"  No changed articles in {edition_year}, keeping existing embeddings")
        return 0, 0

    print(f"  {len(articles):,} articles to embed")

    # Chunk all articles
    texts_to_embed = []
    chunk_metas = []

    for art in articles:
        if paragraph_mode:
            units = paragraphs_from_text(art.get("text", ""))
        else:
            units = chunk_text(art.get("text", ""))
        for ci, u in enumerate(units):
            uid = f"{art['article_id']}__{unit}_{ci}"
            meta = {
                f"{unit}_id": uid,
                "article_id": art["article_id"],
                "title": art["title"],
                "edition_year": art["edition_year"],
                "volume": art.get("volume", 0),
                f"{unit}_index": ci,
                f"total_{suffix}": len(units),
                "char_start": u["char_start"],
                "char_end": u["char_end"],
                "word_count": u["word_count"],
            }
            texts_to_embed.append(u["text"])
            chunk_metas.append(meta)

    if not texts_to_embed:
        print(f"  Nothing to embed")
        return 0, 0

    print(f"  {len(texts_to_embed):,} {suffix} to embed...")
    t0 = time.time()
    total_tokens = 0

    # Embed in batches
    all_embeddings = []
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i:i + BATCH_SIZE]

        # Retry with backoff on rate limits
        for attempt in range(5):
            try:
                embs = embed_batch(client, batch, model_name)
                all_embeddings.append(embs)
                total_tokens += sum(len(t.split()) * 1.3 for t in batch)  # rough token estimate
                break
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        done = min(i + BATCH_SIZE, len(texts_to_embed))
        if done % (BATCH_SIZE * 5) == 0 or done == len(texts_to_embed):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:,}/{len(texts_to_embed):,} {suffix} ({rate:.0f}/sec)")

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts_to_embed)/elapsed:.0f} {suffix}/sec)")

    # If incremental, remove old chunks for changed articles
    if changed_ids is not None:
        changed_article_ids = {art["article_id"] for art in articles}
        kept = {k: v for k, v in existing.items()
                if v["article_id"] not in changed_article_ids}
    else:
        kept = {}

    # Write output
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in kept.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for meta, emb in zip(chunk_metas, embeddings):
            meta["embedding"] = emb.tolist()
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    total = len(kept) + len(chunk_metas)
    print(f"  Wrote {total:,} {suffix} to {output_path.name}")
    return len(chunk_metas), int(total_tokens)


def main():
    parser = argparse.ArgumentParser(description="Embed articles for GraphRAG via Voyage AI")
    parser.add_argument("--edition-year", type=int,
                        help="Process single edition (default: all)")
    parser.add_argument("--max-articles", type=int,
                        help="Limit articles per edition (for testing)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-embed changed articles")
    parser.add_argument("--paragraph", action="store_true",
                        help="Embed at paragraph level instead of 1500-word chunks")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Voyage model (default: {MODEL_NAME})")
    args = parser.parse_args()

    years = [args.edition_year] if args.edition_year else EDITION_YEARS

    changed_ids = None
    if args.incremental:
        changed_ids = load_changed_ids()
        if changed_ids:
            print(f"Incremental mode: {len(changed_ids):,} changed articles")
        else:
            print("No changes detected, nothing to embed.")
            return

    mode = "paragraph" if args.paragraph else "chunk"
    print(f"Mode: {mode}")

    client = init_voyage(args.model)

    total_units = 0
    total_tokens = 0
    t_total = time.time()

    for year in years:
        print(f"\n{'='*60}")
        print(f"Edition {year}")
        print(f"{'='*60}")
        units, tokens = process_edition(year, client, args.model,
                                        args.max_articles, changed_ids,
                                        paragraph_mode=args.paragraph)
        total_units += units
        total_tokens += tokens

    elapsed = time.time() - t_total
    est_cost = total_tokens / 1_000_000 * 0.12
    print(f"\n{'='*60}")
    print(f"Total: {total_units:,} {mode}s embedded in {elapsed:.0f}s")
    print(f"Estimated tokens: {total_tokens:,} (~${est_cost:.2f})")
    print(f"Output: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
