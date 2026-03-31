#!/usr/bin/env python3
"""Embed all articles for GraphRAG retrieval using chunked embeddings.

Splits articles into 1500-word chunks with 200-word overlap, embeds each
chunk with nomic-embed-text-v1.5, and saves per-edition output files.

Supports incremental mode: only re-embeds articles that changed (via
article_manifest.diff.json).

Usage:
    # Full corpus (all editions):
    python graphrag/embed_articles.py

    # Single edition:
    python graphrag/embed_articles.py --edition-year 1771

    # Incremental (only changed articles):
    python graphrag/embed_articles.py --incremental

    # Quick test:
    python graphrag/embed_articles.py --edition-year 1771 --max-articles 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = REPO_ROOT / "data" / "export"
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
MANIFEST_DIFF_PATH = REPO_ROOT / "data" / "article_manifest.diff.json"

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
CHUNK_WORDS = 1500
CHUNK_OVERLAP = 200
MIN_WORDS = 10
BATCH_SIZE = 16  # 8B model on A100 80GB — conservative for 1500-word chunks
EMBED_DIM = 1024  # Matryoshka: use 1024 for quality/storage balance
EDITION_YEARS = [1771, 1778, 1797, 1810, 1815, 1823, 1842, 1860]

# Instruction prefix for Qwen3-Embedding (improves retrieval 1-5%)
# Qwen3-Embedding: documents get NO prompt prefix.
# Queries use prompt_name="query" at search time.
# Custom query instruction for this domain (used at search time, not here):
QUERY_INSTRUCTION = (
    "Instruct: Find relevant passages from historical Encyclopedia "
    "Britannica articles about this topic\nQuery: "
)


def find_export_file(edition_year: int) -> Path:
    matches = list(EXPORT_DIR.glob(f"eb_*_{edition_year}.jsonl"))
    if len(matches) == 1:
        return matches[0]
    sys.exit(f"Error: {'No' if not matches else 'Multiple'} export file(s) for {edition_year}")


def chunk_text(text: str) -> list[dict]:
    """Split text into overlapping chunks. Returns list of {text, char_start, char_end, word_count}."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [{"text": text, "char_start": 0, "char_end": len(text), "word_count": len(words)}]

    chunks = []
    word_idx = 0
    while word_idx < len(words):
        end_idx = min(word_idx + CHUNK_WORDS, len(words))
        chunk_words = words[word_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        # Find char positions in original text
        if word_idx == 0:
            char_start = 0
        else:
            # Approximate char_start by counting characters of preceding words
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


def load_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding_side="left", trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()

    print(f"  Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
    return (model, tokenizer), device


def _last_token_pool(hidden_states, attention_mask):
    """Pool the last non-padding token (Qwen3-Embedding pooling strategy)."""
    import torch
    seq_lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[
        torch.arange(hidden_states.shape[0], device=hidden_states.device),
        seq_lengths,
    ]


def embed_batch(model_tuple, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    model, tokenizer = model_tuple
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=8192, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            embs = _last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embs = F.normalize(embs[:, :EMBED_DIM], p=2, dim=1)
            all_embs.append(embs.cpu().float().numpy())

    return np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]


def process_edition(edition_year: int, model_tuple, device: str,
                    max_articles: int = None,
                    changed_ids: set[str] | None = None) -> int:
    """Process one edition. Returns number of chunks embedded."""
    export_file = find_export_file(edition_year)
    output_path = EMBEDDINGS_DIR / f"eb_{edition_year}.chunks.jsonl"
    checkpoint_path = EMBEDDINGS_DIR / f"eb_{edition_year}.checkpoint.json"

    # Load existing embeddings for incremental mode
    existing = {}
    if changed_ids is not None and output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["chunk_id"]] = rec
        print(f"  Loaded {len(existing):,} existing chunks")

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
        return 0

    print(f"  {len(articles):,} articles to embed")

    # Load checkpoint
    checkpoint = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

    # Chunk and embed
    all_chunks = []
    texts_to_embed = []
    chunk_metas = []

    for art in articles:
        if art["article_id"] in checkpoint:
            continue

        chunks = chunk_text(art.get("text", ""))
        for ci, chunk in enumerate(chunks):
            chunk_id = f"{art['article_id']}__chunk_{ci}"
            meta = {
                "chunk_id": chunk_id,
                "article_id": art["article_id"],
                "title": art["title"],
                "edition_year": art["edition_year"],
                "volume": art.get("volume", 0),
                "chunk_index": ci,
                "total_chunks": len(chunks),
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "word_count": chunk["word_count"],
            }
            texts_to_embed.append(chunk["text"])
            chunk_metas.append(meta)

    if not texts_to_embed:
        print(f"  Nothing to embed (all checkpointed)")
        return 0

    print(f"  {len(texts_to_embed):,} chunks to embed...")
    t0 = time.time()

    # Embed in batches with progress
    all_embeddings = []
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        embs = embed_batch(model_tuple, batch)
        all_embeddings.append(embs)

        # Progress
        done = min(i + BATCH_SIZE, len(texts_to_embed))
        if done % (BATCH_SIZE * 10) == 0 or done == len(texts_to_embed):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:,}/{len(texts_to_embed):,} chunks ({rate:.0f}/sec)")

        # Checkpoint every 50 batches
        if (i // BATCH_SIZE) % 50 == 49:
            processed_ids = {m["article_id"] for m in chunk_metas[:done]}
            checkpoint.update({aid: True for aid in processed_ids})
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts_to_embed)/elapsed:.0f} chunks/sec)")

    # If incremental, remove old chunks for changed articles and keep the rest
    if changed_ids is not None:
        changed_article_ids = {art["article_id"] for art in articles}
        kept = {k: v for k, v in existing.items()
                if v["article_id"] not in changed_article_ids}
    else:
        kept = {}

    # Write output
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Write kept chunks (without embeddings — they're already in file)
        for rec in kept.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Write new chunks with embeddings
        for meta, emb in zip(chunk_metas, embeddings):
            meta["embedding"] = emb.tolist()
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    total = len(kept) + len(chunk_metas)
    print(f"  Wrote {total:,} chunks to {output_path.name}")
    return len(chunk_metas)


def main():
    parser = argparse.ArgumentParser(description="Embed articles for GraphRAG")
    parser.add_argument("--edition-year", type=int,
                        help="Process single edition (default: all)")
    parser.add_argument("--max-articles", type=int,
                        help="Limit articles per edition (for testing)")
    parser.add_argument("--incremental", action="store_true",
                        help="Only re-embed changed articles")
    parser.add_argument("--model", default=MODEL_NAME)
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

    model_tuple, device = load_model(args.model)

    total_chunks = 0
    t_total = time.time()

    for year in years:
        print(f"\n{'='*60}")
        print(f"Edition {year}")
        print(f"{'='*60}")
        chunks = process_edition(year, model_tuple, device, args.max_articles, changed_ids)
        total_chunks += chunks

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Total: {total_chunks:,} chunks embedded in {elapsed:.0f}s")
    print(f"Output: {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
