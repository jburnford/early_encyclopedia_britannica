#!/usr/bin/env python3
"""Simple RAG query over Encyclopedia Britannica embeddings.

Usage:
    python graphrag/query.py "What did the encyclopedia say about phlogiston?"
    python graphrag/query.py "Joseph Black chemistry" --top-k 10
    python graphrag/query.py "slavery in the Caribbean" --edition 1797
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings"
INDEX_PATH = EMBEDDINGS_DIR / "all_chunks.npy"
META_PATH = EMBEDDINGS_DIR / "all_chunks_meta.jsonl"
EXPORT_DIR = REPO_ROOT / "data" / "export"


def load_index():
    print("Loading index...", end=" ", flush=True)
    t0 = time.time()
    emb_matrix = np.load(INDEX_PATH)
    meta = []
    with open(META_PATH) as f:
        for line in f:
            meta.append(json.loads(line))
    print(f"{len(meta):,} chunks in {time.time()-t0:.1f}s")
    return emb_matrix, meta


def embed_query(query: str) -> np.ndarray:
    import voyageai
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = client.embed([query], model="voyage-4-large", input_type="query")
    return np.array(result.embeddings[0], dtype=np.float32)


def search(query_emb: np.ndarray, emb_matrix: np.ndarray, meta: list,
           top_k: int = 5, edition: int = None) -> list[dict]:
    # Cosine similarity (embeddings are already normalized)
    scores = emb_matrix @ query_emb

    if edition:
        mask = np.array([m["edition_year"] == edition for m in meta])
        scores = np.where(mask, scores, -1)

    top_indices = np.argsort(scores)[::-1][:top_k * 3]  # oversample to dedup

    # Deduplicate by article (keep best chunk per article)
    seen_articles = set()
    results = []
    for idx in top_indices:
        m = meta[idx]
        article_key = (m["title"], m["edition_year"])
        if article_key in seen_articles:
            continue
        seen_articles.add(article_key)
        results.append({**m, "score": float(scores[idx])})
        if len(results) >= top_k:
            break

    return results


def get_chunk_text(chunk_id: str, edition_year: int) -> str:
    """Retrieve the actual chunk text from the per-edition embedding file."""
    fp = EMBEDDINGS_DIR / f"eb_{edition_year}.chunks.jsonl"
    with open(fp) as f:
        for line in f:
            rec = json.loads(line)
            if rec["chunk_id"] == chunk_id:
                # Text was not stored in chunks — get from export
                break
    # Fall back to export file
    for fp in EXPORT_DIR.glob(f"eb_*_{edition_year}.jsonl"):
        with open(fp) as f:
            for line in f:
                art = json.loads(line)
                if art["article_id"] == chunk_id.split("__chunk_")[0]:
                    words = art["text"].split()
                    # Approximate chunk boundaries
                    chunk_idx = int(chunk_id.split("__chunk_")[1])
                    start = chunk_idx * (1500 - 200)
                    end = start + 1500
                    return " ".join(words[start:end])
    return "(text not found)"


def main():
    parser = argparse.ArgumentParser(description="RAG query over Encyclopedia Britannica")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--edition", type=int, help="Filter to specific edition year")
    parser.add_argument("--show-text", action="store_true", help="Show chunk text")
    args = parser.parse_args()

    emb_matrix, meta = load_index()

    print(f"Query: {args.query}")
    print(f"Embedding query...", end=" ", flush=True)
    query_emb = embed_query(args.query)
    print("done")

    results = search(query_emb, emb_matrix, meta, args.top_k, args.edition)

    print(f"\nTop {len(results)} results:")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['edition_year']}] {r['title']} "
              f"(chunk {r['chunk_index']}, {r['word_count']}w) "
              f"score={r['score']:.3f}")
        if args.show_text:
            text = get_chunk_text(r["chunk_id"], r["edition_year"])
            print(f"   {text[:300]}...")
            print()


if __name__ == "__main__":
    main()
