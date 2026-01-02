#!/usr/bin/env python3
"""
Test voyage-context-3 on 1771 encyclopedia.
Compares contextualized embeddings vs standard embeddings.
"""

import json
import numpy as np
from pathlib import Path
import voyageai
from typing import List, Dict
import time

# Voyage API key
VOYAGE_API_KEY = "pa-x4Plj_6g9coynPFCuq5RmEv2TtrshNSPU3sTYPhsGc3"

# Same test queries as open-source test
TEST_QUERIES = [
    ("phlogiston combustion theory", ["CHEMISTRY", "FIRE", "COMBUSTION"]),
    ("planetary motion orbits", ["ASTRONOMY", "PLANET", "ORBIT"]),
    ("electrical fluid experiments", ["ELECTRICITY", "LIGHTNING", "THUNDER"]),
    ("smallpox inoculation variolation", ["MEDICINE", "INOCULATION", "SMALLPOX"]),
    ("chymistry transmutation metals", ["CHEMISTRY", "ALCHEMY", "METAL"]),
    ("natural philosophy principles", ["PHYSICS", "PHILOSOPHY", "NATURE"]),
    ("phlebotomy bloodletting cure", ["MEDICINE", "SURGERY", "BLOOD"]),
    ("American colonies revolution", ["AMERICA", "COLONY", "REVOLUTION"]),
    ("Scottish enlightenment philosophy", ["SCOTLAND", "PHILOSOPHY", "EDINBURGH"]),
    ("logarithm calculation tables", ["ALGEBRA", "LOGARITHM", "ARITHMETIC"]),
    ("fortification military engineering", ["FORTIFICATION", "MILITARY", "WAR"]),
    ("navigation longitude chronometer", ["NAVIGATION", "LONGITUDE", "GEOGRAPHY"]),
]


def load_articles(jsonl_path: str, max_articles: int = None) -> List[Dict]:
    """Load articles from JSONL, ensuring key test articles are included."""
    # Key headwords that must be in sample for test queries to work
    key_headwords = {'CHEMISTRY', 'ASTRONOMY', 'ELECTRICITY', 'MEDICINE',
                     'SURGERY', 'ALGEBRA', 'FORTIFICATION', 'NAVIGATION',
                     'AMERICA', 'SCOTLAND', 'PHILOSOPHY', 'FIRE', 'COMBUSTION',
                     'PLANET', 'LIGHTNING', 'ALCHEMY', 'METAL', 'PHYSICS',
                     'BLOOD', 'INOCULATION', 'COLONY', 'EDINBURGH', 'LOGARITHM',
                     'ARITHMETIC', 'MILITARY', 'WAR', 'LONGITUDE', 'GEOGRAPHY'}

    all_articles = []
    with open(jsonl_path) as f:
        for line in f:
            all_articles.append(json.loads(line))

    # Separate key articles and others
    key_articles = [a for a in all_articles if a['headword'] in key_headwords]
    other_articles = [a for a in all_articles if a['headword'] not in key_headwords]

    # Start with all key articles
    sample = list(key_articles)

    # Fill remaining slots with other articles
    if max_articles and len(sample) < max_articles:
        remaining = max_articles - len(sample)
        # Take evenly distributed sample from rest
        step = max(1, len(other_articles) // remaining)
        for i in range(0, len(other_articles), step):
            if len(sample) >= max_articles:
                break
            sample.append(other_articles[i])

    print(f"  Key articles found: {len(key_articles)}")
    print(f"  Total sample size: {len(sample)}")
    return sample[:max_articles] if max_articles else sample


def chunk_article(article: Dict, max_chars: int = 1500) -> List[Dict]:
    """Paragraph-based chunking."""
    text = article['text']
    headword = article['headword']

    if len(text) < 500:
        return [{'headword': headword, 'chunk_id': 0, 'text': text}]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ''
    chunk_id = 0

    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append({'headword': headword, 'chunk_id': chunk_id, 'text': current})
                chunk_id += 1
                current = ''
            chunks.append({'headword': headword, 'chunk_id': chunk_id, 'text': p})
            chunk_id += 1
        elif len(current) + len(p) + 2 > max_chars:
            chunks.append({'headword': headword, 'chunk_id': chunk_id, 'text': current})
            chunk_id += 1
            current = p
        else:
            current = (current + '\n\n' + p) if current else p

    if current:
        chunks.append({'headword': headword, 'chunk_id': chunk_id, 'text': current})

    return chunks


def embed_with_voyage(client, texts: List[str], model: str = "voyage-3") -> np.ndarray:
    """Embed texts using Voyage API."""
    # Reduced batch size for free tier (10K TPM limit)
    # Average text ~500 tokens, so batch of 8-10 should be safe
    all_embeddings = []
    batch_size = 8

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            result = client.embed(batch, model=model, input_type="document")
            all_embeddings.extend(result.embeddings)
            print(f"  Embedded {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        except Exception as e:
            print(f"  Rate limit hit, waiting 60s...")
            time.sleep(60)
            result = client.embed(batch, model=model, input_type="document")
            all_embeddings.extend(result.embeddings)

        if i + batch_size < len(texts):
            time.sleep(25)  # 3 RPM = 20s between requests, add buffer

    return np.array(all_embeddings)


def embed_with_voyage_context(client, documents: List[Dict]) -> np.ndarray:
    """
    Embed using voyage-context-3 with contextualized_embed API.
    Groups chunks by article and embeds with full document context.

    voyage-context-3 uses the contextualized_embed() method with nested list format:
    inputs = [[doc1_chunk1, doc1_chunk2, ...], [doc2_chunk1, doc2_chunk2, ...]]
    """
    # Group chunks by headword (article)
    by_article = {}

    for i, doc in enumerate(documents):
        hw = doc['headword']
        if hw not in by_article:
            by_article[hw] = {'chunks': [], 'indices': []}
        by_article[hw]['chunks'].append(doc['text'])
        by_article[hw]['indices'].append(i)

    # Prepare nested inputs: each inner list is chunks from one article
    nested_inputs = []
    index_mapping = []  # Maps (article_idx, chunk_idx) -> global index

    for article_idx, (hw, article_data) in enumerate(by_article.items()):
        nested_inputs.append(article_data['chunks'])
        for chunk_idx, global_idx in enumerate(article_data['indices']):
            index_mapping.append((article_idx, chunk_idx, global_idx))

    # Embed all articles at once with contextualized_embed API
    all_embeddings = [None] * len(documents)

    # Process in batches to respect rate limits
    batch_size = 5  # Articles per batch (conservative for rate limits)
    total_articles = len(nested_inputs)

    for batch_start in range(0, total_articles, batch_size):
        batch_end = min(batch_start + batch_size, total_articles)
        batch_inputs = nested_inputs[batch_start:batch_end]

        print(f"  Embedding articles {batch_start+1}-{batch_end}/{total_articles}...")

        try:
            result = client.contextualized_embed(
                inputs=batch_inputs,
                model="voyage-context-3",
                input_type="document"
            )

            # result.results is a list of ContextualizedEmbeddingsResult objects
            # Each has .embeddings (list of chunk embeddings for that document)
            for article_offset, article_result in enumerate(result.results):
                article_idx = batch_start + article_offset
                # Find the global indices for this article
                article_hw = list(by_article.keys())[article_idx]
                article_indices = by_article[article_hw]['indices']
                for chunk_idx, emb in enumerate(article_result.embeddings):
                    global_idx = article_indices[chunk_idx]
                    all_embeddings[global_idx] = emb

        except Exception as e:
            if "rate" in str(e).lower():
                print(f"  Rate limit hit, waiting 60s...")
                time.sleep(60)
                result = client.contextualized_embed(
                    inputs=batch_inputs,
                    model="voyage-context-3",
                    input_type="document"
                )
                for article_offset, article_result in enumerate(result.results):
                    article_idx = batch_start + article_offset
                    article_hw = list(by_article.keys())[article_idx]
                    article_indices = by_article[article_hw]['indices']
                    for chunk_idx, emb in enumerate(article_result.embeddings):
                        global_idx = article_indices[chunk_idx]
                        all_embeddings[global_idx] = emb
            else:
                print(f"Error: {e}")
                raise

        if batch_end < total_articles:
            time.sleep(25)  # Rate limiting: 3 RPM

    return np.array(all_embeddings)


def embed_query(client, query: str, model: str = "voyage-3") -> np.ndarray:
    """Embed a query."""
    result = client.embed([query], model=model, input_type="query")
    return np.array(result.embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def evaluate(chunks: List[Dict], embeddings: np.ndarray, client, model: str, queries: List) -> Dict:
    """Evaluate retrieval quality."""
    results = {'model': model, 'queries': [], 'mrr': 0.0, 'recall_at_5': 0.0}

    for i, (query, expected) in enumerate(queries):
        print(f"  Evaluating query {i+1}/{len(queries)}: {query[:30]}...")
        # Rate limiting for query embedding
        try:
            query_emb = embed_query(client, query, model.replace("-context", ""))
        except Exception as e:
            print(f"  Rate limit on query {i+1}, waiting 60s...")
            time.sleep(60)
            query_emb = embed_query(client, query, model.replace("-context", ""))

        similarities = cosine_similarity(query_emb, embeddings)[0]
        time.sleep(25)  # Rate limiting between queries

        top_indices = np.argsort(similarities)[::-1][:10]
        top_chunks = [chunks[idx] for idx in top_indices]
        top_scores = [similarities[idx] for idx in top_indices]
        retrieved = [c['headword'] for c in top_chunks]

        first_relevant = None
        for rank, hw in enumerate(retrieved):
            if hw in expected:
                first_relevant = rank + 1
                break

        recall_5 = len(set(retrieved[:5]) & set(expected)) / len(expected)

        results['queries'].append({
            'query': query,
            'expected': expected,
            'retrieved': retrieved[:5],
            'scores': [f"{s:.3f}" for s in top_scores[:5]],
            'first_relevant_rank': first_relevant,
        })

        if first_relevant:
            results['mrr'] += 1.0 / first_relevant
        results['recall_at_5'] += recall_5

    n = len(queries)
    results['mrr'] /= n
    results['recall_at_5'] /= n

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file', help='Articles JSONL file')
    parser.add_argument('--max-articles', type=int, default=200, help='Max articles to test')
    args = parser.parse_args()

    # Initialize Voyage client
    client = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Load and chunk articles
    print("Loading articles...")
    articles = load_articles(args.jsonl_file, args.max_articles)
    print(f"Loaded {len(articles)} articles")

    print("Chunking...")
    all_chunks = []
    for a in articles:
        all_chunks.extend(chunk_article(a))
    print(f"Created {len(all_chunks)} chunks")

    chunk_texts = [c['text'] for c in all_chunks]

    # Test voyage-3 (standard)
    print("\n=== Testing voyage-3 (standard) ===")
    start = time.time()
    emb_standard = embed_with_voyage(client, chunk_texts, "voyage-3")
    print(f"Embedded in {time.time()-start:.1f}s")
    results_standard = evaluate(all_chunks, emb_standard, client, "voyage-3", TEST_QUERIES)

    # Test voyage-context-3 (contextualized)
    print("\n=== Testing voyage-context-3 (contextualized) ===")
    start = time.time()
    emb_context = embed_with_voyage_context(client, all_chunks)
    print(f"Embedded in {time.time()-start:.1f}s")
    results_context = evaluate(all_chunks, emb_context, client, "voyage-context-3", TEST_QUERIES)

    # Print comparison
    print("\n" + "=" * 60)
    print("VOYAGE EMBEDDING COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'MRR':>10} {'Recall@5':>10}")
    print("-" * 50)
    print(f"{'voyage-3 (standard)':<25} {results_standard['mrr']:>10.3f} {results_standard['recall_at_5']:>10.3f}")
    print(f"{'voyage-context-3':<25} {results_context['mrr']:>10.3f} {results_context['recall_at_5']:>10.3f}")

    # Detailed comparison for queries where they differ
    print("\n=== Query-by-Query Comparison ===")
    for q_std, q_ctx in zip(results_standard['queries'], results_context['queries']):
        std_rank = q_std['first_relevant_rank']
        ctx_rank = q_ctx['first_relevant_rank']

        if std_rank != ctx_rank:
            better = "CONTEXT" if (ctx_rank and (not std_rank or ctx_rank < std_rank)) else "STANDARD"
            print(f"\n{q_std['query'][:40]}...")
            print(f"  Standard: rank {std_rank}, retrieved {q_std['retrieved'][:3]}")
            print(f"  Context:  rank {ctx_rank}, retrieved {q_ctx['retrieved'][:3]}")
            print(f"  Winner: {better}")

    # Save results
    all_results = {
        'voyage-3': results_standard,
        'voyage-context-3': results_context,
        'num_articles': len(articles),
        'num_chunks': len(all_chunks),
    }

    output_file = Path(args.jsonl_file).stem + "_voyage_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
