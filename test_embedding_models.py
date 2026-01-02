#!/usr/bin/env python3
"""
Test embedding models on 18th century encyclopedia text.
Compare retrieval quality across models before full corpus embedding.

Usage:
    python test_embedding_models.py articles_1771.jsonl

Run on Nibi with GPU for speed.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time

# Test queries with expected article matches
TEST_QUERIES = [
    # Scientific concepts
    ("phlogiston combustion theory", ["CHEMISTRY", "FIRE", "COMBUSTION"]),
    ("planetary motion orbits", ["ASTRONOMY", "PLANET", "ORBIT"]),
    ("electrical fluid experiments", ["ELECTRICITY", "LIGHTNING", "THUNDER"]),
    ("smallpox inoculation variolation", ["MEDICINE", "INOCULATION", "SMALLPOX"]),

    # Historical/archaic terms
    ("chymistry transmutation metals", ["CHEMISTRY", "ALCHEMY", "METAL"]),
    ("natural philosophy principles", ["PHYSICS", "PHILOSOPHY", "NATURE"]),
    ("phlebotomy bloodletting cure", ["MEDICINE", "SURGERY", "BLOOD"]),

    # Geographic/historical
    ("American colonies revolution", ["AMERICA", "COLONY", "REVOLUTION"]),
    ("Scottish enlightenment philosophy", ["SCOTLAND", "PHILOSOPHY", "EDINBURGH"]),

    # Technical terms
    ("logarithm calculation tables", ["ALGEBRA", "LOGARITHM", "ARITHMETIC"]),
    ("fortification military engineering", ["FORTIFICATION", "MILITARY", "WAR"]),
    ("navigation longitude chronometer", ["NAVIGATION", "LONGITUDE", "GEOGRAPHY"]),
]

MODELS_TO_TEST = [
    # Baseline
    "sentence-transformers/all-mpnet-base-v2",      # 768-dim baseline

    # Standard large models
    "BAAI/bge-large-en-v1.5",                       # 1024-dim, top MTEB performer
    "intfloat/e5-large-v2",                         # 1024-dim, diverse training

    # Historical text specialists
    "nomic-ai/modernbert-embed-base",               # 768-dim, better vocab than mpnet
    "jinaai/jina-embeddings-v3",                    # 1024-dim, OCR-robust (handles 'f' for 's')
    "nvidia/NV-Embed-v2",                           # 4096-dim, latent attention for archaic syntax

    # Note: voyage-3 requires API key, not open-source
]


def load_sample_articles(jsonl_path: str, n_samples: int = 200) -> List[Dict]:
    """Load a diverse sample of articles."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            articles.append(json.loads(line))

    # Get mix of short and long articles
    articles.sort(key=lambda x: len(x['text']))

    # Sample: 50 short, 100 medium, 50 long
    n = len(articles)
    sample = (
        articles[:50] +                           # shortest
        articles[n//3:n//3+100] +                 # medium
        articles[-50:]                            # longest (treatises)
    )

    # Also ensure we have key test articles
    key_headwords = {'CHEMISTRY', 'ASTRONOMY', 'ELECTRICITY', 'MEDICINE',
                     'SURGERY', 'ALGEBRA', 'FORTIFICATION', 'NAVIGATION',
                     'AMERICA', 'SCOTLAND', 'PHILOSOPHY'}

    for a in articles:
        if a['headword'] in key_headwords and a not in sample:
            sample.append(a)

    print(f"Loaded {len(sample)} sample articles")
    return sample[:n_samples]


def chunk_pure_paragraph(article: Dict) -> List[Dict]:
    """Method 1: Pure paragraph chunking - faithful to 1771 structure."""
    text = article['text']
    headword = article['headword']

    if len(text) < 500:
        return [{'headword': headword, 'chunk_id': 0, 'text': text, 'method': 'pure_para'}]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return [
        {'headword': headword, 'chunk_id': i, 'text': p, 'method': 'pure_para'}
        for i, p in enumerate(paragraphs) if p
    ]


def chunk_context_aware(article: Dict) -> List[Dict]:
    """Method 2: Context-aware paragraph - prepend article title for global context."""
    text = article['text']
    headword = article['headword']
    edition = article.get('edition_name', 'Britannica 1771')

    if len(text) < 500:
        contextualized = f"[{headword}, {edition}] {text}"
        return [{'headword': headword, 'chunk_id': 0, 'text': contextualized, 'method': 'context_aware'}]

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return [
        {
            'headword': headword,
            'chunk_id': i,
            'text': f"[{headword}, {edition}] {p}",
            'method': 'context_aware'
        }
        for i, p in enumerate(paragraphs) if p
    ]


def chunk_sliding_window(article: Dict, window_chars: int = 4000, overlap_chars: int = 800) -> List[Dict]:
    """Method 3: Semantic sliding window - no concept cut in half."""
    text = article['text']
    headword = article['headword']

    if len(text) < window_chars:
        return [{'headword': headword, 'chunk_id': 0, 'text': text, 'method': 'sliding'}]

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + window_chars

        # Try to break at paragraph boundary
        if end < len(text):
            # Look for paragraph break near the end
            break_point = text.rfind('\n\n', start + window_chars - overlap_chars, end)
            if break_point > start:
                end = break_point

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                'headword': headword,
                'chunk_id': chunk_id,
                'text': chunk_text,
                'method': 'sliding'
            })
            chunk_id += 1

        # Move start, accounting for overlap
        start = end - overlap_chars if end < len(text) else len(text)

    return chunks


def chunk_article(article: Dict, method: str = 'context_aware', max_chars: int = 1500) -> List[Dict]:
    """Split article into chunks using specified method."""
    if method == 'pure_para':
        return chunk_pure_paragraph(article)
    elif method == 'context_aware':
        return chunk_context_aware(article)
    elif method == 'sliding':
        return chunk_sliding_window(article)
    else:
        # Default: context-aware
        return chunk_context_aware(article)


def load_model(model_name: str):
    """Load embedding model."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading {model_name}...")
    start = time.time()

    # Special handling for instructor models
    if 'instructor' in model_name.lower():
        from InstructorEmbedding import INSTRUCTOR
        model = INSTRUCTOR(model_name)
        model.instruction = "Represent the 18th century encyclopedia article for retrieval: "
    else:
        model = SentenceTransformer(model_name)

    print(f"  Loaded in {time.time() - start:.1f}s")
    return model


def embed_texts(model, texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for texts."""
    print(f"  Embedding {len(texts)} texts...")
    start = time.time()

    if 'instructor' in model_name.lower():
        # Instructor models need instruction prefix
        instruction = "Represent the 18th century encyclopedia article for retrieval: "
        texts_with_instruction = [[instruction, t] for t in texts]
        embeddings = model.encode(texts_with_instruction, batch_size=batch_size, show_progress_bar=True)
    else:
        # E5 models need "passage: " prefix for documents
        if 'e5' in model_name.lower():
            texts = ["passage: " + t for t in texts]
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    print(f"  Embedded in {time.time() - start:.1f}s ({len(texts)/(time.time()-start):.0f} texts/sec)")
    return embeddings


def embed_query(model, query: str, model_name: str) -> np.ndarray:
    """Embed a single query."""
    if 'instructor' in model_name.lower():
        instruction = "Represent the question for retrieving historical encyclopedia articles: "
        return model.encode([[instruction, query]])
    elif 'e5' in model_name.lower():
        return model.encode(["query: " + query])
    else:
        return model.encode([query])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and documents."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def evaluate_retrieval(
    model,
    model_name: str,
    chunks: List[Dict],
    chunk_embeddings: np.ndarray,
    queries: List[Tuple[str, List[str]]],
    top_k: int = 10
) -> Dict:
    """Evaluate retrieval quality."""
    results = {
        'model': model_name,
        'queries': [],
        'mrr': 0.0,
        'recall_at_5': 0.0,
        'recall_at_10': 0.0,
    }

    for query, expected_headwords in queries:
        query_emb = embed_query(model, query, model_name)
        similarities = cosine_similarity(query_emb, chunk_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]

        # Check if expected headwords are in results
        retrieved_headwords = [c['headword'] for c in top_chunks]

        # Find first relevant result (for MRR)
        first_relevant = None
        for i, hw in enumerate(retrieved_headwords):
            if hw in expected_headwords:
                first_relevant = i + 1
                break

        # Recall at k
        recall_5 = len(set(retrieved_headwords[:5]) & set(expected_headwords)) / len(expected_headwords)
        recall_10 = len(set(retrieved_headwords[:10]) & set(expected_headwords)) / len(expected_headwords)

        results['queries'].append({
            'query': query,
            'expected': expected_headwords,
            'retrieved': retrieved_headwords[:5],
            'scores': [f"{s:.3f}" for s in top_scores[:5]],
            'first_relevant_rank': first_relevant,
            'recall_at_5': recall_5,
        })

        if first_relevant:
            results['mrr'] += 1.0 / first_relevant
        results['recall_at_5'] += recall_5
        results['recall_at_10'] += recall_10

    n_queries = len(queries)
    results['mrr'] /= n_queries
    results['recall_at_5'] /= n_queries
    results['recall_at_10'] /= n_queries

    return results


def print_results(all_results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL COMPARISON - HISTORICAL TEXT RETRIEVAL")
    print("=" * 80)

    # Summary table
    print("\n### SUMMARY METRICS ###\n")
    print(f"{'Model':<40} {'MRR':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 66)
    for r in all_results:
        model_short = r['model'].split('/')[-1]
        print(f"{model_short:<40} {r['mrr']:>8.3f} {r['recall_at_5']:>8.3f} {r['recall_at_10']:>8.3f}")

    # Best model
    best = max(all_results, key=lambda x: x['mrr'])
    print(f"\nBest model by MRR: {best['model'].split('/')[-1]}")

    # Detailed query results for best model
    print(f"\n### DETAILED RESULTS ({best['model'].split('/')[-1]}) ###\n")
    for q in best['queries']:
        status = "✓" if q['first_relevant_rank'] and q['first_relevant_rank'] <= 3 else "✗"
        print(f"{status} Query: \"{q['query'][:50]}...\"")
        print(f"  Expected: {q['expected']}")
        print(f"  Retrieved: {q['retrieved']} (scores: {q['scores']})")
        if q['first_relevant_rank']:
            print(f"  First relevant at rank: {q['first_relevant_rank']}")
        print()


CHUNKING_METHODS = ['pure_para', 'context_aware', 'sliding']


def main():
    parser = argparse.ArgumentParser(description='Test embedding models on historical text')
    parser.add_argument('jsonl_file', help='Path to articles JSONL file')
    parser.add_argument('--samples', type=int, default=200, help='Number of articles to sample')
    parser.add_argument('--models', nargs='+', default=MODELS_TO_TEST, help='Models to test')
    parser.add_argument('--methods', nargs='+', default=CHUNKING_METHODS, help='Chunking methods')
    args = parser.parse_args()

    # Load articles
    print("Loading articles...")
    articles = load_sample_articles(args.jsonl_file, args.samples)

    # Test each model × chunking method combination
    all_results = []

    for model_name in args.models:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print('='*70)

        try:
            model = load_model(model_name)

            for method in args.methods:
                print(f"\n--- Chunking: {method} ---")

                # Chunk articles with this method
                all_chunks = []
                for a in articles:
                    all_chunks.extend(chunk_article(a, method=method))
                print(f"Created {len(all_chunks)} chunks")

                # Extract texts and embed
                chunk_texts = [c['text'] for c in all_chunks]
                embeddings = embed_texts(model, chunk_texts, model_name)

                # Evaluate
                results = evaluate_retrieval(model, model_name, all_chunks, embeddings, TEST_QUERIES)
                results['chunking_method'] = method
                results['num_chunks'] = len(all_chunks)
                all_results.append(results)

            # Free memory
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    print_results_matrix(all_results)

    # Save results
    output_file = Path(args.jsonl_file).stem + "_embedding_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def print_results_matrix(all_results: List[Dict]):
    """Print comparison matrix: models × chunking methods."""
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL × CHUNKING METHOD COMPARISON")
    print("=" * 80)

    # Group by model
    from collections import defaultdict
    by_model = defaultdict(dict)
    for r in all_results:
        model = r['model'].split('/')[-1]
        method = r['chunking_method']
        by_model[model][method] = r

    # Print matrix
    methods = ['pure_para', 'context_aware', 'sliding']
    print(f"\n{'Model':<30} | {'pure_para':>12} | {'context_aware':>14} | {'sliding':>12}")
    print("-" * 80)

    best_score = 0
    best_combo = None

    for model, method_results in by_model.items():
        row = f"{model:<30}"
        for method in methods:
            if method in method_results:
                mrr = method_results[method]['mrr']
                row += f" | {mrr:>12.3f}"
                if mrr > best_score:
                    best_score = mrr
                    best_combo = (model, method)
            else:
                row += f" | {'N/A':>12}"
        print(row)

    print("-" * 80)
    if best_combo:
        print(f"\nBEST: {best_combo[0]} + {best_combo[1]} (MRR: {best_score:.3f})")

    # Detailed results for best combo
    if best_combo:
        print(f"\n### DETAILED RESULTS ({best_combo[0]} + {best_combo[1]}) ###\n")
        for r in all_results:
            if r['model'].split('/')[-1] == best_combo[0] and r['chunking_method'] == best_combo[1]:
                for q in r['queries'][:5]:  # Show first 5
                    status = "✓" if q['first_relevant_rank'] and q['first_relevant_rank'] <= 3 else "✗"
                    print(f"{status} \"{q['query'][:45]}...\"")
                    print(f"   Retrieved: {q['retrieved'][:3]}")
                break


if __name__ == '__main__':
    main()
