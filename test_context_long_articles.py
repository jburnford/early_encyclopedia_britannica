#!/usr/bin/env python3
"""
Test voyage-context-3 on LONG articles where context should matter.

This test uses articles near the 32K token limit with AMBIGUOUS queries
that require document context to resolve.

Hypothesis: voyage-context-3 should outperform voyage-3 when:
1. Articles are long (100+ chunks)
2. Queries match content that appears in multiple articles
3. The chunk alone doesn't reveal which article it's from
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

import voyageai

# =============================================================================
# TARGET ARTICLES - Long articles near 32K token limit
# =============================================================================

TARGET_ARTICLES = ['GLASS', 'PRINTING', 'EXPERIMENTAL PHILOSOPHY']

# =============================================================================
# TEST QUERIES - Designed to be AMBIGUOUS without context
# =============================================================================

# These queries could match multiple articles - context should help disambiguate
AMBIGUOUS_QUERIES = [
    # "The French method" appears in both GLASS and PRINTING
    {
        'query': "the French improvements to this art and their methods",
        'expected': ['GLASS', 'PRINTING'],
        'type': 'cross_article_ambiguous',
        'note': 'Both articles discuss French contributions',
    },
    {
        'query': "the inventor who first discovered this technique in ancient times",
        'expected': ['GLASS', 'PRINTING'],
        'type': 'cross_article_ambiguous',
        'note': 'Both discuss invention history',
    },
    {
        'query': "the process requires careful preparation of materials beforehand",
        'expected': ['GLASS', 'PRINTING'],
        'type': 'cross_article_ambiguous',
        'note': 'Generic process language',
    },
    {
        'query': "the practice was known among the ancient Romans and Greeks",
        'expected': ['GLASS', 'PRINTING'],
        'type': 'cross_article_ambiguous',
        'note': 'Both discuss ancient origins',
    },

    # Article-specific queries - should find the RIGHT article
    {
        'query': "melting sand with alkaline salts in a furnace produces transparent material",
        'expected': ['GLASS'],
        'type': 'article_specific',
        'note': 'Glass manufacturing process',
    },
    {
        'query': "moveable type letters arranged for impressions on paper",
        'expected': ['PRINTING'],
        'type': 'article_specific',
        'note': 'Printing specific terminology',
    },
    {
        'query': "vitrification of materials at high temperatures",
        'expected': ['GLASS'],
        'type': 'article_specific',
        'note': 'Glass-specific term',
    },
    {
        'query': "copper plate engraving for producing illustrations",
        'expected': ['PRINTING'],
        'type': 'article_specific',
        'note': 'Printing specific process',
    },
    {
        'query': "experiments and observations as the foundation of knowledge",
        'expected': ['EXPERIMENTAL PHILOSOPHY'],
        'type': 'article_specific',
        'note': 'Philosophy of science',
    },
    {
        'query': "ocular demonstration and sensory evidence for truth",
        'expected': ['EXPERIMENTAL PHILOSOPHY'],
        'type': 'article_specific',
        'note': 'Empiricism concepts',
    },

    # Within-article retrieval - find specific SECTION in long article
    {
        'query': "the color of glass and how to produce different tints",
        'expected': ['GLASS'],
        'type': 'within_article_section',
        'note': 'Should find coloring section in GLASS',
    },
    {
        'query': "window glass versus mirror glass manufacturing differences",
        'expected': ['GLASS'],
        'type': 'within_article_section',
        'note': 'Should find specific glass types section',
    },
    {
        'query': "ink preparation and composition for printing",
        'expected': ['PRINTING'],
        'type': 'within_article_section',
        'note': 'Should find ink section in PRINTING',
    },
    {
        'query': "the press mechanism and how it applies pressure",
        'expected': ['PRINTING'],
        'type': 'within_article_section',
        'note': 'Should find press mechanics section',
    },
]


@dataclass
class Article:
    headword: str
    text: str
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            self.article_id = f"{self.headword}_{hash(self.text[:100]) % 10000}"


@dataclass
class Chunk:
    headword: str
    article_id: str
    chunk_id: int
    text: str


def load_target_articles(jsonl_path: str) -> List[Article]:
    """Load the target long articles."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data['headword'] in TARGET_ARTICLES:
                articles.append(Article(
                    headword=data['headword'],
                    text=data['text'],
                ))

    # Sort by target order
    articles.sort(key=lambda a: TARGET_ARTICLES.index(a.headword))
    return articles


def chunk_article(article: Article, chunk_size: int = 800) -> List[Chunk]:
    """Create chunks from article using paragraph-aware splitting."""
    text = article.text
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_text = ''
    chunk_id = 0

    for para in paragraphs:
        if len(current_text) + len(para) + 2 > chunk_size and current_text:
            chunks.append(Chunk(
                headword=article.headword,
                article_id=article.article_id,
                chunk_id=chunk_id,
                text=current_text.strip(),
            ))
            chunk_id += 1
            current_text = para
        else:
            current_text = (current_text + '\n\n' + para) if current_text else para

    if current_text:
        chunks.append(Chunk(
            headword=article.headword,
            article_id=article.article_id,
            chunk_id=chunk_id,
            text=current_text.strip(),
        ))

    return chunks


class VoyageEmbedder:
    """Voyage AI embedder with support for both standard and contextual."""

    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)

    def embed_standard(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Embed using voyage-3 (standard)."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            result = self.client.embed(batch, model="voyage-3", input_type="document")
            all_embeddings.extend(result.embeddings)
            print(f"    voyage-3: {min(i+batch_size, len(texts))}/{len(texts)} chunks")
        return np.array(all_embeddings)

    def embed_contextual(self, articles_chunks: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Embed using voyage-context-3 with article grouping."""
        article_ids = list(articles_chunks.keys())
        nested_inputs = [articles_chunks[aid] for aid in article_ids]

        result = self.client.contextualized_embed(
            inputs=nested_inputs,
            model="voyage-context-3",
            input_type="document",
            output_dimension=1024,
        )

        all_results = {}
        for i, aid in enumerate(article_ids):
            all_results[aid] = np.array(result.results[i].embeddings)

        print(f"    voyage-context-3: {len(article_ids)} articles embedded")
        return all_results

    def embed_query(self, query: str, model: str = "voyage-3") -> np.ndarray:
        """Embed a query."""
        if model == "voyage-context-3":
            result = self.client.contextualized_embed(
                inputs=[[query]],
                model="voyage-context-3",
                input_type="query",
                output_dimension=1024,
            )
            return np.array(result.results[0].embeddings[0])
        else:
            result = self.client.embed([query], model=model, input_type="query")
            return np.array(result.embeddings[0])


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def evaluate(chunks: List[Chunk], embeddings: np.ndarray,
             queries: List[Dict], embedder: VoyageEmbedder,
             query_model: str) -> Dict:
    """Evaluate retrieval on test queries."""
    results = {
        'queries': [],
        'mrr': 0.0,
        'recall_at_5': 0.0,
        'by_type': {},
    }

    for q in queries:
        query_emb = embedder.embed_query(q['query'], model=query_model)
        similarities = cosine_similarity(query_emb, embeddings)

        top_indices = np.argsort(similarities)[::-1][:10]
        top_chunks = [chunks[idx] for idx in top_indices]
        top_scores = [similarities[idx] for idx in top_indices]
        retrieved = [c.headword for c in top_chunks]

        # Find first relevant
        expected_upper = [hw.upper() for hw in q['expected']]
        first_relevant = None
        for rank, hw in enumerate(retrieved, 1):
            if hw.upper() in expected_upper:
                first_relevant = rank
                break

        recall_5 = len(set(r.upper() for r in retrieved[:5]) & set(expected_upper)) / len(expected_upper)

        result = {
            'query': q['query'][:60],
            'type': q['type'],
            'expected': q['expected'],
            'retrieved': retrieved[:5],
            'scores': [f"{s:.3f}" for s in top_scores[:5]],
            'first_relevant_rank': first_relevant,
            'recall_5': recall_5,
        }
        results['queries'].append(result)

        if first_relevant:
            results['mrr'] += 1.0 / first_relevant
        results['recall_at_5'] += recall_5

        # Track by query type
        qtype = q['type']
        if qtype not in results['by_type']:
            results['by_type'][qtype] = {'mrr': 0, 'count': 0}
        results['by_type'][qtype]['count'] += 1
        if first_relevant:
            results['by_type'][qtype]['mrr'] += 1.0 / first_relevant

    n = len(queries)
    results['mrr'] /= n
    results['recall_at_5'] /= n

    for qtype in results['by_type']:
        count = results['by_type'][qtype]['count']
        results['by_type'][qtype]['mrr'] /= count

    return results


def main():
    print("=" * 70)
    print("CONTEXT TEST: Long Articles with Ambiguous Queries")
    print("=" * 70)

    # Load articles
    print("\nLoading target articles...")
    articles = load_target_articles('articles_1815_clean.jsonl')

    for a in articles:
        words = len(a.text.split())
        tokens = int(words * 1.3)
        print(f"  {a.headword}: {words:,} words (~{tokens:,} tokens)")

    # Create chunks
    print("\nCreating chunks...")
    all_chunks: List[Chunk] = []
    articles_chunks: Dict[str, List[str]] = {}

    for article in articles:
        chunks = chunk_article(article, chunk_size=800)
        all_chunks.extend(chunks)
        articles_chunks[article.article_id] = [c.text for c in chunks]
        print(f"  {article.headword}: {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(articles)} articles")

    # Estimate tokens
    total_tokens = sum(len(' '.join(chunks).split()) * 1.3
                       for chunks in articles_chunks.values())
    print(f"Estimated tokens: ~{int(total_tokens):,} (limit: 120K)")

    # Initialize embedder
    embedder = VoyageEmbedder(VOYAGE_API_KEY)

    # === Embed with voyage-3 (standard) ===
    print("\n" + "=" * 50)
    print("Embedding with voyage-3 (standard)...")
    start = time.time()
    chunk_texts = [c.text for c in all_chunks]
    emb_standard = embedder.embed_standard(chunk_texts)
    time_standard = time.time() - start
    print(f"  Time: {time_standard:.1f}s")

    # === Embed with voyage-context-3 ===
    print("\nEmbedding with voyage-context-3 (contextual)...")
    start = time.time()
    emb_contextual_dict = embedder.embed_contextual(articles_chunks)
    time_contextual = time.time() - start
    print(f"  Time: {time_contextual:.1f}s")

    # Flatten contextual embeddings
    emb_contextual = []
    for chunk in all_chunks:
        article_id = chunk.article_id
        article_chunks = articles_chunks[article_id]
        chunk_idx = article_chunks.index(chunk.text)
        emb_contextual.append(emb_contextual_dict[article_id][chunk_idx])
    emb_contextual = np.array(emb_contextual)

    # === Evaluate ===
    print("\n" + "=" * 50)
    print("Evaluating retrieval...")

    print("\n  voyage-3 queries...")
    results_standard = evaluate(all_chunks, emb_standard, AMBIGUOUS_QUERIES,
                                embedder, query_model="voyage-3")

    print("  voyage-context-3 queries...")
    results_contextual = evaluate(all_chunks, emb_contextual, AMBIGUOUS_QUERIES,
                                  embedder, query_model="voyage-context-3")

    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<25} {'MRR':>10} {'Recall@5':>10} {'Time':>10}")
    print("-" * 55)
    print(f"{'voyage-3 (standard)':<25} {results_standard['mrr']:>10.3f} "
          f"{results_standard['recall_at_5']:>10.3f} {time_standard:>9.1f}s")
    print(f"{'voyage-context-3':<25} {results_contextual['mrr']:>10.3f} "
          f"{results_contextual['recall_at_5']:>10.3f} {time_contextual:>9.1f}s")

    # Improvement
    if results_standard['mrr'] > 0:
        improvement = (results_contextual['mrr'] - results_standard['mrr']) / results_standard['mrr'] * 100
        print(f"\nContext improvement: {improvement:+.1f}% MRR")

    # Results by query type
    print("\n" + "-" * 55)
    print("Results by Query Type:")
    for qtype in ['cross_article_ambiguous', 'article_specific', 'within_article_section']:
        if qtype in results_standard['by_type']:
            std_mrr = results_standard['by_type'][qtype]['mrr']
            ctx_mrr = results_contextual['by_type'][qtype]['mrr']
            diff = ctx_mrr - std_mrr
            winner = "CONTEXT" if diff > 0.01 else "STANDARD" if diff < -0.01 else "TIE"
            print(f"  {qtype}:")
            print(f"    Standard: {std_mrr:.3f} | Context: {ctx_mrr:.3f} | {winner}")

    # Query-by-query comparison
    print("\n" + "=" * 70)
    print("QUERY-BY-QUERY ANALYSIS")
    print("=" * 70)

    context_wins = 0
    standard_wins = 0
    ties = 0

    for q_std, q_ctx in zip(results_standard['queries'], results_contextual['queries']):
        std_rank = q_std['first_relevant_rank']
        ctx_rank = q_ctx['first_relevant_rank']

        if ctx_rank and (not std_rank or ctx_rank < std_rank):
            status = "CONTEXT WINS"
            context_wins += 1
        elif std_rank and (not ctx_rank or std_rank < ctx_rank):
            status = "STANDARD WINS"
            standard_wins += 1
        else:
            status = "TIE"
            ties += 1

        print(f"\n[{q_std['type']}] {q_std['query']}...")
        print(f"  Expected: {q_std['expected']}")
        print(f"  Standard: rank={std_rank}, got {q_std['retrieved'][:3]}")
        print(f"  Context:  rank={ctx_rank}, got {q_ctx['retrieved'][:3]}")
        print(f"  Result: {status}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: Context wins: {context_wins} | Standard wins: {standard_wins} | Ties: {ties}")
    print("=" * 70)

    # Save results
    output = {
        'articles': [a.headword for a in articles],
        'num_chunks': len(all_chunks),
        'voyage-3': results_standard,
        'voyage-context-3': results_contextual,
        'context_wins': context_wins,
        'standard_wins': standard_wins,
        'ties': ties,
    }

    with open('context_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to context_test_results.json")


if __name__ == '__main__':
    main()
