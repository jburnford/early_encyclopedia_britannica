#!/usr/bin/env python3
"""
Test voyage-context-3 with TRULY ambiguous queries.

These queries are designed to match chunks that:
1. Don't explicitly mention their article's topic
2. Could plausibly come from multiple articles
3. Require document context to disambiguate
"""

import json
import os
import time
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

import voyageai

TARGET_ARTICLES = ['GLASS', 'PRINTING', 'EXPERIMENTAL PHILOSOPHY']

# Queries designed to match AMBIGUOUS chunks (that don't mention their topic)
TRULY_AMBIGUOUS_QUERIES = [
    # Matches GLASS chunk about "manufactory" and "commerce" - no mention of glass
    {
        'query': "manufactory established in Lancashire commerce proprietors",
        'expected': ['GLASS'],
        'note': 'Chunk discusses glass manufactory but never says "glass"',
    },
    # Matches GLASS chunk about "vitrification" theory
    {
        'query': "theory of vitrification solid bodies vehement action of fire vapour",
        'expected': ['GLASS'],
        'note': 'Technical discussion without explicit glass mention',
    },
    # Matches GLASS chunk about "tubes swelling" near fire
    {
        'query': "swelling of the tubes towards the fire heat expand",
        'expected': ['GLASS'],
        'note': 'Scientific observation that could apply to many materials',
    },

    # Matches PRINTING chunk about "religion promoted by the art"
    {
        'query': "cause of religion promoted by the art sacred books",
        'expected': ['PRINTING'],
        'note': 'Discusses printing\'s religious impact without saying "print"',
    },
    # Matches PRINTING chunk about Junius and Cornelius narrative
    {
        'query': "narrative of Junius Nicolaus Galius Cornelius roguery servant",
        'expected': ['PRINTING'],
        'note': 'Historical anecdote from printing history',
    },
    # Matches PRINTING chunk about Laurentius and "the art"
    {
        'query': "Laurentius first devised rough specimen of the art wooden types",
        'expected': ['PRINTING'],
        'note': 'About printing inventor but says "the art" not "printing"',
    },

    # Generic queries that could match either article
    {
        'query': "the art was established through spirited exertions of proprietors",
        'expected': ['GLASS', 'PRINTING'],
        'note': 'Generic "art" + commerce language',
    },
    {
        'query': "the difficulties attendant upon new undertakings foreign establishments",
        'expected': ['GLASS', 'PRINTING'],
        'note': 'Business/trade challenges - could be either',
    },
    {
        'query': "the theory and phenomena observed by natural philosophers",
        'expected': ['GLASS', 'EXPERIMENTAL PHILOSOPHY'],
        'note': 'Scientific observation language',
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


def load_articles(jsonl_path: str) -> List[Article]:
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data['headword'] in TARGET_ARTICLES:
                articles.append(Article(headword=data['headword'], text=data['text']))
    articles.sort(key=lambda a: TARGET_ARTICLES.index(a.headword))
    return articles


def chunk_article(article: Article, chunk_size: int = 800) -> List[Chunk]:
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
    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)

    def embed_standard(self, texts: List[str]) -> np.ndarray:
        result = self.client.embed(texts, model="voyage-3", input_type="document")
        return np.array(result.embeddings)

    def embed_contextual(self, articles_chunks: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        article_ids = list(articles_chunks.keys())
        nested_inputs = [articles_chunks[aid] for aid in article_ids]
        result = self.client.contextualized_embed(
            inputs=nested_inputs,
            model="voyage-context-3",
            input_type="document",
            output_dimension=1024,
        )
        return {aid: np.array(result.results[i].embeddings)
                for i, aid in enumerate(article_ids)}

    def embed_query(self, query: str, contextual: bool = False) -> np.ndarray:
        if contextual:
            result = self.client.contextualized_embed(
                inputs=[[query]],
                model="voyage-context-3",
                input_type="query",
                output_dimension=1024,
            )
            return np.array(result.results[0].embeddings[0])
        else:
            result = self.client.embed([query], model="voyage-3", input_type="query")
            return np.array(result.embeddings[0])


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def main():
    print("=" * 70)
    print("TRULY AMBIGUOUS QUERY TEST")
    print("Testing chunks that don't mention their article topic")
    print("=" * 70)

    # Load and chunk
    articles = load_articles('articles_1815_clean.jsonl')
    all_chunks = []
    articles_chunks = {}

    for article in articles:
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
        articles_chunks[article.article_id] = [c.text for c in chunks]
        print(f"  {article.headword}: {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks")

    # Embed
    embedder = VoyageEmbedder(VOYAGE_API_KEY)

    print("\nEmbedding with voyage-3...")
    chunk_texts = [c.text for c in all_chunks]
    emb_standard = embedder.embed_standard(chunk_texts)

    print("Embedding with voyage-context-3...")
    emb_ctx_dict = embedder.embed_contextual(articles_chunks)
    emb_contextual = []
    for chunk in all_chunks:
        aid = chunk.article_id
        idx = articles_chunks[aid].index(chunk.text)
        emb_contextual.append(emb_ctx_dict[aid][idx])
    emb_contextual = np.array(emb_contextual)

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    std_mrr, ctx_mrr = 0, 0
    context_wins, standard_wins, ties = 0, 0, 0

    for q in TRULY_AMBIGUOUS_QUERIES:
        # Standard
        q_std = embedder.embed_query(q['query'], contextual=False)
        sims_std = cosine_similarity(q_std, emb_standard)
        top_std = np.argsort(sims_std)[::-1][:5]
        retrieved_std = [all_chunks[i].headword for i in top_std]
        scores_std = [sims_std[i] for i in top_std]

        # Contextual
        q_ctx = embedder.embed_query(q['query'], contextual=True)
        sims_ctx = cosine_similarity(q_ctx, emb_contextual)
        top_ctx = np.argsort(sims_ctx)[::-1][:5]
        retrieved_ctx = [all_chunks[i].headword for i in top_ctx]
        scores_ctx = [sims_ctx[i] for i in top_ctx]

        # Find first relevant rank
        expected = [e.upper() for e in q['expected']]

        rank_std = None
        for i, hw in enumerate(retrieved_std, 1):
            if hw.upper() in expected:
                rank_std = i
                break

        rank_ctx = None
        for i, hw in enumerate(retrieved_ctx, 1):
            if hw.upper() in expected:
                rank_ctx = i
                break

        if rank_std:
            std_mrr += 1.0 / rank_std
        if rank_ctx:
            ctx_mrr += 1.0 / rank_ctx

        # Determine winner
        if rank_ctx and (not rank_std or rank_ctx < rank_std):
            winner = "CONTEXT WINS"
            context_wins += 1
        elif rank_std and (not rank_ctx or rank_std < rank_ctx):
            winner = "STANDARD WINS"
            standard_wins += 1
        else:
            winner = "TIE"
            ties += 1

        print(f"\nQuery: {q['query'][:55]}...")
        print(f"  Note: {q['note']}")
        print(f"  Expected: {q['expected']}")
        print(f"  Standard: rank={rank_std}, got {retrieved_std[:3]}, scores={[f'{s:.3f}' for s in scores_std[:3]]}")
        print(f"  Context:  rank={rank_ctx}, got {retrieved_ctx[:3]}, scores={[f'{s:.3f}' for s in scores_ctx[:3]]}")
        print(f"  --> {winner}")

    n = len(TRULY_AMBIGUOUS_QUERIES)
    print("\n" + "=" * 70)
    print(f"FINAL SCORES:")
    print(f"  voyage-3 MRR:         {std_mrr/n:.3f}")
    print(f"  voyage-context-3 MRR: {ctx_mrr/n:.3f}")
    print(f"  Improvement: {(ctx_mrr - std_mrr) / max(std_mrr, 0.001) * 100:+.1f}%")
    print(f"\n  Context wins: {context_wins} | Standard wins: {standard_wins} | Ties: {ties}")
    print("=" * 70)


if __name__ == '__main__':
    main()
