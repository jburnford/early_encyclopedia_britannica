#!/usr/bin/env python3
"""
Test voyage-context-3 on historical encyclopedia articles.

Focuses on:
1. Leather/tanning domain (user interest)
2. Semantic drift cases (historical vs modern meanings)
3. Hierarchical embeddings (section + chunk level)

Usage:
    python test_voyage_context_v2.py articles_1815_clean.jsonl --domain leather
    python test_voyage_context_v2.py articles_1771.jsonl --domain semantic_drift
    python test_voyage_context_v2.py articles_1815_clean.jsonl --domain all --hierarchical
"""

import json
import os
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY not found in .env file")

import voyageai

# =============================================================================
# TEST DOMAINS - Articles to load for each test domain
# =============================================================================

LEATHER_DOMAIN = {
    'headwords': [
        'TANNING', 'LEATHER', 'HIDE', 'BARK', 'TAN', 'TANNER',
        'FULLER', 'CURRIER', 'CORDWAINER', 'FELLMONGER',
        'CHAMOIS', 'MOROCCO', 'VELLUM', 'PARCHMENT',
        'OAK', 'SUMACH', 'ALUM', 'LIME',
    ],
    'queries': [
        # Process queries
        ("vegetable tanning oak bark process pit",
         ['TANNING', 'BARK', 'OAK', 'LEATHER']),
        ("hide preparation lime soaking unhairing",
         ['HIDE', 'TANNING', 'LEATHER', 'LIME']),
        ("currying finishing leather oil grease",
         ['LEATHER', 'TANNING', 'CURRIER']),
        ("morocco red leather Turkey dyeing goatskin",
         ['LEATHER', 'MOROCCO', 'TANNING']),
        ("tanning liquor ooze tan yard handler",
         ['TANNING', 'TAN', 'BARK']),

        # Material queries
        ("raw hides salted green dried",
         ['HIDE', 'LEATHER', 'TANNING']),
        ("sole leather butts backs thick",
         ['TANNING', 'LEATHER']),
        ("chamois soft white leather oil tawed",
         ['CHAMOIS', 'LEATHER']),
        ("parchment vellum animal skin writing",
         ['PARCHMENT', 'VELLUM', 'LEATHER']),

        # Trade queries
        ("cordwainer shoemaker leather craft",
         ['CORDWAINER', 'LEATHER']),
        ("fuller cloth wool process",
         ['FULLER']),
    ],
}

SEMANTIC_DRIFT_DOMAIN = {
    'headwords': [
        # Major treatises (legitimate large articles - will be chunked)
        'CHEMISTRY', 'ASTRONOMY', 'ELECTRICITY', 'MEDICINE',
        # Obsolete science
        'PHLOGISTON', 'CALORIC', 'AETHER', 'ETHER',
        # Changed meanings
        'ENTHUSIASM', 'NICE', 'AWFUL', 'BROADCAST',
        'ATOM', 'ENGINE', 'MANUFACTURE', 'MECHANIC',
        'PHYSIC', 'CONSUMPTION', 'VAPOURS', 'HUMOUR',
        # Historical trades
        'FULLER', 'CHANDLER', 'APOTHECARY',
        # Period science
        'MAGNETISM', 'OPTICS',
        'NATURAL PHILOSOPHY', 'EXPERIMENTAL PHILOSOPHY',
    ],
    'queries': [
        # Major treatise queries - test chunking of large articles
        ("phlogiston combustion theory dephlogisticated air",
         ['CHEMISTRY', 'PHLOGISTON']),
        ("acids alkalis salts chemical affinity",
         ['CHEMISTRY']),
        ("planetary motion elliptical orbits Kepler Newton",
         ['ASTRONOMY']),
        ("electrical fluid Franklin Leyden jar spark",
         ['ELECTRICITY']),

        # Obsolete science - should find period articles
        ("combustion burning fire theory substance",
         ['PHLOGISTON', 'FIRE', 'CHEMISTRY']),
        ("heat fluid caloric warmth transfer",
         ['CALORIC', 'HEAT']),
        ("light medium transmission ether space",
         ['AETHER', 'ETHER', 'LIGHT', 'OPTICS']),

        # Semantic drift - historical meanings
        ("scattering seeds sowing field agriculture hand",
         ['BROADCAST', 'AGRICULTURE', 'HUSBANDRY']),
        ("religious zeal fanaticism sect inspiration",
         ['ENTHUSIASM']),
        ("indivisible matter particle smallest unit",
         ['ATOM', 'MATTER']),
        ("machine device mechanical contrivance apparatus",
         ['ENGINE', 'MECHANIC', 'MACHINE']),
        ("making by hand craft artisan workshop",
         ['MANUFACTURE', 'MECHANIC']),

        # Medical drift
        ("wasting disease tubercular lung decline",
         ['CONSUMPTION', 'MEDICINE']),
        ("nervous disorder melancholy depression spirits",
         ['VAPOURS', 'MEDICINE', 'PHYSIC']),
        ("bodily fluids temperament blood bile phlegm",
         ['HUMOUR', 'MEDICINE', 'PHYSIC']),
        ("medicine physic treatment cure remedy",
         ['PHYSIC', 'MEDICINE', 'APOTHECARY']),

        # Period electricity (pre-modern understanding)
        ("electrical fluid spark leyden jar attraction",
         ['ELECTRICITY', 'MAGNETISM']),
    ],
}

ALL_DOMAIN = {
    'headwords': list(set(LEATHER_DOMAIN['headwords'] + SEMANTIC_DRIFT_DOMAIN['headwords'])),
    'queries': LEATHER_DOMAIN['queries'] + SEMANTIC_DRIFT_DOMAIN['queries'],
}

DOMAINS = {
    'leather': LEATHER_DOMAIN,
    'semantic_drift': SEMANTIC_DRIFT_DOMAIN,
    'all': ALL_DOMAIN,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Article:
    headword: str
    text: str
    edition_year: int
    volume: int = 0
    word_count: int = 0
    article_id: str = ""  # Unique ID for handling duplicate headwords

    def __post_init__(self):
        self.word_count = len(self.text.split())
        if not self.article_id:
            # Create unique ID from headword + text hash
            self.article_id = f"{self.headword}_{hash(self.text[:100]) % 10000}"


@dataclass
class Section:
    """Section-level chunk (coarse granularity)"""
    headword: str
    article_id: str  # Unique ID for handling duplicate headwords
    section_id: int
    text: str
    char_start: int = 0
    char_end: int = 0


@dataclass
class Chunk:
    """Chunk-level unit (fine granularity)"""
    headword: str
    article_id: str  # Unique ID for handling duplicate headwords
    section_id: int
    chunk_id: int
    text: str
    embedding: Optional[np.ndarray] = None


# =============================================================================
# ARTICLE LOADING
# =============================================================================

# Known legitimate large treatises (not parsing errors)
LEGITIMATE_TREATISES = {
    'CHEMISTRY', 'ASTRONOMY', 'MEDICINE', 'LAW', 'SURGERY',
    'ANATOMY', 'MORAL PHILOSOPHY', 'ALGEBRA', 'GEOMETRY',
    'ELECTRICITY', 'MAGNETISM', 'OPTICS', 'FARRIERY',
}


def load_articles_by_headword(jsonl_path: str, target_headwords: List[str],
                               max_word_count: int = 50000,
                               treatise_sample_words: int = 15000) -> List[Article]:
    """
    Load specific articles by headword.

    For known treatises, samples first N words to keep test manageable.
    Filters out excessively long articles that aren't known treatises.
    """
    target_set = {hw.upper() for hw in target_headwords}
    articles = []

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            hw = data['headword'].upper()

            if hw in target_set:
                text = data['text']
                word_count = len(text.split())

                # Handle large articles
                if word_count > max_word_count:
                    if hw in LEGITIMATE_TREATISES:
                        # Sample from treatise (first portion)
                        words = text.split()
                        sampled_text = ' '.join(words[:treatise_sample_words])
                        print(f"  Sampling {hw}: {word_count:,} words -> {treatise_sample_words:,} words")
                        text = sampled_text
                    else:
                        print(f"  Skipping {hw} ({word_count:,} words - likely parsing error)")
                        continue

                articles.append(Article(
                    headword=data['headword'],
                    text=text,
                    edition_year=data.get('edition_year', 0),
                    volume=data.get('volume', 0),
                ))

    return articles


def load_background_articles(jsonl_path: str, exclude_headwords: List[str],
                              sample_size: int = 50, max_word_count: int = 5000) -> List[Article]:
    """
    Load random background articles for retrieval testing.
    These should NOT match test queries.
    """
    exclude_set = {hw.upper() for hw in exclude_headwords}
    candidates = []

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            hw = data['headword'].upper()

            if hw not in exclude_set:
                word_count = len(data['text'].split())
                if 50 < word_count < max_word_count:  # Medium-sized articles
                    candidates.append(Article(
                        headword=data['headword'],
                        text=data['text'],
                        edition_year=data.get('edition_year', 0),
                        volume=data.get('volume', 0),
                    ))

    # Sample evenly across alphabet
    candidates.sort(key=lambda a: a.headword)
    step = max(1, len(candidates) // sample_size)
    return [candidates[i] for i in range(0, len(candidates), step)][:sample_size]


# =============================================================================
# CHUNKING - Hierarchical (Section + Chunk)
# =============================================================================

def create_sections(article: Article, target_size: int = 2000) -> List[Section]:
    """
    Create section-level chunks from article.
    Respects paragraph boundaries.
    """
    text = article.text
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if not paragraphs:
        return [Section(
            headword=article.headword,
            article_id=article.article_id,
            section_id=0,
            text=text,
        )]

    sections = []
    current_text = ''
    section_id = 0

    for para in paragraphs:
        if len(current_text) + len(para) + 2 > target_size and current_text:
            sections.append(Section(
                headword=article.headword,
                article_id=article.article_id,
                section_id=section_id,
                text=current_text.strip(),
            ))
            section_id += 1
            current_text = para
        else:
            current_text = (current_text + '\n\n' + para) if current_text else para

    if current_text:
        sections.append(Section(
            headword=article.headword,
            article_id=article.article_id,
            section_id=section_id,
            text=current_text.strip(),
        ))

    return sections


def create_chunks(section: Section, chunk_size: int = 800) -> List[Chunk]:
    """
    Create chunk-level units from a section.
    Uses simple character-based splitting with sentence awareness.
    """
    text = section.text

    if len(text) <= chunk_size:
        return [Chunk(
            headword=section.headword,
            article_id=section.article_id,
            section_id=section.section_id,
            chunk_id=0,
            text=text,
        )]

    # Split on sentence boundaries (approximate)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_text = ''
    chunk_id = 0

    for sent in sentences:
        if len(current_text) + len(sent) + 1 > chunk_size and current_text:
            chunks.append(Chunk(
                headword=section.headword,
                article_id=section.article_id,
                section_id=section.section_id,
                chunk_id=chunk_id,
                text=current_text.strip(),
            ))
            chunk_id += 1
            current_text = sent
        else:
            current_text = (current_text + ' ' + sent) if current_text else sent

    if current_text:
        chunks.append(Chunk(
            headword=section.headword,
            article_id=section.article_id,
            section_id=section.section_id,
            chunk_id=chunk_id,
            text=current_text.strip(),
        ))

    return chunks


def create_flat_chunks(article: Article, chunk_size: int = 1000) -> List[Chunk]:
    """
    Create flat chunks (no section hierarchy).
    For comparison with hierarchical approach.
    """
    sections = create_sections(article, target_size=chunk_size * 2)
    all_chunks = []

    for section in sections:
        chunks = create_chunks(section, chunk_size=chunk_size)
        all_chunks.extend(chunks)

    return all_chunks


# =============================================================================
# VOYAGE API - Embedding
# =============================================================================

class VoyageEmbedder:
    """
    Wrapper for Voyage AI embedding API with rate limiting.

    Rate limits by tier:
    - Free: 3 RPM, 10K TPM
    - Tier 1 (paid): 2,000 RPM, 8M TPM
    - Tier 2: 4,000 RPM, 16M TPM
    """

    def __init__(self, api_key: str, paid_tier: bool = True):
        self.client = voyageai.Client(api_key=api_key)
        self.last_request = 0
        # Paid tier has 2000 RPM = no practical limit for our use
        # Free tier has 3 RPM = 20s between requests
        self.min_interval = 0.1 if paid_tier else 21

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            wait = self.min_interval - elapsed
            if wait > 1:  # Only print if actually waiting
                print(f"    Rate limiting: waiting {wait:.1f}s...")
            time.sleep(wait)
        self.last_request = time.time()

    def embed_standard(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        """Embed using voyage-3 (standard, non-contextual)."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            self._rate_limit()

            try:
                result = self.client.embed(batch, model="voyage-3", input_type="document")
                all_embeddings.extend(result.embeddings)
                print(f"    voyage-3: {min(i+batch_size, len(texts))}/{len(texts)} chunks")
            except Exception as e:
                if "rate" in str(e).lower():
                    print(f"    Rate limit hit, waiting 60s...")
                    time.sleep(60)
                    result = self.client.embed(batch, model="voyage-3", input_type="document")
                    all_embeddings.extend(result.embeddings)
                else:
                    raise

        return np.array(all_embeddings)

    def embed_contextual(self, articles_chunks: Dict[str, List[str]],
                          batch_size: int = 20) -> Dict[str, np.ndarray]:
        """
        Embed using voyage-context-3 (contextual).
        Groups chunks by article for shared context.

        Args:
            articles_chunks: {headword: [chunk1, chunk2, ...]}

        Returns:
            {headword: embeddings_array}
        """
        headwords = list(articles_chunks.keys())
        all_results = {}

        for i in range(0, len(headwords), batch_size):
            batch_headwords = headwords[i:i+batch_size]
            # Build nested input: [[article1_chunks], [article2_chunks], ...]
            nested_inputs = [articles_chunks[hw] for hw in batch_headwords]

            self._rate_limit()

            try:
                result = self.client.contextualized_embed(
                    inputs=nested_inputs,
                    model="voyage-context-3",
                    input_type="document",
                    output_dimension=1024,
                )

                # Map results back to headwords
                for j, hw in enumerate(batch_headwords):
                    all_results[hw] = np.array(result.results[j].embeddings)

                print(f"    voyage-context-3: {min(i+batch_size, len(headwords))}/{len(headwords)} articles")

            except Exception as e:
                if "rate" in str(e).lower():
                    print(f"    Rate limit hit, waiting 60s...")
                    time.sleep(60)
                    result = self.client.contextualized_embed(
                        inputs=nested_inputs,
                        model="voyage-context-3",
                        input_type="document",
                        output_dimension=1024,
                    )
                    for j, hw in enumerate(batch_headwords):
                        all_results[hw] = np.array(result.results[j].embeddings)
                else:
                    raise

        return all_results

    def embed_query(self, query: str, model: str = "voyage-3") -> np.ndarray:
        """Embed a query for retrieval."""
        self._rate_limit()

        if model == "voyage-context-3":
            # For voyage-context-3, use contextualized_embed with single-element list
            result = self.client.contextualized_embed(
                inputs=[[query]],  # Single query as single-element list
                model="voyage-context-3",
                input_type="query",
                output_dimension=1024,
            )
            return np.array(result.results[0].embeddings[0])
        else:
            result = self.client.embed([query], model=model, input_type="query")
            return np.array(result.embeddings[0])


# =============================================================================
# EVALUATION
# =============================================================================

def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and documents."""
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def evaluate_retrieval(chunks: List[Chunk], embeddings: np.ndarray,
                        queries: List[Tuple[str, List[str]]],
                        embedder: VoyageEmbedder,
                        model_name: str,
                        query_model: str = "voyage-3") -> Dict:
    """
    Evaluate retrieval quality on test queries.

    Returns metrics: MRR, Recall@5, Recall@10, per-query results
    """
    results = {
        'model': model_name,
        'queries': [],
        'mrr': 0.0,
        'recall_at_5': 0.0,
        'recall_at_10': 0.0,
    }

    for query_text, expected_headwords in queries:
        query_emb = embedder.embed_query(query_text, model=query_model)
        similarities = cosine_similarity(query_emb, embeddings)

        # Get top-10 results
        top_indices = np.argsort(similarities)[::-1][:10]
        top_chunks = [chunks[idx] for idx in top_indices]
        top_scores = [similarities[idx] for idx in top_indices]
        retrieved_headwords = [c.headword.upper() for c in top_chunks]

        # Find first relevant result
        expected_upper = [hw.upper() for hw in expected_headwords]
        first_relevant_rank = None
        for rank, hw in enumerate(retrieved_headwords, 1):
            if hw in expected_upper:
                first_relevant_rank = rank
                break

        # Calculate recall
        recall_5 = len(set(retrieved_headwords[:5]) & set(expected_upper)) / len(expected_upper)
        recall_10 = len(set(retrieved_headwords[:10]) & set(expected_upper)) / len(expected_upper)

        results['queries'].append({
            'query': query_text,
            'expected': expected_headwords,
            'retrieved': retrieved_headwords[:5],
            'scores': [f"{s:.3f}" for s in top_scores[:5]],
            'first_relevant_rank': first_relevant_rank,
            'recall_5': recall_5,
        })

        if first_relevant_rank:
            results['mrr'] += 1.0 / first_relevant_rank
        results['recall_at_5'] += recall_5
        results['recall_at_10'] += recall_10

    n = len(queries)
    results['mrr'] /= n
    results['recall_at_5'] /= n
    results['recall_at_10'] /= n

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test voyage-context-3 on historical encyclopedias')
    parser.add_argument('jsonl_file', help='Articles JSONL file')
    parser.add_argument('--domain', choices=['leather', 'semantic_drift', 'all'],
                        default='leather', help='Test domain')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Use hierarchical chunking (section + chunk)')
    parser.add_argument('--background-size', type=int, default=50,
                        help='Number of background articles for noise')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file (default: auto-generated)')
    args = parser.parse_args()

    domain = DOMAINS[args.domain]

    print(f"=== voyage-context-3 Test: {args.domain} domain ===\n")

    # Load target articles
    print("Loading target articles...")
    target_articles = load_articles_by_headword(
        args.jsonl_file,
        domain['headwords'],
        max_word_count=20000,  # Skip parsing errors
    )
    print(f"  Loaded {len(target_articles)} target articles")

    for a in sorted(target_articles, key=lambda x: -x.word_count)[:10]:
        print(f"    {a.headword}: {a.word_count:,} words")

    # Load background articles (for realistic retrieval testing)
    print("\nLoading background articles...")
    background_articles = load_background_articles(
        args.jsonl_file,
        exclude_headwords=domain['headwords'],
        sample_size=args.background_size,
    )
    print(f"  Loaded {len(background_articles)} background articles")

    all_articles = target_articles + background_articles

    # Create chunks
    print("\nCreating chunks...")
    all_chunks: List[Chunk] = []
    articles_chunks: Dict[str, List[str]] = {}  # For contextual embedding (keyed by article_id)

    for article in all_articles:
        if args.hierarchical:
            sections = create_sections(article)
            article_chunks = []
            for section in sections:
                article_chunks.extend(create_chunks(section))
        else:
            article_chunks = create_flat_chunks(article)

        all_chunks.extend(article_chunks)
        # Use article_id as key to handle duplicate headwords (e.g., BARK x3)
        articles_chunks[article.article_id] = [c.text for c in article_chunks]

    print(f"  Created {len(all_chunks)} chunks from {len(all_articles)} articles")

    # Initialize embedder
    embedder = VoyageEmbedder(VOYAGE_API_KEY)

    # === Test 1: voyage-3 (standard) ===
    print("\n=== Testing voyage-3 (standard) ===")
    chunk_texts = [c.text for c in all_chunks]

    start = time.time()
    emb_standard = embedder.embed_standard(chunk_texts)
    time_standard = time.time() - start
    print(f"  Embedded in {time_standard:.1f}s")

    results_standard = evaluate_retrieval(
        all_chunks, emb_standard, domain['queries'], embedder, "voyage-3"
    )

    # === Test 2: voyage-context-3 (contextual) ===
    print("\n=== Testing voyage-context-3 (contextual) ===")

    start = time.time()
    emb_contextual_dict = embedder.embed_contextual(articles_chunks)
    time_contextual = time.time() - start
    print(f"  Embedded in {time_contextual:.1f}s")

    # Flatten contextual embeddings to match chunk order
    emb_contextual = []
    for chunk in all_chunks:
        article_id = chunk.article_id
        # Find chunk index within this article
        article_chunk_texts = articles_chunks[article_id]
        chunk_idx = article_chunk_texts.index(chunk.text)
        emb_contextual.append(emb_contextual_dict[article_id][chunk_idx])
    emb_contextual = np.array(emb_contextual)

    # Use voyage-context-3 for queries to match document embedding space
    results_contextual = evaluate_retrieval(
        all_chunks, emb_contextual, domain['queries'], embedder, "voyage-context-3",
        query_model="voyage-context-3"
    )

    # === Print Results ===
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<25} {'MRR':>10} {'Recall@5':>10} {'Recall@10':>10} {'Time':>10}")
    print("-" * 65)
    print(f"{'voyage-3 (standard)':<25} {results_standard['mrr']:>10.3f} "
          f"{results_standard['recall_at_5']:>10.3f} {results_standard['recall_at_10']:>10.3f} "
          f"{time_standard:>9.1f}s")
    print(f"{'voyage-context-3':<25} {results_contextual['mrr']:>10.3f} "
          f"{results_contextual['recall_at_5']:>10.3f} {results_contextual['recall_at_10']:>10.3f} "
          f"{time_contextual:>9.1f}s")

    # Improvement
    mrr_improvement = (results_contextual['mrr'] - results_standard['mrr']) / max(results_standard['mrr'], 0.001) * 100
    print(f"\nContext improvement: {mrr_improvement:+.1f}% MRR")

    # Per-query comparison
    print("\n=== Query-by-Query Analysis ===")
    for q_std, q_ctx in zip(results_standard['queries'], results_contextual['queries']):
        std_rank = q_std['first_relevant_rank']
        ctx_rank = q_ctx['first_relevant_rank']

        status = ""
        if ctx_rank and (not std_rank or ctx_rank < std_rank):
            status = "CONTEXT WINS"
        elif std_rank and (not ctx_rank or std_rank < ctx_rank):
            status = "STANDARD WINS"
        elif std_rank == ctx_rank:
            status = "TIE"
        else:
            status = "BOTH FAILED"

        print(f"\n{q_std['query'][:50]}...")
        print(f"  Expected: {q_std['expected'][:3]}")
        print(f"  Standard: rank={std_rank}, got {q_std['retrieved'][:3]}")
        print(f"  Context:  rank={ctx_rank}, got {q_ctx['retrieved'][:3]}")
        print(f"  Result: {status}")

    # Save results
    output_file = args.output or f"{Path(args.jsonl_file).stem}_{args.domain}_results.json"
    all_results = {
        'domain': args.domain,
        'hierarchical': args.hierarchical,
        'num_target_articles': len(target_articles),
        'num_background_articles': len(background_articles),
        'num_chunks': len(all_chunks),
        'time_standard': time_standard,
        'time_contextual': time_contextual,
        'voyage-3': results_standard,
        'voyage-context-3': results_contextual,
        'mrr_improvement_pct': mrr_improvement,
    }

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
