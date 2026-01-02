#!/usr/bin/env python3
"""
Embed encyclopedia chunks and load into Neo4j.

Creates:
- EB_Article nodes (headword, edition_year)
- EB_Chunk nodes (text, embedding, section metadata)
- Vector index for similarity search
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

import voyageai
from neo4j import GraphDatabase


# Neo4j connection (from CLAUDE.md)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://206.12.90.118:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "hl;kn258*vcA7492")

# Voyage API
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Embedding settings
EMBEDDING_MODEL = "voyage-3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 128  # voyage-3 supports up to 128 texts per batch


@dataclass
class Chunk:
    """Encyclopedia chunk with metadata."""
    text: str
    index: int
    parent_headword: str
    edition_year: int
    char_start: int
    char_end: int
    section_title: str
    section_index: int


def load_chunks(jsonl_path: str) -> List[Chunk]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(
                text=data['text'],
                index=data['index'],
                parent_headword=data['parent_headword'],
                edition_year=data['edition_year'],
                char_start=data['char_start'],
                char_end=data['char_end'],
                section_title=data.get('section_title', ''),
                section_index=data.get('section_index', 0)
            ))
    return chunks


def embed_chunks(chunks: List[Chunk], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """Embed chunks using voyage-3."""
    client = voyageai.Client(api_key=VOYAGE_API_KEY)

    texts = [c.text for c in chunks]
    embeddings = []

    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]

        # Rate limiting - voyage allows ~300 RPM
        if i > 0:
            time.sleep(0.2)

        result = client.embed(
            texts=batch,
            model=EMBEDDING_MODEL,
            input_type="document"
        )
        embeddings.extend(result.embeddings)

    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    return embeddings


def setup_neo4j_schema(driver):
    """Create constraints and indexes."""
    with driver.session() as session:
        # Constraints
        session.run("""
            CREATE CONSTRAINT eb_article_id IF NOT EXISTS
            FOR (a:EB_Article) REQUIRE (a.headword, a.edition_year) IS UNIQUE
        """)

        session.run("""
            CREATE CONSTRAINT eb_chunk_id IF NOT EXISTS
            FOR (c:EB_Chunk) REQUIRE c.chunk_id IS UNIQUE
        """)

        # Indexes
        session.run("""
            CREATE INDEX eb_article_headword IF NOT EXISTS
            FOR (a:EB_Article) ON (a.headword)
        """)

        session.run("""
            CREATE INDEX eb_chunk_headword IF NOT EXISTS
            FOR (c:EB_Chunk) ON (c.headword)
        """)

        # Vector index for similarity search
        session.run("""
            CREATE VECTOR INDEX eb_chunk_embedding IF NOT EXISTS
            FOR (c:EB_Chunk) ON (c.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }
            }
        """)

        print("Neo4j schema created (constraints, indexes, vector index)")


def load_articles(driver, chunks: List[Chunk]):
    """Create article nodes from chunks."""
    # Get unique articles
    articles = {}
    for c in chunks:
        key = (c.parent_headword, c.edition_year)
        if key not in articles:
            articles[key] = {
                'headword': c.parent_headword,
                'edition_year': c.edition_year,
                'chunk_count': 0
            }
        articles[key]['chunk_count'] += 1

    print(f"Loading {len(articles)} articles into Neo4j...")

    with driver.session() as session:
        for article in tqdm(articles.values(), desc="Articles"):
            session.run("""
                MERGE (a:EB_Article {headword: $headword, edition_year: $edition_year})
                SET a.chunk_count = $chunk_count
            """, article)

    print(f"Created {len(articles)} EB_Article nodes")


def load_chunks_with_embeddings(driver, chunks: List[Chunk], embeddings: List[List[float]], batch_size: int = 100):
    """Load chunks with embeddings into Neo4j."""
    print(f"Loading {len(chunks)} chunks into Neo4j...")

    with driver.session() as session:
        for i in tqdm(range(0, len(chunks), batch_size), desc="Chunks"):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                # Create unique chunk ID
                chunk_id = f"{chunk.parent_headword}_{chunk.edition_year}_{chunk.index}"

                session.run("""
                    MERGE (c:EB_Chunk {chunk_id: $chunk_id})
                    SET c.text = $text,
                        c.headword = $headword,
                        c.edition_year = $edition_year,
                        c.chunk_index = $chunk_index,
                        c.section_title = $section_title,
                        c.section_index = $section_index,
                        c.char_start = $char_start,
                        c.char_end = $char_end,
                        c.embedding = $embedding

                    WITH c
                    MATCH (a:EB_Article {headword: $headword, edition_year: $edition_year})
                    MERGE (a)-[:HAS_CHUNK]->(c)
                """, {
                    'chunk_id': chunk_id,
                    'text': chunk.text,
                    'headword': chunk.parent_headword,
                    'edition_year': chunk.edition_year,
                    'chunk_index': chunk.index,
                    'section_title': chunk.section_title,
                    'section_index': chunk.section_index,
                    'char_start': chunk.char_start,
                    'char_end': chunk.char_end,
                    'embedding': embedding
                })

    print(f"Created {len(chunks)} EB_Chunk nodes with embeddings")


def test_similarity_search(driver, query: str, top_k: int = 5):
    """Test vector similarity search."""
    client = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Embed query
    result = client.embed(
        texts=[query],
        model=EMBEDDING_MODEL,
        input_type="query"
    )
    query_embedding = result.embeddings[0]

    # Search in Neo4j
    with driver.session() as session:
        results = session.run("""
            CALL db.index.vector.queryNodes('eb_chunk_embedding', $top_k, $embedding)
            YIELD node, score
            RETURN node.headword AS headword,
                   node.section_title AS section,
                   left(node.text, 200) AS text_preview,
                   score
            ORDER BY score DESC
        """, {'embedding': query_embedding, 'top_k': top_k})

        print(f"\nQuery: {query}")
        print("-" * 60)
        for record in results:
            print(f"[{record['score']:.3f}] {record['headword']} - {record['section']}")
            print(f"    {record['text_preview']}...")
            print()


def main():
    parser = argparse.ArgumentParser(description='Embed encyclopedia chunks and load to Neo4j')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to chunks JSONL file')
    parser.add_argument('--test-query', type=str, default=None,
                        help='Run a test similarity search after loading')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of chunks to process (for testing)')
    parser.add_argument('--skip-embed', action='store_true',
                        help='Skip embedding (use cached embeddings)')
    parser.add_argument('--embeddings-cache', type=str, default=None,
                        help='Path to cached embeddings JSON')
    args = parser.parse_args()

    # Validate API key
    if not VOYAGE_API_KEY and not args.skip_embed:
        print("Error: VOYAGE_API_KEY environment variable not set")
        print("Set it with: export VOYAGE_API_KEY=your-key")
        sys.exit(1)

    # Load chunks
    print(f"Loading chunks from {args.input}...")
    chunks = load_chunks(args.input)

    if args.limit:
        chunks = chunks[:args.limit]
        print(f"Limited to {len(chunks)} chunks")

    # Generate or load embeddings
    if args.skip_embed and args.embeddings_cache:
        print(f"Loading cached embeddings from {args.embeddings_cache}...")
        with open(args.embeddings_cache, 'r') as f:
            embeddings = json.load(f)
    else:
        embeddings = embed_chunks(chunks)

        # Cache embeddings
        cache_path = args.embeddings_cache or args.input.replace('.jsonl', '_embeddings.json')
        print(f"Caching embeddings to {cache_path}...")
        with open(cache_path, 'w') as f:
            json.dump(embeddings, f)

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Verify connection
        driver.verify_connectivity()
        print("Connected to Neo4j")

        # Setup schema
        setup_neo4j_schema(driver)

        # Load articles
        load_articles(driver, chunks)

        # Load chunks with embeddings
        load_chunks_with_embeddings(driver, chunks, embeddings)

        # Test search if requested
        if args.test_query:
            test_similarity_search(driver, args.test_query)

        # Print summary
        with driver.session() as session:
            result = session.run("""
                MATCH (a:EB_Article) WITH count(a) as articles
                MATCH (c:EB_Chunk) WITH articles, count(c) as chunks
                RETURN articles, chunks
            """)
            record = result.single()
            print(f"\nSummary:")
            print(f"  EB_Article nodes: {record['articles']}")
            print(f"  EB_Chunk nodes: {record['chunks']}")

    finally:
        driver.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
