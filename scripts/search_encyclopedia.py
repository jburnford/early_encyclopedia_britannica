#!/usr/bin/env python3
"""
Search encyclopedia chunks using vector similarity.

Usage:
    python search_encyclopedia.py "your query here"
    python search_encyclopedia.py --interactive
"""

import argparse
import os
import sys

import voyageai
from neo4j import GraphDatabase


# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://206.12.90.118:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "hl;kn258*vcA7492")

# Voyage API
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
EMBEDDING_MODEL = "voyage-3"


class EncyclopediaSearch:
    """Vector search over encyclopedia chunks in Neo4j."""

    def __init__(self):
        self.voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def embed_query(self, query: str) -> list:
        """Embed a query string."""
        result = self.voyage_client.embed(
            texts=[query],
            model=EMBEDDING_MODEL,
            input_type="query"
        )
        return result.embeddings[0]

    def search(self, query: str, top_k: int = 5, edition_year: int = None) -> list:
        """Search for chunks similar to query."""
        query_embedding = self.embed_query(query)

        with self.driver.session() as session:
            # Base vector search
            cypher = """
                CALL db.index.vector.queryNodes('eb_chunk_embedding', $top_k, $embedding)
                YIELD node, score
            """

            # Optional filter by edition
            if edition_year:
                cypher += " WHERE node.edition_year = $edition_year"

            cypher += """
                RETURN node.chunk_id AS id,
                       node.headword AS headword,
                       node.section_title AS section,
                       node.edition_year AS year,
                       node.text AS text,
                       score
                ORDER BY score DESC
            """

            params = {'embedding': query_embedding, 'top_k': top_k}
            if edition_year:
                params['edition_year'] = edition_year

            results = session.run(cypher, params)
            return [dict(record) for record in results]

    def get_article_context(self, headword: str, edition_year: int = 1778) -> list:
        """Get all chunks for an article."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (a:EB_Article {headword: $headword, edition_year: $edition_year})-[:HAS_CHUNK]->(c:EB_Chunk)
                RETURN c.chunk_id AS id,
                       c.section_title AS section,
                       c.chunk_index AS index,
                       c.text AS text
                ORDER BY c.chunk_index
            """, {'headword': headword, 'edition_year': edition_year})
            return [dict(record) for record in results]

    def find_related_articles(self, headword: str, top_k: int = 5) -> list:
        """Find articles with similar content to the given article."""
        # Get the article's chunks
        chunks = self.get_article_context(headword)
        if not chunks:
            return []

        # Combine chunk texts and embed
        combined_text = " ".join(c['text'][:500] for c in chunks[:3])
        query_embedding = self.embed_query(combined_text)

        with self.driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('eb_chunk_embedding', $top_k * 3, $embedding)
                YIELD node, score
                WHERE node.headword <> $headword
                WITH node.headword AS headword, avg(score) AS avg_score
                RETURN headword, avg_score
                ORDER BY avg_score DESC
                LIMIT $top_k
            """, {
                'embedding': query_embedding,
                'top_k': top_k,
                'headword': headword
            })
            return [dict(record) for record in results]


def format_result(result: dict, show_full: bool = False) -> str:
    """Format a search result for display."""
    output = []
    output.append(f"[{result['score']:.3f}] {result['headword']} ({result['year']})")
    output.append(f"        Section: {result['section']}")

    text = result['text']
    if not show_full and len(text) > 300:
        text = text[:300] + "..."
    output.append(f"        {text}")
    output.append("")

    return "\n".join(output)


def interactive_mode(searcher: EncyclopediaSearch):
    """Interactive search mode."""
    print("Encyclopedia Search (type 'quit' to exit, 'help' for commands)")
    print("-" * 60)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == 'quit':
            break

        if query.lower() == 'help':
            print("""
Commands:
  <query>           - Search for chunks matching query
  article:<name>    - Show all chunks for article
  related:<name>    - Find articles related to article
  quit              - Exit interactive mode
            """)
            continue

        # Parse commands
        if query.startswith('article:'):
            headword = query.split(':', 1)[1].strip().upper()
            chunks = searcher.get_article_context(headword)
            if not chunks:
                print(f"No article found: {headword}")
            else:
                print(f"\n{headword} ({len(chunks)} chunks)")
                print("=" * 60)
                for chunk in chunks:
                    print(f"\n[{chunk['index']}] {chunk['section']}")
                    print(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])

        elif query.startswith('related:'):
            headword = query.split(':', 1)[1].strip().upper()
            related = searcher.find_related_articles(headword)
            if not related:
                print(f"No related articles found for: {headword}")
            else:
                print(f"\nArticles related to {headword}:")
                for r in related:
                    print(f"  [{r['avg_score']:.3f}] {r['headword']}")

        else:
            # Regular search
            results = searcher.search(query, top_k=5)
            if not results:
                print("No results found")
            else:
                print(f"\nFound {len(results)} results:")
                print("-" * 60)
                for result in results:
                    print(format_result(result))


def main():
    parser = argparse.ArgumentParser(description='Search encyclopedia chunks')
    parser.add_argument('query', nargs='?', default=None,
                        help='Search query')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive search mode')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of results to return')
    parser.add_argument('--edition', '-e', type=int, default=None,
                        help='Filter by edition year')
    parser.add_argument('--full', '-f', action='store_true',
                        help='Show full text (not truncated)')
    args = parser.parse_args()

    # Validate API key
    if not VOYAGE_API_KEY:
        print("Error: VOYAGE_API_KEY environment variable not set")
        sys.exit(1)

    searcher = EncyclopediaSearch()

    try:
        if args.interactive:
            interactive_mode(searcher)
        elif args.query:
            results = searcher.search(args.query, top_k=args.top_k, edition_year=args.edition)
            if not results:
                print("No results found")
            else:
                print(f"Found {len(results)} results for: {args.query}")
                print("-" * 60)
                for result in results:
                    print(format_result(result, show_full=args.full))
        else:
            parser.print_help()
    finally:
        searcher.close()


if __name__ == '__main__':
    main()
