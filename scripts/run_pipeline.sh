#!/bin/bash
# Run the embedding and Neo4j loading pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -v '^#' "$PROJECT_DIR/.env" | xargs)
fi

# Default input file
INPUT_FILE="${1:-$PROJECT_DIR/output/chunks_1778.jsonl}"

echo "=============================================="
echo "Encyclopedia Britannica Embedding Pipeline"
echo "=============================================="
echo "Input: $INPUT_FILE"
echo "Neo4j: $NEO4J_URI"
echo ""

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Count chunks
CHUNK_COUNT=$(wc -l < "$INPUT_FILE")
echo "Chunks to process: $CHUNK_COUNT"
echo ""

# Estimate cost (voyage-3 is $0.06/1M tokens, ~300 tokens per chunk average)
TOKENS=$((CHUNK_COUNT * 300))
COST=$(echo "scale=4; $TOKENS * 0.00006 / 1000" | bc)
echo "Estimated embedding cost: ~\$$COST"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Run pipeline
python3 "$SCRIPT_DIR/embed_and_load_neo4j.py" \
    --input "$INPUT_FILE" \
    --test-query "combustion burning fire theory"

echo ""
echo "Pipeline complete!"
echo ""
echo "To search, run:"
echo "  python3 $SCRIPT_DIR/search_encyclopedia.py --interactive"
