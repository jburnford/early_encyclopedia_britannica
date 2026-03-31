#!/bin/bash
#SBATCH --job-name=eb_embed_all
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_DIR=~/projects/def-jic823/1815EncyclopediaBritannicaNLS
VENV=~/projects/def-jic823/embed_venv

source "$VENV/bin/activate"

export HF_HOME=~/projects/def-jic823/models/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

cd "$REPO_DIR"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=== Embedding all articles ==="
python3 graphrag/embed_articles.py "$@"

echo ""
echo "=== Done ==="
