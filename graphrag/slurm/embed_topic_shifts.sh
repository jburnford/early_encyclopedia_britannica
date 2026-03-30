#!/bin/bash
#SBATCH --job-name=eb_topic_shift
#SBATCH --partition=plato_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-%j.out

# ============================================================================
# Topic shift detection via embeddings
# Run from: ~/projects/def-jic823/1815EncyclopediaBritannicaNLS
#
# Setup (one-time):
#   cd ~/projects/def-jic823
#   git clone git@github.com:jburnford/1815EncyclopediaBritannicaNLS.git
#   python3 -m venv ~/projects/def-jic823/embed_venv
#   source ~/projects/def-jic823/embed_venv/bin/activate
#   pip install sentence-transformers einops scipy numpy
#
# Submit:
#   cd ~/projects/def-jic823/1815EncyclopediaBritannicaNLS
#   sbatch graphrag/slurm/embed_topic_shifts.sh
# ============================================================================

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
echo "=== Running topic shift detection ==="
python3 graphrag/embed_topic_shifts.py "$@"

echo ""
echo "=== Done ==="
