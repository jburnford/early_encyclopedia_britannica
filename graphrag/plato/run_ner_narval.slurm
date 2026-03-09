#!/bin/bash
#SBATCH --job-name=ner-eb
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=ner-%j.out
#
# Run EarlyModernNER on one edition of Encyclopedia Britannica.
# Narval cluster (DRAC) — A100 40GB GPUs.
#
# Usage:
#   sbatch graphrag/plato/run_ner_narval.slurm 1771
#   sbatch graphrag/plato/run_ner_narval.slurm 1771 --max-articles 100

set -euo pipefail

EDITION_YEAR="${1:?Usage: sbatch run_ner_narval.slurm <edition_year> [extra args...]}"
shift

echo "=== EarlyModernNER: ${EDITION_YEAR} edition ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

module load python/3.11 cuda/12.6

source "/project/def-jic823/ner_venv/bin/activate"
export HF_HOME="/project/def-jic823/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi -L 2>/dev/null || echo "GPU: not visible"
echo ""

REPO="/project/def-jic823/1815EncyclopediaBritannicaNLS"
if [ ! -f "$REPO/graphrag/run_ner.py" ]; then
    echo "Error: Cannot find $REPO/graphrag/run_ner.py"
    exit 1
fi

echo "Repo: $REPO"
echo ""

cd "$REPO"
python graphrag/run_ner.py \
    --edition-year "$EDITION_YEAR" \
    --batch-size 500 \
    --verbose \
    "$@"

echo ""
echo "Finished: $(date)"
