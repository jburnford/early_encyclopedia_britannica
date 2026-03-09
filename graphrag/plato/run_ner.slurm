#!/bin/bash
#SBATCH --job-name=ner-eb
#SBATCH --account=hpc_p_clifford
#SBATCH --gpus-per-node=1g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=ner-%j.out
#
# Run EarlyModernNER on one edition of Encyclopedia Britannica.
#
# Usage:
#   sbatch graphrag/plato/run_ner.slurm 1771
#   sbatch --time=12:00:00 graphrag/plato/run_ner.slurm 1842
#
# For a quick test (100 articles):
#   sbatch --time=1:00:00 graphrag/plato/run_ner.slurm 1771 --max-articles 100

set -euo pipefail

EDITION_YEAR="${1:?Usage: sbatch run_ner.slurm <edition_year> [extra args...]}"
shift  # remaining args passed to run_ner.py

echo "=== EarlyModernNER: ${EDITION_YEAR} edition ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

# Load modules
module load gcc/12.3 cuda/12.6 python/3.11

# Activate venv (on project disk — home quota is full)
source "/project/clifford/ner_venv/bin/activate"
export HF_HOME="/project/clifford/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
echo "Python:  $(which python)"
echo "Torch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"

# GPU info (may not be visible until job starts on node)
nvidia-smi -L 2>/dev/null || echo "GPU: not yet visible"
echo ""

# Find the repo
REPO=""
for dir in \
    "/project/clifford/1815EncyclopediaBritannicaNLS" \
    "$HOME/1815EncyclopediaBritannicaNLS"; do
    if [ -f "$dir/graphrag/run_ner.py" ]; then
        REPO="$dir"
        break
    fi
done

if [ -z "$REPO" ]; then
    echo "Error: Cannot find repo with graphrag/run_ner.py"
    echo "Searched: ~/projects/def-jic823/ and ~/"
    exit 1
fi

echo "Repo: $REPO"
echo ""

# Run NER
cd "$REPO"
python graphrag/run_ner.py \
    --edition-year "$EDITION_YEAR" \
    --batch-size 500 \
    --verbose \
    "$@"

echo ""
echo "Finished: $(date)"
