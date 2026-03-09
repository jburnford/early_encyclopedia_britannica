#!/bin/bash
# Submit NER jobs for all editions and volumes on Narval.
#
# Each volume gets its own job → maximum parallelism.
# Results are per-volume JSONL files that can be concatenated later.
#
# Usage:
#   bash graphrag/plato/submit_all_narval.sh          # submit everything
#   bash graphrag/plato/submit_all_narval.sh --dry-run # show what would be submitted
#
# Prerequisites:
#   1. Delete old checkpoints first:
#      rm /project/def-jic823/1815EncyclopediaBritannicaNLS/data/ner/.checkpoint_*.json
#   2. git pull to get latest run_ner.py with --volume support

set -euo pipefail

REPO="/project/def-jic823/1815EncyclopediaBritannicaNLS"
SLURM_SCRIPT="$REPO/graphrag/plato/run_ner_narval.slurm"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Activate venv to run list_volumes
module load python/3.11 2>/dev/null || true
source "/project/def-jic823/ner_venv/bin/activate" 2>/dev/null || true
export HF_HOME="/project/def-jic823/hf_cache"

cd "$REPO"

total=0
for year in 1771 1778 1797 1810 1815 1823 1842 1860; do
    # Get volume list for this edition
    volumes=$(python graphrag/run_ner.py --edition-year "$year" --list-volumes 2>/dev/null \
        | grep -oP '\d+' | tr '\n' ' ')

    if [[ -z "$volumes" ]]; then
        echo "WARNING: No volumes found for $year, submitting whole-edition job"
        if $DRY_RUN; then
            echo "  [dry-run] sbatch $SLURM_SCRIPT $year"
        else
            sbatch "$SLURM_SCRIPT" "$year"
        fi
        total=$((total + 1))
        continue
    fi

    for vol in $volumes; do
        if $DRY_RUN; then
            echo "  [dry-run] sbatch $SLURM_SCRIPT $year --volume $vol"
        else
            jobid=$(sbatch --parsable "$SLURM_SCRIPT" "$year" --volume "$vol")
            echo "  Submitted $year v$vol → job $jobid"
        fi
        total=$((total + 1))
    done
done

echo ""
echo "Total jobs: $total"
