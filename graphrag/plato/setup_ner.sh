#!/bin/bash
# One-time setup for EarlyModernNER on Plato cluster.
# Creates a dedicated venv, installs dependencies, downloads model + adapters.
#
# Usage: bash graphrag/plato/setup_ner.sh

set -euo pipefail

echo "=== EarlyModernNER Setup for Plato ==="

# Load required modules
module load gcc/12.3 cuda/12.6 python/3.11
echo "Modules loaded: gcc/12.3, cuda/12.6, python/3.11"

VENV_DIR="/project/clifford/ner_venv"

# HuggingFace cache also on project disk (home is full)
export HF_HOME="/project/clifford/hf_cache"
mkdir -p "$HF_HOME"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR"
else
    echo "Creating venv at $VENV_DIR..."
    python3.11 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Install PyTorch (CUDA 12.4 wheels — compatible with cluster CUDA 12.6)
echo ""
echo "=== Installing PyTorch ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install earlymodernner + dependencies
echo ""
echo "=== Installing earlymodernner ==="
pip install earlymodernner transformers peft bitsandbytes accelerate

# Pre-download LoRA adapters (~680MB from HuggingFace)
echo ""
echo "=== Downloading LoRA adapters ==="
python -m earlymodernner --download

# Pre-download base model weights (~8GB from HuggingFace)
# This avoids download during SLURM jobs where HTTPS may be flaky.
echo ""
echo "=== Downloading base model (Qwen3-4B) ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Instruct-2507')
print('Base model cached.')
"

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV_DIR"
echo ""
echo "Test it:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -c 'import earlymodernner; print(\"OK\")'"
echo "  python -c 'import torch; print(\"CUDA:\", torch.cuda.is_available())'"
