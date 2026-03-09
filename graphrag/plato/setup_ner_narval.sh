#!/bin/bash
# One-time setup for EarlyModernNER on Narval (DRAC).
# Creates a dedicated venv, installs dependencies, downloads model + adapters.
#
# Usage: bash graphrag/plato/setup_ner_narval.sh

set -euo pipefail

echo "=== EarlyModernNER Setup for Narval ==="

module load python/3.11 cuda/12.6
echo "Modules loaded: python/3.11, cuda/12.6"

VENV_DIR="/project/def-jic823/ner_venv"
export HF_HOME="/project/def-jic823/hf_cache"
mkdir -p "$HF_HOME"

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR"
else
    echo "Creating venv at $VENV_DIR..."
    virtualenv --no-download "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Install PyTorch (use Alliance wheelhouse)
echo ""
echo "=== Installing PyTorch ==="
pip install --no-index torch

# Install earlymodernner + dependencies
echo ""
echo "=== Installing earlymodernner ==="
pip install --no-index bitsandbytes accelerate transformers peft safetensors
pip install earlymodernner

# Pre-download LoRA adapters (~680MB from HuggingFace)
echo ""
echo "=== Downloading LoRA adapters ==="
python -m earlymodernner --download

# Pre-download base model weights (~8GB from HuggingFace)
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
echo "HF cache: $HF_HOME"
echo ""
echo "Test:"
echo "  module load python/3.11 cuda/12.6"
echo "  source $VENV_DIR/bin/activate"
echo "  python -c 'import earlymodernner; print(\"OK\")'"
