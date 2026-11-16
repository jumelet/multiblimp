#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH -p gpu_h100
#SBATCH --gpus=1

# Submit from this directory; all arguments are passed on to eval_model.py, e.g.
#   sbatch job.sh --model swiss-ai/Apertus-8B-2509 --langs nld deu
# sbatch options go before the script name, e.g. sbatch -t 8:00:00 -p gpu_h100 job.sh ...

set -euo pipefail

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

VENV_DIR="$HOME/venvs/babylm"

echo "starting..."

if ! "$VENV_DIR/bin/python" -c "import torch, transformers, lm_eval, pandas, huggingface_hub" 2>/dev/null; then
    echo "creating venv..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    python -m pip install transformers "lm_eval[hf]" pandas pyarrow
else
    source "$VENV_DIR/bin/activate"
fi

# Lets PyTorch reuse fragmented GPU memory: without it, salamandra-2b went OOM with
# 23 GiB reserved but unused, as the auto-detected batch size leaves little headroom.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_model.py "$@"
