#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"
NVIDIA_SITE_PKGS="$(pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/curand/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvjitlink/lib:$NVIDIA_SITE_PKGS/nvtx/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-20}"
SEED="${SEED:-42}"

uv run python scripts/video2lora/import_msvd_qa.py \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --seed "$SEED" \
  --train-out /data/video2lora/processed/train.jsonl \
  --val-out /data/video2lora/processed/val.jsonl
