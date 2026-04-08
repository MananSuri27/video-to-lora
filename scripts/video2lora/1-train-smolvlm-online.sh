#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME="$(readlink -f /usr/local/cuda)"
  elif command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    CUDA_HOME="/usr/local/cuda"
  fi
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "[gpu] 1-run using CUDA_HOME=${CUDA_HOME}"
NVIDIA_SITE_PKGS="$(pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/curand/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvjitlink/lib:$NVIDIA_SITE_PKGS/nvtx/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

uv run python scripts/video2lora/train_smolvlm_online.py \
  --smolvlm-name-or-path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --base-lm-name-or-path HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --train-manifest /data/video2lora/processed/train.jsonl \
  --val-manifest /data/video2lora/processed/val.jsonl \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --video-fps 1.0 \
  --max-frames 16 \
  --wandb-project video2lora
