#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"
NVIDIA_SITE_PKGS="$(pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/curand/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvjitlink/lib:$NVIDIA_SITE_PKGS/nvtx/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

INPUT_MANIFEST="${INPUT_MANIFEST:-/data/video2lora/raw/manifest.jsonl}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-10}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-video2lora}"
SMOLVLM_MODEL="${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
BASE_LM_MODEL="${BASE_LM_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
VIDEO_FPS="${VIDEO_FPS:-1.0}"
MAX_FRAMES="${MAX_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

uv run python scripts/video2lora/build_subset_manifest.py \
  --input-manifest "$INPUT_MANIFEST" \
  --train-out /data/video2lora/processed/train.jsonl \
  --val-out /data/video2lora/processed/val.jsonl \
  --total-samples "$TOTAL_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --seed "$SEED" \
  --require-video-exists

uv run python scripts/video2lora/train_smolvlm_online.py \
  --smolvlm-name-or-path "$SMOLVLM_MODEL" \
  --base-lm-name-or-path "$BASE_LM_MODEL" \
  --train-manifest /data/video2lora/processed/train.jsonl \
  --val-manifest /data/video2lora/processed/val.jsonl \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --video-fps "$VIDEO_FPS" \
  --max-frames "$MAX_FRAMES" \
  --wandb-project "$WANDB_PROJECT"
