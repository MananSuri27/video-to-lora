#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export PATH="$CUDA_HOME/bin:$PATH"
NVIDIA_SITE_PKGS="$(pwd)/.venv/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/curand/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvjitlink/lib:$NVIDIA_SITE_PKGS/nvtx/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-$(pwd)/wandb_key}"
if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f "$WANDB_KEY_FILE" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"
VAL_SAMPLES="${VAL_SAMPLES:-0}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-video2lora}"
SMOLVLM_MODEL="${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
BASE_LM_MODEL="${BASE_LM_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
VIDEO_FPS="${VIDEO_FPS:-}"
MAX_FRAMES="${MAX_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
EPOCHS="${EPOCHS:-1000}"
MAX_STEPS="${MAX_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-250}"
NUM_WORKERS="${NUM_WORKERS:-8}"

uv run python scripts/video2lora/import_msvd_qa.py \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --seed "$SEED" \
  --train-out /data/video2lora/processed/train.jsonl \
  --val-out /data/video2lora/processed/val.jsonl

train_args=(
  --smolvlm-name-or-path "$SMOLVLM_MODEL"
  --base-lm-name-or-path "$BASE_LM_MODEL"
  --train-manifest /data/video2lora/processed/train.jsonl
  --val-manifest /data/video2lora/processed/val.jsonl
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --learning-rate "$LEARNING_RATE"
  --warmup-steps "$WARMUP_STEPS"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --max-frames "$MAX_FRAMES"
  --wandb-project "$WANDB_PROJECT"
)

if [[ -n "$VIDEO_FPS" ]]; then
  train_args+=(--video-fps "$VIDEO_FPS")
fi

uv run python scripts/video2lora/train_smolvlm_online.py "${train_args[@]}"
