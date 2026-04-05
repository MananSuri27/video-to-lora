#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
echo "[gpu] 4-run using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[gpu] 4-run using PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
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

DATASET_ROOT="${DATASET_ROOT:-/data/video2lora/raw/MSRVTT-QA}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-0}"
VAL_SAMPLES="${VAL_SAMPLES:-0}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-video2lora}"
SMOLVLM_MODEL="${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
BASE_LM_MODEL="${BASE_LM_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
VIDEO_FPS="${VIDEO_FPS:-}"
MAX_FRAMES="${MAX_FRAMES:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-18}"
EPOCHS="${EPOCHS:-1000}"
MAX_STEPS="${MAX_STEPS:-5000}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
LORA_R="${LORA_R:-16}"
LATENT_SIZE="${LATENT_SIZE:-768}"
N_LATENT_QUERIES="${N_LATENT_QUERIES:-208}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
NUM_SELF_ATTN_PER_BLOCK="${NUM_SELF_ATTN_PER_BLOCK:-1}"
WANDB_GROUP="${WANDB_GROUP:-msrvtt-qa}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-msrvtt-qa-smolvlm-r${LORA_R}-bs${BATCH_SIZE}}"

TRAIN_MANIFEST="/data/video2lora/processed/msrvtt-qa-train.jsonl"
VAL_MANIFEST="/data/video2lora/processed/msrvtt-qa-val.jsonl"

uv run python scripts/video2lora/bootstrap_msrvtt_qa.py \
  --dataset-root "$DATASET_ROOT"

uv run python scripts/video2lora/import_msrvtt_qa.py \
  --dataset-root "$DATASET_ROOT" \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --seed "$SEED" \
  --train-out "$TRAIN_MANIFEST" \
  --val-out "$VAL_MANIFEST"

train_args=(
  --smolvlm-name-or-path "$SMOLVLM_MODEL"
  --base-lm-name-or-path "$BASE_LM_MODEL"
  --train-manifest "$TRAIN_MANIFEST"
  --val-manifest "$VAL_MANIFEST"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --num-workers "$NUM_WORKERS"
  --learning-rate "$LEARNING_RATE"
  --warmup-steps "$WARMUP_STEPS"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --max-val-samples "$MAX_VAL_SAMPLES"
  --lora-r "$LORA_R"
  --latent-size "$LATENT_SIZE"
  --n-latent-queries "$N_LATENT_QUERIES"
  --num-blocks "$NUM_BLOCKS"
  --num-self-attn-per-block "$NUM_SELF_ATTN_PER_BLOCK"
  --max-frames "$MAX_FRAMES"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP"
  --wandb-run-name "$WANDB_RUN_NAME"
)

if [[ -n "$VIDEO_FPS" ]]; then
  train_args+=(--video-fps "$VIDEO_FPS")
fi

uv run python scripts/video2lora/train_smolvlm_online.py "${train_args[@]}"
