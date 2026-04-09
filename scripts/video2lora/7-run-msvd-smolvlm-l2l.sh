#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "[gpu] 7-run using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[gpu] 7-run using PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
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
echo "[gpu] 7-run using CUDA_HOME=${CUDA_HOME}"
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
WANDB_PROJECT="${WANDB_PROJECT:-video2lora-video-centric}"
WANDB_MODE="${WANDB_MODE:-auto}"
SMOLVLM_MODEL="${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM2-2.2B-Instruct}"
BASE_LM_MODEL="${BASE_LM_MODEL:-HuggingFaceTB/SmolLM2-1.7B-Instruct}"
VIDEO_FPS="${VIDEO_FPS:-}"
MAX_FRAMES="${MAX_FRAMES:-16}"
QUESTIONS_PER_VIDEO="${QUESTIONS_PER_VIDEO:-4}"
FRAME_POOLING="${FRAME_POOLING:-mean}"
CTX_FEATURE_MODE="${CTX_FEATURE_MODE:-l2l_fused_text}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-1000}"
MAX_STEPS="${MAX_STEPS:-1000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"
LORA_R="${LORA_R:-8}"
TARGET_MODULES="${TARGET_MODULES:-down_proj}"
LATENT_SIZE="${LATENT_SIZE:-512}"
N_LATENT_QUERIES="${N_LATENT_QUERIES:-8}"
NUM_BLOCKS="${NUM_BLOCKS:-9}"
NUM_SELF_ATTN_PER_BLOCK="${NUM_SELF_ATTN_PER_BLOCK:-0}"
KL_WEIGHT="${KL_WEIGHT:-0.0}"
KL_TEMPERATURE="${KL_TEMPERATURE:-2.0}"
WANDB_GROUP="${WANDB_GROUP:-msvd-qa-video-centric-l2l}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-msvd-qa-l2l-qpv${QUESTIONS_PER_VIDEO}-f${MAX_FRAMES}-r${LORA_R}-bs${BATCH_SIZE}}"

TRAIN_MANIFEST="/data/video2lora/processed/msvd-qa-train.jsonl"
VAL_MANIFEST="/data/video2lora/processed/msvd-qa-val.jsonl"

uv run python scripts/video2lora/import_msvd_qa.py \
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
  --target-modules "$TARGET_MODULES"
  --latent-size "$LATENT_SIZE"
  --n-latent-queries "$N_LATENT_QUERIES"
  --num-blocks "$NUM_BLOCKS"
  --num-self-attn-per-block "$NUM_SELF_ATTN_PER_BLOCK"
  --max-frames "$MAX_FRAMES"
  --questions-per-video "$QUESTIONS_PER_VIDEO"
  --frame-pooling "$FRAME_POOLING"
  --ctx-feature-mode "$CTX_FEATURE_MODE"
  --kl-weight "$KL_WEIGHT"
  --kl-temperature "$KL_TEMPERATURE"
  --wandb-project "$WANDB_PROJECT"
  --wandb-mode "$WANDB_MODE"
  --wandb-group "$WANDB_GROUP"
  --wandb-run-name "$WANDB_RUN_NAME"
)

if [[ -n "$VIDEO_FPS" ]]; then
  train_args+=(--video-fps "$VIDEO_FPS")
fi

uv run python scripts/video2lora/train_smolvlm_online.py "${train_args[@]}"
