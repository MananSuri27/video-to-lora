#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

INPUT_MANIFEST="${INPUT_MANIFEST:-/data/video2lora/raw/manifest.jsonl}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100}"
VAL_SAMPLES="${VAL_SAMPLES:-10}"
SEED="${SEED:-42}"

uv run python scripts/video2lora/build_subset_manifest.py \
  --input-manifest "$INPUT_MANIFEST" \
  --train-out /data/video2lora/processed/train.jsonl \
  --val-out /data/video2lora/processed/val.jsonl \
  --total-samples "$TOTAL_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --seed "$SEED" \
  --require-video-exists
