# Prototype Args

Example `uv` command:

```bash
uv run python scripts/video2lora/train.py \
  --model-name-or-path google/gemma-2-2b-it \
  --ctx-feature-size 2048 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --latent-size 512 \
  --n-latent-queries 128 \
  --num-blocks 4 \
  --wandb-project video2lora
```

Use this as a starting point while we decide on the exact VLM feature extractor.

For the current preferred path, use SmolVLM online extraction instead via:

```bash
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
```
