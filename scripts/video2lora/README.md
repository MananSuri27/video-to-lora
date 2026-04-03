# Video2LoRA Prototype Training

This folder now contains two prototype paths:

- `import_msvd_qa.py`
  - downloads a small MSVD-QA subset, writes local `.mp4` files, and creates train/val manifests
- `0-import-msvd-qa.sh`
  - shell wrapper for importing a tiny MSVD-QA subset
- `build_subset_manifest.py`
  - build a small train/val subset manifest from a larger raw manifest
- `0-build-subset.sh`
  - shell wrapper for the subset builder
- `train.py`
  - precomputed feature path
- `train_smolvlm_online.py`
  - online SmolVLM path
- `2-run-smolvlm-prototype.sh`
  - one-command subset-build + train runner
- `3-run-msvd-smolvlm.sh`
  - one-command MSVD-QA import + train runner

The online path is the preferred starting point because it matches the current D2L design more closely:

- load videos online inside the training step
- extract LM-aligned video states from SmolVLM
- feed them into the perceiver + hypernet
- generate LoRA for the frozen text decoder
- train on video QA

## Data Layout

The script creates and uses these directories:

- `/data/video2lora/raw`
- `/data/video2lora/processed`
- `/data/video2lora/features`
- `/data/video2lora/runs`
- `/data/video2lora/cache`

## Manifest Format

Training data lives in JSONL files such as:

- `/data/video2lora/processed/train.jsonl`
- `/data/video2lora/processed/val.jsonl`

Each line should look like:

```json
{
  "id": "sample-0001",
  "video_path": "raw/sample-0001.mp4",
  "question": "What does the person pick up from the table?",
  "answer": "A red cup."
}
```

`video_path` may be absolute or relative to `/data/video2lora`.

## Build A 100-Sample Subset

Place a source manifest at `/data/video2lora/raw/manifest.jsonl` with rows like:

```json
{"id":"sample-0001","video_path":"raw/sample-0001.mp4","question":"What does the person pick up?","answer":"A red cup."}
```

Then run:

```bash
bash scripts/video2lora/0-build-subset.sh
```

This writes:

- `/data/video2lora/processed/train.jsonl`
- `/data/video2lora/processed/val.jsonl`

## Preferred Script

Run the online SmolVLM variant first:

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

This script extracts video states online using SmolVLM's projected multimodal hidden states and feeds them directly into the D2L hypernet.

## One-Command Run

If your raw manifest is already at `/data/video2lora/raw/manifest.jsonl`, you can run the whole tiny prototype with:

```bash
bash scripts/video2lora/2-run-smolvlm-prototype.sh
```

Useful overrides:

```bash
INPUT_MANIFEST=/data/video2lora/raw/my_dataset.jsonl TOTAL_SAMPLES=100 VAL_SAMPLES=10 bash scripts/video2lora/2-run-smolvlm-prototype.sh
```

## MSVD-QA Quick Start

For the first real run, we now support importing a tiny MSVD-QA subset directly from Hugging Face.

Import only:

```bash
bash scripts/video2lora/0-import-msvd-qa.sh
```

Import and train:

```bash
bash scripts/video2lora/3-run-msvd-smolvlm.sh
```

Useful overrides:

```bash
TRAIN_SAMPLES=100 VAL_SAMPLES=20 WANDB_PROJECT=video2lora bash scripts/video2lora/3-run-msvd-smolvlm.sh
```

## Loss

The current SmolVLM Video2LoRA path uses standard supervised fine-tuning loss.

That is the right default for MSVD-QA:

- supervised QA labels already exist
- we do not need teacher log-prob distributions to get the first prototype running
- KL can be added later as an ablation or distillation variant

## Runtime Notes

SmolVLM2's official path expects a few extra packages in the environment:

- `tokenizers`
- `num2words`
- `decord`

These have been added to `pyproject.toml` so `uv` can install them for the project.

## Precomputed Fallback

The precomputed path is still available if we later want to cache video states:

```bash
uv run python scripts/video2lora/train.py \
  --model-name-or-path google/gemma-2-2b-it \
  --ctx-feature-size 2048 \
  --train-manifest /data/video2lora/processed/train.jsonl \
  --val-manifest /data/video2lora/processed/val.jsonl \
  --wandb-project video2lora
```
