# Video2LoRA First-Attempt Assumptions

This note captures the assumptions, simplifications, and known discounts in the current `Video2LoRA` prototype. The goal of this first attempt is not to finalize the architecture. The goal is to get one real end-to-end training path running and learn where the true integration risks are.

## Prototype Goal

The current prototype is trying to validate one core claim:

- online video encoding can produce LM-aligned context features
- those features can be fed into the Doc-to-LoRA hypernet/perceiver path
- the hypernet can generate LoRA weights for a small language model
- the resulting model can be trained on a small video-QA task end to end

This is a feasibility pass, not a final training system.

## Main Assumptions

- We use `HuggingFaceTB/SmolVLM2-2.2B-Instruct` as the online video feature extractor.
- We use `HuggingFaceTB/SmolLM2-1.7B-Instruct` as the LoRA target language model.
- We assume SmolVLM's returned LM-aligned visual states are a reasonable input representation for the D2L hypernet.
- We assume a small supervised video-QA dataset is enough to validate the mechanism before moving to harder tasks.
- We assume keeping the video encoder frozen is acceptable for the first pass.
- We assume keeping the base LM frozen and training only the hypernet-side adaptation is the right initial setup.

## Dataset Assumptions

- First dataset: `MSVD-QA`.
- First scale: a very small subset, currently `100` train examples and `20` validation examples.
- First task: supervised question answering over videos.
- Training objective: standard SFT / cross-entropy, not KL distillation.

Why:

- `MSVD-QA` is easy to operationalize quickly.
- The task is simple enough to expose plumbing issues early.
- SFT is the most direct objective for this dataset and easier to debug than a teacher-based KL setup.

## Deliberate Simplifications

### 1. Single-GPU First

- The machine has `8x A100 80GB`, but the current script is still a single-GPU training path.
- DeepSpeed remains installed and importable, but we are not yet launching distributed training.

Reason:

- The first blocker was not raw compute.
- The first blocker was environment and integration stability.

### 2. SmolVLM As Encoder, SmolLM As Target LM

- We are not currently training by directly reusing the entire VLM object as the inference-time model.
- Instead, we use the VLM online to produce context features and apply generated LoRA to the aligned small text LM.

Reason:

- This is the shortest path to validate the D2L-style mechanism.
- It keeps the video path and the LoRA target path interpretable.

This is a prototype compromise, not necessarily the final architecture.

### 3. Narrow LoRA Target Set

- For the current prototype run, the default target modules were narrowed to `down_proj`.

Reason:

- The broader target set (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) exposed integration bugs in the custom LoRA-hook path.
- `down_proj` is a safer first target and still lets us test whether video-conditioned LoRA generation works at all.

This is a pragmatic debugging choice, not a claim that `down_proj` is optimal.

### 4. Per-Rank Generation Enabled

- The current prototype explicitly sets `per_rank_gen=True`.

Reason:

- The active hypernet head path expects an explicit rank dimension.
- This avoided a shape mismatch between aggregator output and the hypernet head for the current implementation.

This may not remain necessary after a cleaner refactor.

### 5. Sequence Packing Disabled

- The Video2LoRA training script sets `use_sequence_packing=False`.

Reason:

- The current dataloader produces padded batches, not the packed representation used by parts of the original repo.
- The packed LoRA forward path expects extra metadata that the current prototype does not supply.

This is a prototype-path simplification.

### 6. Non-Flash Perceiver Path

- The local perceiver path was adapted to run without relying on FlashAttention.

Reason:

- FlashAttention was not stable against the current locked torch environment on this machine during this attempt.
- The priority was to keep the environment stable and continue debugging the actual training path.

This is an environment-driven compromise, not a long-term recommendation.

## Environment Assumptions

- Project root: `/home/nvidia/manans/doc-to-lora`
- Project virtualenv: `/home/nvidia/manans/doc-to-lora/.venv`
- CUDA toolkit is installed at `/usr/local/cuda-13.2`
- The repo lockfile still expects `torch 2.6.0`
- NVIDIA runtime libraries are being loaded from the virtualenv package directories

Important environment choice:

- We restored the project to the lockfile torch stack after FlashAttention installation attempts destabilized the runtime.

## What This Attempt Is Discounting

The current attempt is intentionally discounting:

- multi-GPU efficiency
- aggressive performance optimization
- FlashAttention integration
- full target-module coverage
- polished dataset tooling beyond MSVD-QA
- long-video benchmark quality
- architectural finality around "same L" in one unified VLM object

These are all important, but they are not the bottleneck for the first successful run.

## Current Known Limitations

At the time of writing, this prototype still has active integration issues in the training path. The latest blockers have been shape/interface mismatches between:

- the perceiver output and hypernet head
- packed vs non-packed LoRA hook paths
- target-module assumptions and actual model execution shapes
- final logits/labels or projection-shape behavior during supervised loss computation

So this prototype is still in the "make the first run truly train" stage.

## Success Criteria For This First Attempt

This attempt should be considered successful if we can:

1. Launch training end to end on MSVD-QA without environment failures.
2. Complete forward and backward passes on a real batch.
3. Save checkpoints and log metrics to `wandb`.
4. Show that video-conditioned context features can drive generated LoRA updates on the target LM.

Only after that should we spend serious time on:

- expanding target modules
- multi-GPU training
- better datasets
- better objectives
- stronger evaluation
- more faithful unified-VLM inference designs

## Current Training Regime

The current training path should now be treated as a `stage 1` setup:

- single-chunk internalization only
- no compositional chunk training yet
- checkpoints from this phase are intended to seed a later `stage 2`

For the current MSVD-QA prototype launcher, the intended defaults are:

- `TRAIN_SAMPLES=5000`
- `VAL_SAMPLES=500`
- `BATCH_SIZE=16`
- `GRAD_ACCUM_STEPS=8`
- `MAX_FRAMES=16`
- full target set:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`

This is explicitly meant to produce useful `stage 1` checkpoints that can later be resumed for chunk-composition training.
