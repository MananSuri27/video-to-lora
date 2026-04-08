# Video2LoRA Prototype Notes

## Additional Notes

- [Video2LoRA Musings (2026-04-07)](musings_2026-04-07.md)
- [Video2LoRA Datasets And Research Notes (2026-04-07)](datasets_and_research_2026-04-07.md)

## Goal
Adapt Doc-to-LoRA so a small video-language model (VLM) can internalize a video and emit LoRA weights for a small base LM, using the same long-context perceiver-style aggregation strategy that the current project uses for document chunks.

## High-Level Idea
- Keep the current overall shape:
  1. Encode context.
  2. Aggregate long-context features with the perceiver bottleneck.
  3. Generate LoRA weights with the hypernetwork.
  4. Answer downstream queries with the frozen base LM plus generated LoRA.
- Replace text context with video context.
- Start with a tiny prototype that values fast iteration over full generality.

## Recommended Prototype Architecture
### Base LM
- Keep the generated-LoRA target as a small causal LM.
- Best practical first target in this repo: a small text LM already supported by the codepath.
- If we want to stay closest to the original setup, keep `google/gemma-2-2b-it` or switch to a small Qwen/Gemma text model that already works here.

### Video Context Encoder
- Use a tiny video-capable encoder to produce per-frame or short-clip features.
- For the prototype, the safest design is not "one monolithic video LLM end-to-end", but:
  - a frozen lightweight video encoder or image encoder applied to sampled frames,
  - an optional temporal adapter,
  - then the existing perceiver aggregator to compress long sequences of video tokens/features.
- This reduces code churn because the current hypernet stack already expects a sequence of context features plus an attention mask.

### Hypernetwork
- Keep the current hypernet strong enough to absorb modality mismatch:
  - keep the perceiver aggregator,
  - keep `per_layer_activations` or a similarly rich context encoder output path,
  - keep LoRA generation across the main attention/MLP target modules.
- The "hypernet strength" should come from:
  - enough latent size,
  - enough latent queries,
  - enough perceiver depth,
  - not over-shrinking temporal information too early.

## What Would Need To Change In This Codebase
### 1. Context representation: text tokens -> video features
Current code assumes `ctx_ids` are token IDs produced by a tokenizer. For video we likely want `ctx_features` or `video_features`.

Likely touchpoints:
- `src/ctx_to_lora/data/processing.py`
  - add a video preprocessing path that loads videos, samples frames/clips, and stores feature tensors or frame-token tensors.
- `src/ctx_to_lora/data/collator.py`
  - support batching padded video features and temporal masks, not just token IDs.
- `src/ctx_to_lora/data/packing.py`
  - current packing logic is written around tokenized `ctx_ids`; this should become modality-agnostic or get a separate video packing path.

### 2. Context encoder API
Current `ctx_encoder.py` expects a transformer-like model called with `input_ids` and `attention_mask`.

Needed change:
- introduce a video context encoder class, likely alongside:
  - `EarlyExit`
  - `EmbeddingOnly`
  - `PerLayerActivations`
- new variant could accept:
  - `pixel_values`,
  - `video_features`,
  - `frame_attention_mask`,
  - optional `position_ids` or temporal indices.

### 3. Aggregator input contract
This is the best reuse point.

`src/ctx_to_lora/modeling/aggregator.py` already wants:
- `ctx_features`
- `ctx_attn_mask`
- `ctx_position_ids`

That maps naturally to video:
- `ctx_features`: frame/clip embeddings
- `ctx_attn_mask`: valid frame mask
- `ctx_position_ids`: temporal positions

This part can remain mostly unchanged if we normalize video into a long sequence of features before the hypernet.

### 4. Model loading
`src/ctx_to_lora/model_loading.py` has limited vision-model handling and strips Gemma vision models down to the language model.

Needed change:
- add explicit support for a video encoder or VLM backbone as the context encoder,
- avoid collapsing the vision/video tower away when we actually need its outputs,
- likely split:
  - base LM loader
  - context encoder loader
  - multimodal/video feature extractor loader

### 5. Training config surface
`src/ctx_to_lora/configs.py` needs new args for video:
- frame sampling strategy
- fps or target clip stride
- max frames
- clip length
- spatial resolution
- whether to precompute video embeddings
- temporal pooling mode
- audio on/off if ever added later

### 6. Evaluation path
`src/ctx_to_lora/eval_utils.py` and metrics assume text context display/decoding in places.

Needed change:
- support video sample metadata,
- avoid trying to decode `ctx_ids` as text,
- add retrieval-style or QA-style metrics for video-conditioned answers.

## Small Prototype Recommendation
### Practical first version
- Base LM: small text LM already supported here.
- Video side: convert each video to a sequence of sampled frames.
- Encode frames with a small frozen vision backbone.
- Add a lightweight temporal projector or temporal transformer.
- Feed the resulting frame sequence into the existing perceiver aggregator.
- Generate LoRA for the base LM.
- Train on short video QA or caption-conditioned QA.

This gives us "Video2LoRA" without needing a full end-to-end tiny VLM stack on day one.

## Candidate Datasets
### Best for a fast proof-of-concept
- MSR-VTT QA or caption-style subsets
- TGIF-QA
- Next-QA
- MSVD-QA

### Best for a tiny contrived validation
- A synthetic dataset built from a very small public video set:
  - sample 1 to 8 frames per clip,
  - generate simple event/entity questions,
  - ask questions that require the video but not world knowledge.

### If we want a stronger benchmark later
- ActivityNet-QA
- EgoSchema
- Perception Test

## Contrived Tasks That Fit The Method
### Best first task: closed-book video QA after internalization
Train the model so that:
- context = video,
- query = question about the video,
- answer must be produced after LoRA internalization,
- no raw video is shown to the LM at generation time.

Example tasks:
- "What color was the car that passed in front of the camera?"
- "Did the person pick up the cup before opening the door?"
- "How many times did the ball bounce?"

### Other good contrived tasks
- temporal order QA
- action presence detection framed as QA
- object persistence QA
- short video-to-report generation
- contrastive QA with hard negatives from nearby clips

## Small-Subset Validation Plan
### Goal
Validate that generated LoRA helps the base LM answer questions about a specific video after seeing only the compressed internalized representation.

### Minimal setup
- 500 to 2,000 training samples
- 100 to 300 validation samples
- 1 dataset only
- 8 to 16 sampled frames per video
- single-turn QA
- short answers only

### Strongest quick check
Compare:
- base LM with no video
- base LM with a text caption only
- Video2LoRA with sampled-frame features

If Video2LoRA beats the no-video baseline and is competitive with caption-only conditioning on a tiny subset, the idea is worth scaling.

## Video-Specific Considerations Versus Documents
### Temporal structure
Video is not just long context; order matters much more. Temporal position encoding and frame sampling policy are first-class concerns.

### Redundancy
Neighboring frames are highly redundant. Aggressive temporal subsampling, deduplication, or shot-aware sampling matters much more than in text.

### Expensive context encoding
Video encoding is far costlier than tokenizing text. Precomputing video features is likely important for prototype speed.

### Variable length and sparsity
Important events can be brief and rare. Uniform sampling may miss the answer signal entirely.

### Multi-scale reasoning
Some questions need a single frame; others need cross-frame temporal reasoning. The model may need both local clip tokens and global summary tokens.

### Modality mismatch
The generated LoRA still steers a text LM. That means the hypernetwork has to bridge visual-temporal features into language-model weight updates. This is harder than the document setting and argues for keeping the hypernet reasonably expressive.

## Recommended First Refactor Plan
1. Add a modality-aware context batch format that supports either `ctx_ids` or `ctx_features`.
2. Add a video preprocessing pipeline that outputs sampled-frame embeddings and masks.
3. Add a video context encoder wrapper that can return per-layer or final video features.
4. Reuse the existing perceiver aggregator with temporal position IDs.
5. Train on a tiny closed-book video QA subset.
6. Compare against no-context and caption-only baselines.

## Suggested First Experiments
- Experiment A: precomputed frame embeddings -> perceiver -> hypernet -> Gemma/Qwen 2B LM
- Experiment B: same as A but with temporal-order questions only
- Experiment C: caption-only pseudo-video baseline
- Experiment D: random-frame ablation to test whether temporal signal matters

## Directory Intent
- `docs/video2lora/`: design notes, experiment writeups, task definitions
- `configs/video2lora/`: future YAML configs for prototype runs
