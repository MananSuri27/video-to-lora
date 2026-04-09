# Video2LoRA Layer-To-Layer Notes (2026-04-08)

## Scope
This note captures the latest discussion around the "layer-to-layer" path in the repo and how it differs from the current Video2LoRA training path.

The main questions were:
- What does `per_layer_activations` actually return?
- Does layer-to-layer keep all token positions or pool them?
- How is that different from the current video setup?
- If we move Video2LoRA toward layer-to-layer, what should the source activations be?
- Does that help with the visual-token bottleneck?

## Short Answer
- Yes, the repo's `per_layer_activations` path starts from activations for all positions.
- It returns a tensor shaped like `[batch, num_layers, seq_len, hidden]`.
- The position-wise representations are not pooled inside the context encoder.
- Compression happens later in the perceiver aggregator.
- The current Video2LoRA path does not use that mode. It passes a single sequence of pooled video features shaped like `[batch, seq_len, hidden]`.

## What `per_layer_activations` Returns
The relevant code is in [`ctx_encoder.py`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/ctx_encoder.py).

`PerLayerActivations.forward()`:
- forces `output_hidden_states=True`
- runs the context encoder
- returns `torch.stack(outputs.hidden_states, dim=1)`

Code reference:
- [`ctx_encoder.py#L134`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/ctx_encoder.py#L134)
- [`ctx_encoder.py#L145`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/ctx_encoder.py#L145)

The inline comment in the code says the shape is:
- `(batch_size, num_layers, seq_len, hidden_size)`

So in layer-to-layer mode, the hypernet input starts from:
- every returned layer
- every token position in each layer
- the full hidden size at each position

It is not a pooled document embedding.

## What The Aggregator Does With Those Activations
The relevant code is in [`aggregator.py`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py).

The perceiver aggregator is told whether the context encoder is layer-to-layer by:
- `layer_to_layer_ctx_encoder=True`

Code reference:
- [`hypernet.py#L94`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/hypernet.py#L94)
- [`hypernet.py#L106`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/hypernet.py#L106)

Inside the perceiver aggregator, there are two main cases.

### Case 1: Per-layer batched path
If `ctx_attn_mask` is present, the code reshapes:
- `ctx_features` from `[bs, num_layers, seq_len, d]`
- to `[(num_layers * bs), seq_len, d]`

and repeats the attention mask across layers.

Code reference:
- [`aggregator.py#L140`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py#L140)
- [`aggregator.py#L150`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py#L150)

Interpretation:
- each layer keeps all its token positions
- the perceiver processes each layer's sequence separately

### Case 2: Packed-context path
If `ctx_position_ids` is present, the code reshapes:
- `ctx_features` from `[1, num_layers, seq_len, d]`
- to `[1, num_layers * seq_len, d]`

and repeats the position ids accordingly.

Code reference:
- [`aggregator.py#L151`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py#L151)
- [`aggregator.py#L160`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py#L160)

Interpretation:
- all positions are still kept
- but now they are flattened across layers into one long context sequence before the perceiver

### Common Point
In both cases, the context encoder does not pool away token positions.

The actual compression happens in:
- the perceiver bottleneck
- the decoder latents that become LoRA embeddings

Code reference:
- [`aggregator.py#L162`](/home/nvidia/manans/video-to-lora/src/ctx_to_lora/modeling/aggregator.py#L162)

## How This Differs From The Current Video Path
The current online Video2LoRA trainer uses precomputed video features, not `per_layer_activations`.

In [`train_smolvlm_online.py`](/home/nvidia/manans/video-to-lora/scripts/video2lora/train_smolvlm_online.py):
- SmolVLM is run on sampled video frames
- the trainer reads `outputs.image_hidden_states`
- those features are grouped by valid visual units
- patch tokens for each visual unit are mean-pooled
- the resulting frame or visual-unit sequence is padded into `ctx_features`, `ctx_attn_mask`, and `ctx_position_ids`

Code reference:
- [`train_smolvlm_online.py#L598`](/home/nvidia/manans/video-to-lora/scripts/video2lora/train_smolvlm_online.py#L598)
- [`train_smolvlm_online.py#L660`](/home/nvidia/manans/video-to-lora/scripts/video2lora/train_smolvlm_online.py#L660)

That means the current hypernet input is:
- `[batch, seq_len, hidden]`

not:
- `[batch, num_layers, seq_len, hidden]`

So today we are using:
- one visual feature stream per video

rather than:
- one context feature stream per model layer

## Inductive Bias Comparison
### Current single-stream video path
Strengths:
- simpler
- easier to get running
- cheaper than feeding a full per-layer stack

Weaknesses:
- one shared representation has to generate LoRAs for all target LM layers
- less explicit alignment between low-level, mid-level, and high-level features

### Layer-to-layer path
Strengths:
- better inductive bias for generating different LoRAs for different target layers
- lower target layers can condition on lower-level context
- upper target layers can condition on more abstract context
- more natural fit for a deep LM being modulated layer-by-layer

Weaknesses:
- more expensive
- more memory-heavy
- more moving parts

For research, layer-to-layer looks like the stronger bias.
For a PoC, the current single-stream path is still a reasonable simplification.

## If We Shift Video2LoRA Toward Layer-To-Layer, What Activations Should We Use?
The best answer is:
- fused VLM text hidden states before the final `lm_head`

Not:
- final vocab logits

Usually not ideal as the first choice:
- raw visual-tower activations alone

Reasoning:
- the LoRAs are applied to the base LM's language transformer modules
- so the source representation should ideally already live in a language-model-like space
- fused text-side hidden states are a better match for that than final logits or purely visual features

So the preferred future path is:
1. run the VLM with visual input
2. let the VLM perform multimodal fusion
3. extract per-layer fused text hidden states
4. feed those into the hypernet in a layer-to-layer way

## Does Layer-To-Layer Remove The Visual Token Bottleneck?
No.

It can improve where compression happens, but it does not remove the upstream sampling problem.

### What it helps with
Layer-to-layer on the fused text side means:
- the VLM gets to process visual tokens first
- the multimodal model can perform some semantic compression before the hypernet sees the signal
- the hypernet can work with a more language-aligned representation

This is often better than compressing very early on raw visual features.

### What it does not solve
The VLM still has to ingest visual tokens upstream.
So we still care about:
- number of sampled frames
- number of tokens per frame
- whether the sampling policy missed the important event entirely

If a crucial event is not present in the sampled video tokens, no later layer-to-layer trick can recover it.

## Practical Framing
The tradeoff can be framed like this:

- Current setup:
  - compress early
  - use pooled visual features before deep multimodal reasoning

- Future layer-to-layer setup:
  - compress later
  - use fused text-side hidden states after multimodal reasoning

The second option is usually the better research direction.
But it still needs:
- frame or clip selection
- temporal coverage
- possibly hierarchical memory or retrieval for long videos
- robustness to different frame samples

## Current Takeaway
The repo's layer-to-layer path does use all positions.
The current Video2LoRA path does not.

So if Video2LoRA later moves to a true layer-to-layer design, the likely strongest version is:
- per-layer fused VLM text hidden states
- not logits
- not a single pooled video vector
- and still with explicit care for frame/token budget
