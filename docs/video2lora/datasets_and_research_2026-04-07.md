# Video2LoRA Datasets And Research Notes

Date: 2026-04-07

This note captures:

- what was downloaded locally
- what the current datasets are actually good for
- where the current data is weak for Video2LoRA
- what longer-video work is most relevant as inspiration

## Local Data Inventory

Materialized data lives under `/data/video2lora`.

Observed local footprint:

- `/data/video2lora`: about 14G
- `~/.cache/huggingface/datasets/morpheushoc___msvd-qa`: about 16G
- `~/.cache/huggingface/datasets/MMInstruction___m3_it`: about 4.8G

### Downloaded Datasets

| Dataset | Local manifest(s) | Train rows | Val rows | Unique videos | Avg QA / video | What it really is |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MSVD-QA | `/data/video2lora/processed/train.jsonl`, `/data/video2lora/processed/val.jsonl` | 30,933 | 6,415 | 1,406 | 26.56 | Real short videos, many QAs per video |
| MSRVTT-QA canonical | `/data/video2lora/processed/msrvtt-qa-canonical-train.jsonl`, `/data/video2lora/processed/msrvtt-qa-canonical-val.jsonl` | 158,581 | 12,278 | 7,010 | 24.37 | Real short videos, many QAs per video |
| M3IT `msrvtt-qa` | `/data/video2lora/processed/msrvtt-qa-train.jsonl`, `/data/video2lora/processed/msrvtt-qa-val.jsonl` | 6,513 | 497 | 7,010 | 1.00 | One 8-frame synthetic clip per QA |
| M3IT `activitynet-qa` | `/data/video2lora/processed/activitynet-qa-train.jsonl`, `/data/video2lora/processed/activitynet-qa-val.jsonl` | 3,200 | 1,800 | 5,000 | 1.00 | One 8-frame synthetic clip per QA |

### Video Characteristics

The sampled video stats below come from local files after materialization.

| Dataset | Duration sample | Frame-count sample | FPS sample | Notes |
| --- | --- | --- | --- | --- |
| MSVD-QA | median about 3.0s, mean about 3.54s, max about 15.67s | median about 9 frames | fixed 3 FPS | Extremely short clips |
| MSRVTT-QA canonical | median about 12.01s, mean about 14.86s, max about 30.03s | median about 359.5 frames | median about 29.97 FPS | Best current real-video PoC source |
| M3IT `msrvtt-qa` | always about 2.67s | always 8 frames | fixed 3 FPS | Debugging-friendly, not long-video-like |
| M3IT `activitynet-qa` | always about 2.67s | always 8 frames | fixed 3 FPS | Better QA variety than MSVD, still not long-video supervision |

## What The Current Data Tells Me

### 1. We do have enough data for a short-video PoC

Especially because:

- MSVD-QA and canonical MSRVTT-QA both provide many questions per video
- canonical MSRVTT-QA is reasonably large for a first end-to-end attempt
- the local pipelines now work and materialize real `.mp4` files

So for a first "can video-conditioned LoRA internalization work at all?" experiment, we are not blocked on data.

### 2. We do not yet have good long-video training data in the current local bundle

This is the important caveat.

The current local datasets are mostly:

- short real videos, or
- synthetic 8-frame projections of larger datasets

That is good for plumbing and debugging, but it is not enough to claim we solved long-video internalization.

### 3. The current QA data has strong shortcut pressure

Observed biases:

- MSVD-QA answers are always 1 word in the imported format
- canonical MSRVTT-QA answers are also always 1 word
- the top 10 answers cover about 35.9% of MSVD-QA
- the top 10 answers cover about 32.6% of canonical MSRVTT-QA

That means:

- train loss can improve from answer priors
- a brittle adapter may still look good in training
- validation loss can rise even while the model gets better at exploiting dataset regularities

### 4. Canonical MSRVTT-QA is more useful than the M3IT `msrvtt-qa` version for Video2LoRA

The canonical set gives:

- real video durations
- many questions per video
- a natural setup for video-centric training

The M3IT version gives:

- exactly one QA per video
- a fixed 8-frame clip
- templated, longer answers

So M3IT `msrvtt-qa` is useful as a lightweight debugging dataset, but canonical MSRVTT-QA is the more faithful D2L-style training source.

## Do We Need More Data Sources?

Yes, if the goal includes long-video Video2LoRA rather than only a short-video PoC.

### My answer by phase

For a short-video PoC:

- no, we can start now with MSVD-QA + canonical MSRVTT-QA

For medium/long-video Video2LoRA research:

- yes, we need more data sources

## What I Would Add Next

### Training data I would prioritize

1. Canonical ActivityNet-QA, not only the M3IT 8-frame version.
2. TVQA / TVQA+, because subtitle-heavy supervision is useful for internalization.
3. NExT-QA or STAR, for temporal and causal reasoning.
4. Ego4D-derived QA or EgoSchema-like sources, if we want long egocentric memory.
5. A synthetic multi-question-per-video dataset generated from long videos plus subtitles, because the training unit we want is "one video -> many questions."

### Evaluation data I would prioritize

1. LongVideoBench
2. Video-MME
3. MLVU
4. LVBench
5. synthetic visual needle-in-a-haystack tests

These are more useful for "does Video2LoRA really preserve long context?" than short-video QA alone.

## Research Inspiration Worth Borrowing

### Doc-to-LoRA

Source:

- https://arxiv.org/abs/2602.15902

Why it matters:

- D2L is about generating an adapter from a context in one pass
- the adapter should support multiple downstream queries
- it is explicitly motivated by avoiding repeated consumption of long contexts

The main lesson for Video2LoRA is that training should be context-centric, not single-QA-centric.

### LongVideoBench

Sources:

- https://arxiv.org/abs/2407.15754
- https://github.com/longvideobench/LongVideoBench

What I want to borrow:

- long-context video plus subtitle inputs
- referred-context reasoning
- evaluation that actually depends on retrieving the right moment from a long sequence

The benchmark includes 3,763 videos, 6,678 human-annotated multiple-choice questions, and inputs up to an hour long.

### MLVU

Source:

- https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_MLVU_Benchmarking_Multi-task_Long_Video_Understanding_CVPR_2025_paper.html

What I want to borrow:

- task diversity
- explicit emphasis on multi-detail reasoning
- the observation that many models collapse on action order, action count, and other multi-detail tasks

This matters because Video2LoRA should probably be judged on nuanced detail retention, not just gist.

### LongVU

Source:

- https://arxiv.org/abs/2410.17434

What I want to borrow:

- adaptive temporal pruning
- adaptive spatial compression
- using redundancy reduction before the language model

For Video2LoRA, this translates into: do not treat every frame patch equally before the hypernetwork.

### Hour-LLaVA / VideoMarathon

Source:

- https://videomarathon.github.io/

What I want to borrow:

- cached full-video features
- question-relevant memory augmentation
- 1-FPS long-video modeling with a memory bank rather than brute-force full-context decoding every time

This is very relevant to a future "global adapter plus retrieved segment memory" Video2LoRA design.

### Temporal Reasoning Transfer from Text to Video

Source:

- https://arxiv.org/abs/2410.06166

What I want to borrow:

- the idea that temporal reasoning is partly a language-model bottleneck
- text-only temporal supervision can improve video temporal reasoning

This suggests Video2LoRA should not rely only on video QA data. We may also want text-only temporal training or distillation.

## My Current Recommendation

### For the next practical step

Train on:

- MSVD-QA
- canonical MSRVTT-QA

Use M3IT versions only for:

- smoke tests
- speed debugging
- ablations

### For the next research step

Add at least one real longer-video source plus one harder benchmark:

- canonical ActivityNet-QA or TVQA / TVQA+
- LongVideoBench or MLVU for evaluation

### For the long-video version of Video2LoRA

I think the final system should probably combine:

- coarse global video compression
- segment-level memory
- multiple questions per video during training
- optional subtitles / ASR
- question-time retrieval or adapter composition

not a single flat adapter generated from a blindly flattened patch sequence.

