import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    get_cosine_schedule_with_warmup,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.model_loading import get_lora_config
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)


DATA_ROOT = Path("/data/video2lora")


@dataclass
class TrainArgs:
    smolvlm_name_or_path: str
    base_lm_name_or_path: str
    train_manifest: str
    val_manifest: str | None
    output_dir: str
    epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    seed: int
    log_every: int
    eval_every: int
    save_every: int
    max_train_samples: int | None
    max_val_samples: int | None
    num_workers: int
    lora_r: int
    lora_dropout: float
    target_modules: list[str]
    latent_size: int
    dropout_rate: float
    n_latent_queries: int
    num_blocks: int
    num_self_attn_per_block: int
    video_fps: float
    max_frames: int
    wandb_project: str
    wandb_mode: str
    wandb_run_name: str | None
    wandb_notes: str | None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train Video2LoRA with online SmolVLM video feature extraction."
    )
    parser.add_argument(
        "--smolvlm-name-or-path",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    )
    parser.add_argument(
        "--base-lm-name-or-path",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    parser.add_argument(
        "--train-manifest",
        default=str(DATA_ROOT / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--val-manifest",
        default=str(DATA_ROOT / "processed" / "val.jsonl"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--n-latent-queries", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-self-attn-per-block", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--wandb-project", default="video2lora")
    parser.add_argument(
        "--wandb-mode",
        default="auto",
        choices=("auto", "online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output_dir = str(DATA_ROOT / "runs" / f"{timestamp}-smolvlm-online")

    val_manifest = parsed.val_manifest
    if val_manifest and not os.path.exists(val_manifest):
        val_manifest = None

    return TrainArgs(
        smolvlm_name_or_path=parsed.smolvlm_name_or_path,
        base_lm_name_or_path=parsed.base_lm_name_or_path,
        train_manifest=parsed.train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        grad_accum_steps=parsed.grad_accum_steps,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        warmup_steps=parsed.warmup_steps,
        max_grad_norm=parsed.max_grad_norm,
        seed=parsed.seed,
        log_every=parsed.log_every,
        eval_every=parsed.eval_every,
        save_every=parsed.save_every,
        max_train_samples=parsed.max_train_samples,
        max_val_samples=parsed.max_val_samples,
        num_workers=parsed.num_workers,
        lora_r=parsed.lora_r,
        lora_dropout=parsed.lora_dropout,
        target_modules=[x.strip() for x in parsed.target_modules.split(",") if x.strip()],
        latent_size=parsed.latent_size,
        dropout_rate=parsed.dropout_rate,
        n_latent_queries=parsed.n_latent_queries,
        num_blocks=parsed.num_blocks,
        num_self_attn_per_block=parsed.num_self_attn_per_block,
        video_fps=parsed.video_fps,
        max_frames=parsed.max_frames,
        wandb_project=parsed.wandb_project,
        wandb_mode=parsed.wandb_mode,
        wandb_run_name=parsed.wandb_run_name,
        wandb_notes=parsed.wandb_notes,
    )


def ensure_layout(output_dir: Path) -> None:
    for path in (
        DATA_ROOT,
        DATA_ROOT / "raw",
        DATA_ROOT / "processed",
        DATA_ROOT / "features",
        DATA_ROOT / "runs",
        DATA_ROOT / "cache",
        output_dir,
        output_dir / "checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


class VideoQADataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        video_path = Path(row["video_path"])
        if not video_path.is_absolute():
            video_path = DATA_ROOT / video_path
        return {
            "id": row.get("id", str(idx)),
            "video_path": str(video_path),
            "question": row["question"],
            "answer": row["answer"],
            "metadata": row.get("metadata", {}),
        }


def build_labels(tokenizer, question: str, answer: str):
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100
    return input_ids, attention_mask, labels


class SmolVLMOnlineCollator:
    def __init__(self, base_tokenizer, video_fps: float, max_frames: int):
        self.base_tokenizer = base_tokenizer
        self.video_fps = video_fps
        self.max_frames = max_frames

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_masks, labels = [], [], []
        ids = []
        video_messages = []

        for example in batch:
            inp_ids, attn_mask, lbl = build_labels(
                self.base_tokenizer,
                question=example["question"],
                answer=example["answer"],
            )
            input_ids.append(inp_ids)
            attention_masks.append(attn_mask)
            labels.append(lbl)
            ids.append(example["id"])
            video_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "path": example["video_path"]},
                            {
                                "type": "text",
                                "text": "Internalize this video for later question answering.",
                            },
                        ],
                    }
                ]
            )

        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.base_tokenizer.pad_token_id,
        )
        batch_attention_mask = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "ids": ids,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "video_messages": video_messages,
            "video_fps": self.video_fps,
            "max_frames": self.max_frames,
        }


def move_text_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_base_model(args: TrainArgs, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_lm_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_config = get_lora_config(
        args.base_lm_name_or_path,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_lm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device.type,
    )
    from peft import PeftModel

    base_model = PeftModel(base_model, peft_config)
    base_model.train()
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(base_model, "generation_config", None):
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return base_model, tokenizer


def build_video2lora_model(args: TrainArgs, device: torch.device):
    base_model, base_tokenizer = build_base_model(args, device)

    processor = AutoProcessor.from_pretrained(
        args.smolvlm_name_or_path,
        trust_remote_code=True,
    )
    vlm = AutoModelForImageTextToText.from_pretrained(
        args.smolvlm_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device.type,
    )
    vlm.eval()
    for param in vlm.parameters():
        param.requires_grad = False

    ctx_hidden_size = vlm.config.text_config.hidden_size
    hypernet_args = HypernetArguments(
        latent_size=args.latent_size,
        dropout_rate=args.dropout_rate,
        per_rank_gen=True,
    )
    aggregator_args = AggregatorArguments(
        aggregator_type="perceiver",
        n_latent_queries=args.n_latent_queries,
        num_blocks=args.num_blocks,
        num_self_attn_per_block=args.num_self_attn_per_block,
    )
    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path="precomputed",
        ctx_encoder_type="early_exit",
    )
    ctx_config = PretrainedConfig(hidden_size=ctx_hidden_size)
    hypernet_config = get_hypernet_config(
        base_model,
        ctx_config,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    )
    model = ModulatedPretrainedModel(
        base_model,
        hypernet_config,
        ctx_encoder_args,
        use_sequence_packing=False,
    )
    model.to(device)
    model.train()
    return model, base_tokenizer, processor, vlm


def prepare_smolvlm_inputs(processor, video_messages, device, video_fps: float, max_frames: int):
    vlm_inputs = processor.apply_chat_template(
        video_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        video_fps=video_fps,
        num_frames=max_frames,
    )
    moved = {}
    for key, value in vlm_inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def extract_video_features(vlm, vlm_inputs):
    outputs = vlm.model(
        input_ids=vlm_inputs["input_ids"],
        attention_mask=vlm_inputs.get("attention_mask"),
        pixel_values=vlm_inputs.get("pixel_values"),
        pixel_attention_mask=vlm_inputs.get("pixel_attention_mask"),
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
    )
    ctx_features = outputs.image_hidden_states
    if ctx_features is None:
        raise ValueError("SmolVLM did not return image_hidden_states for the provided videos.")
    batch_size = vlm_inputs["input_ids"].shape[0]
    if ctx_features.ndim != 3:
        raise ValueError(
            f"Expected SmolVLM image_hidden_states to be rank-3, got shape {tuple(ctx_features.shape)}."
        )
    pixel_attention_mask = vlm_inputs.get("pixel_attention_mask")
    if pixel_attention_mask is None:
        raise ValueError("Expected pixel_attention_mask in SmolVLM inputs for video batching.")

    # `image_hidden_states` is flattened across all valid visual units in the batch.
    # Recover per-example counts from the pixel attention mask, then concatenate each
    # example's valid visual units into one long token sequence and pad across examples.
    valid_visual_units = pixel_attention_mask.view(batch_size, pixel_attention_mask.shape[1], -1)
    valid_visual_units = valid_visual_units.any(dim=-1).sum(dim=-1).tolist()
    if sum(valid_visual_units) != ctx_features.shape[0]:
        raise ValueError(
            "SmolVLM visual feature count mismatch: "
            f"sum(valid_visual_units)={sum(valid_visual_units)} "
            f"vs image_hidden_states.shape[0]={ctx_features.shape[0]}."
        )

    seq_len = ctx_features.shape[1]
    split_features = []
    offset = 0
    for n_units in valid_visual_units:
        sample_features = ctx_features[offset : offset + n_units]
        split_features.append(sample_features.reshape(n_units * seq_len, ctx_features.shape[-1]))
        offset += n_units

    ctx_features = pad_sequence(split_features, batch_first=True, padding_value=0.0)
    ctx_attn_mask = torch.zeros(
        ctx_features.shape[:2], dtype=torch.long, device=ctx_features.device
    )
    ctx_position_ids = torch.zeros_like(ctx_attn_mask)
    for sample_idx, sample_features in enumerate(split_features):
        sample_len = sample_features.shape[0]
        ctx_attn_mask[sample_idx, :sample_len] = 1
        ctx_position_ids[sample_idx, :sample_len] = torch.arange(
            sample_len, dtype=torch.long, device=ctx_features.device
        )
    return ctx_features, ctx_attn_mask, ctx_position_ids


@torch.no_grad()
def evaluate(model, processor, vlm, data_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in data_loader:
        text_batch = move_text_batch_to_device(batch, device)
        vlm_inputs = prepare_smolvlm_inputs(
            processor,
            batch["video_messages"],
            device,
            video_fps=batch["video_fps"],
            max_frames=batch["max_frames"],
        )
        ctx_features, ctx_attn_mask, ctx_position_ids = extract_video_features(vlm, vlm_inputs)
        outputs = model(
            ctx_features=ctx_features,
            ctx_attn_mask=ctx_attn_mask,
            ctx_position_ids=ctx_position_ids,
            n_ctx_chunks=torch.ones(ctx_features.shape[0], dtype=torch.int32, device=device),
            input_ids=text_batch["input_ids"],
            attention_mask=text_batch["attention_mask"],
            labels=text_batch["labels"],
        )
        labels = text_batch["labels"]
        valid_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * valid_tokens
        total_tokens += valid_tokens
    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "ppl": float("nan")}
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "ppl": math.exp(min(mean_loss, 20))}


def save_checkpoint(model, output_dir: Path, step: int) -> Path:
    ckpt_path = output_dir / "checkpoints" / f"step-{step}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def resolve_wandb_mode(requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    return "offline"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_layout(output_dir)
    set_seed(args.seed)

    if not os.path.exists(args.train_manifest):
        raise FileNotFoundError(f"Train manifest not found: {args.train_manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_jsonl(args.train_manifest, args.max_train_samples)
    val_rows = (
        load_jsonl(args.val_manifest, args.max_val_samples) if args.val_manifest else []
    )

    model, base_tokenizer, processor, vlm = build_video2lora_model(args, device)
    collator = SmolVLMOnlineCollator(
        base_tokenizer=base_tokenizer,
        video_fps=args.video_fps,
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        VideoQADataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = None
    if val_rows:
        val_loader = DataLoader(
            VideoQADataset(val_rows),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    run_name = args.wandb_run_name or output_dir.name
    wandb_mode = resolve_wandb_mode(args.wandb_mode)
    if wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    print(f"Using wandb mode: {wandb_mode}")
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        notes=args.wandb_notes,
        config=asdict(args),
        mode=wandb_mode,
    )
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            text_batch = move_text_batch_to_device(batch, device)
            vlm_inputs = prepare_smolvlm_inputs(
                processor,
                batch["video_messages"],
                device,
                video_fps=batch["video_fps"],
                max_frames=batch["max_frames"],
            )
            with torch.no_grad():
                ctx_features, ctx_attn_mask, ctx_position_ids = extract_video_features(
                    vlm,
                    vlm_inputs,
                )

            outputs = model(
                ctx_features=ctx_features,
                ctx_attn_mask=ctx_attn_mask,
                ctx_position_ids=ctx_position_ids,
                n_ctx_chunks=torch.ones(
                    ctx_features.shape[0],
                    dtype=torch.int32,
                    device=device,
                ),
                input_ids=text_batch["input_ids"],
                attention_mask=text_batch["attention_mask"],
                labels=text_batch["labels"],
            )
            raw_loss = outputs.loss
            loss = raw_loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    wandb.log(
                        {
                            "train/loss": raw_loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + (step / max(len(train_loader), 1)),
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

                if val_loader is not None and global_step % args.eval_every == 0:
                    metrics = evaluate(model, processor, vlm, val_loader, device)
                    wandb.log(
                        {
                            "val/loss": metrics["loss"],
                            "val/ppl": metrics["ppl"],
                        },
                        step=global_step,
                    )

                if global_step % args.save_every == 0:
                    ckpt_path = save_checkpoint(model, output_dir, global_step)
                    wandb.log({"checkpoint/step": global_step}, step=global_step)
                    print(f"Saved checkpoint to {ckpt_path}")

    final_ckpt = save_checkpoint(model, output_dir, global_step or 0)
    if val_loader is not None:
        metrics = evaluate(model, processor, vlm, val_loader, device)
        wandb.log(
            {
                "val/final_loss": metrics["loss"],
                "val/final_ppl": metrics["ppl"],
            },
            step=max(global_step, 1),
        )
    wandb.finish()
    print(f"Training finished. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
