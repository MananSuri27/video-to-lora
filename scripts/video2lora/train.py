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
    PretrainedConfig,
    get_cosine_schedule_with_warmup,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)


DATA_ROOT = Path("/data/video2lora")


@dataclass
class TrainArgs:
    model_name_or_path: str
    train_manifest: str
    val_manifest: str | None
    output_dir: str
    ctx_feature_size: int
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
    wandb_project: str
    wandb_run_name: str | None
    wandb_notes: str | None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Train a Video2LoRA prototype from precomputed video features."
    )
    parser.add_argument("--model-name-or-path", default="google/gemma-2-2b-it")
    parser.add_argument(
        "--train-manifest",
        default=str(DATA_ROOT / "processed" / "train.jsonl"),
    )
    parser.add_argument("--val-manifest", default=str(DATA_ROOT / "processed" / "val.jsonl"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ctx-feature-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
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
    parser.add_argument("--wandb-project", default="video2lora")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-notes", default=None)
    parsed = parser.parse_args()

    output_dir = parsed.output_dir
    if not output_dir:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        model_stub = parsed.model_name_or_path.split("/")[-1]
        output_dir = str(DATA_ROOT / "runs" / f"{timestamp}-{model_stub}")

    val_manifest = parsed.val_manifest
    if val_manifest and not os.path.exists(val_manifest):
        val_manifest = None

    return TrainArgs(
        model_name_or_path=parsed.model_name_or_path,
        train_manifest=parsed.train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        ctx_feature_size=parsed.ctx_feature_size,
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
        wandb_project=parsed.wandb_project,
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


class VideoFeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = DATA_ROOT / feature_path
        feature_blob = torch.load(feature_path, map_location="cpu", weights_only=False)

        if isinstance(feature_blob, torch.Tensor):
            feature_blob = {"ctx_features": feature_blob}

        if "question" in row:
            prompt = row["question"]
        else:
            prompt = row["prompt"]

        answer = row["answer"]
        return {
            "id": row.get("id", str(idx)),
            "prompt": prompt,
            "answer": answer,
            "ctx_features": feature_blob["ctx_features"],
            "ctx_attn_mask": feature_blob.get("ctx_attn_mask"),
            "ctx_position_ids": feature_blob.get("ctx_position_ids"),
            "metadata": row.get("metadata", {}),
        }


def to_chunk_major(
    tensor: torch.Tensor | None,
    *,
    seq_len: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if seq_len is not None and tensor.shape[-1] != seq_len:
        raise ValueError(f"Expected sequence length {seq_len}, got {tensor.shape[-1]}")
    return tensor


def build_labels(
    tokenizer,
    prompt: str,
    answer: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        {"role": "user", "content": prompt},
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


class VideoFeatureCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids, attention_masks, labels = [], [], []
        ctx_features, ctx_attn_masks, ctx_position_ids = [], [], []
        n_ctx_chunks = []
        ids = []

        for example in batch:
            inp_ids, attn_mask, lbl = build_labels(
                self.tokenizer,
                prompt=example["prompt"],
                answer=example["answer"],
            )
            input_ids.append(inp_ids)
            attention_masks.append(attn_mask)
            labels.append(lbl)
            ids.append(example["id"])

            feature_chunks = to_chunk_major(example["ctx_features"], dtype=torch.float32)
            n_chunks, seq_len = feature_chunks.shape[0], feature_chunks.shape[1]
            ctx_features.extend(feature_chunks.unbind(dim=0))
            n_ctx_chunks.append(n_chunks)

            attn_chunks = to_chunk_major(
                example["ctx_attn_mask"],
                seq_len=seq_len,
                dtype=torch.long,
            )
            if attn_chunks is None:
                attn_chunks = torch.ones((n_chunks, seq_len), dtype=torch.long)
            ctx_attn_masks.extend(attn_chunks.unbind(dim=0))

            pos_chunks = to_chunk_major(
                example["ctx_position_ids"],
                seq_len=seq_len,
                dtype=torch.long,
            )
            if pos_chunks is None:
                base = torch.arange(seq_len, dtype=torch.long)
                pos_chunks = base.unsqueeze(0).repeat(n_chunks, 1)
            ctx_position_ids.extend(pos_chunks.unbind(dim=0))

        batch_input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
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

        padded_ctx_features = pad_sequence(
            ctx_features,
            batch_first=True,
            padding_value=0.0,
        )
        padded_ctx_attn = pad_sequence(
            ctx_attn_masks,
            batch_first=True,
            padding_value=0,
        )
        padded_ctx_pos = pad_sequence(
            ctx_position_ids,
            batch_first=True,
            padding_value=0,
        )

        return {
            "ids": ids,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "ctx_features": padded_ctx_features,
            "ctx_attn_mask": padded_ctx_attn,
            "ctx_position_ids": padded_ctx_pos,
            "n_ctx_chunks": torch.tensor(n_ctx_chunks, dtype=torch.int32),
        }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_model(args: TrainArgs, device: torch.device):
    peft_config = get_lora_config(
        args.model_name_or_path,
        lora_r=args.lora_r,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    base_model, tokenizer = get_model_and_tokenizer(
        args.model_name_or_path,
        train=True,
        requires_grad=False,
        peft_config=peft_config,
        device=device.type,
    )

    hypernet_args = HypernetArguments(
        latent_size=args.latent_size,
        dropout_rate=args.dropout_rate,
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
    ctx_config = PretrainedConfig(hidden_size=args.ctx_feature_size)
    hypernet_config = get_hypernet_config(
        base_model,
        ctx_config,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    )
    model = ModulatedPretrainedModel(base_model, hypernet_config, ctx_encoder_args)
    model.to(device)
    model.train()
    return model, tokenizer


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            ctx_features=batch["ctx_features"],
            ctx_attn_mask=batch["ctx_attn_mask"],
            ctx_position_ids=batch["ctx_position_ids"],
            n_ctx_chunks=batch["n_ctx_chunks"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        labels = batch["labels"]
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

    model, tokenizer = build_model(args, device)
    collator = VideoFeatureCollator(tokenizer)
    train_loader = DataLoader(
        VideoFeatureDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    val_loader = None
    if val_rows:
        val_loader = DataLoader(
            VideoFeatureDataset(val_rows),
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
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        notes=args.wandb_notes,
        config=asdict(args),
    )

    with open(output_dir / "train_args.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(
                ctx_features=batch["ctx_features"],
                ctx_attn_mask=batch["ctx_attn_mask"],
                ctx_position_ids=batch["ctx_position_ids"],
                n_ctx_chunks=batch["n_ctx_chunks"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / args.grad_accum_steps
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
                            "train/loss": loss.item() * args.grad_accum_steps,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch + (step / max(len(train_loader), 1)),
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

                if val_loader is not None and global_step % args.eval_every == 0:
                    metrics = evaluate(model, val_loader, device)
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
        metrics = evaluate(model, val_loader, device)
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
