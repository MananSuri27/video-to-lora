import argparse
import base64
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


DATA_ROOT = Path("/data/video2lora")
DATASET_ID = "MMInstruction/M3IT"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import an M3IT video QA dataset into the Video2LoRA manifest format."
    )
    parser.add_argument(
        "--dataset-name",
        default="msrvtt-qa",
        help="M3IT dataset name, for example msrvtt-qa or activitynet-qa.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="Number of QA rows to keep. Use 0 or a negative value to keep the full train split.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=0,
        help="Number of QA rows to keep. Use 0 or a negative value to keep the full validation split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="FPS to use when materializing videos from frame lists.",
    )
    parser.add_argument(
        "--raw-video-dir",
        default="",
        help="Where to write materialized videos. Defaults to /data/video2lora/raw/<dataset-name>.",
    )
    parser.add_argument(
        "--train-out",
        default="",
        help="Train manifest path. Defaults to /data/video2lora/processed/<dataset-name>-train.jsonl.",
    )
    parser.add_argument(
        "--val-out",
        default="",
        help="Val manifest path. Defaults to /data/video2lora/processed/<dataset-name>-val.jsonl.",
    )
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def normalize_split_name(split: str) -> str:
    if split == "val":
        return "validation"
    return split


def list_split_files(dataset_name: str, split: str) -> list[str]:
    normalized = normalize_split_name(split)
    prefix = f"{dataset_name}/"
    files = list_repo_files(DATASET_ID, repo_type="dataset")
    candidates = [
        f
        for f in files
        if f.startswith(prefix)
        and normalized in Path(f).name
        and f.endswith(".parquet")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found for dataset={dataset_name!r} split={split!r} in {DATASET_ID}."
        )
    return sorted(candidates)


def load_split(dataset_name: str, split: str) -> pd.DataFrame:
    shard_paths = [
        hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=filename)
        for filename in list_split_files(dataset_name, split)
    ]
    frames = [pd.read_parquet(path) for path in shard_paths]
    return pd.concat(frames, ignore_index=True)


def decode_frame(frame_str: str) -> np.ndarray:
    raw = base64.b64decode(frame_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode frame from M3IT video_str/image_base64_str entry.")
    return frame


def materialize_video(row, output_dir: Path, overwrite: bool, fps: float) -> Path:
    sample_id = str(row.get("id", row.name))
    output_path = output_dir / f"{sample_id}.mp4"
    if output_path.exists() and not overwrite:
        return output_path

    frame_strings = row.get("image_base64_str")
    if not isinstance(frame_strings, list) or not frame_strings:
        raise ValueError(f"Expected a non-empty image_base64_str list for sample {sample_id}.")

    frames = [decode_frame(frame_str) for frame_str in frame_strings]
    height, width = frames[0].shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def assign_sample_ids(df: pd.DataFrame, split: str, dataset_name: str) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["sample_id"] = [f"{split}-{dataset_name}-{idx}" for idx in range(len(df))]
    return df


def build_rows(df: pd.DataFrame, dataset_name: str, raw_dir_name: str) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        rows.append(
            {
                "id": sample_id,
                "video_path": f"raw/{raw_dir_name}/{sample_id}.mp4",
                "question": str(row["inputs"]),
                "answer": str(row["outputs"]),
                "metadata": {
                    "dataset": dataset_name,
                    "source_repo": DATASET_ID,
                },
            }
        )
    return rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_slug = args.dataset_name.replace("/", "-")
    raw_dir_name = dataset_slug.replace("_", "-")
    output_dir = Path(args.raw_video_dir) if args.raw_video_dir else DATA_ROOT / "raw" / raw_dir_name
    train_out = Path(args.train_out) if args.train_out else DATA_ROOT / "processed" / f"{dataset_slug}-train.jsonl"
    val_out = Path(args.val_out) if args.val_out else DATA_ROOT / "processed" / f"{dataset_slug}-val.jsonl"

    train_df = load_split(args.dataset_name, "train")
    val_df = load_split(args.dataset_name, "val")

    if args.train_samples > 0:
        train_df = train_df.sample(n=min(len(train_df), args.train_samples), random_state=args.seed)
    if args.val_samples > 0:
        val_df = val_df.sample(n=min(len(val_df), args.val_samples), random_state=args.seed)

    train_df = assign_sample_ids(train_df, split="train", dataset_name=dataset_slug)
    val_df = assign_sample_ids(val_df, split="val", dataset_name=dataset_slug)

    for split_df in (train_df, val_df):
        for row_idx, row in split_df.iterrows():
            materialize_video(
                row={"id": row["sample_id"], **row.to_dict()},
                output_dir=output_dir,
                overwrite=args.overwrite_videos,
                fps=args.fps,
            )

    train_rows = build_rows(train_df, dataset_name=dataset_slug, raw_dir_name=raw_dir_name)
    val_rows = build_rows(val_df, dataset_name=dataset_slug, raw_dir_name=raw_dir_name)
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    dump_jsonl(train_out, train_rows)
    dump_jsonl(val_out, val_rows)

    print(f"Wrote {len(train_rows)} train rows to {train_out}")
    print(f"Wrote {len(val_rows)} val rows to {val_out}")
    print(f"Videos stored under {output_dir}")


if __name__ == "__main__":
    main()
