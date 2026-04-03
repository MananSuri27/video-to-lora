import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


DATA_ROOT = Path("/data/video2lora")
DATASET_ID = "morpheushoc/msvd-qa"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a tiny MSVD-QA subset into the Video2LoRA manifest format."
    )
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-video-dir",
        default=str(DATA_ROOT / "raw" / "msvd_qa"),
    )
    parser.add_argument(
        "--train-out",
        default=str(DATA_ROOT / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--val-out",
        default=str(DATA_ROOT / "processed" / "val.jsonl"),
    )
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def list_split_files(split: str) -> list[str]:
    files = list_repo_files(DATASET_ID, repo_type="dataset")
    prefix = f"data/{split}-"
    return sorted([f for f in files if f.startswith(prefix) and f.endswith(".parquet")])


def load_split(split: str) -> pd.DataFrame:
    shard_paths = [
        hf_hub_download(repo_id=DATASET_ID, repo_type="dataset", filename=filename)
        for filename in list_split_files(split)
    ]
    frames = [pd.read_parquet(path) for path in shard_paths]
    return pd.concat(frames, ignore_index=True)


def video_rel_path(source_path: str) -> str:
    stem = Path(source_path).stem
    return f"raw/msvd_qa/{stem}.mp4"


def materialize_video(
    row,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    source_path = row["video_path"]
    output_path = output_dir / f"{Path(source_path).stem}.mp4"
    if output_path.exists() and not overwrite:
        return output_path

    num_frames = int(row["num_frames"])
    height = int(row["height"])
    width = int(row["width"])
    channels = int(row["channels"])
    raw = np.frombuffer(row["binary_frames"], dtype=np.uint8)
    expected = num_frames * height * width * channels
    if raw.size != expected:
        raise ValueError(
            f"Unexpected frame buffer size for {source_path}: got {raw.size}, expected {expected}"
        )
    frames = raw.reshape(num_frames, height, width, channels)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        3.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame in frames:
            if channels == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)
    finally:
        writer.release()
    return output_path


def explode_rows(df: pd.DataFrame, split: str) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        rel_video_path = video_rel_path(row["video_path"])
        for qa_idx, qa_pair in enumerate(row["qa"]):
            if len(qa_pair) < 2:
                continue
            question, answer = qa_pair[0], qa_pair[1]
            rows.append(
                {
                    "id": f"{split}-{Path(row['video_path']).stem}-{qa_idx}",
                    "video_path": rel_video_path,
                    "question": str(question),
                    "answer": str(answer),
                    "metadata": {
                        "dataset": "msvd-qa",
                        "source_video_path": row["video_path"],
                        "caption_count": len(row["caption"]),
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
    output_dir = Path(args.raw_video_dir)

    train_df = load_split("train")
    val_df = load_split("val")

    train_examples = train_df.sample(
        n=min(len(train_df), max(1, args.train_samples // 5)),
        random_state=args.seed,
    )
    val_examples = val_df.sample(
        n=min(len(val_df), max(1, args.val_samples // 5)),
        random_state=args.seed,
    )

    for _, row in pd.concat([train_examples, val_examples]).drop_duplicates(
        subset=["video_path"]
    ).iterrows():
        materialize_video(row, output_dir, overwrite=args.overwrite_videos)

    train_rows = explode_rows(train_examples, split="train")
    val_rows = explode_rows(val_examples, split="val")
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    train_rows = train_rows[: args.train_samples]
    val_rows = val_rows[: args.val_samples]

    dump_jsonl(Path(args.train_out), train_rows)
    dump_jsonl(Path(args.val_out), val_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(val_rows)} val rows to {args.val_out}")
    print(f"Videos stored under {output_dir}")


if __name__ == "__main__":
    main()
