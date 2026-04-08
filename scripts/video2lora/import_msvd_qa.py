import argparse
import json
import random
from pathlib import Path

import cv2
from datasets import Dataset, load_dataset
import numpy as np


DATA_ROOT = Path("/data/video2lora")
DATASET_ID = "morpheushoc/msvd-qa"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import MSVD-QA into the Video2LoRA manifest format."
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
        help="Number of QA rows to keep. Use 0 or a negative value to keep the full val split.",
    )
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


def load_split(split: str) -> Dataset:
    return load_dataset(DATASET_ID, split=split)


def sample_examples(ds: Dataset, n_qa_rows: int, seed: int) -> Dataset:
    if n_qa_rows <= 0:
        return ds
    # MSVD-QA has about five QA pairs per source video, so sample videos first.
    n_video_rows = min(len(ds), max(1, n_qa_rows // 5))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), k=n_video_rows))
    return ds.select(indices)


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


def explode_rows(ds: Dataset, split: str) -> list[dict]:
    rows = []
    for row in ds:
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

    train_examples = sample_examples(train_df, args.train_samples, args.seed)
    val_examples = sample_examples(val_df, args.val_samples, args.seed)

    rows_to_materialize: dict[str, dict] = {}
    for row in train_examples:
        rows_to_materialize[row["video_path"]] = row
    for row in val_examples:
        rows_to_materialize[row["video_path"]] = row

    for row in rows_to_materialize.values():
        materialize_video(row, output_dir, overwrite=args.overwrite_videos)

    train_rows = explode_rows(train_examples, split="train")
    val_rows = explode_rows(val_examples, split="val")
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    if args.train_samples > 0:
        train_rows = train_rows[: args.train_samples]
    if args.val_samples > 0:
        val_rows = val_rows[: args.val_samples]

    dump_jsonl(Path(args.train_out), train_rows)
    dump_jsonl(Path(args.val_out), val_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(val_rows)} val rows to {args.val_out}")
    print(f"Videos stored under {output_dir}")


if __name__ == "__main__":
    main()
