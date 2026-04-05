import argparse
import json
import random
from pathlib import Path


DATA_ROOT = Path("/data/video2lora")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import canonical MSRVTT-QA into the Video2LoRA manifest format."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DATA_ROOT / "raw" / "MSRVTT-QA"),
        help="Directory containing train_qa.json, val_qa.json, test_qa.json, and the video folder.",
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
        "--train-out",
        default=str(DATA_ROOT / "processed" / "msrvtt-qa-train.jsonl"),
    )
    parser.add_argument(
        "--val-out",
        default=str(DATA_ROOT / "processed" / "msrvtt-qa-val.jsonl"),
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def resolve_video_dir(dataset_root: Path) -> Path:
    for candidate in ("video", "videos", "Video", "Videos"):
        candidate_path = dataset_root / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"Could not find a video directory under {dataset_root}. Expected one of: video, videos, Video, Videos."
    )


def build_rows(
    records: list[dict],
    split: str,
    dataset_root: Path,
    video_dir: Path,
) -> list[dict]:
    rows = []
    for idx, record in enumerate(records):
        video_name = str(record["video"])
        video_path = video_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file for QA record: {video_path}")

        record_id = str(record.get("question_id", f"{split}_{idx}"))

        rows.append(
            {
                "id": f"{split}-msrvtt-qa-{record_id}",
                "video_path": str(video_path),
                "question": str(record["question"]),
                "answer": str(record["answer"]),
                "metadata": {
                    "dataset": "msrvtt-qa",
                    "question_id": record_id,
                    "video_name": video_name,
                    "dataset_root": str(dataset_root),
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

    dataset_root = Path(args.dataset_root)
    train_json = dataset_root / "train_qa.json"
    val_json = dataset_root / "val_qa.json"
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            "Expected canonical MSRVTT-QA annotation files under "
            f"{dataset_root}: train_qa.json and val_qa.json"
        )

    video_dir = resolve_video_dir(dataset_root)
    train_records = load_json(train_json)
    val_records = load_json(val_json)

    if args.train_samples > 0:
        train_records = random.sample(train_records, k=min(len(train_records), args.train_samples))
    if args.val_samples > 0:
        val_records = random.sample(val_records, k=min(len(val_records), args.val_samples))

    train_rows = build_rows(train_records, split="train", dataset_root=dataset_root, video_dir=video_dir)
    val_rows = build_rows(val_records, split="val", dataset_root=dataset_root, video_dir=video_dir)

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    dump_jsonl(Path(args.train_out), train_rows)
    dump_jsonl(Path(args.val_out), val_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.train_out}")
    print(f"Wrote {len(val_rows)} val rows to {args.val_out}")
    print(f"Using videos from {video_dir}")


if __name__ == "__main__":
    main()
