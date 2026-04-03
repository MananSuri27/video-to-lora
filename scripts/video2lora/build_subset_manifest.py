import argparse
import json
import random
from pathlib import Path


DATA_ROOT = Path("/data/video2lora")
DEFAULT_INPUT_CANDIDATES = [
    DATA_ROOT / "raw" / "manifest.jsonl",
    DATA_ROOT / "raw" / "all.jsonl",
    DATA_ROOT / "raw" / "qa.jsonl",
]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a tiny train/val subset manifest for Video2LoRA."
    )
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument(
        "--train-out",
        default=str(DATA_ROOT / "processed" / "train.jsonl"),
    )
    parser.add_argument(
        "--val-out",
        default=str(DATA_ROOT / "processed" / "val.jsonl"),
    )
    parser.add_argument("--total-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-video-exists", action="store_true")
    return parser.parse_args()


def resolve_input_manifest(input_manifest: str | None) -> Path:
    if input_manifest:
        path = Path(input_manifest)
        if not path.exists():
            raise FileNotFoundError(f"Input manifest not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(x) for x in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No input manifest found. Provide --input-manifest or create one of:\n"
        f"{searched}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "question" not in row or "answer" not in row or "video_path" not in row:
                raise ValueError(
                    f"Manifest row {line_no} must contain video_path, question, and answer."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def normalize_video_path(video_path: str) -> Path:
    path = Path(video_path)
    if not path.is_absolute():
        path = DATA_ROOT / path
    return path


def validate_rows(rows: list[dict], require_video_exists: bool) -> list[dict]:
    valid_rows = []
    for row in rows:
        video_path = normalize_video_path(row["video_path"])
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if require_video_exists and not video_path.exists():
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError("No valid rows found after video-path validation.")
    return valid_rows


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)

    input_manifest = resolve_input_manifest(args.input_manifest)
    rows = load_jsonl(input_manifest)
    rows = validate_rows(rows, require_video_exists=args.require_video_exists)
    random.shuffle(rows)

    total_samples = min(args.total_samples, len(rows))
    rows = rows[:total_samples]
    val_samples = min(args.val_samples, max(1, total_samples // 10), len(rows) - 1)
    train_rows = rows[val_samples:]
    val_rows = rows[:val_samples]

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    dump_jsonl(train_out, train_rows)
    dump_jsonl(val_out, val_rows)

    print(f"Input manifest: {input_manifest}")
    print(f"Train rows: {len(train_rows)} -> {train_out}")
    print(f"Val rows: {len(val_rows)} -> {val_out}")


if __name__ == "__main__":
    main()
