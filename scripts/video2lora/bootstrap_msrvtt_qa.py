import argparse
import re
import shutil
import zipfile
from pathlib import Path

import requests


ANNOTATION_FILES = {
    "train_qa.json": "12dJq5_7v8FytrJwrPB_f22tET1MmGCNh",
    "val_qa.json": "138q-A-V8fCC2nBYJgqkQa3gBfXVNbNNd",
    "test_qa.json": "13IiEcUMHiNppWhGwVY1eAaip6iSJM35A",
}
VIDEO_ARCHIVES = {
    "train_val_videos.zip": "https://www.mediafire.com/file/x3rrbe4hwp04e6w/train_val_videos.zip/file",
    "test_videos.zip": "https://www.mediafire.com/file/czh8sezbo9s4692/test_videos.zip/file",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare canonical-style MSRVTT-QA data."
    )
    parser.add_argument(
        "--dataset-root",
        default="/data/video2lora/raw/MSRVTT-QA",
        help="Destination directory for annotations and videos.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download test_qa.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and reextract files even if they already exist.",
    )
    return parser.parse_args()


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def confirm_token_from_text(text: str) -> str | None:
    patterns = [
        r'confirm=([0-9A-Za-z_]+)',
        r'"downloadUrl":"[^"]*confirm=([0-9A-Za-z_]+)',
        r"confirm=([0-9A-Za-z_-]+)&",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def resolve_mediafire_download_url(page_url: str) -> str:
    response = requests.get(page_url, timeout=30)
    response.raise_for_status()
    match = re.search(r'href="(https://download[^"]+)"', response.text)
    if not match:
        raise ValueError(f"Could not resolve MediaFire download link from {page_url}")
    return match.group(1).replace("&amp;", "&")


def download_stream(url: str, destination: Path, *, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print(f"[download] {label} -> {destination}")
    response = session.get(url, stream=True)
    response.raise_for_status()
    total_bytes = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
    print(f"[done] Saved {destination.name} ({format_size(total_bytes)})")


def download_drive_file(file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    print(f"[download] Google Drive file {file_id} -> {destination}")
    response = session.get(url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    needs_confirm = "text/html" in content_type.lower()
    if needs_confirm:
        response_text = response.text
        token = confirm_token_from_text(response_text)
        if token:
            print(f"[download] Confirmation token detected for {destination.name}; retrying confirmed download")
            response = session.get(
                url,
                params={"id": file_id, "confirm": token},
                stream=True,
            )
            response.raise_for_status()

    total_bytes = 0
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)
    print(f"[done] Saved {destination.name} ({format_size(total_bytes)})")


def ensure_annotations(dataset_root: Path, include_test: bool, force: bool) -> None:
    needed = ["train_qa.json", "val_qa.json"]
    if include_test:
        needed.append("test_qa.json")
    print(f"[stage] Ensuring annotations under {dataset_root}")
    for filename in needed:
        destination = dataset_root / filename
        if destination.exists() and not force:
            print(f"[skip] Reusing existing annotation {destination}")
            continue
        print(f"[fetch] Downloading annotation {filename}")
        download_drive_file(ANNOTATION_FILES[filename], destination)


def extract_video_zip(zip_path: Path, video_dir: Path, force: bool) -> None:
    existing_videos = list(video_dir.glob("video*.mp4"))
    if existing_videos and not force:
        print(
            f"[skip] Reusing existing extracted videos in {video_dir} "
            f"({len(existing_videos)} files found)"
        )
        return

    staging_dir = video_dir.parent / "_msrvtt_extract_tmp"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[extract] Unpacking {zip_path} into {staging_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(staging_dir)

        extracted_videos = sorted(staging_dir.rglob("video*.mp4"))
        if not extracted_videos:
            raise FileNotFoundError(f"No video*.mp4 files found after extracting {zip_path}")

        print(f"[extract] Moving {len(extracted_videos)} videos into {video_dir}")
        for src in extracted_videos:
            dst = video_dir / src.name
            if dst.exists():
                if force:
                    dst.unlink()
                else:
                    continue
            shutil.move(str(src), str(dst))
        print(f"[done] Video extraction complete: {len(extracted_videos)} files available in {video_dir}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return zipfile.is_zipfile(path)


def ensure_videos(dataset_root: Path, include_test: bool, force: bool) -> None:
    video_dir = dataset_root / "video"
    existing_videos = list(video_dir.glob("video*.mp4"))
    if existing_videos and not force:
        print(
            f"[skip] Reusing downloaded videos in {video_dir} "
            f"({len(existing_videos)} files found)"
        )
        return

    cache_dir = dataset_root / "_downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stage] Ensuring videos under {video_dir}")
    needed_archives = {"train_val_videos.zip": VIDEO_ARCHIVES["train_val_videos.zip"]}
    if include_test:
        needed_archives["test_videos.zip"] = VIDEO_ARCHIVES["test_videos.zip"]

    for archive_name, url in needed_archives.items():
        zip_path = cache_dir / archive_name
        if zip_path.exists() and not force and is_valid_zip(zip_path):
            print(f"[skip] Reusing cached archive {zip_path} ({format_size(zip_path.stat().st_size)})")
        else:
            if zip_path.exists() and not is_valid_zip(zip_path):
                print(f"[warn] Cached archive {zip_path} is not a valid zip; redownloading")
                zip_path.unlink()
            print(f"[fetch] Resolving MediaFire download for {archive_name}")
            resolved_url = resolve_mediafire_download_url(url)
            print(f"[fetch] Downloading {archive_name} into {cache_dir}")
            download_stream(resolved_url, zip_path, label=resolved_url)
        if zip_path.exists():
            print(f"[done] Video archive ready at {zip_path} ({format_size(zip_path.stat().st_size)})")
        extract_video_zip(zip_path, video_dir, force=force)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"[start] Preparing MSRVTT-QA under {dataset_root}")
    ensure_annotations(dataset_root, include_test=args.include_test, force=args.force)
    ensure_videos(dataset_root, include_test=args.include_test, force=args.force)
    video_count = len(list((dataset_root / "video").glob("video*.mp4")))
    print(f"[ready] MSRVTT-QA ready under {dataset_root} with {video_count} videos")


if __name__ == "__main__":
    main()
