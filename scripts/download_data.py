from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

import kagglehub

DATASET_HANDLE = "boltzmannbrain/nab"
LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "nab"


def download_dataset(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(output_dir),
    )
    return Path(downloaded_path)


def download_labels(output_dir: Path) -> Path:
    labels_path = output_dir.expanduser().resolve() / "labels" / "combined_windows.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(LABELS_URL, labels_path)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            "Could not download NAB labels automatically. Download combined_windows.json "
            f"manually from {LABELS_URL} and save it to {labels_path}."
        ) from exc
    return labels_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NAB dataset and labels.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_dataset:
        dataset_path = download_dataset(args.output_dir)
        print(f"Dataset downloaded to {dataset_path}.")
    if not args.skip_labels:
        labels_path = download_labels(args.output_dir)
        print(f"Labels downloaded to {labels_path}.")


if __name__ == "__main__":
    main()
