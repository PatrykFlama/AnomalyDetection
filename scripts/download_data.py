from pathlib import Path

import kagglehub

DATASET_HANDLE = "boltzmannbrain/nab"
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


def main() -> None:
    download_dataset(OUTPUT_DIR)


if __name__ == "__main__":
    main()
