from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "nab"


def iter_series_paths(
    dataset_dir: Path = DATASET_DIR,
    categories: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield NAB CSV paths"""
    dataset_dir = dataset_dir.expanduser().resolve()

    selected_categories = set(categories) if categories is not None else None
    for path in sorted(dataset_dir.glob("*/*/*.csv")):
        category = path.relative_to(dataset_dir).parts[0]
        if selected_categories is None or category in selected_categories:
            yield path


def load_series(path: Path) -> pd.DataFrame:
    """Load and validate one NAB time series"""
    path = path.expanduser().resolve()
    frame = pd.read_csv(path)

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_dataset(
    dataset_dir: Path = DATASET_DIR,
    categories: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load NAB series keyed by their category and filename"""
    dataset_dir = dataset_dir.expanduser().resolve()
    selected_categories = None if categories is None else tuple(categories)
    paths = list(iter_series_paths(dataset_dir, selected_categories))

    return {
        f"{path.relative_to(dataset_dir).parts[0]}/{path.name}": load_series(path) for path in paths
    }
