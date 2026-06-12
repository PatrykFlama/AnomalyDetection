from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.data import DATASET_DIR, PROJECT_ROOT

DEFAULT_LABELS_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
)


def default_label_paths(dataset_dir: Path = DATASET_DIR) -> tuple[Path, ...]:
    dataset_dir = dataset_dir.expanduser().resolve()
    return (
        dataset_dir / "labels" / "combined_windows.json",
        dataset_dir / "combined_windows.json",
        PROJECT_ROOT / "data" / "nab_labels" / "combined_windows.json",
        PROJECT_ROOT / "labels" / "combined_windows.json",
    )


def download_labels(dataset_dir: Path = DATASET_DIR) -> Path:
    destination = dataset_dir.expanduser().resolve() / "labels" / "combined_windows.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(DEFAULT_LABELS_URL, destination)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            "Could not download NAB labels. Download combined_windows.json manually and pass "
            f"it with --labels-file. URL: {DEFAULT_LABELS_URL}"
        ) from exc
    return destination


def find_labels_file(
    explicit_path: Path | None = None,
    dataset_dir: Path = DATASET_DIR,
    auto_download: bool = True,
) -> Path | None:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Labels file does not exist: {path}")
        return path
    for path in default_label_paths(dataset_dir):
        if path.is_file():
            return path.resolve()
    if auto_download:
        return download_labels(dataset_dir)
    return None


def load_label_windows(labels_file: Path | None) -> dict[str, Any]:
    if labels_file is None:
        return {}
    data = json.loads(labels_file.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(data, dict) and "windows" in data and isinstance(data["windows"], dict):
        data = data["windows"]
    if not isinstance(data, dict):
        raise ValueError("Labels file must contain a JSON object mapping series names to windows.")
    return {str(key).replace("\\", "/"): value for key, value in data.items()}


def canonical_label_keys(series_name: str) -> tuple[str, ...]:
    path = Path(series_name.replace("\\", "/"))
    return (
        series_name,
        series_name.replace("\\", "/"),
        f"{path.parts[0]}/{path.name}" if len(path.parts) > 1 else path.name,
        path.name,
        path.stem,
    )


def lookup_label_entry(labels: dict[str, Any], series_name: str) -> Any | None:
    for key in canonical_label_keys(series_name):
        normalized = key.replace("\\", "/")
        if normalized in labels:
            return labels[normalized]
    for key, value in labels.items():
        if key.endswith(series_name) or key.endswith(Path(series_name).name):
            return value
    return None


def labels_from_entry(
    entry: Any,
    timestamps: pd.Series,
    series_name: str,
    missing_labels_as_normal: bool = False,
) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=np.int8)
    if entry is None:
        if missing_labels_as_normal or series_name.replace("\\", "/").startswith(
            "artificialNoAnomaly/"
        ):
            return labels
        raise ValueError(
            f"No labels found for {series_name}. Pass --missing-labels-as-normal only for "
            "series that are truly normal."
        )

    if isinstance(entry, dict) and "labels" in entry:
        entry = entry["labels"]

    if (
        isinstance(entry, list)
        and len(entry) == len(timestamps)
        and all(isinstance(value, int | bool | float) for value in entry)
    ):
        return np.asarray(entry, dtype=np.int8)

    if not isinstance(entry, list):
        raise ValueError(f"Unsupported label entry for {series_name}: expected a list.")

    ts = pd.to_datetime(timestamps)
    for window in entry:
        if isinstance(window, dict):
            start = window.get("start") or window.get("begin") or window.get("startTime")
            end = window.get("end") or window.get("stop") or window.get("endTime")
        else:
            if not isinstance(window, list | tuple) or len(window) != 2:
                raise ValueError(f"Unsupported anomaly window for {series_name}: {window!r}")
            start, end = window
        start_ts = pd.to_datetime(start, errors="raise")
        end_ts = pd.to_datetime(end, errors="raise")
        labels[(ts >= start_ts) & (ts <= end_ts)] = 1
    return labels


def labels_for_series(
    labels: dict[str, Any],
    series_name: str,
    timestamps: pd.Series,
    missing_labels_as_normal: bool = False,
) -> np.ndarray:
    return labels_from_entry(
        lookup_label_entry(labels, series_name),
        timestamps,
        series_name,
        missing_labels_as_normal=missing_labels_as_normal,
    )
