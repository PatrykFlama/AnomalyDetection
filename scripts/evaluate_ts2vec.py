#!/usr/bin/env python3
"""Evaluate a trained TS2Vec anomaly detector on NAB time series.

Changes vs the original script:
- separates binary precision/recall/F1 from continuous ROC-AUC/PR-AUC;
- computes event-wise aggregate per series instead of on one concatenated pseudo-series;
- adds threshold-sweep diagnostics for point-wise and event-wise F1;
- adds less brittle TS2Vec downstream scores: centroid and kNN distance to normal prefix;
- keeps the original mask-difference score available as --score-method mask-diff;
- uses fairer event-AUC negative examples by default: fixed normal windows, not whole normal segments;
- adds one-to-one event matching so one long predicted event cannot detect many true events for free.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.data import DATASET_DIR, load_dataset  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "ts2vec"
DEFAULT_LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
DEFAULT_LABELS_DOWNLOAD_PATH = DATASET_DIR / "labels" / "combined_windows.json"
DEFAULT_TS2VEC_PATHS = (
    PROJECT_ROOT / "src" / "anomaly_detection" / "vendor" / "ts2vec",
    PROJECT_ROOT / "vendor" / "ts2vec",
    PROJECT_ROOT / "ts2vec",
    PROJECT_ROOT.parent / "ts2vec",
)
DEFAULT_LABEL_PATHS = (
    DATASET_DIR / "labels" / "combined_windows.json",
    DATASET_DIR / "combined_windows.json",
    PROJECT_ROOT / "data" / "nab_labels" / "combined_windows.json",
    PROJECT_ROOT / "labels" / "combined_windows.json",
)
EPS = 1e-8


@dataclass(frozen=True)
class SeriesResult:
    name: str
    n_points: int
    n_anomaly_points: int
    n_true_events: int
    n_pred_events: int
    threshold: float
    pointwise: dict[str, float | int | None]
    eventwise: dict[str, float | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained TS2Vec model, score NAB time series, and report point-wise "
            "and event-wise precision, recall, F1, ROC-AUC, and PR-AUC."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--series-metadata-path", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--no-download-labels", action="store_true")
    parser.add_argument("--missing-labels-as-normal", action="store_true")
    parser.add_argument("--ts2vec-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256, help="TS2Vec encode batch size.")
    parser.add_argument("--sliding-length", type=int, default=32, help="TS2Vec sliding inference length.")
    parser.add_argument("--sliding-padding", type=int, default=200, help="TS2Vec sliding padding.")
    parser.add_argument(
        "--score-method",
        choices=("knn", "centroid", "mask-diff"),
        default="knn",
        help=(
            "How to convert TS2Vec representations into anomaly scores. 'knn' and 'centroid' "
            "compare each point representation to the normal training prefix. 'mask-diff' "
            "keeps the original masked-vs-unmasked representation difference."
        ),
    )
    parser.add_argument("--knn-k", type=int, default=5, help="k for --score-method knn.")
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.15,
        help=(
            "Fallback normal-prefix fraction used for centroid/kNN scoring when series.json does "
            "not contain train_length for a series."
        ),
    )
    parser.add_argument(
        "--score-adjust-window",
        type=int,
        default=0,
        help=(
            "Optional past rolling-mean window used to normalize raw scores. Defaults to 0 "
            "because rolling relative normalization can distort ranking/AUC diagnostics."
        ),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Global score quantile used as threshold when --threshold is omitted.",
    )
    parser.add_argument(
        "--event-overlap-policy",
        choices=("one-to-one", "any", "point-adjust"),
        default="one-to-one",
        help=(
            "Event-wise binary metric policy. 'one-to-one' is the fairest default: each "
            "predicted event can match at most one true event and each true event can be "
            "detected at most once. 'any' is the more permissive legacy policy. "
            "'point-adjust' expands detected true events for point-wise-style scoring."
        ),
    )
    parser.add_argument(
        "--event-min-overlap-points",
        type=int,
        default=1,
        help="Minimum number of overlapping points required for a predicted/true event match.",
    )
    parser.add_argument(
        "--event-min-true-overlap-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of the true event that must be covered by the predicted event. "
            "Use e.g. 0.1 to require at least 10%% coverage of the anomaly window."
        ),
    )
    parser.add_argument(
        "--event-min-pred-overlap-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum fraction of the predicted event that must overlap a true event. "
            "Use e.g. 0.1 to penalize very long alerts that barely touch an anomaly."
        ),
    )
    parser.add_argument(
        "--event-auc-negative-policy",
        choices=("fixed-windows", "normal-segments"),
        default="fixed-windows",
        help=(
            "How to construct negative examples for event-wise AUC. fixed-windows is usually "
            "fairer than taking max score over entire long normal segments."
        ),
    )
    parser.add_argument(
        "--event-auc-window-length",
        type=int,
        default=None,
        help="Normal-window length for event-wise AUC. Defaults to median true-event length.",
    )
    parser.add_argument("--threshold-sweep-steps", type=int, default=200)
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    if args.run_dir is None and args.model_path is None:
        raise SystemExit("Provide --run-dir or --model-path.")

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    model_path = args.model_path or run_dir / "ts2vec_model.pt"
    metadata_path = args.metadata_path or (run_dir / "metadata.json" if run_dir else None)
    series_metadata_path = args.series_metadata_path or (run_dir / "series.json" if run_dir else None)
    return model_path.expanduser().resolve(), metadata_path, series_metadata_path


def resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def import_ts2vec(ts2vec_dir: Path | None) -> tuple[type, Path | None]:
    candidates = [ts2vec_dir.expanduser().resolve()] if ts2vec_dir is not None else []
    candidates.extend(path for path in DEFAULT_TS2VEC_PATHS if path not in candidates)

    for candidate in candidates:
        if (candidate / "ts2vec.py").is_file():
            sys.path.insert(0, str(candidate))
            from ts2vec import TS2Vec

            return TS2Vec, candidate

    try:
        from ts2vec import TS2Vec

        return TS2Vec, None
    except ImportError as exc:
        searched = "\n  - ".join(str(path) for path in candidates)
        raise SystemExit(
            "Could not import the official TS2Vec implementation. Pass --ts2vec-dir vendor/ts2vec.\n"
            f"Searched:\n  - {searched}"
        ) from exc


def download_default_labels() -> Path:
    destination = DEFAULT_LABELS_DOWNLOAD_PATH.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NAB labels from {DEFAULT_LABELS_URL} to {destination}.")
    try:
        urllib.request.urlretrieve(DEFAULT_LABELS_URL, destination)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            "Could not download official NAB labels automatically. Download combined_windows.json "
            "manually and pass it with --labels-file.\n"
            f"URL: {DEFAULT_LABELS_URL}\nError: {exc}"
        ) from exc
    return destination


def find_labels_file(explicit_path: Path | None, auto_download: bool) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Labels file does not exist: {path}")
        return path

    for path in DEFAULT_LABEL_PATHS:
        if path.is_file():
            return path.resolve()

    if auto_download:
        return download_default_labels()

    searched = "\n  - ".join(str(path) for path in DEFAULT_LABEL_PATHS)
    raise SystemExit(
        "Could not find NAB anomaly window labels. Download or copy combined_windows.json "
        "and pass it with --labels-file, or omit --no-download-labels.\n"
        f"Official URL: {DEFAULT_LABELS_URL}\nSearched:\n  - {searched}"
    )


def make_output_dir(output_dir: Path | None, run_dir: Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = run_dir / "evaluation" if run_dir is not None else DEFAULT_OUTPUT_ROOT
        output_dir = base / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def model_kwargs_from_metadata(metadata: dict[str, Any], device: str, batch_size: int) -> dict[str, Any]:
    train_args = metadata.get("args", {})
    return {
        "input_dims": 1,
        "output_dims": int(train_args.get("output_dims", 320)),
        "hidden_dims": int(train_args.get("hidden_dims", 64)),
        "depth": int(train_args.get("depth", 10)),
        "device": device,
        "batch_size": batch_size,
    }


def build_series_metadata_map(series_metadata: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if series_metadata is None:
        return {}
    return {item["name"]: item for item in series_metadata}


def normalize_values(
    name: str,
    values: np.ndarray,
    normalization: str,
    series_metadata: dict[str, dict[str, Any]],
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if normalization == "none":
        return values

    stats = series_metadata.get(name, {}).get("normalization")
    if stats is not None and "mean" in stats and "std" in stats:
        mean = float(stats["mean"])
        std = float(stats["std"])
    else:
        # Fallback for old runs: avoid using the full test series if possible.
        fallback_len = max(2, int(np.ceil(len(values) * 0.15)))
        prefix = values[:fallback_len]
        mean = float(prefix.mean())
        std = float(prefix.std())
    std = 1.0 if std == 0 else std
    return (values - mean) / std


def reference_length_for_series(
    name: str,
    n_points: int,
    series_metadata: dict[str, dict[str, Any]],
    reference_fraction: float,
) -> int:
    if name in series_metadata and "train_length" in series_metadata[name]:
        return max(2, min(n_points, int(series_metadata[name]["train_length"])))
    if not 0 < reference_fraction <= 1:
        raise SystemExit("--reference-fraction must be in the interval (0, 1].")
    return max(2, min(n_points, int(np.ceil(n_points * reference_fraction))))


def canonical_label_keys(series_name: str) -> tuple[str, ...]:
    path = Path(series_name)
    return (
        series_name,
        series_name.replace("\\", "/"),
        f"{path.parts[0]}/{path.name}" if len(path.parts) > 1 else path.name,
        path.name,
        path.stem,
    )


def load_label_windows(labels_file: Path) -> dict[str, Any]:
    data = load_json(labels_file)
    if isinstance(data, dict) and "windows" in data and isinstance(data["windows"], dict):
        data = data["windows"]
    if not isinstance(data, dict):
        raise SystemExit("Labels file must contain a JSON object mapping series names to windows.")
    return data


def lookup_label_entry(labels: dict[str, Any], series_name: str) -> Any | None:
    normalized = {str(key).replace("\\", "/"): value for key, value in labels.items()}
    for key in canonical_label_keys(series_name):
        if key in normalized:
            return normalized[key]
    for key, value in normalized.items():
        if key.endswith(series_name) or key.endswith(Path(series_name).name):
            return value
    return None


def labels_from_entry(
    entry: Any,
    timestamps: pd.Series,
    series_name: str,
    missing_labels_as_normal: bool,
) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=np.int8)
    if entry is None:
        if missing_labels_as_normal:
            return labels
        raise SystemExit(
            f"No labels found for {series_name}. Pass a labels file with matching keys or use "
            "--missing-labels-as-normal if this series is truly normal."
        )

    if isinstance(entry, dict) and "labels" in entry:
        entry = entry["labels"]

    if (
        isinstance(entry, list)
        and len(entry) == len(timestamps)
        and all(isinstance(value, (int, bool, float)) for value in entry)
    ):
        return np.asarray(entry, dtype=np.int8)

    if not isinstance(entry, list):
        raise SystemExit(f"Unsupported label entry for {series_name}: expected list of windows.")

    ts = pd.to_datetime(timestamps)
    for window in entry:
        if isinstance(window, dict):
            start = window.get("start") or window.get("begin") or window.get("startTime")
            end = window.get("end") or window.get("stop") or window.get("endTime")
        else:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise SystemExit(f"Unsupported anomaly window for {series_name}: {window!r}")
            start, end = window
        start_ts = pd.to_datetime(start, errors="raise")
        end_ts = pd.to_datetime(end, errors="raise")
        labels[(ts >= start_ts) & (ts <= end_ts)] = 1
    return labels


def contiguous_events(binary: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return []
    changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def safe_auc(metric_name: str, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    if metric_name == "roc_auc":
        return float(roc_auc_score(labels, scores))
    if metric_name == "pr_auc":
        return float(average_precision_score(labels, scores))
    raise ValueError(metric_name)


def confusion_counts(labels: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    labels_bool = labels.astype(bool)
    pred_bool = pred.astype(bool)
    return {
        "tp": int(np.logical_and(labels_bool, pred_bool).sum()),
        "tn": int(np.logical_and(~labels_bool, ~pred_bool).sum()),
        "fp": int(np.logical_and(~labels_bool, pred_bool).sum()),
        "fn": int(np.logical_and(labels_bool, ~pred_bool).sum()),
    }


def point_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **counts}


def binary_point_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **confusion_counts(labels, pred),
        "support": int(labels.sum()),
    }


def continuous_point_auc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    return {
        "roc_auc": safe_auc("roc_auc", labels, scores),
        "pr_auc": safe_auc("pr_auc", labels, scores),
    }


def overlap_count(primary: list[tuple[int, int]], secondary: list[tuple[int, int]]) -> int:
    count = 0
    for start, end in primary:
        if any(start < other_end and end > other_start for other_start, other_end in secondary):
            count += 1
    return count


def point_adjust_predictions(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    adjusted = pred.astype(np.int8, copy=True)
    for start, end in contiguous_events(labels):
        adjusted[start:end] = 1 if pred[start:end].any() else 0
    return adjusted


def event_overlap_details(
    true_event: tuple[int, int],
    pred_event: tuple[int, int],
) -> tuple[int, float, float]:
    """Return overlap points, true-event coverage, and predicted-event purity."""
    true_start, true_end = true_event
    pred_start, pred_end = pred_event
    overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    true_len = max(1, true_end - true_start)
    pred_len = max(1, pred_end - pred_start)
    return overlap, overlap / true_len, overlap / pred_len


def match_qualifies(
    true_event: tuple[int, int],
    pred_event: tuple[int, int],
    min_overlap_points: int,
    min_true_overlap_fraction: float,
    min_pred_overlap_fraction: float,
) -> tuple[bool, int, float, float]:
    overlap, true_fraction, pred_fraction = event_overlap_details(true_event, pred_event)
    qualifies = (
        overlap >= min_overlap_points
        and true_fraction >= min_true_overlap_fraction
        and pred_fraction >= min_pred_overlap_fraction
    )
    return qualifies, overlap, true_fraction, pred_fraction


def any_overlap_event_counts(
    labels: np.ndarray,
    pred: np.ndarray,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
) -> dict[str, int]:
    """Legacy permissive counts: one event may match many events on the other side."""
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)
    true_detected = 0
    for true_event in true_events:
        if any(
            match_qualifies(
                true_event,
                pred_event,
                min_overlap_points,
                min_true_overlap_fraction,
                min_pred_overlap_fraction,
            )[0]
            for pred_event in pred_events
        ):
            true_detected += 1
    pred_matched = 0
    for pred_event in pred_events:
        if any(
            match_qualifies(
                true_event,
                pred_event,
                min_overlap_points,
                min_true_overlap_fraction,
                min_pred_overlap_fraction,
            )[0]
            for true_event in true_events
        ):
            pred_matched += 1
    return {
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "true_detected_events": true_detected,
        "pred_matched_events": pred_matched,
        "matched_event_pairs": min(true_detected, pred_matched),
        "fp_events": len(pred_events) - pred_matched,
        "fn_events": len(true_events) - true_detected,
    }


def one_to_one_event_counts(
    labels: np.ndarray,
    pred: np.ndarray,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
) -> dict[str, int]:
    """Fair event counts with greedy one-to-one matching by largest overlap.

    A predicted event can match at most one true event, and a true event can be
    detected at most once. This prevents long alerts from covering many anomaly
    windows and receiving perfect precision/recall for free.
    """
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)
    candidate_pairs: list[tuple[int, float, float, int, int]] = []
    for true_idx, true_event in enumerate(true_events):
        for pred_idx, pred_event in enumerate(pred_events):
            qualifies, overlap, true_fraction, pred_fraction = match_qualifies(
                true_event,
                pred_event,
                min_overlap_points,
                min_true_overlap_fraction,
                min_pred_overlap_fraction,
            )
            if qualifies:
                # Sort primarily by overlap points, then by true coverage and predicted purity.
                candidate_pairs.append((overlap, true_fraction, pred_fraction, true_idx, pred_idx))

    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    for _overlap, _true_fraction, _pred_fraction, true_idx, pred_idx in sorted(
        candidate_pairs,
        reverse=True,
    ):
        if true_idx in matched_true or pred_idx in matched_pred:
            continue
        matched_true.add(true_idx)
        matched_pred.add(pred_idx)

    tp_events = len(matched_true)
    return {
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "true_detected_events": tp_events,
        "pred_matched_events": tp_events,
        "matched_event_pairs": tp_events,
        "fp_events": len(pred_events) - tp_events,
        "fn_events": len(true_events) - tp_events,
    }


def event_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp_events = counts.get("matched_event_pairs", min(counts["true_detected_events"], counts["pred_matched_events"]))
    precision = tp_events / counts["pred_events"] if counts["pred_events"] else 0.0
    recall = tp_events / counts["true_events"] if counts["true_events"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fp_events = counts.get("fp_events", counts["pred_events"] - tp_events)
    fn_events = counts.get("fn_events", counts["true_events"] - tp_events)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_events": tp_events,
        "fp_events": fp_events,
        "fn_events": fn_events,
        **counts,
    }


def event_binary_metrics(
    labels: np.ndarray,
    pred: np.ndarray,
    policy: str,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
) -> dict[str, float | int | None]:
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)

    if policy == "point-adjust":
        adjusted = point_adjust_predictions(labels, pred)
        metrics = binary_point_metrics(labels, adjusted)
        metrics["true_events"] = len(true_events)
        metrics["pred_events"] = len(pred_events)
        return metrics

    if policy == "one-to-one":
        counts = one_to_one_event_counts(
            labels,
            pred,
            min_overlap_points,
            min_true_overlap_fraction,
            min_pred_overlap_fraction,
        )
    elif policy == "any":
        counts = any_overlap_event_counts(
            labels,
            pred,
            min_overlap_points,
            min_true_overlap_fraction,
            min_pred_overlap_fraction,
        )
    else:
        raise ValueError(policy)
    return event_metrics_from_counts(counts)


def normal_segments_from_labels(labels: np.ndarray) -> list[tuple[int, int]]:
    return contiguous_events(1 - labels.astype(np.int8))


def default_event_auc_window_length(labels_list: Iterable[np.ndarray]) -> int:
    lengths = []
    for labels in labels_list:
        lengths.extend(end - start for start, end in contiguous_events(labels))
    if not lengths:
        return 1
    return max(1, int(np.median(lengths)))


def event_auc_examples(
    labels: np.ndarray,
    scores: np.ndarray,
    negative_policy: str,
    window_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []

    for start, end in contiguous_events(labels):
        y_true.append(1)
        y_score.append(float(np.nanmax(scores[start:end])))

    if negative_policy == "normal-segments":
        for start, end in normal_segments_from_labels(labels):
            if end > start:
                y_true.append(0)
                y_score.append(float(np.nanmax(scores[start:end])))
    else:
        step = max(1, window_length)
        min_tail = max(1, step // 2)
        for seg_start, seg_end in normal_segments_from_labels(labels):
            start = seg_start
            while start < seg_end:
                end = min(seg_end, start + step)
                if end - start >= min_tail:
                    y_true.append(0)
                    y_score.append(float(np.nanmax(scores[start:end])))
                start += step

    return np.asarray(y_true, dtype=np.int8), np.asarray(y_score, dtype=np.float64)


def event_auc_metrics_from_examples(event_labels: np.ndarray, event_scores: np.ndarray) -> dict[str, float | None]:
    return {
        "roc_auc": safe_auc("roc_auc", event_labels, event_scores),
        "pr_auc": safe_auc("pr_auc", event_labels, event_scores),
    }


def shifted_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return np.ones_like(values, dtype=np.float32)
    series = pd.Series(values)
    mean = series.rolling(window=window, min_periods=1).mean().shift(1)
    mean = mean.bfill().fillna(float(np.nanmean(values) if values.size else 1.0))
    return mean.to_numpy(dtype=np.float32)


def adjusted_scores(raw_scores: np.ndarray, window: int) -> np.ndarray:
    raw_scores = raw_scores.astype(np.float32, copy=False)
    if window <= 0:
        return np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    baseline = shifted_rolling_mean(raw_scores, window)
    scores = (raw_scores - baseline) / (np.abs(baseline) + EPS)
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def encode_representations(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    sliding_length: int,
    sliding_padding: int,
) -> np.ndarray:
    data = values.reshape(1, -1, 1).astype(np.float32)
    encoded = model.encode(
        data,
        causal=True,
        sliding_length=sliding_length,
        sliding_padding=sliding_padding,
        batch_size=batch_size,
    ).squeeze(0)
    return np.asarray(encoded[: len(values)], dtype=np.float32)


def score_mask_difference(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    sliding_length: int,
    sliding_padding: int,
) -> np.ndarray:
    data = values.reshape(1, -1, 1).astype(np.float32)
    repr_masked = model.encode(
        data,
        mask="mask_last",
        causal=True,
        sliding_length=sliding_length,
        sliding_padding=sliding_padding,
        batch_size=batch_size,
    ).squeeze(0)
    repr_unmasked = model.encode(
        data,
        causal=True,
        sliding_length=sliding_length,
        sliding_padding=sliding_padding,
        batch_size=batch_size,
    ).squeeze(0)
    return np.abs(repr_unmasked - repr_masked).sum(axis=1)[: len(values)]


def score_centroid(embeddings: np.ndarray, reference_length: int) -> np.ndarray:
    reference = embeddings[:reference_length]
    centroid = np.nanmean(reference, axis=0, keepdims=True)
    return np.linalg.norm(embeddings - centroid, axis=1)


def score_knn(embeddings: np.ndarray, reference_length: int, k: int, chunk_size: int = 4096) -> np.ndarray:
    reference = embeddings[:reference_length].astype(np.float32, copy=False)
    k = max(1, min(k, len(reference)))
    output = np.empty(len(embeddings), dtype=np.float32)
    ref_norm = np.sum(reference * reference, axis=1, keepdims=True).T
    for start in range(0, len(embeddings), chunk_size):
        end = min(len(embeddings), start + chunk_size)
        chunk = embeddings[start:end].astype(np.float32, copy=False)
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        d2 = chunk_norm + ref_norm - 2.0 * chunk @ reference.T
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        output[start:end] = np.sqrt(np.mean(nearest, axis=1))
    return output


def score_series(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    sliding_length: int,
    sliding_padding: int,
    score_adjust_window: int,
    score_method: str,
    reference_length: int,
    knn_k: int,
) -> np.ndarray:
    if score_method == "mask-diff":
        raw_scores = score_mask_difference(model, values, batch_size, sliding_length, sliding_padding)
    else:
        embeddings = encode_representations(model, values, batch_size, sliding_length, sliding_padding)
        if score_method == "centroid":
            raw_scores = score_centroid(embeddings, reference_length)
        elif score_method == "knn":
            raw_scores = score_knn(embeddings, reference_length, knn_k)
        else:
            raise ValueError(score_method)
    return adjusted_scores(raw_scores, score_adjust_window)


def choose_threshold(scores: np.ndarray, threshold: float | None, threshold_quantile: float) -> float:
    if threshold is not None:
        return float(threshold)
    if not 0 <= threshold_quantile <= 1:
        raise SystemExit("--threshold-quantile must be in [0, 1].")
    return float(np.quantile(scores, threshold_quantile))


def threshold_grid(scores: np.ndarray, steps: int) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0], dtype=np.float64)
    if steps <= 1:
        return np.array([float(np.median(finite))], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, steps)
    return np.unique(np.quantile(finite, quantiles))


def sweep_point_f1(labels: np.ndarray, scores: np.ndarray, steps: int) -> dict[str, float | int | None]:
    best: dict[str, float | int | None] = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in threshold_grid(scores, steps):
        pred = (scores >= threshold).astype(np.int8)
        metrics = binary_point_metrics(labels, pred)
        if float(metrics["f1"]) > float(best["f1"]):
            best = {"threshold": float(threshold), **metrics}
    return best


def sweep_event_f1(
    series_scores_labels: list[tuple[np.ndarray, np.ndarray]],
    steps: int,
    policy: str,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
) -> dict[str, float | int | None]:
    all_scores = np.concatenate([scores for scores, _ in series_scores_labels])
    best: dict[str, float | int | None] = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in threshold_grid(all_scores, steps):
        if policy == "point-adjust":
            counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
            true_events = pred_events = 0
            for scores, labels in series_scores_labels:
                pred = (scores >= threshold).astype(np.int8)
                adjusted = point_adjust_predictions(labels, pred)
                cc = confusion_counts(labels, adjusted)
                for key in counts:
                    counts[key] += cc[key]
                true_events += len(contiguous_events(labels))
                pred_events += len(contiguous_events(pred))
            metrics = {**point_metrics_from_counts(counts), "true_events": true_events, "pred_events": pred_events}
        else:
            counts = {
                "true_events": 0,
                "pred_events": 0,
                "true_detected_events": 0,
                "pred_matched_events": 0,
                "matched_event_pairs": 0,
                "fp_events": 0,
                "fn_events": 0,
            }
            for scores, labels in series_scores_labels:
                pred = (scores >= threshold).astype(np.int8)
                if policy == "one-to-one":
                    ec = one_to_one_event_counts(
                        labels,
                        pred,
                        min_overlap_points,
                        min_true_overlap_fraction,
                        min_pred_overlap_fraction,
                    )
                elif policy == "any":
                    ec = any_overlap_event_counts(
                        labels,
                        pred,
                        min_overlap_points,
                        min_true_overlap_fraction,
                        min_pred_overlap_fraction,
                    )
                else:
                    raise ValueError(policy)
                for key in counts:
                    counts[key] += ec[key]
            metrics = event_metrics_from_counts(counts)
        if float(metrics["f1"]) > float(best["f1"]):
            best = {"threshold": float(threshold), **metrics}
    return best


def aggregate_metric_dict(metric_dicts: list[dict[str, float | int | None]]) -> dict[str, float | None]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    output: dict[str, float | None] = {}
    for key in keys:
        values = [metrics[key] for metrics in metric_dicts if metrics.get(key) is not None]
        if not values:
            output[key] = None
        elif key.endswith("events") or key in {"support", "tp", "tn", "fp", "fn"}:
            output[key] = float(sum(float(value) for value in values))
        else:
            output[key] = float(np.mean(values))
    return output


def main() -> None:
    args = parse_args()
    model_path, metadata_path, series_metadata_path = resolve_run_paths(args)
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None

    metadata = load_json(metadata_path) if metadata_path is not None and metadata_path.is_file() else {}
    series_metadata = (
        load_json(series_metadata_path)
        if series_metadata_path is not None and series_metadata_path.is_file()
        else None
    )
    series_metadata_map = build_series_metadata_map(series_metadata)

    dataset_dir = args.dataset_dir or Path(metadata.get("dataset_dir", DATASET_DIR))
    categories = args.categories or metadata.get("categories")
    normalization = metadata.get("normalization", "per-series")

    TS2Vec, ts2vec_source = import_ts2vec(args.ts2vec_dir)
    device = resolve_device(args.device)
    model = TS2Vec(**model_kwargs_from_metadata(metadata, device, args.batch_size))
    model.load(str(model_path))

    labels_file = find_labels_file(args.labels_file, auto_download=not args.no_download_labels)
    label_windows = load_label_windows(labels_file)
    dataset = load_dataset(dataset_dir, categories)
    if args.limit_series is not None:
        if args.limit_series <= 0:
            raise SystemExit("--limit-series must be positive.")
        dataset = dict(list(dataset.items())[: args.limit_series])
    output_dir = make_output_dir(args.output_dir, run_dir)

    score_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    print(f"Loaded model from {model_path}.")
    print(f"Loaded labels from {labels_file}.")
    print(f"Scoring {len(dataset)} series on {device}; outputs will be written to {output_dir}.")

    for name, frame in dataset.items():
        values = frame["value"].to_numpy(dtype=np.float32)
        normalized = normalize_values(name, values, normalization, series_metadata_map)
        reference_length = reference_length_for_series(
            name, len(normalized), series_metadata_map, args.reference_fraction
        )
        scores = score_series(
            model=model,
            values=normalized,
            batch_size=args.batch_size,
            sliding_length=args.sliding_length,
            sliding_padding=args.sliding_padding,
            score_adjust_window=args.score_adjust_window,
            score_method=args.score_method,
            reference_length=reference_length,
            knn_k=args.knn_k,
        )
        labels = labels_from_entry(
            lookup_label_entry(label_windows, name),
            frame["timestamp"],
            name,
            args.missing_labels_as_normal,
        )
        if len(scores) != len(labels):
            raise SystemExit(f"Score/label length mismatch for {name}: {len(scores)} != {len(labels)}")
        all_scores.append(scores)
        all_labels.append(labels)
        score_cache[name] = (scores, labels, frame["timestamp"].to_numpy(), reference_length)

    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels)
    threshold = choose_threshold(flat_scores, args.threshold, args.threshold_quantile)
    flat_pred = (flat_scores >= threshold).astype(np.int8)

    pointwise = binary_point_metrics(flat_labels, flat_pred)
    pointwise.update(continuous_point_auc(flat_labels, flat_scores))

    event_auc_window_length = args.event_auc_window_length or default_event_auc_window_length(all_labels)
    per_series_payload: list[dict[str, Any]] = []
    aggregate_event_counts = {
        "true_events": 0,
        "pred_events": 0,
        "true_detected_events": 0,
        "pred_matched_events": 0,
        "matched_event_pairs": 0,
        "fp_events": 0,
        "fn_events": 0,
    }
    aggregate_point_adjust_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    aggregate_point_adjust_true_events = 0
    aggregate_point_adjust_pred_events = 0
    event_auc_labels: list[np.ndarray] = []
    event_auc_scores: list[np.ndarray] = []

    for name, (scores, labels, timestamps, reference_length) in score_cache.items():
        pred = (scores >= threshold).astype(np.int8)
        point_metrics = binary_point_metrics(labels, pred)
        point_metrics.update(continuous_point_auc(labels, scores))
        event_metrics = event_binary_metrics(
            labels,
            pred,
            args.event_overlap_policy,
            args.event_min_overlap_points,
            args.event_min_true_overlap_fraction,
            args.event_min_pred_overlap_fraction,
        )

        if args.event_overlap_policy == "point-adjust":
            adjusted = point_adjust_predictions(labels, pred)
            cc = confusion_counts(labels, adjusted)
            for key in aggregate_point_adjust_counts:
                aggregate_point_adjust_counts[key] += cc[key]
            aggregate_point_adjust_true_events += len(contiguous_events(labels))
            aggregate_point_adjust_pred_events += len(contiguous_events(pred))
        else:
            if args.event_overlap_policy == "one-to-one":
                ec = one_to_one_event_counts(
                    labels,
                    pred,
                    args.event_min_overlap_points,
                    args.event_min_true_overlap_fraction,
                    args.event_min_pred_overlap_fraction,
                )
            elif args.event_overlap_policy == "any":
                ec = any_overlap_event_counts(
                    labels,
                    pred,
                    args.event_min_overlap_points,
                    args.event_min_true_overlap_fraction,
                    args.event_min_pred_overlap_fraction,
                )
            else:
                raise ValueError(args.event_overlap_policy)
            for key in aggregate_event_counts:
                aggregate_event_counts[key] += ec[key]

        ey, es = event_auc_examples(
            labels,
            scores,
            args.event_auc_negative_policy,
            event_auc_window_length,
        )
        event_auc_labels.append(ey)
        event_auc_scores.append(es)
        event_metrics.update(event_auc_metrics_from_examples(ey, es))

        per_series_payload.append(
            SeriesResult(
                name=name,
                n_points=int(len(labels)),
                n_anomaly_points=int(labels.sum()),
                n_true_events=len(contiguous_events(labels)),
                n_pred_events=len(contiguous_events(pred)),
                threshold=threshold,
                pointwise=point_metrics,
                eventwise=event_metrics,
            ).__dict__
        )
        per_series_payload[-1]["reference_length"] = int(reference_length)

        if args.save_scores:
            safe_name = name.replace("/", "__").replace(".csv", "")
            np.savez_compressed(
                output_dir / f"scores__{safe_name}.npz",
                timestamps=timestamps.astype(str),
                scores=scores,
                labels=labels,
                predictions=pred,
            )

    if args.event_overlap_policy == "point-adjust":
        eventwise = {
            **point_metrics_from_counts(aggregate_point_adjust_counts),
            "support": int(sum(labels.sum() for labels in all_labels)),
            "true_events": aggregate_point_adjust_true_events,
            "pred_events": aggregate_point_adjust_pred_events,
        }
    else:
        eventwise = event_metrics_from_counts(aggregate_event_counts)

    all_event_labels = np.concatenate(event_auc_labels) if event_auc_labels else np.array([], dtype=np.int8)
    all_event_scores = np.concatenate(event_auc_scores) if event_auc_scores else np.array([], dtype=np.float64)
    eventwise.update(event_auc_metrics_from_examples(all_event_labels, all_event_scores))

    series_scores_labels = [(scores, labels) for scores, labels in zip(all_scores, all_labels, strict=True)]
    threshold_diagnostics = {
        "pointwise_best_f1": sweep_point_f1(flat_labels, flat_scores, args.threshold_sweep_steps),
        "eventwise_best_f1": sweep_event_f1(
            series_scores_labels,
            args.threshold_sweep_steps,
            args.event_overlap_policy,
            args.event_min_overlap_points,
            args.event_min_true_overlap_fraction,
            args.event_min_pred_overlap_fraction,
        ),
    }

    metrics = {
        "model_path": str(model_path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "series_metadata_path": None if series_metadata_path is None else str(series_metadata_path),
        "dataset_dir": str(dataset_dir.expanduser().resolve()),
        "labels_file": str(labels_file),
        "ts2vec_source": None if ts2vec_source is None else str(ts2vec_source),
        "device": device,
        "categories": categories,
        "normalization": normalization,
        "score_method": args.score_method,
        "knn_k": args.knn_k if args.score_method == "knn" else None,
        "sliding_length": args.sliding_length,
        "sliding_padding": args.sliding_padding,
        "score_adjust_window": args.score_adjust_window,
        "threshold": threshold,
        "threshold_quantile": args.threshold_quantile if args.threshold is None else None,
        "event_overlap_policy": args.event_overlap_policy,
        "event_min_overlap_points": args.event_min_overlap_points,
        "event_min_true_overlap_fraction": args.event_min_true_overlap_fraction,
        "event_min_pred_overlap_fraction": args.event_min_pred_overlap_fraction,
        "event_auc_negative_policy": args.event_auc_negative_policy,
        "event_auc_window_length": event_auc_window_length,
        "aggregate": {"pointwise": pointwise, "eventwise": eventwise},
        "threshold_diagnostics": threshold_diagnostics,
        "macro_average": {
            "pointwise": aggregate_metric_dict([item["pointwise"] for item in per_series_payload]),
            "eventwise": aggregate_metric_dict([item["eventwise"] for item in per_series_payload]),
        },
        "per_series": per_series_payload,
    }
    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(per_series_payload).to_json(
        output_dir / "per_series_metrics.jsonl", orient="records", lines=True
    )

    print(json.dumps(metrics["aggregate"], indent=2))
    print("Threshold diagnostics:")
    print(json.dumps(metrics["threshold_diagnostics"], indent=2))
    print(f"Saved metrics to {output_dir / 'metrics.json'}.")


if __name__ == "__main__":
    main()
