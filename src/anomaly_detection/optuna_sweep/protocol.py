from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.labels import labels_for_series
from anomaly_detection.metrics import contiguous_events
from anomaly_detection.optuna_sweep.types import SweepContext, TrainWindowInfo
from anomaly_detection.optuna_sweep.utils import save_json
from anomaly_detection.preprocessing import EPS, finite_array, robust_scale_stats
from anomaly_detection.protocol import select_training_window


def category_of(series_name: str) -> str:
    return series_name.split("/", 1)[0]


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    center = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - center)))


def lag1_autocorr(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 3 or float(np.nanstd(values)) < EPS:
        return None
    corr = np.corrcoef(values[:-1], values[1:])[0, 1]
    return None if not np.isfinite(corr) else float(corr)


def trend_proxy(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2:
        return None
    x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    try:
        slope = np.polyfit(x, values, 1)[0]
    except np.linalg.LinAlgError:
        return None
    return None if not np.isfinite(slope) else float(slope)


def series_metadata(
    dataset: dict[str, pd.DataFrame], label_windows: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = []
    for name, frame in dataset.items():
        values = frame["value"].to_numpy(dtype=np.float32)
        labels = labels_for_series(
            label_windows, name, frame["timestamp"], missing_labels_as_normal=True
        )
        events = contiguous_events(labels)
        metadata.append(
            {
                "category": category_of(name),
                "series": name,
                "n_points": int(len(frame)),
                "n_anomaly_points": int(labels.sum()),
                "anomaly_fraction": float(labels.mean()) if len(labels) else 0.0,
                "n_events": int(len(events)),
                "value_mean": float(np.nanmean(values)),
                "value_std": float(np.nanstd(values)),
                "value_mad": robust_mad(values),
                "missing_ratio": float(frame.isna().to_numpy().mean()),
                "lag1_autocorrelation": lag1_autocorr(values),
                "trend_proxy": trend_proxy(values),
            }
        )
    return metadata


def candidate_rank(
    item: dict[str, Any], category_medians: dict[str, dict[str, float]]
) -> tuple[float, float, str]:
    medians = category_medians[item["category"]]
    anomaly_delta = abs(item["anomaly_fraction"] - medians["anomaly_fraction"])
    length_delta = abs(math.log1p(item["n_points"]) - math.log1p(medians["n_points"]))
    return anomaly_delta, length_delta, item["series"]


def select_validation_subset(
    dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    subset_file: Path,
    validation_size: int,
    seed: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    if subset_file.is_file():
        payload = json.loads(subset_file.read_text(encoding="utf-8"))
        return list(payload["selected_series"]), list(payload["metadata"])

    metadata = series_metadata(dataset, label_windows)
    anomaly_candidates = [item for item in metadata if item["n_events"] > 0]
    candidates = anomaly_candidates or metadata
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        by_category.setdefault(item["category"], []).append(item)

    category_medians = {
        category: {
            "anomaly_fraction": float(np.median([item["anomaly_fraction"] for item in items])),
            "n_points": float(np.median([item["n_points"] for item in items])),
        }
        for category, items in by_category.items()
    }

    selected: list[dict[str, Any]] = []
    for category in sorted(by_category):
        if len(selected) >= validation_size:
            break
        ranked = sorted(
            by_category[category],
            key=lambda item: candidate_rank(item, category_medians),
        )
        selected.append(ranked[0])

    selected_names = {row["series"] for row in selected}
    remaining = [item for item in candidates if item["series"] not in selected_names]
    rng = np.random.default_rng(seed)
    for item in remaining:
        item["_diversity_jitter"] = float(rng.uniform(0.0, 1e-6))
    remaining.sort(
        key=lambda item: (
            -len(by_category[item["category"]]),
            -item["n_events"],
            item["_diversity_jitter"],
            item["series"],
        )
    )
    selected.extend(remaining[: max(0, validation_size - len(selected))])

    selected_names_list = [item["series"] for item in selected]
    clean_metadata = [
        {key: value for key, value in item.items() if not key.startswith("_")} for item in metadata
    ]
    payload = {
        "seed": seed,
        "validation_size_requested": validation_size,
        "selected_series": selected_names_list,
        "selection_strategy": "category representative plus deterministic diverse fill",
        "metadata": clean_metadata,
        "selected_metadata": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in selected
        ],
    }
    save_json(subset_file, payload)
    return selected_names_list, clean_metadata


def score_smoothing(scores: np.ndarray, window: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if window <= 1:
        return finite_array(scores)
    smoothed = pd.Series(scores).rolling(window=window, min_periods=1).mean().to_numpy(np.float32)
    return finite_array(smoothed)


def prepare_series_labels(context: SweepContext, name: str) -> np.ndarray:
    frame = context.dataset[name]
    return labels_for_series(
        context.labels,
        name,
        frame["timestamp"],
        missing_labels_as_normal=True,
    )


def training_window_info_for_series(context: SweepContext, name: str) -> TrainWindowInfo:
    frame = context.dataset[name]
    labels = prepare_series_labels(context, name)
    fraction = context.args.fallback_train_fraction
    try:
        window = select_training_window(
            n_points=len(frame),
            labels=labels,
            fallback_train_fraction=fraction,
            mode="before_first_anomaly",
            min_length=context.args.min_train_points,
            allow_contaminated=context.args.allow_contaminated_train,
        )
        train_end = int(window.end)
        return TrainWindowInfo(
            train_end=train_end,
            contaminated_train=bool(labels[:train_end].any()),
            used_fallback=False,
            reason="select_training_window",
        )
    except ValueError as exc:
        fallback = max(
            context.args.min_train_points,
            min(len(frame) - 1, int(math.ceil(len(frame) * fraction))),
        )
        if fallback <= 1 or fallback >= len(frame):
            raise
        contaminated = bool(labels[:fallback].any())
        if (
            contaminated
            and context.args.strict_clean_train
            and not context.args.allow_contaminated_train
        ):
            raise ValueError(
                f"No clean training prefix for {name}; "
                f"fallback train_end={fallback} is contaminated."
            ) from exc
        if contaminated and not context.args.allow_contaminated_train:
            warnings.warn(
                f"Using contaminated fallback training prefix for {name}: train_end={fallback}. "
                "Set --strict-clean-train to fail instead or "
                "--allow-contaminated-train to silence.",
                stacklevel=2,
            )
        return TrainWindowInfo(
            train_end=int(fallback),
            contaminated_train=contaminated,
            used_fallback=True,
            reason=f"fallback_after_error: {exc}",
        )


def fit_normalization(
    values: np.ndarray, train_end: int, method: str
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    if method == "none":
        return values.astype(np.float32, copy=True), {"method": "none"}
    train = values[:train_end]
    if method == "per-series":
        mean = float(np.nanmean(train))
        std = float(np.nanstd(train))
        std = 1.0 if not np.isfinite(std) or std < EPS else std
        return ((values - mean) / std).astype(np.float32), {
            "method": method,
            "mean": mean,
            "std": std,
        }
    if method == "robust":
        center = float(np.nanmedian(train))
        scale = robust_scale_stats(train).scale
        scale = 1.0 if not np.isfinite(scale) or scale < EPS else float(scale)
        return ((values - center) / scale).astype(np.float32), {
            "method": method,
            "center": center,
            "scale": scale,
        }
    raise ValueError(f"Unsupported normalization: {method}")
