#!/usr/bin/env python3
"""Optuna sweeps and final NAB evaluation with a fixed, explicit protocol.

This script has two modes:

1. sweep
   Selects or reuses a frozen validation subset and runs Optuna only on
   model-specific hyperparameters. Shared protocol parameters such as the
   training window, preprocessing normalization, thresholding, score smoothing,
   and event matching are CLI/config values, not sampled Optuna parameters.

2. final
   Loads an exact params JSON and evaluates the chosen model with the same fixed
   protocol on the requested final split. The default final split is ``all``:
   every loaded NAB series is used. Local statistical baselines are calibrated
   per series, while trainable models are trained globally on pooled normal-prefix
   data from all selected series. Metrics are computed after each series'
   ``train_end``.

W&B directories are controlled by standard environment variables when needed:
WANDB_DIR, WANDB_DATA_DIR, WANDB_CACHE_DIR, and WANDB_CONFIG_DIR.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.data import DATASET_DIR, load_dataset  # noqa: E402
from anomaly_detection.detectors import (  # noqa: E402
    IsolationForestDetector,
    MLPAutoencoderDetector,
    RNNPredictorDetector,
    RobustZScoreDetector,
)
from anomaly_detection.labels import (  # noqa: E402
    find_labels_file,
    labels_for_series,
    load_label_windows,
)
from anomaly_detection.metrics import (  # noqa: E402
    SeriesScores,
    choose_threshold,
    contiguous_events,
    event_counts,
    event_metrics_from_counts,
    macro_average,
    point_metrics_from_predictions,
    predict,
    safe_auc,
    sweep_event_f1,
    sweep_point_f1,
)
from anomaly_detection.preprocessing import (  # noqa: E402
    EPS,
    apply_scale,
    finite_array,
    robust_scale_stats,
    rolling_mad,
    shifted_rolling,
    standard_scale_stats,
)
from anomaly_detection.protocol import select_training_window  # noqa: E402
from anomaly_detection.ts2vec_support import import_ts2vec, resolve_torch_device  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover - dependency is optional at import time
    wandb = None

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optuna"
DEFAULT_FINAL_OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_from_optuna"
MODEL_CHOICES = (
    "robust_zscore",
    "rolling_residual",
    "isolation_forest",
    "mlp_autoencoder",
    "lstm",
    "gru",
    "ts2vec",
)
DEFAULT_OBJECTIVE = "macro_average.eventwise.f1"
PROTOCOL_PARAM_KEYS = {
    "train_fraction",
    "normalization",
    "threshold_quantile",
    "min_overlap_points",
    "min_true_overlap_fraction",
    "min_pred_overlap_fraction",
    "score_smoothing_window",
    "score_adjust_window",
}
MODEL_PARAM_KEYS: dict[str, set[str]] = {
    "robust_zscore": {"scale_estimator"},
    "rolling_residual": {"rolling_window", "baseline_statistic", "residual_transform"},
    "isolation_forest": {"window_size", "n_estimators", "contamination"},
    "mlp_autoencoder": {
        "window_size",
        "latent_dim",
        "hidden_dim",
        "learning_rate",
        "batch_size",
        "epochs",
    },
    "lstm": {"window_size", "hidden_dim", "learning_rate", "batch_size", "epochs"},
    "gru": {"window_size", "hidden_dim", "learning_rate", "batch_size", "epochs"},
    "ts2vec": {
        "output_dims",
        "hidden_dims",
        "depth",
        "batch_size_train",
        "learning_rate",
        "iters",
        "max_train_length",
        "temporal_unit",
        "score_method",
        "knn_k",
        "batch_size_eval",
        "sliding_length",
        "sliding_padding",
    },
}

LOCAL_BASELINE_MODELS = {"robust_zscore", "rolling_residual"}
GLOBAL_TRAINABLE_MODELS = {"isolation_forest", "mlp_autoencoder", "lstm", "gru", "ts2vec"}


@dataclass(frozen=True)
class SweepContext:
    args: argparse.Namespace
    dataset: dict[str, pd.DataFrame]
    labels: dict[str, Any]
    subset: list[str]
    subset_metadata: list[dict[str, Any]]
    labels_file: Path | None
    output_root: Path
    git_commit: str | None


@dataclass
class TrainWindowInfo:
    train_end: int
    contaminated_train: bool
    used_fallback: bool
    reason: str


@dataclass
class TrialResult:
    series_scores: list[SeriesScores]
    operational: dict[str, Any]
    artifact_paths: dict[str, str | None]
    series_info: dict[str, dict[str, Any]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna sweeps and fixed-protocol final evaluation on NAB."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--mode",
        choices=("sweep", "final"),
        default="sweep",
        help="sweep runs Optuna; final evaluates exact --params-file with fixed protocol.",
    )
    parser.add_argument("--params-file", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default="sqlite:///optuna_nab_sweeps.db")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--wandb-project", default="anomaly-detection-nab-sweeps")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument("--validation-subset-file", type=Path, default=None)
    parser.add_argument("--validation-size", type=int, default=12)
    parser.add_argument("--objective-key", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("all", "validation", "test", "validation_and_test"),
        default="all",
        help=(
            "Final-mode reporting split. 'all' evaluates all loaded series. "
            "'validation' uses the frozen validation subset. 'test' uses all loaded "
            "series except the frozen validation subset. 'validation_and_test' reports both."
        ),
    )
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--load-models-dir",
        type=Path,
        default=None,
        help="Load fitted per-series models/checkpoints from this directory and skip fitting.",
    )
    parser.add_argument("--ts2vec-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--training-scope",
        choices=("auto", "per_series", "global"),
        default="auto",
        help=(
            "Training scope for model fitting. Default 'auto' uses per-series calibration "
            "for robust_zscore/rolling_residual and global pooled-prefix training for "
            "isolation_forest, mlp_autoencoder, lstm, gru and ts2vec. 'per_series' is "
            "available for ablations; 'global' is ignored for local statistical baselines."
        ),
    )
    parser.add_argument(
        "--ts2vec-training-scope",
        choices=("per_series", "global"),
        default=None,
        help=(
            "Deprecated compatibility flag. Prefer --training-scope. When provided for "
            "--model ts2vec, it overrides --training-scope."
        ),
    )

    # Shared protocol parameters. These are fixed across models and are not sampled by Optuna.
    parser.add_argument("--threshold-source", choices=("train", "all"), default="train")
    parser.add_argument("--threshold-scope", choices=("per_series", "global"), default="per_series")
    parser.add_argument("--threshold-sweep-steps", type=int, default=100)
    parser.add_argument("--protocol-threshold-quantile", type=float, default=0.99)
    parser.add_argument("--protocol-score-smoothing-window", type=int, default=1)
    parser.add_argument(
        "--protocol-normalization",
        choices=("none", "per-series", "robust"),
        default="robust",
        help="Common preprocessing fitted only on the training/calibration prefix.",
    )
    parser.add_argument("--protocol-min-overlap-points", type=int, default=1)
    parser.add_argument("--protocol-min-true-overlap-fraction", type=float, default=0.0)
    parser.add_argument("--protocol-min-pred-overlap-fraction", type=float, default=0.0)
    parser.add_argument(
        "--protocol-min-event-length",
        type=int,
        default=1,
        help="Drop predicted anomaly events shorter than this many points after thresholding.",
    )
    parser.add_argument(
        "--protocol-merge-gap",
        type=int,
        default=0,
        help="Merge predicted anomaly events separated by at most this many normal points.",
    )
    parser.add_argument(
        "--protocol-cooldown",
        type=int,
        default=0,
        help="Suppress new predicted events that start within this many points after a kept event.",
    )
    parser.add_argument(
        "--protocol-alert-mode",
        choices=("full_event", "peak_window"),
        default="full_event",
        help=(
            "How cleaned-up predicted events become point-wise alerts. "
            "'full_event' marks the whole event span; 'peak_window' marks only "
            "a fixed window around the highest score inside each event."
        ),
    )
    parser.add_argument(
        "--protocol-alert-window",
        type=int,
        default=1,
        help="Number of points to mark around each event peak when using peak-window alerts.",
    )
    parser.add_argument(
        "--protocol-max-alerts-per-event",
        type=int,
        default=1,
        help="Maximum number of separated peak windows to emit inside each cleaned event.",
    )
    parser.add_argument(
        "--protocol-peak-min-distance",
        type=int,
        default=None,
        help=(
            "Minimum distance between selected peaks inside one cleaned event. "
            "Defaults to --protocol-alert-window."
        ),
    )
    parser.add_argument("--min-train-points", type=int, default=32)
    parser.add_argument("--fallback-train-fraction", type=float, default=0.05)
    parser.add_argument(
        "--allow-contaminated-train",
        action="store_true",
        help=(
            "Allow fallback training prefixes containing anomaly labels. When this is not set, "
            "contamination is still recorded and warned about if no clean prefix can be formed."
        ),
    )
    parser.add_argument(
        "--strict-clean-train",
        action="store_true",
        help="Fail a series instead of using a contaminated fallback training prefix.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "sweep" and args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive.")
    if args.mode == "final" and args.params_file is None:
        raise SystemExit("--params-file is required when --mode final.")
    if args.params_file is not None and not args.params_file.expanduser().is_file():
        raise SystemExit(f"Params file does not exist: {args.params_file}")
    if args.limit_series is not None and args.limit_series <= 0:
        raise SystemExit("--limit-series must be positive.")
    if args.validation_size <= 0:
        raise SystemExit("--validation-size must be positive.")
    if args.threshold_sweep_steps <= 1:
        raise SystemExit("--threshold-sweep-steps must be greater than 1.")
    if not 0 <= args.protocol_threshold_quantile <= 1:
        raise SystemExit("--protocol-threshold-quantile must be in [0, 1].")
    if args.protocol_score_smoothing_window < 1:
        raise SystemExit("--protocol-score-smoothing-window must be at least 1.")
    if args.protocol_min_overlap_points < 1:
        raise SystemExit("--protocol-min-overlap-points must be at least 1.")
    if not 0 <= args.protocol_min_true_overlap_fraction <= 1:
        raise SystemExit("--protocol-min-true-overlap-fraction must be in [0, 1].")
    if not 0 <= args.protocol_min_pred_overlap_fraction <= 1:
        raise SystemExit("--protocol-min-pred-overlap-fraction must be in [0, 1].")
    if args.protocol_min_event_length < 1:
        raise SystemExit("--protocol-min-event-length must be at least 1.")
    if args.protocol_merge_gap < 0:
        raise SystemExit("--protocol-merge-gap must be non-negative.")
    if args.protocol_cooldown < 0:
        raise SystemExit("--protocol-cooldown must be non-negative.")
    if args.protocol_alert_window < 1:
        raise SystemExit("--protocol-alert-window must be at least 1.")
    if args.protocol_max_alerts_per_event < 1:
        raise SystemExit("--protocol-max-alerts-per-event must be at least 1.")
    if args.protocol_peak_min_distance is not None and args.protocol_peak_min_distance < 1:
        raise SystemExit("--protocol-peak-min-distance must be at least 1 when provided.")
    if not 0 < args.fallback_train_fraction <= 1:
        raise SystemExit("--fallback-train-fraction must be in the interval (0, 1].")
    if args.min_train_points <= 1:
        raise SystemExit("--min-train-points must be greater than 1.")


def set_reproducible_seed(seed: int, device: str | torch.device | None = None) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path | torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_dict(value, name))
        elif isinstance(value, list | tuple):
            continue
        else:
            output[name] = value
    return output


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


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


def series_metadata(dataset: dict[str, pd.DataFrame], label_windows: dict[str, Any]) -> list[dict[str, Any]]:
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


def candidate_rank(item: dict[str, Any], category_medians: dict[str, dict[str, float]]) -> tuple[float, float, str]:
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

    category_medians = {}
    for category, items in by_category.items():
        category_medians[category] = {
            "anomaly_fraction": float(np.median([item["anomaly_fraction"] for item in items])),
            "n_points": float(np.median([item["n_points"] for item in items])),
        }

    selected: list[dict[str, Any]] = []
    for category in sorted(by_category):
        if len(selected) >= validation_size:
            break
        ranked = sorted(by_category[category], key=lambda item: candidate_rank(item, category_medians))
        selected.append(ranked[0])

    remaining = [
        item for item in candidates if item["series"] not in {row["series"] for row in selected}
    ]
    rng = np.random.default_rng(seed)
    for item in remaining:
        item["_diversity_jitter"] = float(rng.uniform(0.0, 1e-6))
    remaining = sorted(
        remaining,
        key=lambda item: (
            -len(by_category[item["category"]]),
            -item["n_events"],
            item["_diversity_jitter"],
            item["series"],
        ),
    )
    for item in remaining:
        if len(selected) >= validation_size:
            break
        selected.append(item)

    selected_names = [item["series"] for item in selected]
    clean_metadata = [
        {key: value for key, value in item.items() if not key.startswith("_")} for item in metadata
    ]
    payload = {
        "seed": seed,
        "validation_size_requested": validation_size,
        "selected_series": selected_names,
        "selection_strategy": "category representative plus deterministic diverse fill",
        "metadata": clean_metadata,
        "selected_metadata": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in selected
        ],
    }
    save_json(subset_file, payload)
    return selected_names, clean_metadata


def effective_training_scope(args: argparse.Namespace, model: str) -> str:
    """Return the actual training scope used for a model.

    The default benchmark protocol is hybrid:
    - robust_zscore and rolling_residual are local per-series calibrators;
    - isolation_forest, mlp_autoencoder, lstm, gru and ts2vec are trained once
      on pooled normal-prefix data from all selected series.

    This keeps local baselines local while allowing trainable models to use the
    full NAB training signal without looking beyond each series' train_end.
    """
    if model in LOCAL_BASELINE_MODELS:
        return "per_series"
    if model == "ts2vec" and getattr(args, "ts2vec_training_scope", None) is not None:
        return args.ts2vec_training_scope
    requested = getattr(args, "training_scope", "auto")
    if requested == "auto":
        return "global" if model in GLOBAL_TRAINABLE_MODELS else "per_series"
    if requested == "global" and model not in GLOBAL_TRAINABLE_MODELS:
        warnings.warn(
            f"Global training requested for local baseline {model}; using per_series instead.",
            stacklevel=2,
        )
        return "per_series"
    return requested


def protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_window_mode": "before_first_anomaly_with_fallback",
        "fallback_train_fraction": args.fallback_train_fraction,
        "min_train_points": args.min_train_points,
        "allow_contaminated_train": args.allow_contaminated_train,
        "strict_clean_train": args.strict_clean_train,
        "normalization": args.protocol_normalization,
        "score_smoothing_window": args.protocol_score_smoothing_window,
        "threshold_source": args.threshold_source,
        "threshold_scope": args.threshold_scope,
        "threshold_quantile": args.protocol_threshold_quantile,
        "min_overlap_points": args.protocol_min_overlap_points,
        "min_true_overlap_fraction": args.protocol_min_true_overlap_fraction,
        "min_pred_overlap_fraction": args.protocol_min_pred_overlap_fraction,
        "min_event_length": args.protocol_min_event_length,
        "merge_gap": args.protocol_merge_gap,
        "cooldown": args.protocol_cooldown,
        "alert_mode": args.protocol_alert_mode,
        "alert_window": args.protocol_alert_window,
        "max_alerts_per_event": args.protocol_max_alerts_per_event,
        "peak_min_distance": args.protocol_peak_min_distance,
        "training_scope_cli": args.training_scope,
        "ts2vec_training_scope_override": args.ts2vec_training_scope,
        "evaluation_region": "after_train_end",
        "benchmark_training_protocol": (
            "hybrid: local statistical baselines are per-series; trainable models are global "
            "on pooled normalized normal-prefix data"
        ),
    }


def score_smoothing(scores: np.ndarray, window: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if window <= 1:
        return finite_array(scores)
    smoothed = pd.Series(scores).rolling(window=window, min_periods=1).mean().to_numpy(np.float32)
    return finite_array(smoothed)


def prediction_postprocessing_kwargs(args: argparse.Namespace) -> dict[str, int | str]:
    return {
        "min_event_length": args.protocol_min_event_length,
        "merge_gap": args.protocol_merge_gap,
        "cooldown": args.protocol_cooldown,
        "alert_mode": args.protocol_alert_mode,
        "alert_window": args.protocol_alert_window,
        "max_alerts_per_event": args.protocol_max_alerts_per_event,
        "peak_min_distance": args.protocol_peak_min_distance,
    }


def safe_series_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(".csv", "")


def resolved_models_dir(args: argparse.Namespace, output_root: Path) -> Path:
    if args.models_dir is not None:
        return args.models_dir.expanduser().resolve()
    return output_root / "models" / args.model


def model_payload_path(models_dir: Path, series_name: str, suffix: str = ".pkl") -> Path:
    return models_dir / f"{safe_series_name(series_name)}{suffix}"


def global_model_path(models_dir: Path, suffix: str = ".pkl") -> Path:
    return models_dir / f"global_model{suffix}"


def ts2vec_checkpoint_path(models_dir: Path, series_name: str | None = None) -> Path:
    if series_name is not None:
        return model_payload_path(models_dir, series_name, suffix=".pt")
    direct = models_dir / "ts2vec_model.pt"
    if direct.is_file():
        return direct
    nested = models_dir / "artifacts" / "ts2vec_model.pt"
    if nested.is_file():
        return nested
    return direct


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def rolling_residual_score(values: np.ndarray, window: int, baseline: str, transform: str) -> np.ndarray:
    baseline_values = shifted_rolling(values, window, baseline)
    residual = np.asarray(values, dtype=np.float32) - baseline_values
    if transform == "squared":
        residual = residual * residual
    else:
        residual = np.abs(residual)
    scale = rolling_mad(values, window)
    return finite_array(residual / np.where(scale < EPS, 1.0, scale))


def score_loaded_payload(payload: Any, values: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if isinstance(payload, dict):
        kind = payload.get("kind")
        if kind == "robust_iqr":
            center = float(payload["center"])
            scale = float(payload["scale"])
            return finite_array(np.abs((values - center) / scale))
        if kind == "rolling_residual":
            return rolling_residual_score(
                values,
                params["rolling_window"],
                params["baseline_statistic"],
                params["residual_transform"],
            )
        raise ValueError(f"Unsupported saved model payload kind: {kind}")
    if hasattr(payload, "score"):
        return payload.score(values)
    raise ValueError(f"Unsupported saved model payload: {type(payload)!r}")


def prepare_series_labels(context: SweepContext, name: str) -> np.ndarray:
    frame = context.dataset[name]
    return labels_for_series(context.labels, name, frame["timestamp"], missing_labels_as_normal=True)


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
        contaminated = bool(labels[:train_end].any())
        return TrainWindowInfo(
            train_end=train_end,
            contaminated_train=contaminated,
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
        if contaminated and context.args.strict_clean_train and not context.args.allow_contaminated_train:
            raise ValueError(
                f"No clean training prefix for {name}; fallback train_end={fallback} is contaminated."
            ) from exc
        if contaminated and not context.args.allow_contaminated_train:
            warnings.warn(
                f"Using contaminated fallback training prefix for {name}: train_end={fallback}. "
                "Set --strict-clean-train to fail instead or --allow-contaminated-train to silence.",
                stacklevel=2,
            )
        return TrainWindowInfo(
            train_end=int(fallback),
            contaminated_train=contaminated,
            used_fallback=True,
            reason=f"fallback_after_error: {exc}",
        )


def fit_normalization(values: np.ndarray, train_end: int, method: str) -> tuple[np.ndarray, dict[str, Any]]:
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


def parameter_count_from_model(model: Any) -> int:
    network = getattr(model, "model_", None) or getattr(model, "_net", None) or getattr(model, "net", None)
    if network is None:
        return 0
    if hasattr(network, "module"):
        network = network.module
    if hasattr(network, "parameters"):
        return int(sum(parameter.numel() for parameter in network.parameters()))
    return 0


def cuda_device_index(device: str | torch.device | None) -> int | None:
    if device is None or not torch.cuda.is_available():
        return None
    try:
        torch_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        warnings.warn(f"Invalid torch device {device!r}; CUDA metrics disabled: {exc}", stacklevel=2)
        return None
    if torch_device.type != "cuda":
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None
    index = 0 if torch_device.index is None else int(torch_device.index)
    if index < 0 or index >= device_count:
        warnings.warn(
            f"CUDA device cuda:{index} is unavailable; device_count={device_count}. CUDA metrics disabled.",
            stacklevel=2,
        )
        return None
    return index


def reset_cuda_peak_memory_stats(device: str | torch.device | None) -> None:
    index = cuda_device_index(device)
    if index is None:
        return
    try:
        torch.cuda.set_device(index)
        torch.empty(1, device=f"cuda:{index}")
        torch.cuda.synchronize(index)
        torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        warnings.warn(
            f"Could not reset CUDA peak memory stats for device={device!r} (resolved cuda:{index}): {exc}",
            stacklevel=2,
        )


def peak_gpu_memory(device: str | torch.device | None) -> int | None:
    index = cuda_device_index(device)
    if index is None:
        return None
    try:
        torch.cuda.set_device(index)
        torch.cuda.synchronize(index)
        return int(torch.cuda.max_memory_allocated())
    except Exception as exc:
        warnings.warn(
            f"Could not read CUDA peak memory for device={device!r} (resolved cuda:{index}): {exc}",
            stacklevel=2,
        )
        return None


def fit_and_score_single_series(
    context: SweepContext,
    name: str,
    params: dict[str, Any],
    device: str | torch.device,
    models_dir: Path,
    save_models: bool,
    load_models_dir: Path | None,
) -> tuple[SeriesScores, dict[str, Any], float, float, int, int, int | None, str | None]:
    model_name = context.args.model
    frame = context.dataset[name]
    raw_values = frame["value"].to_numpy(dtype=np.float32)
    labels = prepare_series_labels(context, name)
    train_info = training_window_info_for_series(context, name)
    values, norm_info = fit_normalization(raw_values, train_info.train_end, context.args.protocol_normalization)

    detector: Any = None
    model_payload: Any = None
    saved_path: str | None = None
    fit_seconds = 0.0
    score_seconds = 0.0
    param_count = 0

    if load_models_dir is not None:
        load_path = model_payload_path(load_models_dir, name)
        if not load_path.is_file():
            raise FileNotFoundError(f"Saved model not found for {name}: {load_path}")
        model_payload = load_pickle(load_path)
        detector = None if isinstance(model_payload, dict) else model_payload
        score_started = time.perf_counter()
        scores = score_loaded_payload(model_payload, values, params)
        score_seconds += time.perf_counter() - score_started
    elif model_name == "robust_zscore":
        fit_started = time.perf_counter()
        scale_estimator = params["scale_estimator"]
        if scale_estimator == "std":
            stats = standard_scale_stats(values[: train_info.train_end])
            detector = RobustZScoreDetector().fit(values, train_info.train_end)
            detector.stats_ = stats
            model_payload = detector
        elif scale_estimator == "iqr":
            train = np.asarray(values[: train_info.train_end], dtype=np.float32)
            center = float(np.nanmedian(train))
            q75, q25 = np.nanpercentile(train, [75, 25])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale < EPS:
                scale = robust_scale_stats(train).scale
            detector = None
            model_payload = {"kind": "robust_iqr", "center": center, "scale": scale}
        else:
            detector = RobustZScoreDetector().fit(values, train_info.train_end)
            model_payload = detector
        fit_seconds += time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = score_loaded_payload(model_payload, values, params)
        score_seconds += time.perf_counter() - score_started
    elif model_name == "rolling_residual":
        fit_started = time.perf_counter()
        detector = None
        model_payload = {"kind": "rolling_residual", "params": params}
        fit_seconds += time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = rolling_residual_score(
            values,
            params["rolling_window"],
            params["baseline_statistic"],
            params["residual_transform"],
        )
        score_seconds += time.perf_counter() - score_started
    elif model_name == "isolation_forest":
        detector = IsolationForestDetector(
            rolling_window=params["window_size"],
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            seed=context.args.seed,
        )
        fit_started = time.perf_counter()
        detector.fit(values, train_info.train_end)
        fit_seconds += time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = detector.score(values)
        score_seconds += time.perf_counter() - score_started
        model_payload = detector
    elif model_name == "mlp_autoencoder":
        detector = MLPAutoencoderDetector(
            window=params["window_size"],
            hidden_dim=params["hidden_dim"],
            bottleneck_dim=params["latent_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
            device=device,
            seed=context.args.seed,
        )
        fit_started = time.perf_counter()
        detector.fit(values, train_info.train_end)
        fit_seconds += time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = detector.score(values)
        score_seconds += time.perf_counter() - score_started
        model_payload = detector
    elif model_name in {"lstm", "gru"}:
        detector = RNNPredictorDetector(
            kind=model_name,
            window=params["window_size"],
            hidden_dim=params["hidden_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
            device=device,
            seed=context.args.seed,
        )
        fit_started = time.perf_counter()
        detector.fit(values, train_info.train_end)
        fit_seconds += time.perf_counter() - fit_started
        score_started = time.perf_counter()
        scores = detector.score(values)
        score_seconds += time.perf_counter() - score_started
        model_payload = detector
    else:
        raise ValueError(model_name)

    smoothing_started = time.perf_counter()
    scores = score_smoothing(scores, context.args.protocol_score_smoothing_window)
    score_seconds += time.perf_counter() - smoothing_started

    if save_models and load_models_dir is None:
        path = model_payload_path(models_dir, name)
        save_pickle(path, model_payload)
        saved_path = str(path)

    if detector is not None and hasattr(detector, "parameter_count"):
        try:
            param_count = int(detector.parameter_count())
        except Exception:
            param_count = parameter_count_from_model(detector)
    elif detector is not None:
        param_count = parameter_count_from_model(detector)

    trainable_examples = max(0, train_info.train_end - params.get("window_size", 1) + 1)
    info = {
        "train_end": train_info.train_end,
        "contaminated_train": train_info.contaminated_train,
        "used_fallback_train_window": train_info.used_fallback,
        "train_window_reason": train_info.reason,
        "normalization": norm_info,
    }
    return (
        SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end),
        info,
        fit_seconds,
        score_seconds,
        trainable_examples,
        param_count,
        len(values),
        saved_path,
    )


def score_classical_or_torch(
    context: SweepContext,
    trial: optuna.trial.Trial | None,
    params: dict[str, Any],
) -> TrialResult:
    model_name = context.args.model
    if effective_training_scope(context.args, model_name) == "global" and model_name in {"isolation_forest", "mlp_autoencoder", "lstm", "gru"}:
        return score_trainable_global(context, trial, params)

    device = resolve_torch_device(context.args.device) if model_name in {"mlp_autoencoder", "lstm", "gru"} else "cpu"
    reset_cuda_peak_memory_stats(device)
    set_reproducible_seed(context.args.seed, device)

    save_models = bool(getattr(context.args, "save_models", False))
    load_models_dir = getattr(context.args, "load_models_dir", None)
    models_dir = resolved_models_dir(context.args, context.output_root)
    if load_models_dir is not None:
        models_dir = load_models_dir.expanduser().resolve()
    elif save_models and getattr(context.args, "mode", None) == "sweep" and trial is not None:
        models_dir = trial_directory(context, trial.number) / "models" / model_name

    series_scores: list[SeriesScores] = []
    series_info: dict[str, dict[str, Any]] = {}
    training_seconds = 0.0
    inference_seconds = 0.0
    inference_points = 0
    trainable_examples = 0
    parameter_counts = []
    saved_paths: list[str] = []

    for name in context.subset:
        (
            item,
            info,
            fit_seconds,
            score_seconds,
            examples,
            param_count,
            n_points,
            saved_path,
        ) = fit_and_score_single_series(
            context=context,
            name=name,
            params=params,
            device=device,
            models_dir=models_dir,
            save_models=save_models,
            load_models_dir=load_models_dir,
        )
        series_scores.append(item)
        series_info[name] = info
        training_seconds += fit_seconds
        inference_seconds += score_seconds
        inference_points += n_points
        trainable_examples += examples
        parameter_counts.append(param_count)
        if saved_path is not None:
            saved_paths.append(saved_path)

    operational = {
        "training_seconds": 0.0 if load_models_dir is not None else training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds) + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points if inference_points else None,
        "parameter_count": int(max(parameter_counts)) if parameter_counts else 0,
        "parameter_count_mean": float(np.mean(parameter_counts)) if parameter_counts else 0.0,
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "device": str(device),
        "model_size_bytes": sum(Path(path).stat().st_size for path in saved_paths if Path(path).is_file()) if saved_paths else None,
        "trainable_examples": int(trainable_examples),
        "validation_series": len(context.subset),
        "models_dir": str(models_dir) if save_models or load_models_dir is not None else None,
        "loaded_models": load_models_dir is not None,
        "training_scope": "per_series",
    }
    return TrialResult(series_scores, operational, {"models_dir": operational["models_dir"]}, series_info)



def build_pooled_training_values(prefixes: list[np.ndarray], separator_length: int) -> np.ndarray:
    """Pool per-series normal prefixes without using evaluation suffixes.

    A short zero separator reduces the number of cross-series windows learned by
    window-based detectors. Since every series is normalized using only its own
    train prefix, zero is the local baseline for robust/per-series normalization.
    """
    chunks: list[np.ndarray] = []
    separator_length = max(0, int(separator_length))
    separator = np.zeros(separator_length, dtype=np.float32)
    for prefix in prefixes:
        prefix = finite_array(np.asarray(prefix, dtype=np.float32))
        if len(prefix) == 0:
            continue
        if chunks and separator_length > 0:
            chunks.append(separator.copy())
        chunks.append(prefix)
    if not chunks:
        raise ValueError("No training values available for global model fitting.")
    return finite_array(np.concatenate(chunks).astype(np.float32))


def make_trainable_detector(
    model_name: str,
    params: dict[str, Any],
    device: str | torch.device,
    seed: int,
) -> Any:
    if model_name == "isolation_forest":
        return IsolationForestDetector(
            rolling_window=params["window_size"],
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            seed=seed,
        )
    if model_name == "mlp_autoencoder":
        return MLPAutoencoderDetector(
            window=params["window_size"],
            hidden_dim=params["hidden_dim"],
            bottleneck_dim=params["latent_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
            device=device,
            seed=seed,
        )
    if model_name in {"lstm", "gru"}:
        return RNNPredictorDetector(
            kind=model_name,
            window=params["window_size"],
            hidden_dim=params["hidden_dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["learning_rate"],
            device=device,
            seed=seed,
        )
    raise ValueError(f"Unsupported global trainable model: {model_name}")


def detector_parameter_count(detector: Any) -> int:
    if detector is not None and hasattr(detector, "parameter_count"):
        try:
            return int(detector.parameter_count())
        except Exception:
            pass
    return parameter_count_from_model(detector)


def score_trainable_global(
    context: SweepContext,
    trial: optuna.trial.Trial | None,
    params: dict[str, Any],
) -> TrialResult:
    """Fit one global trainable model on pooled normal-prefix data.

    This is the default protocol for isolation_forest, mlp_autoencoder, lstm and
    gru. Each series is normalized using only its own train prefix, only train
    prefixes are pooled for fitting, and each full series is scored separately.
    Thresholds are still chosen later from train scores per the shared protocol.
    """
    model_name = context.args.model
    if model_name not in {"isolation_forest", "mlp_autoencoder", "lstm", "gru"}:
        raise ValueError(f"score_trainable_global called for unsupported model: {model_name}")

    device = resolve_torch_device(context.args.device) if model_name in {"mlp_autoencoder", "lstm", "gru"} else "cpu"
    reset_cuda_peak_memory_stats(device)
    set_reproducible_seed(context.args.seed, device)

    save_models = bool(getattr(context.args, "save_models", False))
    load_models_dir = getattr(context.args, "load_models_dir", None)
    models_dir = resolved_models_dir(context.args, context.output_root)
    if load_models_dir is not None:
        models_dir = load_models_dir.expanduser().resolve()
    elif save_models and getattr(context.args, "mode", None) == "sweep" and trial is not None:
        models_dir = trial_directory(context, trial.number) / "models" / model_name

    per_series: list[tuple[str, np.ndarray, np.ndarray, TrainWindowInfo, dict[str, Any]]] = []
    prefixes: list[np.ndarray] = []
    trainable_examples = 0
    inference_points = 0
    window_size = int(params.get("window_size", 1))

    for name in context.subset:
        frame = context.dataset[name]
        raw_values = frame["value"].to_numpy(dtype=np.float32)
        labels = prepare_series_labels(context, name)
        train_info = training_window_info_for_series(context, name)
        values, norm_info = fit_normalization(raw_values, train_info.train_end, context.args.protocol_normalization)
        prefixes.append(values[: train_info.train_end])
        trainable_examples += max(0, train_info.train_end - window_size + 1)
        inference_points += len(values)
        per_series.append((name, values, labels, train_info, norm_info))

    model_path = global_model_path(models_dir)
    training_seconds = 0.0
    detector: Any
    if load_models_dir is not None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved global model not found: {model_path}")
        detector = load_pickle(model_path)
    else:
        pooled_train = build_pooled_training_values(prefixes, separator_length=window_size)
        detector = make_trainable_detector(model_name, params, device, context.args.seed)
        fit_started = time.perf_counter()
        detector.fit(pooled_train, len(pooled_train))
        training_seconds = time.perf_counter() - fit_started
        if save_models:
            save_pickle(model_path, detector)

    series_scores: list[SeriesScores] = []
    series_info: dict[str, dict[str, Any]] = {}
    inference_seconds = 0.0
    for name, values, labels, train_info, norm_info in per_series:
        score_started = time.perf_counter()
        scores = detector.score(values)
        scores = score_smoothing(scores, context.args.protocol_score_smoothing_window)
        inference_seconds += time.perf_counter() - score_started
        series_scores.append(SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end))
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
            "training_scope": "global",
        }

    saved_model_size = model_path.stat().st_size if (save_models or load_models_dir is not None) and model_path.is_file() else None
    operational = {
        "training_seconds": 0.0 if load_models_dir is not None else training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds) + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points if inference_points else None,
        "parameter_count": detector_parameter_count(detector),
        "parameter_count_mean": detector_parameter_count(detector),
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "device": str(device),
        "model_size_bytes": saved_model_size,
        "trainable_examples": int(trainable_examples),
        "validation_series": len(context.subset),
        "models_dir": str(models_dir) if save_models or load_models_dir is not None else None,
        "loaded_models": load_models_dir is not None,
        "training_scope": "global",
        "global_training_separator_points": window_size,
    }
    return TrialResult(
        series_scores,
        operational,
        {"model": str(model_path) if (save_models or load_models_dir is not None) else None},
        series_info,
    )

def build_ts2vec_eval_args(params: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        score_method=params["score_method"],
        batch_size=params["batch_size_eval"],
        sliding_length=params["sliding_length"],
        sliding_padding=params["sliding_padding"],
        knn_k=params["knn_k"],
    )


def ts2vec_score_with_model(model: Any, values: np.ndarray, params: dict[str, Any], train_end: int) -> np.ndarray:
    from evaluate_ts2vec import score_series as ts2vec_score_series

    scores = ts2vec_score_series(model, values, build_ts2vec_eval_args(params), train_end)
    return score_smoothing(scores, window=1)  # common smoothing is applied by caller


def score_ts2vec_per_series(
    context: SweepContext,
    params: dict[str, Any],
    trial_dir: Path,
    TS2Vec: type,
    ts2vec_source: Path | None,
) -> TrialResult:
    device = resolve_torch_device(context.args.device)
    reset_cuda_peak_memory_stats(device)
    set_reproducible_seed(context.args.seed, device)

    save_models = bool(getattr(context.args, "save_models", False))
    load_models_dir = getattr(context.args, "load_models_dir", None)
    models_dir = resolved_models_dir(context.args, context.output_root)
    if load_models_dir is not None:
        models_dir = load_models_dir.expanduser().resolve()
    elif save_models and getattr(context.args, "mode", None) == "sweep":
        models_dir = trial_dir / "models" / "ts2vec"

    series_scores: list[SeriesScores] = []
    series_info: dict[str, dict[str, Any]] = {}
    checkpoint_paths: list[str] = []
    training_seconds = 0.0
    inference_seconds = 0.0
    inference_points = 0
    trainable_examples = 0
    parameter_counts = []

    for name in context.subset:
        frame = context.dataset[name]
        raw_values = frame["value"].to_numpy(dtype=np.float32)
        labels = prepare_series_labels(context, name)
        train_info = training_window_info_for_series(context, name)
        values, norm_info = fit_normalization(raw_values, train_info.train_end, context.args.protocol_normalization)

        model = TS2Vec(
            input_dims=1,
            output_dims=params["output_dims"],
            hidden_dims=params["hidden_dims"],
            depth=params["depth"],
            device=device,
            lr=params["learning_rate"],
            batch_size=params["batch_size_train"],
            max_train_length=params["max_train_length"],
            temporal_unit=params["temporal_unit"],
        )
        checkpoint_path = ts2vec_checkpoint_path(models_dir, name) if (save_models or load_models_dir is not None) else trial_dir / "artifacts" / f"{safe_series_name(name)}.pt"
        if load_models_dir is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Saved TS2Vec checkpoint not found for {name}: {checkpoint_path}")
            model.load(str(checkpoint_path))
        else:
            train_data = values[: train_info.train_end].reshape(1, -1, 1).astype(np.float32)
            started = time.perf_counter()
            model.fit(train_data, n_iters=params["iters"], verbose=False)
            training_seconds += time.perf_counter() - started
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                model.save(str(checkpoint_path))
                checkpoint_paths.append(str(checkpoint_path))
            except Exception as exc:
                warnings.warn(f"Could not save TS2Vec checkpoint for {name}: {exc}", stacklevel=2)

        started = time.perf_counter()
        scores = ts2vec_score_with_model(model, values, params, train_info.train_end)
        scores = score_smoothing(scores, context.args.protocol_score_smoothing_window)
        inference_seconds += time.perf_counter() - started
        inference_points += len(values)
        trainable_examples += train_info.train_end
        parameter_counts.append(parameter_count_from_model(model))
        series_scores.append(SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end))
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
        }

    model_size = sum(Path(path).stat().st_size for path in checkpoint_paths if Path(path).is_file()) if checkpoint_paths else None
    operational = {
        "training_seconds": 0.0 if load_models_dir is not None else training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds) + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points if inference_points else None,
        "parameter_count": int(max(parameter_counts)) if parameter_counts else 0,
        "parameter_count_mean": float(np.mean(parameter_counts)) if parameter_counts else 0.0,
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "device": str(device),
        "model_size_bytes": model_size,
        "trainable_examples": int(trainable_examples),
        "validation_series": len(context.subset),
        "ts2vec_source": None if ts2vec_source is None else str(ts2vec_source),
        "models_dir": str(models_dir) if save_models or load_models_dir is not None else None,
        "loaded_models": load_models_dir is not None,
        "training_scope": "per_series",
    }
    return TrialResult(series_scores, operational, {"models_dir": operational["models_dir"]}, series_info)


def score_ts2vec_global(
    context: SweepContext,
    params: dict[str, Any],
    trial_dir: Path,
    TS2Vec: type,
    ts2vec_source: Path | None,
) -> TrialResult:
    device = resolve_torch_device(context.args.device)
    reset_cuda_peak_memory_stats(device)
    set_reproducible_seed(context.args.seed, device)

    save_models = bool(getattr(context.args, "save_models", False))
    load_models_dir = getattr(context.args, "load_models_dir", None)
    models_dir = resolved_models_dir(context.args, context.output_root)
    if load_models_dir is not None:
        models_dir = load_models_dir.expanduser().resolve()
    elif save_models and getattr(context.args, "mode", None) == "sweep":
        models_dir = trial_dir / "models" / "ts2vec"

    train_arrays = []
    per_series: list[tuple[str, np.ndarray, np.ndarray, TrainWindowInfo, dict[str, Any]]] = []
    for name in context.subset:
        frame = context.dataset[name]
        raw_values = frame["value"].to_numpy(dtype=np.float32)
        labels = prepare_series_labels(context, name)
        train_info = training_window_info_for_series(context, name)
        values, norm_info = fit_normalization(raw_values, train_info.train_end, context.args.protocol_normalization)
        train_arrays.append(values[: train_info.train_end])
        per_series.append((name, values, labels, train_info, norm_info))

    max_len = max(len(values) for values in train_arrays)
    train_data = np.full((len(train_arrays), max_len, 1), np.nan, dtype=np.float32)
    for index, values in enumerate(train_arrays):
        train_data[index, : len(values), 0] = values

    model = TS2Vec(
        input_dims=1,
        output_dims=params["output_dims"],
        hidden_dims=params["hidden_dims"],
        depth=params["depth"],
        device=device,
        lr=params["learning_rate"],
        batch_size=params["batch_size_train"],
        max_train_length=params["max_train_length"],
        temporal_unit=params["temporal_unit"],
    )
    checkpoint_path = ts2vec_checkpoint_path(models_dir) if (save_models or load_models_dir is not None) else trial_dir / "artifacts" / "ts2vec_model.pt"
    if load_models_dir is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Saved TS2Vec checkpoint not found: {checkpoint_path}")
        training_seconds = 0.0
        model.load(str(checkpoint_path))
    else:
        training_started = time.perf_counter()
        model.fit(train_data, n_iters=params["iters"], verbose=False)
        training_seconds = time.perf_counter() - training_started
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(checkpoint_path))
        except Exception as exc:
            warnings.warn(f"Could not save TS2Vec checkpoint: {exc}", stacklevel=2)
            checkpoint_path = None

    series_scores = []
    series_info: dict[str, dict[str, Any]] = {}
    inference_seconds = 0.0
    inference_points = 0
    for name, values, labels, train_info, norm_info in per_series:
        score_started = time.perf_counter()
        scores = ts2vec_score_with_model(model, values, params, train_info.train_end)
        scores = score_smoothing(scores, context.args.protocol_score_smoothing_window)
        inference_seconds += time.perf_counter() - score_started
        inference_points += len(values)
        series_scores.append(SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end))
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
        }

    model_size = checkpoint_path.stat().st_size if checkpoint_path is not None and checkpoint_path.is_file() else None
    operational = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": training_seconds + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points if inference_points else None,
        "parameter_count": parameter_count_from_model(model),
        "parameter_count_mean": parameter_count_from_model(model),
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "device": str(device),
        "model_size_bytes": model_size,
        "trainable_examples": int(np.isfinite(train_data[:, :, 0]).sum()),
        "validation_series": len(context.subset),
        "ts2vec_source": None if ts2vec_source is None else str(ts2vec_source),
        "models_dir": str(models_dir) if save_models or load_models_dir is not None else None,
        "loaded_models": load_models_dir is not None,
        "training_scope": "global",
    }
    return TrialResult(
        series_scores,
        operational,
        {"model_checkpoint": None if checkpoint_path is None else str(checkpoint_path)},
        series_info,
    )


def score_ts2vec(context: SweepContext, trial: optuna.trial.Trial | None, params: dict[str, Any], trial_dir: Path) -> TrialResult:
    TS2Vec, ts2vec_source = import_ts2vec(context.args.ts2vec_dir)
    if effective_training_scope(context.args, "ts2vec") == "global":
        return score_ts2vec_global(context, params, trial_dir, TS2Vec, ts2vec_source)
    return score_ts2vec_per_series(context, params, trial_dir, TS2Vec, ts2vec_source)


def sample_params(trial: optuna.trial.Trial, model: str) -> dict[str, Any]:
    if model == "robust_zscore":
        return {"scale_estimator": trial.suggest_categorical("scale_estimator", ["mad", "iqr", "std"])}
    if model == "rolling_residual":
        return {
            "rolling_window": trial.suggest_categorical("rolling_window", [16, 32, 64, 128, 256]),
            "baseline_statistic": trial.suggest_categorical("baseline_statistic", ["mean", "median"]),
            "residual_transform": trial.suggest_categorical("residual_transform", ["absolute", "squared"]),
        }
    if model == "isolation_forest":
        return {
            "window_size": trial.suggest_categorical("window_size", [16, 32, 64, 128]),
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 400]),
            "contamination": trial.suggest_categorical("contamination", ["auto"]),
        }
    if model == "mlp_autoencoder":
        return {
            "window_size": trial.suggest_categorical("window_size", [32, 64, 128]),
            "latent_dim": trial.suggest_categorical("latent_dim", [8, 16, 32]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": trial.suggest_int("epochs", 5, 15),
        }
    if model in {"lstm", "gru"}:
        return {
            "window_size": trial.suggest_categorical("window_size", [32, 64, 128]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
            "epochs": trial.suggest_int("epochs", 5, 15),
        }
    if model == "ts2vec":
        return {
            "output_dims": trial.suggest_categorical("output_dims", [80, 160]),
            "hidden_dims": trial.suggest_categorical("hidden_dims", [32, 64]),
            "depth": trial.suggest_categorical("depth", [4, 6]),
            "batch_size_train": trial.suggest_categorical("batch_size_train", [2, 4]),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 1e-3, log=True),
            "iters": trial.suggest_categorical("iters", [50, 100, 500, 1000, 2000]),
            "max_train_length": trial.suggest_categorical("max_train_length", [128, 256]),
            "temporal_unit": trial.suggest_categorical("temporal_unit", [0, 1]),
            "score_method": trial.suggest_categorical("score_method", ["knn", "centroid", "mask-diff"]),
            "knn_k": trial.suggest_categorical("knn_k", [1, 3, 5]),
            "batch_size_eval": trial.suggest_categorical("batch_size_eval", [64, 128]),
            "sliding_length": trial.suggest_categorical("sliding_length", [16, 32]),
            "sliding_padding": trial.suggest_categorical("sliding_padding", [64, 128]),
        }
    raise ValueError(model)


def sanitize_params_for_model(raw_params: dict[str, Any], model: str) -> dict[str, Any]:
    allowed = MODEL_PARAM_KEYS[model]
    cleaned = {key: value for key, value in raw_params.items() if key in allowed}
    ignored = sorted(set(raw_params) - allowed)
    if ignored:
        warnings.warn(
            f"Ignoring non-model/protocol/unused params for {model}: {ignored}. "
            "Use CLI protocol flags for thresholding, smoothing, normalization and event matching.",
            stacklevel=2,
        )
    missing = sorted(allowed - set(cleaned))
    if missing:
        raise SystemExit(f"Params file is missing required model params for {model}: {missing}")
    return cleaned


def event_auc_placeholders(metrics: dict[str, Any]) -> None:
    sections = [metrics["aggregate"], metrics["macro_average"], metrics["category_macro_average"], *metrics.get("category_macro", {}).values()]
    for section in sections:
        eventwise = section.get("eventwise", {})
        eventwise.setdefault("roc_auc", None)
        eventwise.setdefault("pr_auc", None)
        if "matched_events" in eventwise:
            eventwise.setdefault("true_detected_events", eventwise["matched_events"])
            eventwise.setdefault("pred_matched_events", eventwise["matched_events"])
            eventwise.setdefault("tp_events", eventwise["matched_events"])


def evaluate_scores(
    series_scores: list[SeriesScores],
    context: SweepContext,
    params: dict[str, Any],
    series_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not series_scores:
        raise ValueError("No series scores to evaluate.")
    series_info = series_info or {}
    postprocess_kwargs = prediction_postprocessing_kwargs(context.args)
    thresholds: dict[str, float] = {}
    all_scores = np.concatenate([item.scores[item.train_end :] for item in series_scores])
    all_labels = np.concatenate([item.labels[item.train_end :] for item in series_scores])
    if context.args.threshold_scope == "global":
        threshold_source = (
            np.concatenate([item.scores[: item.train_end] for item in series_scores])
            if context.args.threshold_source == "train"
            else all_scores
        )
        threshold = choose_threshold(threshold_source, None, context.args.protocol_threshold_quantile)
        thresholds = {item.name: threshold for item in series_scores}
    else:
        for item in series_scores:
            eval_scores = item.scores[item.train_end :]
            threshold_source = item.scores[: item.train_end] if context.args.threshold_source == "train" else eval_scores
            thresholds[item.name] = choose_threshold(threshold_source, None, context.args.protocol_threshold_quantile)

    all_pred = []
    per_series = []
    aggregate_counts = {
        "true_events": 0,
        "pred_events": 0,
        "matched_events": 0,
        "fp_events": 0,
        "fn_events": 0,
        "detection_delay_sum": 0,
        "detection_delay_max": 0,
    }
    aggregate_raw_pred_events = 0
    for item in series_scores:
        eval_scores = item.scores[item.train_end :]
        eval_labels = item.labels[item.train_end :]
        raw_pred = predict(eval_scores, thresholds[item.name])
        pred = predict(eval_scores, thresholds[item.name], **postprocess_kwargs)
        raw_pred_events = len(contiguous_events(raw_pred))
        aggregate_raw_pred_events += raw_pred_events
        all_pred.append(pred)
        counts = event_counts(
            eval_labels,
            pred,
            min_overlap_points=context.args.protocol_min_overlap_points,
            min_true_overlap_fraction=context.args.protocol_min_true_overlap_fraction,
            min_pred_overlap_fraction=context.args.protocol_min_pred_overlap_fraction,
        )
        for key, value in counts.items():
            if key == "detection_delay_max":
                aggregate_counts[key] = max(aggregate_counts[key], value)
            else:
                aggregate_counts[key] += value
        eventwise = event_metrics_from_counts(counts)
        pointwise = point_metrics_from_predictions(eval_labels, eval_scores, pred)
        info = series_info.get(item.name, {})
        per_series.append(
            {
                "name": item.name,
                "category": category_of(item.name),
                "threshold": thresholds[item.name],
                "n_points": int(len(item.labels)),
                "n_train_points": int(item.train_end),
                "n_evaluation_points": int(len(eval_labels)),
                "n_anomaly_points": int(eval_labels.sum()),
                "n_true_events": len(contiguous_events(eval_labels)),
                "n_raw_pred_events": raw_pred_events,
                "n_pred_events": len(contiguous_events(pred)),
                "contaminated_train": bool(info.get("contaminated_train", False)),
                "used_fallback_train_window": bool(info.get("used_fallback_train_window", False)),
                "train_window_reason": info.get("train_window_reason"),
                "normalization": info.get("normalization"),
                "pointwise": pointwise,
                "eventwise": eventwise,
            }
        )

    aggregate = {
        "pointwise": point_metrics_from_predictions(all_labels, all_scores, np.concatenate(all_pred)),
        "eventwise": event_metrics_from_counts(aggregate_counts),
    }
    aggregate["eventwise"]["raw_pred_events"] = aggregate_raw_pred_events
    aggregate["pointwise"]["roc_auc"] = safe_auc("roc_auc", all_labels, all_scores)
    aggregate["pointwise"]["pr_auc"] = safe_auc("pr_auc", all_labels, all_scores)

    category_macro = {}
    for category in sorted({item["category"] for item in per_series}):
        rows = [item for item in per_series if item["category"] == category]
        category_macro[category] = {
            "pointwise": macro_average([item["pointwise"] for item in rows]),
            "eventwise": macro_average([item["eventwise"] for item in rows]),
        }
    category_macro_average = {
        "pointwise": macro_average([value["pointwise"] for value in category_macro.values()]),
        "eventwise": macro_average([value["eventwise"] for value in category_macro.values()]),
    }
    diagnostics_series = [
        SeriesScores(item.name, item.scores[item.train_end :], item.labels[item.train_end :], 0)
        for item in series_scores
    ]
    metrics = {
        "threshold": None if context.args.threshold_scope == "per_series" else next(iter(thresholds.values())),
        "thresholds": thresholds,
        "protocol": protocol_config(context.args),
        "threshold_quantile": context.args.protocol_threshold_quantile,
        "threshold_source": context.args.threshold_source,
        "threshold_scope": context.args.threshold_scope,
        "event_matching": {
            "min_overlap_points": context.args.protocol_min_overlap_points,
            "min_true_overlap_fraction": context.args.protocol_min_true_overlap_fraction,
            "min_pred_overlap_fraction": context.args.protocol_min_pred_overlap_fraction,
        },
        "event_postprocessing": postprocess_kwargs,
        "aggregate": aggregate,
        "macro_average": {
            "pointwise": macro_average([item["pointwise"] for item in per_series]),
            "eventwise": macro_average([item["eventwise"] for item in per_series]),
        },
        "category_macro": category_macro,
        "category_macro_average": category_macro_average,
        "threshold_diagnostics": {
            "pointwise_best_f1": sweep_point_f1(
                all_labels,
                all_scores,
                context.args.threshold_sweep_steps,
                **postprocess_kwargs,
            ),
            "eventwise_best_f1": sweep_event_f1(
                diagnostics_series,
                context.args.threshold_sweep_steps,
                min_overlap_points=context.args.protocol_min_overlap_points,
                min_true_overlap_fraction=context.args.protocol_min_true_overlap_fraction,
                min_pred_overlap_fraction=context.args.protocol_min_pred_overlap_fraction,
                **postprocess_kwargs,
            ),
        },
        "per_series": per_series,
    }
    event_auc_placeholders(metrics)
    return metrics


def resolve_objective(metrics: dict[str, Any], key: str) -> float:
    aliases = {"category_macro.eventwise.f1": "category_macro_average.eventwise.f1"}
    key = aliases.get(key, key)
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Objective key not found: {key}")
        value = value[part]
    if value is None:
        return float("-inf")
    return float(value)


def init_wandb_run(context: SweepContext, trial: optuna.trial.Trial, params: dict[str, Any]):
    if context.args.wandb_mode == "disabled":
        return None
    if wandb is None:
        warnings.warn("wandb is not installed; continuing without W&B logging.", stacklevel=2)
        return None
    config = {
        **params,
        "protocol": protocol_config(context.args),
        "effective_training_scope": effective_training_scope(context.args, context.args.model),
        "model": context.args.model,
        "trial_number": trial.number,
        "study_name": context.args.study_name,
        "objective_key": context.args.objective_key,
        "seed": context.args.seed,
        "validation_series": context.subset,
        "validation_subset_file": str(validation_subset_path(context.args)),
        "git_commit": context.git_commit,
    }
    return wandb.init(
        project=context.args.wandb_project,
        entity=context.args.wandb_entity,
        mode=context.args.wandb_mode,
        group=context.args.study_name,
        job_type=context.args.model,
        name=f"{context.args.model}-trial-{trial.number:04d}",
        config=jsonable(config),
        reinit=True,
    )


def log_wandb_artifacts(run: Any, context: SweepContext, trial_dir: Path, paths: dict[str, str | None]) -> None:
    if run is None or not context.args.wandb_log_artifacts:
        return
    try:
        artifact = wandb.Artifact(f"{context.args.model}-trial-{trial_dir.name}", type="optuna-trial")
        for filename in ["params.json", "metrics.json", "operational_metrics.json", "commands.json"]:
            path = trial_dir / filename
            if path.is_file():
                artifact.add_file(str(path))
        subset = validation_subset_path(context.args)
        if subset.is_file():
            artifact.add_file(str(subset))
        for path_str in paths.values():
            if path_str is None:
                continue
            path = Path(path_str)
            if path.is_file():
                artifact.add_file(str(path))
            elif path.is_dir():
                artifact.add_dir(str(path))
        run.log_artifact(artifact)
    except Exception as exc:
        warnings.warn(f"W&B artifact logging failed: {exc}", stacklevel=2)


def validation_subset_path(args: argparse.Namespace) -> Path:
    if args.validation_subset_file is not None:
        return args.validation_subset_file.expanduser().resolve()
    study_name = args.study_name or f"nab-{args.model}"
    base = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_ROOT
    return (base / study_name / "validation_subset.json").expanduser().resolve()


def trial_directory(context: SweepContext, trial_number: int) -> Path:
    return context.output_root / context.args.study_name / context.args.model / f"trial_{trial_number:04d}"


def objective(context: SweepContext) -> Callable[[optuna.trial.Trial], float]:
    def _objective(trial: optuna.trial.Trial) -> float:
        trial_started = time.perf_counter()
        trial_dir = trial_directory(context, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "logs").mkdir(exist_ok=True)
        (trial_dir / "artifacts").mkdir(exist_ok=True)
        params = sample_params(trial, context.args.model)
        save_json(trial_dir / "params.json", params)
        save_json(
            trial_dir / "commands.json",
            {
                "argv": sys.argv,
                "cwd": str(PROJECT_ROOT),
                "protocol": protocol_config(context.args),
                "effective_training_scope": effective_training_scope(context.args, context.args.model),
                "wandb_env_vars": ["WANDB_DIR", "WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"],
            },
        )
        run = init_wandb_run(context, trial, params)
        try:
            if context.args.model == "ts2vec":
                result = score_ts2vec(context, trial, params, trial_dir)
            else:
                result = score_classical_or_torch(context, trial, params)
            metrics = evaluate_scores(result.series_scores, context, params, result.series_info)
            objective_value = resolve_objective(metrics, context.args.objective_key)
            operational = {
                **result.operational,
                "total_trial_seconds": time.perf_counter() - trial_started,
                "objective_key": context.args.objective_key,
                "objective_value": objective_value,
                "trial_number": trial.number,
                "model": context.args.model,
                "seed": context.args.seed,
                "validation_subset_file": str(validation_subset_path(context.args)),
                "output_dir": str(trial_dir),
                "git_commit": context.git_commit,
            }
            metrics["operational"] = operational
            metrics["params"] = params
            metrics["model"] = context.args.model
            metrics["trial_number"] = trial.number
            metrics["objective_key"] = context.args.objective_key
            metrics["objective_value"] = objective_value
            save_json(trial_dir / "metrics.json", metrics)
            save_json(trial_dir / "operational_metrics.json", operational)

            trial.set_user_attr("model", context.args.model)
            trial.set_user_attr("objective_key", context.args.objective_key)
            trial.set_user_attr("objective_value", objective_value)
            trial.set_user_attr("metrics_path", str(trial_dir / "metrics.json"))
            trial.set_user_attr("params_path", str(trial_dir / "params.json"))
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("validation_subset_file", str(validation_subset_path(context.args)))
            trial.set_user_attr("status", "ok")

            if run is not None:
                run.log({**flatten_dict(metrics), **flatten_dict(operational)})
                log_wandb_artifacts(run, context, trial_dir, result.artifact_paths)
            return objective_value
        except RuntimeError as exc:
            message = str(exc)
            if "out of memory" in message.lower() or "cuda" in message.lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr("status", "pruned")
                trial.set_user_attr("error", message)
                save_json(trial_dir / "error.json", {"error": message, "traceback": traceback.format_exc()})
                if run is not None:
                    run.log({"status": "pruned", "error": message})
                raise optuna.exceptions.TrialPruned(message) from exc
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error", message)
            save_json(trial_dir / "error.json", {"error": message, "traceback": traceback.format_exc()})
            raise
        except Exception as exc:
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error", str(exc))
            save_json(trial_dir / "error.json", {"error": str(exc), "traceback": traceback.format_exc()})
            if run is not None:
                run.log({"status": "failed", "error": str(exc)})
            raise
        finally:
            if run is not None:
                run.finish()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _objective


def load_dataset_and_labels(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path | None]:
    labels_file = find_labels_file(args.labels_file, dataset_dir=args.dataset_dir, auto_download=True)
    label_windows = load_label_windows(labels_file)
    dataset = load_dataset(args.dataset_dir, args.categories)
    if args.limit_series is not None:
        dataset = dict(list(dataset.items())[: args.limit_series])
    if not dataset:
        raise SystemExit("No NAB series were loaded. Check --dataset-dir/--categories.")
    return dataset, label_windows, labels_file


def create_context(args: argparse.Namespace) -> SweepContext:
    dataset, label_windows, labels_file = load_dataset_and_labels(args)
    subset_file = validation_subset_path(args)
    subset, metadata = select_validation_subset(
        dataset, label_windows, subset_file, validation_size=args.validation_size, seed=args.seed
    )
    missing = sorted(set(subset) - set(dataset))
    if missing:
        raise SystemExit(f"Validation subset references missing series: {missing}")
    output_root = (args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    args.output_dir = output_root
    return SweepContext(
        args=args,
        dataset=dataset,
        labels=label_windows,
        subset=subset,
        subset_metadata=metadata,
        labels_file=labels_file,
        output_root=output_root,
        git_commit=git_commit(),
    )


def load_params(path: Path, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    raw_params = payload.get("params", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_params, dict):
        raise SystemExit("Params file must contain a JSON object.")
    cleaned = sanitize_params_for_model(raw_params, model)
    return cleaned, raw_params


def make_final_output_dir(path: Path | None, model: str) -> Path:
    if path is None:
        path = DEFAULT_FINAL_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S") / model
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_split_names(args: argparse.Namespace, dataset: dict[str, pd.DataFrame], labels: dict[str, Any]) -> dict[str, list[str]]:
    all_names = sorted(dataset)
    if args.split == "all":
        return {"all": all_names}
    subset_file = validation_subset_path(args)
    validation, _metadata = select_validation_subset(
        dataset, labels, subset_file, validation_size=args.validation_size, seed=args.seed
    )
    validation_set = set(validation)
    test = [name for name in all_names if name not in validation_set]
    if args.split == "validation":
        return {"validation": sorted(validation_set)}
    if args.split == "test":
        return {"test": test}
    return {"validation": sorted(validation_set), "test": test}


def metric_summary_row(model: str, split_name: str, metrics: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pointwise = metrics["aggregate"]["pointwise"]
    eventwise = metrics["aggregate"]["eventwise"]
    diagnostics = metrics["threshold_diagnostics"]
    operational = metrics.get("operational", {})
    return {
        "method": model,
        "split": split_name,
        "threshold_quantile": metrics["threshold_quantile"],
        "threshold_source": metrics["threshold_source"],
        "threshold_scope": metrics["threshold_scope"],
        "postprocess_min_event_length": metrics["event_postprocessing"]["min_event_length"],
        "postprocess_merge_gap": metrics["event_postprocessing"]["merge_gap"],
        "postprocess_cooldown": metrics["event_postprocessing"]["cooldown"],
        "postprocess_alert_mode": metrics["event_postprocessing"]["alert_mode"],
        "postprocess_alert_window": metrics["event_postprocessing"]["alert_window"],
        "postprocess_max_alerts_per_event": metrics["event_postprocessing"]["max_alerts_per_event"],
        "postprocess_peak_min_distance": metrics["event_postprocessing"]["peak_min_distance"],
        "point_precision": pointwise["precision"],
        "point_recall": pointwise["recall"],
        "point_f1": pointwise["f1"],
        "point_roc_auc": pointwise["roc_auc"],
        "point_pr_auc": pointwise["pr_auc"],
        "point_tp": pointwise.get("tp"),
        "point_tn": pointwise.get("tn"),
        "point_fp": pointwise.get("fp"),
        "point_fn": pointwise.get("fn"),
        "event_precision": eventwise["precision"],
        "event_recall": eventwise["recall"],
        "event_f1": eventwise["f1"],
        "event_true_events": eventwise.get("true_events"),
        "event_raw_pred_events": eventwise.get("raw_pred_events"),
        "event_pred_events": eventwise.get("pred_events"),
        "event_matched_events": eventwise.get("matched_events"),
        "event_fp_events": eventwise.get("fp_events"),
        "event_fn_events": eventwise.get("fn_events"),
        "event_mean_detection_delay": eventwise.get("mean_detection_delay"),
        "event_max_detection_delay": eventwise.get("max_detection_delay"),
        "point_best_f1_oracle": diagnostics["pointwise_best_f1"]["f1"],
        "event_best_f1_oracle": diagnostics["eventwise_best_f1"]["f1"],
        "training_seconds": operational.get("training_seconds"),
        "inference_seconds": operational.get("inference_seconds"),
        "total_scoring_seconds": operational.get("total_scoring_seconds"),
        "parameter_count": operational.get("parameter_count"),
        "device": operational.get("device"),
        "models_dir": operational.get("models_dir"),
        "loaded_models": operational.get("loaded_models"),
        "training_scope": operational.get("training_scope"),
    }


def per_series_rows(model: str, split_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in metrics["per_series"]:
        pointwise = item["pointwise"]
        eventwise = item["eventwise"]
        rows.append(
            {
                "method": model,
                "split": split_name,
                "series": item["name"],
                "category": item["category"],
                "threshold": item["threshold"],
                "n_points": item["n_points"],
                "n_train_points": item["n_train_points"],
                "n_evaluation_points": item["n_evaluation_points"],
                "n_anomaly_points": item["n_anomaly_points"],
                "n_true_events": item["n_true_events"],
                "n_raw_pred_events": item.get("n_raw_pred_events"),
                "n_pred_events": item["n_pred_events"],
                "contaminated_train": item.get("contaminated_train"),
                "used_fallback_train_window": item.get("used_fallback_train_window"),
                "point_precision": pointwise["precision"],
                "point_recall": pointwise["recall"],
                "point_f1": pointwise["f1"],
                "point_roc_auc": pointwise["roc_auc"],
                "point_pr_auc": pointwise["pr_auc"],
                "point_tp": pointwise.get("tp"),
                "point_tn": pointwise.get("tn"),
                "point_fp": pointwise.get("fp"),
                "point_fn": pointwise.get("fn"),
                "event_precision": eventwise["precision"],
                "event_recall": eventwise["recall"],
                "event_f1": eventwise["f1"],
                "event_true_events": eventwise.get("true_events"),
                "event_pred_events": eventwise.get("pred_events"),
                "event_matched_events": eventwise.get("matched_events"),
                "event_fp_events": eventwise.get("fp_events"),
                "event_fn_events": eventwise.get("fn_events"),
                "event_mean_detection_delay": eventwise.get("mean_detection_delay"),
                "event_max_detection_delay": eventwise.get("max_detection_delay"),
            }
        )
    return rows


def save_predictions(
    output_dir: Path,
    model: str,
    split_name: str,
    dataset: dict[str, pd.DataFrame],
    result: TrialResult,
    metrics: dict[str, Any],
) -> None:
    rows = []
    postprocess_kwargs = metrics.get("event_postprocessing", {})
    for item in result.series_scores:
        frame = dataset[item.name]
        threshold = metrics["thresholds"][item.name]
        raw_predictions = np.zeros_like(item.labels, dtype=np.int8)
        predictions = np.zeros_like(item.labels, dtype=np.int8)
        eval_scores = item.scores[item.train_end :]
        raw_predictions[item.train_end :] = predict(eval_scores, threshold)
        predictions[item.train_end :] = predict(eval_scores, threshold, **postprocess_kwargs)
        timestamps = frame["timestamp"].astype(str).to_numpy()
        values = frame["value"].to_numpy(dtype=np.float32)
        for index in range(item.train_end, len(item.scores)):
            rows.append(
                {
                    "method": model,
                    "split": split_name,
                    "series": item.name,
                    "index": index,
                    "timestamp": timestamps[index],
                    "value": values[index],
                    "score": item.scores[index],
                    "label": item.labels[index],
                    "threshold": threshold,
                    "raw_prediction": raw_predictions[index],
                    "prediction": predictions[index],
                    "train_end": item.train_end,
                }
            )
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(predictions_dir / f"{split_name}_{model}.csv", index=False)


def filter_result(result: TrialResult, names: set[str]) -> TrialResult:
    return TrialResult(
        series_scores=[item for item in result.series_scores if item.name in names],
        operational=result.operational,
        artifact_paths=result.artifact_paths,
        series_info={key: value for key, value in result.series_info.items() if key in names},
    )


def make_final_context(
    args: argparse.Namespace,
    dataset: dict[str, pd.DataFrame],
    labels: dict[str, Any],
    labels_file: Path | None,
    names: list[str],
    output_dir: Path,
) -> SweepContext:
    return SweepContext(
        args=args,
        dataset=dataset,
        labels=labels,
        subset=names,
        subset_metadata=[],
        labels_file=labels_file,
        output_root=output_dir,
        git_commit=git_commit(),
    )


def run_final_mode(args: argparse.Namespace) -> None:
    params, raw_params = load_params(args.params_file, args.model)
    output_dir = make_final_output_dir(args.output_dir, args.model)
    args.output_dir = output_dir
    dataset, labels, labels_file = load_dataset_and_labels(args)
    split_map = final_split_names(args, dataset, labels)
    target_names = sorted({name for names in split_map.values() for name in names})
    context = make_final_context(args, dataset, labels, labels_file, target_names, output_dir)
    trial_dir = output_dir / "artifacts" / args.model

    save_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "raw_params_file": str(args.params_file.expanduser().resolve()),
            "raw_params": raw_params,
            "model_params_used": params,
            "ignored_params": sorted(set(raw_params) - set(params)),
            "protocol": protocol_config(args),
            "effective_training_scope": effective_training_scope(args, args.model),
            "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
            "labels_file": None if labels_file is None else str(labels_file),
            "splits": split_map,
            "note": (
                "Fixed-protocol final evaluation. Optuna params are used only for model-specific "
                "hyperparameters; thresholding, smoothing, preprocessing, event matching and "
                "training-window protocol are CLI/config values."
            ),
        },
    )

    print(f"Loaded {len(dataset)} NAB series.")
    print(f"Using params: {args.params_file}")
    print(f"Protocol: {json.dumps(protocol_config(args), indent=2)}")
    print(f"Effective training scope for {args.model}: {effective_training_scope(args, args.model)}")
    print(f"Scoring series for split(s): {', '.join(split_map)}")
    print(f"Output: {output_dir}")
    if args.save_models:
        print(f"Saving fitted models to: {resolved_models_dir(args, output_dir)}")
    if args.load_models_dir is not None:
        print(f"Loading fitted models from: {args.load_models_dir}")

    if args.model == "ts2vec":
        result = score_ts2vec(context, trial=None, params=params, trial_dir=trial_dir)
    else:
        result = score_classical_or_torch(context, trial=None, params=params)

    summary_rows = []
    per_series = []
    for split_name, names in split_map.items():
        split_result = filter_result(result, set(names))
        split_context = make_final_context(args, dataset, labels, labels_file, names, output_dir)
        metrics = evaluate_scores(split_result.series_scores, split_context, params, split_result.series_info)
        metrics["params"] = params
        metrics["raw_params"] = raw_params
        metrics["model"] = args.model
        metrics["split"] = split_name
        metrics["operational"] = {
            **result.operational,
            "model": args.model,
            "split": split_name,
            "params_file": str(args.params_file.expanduser().resolve()),
            "output_dir": str(output_dir),
            "git_commit": context.git_commit,
        }
        save_json(output_dir / f"metrics_{split_name}_{args.model}.json", metrics)
        save_json(output_dir / f"operational_{split_name}_{args.model}.json", metrics["operational"])
        if args.save_scores:
            save_predictions(output_dir, args.model, split_name, dataset, split_result, metrics)
        summary_rows.append(metric_summary_row(args.model, split_name, metrics, params))
        per_series.extend(per_series_rows(args.model, split_name, metrics))

    summary = pd.DataFrame(summary_rows).sort_values(["split", "event_f1"], ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(per_series).sort_values(["split", "series"]).to_csv(output_dir / "per_series_metrics.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output_dir / 'summary.csv'}.")


def run_sweep_mode(args: argparse.Namespace) -> None:
    if args.study_name is None:
        args.study_name = f"nab-{args.model}"
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT
    context = create_context(args)

    print(f"Loaded {len(context.dataset)} NAB series.")
    print(f"Validation subset: {len(context.subset)} series -> {validation_subset_path(args)}")
    print(f"Study: {args.study_name}; storage: {args.storage}; model: {args.model}")
    print(f"Effective training scope for {args.model}: {effective_training_scope(args, args.model)}")
    print(f"Fixed protocol: {json.dumps(protocol_config(args), indent=2)}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.dry_run:
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective(context), n_trials=1, timeout=args.timeout, gc_after_trial=True)
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="maximize",
            load_if_exists=True,
            sampler=sampler,
        )
        study.optimize(objective(context), n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    if study.best_trial is not None:
        print("Best trial:")
        print(f"  number: {study.best_trial.number}")
        print(f"  value: {study.best_value:.6f}")
        print(f"  params: {study.best_trial.params}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.mode == "final":
        run_final_mode(args)
    else:
        run_sweep_mode(args)


if __name__ == "__main__":
    main()
