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
import sys
import time
import traceback
import warnings
from collections.abc import Callable
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

from anomaly_detection.data import load_dataset  # noqa: E402
from anomaly_detection.detectors import (  # noqa: E402
    IsolationForestDetector,
    MLPAutoencoderDetector,
    RNNPredictorDetector,
    RobustZScoreDetector,
)
from anomaly_detection.labels import (  # noqa: E402
    find_labels_file,
    load_label_windows,
)
from anomaly_detection.metrics import SeriesScores  # noqa: E402
from anomaly_detection.optuna_sweep.config import (  # noqa: E402
    DEFAULT_FINAL_OUTPUT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MODEL_PARAM_KEYS,
    effective_training_scope,
    parse_args,
    protocol_config,
    validate_args,
)
from anomaly_detection.optuna_sweep.evaluation import (  # noqa: E402
    evaluate_scores,
    filter_result,
    metric_summary_row,
    per_series_rows,
    resolve_objective,
    save_predictions,
)
from anomaly_detection.optuna_sweep.protocol import (  # noqa: E402
    fit_normalization,
    prepare_series_labels,
    score_smoothing,
    select_validation_subset,
    training_window_info_for_series,
)
from anomaly_detection.optuna_sweep.types import (  # noqa: E402
    SweepContext,
    TrainWindowInfo,
    TrialResult,
)
from anomaly_detection.optuna_sweep.utils import (  # noqa: E402
    flatten_dict,
    git_commit,
    global_model_path,
    jsonable,
    load_pickle,
    model_payload_path,
    parameter_count_from_model,
    peak_gpu_memory,
    reset_cuda_peak_memory_stats,
    resolved_models_dir,
    safe_series_name,
    save_json,
    save_pickle,
    set_reproducible_seed,
    ts2vec_checkpoint_path,
)
from anomaly_detection.preprocessing import (  # noqa: E402
    EPS,
    finite_array,
    robust_scale_stats,
    rolling_mad,
    shifted_rolling,
    standard_scale_stats,
)
from anomaly_detection.ts2vec_support import import_ts2vec, resolve_torch_device  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover - dependency is optional at import time
    wandb = None


def rolling_residual_score(
    values: np.ndarray, window: int, baseline: str, transform: str
) -> np.ndarray:
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
    values, norm_info = fit_normalization(
        raw_values, train_info.train_end, context.args.protocol_normalization
    )

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
    if effective_training_scope(context.args, model_name) == "global" and model_name in {
        "isolation_forest",
        "mlp_autoencoder",
        "lstm",
        "gru",
    }:
        return score_trainable_global(context, trial, params)

    device = (
        resolve_torch_device(context.args.device)
        if model_name in {"mlp_autoencoder", "lstm", "gru"}
        else "cpu"
    )
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
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds)
        + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points
        if inference_points
        else None,
        "parameter_count": int(max(parameter_counts)) if parameter_counts else 0,
        "parameter_count_mean": float(np.mean(parameter_counts)) if parameter_counts else 0.0,
        "peak_gpu_memory_bytes": peak_gpu_memory(device),
        "device": str(device),
        "model_size_bytes": sum(
            Path(path).stat().st_size for path in saved_paths if Path(path).is_file()
        )
        if saved_paths
        else None,
        "trainable_examples": int(trainable_examples),
        "validation_series": len(context.subset),
        "models_dir": str(models_dir) if save_models or load_models_dir is not None else None,
        "loaded_models": load_models_dir is not None,
        "training_scope": "per_series",
    }
    return TrialResult(
        series_scores, operational, {"models_dir": operational["models_dir"]}, series_info
    )


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

    device = (
        resolve_torch_device(context.args.device)
        if model_name in {"mlp_autoencoder", "lstm", "gru"}
        else "cpu"
    )
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
        values, norm_info = fit_normalization(
            raw_values, train_info.train_end, context.args.protocol_normalization
        )
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
        series_scores.append(
            SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end)
        )
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
            "training_scope": "global",
        }

    saved_model_size = (
        model_path.stat().st_size
        if (save_models or load_models_dir is not None) and model_path.is_file()
        else None
    )
    operational = {
        "training_seconds": 0.0 if load_models_dir is not None else training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds)
        + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points
        if inference_points
        else None,
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


def ts2vec_score_with_model(
    model: Any, values: np.ndarray, params: dict[str, Any], train_end: int
) -> np.ndarray:
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
        values, norm_info = fit_normalization(
            raw_values, train_info.train_end, context.args.protocol_normalization
        )

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
        checkpoint_path = (
            ts2vec_checkpoint_path(models_dir, name)
            if (save_models or load_models_dir is not None)
            else trial_dir / "artifacts" / f"{safe_series_name(name)}.pt"
        )
        if load_models_dir is not None:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Saved TS2Vec checkpoint not found for {name}: {checkpoint_path}"
                )
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
        series_scores.append(
            SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end)
        )
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
        }

    model_size = (
        sum(Path(path).stat().st_size for path in checkpoint_paths if Path(path).is_file())
        if checkpoint_paths
        else None
    )
    operational = {
        "training_seconds": 0.0 if load_models_dir is not None else training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": (0.0 if load_models_dir is not None else training_seconds)
        + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points
        if inference_points
        else None,
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
    return TrialResult(
        series_scores, operational, {"models_dir": operational["models_dir"]}, series_info
    )


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
        values, norm_info = fit_normalization(
            raw_values, train_info.train_end, context.args.protocol_normalization
        )
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
    checkpoint_path = (
        ts2vec_checkpoint_path(models_dir)
        if (save_models or load_models_dir is not None)
        else trial_dir / "artifacts" / "ts2vec_model.pt"
    )
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
        series_scores.append(
            SeriesScores(name=name, scores=scores, labels=labels, train_end=train_info.train_end)
        )
        series_info[name] = {
            "train_end": train_info.train_end,
            "contaminated_train": train_info.contaminated_train,
            "used_fallback_train_window": train_info.used_fallback,
            "train_window_reason": train_info.reason,
            "normalization": norm_info,
        }

    model_size = (
        checkpoint_path.stat().st_size
        if checkpoint_path is not None and checkpoint_path.is_file()
        else None
    )
    operational = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_scoring_seconds": training_seconds + inference_seconds,
        "inference_seconds_per_point": inference_seconds / inference_points
        if inference_points
        else None,
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


def score_ts2vec(
    context: SweepContext, trial: optuna.trial.Trial | None, params: dict[str, Any], trial_dir: Path
) -> TrialResult:
    TS2Vec, ts2vec_source = import_ts2vec(context.args.ts2vec_dir)
    if effective_training_scope(context.args, "ts2vec") == "global":
        return score_ts2vec_global(context, params, trial_dir, TS2Vec, ts2vec_source)
    return score_ts2vec_per_series(context, params, trial_dir, TS2Vec, ts2vec_source)


def sample_params(trial: optuna.trial.Trial, model: str) -> dict[str, Any]:
    if model == "robust_zscore":
        return {
            "scale_estimator": trial.suggest_categorical("scale_estimator", ["mad", "iqr", "std"])
        }
    if model == "rolling_residual":
        return {
            "rolling_window": trial.suggest_categorical("rolling_window", [16, 32, 64, 128, 256]),
            "baseline_statistic": trial.suggest_categorical(
                "baseline_statistic", ["mean", "median"]
            ),
            "residual_transform": trial.suggest_categorical(
                "residual_transform", ["absolute", "squared"]
            ),
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
            "score_method": trial.suggest_categorical(
                "score_method", ["knn", "centroid", "mask-diff"]
            ),
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


def log_wandb_artifacts(
    run: Any, context: SweepContext, trial_dir: Path, paths: dict[str, str | None]
) -> None:
    if run is None or not context.args.wandb_log_artifacts:
        return
    try:
        artifact = wandb.Artifact(
            f"{context.args.model}-trial-{trial_dir.name}", type="optuna-trial"
        )
        for filename in [
            "params.json",
            "metrics.json",
            "operational_metrics.json",
            "commands.json",
        ]:
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
    return (
        context.output_root
        / context.args.study_name
        / context.args.model
        / f"trial_{trial_number:04d}"
    )


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
                "effective_training_scope": effective_training_scope(
                    context.args, context.args.model
                ),
                "wandb_env_vars": [
                    "WANDB_DIR",
                    "WANDB_DATA_DIR",
                    "WANDB_CACHE_DIR",
                    "WANDB_CONFIG_DIR",
                ],
            },
        )
        run = init_wandb_run(context, trial, params)
        try:
            if context.args.model == "ts2vec":
                result = score_ts2vec(context, trial, params, trial_dir)
            else:
                result = score_classical_or_torch(context, trial, params)
            metrics = evaluate_scores(result.series_scores, context, result.series_info)
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
                save_json(
                    trial_dir / "error.json",
                    {"error": message, "traceback": traceback.format_exc()},
                )
                if run is not None:
                    run.log({"status": "pruned", "error": message})
                raise optuna.exceptions.TrialPruned(message) from exc
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error", message)
            save_json(
                trial_dir / "error.json", {"error": message, "traceback": traceback.format_exc()}
            )
            raise
        except Exception as exc:
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error", str(exc))
            save_json(
                trial_dir / "error.json", {"error": str(exc), "traceback": traceback.format_exc()}
            )
            if run is not None:
                run.log({"status": "failed", "error": str(exc)})
            raise
        finally:
            if run is not None:
                run.finish()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _objective


def load_dataset_and_labels(
    args: argparse.Namespace,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], Path | None]:
    labels_file = find_labels_file(
        args.labels_file, dataset_dir=args.dataset_dir, auto_download=True
    )
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
    output_root = (
        (args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_ROOT)
        .expanduser()
        .resolve()
    )
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


def final_split_names(
    args: argparse.Namespace, dataset: dict[str, pd.DataFrame], labels: dict[str, Any]
) -> dict[str, list[str]]:
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
    print(
        f"Effective training scope for {args.model}: {effective_training_scope(args, args.model)}"
    )
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
        metrics = evaluate_scores(
            split_result.series_scores, split_context, split_result.series_info
        )
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
        save_json(
            output_dir / f"operational_{split_name}_{args.model}.json", metrics["operational"]
        )
        if args.save_scores:
            save_predictions(output_dir, args.model, split_name, dataset, split_result, metrics)
        summary_rows.append(metric_summary_row(args.model, split_name, metrics))
        per_series.extend(per_series_rows(args.model, split_name, metrics))

    summary = pd.DataFrame(summary_rows).sort_values(["split", "event_f1"], ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(per_series).sort_values(["split", "series"]).to_csv(
        output_dir / "per_series_metrics.csv", index=False
    )
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
    print(
        f"Effective training scope for {args.model}: {effective_training_scope(args, args.model)}"
    )
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
        study.optimize(
            objective(context), n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True
        )

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
