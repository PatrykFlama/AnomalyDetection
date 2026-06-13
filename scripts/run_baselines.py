#!/usr/bin/env python3
"""Run classical and neural anomaly-detection baselines on NAB time series."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.data import DATASET_DIR, load_dataset  # noqa: E402
from anomaly_detection.detectors import build_detector  # noqa: E402
from anomaly_detection.labels import (  # noqa: E402
    find_labels_file,
    labels_for_series,
    load_label_windows,
)
from anomaly_detection.metrics import SeriesScores, evaluate_series_scores, predict  # noqa: E402
from anomaly_detection.protocol import TrainingWindow, select_training_window  # noqa: E402
from anomaly_detection.splits import save_split, split_series_names  # noqa: E402

METHODS = (
    "robust-zscore",
    "rolling-residual",
    "isolation-forest",
    "mlp-autoencoder",
    "lstm",
    "gru",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train classical/neural anomaly detectors on a per-series normal prefix. "
            "By default, threshold quantile is selected on validation series and final "
            "metrics are reported on held-out test series."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--no-download-labels", action="store_true")
    parser.add_argument("--missing-labels-as-normal", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument(
        "--train-mode",
        choices=("before_first_anomaly", "fixed_prefix"),
        default="before_first_anomaly",
    )
    parser.add_argument(
        "--fallback-train-fraction",
        type=float,
        default=0.05,
        help="Used only for fixed_prefix mode or series without anomaly windows.",
    )
    parser.add_argument("--min-train-points", type=int, default=32)
    parser.add_argument("--allow-contaminated-train", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument(
        "--split",
        choices=("all", "validation", "test"),
        default="all",
        help="all tunes on validation and reports test; validation/test run only that subset.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument(
        "--threshold-quantiles",
        nargs="+",
        type=float,
        default=[0.99, 0.995],
        help="Candidate quantiles tuned on validation when --split all.",
    )
    parser.add_argument(
        "--threshold-source",
        choices=("train", "all"),
        default="train",
    )
    parser.add_argument(
        "--threshold-scope",
        choices=("per_series", "global"),
        default="per_series",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("event_f1", "point_f1", "point_pr_auc"),
        default="event_f1",
    )
    parser.add_argument("--threshold-sweep-steps", type=int, default=200)
    parser.add_argument("--rolling-window", type=int, default=64)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit_series is not None and args.limit_series <= 0:
        raise SystemExit("--limit-series must be positive.")
    if not 0 < args.fallback_train_fraction <= 1:
        raise SystemExit("--fallback-train-fraction must be in the interval (0, 1].")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in the interval (0, 1).")
    if args.min_train_points <= 1:
        raise SystemExit("--min-train-points must be greater than 1.")
    if args.threshold is None and not 0 <= args.threshold_quantile <= 1:
        raise SystemExit("--threshold-quantile must be in [0, 1].")
    if any(not 0 <= quantile <= 1 for quantile in args.threshold_quantiles):
        raise SystemExit("--threshold-quantiles must all be in [0, 1].")
    if args.rolling_window <= 0 or args.window <= 0:
        raise SystemExit("--rolling-window and --window must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive.")

    if args.methods == ["all"]:
        args.methods = list(METHODS)
    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {', '.join(unknown)}")


def make_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    for key, value in list(output.items()):
        if isinstance(value, Path):
            output[key] = str(value)
    return output


def subset_dataset(dataset: dict[str, pd.DataFrame], names: list[str]) -> dict[str, pd.DataFrame]:
    return {name: dataset[name] for name in names}


def prepare_training_windows(
    dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, TrainingWindow]:
    training_windows: dict[str, TrainingWindow] = {}
    for name, frame in dataset.items():
        labels = labels_for_series(
            label_windows,
            name,
            frame["timestamp"],
            missing_labels_as_normal=args.missing_labels_as_normal,
        )
        try:
            training_windows[name] = select_training_window(
                n_points=len(frame),
                labels=labels,
                fallback_train_fraction=args.fallback_train_fraction,
                mode=args.train_mode,
                min_length=args.min_train_points,
                allow_contaminated=args.allow_contaminated_train,
            )
        except ValueError as exc:
            raise SystemExit(f"{name}: {exc}") from exc
    return training_windows


def score_method(
    method: str,
    split_name: str,
    dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    training_windows: dict[str, TrainingWindow],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[SeriesScores], float]:
    started = time.perf_counter()
    series_scores: list[SeriesScores] = []
    scores_dir = output_dir / "scores" / split_name / method
    if args.save_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)

    for index, (name, frame) in enumerate(dataset.items(), start=1):
        print(f"[{split_name}/{method}] {index}/{len(dataset)} {name}")
        values = frame["value"].to_numpy(dtype=np.float32)
        detector = build_detector(
            method=method,
            rolling_window=args.rolling_window,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        detector.fit(values, training_windows[name].end)
        scores = detector.score(values)
        labels = labels_for_series(
            label_windows,
            name,
            frame["timestamp"],
            missing_labels_as_normal=args.missing_labels_as_normal,
        )
        series_scores.append(
            SeriesScores(
                name=name,
                scores=scores,
                labels=labels,
                train_end=training_windows[name].end,
            )
        )

        if args.save_scores:
            safe_name = name.replace("/", "__").replace(".csv", "")
            np.savez_compressed(
                scores_dir / f"{safe_name}.npz",
                timestamps=frame["timestamp"].astype(str).to_numpy(),
                values=values,
                scores=scores,
                labels=labels,
                train_end=training_windows[name].end,
            )

    return series_scores, time.perf_counter() - started


def evaluate_scores(
    series_scores: list[SeriesScores],
    args: argparse.Namespace,
    threshold_quantile: float,
) -> dict[str, Any]:
    return evaluate_series_scores(
        series_scores,
        threshold=args.threshold,
        threshold_quantile=threshold_quantile,
        threshold_sweep_steps=args.threshold_sweep_steps,
        threshold_source=args.threshold_source,
        threshold_scope=args.threshold_scope,
    )


def selection_metric(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "event_f1":
        value = metrics["aggregate"]["eventwise"]["f1"]
    elif metric_name == "point_f1":
        value = metrics["aggregate"]["pointwise"]["f1"]
    elif metric_name == "point_pr_auc":
        value = metrics["aggregate"]["pointwise"]["pr_auc"]
    else:
        raise ValueError(metric_name)
    return float("-inf") if value is None else float(value)


def tune_threshold_quantile(
    validation_scores: list[SeriesScores],
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    best_quantile = args.threshold_quantile
    best_metrics: dict[str, Any] | None = None
    best_value = float("-inf")
    for quantile in args.threshold_quantiles:
        metrics = evaluate_scores(validation_scores, args, quantile)
        value = selection_metric(metrics, args.selection_metric)
        if value > best_value:
            best_value = value
            best_quantile = quantile
            best_metrics = metrics
    assert best_metrics is not None
    best_metrics["selected_by"] = args.selection_metric
    return best_quantile, best_metrics


def summary_row(
    method: str,
    split_name: str,
    elapsed_seconds: float,
    threshold_quantile: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    pointwise = metrics["aggregate"]["pointwise"]
    eventwise = metrics["aggregate"]["eventwise"]
    diagnostics = metrics["threshold_diagnostics"]
    return {
        "method": method,
        "split": split_name,
        "elapsed_seconds": elapsed_seconds,
        "threshold_quantile": threshold_quantile,
        "threshold_source": metrics["threshold_source"],
        "threshold_scope": metrics["threshold_scope"],
        "point_precision": pointwise["precision"],
        "point_recall": pointwise["recall"],
        "point_f1": pointwise["f1"],
        "point_roc_auc": pointwise["roc_auc"],
        "point_pr_auc": pointwise["pr_auc"],
        "event_precision": eventwise["precision"],
        "event_recall": eventwise["recall"],
        "event_f1": eventwise["f1"],
        "point_best_f1_oracle": diagnostics["pointwise_best_f1"]["f1"],
        "event_best_f1_oracle": diagnostics["eventwise_best_f1"]["f1"],
    }


def per_series_metric_rows(
    method: str,
    split_name: str,
    threshold_quantile: float,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for item in metrics["per_series"]:
        pointwise = item["pointwise"]
        eventwise = item["eventwise"]
        rows.append(
            {
                "method": method,
                "split": split_name,
                "series": item["name"],
                "threshold_quantile": threshold_quantile,
                "threshold_source": metrics["threshold_source"],
                "threshold_scope": metrics["threshold_scope"],
                "threshold": item["threshold"],
                "n_points": item["n_points"],
                "n_train_points": item["n_train_points"],
                "n_evaluation_points": item["n_evaluation_points"],
                "n_anomaly_points": item["n_anomaly_points"],
                "n_true_events": item["n_true_events"],
                "n_pred_events": item["n_pred_events"],
                "train_end": item["train_end"],
                "point_precision": pointwise["precision"],
                "point_recall": pointwise["recall"],
                "point_f1": pointwise["f1"],
                "point_roc_auc": pointwise["roc_auc"],
                "point_pr_auc": pointwise["pr_auc"],
                "event_precision": eventwise["precision"],
                "event_recall": eventwise["recall"],
                "event_f1": eventwise["f1"],
                "event_true_events": eventwise["true_events"],
                "event_pred_events": eventwise["pred_events"],
                "event_matched_events": eventwise["matched_events"],
                "event_fp_events": eventwise["fp_events"],
                "event_fn_events": eventwise["fn_events"],
            }
        )
    return rows


def save_predictions(
    output_dir: Path,
    method: str,
    split_name: str,
    dataset: dict[str, pd.DataFrame],
    series_scores: list[SeriesScores],
    metrics: dict[str, Any],
) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in series_scores:
        frame = dataset[item.name]
        threshold = metrics["thresholds"][item.name]
        predictions = predict(item.scores, threshold)
        timestamps = frame["timestamp"].astype(str).to_numpy()
        values = frame["value"].to_numpy(dtype=np.float32)
        for index in range(item.train_end, len(item.scores)):
            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "series": item.name,
                    "index": index,
                    "timestamp": timestamps[index],
                    "value": values[index],
                    "score": item.scores[index],
                    "label": item.labels[index],
                    "threshold": threshold,
                    "prediction": predictions[index],
                    "train_end": item.train_end,
                }
            )
    pd.DataFrame(rows).to_csv(predictions_dir / f"{split_name}_{method}.csv", index=False)


def save_method_metrics(
    output_dir: Path,
    method: str,
    split_name: str,
    metrics: dict[str, Any],
    training_windows: dict[str, TrainingWindow],
    threshold_quantile: float,
) -> None:
    metrics["split"] = split_name
    metrics["selected_threshold_quantile"] = threshold_quantile
    metrics["training_windows"] = {
        name: window.to_dict() for name, window in training_windows.items()
    }
    save_json(output_dir / f"metrics_{split_name}_{method}.json", metrics)


def run_single_split(
    method: str,
    split_name: str,
    dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    training_windows: dict[str, TrainingWindow],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores, elapsed = score_method(
        method,
        split_name,
        dataset,
        label_windows,
        training_windows,
        args,
        output_dir,
    )
    metrics = evaluate_scores(scores, args, args.threshold_quantile)
    save_method_metrics(
        output_dir, method, split_name, metrics, training_windows, args.threshold_quantile
    )
    if args.save_scores:
        save_predictions(output_dir, method, split_name, dataset, scores, metrics)
    return (
        summary_row(method, split_name, elapsed, args.threshold_quantile, metrics),
        per_series_metric_rows(method, split_name, args.threshold_quantile, metrics),
    )


def run_validation_test(
    method: str,
    validation_dataset: dict[str, pd.DataFrame],
    test_dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    validation_windows: dict[str, TrainingWindow],
    test_windows: dict[str, TrainingWindow],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_scores, validation_elapsed = score_method(
        method,
        "validation",
        validation_dataset,
        label_windows,
        validation_windows,
        args,
        output_dir,
    )
    selected_quantile, validation_metrics = tune_threshold_quantile(validation_scores, args)
    save_method_metrics(
        output_dir,
        method,
        "validation",
        validation_metrics,
        validation_windows,
        selected_quantile,
    )
    if args.save_scores:
        save_predictions(
            output_dir,
            method,
            "validation",
            validation_dataset,
            validation_scores,
            validation_metrics,
        )

    test_scores, test_elapsed = score_method(
        method,
        "test",
        test_dataset,
        label_windows,
        test_windows,
        args,
        output_dir,
    )
    test_metrics = evaluate_scores(test_scores, args, selected_quantile)
    save_method_metrics(output_dir, method, "test", test_metrics, test_windows, selected_quantile)
    if args.save_scores:
        save_predictions(output_dir, method, "test", test_dataset, test_scores, test_metrics)

    return (
        [
            summary_row(
                method, "validation", validation_elapsed, selected_quantile, validation_metrics
            ),
            summary_row(method, "test", test_elapsed, selected_quantile, test_metrics),
        ],
        [
            *per_series_metric_rows(method, "validation", selected_quantile, validation_metrics),
            *per_series_metric_rows(method, "test", selected_quantile, test_metrics),
        ],
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = make_output_dir(args.output_dir)

    dataset = load_dataset(args.dataset_dir, args.categories)
    if args.limit_series is not None:
        dataset = dict(list(dataset.items())[: args.limit_series])
    if not dataset:
        raise SystemExit("No series were loaded. Check --dataset-dir and --categories.")

    labels_file = find_labels_file(
        args.labels_file,
        dataset_dir=args.dataset_dir,
        auto_download=not args.no_download_labels and not args.missing_labels_as_normal,
    )
    label_windows = load_label_windows(labels_file)
    split = split_series_names(list(dataset), args.validation_fraction, args.split_seed)
    save_split(output_dir / "splits.json", split)

    validation_dataset = subset_dataset(dataset, split.validation)
    test_dataset = subset_dataset(dataset, split.test)
    validation_windows = prepare_training_windows(validation_dataset, label_windows, args)
    test_windows = prepare_training_windows(test_dataset, label_windows, args)

    save_json(
        output_dir / "config.json",
        {
            "args": jsonable_args(args),
            "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
            "labels_file": None if labels_file is None else str(labels_file),
            "splits": split.to_dict(),
            "validation_training_windows": {
                name: window.to_dict() for name, window in validation_windows.items()
            },
            "test_training_windows": {
                name: window.to_dict() for name, window in test_windows.items()
            },
            "metrics_note": (
                "Oracle best-F1 diagnostics use labels and are not the primary thresholded result."
            ),
        },
    )

    print(f"Loaded {len(dataset)} series.")
    print(f"Validation series: {len(split.validation)}; test series: {len(split.test)}.")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Outputs: {output_dir}")

    summary_rows = []
    per_series_rows = []
    for method in args.methods:
        if args.split == "all":
            method_summary_rows, method_per_series_rows = run_validation_test(
                method,
                validation_dataset,
                test_dataset,
                label_windows,
                validation_windows,
                test_windows,
                args,
                output_dir,
            )
            summary_rows.extend(method_summary_rows)
            per_series_rows.extend(method_per_series_rows)
        elif args.split == "validation":
            method_summary_row, method_per_series_rows = run_single_split(
                method,
                "validation",
                validation_dataset,
                label_windows,
                validation_windows,
                args,
                output_dir,
            )
            summary_rows.append(method_summary_row)
            per_series_rows.extend(method_per_series_rows)
        else:
            method_summary_row, method_per_series_rows = run_single_split(
                method,
                "test",
                test_dataset,
                label_windows,
                test_windows,
                args,
                output_dir,
            )
            summary_rows.append(method_summary_row)
            per_series_rows.extend(method_per_series_rows)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "event_f1"], ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    per_series = pd.DataFrame(per_series_rows).sort_values(["split", "method", "series"])
    per_series.to_csv(output_dir / "per_series_metrics.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output_dir / 'summary.csv'}.")


if __name__ == "__main__":
    main()
