#!/usr/bin/env python3
"""Evaluate a trained TS2Vec model with the shared project protocol."""

from __future__ import annotations

import argparse
import json
import sys
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
from anomaly_detection.labels import (  # noqa: E402
    find_labels_file,
    labels_for_series,
    load_label_windows,
)
from anomaly_detection.metrics import SeriesScores, evaluate_series_scores, predict  # noqa: E402
from anomaly_detection.splits import save_split, split_series_names  # noqa: E402
from anomaly_detection.ts2vec_support import import_ts2vec, resolve_torch_device  # noqa: E402

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "ts2vec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TS2Vec anomaly scores on NAB.")
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sliding-length", type=int, default=32)
    parser.add_argument("--sliding-padding", type=int, default=200)
    parser.add_argument(
        "--score-method",
        choices=("knn", "centroid", "mask-diff"),
        default="knn",
    )
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument(
        "--reference-fraction",
        type=float,
        default=0.05,
        help="Fallback normal-prefix fraction for older TS2Vec runs without series.json.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
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
        "--threshold-quantiles",
        nargs="+",
        type=float,
        default=[0.99, 0.995],
        help="Candidate quantiles tuned on validation when --split all.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument(
        "--split",
        choices=("all", "validation", "test"),
        default="all",
        help="all tunes on validation and reports test; validation/test run only that subset.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("event_f1", "point_f1", "point_pr_auc"),
        default="event_f1",
    )
    parser.add_argument("--threshold-sweep-steps", type=int, default=200)
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit_series is not None and args.limit_series <= 0:
        raise SystemExit("--limit-series must be positive.")
    if not 0 < args.reference_fraction <= 1:
        raise SystemExit("--reference-fraction must be in the interval (0, 1].")
    if args.threshold is None and not 0 <= args.threshold_quantile <= 1:
        raise SystemExit("--threshold-quantile must be in [0, 1].")
    if any(not 0 <= quantile <= 1 for quantile in args.threshold_quantiles):
        raise SystemExit("--threshold-quantiles must all be in [0, 1].")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("--validation-fraction must be in the interval (0, 1).")
    if args.batch_size <= 0 or args.sliding_length <= 0 or args.sliding_padding < 0:
        raise SystemExit("--batch-size and --sliding-length must be positive.")
    if args.knn_k <= 0:
        raise SystemExit("--knn-k must be positive.")


def load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


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


def resolve_run_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    if args.run_dir is None and args.model_path is None:
        raise SystemExit("Provide --run-dir or --model-path.")

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    model_path = args.model_path or run_dir / "ts2vec_model.pt"
    metadata_path = args.metadata_path or (run_dir / "metadata.json" if run_dir else None)
    series_path = args.series_metadata_path or (run_dir / "series.json" if run_dir else None)
    return model_path.expanduser().resolve(), metadata_path, series_path, run_dir


def make_output_dir(output_dir: Path | None, run_dir: Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (run_dir / "evaluation" if run_dir else DEFAULT_OUTPUT_ROOT) / timestamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def model_kwargs(metadata: dict[str, Any], device: str, batch_size: int) -> dict[str, Any]:
    train_args = metadata.get("args", {})
    return {
        "input_dims": 1,
        "output_dims": int(train_args.get("output_dims", 320)),
        "hidden_dims": int(train_args.get("hidden_dims", 64)),
        "depth": int(train_args.get("depth", 10)),
        "device": device,
        "batch_size": batch_size,
    }


def series_metadata_map(series_metadata: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {} if series_metadata is None else {item["name"]: item for item in series_metadata}


def normalize_values(
    name: str,
    values: np.ndarray,
    normalization: str,
    metadata_by_series: dict[str, dict[str, Any]],
    fallback_fraction: float,
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if normalization == "none":
        return values

    stats = metadata_by_series.get(name, {}).get("normalization")
    if stats is not None and "mean" in stats and "std" in stats:
        mean = float(stats["mean"])
        std = float(stats["std"])
    else:
        prefix_len = max(2, int(np.ceil(len(values) * fallback_fraction)))
        prefix = values[:prefix_len]
        mean = float(prefix.mean())
        std = float(prefix.std())
    std = 1.0 if std == 0 else std
    return ((values - mean) / std).astype(np.float32)


def reference_length(
    name: str,
    n_points: int,
    metadata_by_series: dict[str, dict[str, Any]],
    fallback_fraction: float,
) -> int:
    if name in metadata_by_series and "train_length" in metadata_by_series[name]:
        return max(2, min(n_points, int(metadata_by_series[name]["train_length"])))
    if not 0 < fallback_fraction <= 1:
        raise SystemExit("--reference-fraction must be in the interval (0, 1].")
    return max(2, min(n_points, int(np.ceil(n_points * fallback_fraction))))


def encode_representations(
    model: Any,
    values: np.ndarray,
    batch_size: int,
    sliding_length: int,
    sliding_padding: int,
) -> np.ndarray:
    encoded = model.encode(
        values.reshape(1, -1, 1).astype(np.float32),
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
    masked = model.encode(
        data,
        mask="mask_last",
        causal=True,
        sliding_length=sliding_length,
        sliding_padding=sliding_padding,
        batch_size=batch_size,
    ).squeeze(0)
    unmasked = model.encode(
        data,
        causal=True,
        sliding_length=sliding_length,
        sliding_padding=sliding_padding,
        batch_size=batch_size,
    ).squeeze(0)
    return np.abs(unmasked - masked).sum(axis=1)[: len(values)]


def score_centroid(embeddings: np.ndarray, ref_len: int) -> np.ndarray:
    centroid = np.nanmean(embeddings[:ref_len], axis=0, keepdims=True)
    return np.linalg.norm(embeddings - centroid, axis=1)


def score_knn(embeddings: np.ndarray, ref_len: int, k: int, chunk_size: int = 4096) -> np.ndarray:
    reference = embeddings[:ref_len].astype(np.float32, copy=False)
    k = max(1, min(k, len(reference)))
    output = np.empty(len(embeddings), dtype=np.float32)
    ref_norm = np.sum(reference * reference, axis=1, keepdims=True).T
    for start in range(0, len(embeddings), chunk_size):
        end = min(len(embeddings), start + chunk_size)
        chunk = embeddings[start:end].astype(np.float32, copy=False)
        d2 = np.sum(chunk * chunk, axis=1, keepdims=True) + ref_norm - 2.0 * chunk @ reference.T
        nearest = np.partition(np.maximum(d2, 0.0), kth=k - 1, axis=1)[:, :k]
        output[start:end] = np.sqrt(np.mean(nearest, axis=1))
    return output


def score_series(
    model: Any,
    values: np.ndarray,
    args: argparse.Namespace,
    ref_len: int,
) -> np.ndarray:
    if args.score_method == "mask-diff":
        scores = score_mask_difference(
            model,
            values,
            args.batch_size,
            args.sliding_length,
            args.sliding_padding,
        )
    else:
        embeddings = encode_representations(
            model,
            values,
            args.batch_size,
            args.sliding_length,
            args.sliding_padding,
        )
        scores = (
            score_centroid(embeddings, ref_len)
            if args.score_method == "centroid"
            else score_knn(embeddings, ref_len, args.knn_k)
        )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_scores(
    output_dir: Path,
    split_name: str,
    name: str,
    timestamps: pd.Series,
    scores: np.ndarray,
    labels: np.ndarray,
    train_end: int,
) -> None:
    safe_name = name.replace("/", "__").replace(".csv", "")
    scores_dir = output_dir / "scores" / split_name
    scores_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scores_dir / f"{safe_name}.npz",
        timestamps=timestamps.astype(str).to_numpy(),
        scores=scores,
        labels=labels,
        train_end=train_end,
    )


def score_dataset(
    model: Any,
    split_name: str,
    dataset: dict[str, pd.DataFrame],
    label_windows: dict[str, Any],
    metadata_by_series: dict[str, dict[str, Any]],
    normalization: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[SeriesScores]:
    evaluated: list[SeriesScores] = []
    for index, (name, frame) in enumerate(dataset.items(), start=1):
        print(f"[{split_name}/ts2vec] {index}/{len(dataset)} {name}")
        raw_values = frame["value"].to_numpy(dtype=np.float32)
        values = normalize_values(
            name,
            raw_values,
            normalization,
            metadata_by_series,
            args.reference_fraction,
        )
        ref_len = reference_length(name, len(values), metadata_by_series, args.reference_fraction)
        scores = score_series(model, values, args, ref_len)
        labels = labels_for_series(
            label_windows,
            name,
            frame["timestamp"],
            missing_labels_as_normal=args.missing_labels_as_normal,
        )
        evaluated.append(SeriesScores(name=name, scores=scores, labels=labels, train_end=ref_len))
        if args.save_scores:
            save_scores(output_dir, split_name, name, frame["timestamp"], scores, labels, ref_len)
    return evaluated


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
    split_name: str,
    threshold_quantile: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    pointwise = metrics["aggregate"]["pointwise"]
    eventwise = metrics["aggregate"]["eventwise"]
    diagnostics = metrics["threshold_diagnostics"]
    return {
        "method": "ts2vec",
        "split": split_name,
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
                "method": "ts2vec",
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
                    "method": "ts2vec",
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
    pd.DataFrame(rows).to_csv(predictions_dir / f"{split_name}_ts2vec.csv", index=False)


def base_metadata(
    args: argparse.Namespace,
    model_path: Path,
    metadata_path: Path | None,
    series_path: Path | None,
    dataset_dir: Path,
    labels_file: Path | None,
    ts2vec_source: Path | None,
    device: str,
    categories: list[str] | None,
    normalization: str,
) -> dict[str, Any]:
    return {
        "model_path": str(model_path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "series_metadata_path": None if series_path is None else str(series_path),
        "dataset_dir": str(dataset_dir.expanduser().resolve()),
        "labels_file": None if labels_file is None else str(labels_file),
        "ts2vec_source": None if ts2vec_source is None else str(ts2vec_source),
        "device": device,
        "categories": categories,
        "normalization": normalization,
        "score_method": args.score_method,
        "knn_k": args.knn_k if args.score_method == "knn" else None,
        "args": jsonable_args(args),
    }


def save_metrics(
    output_dir: Path,
    split_name: str,
    metrics: dict[str, Any],
    threshold_quantile: float,
    metadata: dict[str, Any],
) -> None:
    metrics.update(metadata)
    metrics["split"] = split_name
    metrics["selected_threshold_quantile"] = threshold_quantile
    save_json(output_dir / f"metrics_{split_name}_ts2vec.json", metrics)


def main() -> None:
    args = parse_args()
    validate_args(args)
    model_path, metadata_path, series_path, run_dir = resolve_run_paths(args)
    metadata = load_json(metadata_path) or {}
    metadata_by_series = series_metadata_map(load_json(series_path))

    dataset_dir = args.dataset_dir or Path(metadata.get("dataset_dir", DATASET_DIR))
    categories = args.categories or metadata.get("categories")
    normalization = metadata.get("normalization", "per-series")

    TS2Vec, ts2vec_source = import_ts2vec(args.ts2vec_dir)
    device = resolve_torch_device(args.device)
    model = TS2Vec(**model_kwargs(metadata, device, args.batch_size))
    model.load(str(model_path))

    labels_file = find_labels_file(
        args.labels_file,
        dataset_dir=dataset_dir,
        auto_download=not args.no_download_labels and not args.missing_labels_as_normal,
    )
    label_windows = load_label_windows(labels_file)
    dataset = load_dataset(dataset_dir, categories)
    if args.limit_series is not None:
        dataset = dict(list(dataset.items())[: args.limit_series])
    if not dataset:
        raise SystemExit("No series were loaded. Check --dataset-dir and --categories.")
    output_dir = make_output_dir(args.output_dir, run_dir)
    split = split_series_names(list(dataset), args.validation_fraction, args.split_seed)
    save_split(output_dir / "splits.json", split)

    result_metadata = base_metadata(
        args=args,
        model_path=model_path,
        metadata_path=metadata_path,
        series_path=series_path,
        dataset_dir=dataset_dir,
        labels_file=labels_file,
        ts2vec_source=ts2vec_source,
        device=device,
        categories=categories,
        normalization=normalization,
    )
    save_json(
        output_dir / "config.json",
        {
            **result_metadata,
            "splits": split.to_dict(),
            "metrics_note": (
                "Oracle best-F1 diagnostics use labels and are not the primary thresholded result."
            ),
        },
    )

    print(f"Loaded model from {model_path}.")
    print(f"Scoring {len(dataset)} series on {device}; outputs: {output_dir}.")
    print(f"Validation series: {len(split.validation)}; test series: {len(split.test)}.")

    summary_rows = []
    per_series_rows = []

    if args.split == "all":
        validation_dataset = subset_dataset(dataset, split.validation)
        validation_scores = score_dataset(
            model,
            "validation",
            validation_dataset,
            label_windows,
            metadata_by_series,
            normalization,
            args,
            output_dir,
        )
        selected_quantile, validation_metrics = tune_threshold_quantile(validation_scores, args)
        save_metrics(
            output_dir,
            "validation",
            validation_metrics,
            selected_quantile,
            result_metadata,
        )
        if args.save_scores:
            save_predictions(
                output_dir,
                "validation",
                validation_dataset,
                validation_scores,
                validation_metrics,
            )

        test_dataset = subset_dataset(dataset, split.test)
        test_scores = score_dataset(
            model,
            "test",
            test_dataset,
            label_windows,
            metadata_by_series,
            normalization,
            args,
            output_dir,
        )
        test_metrics = evaluate_scores(test_scores, args, selected_quantile)
        save_metrics(output_dir, "test", test_metrics, selected_quantile, result_metadata)
        if args.save_scores:
            save_predictions(output_dir, "test", test_dataset, test_scores, test_metrics)

        summary_rows.extend(
            [
                summary_row("validation", selected_quantile, validation_metrics),
                summary_row("test", selected_quantile, test_metrics),
            ]
        )
        per_series_rows.extend(
            [
                *per_series_metric_rows("validation", selected_quantile, validation_metrics),
                *per_series_metric_rows("test", selected_quantile, test_metrics),
            ]
        )
    else:
        split_names = split.validation if args.split == "validation" else split.test
        selected_dataset = subset_dataset(dataset, split_names)
        scores = score_dataset(
            model,
            args.split,
            selected_dataset,
            label_windows,
            metadata_by_series,
            normalization,
            args,
            output_dir,
        )
        metrics = evaluate_scores(scores, args, args.threshold_quantile)
        save_metrics(output_dir, args.split, metrics, args.threshold_quantile, result_metadata)
        save_json(output_dir / "metrics_ts2vec.json", metrics)
        if args.save_scores:
            save_predictions(output_dir, args.split, selected_dataset, scores, metrics)
        summary_rows.append(summary_row(args.split, args.threshold_quantile, metrics))
        per_series_rows.extend(per_series_metric_rows(args.split, args.threshold_quantile, metrics))

    summary = pd.DataFrame(summary_rows).sort_values(["split", "event_f1"], ascending=False)
    summary.to_csv(output_dir / "summary_ts2vec.csv", index=False)
    per_series = pd.DataFrame(per_series_rows).sort_values(["split", "series"])
    per_series.to_csv(output_dir / "per_series_metrics_ts2vec.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {output_dir / 'summary_ts2vec.csv'}.")


if __name__ == "__main__":
    main()
