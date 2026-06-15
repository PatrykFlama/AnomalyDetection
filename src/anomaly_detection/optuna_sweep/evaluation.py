from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anomaly_detection.metrics import (
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
from anomaly_detection.optuna_sweep.config import (
    prediction_postprocessing_kwargs,
    protocol_config,
)
from anomaly_detection.optuna_sweep.protocol import category_of
from anomaly_detection.optuna_sweep.types import SweepContext, TrialResult


def event_auc_placeholders(metrics: dict[str, Any]) -> None:
    sections = [
        metrics["aggregate"],
        metrics["macro_average"],
        metrics["category_macro_average"],
        *metrics.get("category_macro", {}).values(),
    ]
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
        threshold = choose_threshold(
            threshold_source,
            None,
            context.args.protocol_threshold_quantile,
        )
        thresholds = {item.name: threshold for item in series_scores}
    else:
        for item in series_scores:
            eval_scores = item.scores[item.train_end :]
            threshold_source = (
                item.scores[: item.train_end]
                if context.args.threshold_source == "train"
                else eval_scores
            )
            thresholds[item.name] = choose_threshold(
                threshold_source,
                None,
                context.args.protocol_threshold_quantile,
            )

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
        pred = predict(
            eval_scores,
            thresholds[item.name],
            **postprocess_kwargs,
        )
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
        "pointwise": point_metrics_from_predictions(
            all_labels,
            all_scores,
            np.concatenate(all_pred),
        ),
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
        SeriesScores(
            item.name,
            item.scores[item.train_end :],
            item.labels[item.train_end :],
            0,
        )
        for item in series_scores
    ]
    metrics = {
        "threshold": (
            None
            if context.args.threshold_scope == "per_series"
            else next(iter(thresholds.values()))
        ),
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


def metric_summary_row(
    model: str,
    split_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
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


def per_series_rows(
    model: str,
    split_name: str,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
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
        predictions[item.train_end :] = predict(
            eval_scores,
            threshold,
            **postprocess_kwargs,
        )
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
    pd.DataFrame(rows).to_csv(
        predictions_dir / f"{split_name}_{model}.csv",
        index=False,
    )


def filter_result(result: TrialResult, names: set[str]) -> TrialResult:
    return TrialResult(
        series_scores=[item for item in result.series_scores if item.name in names],
        operational=result.operational,
        artifact_paths=result.artifact_paths,
        series_info={key: value for key, value in result.series_info.items() if key in names},
    )
