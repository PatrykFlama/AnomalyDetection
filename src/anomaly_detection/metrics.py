from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


@dataclass(frozen=True)
class SeriesScores:
    name: str
    scores: np.ndarray
    labels: np.ndarray
    train_end: int


ThresholdScope = str


def contiguous_events(binary: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return []
    changes = np.diff(binary.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _validate_event_postprocessing(
    min_event_length: int,
    merge_gap: int,
    cooldown: int,
    alert_mode: str,
    alert_window: int,
    max_alerts_per_event: int,
    peak_min_distance: int | None,
) -> None:
    if min_event_length < 1:
        raise ValueError("min_event_length must be at least 1")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    if cooldown < 0:
        raise ValueError("cooldown must be non-negative")
    if alert_mode not in {"full_event", "peak_window"}:
        raise ValueError("alert_mode must be 'full_event' or 'peak_window'")
    if alert_window < 1:
        raise ValueError("alert_window must be at least 1")
    if max_alerts_per_event < 1:
        raise ValueError("max_alerts_per_event must be at least 1")
    if peak_min_distance is not None and peak_min_distance < 1:
        raise ValueError("peak_min_distance must be at least 1 when provided")


def postprocess_prediction_events(
    pred: np.ndarray,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
) -> list[tuple[int, int]]:
    pred_bool = np.asarray(pred, dtype=bool)
    events = contiguous_events(pred_bool)
    if not events:
        return []

    if merge_gap > 0:
        merged = [events[0]]
        for start, end in events[1:]:
            previous_start, previous_end = merged[-1]
            if start - previous_end <= merge_gap:
                merged[-1] = (previous_start, end)
            else:
                merged.append((start, end))
        events = merged

    if min_event_length > 1:
        events = [(start, end) for start, end in events if end - start >= min_event_length]

    if cooldown > 0:
        kept: list[tuple[int, int]] = []
        suppress_until = -1
        for start, end in events:
            if start < suppress_until:
                continue
            kept.append((start, end))
            suppress_until = end + cooldown
        events = kept

    return events


def materialize_alerts(
    events: list[tuple[int, int]],
    n_points: int,
    scores: np.ndarray | None = None,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> np.ndarray:
    output = np.zeros(n_points, dtype=np.int8)
    if not events:
        return output

    if alert_mode == "full_event":
        for start, end in events:
            output[start:end] = 1
        return output

    if scores is None:
        raise ValueError("scores are required when alert_mode='peak_window'")
    score_values = np.asarray(scores, dtype=np.float32)
    if score_values.shape[0] != n_points:
        raise ValueError("scores and predictions must have the same length")

    before = (alert_window - 1) // 2
    after = alert_window - 1 - before
    min_distance = peak_min_distance if peak_min_distance is not None else alert_window
    for start, end in events:
        event_scores = score_values[start:end]
        finite_scores = np.where(np.isfinite(event_scores), event_scores, -np.inf)
        selected_peaks: list[int] = []
        if finite_scores.size == 0 or np.all(np.isneginf(finite_scores)):
            selected_peaks.append(start + max(0, (end - start - 1) // 2))
        else:
            candidates = finite_scores.copy()
            for _ in range(min(max_alerts_per_event, len(candidates))):
                if np.all(np.isneginf(candidates)):
                    break
                local_peak = int(np.argmax(candidates))
                selected_peaks.append(start + local_peak)
                suppress_start = max(0, local_peak - min_distance + 1)
                suppress_end = min(len(candidates), local_peak + min_distance)
                candidates[suppress_start:suppress_end] = -np.inf

        for peak in selected_peaks:
            alert_start = max(0, peak - before)
            alert_end = min(n_points, peak + after + 1)
            output[alert_start:alert_end] = 1
    return output


def postprocess_predictions(
    pred: np.ndarray,
    scores: np.ndarray | None = None,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> np.ndarray:
    """Convert threshold crossings into operational alarms.

    The event cleanup stage merges nearby threshold crossings, filters very short
    alarms, and applies cooldown. The alert materialization stage then either
    emits the full event span or a fixed-size window around the peak score in
    each kept event. This is deliberately model-agnostic: every detector still
    produces point scores, and the shared protocol decides how scores become
    alarms.
    """

    _validate_event_postprocessing(
        min_event_length,
        merge_gap,
        cooldown,
        alert_mode,
        alert_window,
        max_alerts_per_event,
        peak_min_distance,
    )
    pred_bool = np.asarray(pred, dtype=bool)
    events = postprocess_prediction_events(
        pred_bool,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
        cooldown=cooldown,
    )
    return materialize_alerts(
        events,
        len(pred_bool),
        scores=scores,
        alert_mode=alert_mode,
        alert_window=alert_window,
        max_alerts_per_event=max_alerts_per_event,
        peak_min_distance=peak_min_distance,
    )


def predict(
    scores: np.ndarray,
    threshold: float,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> np.ndarray:
    score_values = np.asarray(scores)
    pred = (score_values > threshold).astype(np.int8)
    if (
        min_event_length == 1
        and merge_gap == 0
        and cooldown == 0
        and alert_mode == "full_event"
    ):
        return pred
    return postprocess_predictions(
        pred,
        scores=score_values,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
        cooldown=cooldown,
        alert_mode=alert_mode,
        alert_window=alert_window,
        max_alerts_per_event=max_alerts_per_event,
        peak_min_distance=peak_min_distance,
    )


def choose_threshold(scores: np.ndarray, threshold: float | None, quantile: float) -> float:
    if threshold is not None:
        return float(threshold)
    if not 0 <= quantile <= 1:
        raise ValueError("threshold quantile must be in [0, 1]")
    finite = np.asarray(scores, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, quantile))


def confusion_counts(labels: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    labels_bool = np.asarray(labels, dtype=bool)
    pred_bool = np.asarray(pred, dtype=bool)
    return {
        "tp": int(np.logical_and(labels_bool, pred_bool).sum()),
        "tn": int(np.logical_and(~labels_bool, ~pred_bool).sum()),
        "fp": int(np.logical_and(~labels_bool, pred_bool).sum()),
        "fn": int(np.logical_and(labels_bool, ~pred_bool).sum()),
    }


def binary_metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
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
        "support": int(np.asarray(labels).sum()),
    }


def safe_auc(kind: str, labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return None
    if kind == "roc_auc":
        return float(roc_auc_score(labels, scores))
    if kind == "pr_auc":
        return float(average_precision_score(labels, scores))
    raise ValueError(kind)


def point_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> dict[str, Any]:
    pred = predict(
        scores,
        threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
        cooldown=cooldown,
        alert_mode=alert_mode,
        alert_window=alert_window,
        max_alerts_per_event=max_alerts_per_event,
        peak_min_distance=peak_min_distance,
    )
    return {
        **binary_metrics(labels, pred),
        "roc_auc": safe_auc("roc_auc", labels, scores),
        "pr_auc": safe_auc("pr_auc", labels, scores),
    }


def point_metrics_from_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    return {
        **binary_metrics(labels, pred),
        "roc_auc": safe_auc("roc_auc", labels, scores),
        "pr_auc": safe_auc("pr_auc", labels, scores),
    }


def event_overlap(
    true_event: tuple[int, int],
    pred_event: tuple[int, int],
) -> tuple[int, float, float]:
    true_start, true_end = true_event
    pred_start, pred_end = pred_event
    overlap = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    true_len = max(1, true_end - true_start)
    pred_len = max(1, pred_end - pred_start)
    return overlap, overlap / true_len, overlap / pred_len


def event_counts(
    labels: np.ndarray,
    pred: np.ndarray,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
) -> dict[str, int]:
    true_events = contiguous_events(labels)
    pred_events = contiguous_events(pred)
    candidates: list[tuple[int, int, int, int]] = []
    for true_index, true_event in enumerate(true_events):
        for pred_index, pred_event in enumerate(pred_events):
            overlap, true_fraction, pred_fraction = event_overlap(true_event, pred_event)
            if (
                overlap >= min_overlap_points
                and true_fraction >= min_true_overlap_fraction
                and pred_fraction >= min_pred_overlap_fraction
            ):
                detection_index = max(true_event[0], pred_event[0])
                detection_delay = detection_index - true_event[0]
                candidates.append((overlap, detection_delay, true_index, pred_index))

    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    detection_delays = []
    for _, detection_delay, true_index, pred_index in sorted(
        candidates, key=lambda item: (-item[0], item[1])
    ):
        if true_index in matched_true or pred_index in matched_pred:
            continue
        matched_true.add(true_index)
        matched_pred.add(pred_index)
        detection_delays.append(detection_delay)

    matched = len(matched_true)
    return {
        "true_events": len(true_events),
        "pred_events": len(pred_events),
        "matched_events": matched,
        "fp_events": len(pred_events) - matched,
        "fn_events": len(true_events) - matched,
        "detection_delay_sum": int(sum(detection_delays)),
        "detection_delay_max": int(max(detection_delays)) if detection_delays else 0,
    }


def event_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int | None]:
    matched = counts["matched_events"]
    pred_events = counts["pred_events"]
    true_events = counts["true_events"]
    precision = matched / pred_events if pred_events else 0.0
    recall = matched / true_events if true_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_delay = counts["detection_delay_sum"] / matched if matched else None
    max_delay = counts["detection_delay_max"] if matched else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_detection_delay": mean_delay,
        "max_detection_delay": max_delay,
        **counts,
    }


def event_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> dict[str, float | int]:
    return event_metrics_from_counts(
        event_counts(
            labels,
            predict(
                scores,
                threshold,
                min_event_length=min_event_length,
                merge_gap=merge_gap,
                cooldown=cooldown,
                alert_mode=alert_mode,
                alert_window=alert_window,
                max_alerts_per_event=max_alerts_per_event,
                peak_min_distance=peak_min_distance,
            ),
            min_overlap_points=min_overlap_points,
            min_true_overlap_fraction=min_true_overlap_fraction,
            min_pred_overlap_fraction=min_pred_overlap_fraction,
        )
    )


def threshold_grid(scores: np.ndarray, steps: int) -> np.ndarray:
    finite = np.asarray(scores, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([0.0], dtype=np.float32)
    if steps <= 1:
        return np.array([float(np.median(finite))], dtype=np.float32)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, steps)))


def sweep_point_f1(
    labels: np.ndarray,
    scores: np.ndarray,
    steps: int,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> dict[str, Any]:
    best: dict[str, Any] = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in threshold_grid(scores, steps):
        metrics = binary_metrics(
            labels,
            predict(
                scores,
                float(threshold),
                min_event_length=min_event_length,
                merge_gap=merge_gap,
                cooldown=cooldown,
                alert_mode=alert_mode,
                alert_window=alert_window,
                max_alerts_per_event=max_alerts_per_event,
                peak_min_distance=peak_min_distance,
            ),
        )
        if metrics["f1"] > best["f1"]:
            best = {"threshold": float(threshold), **metrics}
    return best


def sweep_event_f1(
    series: Iterable[SeriesScores],
    steps: int,
    min_overlap_points: int = 1,
    min_true_overlap_fraction: float = 0.0,
    min_pred_overlap_fraction: float = 0.0,
    min_event_length: int = 1,
    merge_gap: int = 0,
    cooldown: int = 0,
    alert_mode: str = "full_event",
    alert_window: int = 1,
    max_alerts_per_event: int = 1,
    peak_min_distance: int | None = None,
) -> dict[str, Any]:
    series = list(series)
    all_scores = np.concatenate([item.scores for item in series])
    best: dict[str, Any] = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in threshold_grid(all_scores, steps):
        counts = {
            "true_events": 0,
            "pred_events": 0,
            "matched_events": 0,
            "fp_events": 0,
            "fn_events": 0,
            "detection_delay_sum": 0,
            "detection_delay_max": 0,
        }
        for item in series:
            item_counts = event_counts(
                item.labels,
                predict(
                    item.scores,
                    float(threshold),
                    min_event_length=min_event_length,
                    merge_gap=merge_gap,
                    cooldown=cooldown,
                    alert_mode=alert_mode,
                    alert_window=alert_window,
                    max_alerts_per_event=max_alerts_per_event,
                    peak_min_distance=peak_min_distance,
                ),
                min_overlap_points=min_overlap_points,
                min_true_overlap_fraction=min_true_overlap_fraction,
                min_pred_overlap_fraction=min_pred_overlap_fraction,
            )
            for key in counts:
                if key == "detection_delay_max":
                    counts[key] = max(counts[key], item_counts[key])
                else:
                    counts[key] += item_counts[key]
        metrics = event_metrics_from_counts(counts)
        if metrics["f1"] > best["f1"]:
            best = {"threshold": float(threshold), **metrics}
    return best


def aggregate_event_metrics(
    series: Iterable[SeriesScores], threshold: float
) -> dict[str, float | int]:
    counts = {
        "true_events": 0,
        "pred_events": 0,
        "matched_events": 0,
        "fp_events": 0,
        "fn_events": 0,
        "detection_delay_sum": 0,
        "detection_delay_max": 0,
    }
    for item in series:
        item_counts = event_counts(item.labels, predict(item.scores, threshold))
        for key in counts:
            if key == "detection_delay_max":
                counts[key] = max(counts[key], item_counts[key])
            else:
                counts[key] += item_counts[key]
    return event_metrics_from_counts(counts)


def aggregate_event_metrics_with_thresholds(
    series: Iterable[SeriesScores],
    thresholds: dict[str, float],
) -> dict[str, float | int]:
    counts = {
        "true_events": 0,
        "pred_events": 0,
        "matched_events": 0,
        "fp_events": 0,
        "fn_events": 0,
        "detection_delay_sum": 0,
        "detection_delay_max": 0,
    }
    for item in series:
        item_counts = event_counts(item.labels, predict(item.scores, thresholds[item.name]))
        for key in counts:
            if key == "detection_delay_max":
                counts[key] = max(counts[key], item_counts[key])
            else:
                counts[key] += item_counts[key]
    return event_metrics_from_counts(counts)


def macro_average(items: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for item in items for key in item})
    output: dict[str, float | None] = {}
    sum_keys = {
        "support",
        "tp",
        "tn",
        "fp",
        "fn",
        "true_events",
        "pred_events",
        "matched_events",
        "detection_delay_sum",
    }
    for key in keys:
        values = [item[key] for item in items if item.get(key) is not None]
        if not values:
            output[key] = None
        elif key in sum_keys or key.endswith("_events"):
            output[key] = float(sum(float(value) for value in values))
        else:
            output[key] = float(np.mean(values))
    return output


def evaluate_series_scores(
    series: list[SeriesScores],
    threshold: float | None = None,
    threshold_quantile: float = 0.995,
    threshold_sweep_steps: int = 200,
    threshold_source: str = "train",
    threshold_scope: ThresholdScope = "per_series",
) -> dict[str, Any]:
    eval_scores_by_series = [item.scores[item.train_end :] for item in series]
    eval_labels_by_series = [item.labels[item.train_end :] for item in series]
    if any(len(scores) == 0 for scores in eval_scores_by_series):
        raise ValueError("Every series must contain evaluation points after the training window.")

    eval_series = [
        SeriesScores(
            name=item.name,
            scores=scores,
            labels=labels,
            train_end=0,
        )
        for item, scores, labels in zip(
            series, eval_scores_by_series, eval_labels_by_series, strict=True
        )
    ]

    all_scores = np.concatenate(eval_scores_by_series)
    all_labels = np.concatenate(eval_labels_by_series)
    if threshold_source not in {"train", "all"}:
        raise ValueError("threshold_source must be 'train' or 'all'")
    if threshold_scope not in {"per_series", "global"}:
        raise ValueError("threshold_scope must be 'per_series' or 'global'")

    if threshold_scope == "global":
        threshold_scores = (
            np.concatenate([item.scores[: item.train_end] for item in series])
            if threshold_source == "train"
            else all_scores
        )
        selected_threshold = choose_threshold(threshold_scores, threshold, threshold_quantile)
        thresholds = {item.name: selected_threshold for item in series}
    else:
        thresholds = {}
        for item, eval_scores in zip(series, eval_scores_by_series, strict=True):
            threshold_scores = (
                item.scores[: item.train_end] if threshold_source == "train" else eval_scores
            )
            thresholds[item.name] = choose_threshold(
                threshold_scores, threshold, threshold_quantile
            )
        selected_threshold = None

    all_pred = np.concatenate(
        [
            predict(eval_scores, thresholds[item.name])
            for item, eval_scores in zip(series, eval_scores_by_series, strict=True)
        ]
    )
    aggregate_pointwise = point_metrics_from_predictions(all_labels, all_scores, all_pred)
    aggregate_eventwise = aggregate_event_metrics_with_thresholds(eval_series, thresholds)

    per_series = []
    for item, eval_scores, eval_labels in zip(
        series, eval_scores_by_series, eval_labels_by_series, strict=True
    ):
        per_series.append(
            {
                "name": item.name,
                "n_points": int(len(item.labels)),
                "n_train_points": int(item.train_end),
                "n_evaluation_points": int(len(eval_labels)),
                "n_anomaly_points": int(eval_labels.sum()),
                "n_true_events": len(contiguous_events(eval_labels)),
                "n_pred_events": len(
                    contiguous_events(predict(eval_scores, thresholds[item.name]))
                ),
                "train_end": int(item.train_end),
                "threshold": thresholds[item.name],
                "pointwise": point_metrics(eval_labels, eval_scores, thresholds[item.name]),
                "eventwise": event_metrics(eval_labels, eval_scores, thresholds[item.name]),
            }
        )

    return {
        "threshold": selected_threshold,
        "thresholds": thresholds,
        "threshold_quantile": None if threshold is not None else threshold_quantile,
        "threshold_source": threshold_source,
        "threshold_scope": threshold_scope,
        "aggregate": {
            "pointwise": aggregate_pointwise,
            "eventwise": aggregate_eventwise,
        },
        "macro_average": {
            "pointwise": macro_average([item["pointwise"] for item in per_series]),
            "eventwise": macro_average([item["eventwise"] for item in per_series]),
        },
        "threshold_diagnostics": {
            "pointwise_best_f1": sweep_point_f1(all_labels, all_scores, threshold_sweep_steps),
            "eventwise_best_f1": sweep_event_f1(eval_series, threshold_sweep_steps),
        },
        "per_series": per_series,
    }
