from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from anomaly_detection.metrics import contiguous_events
from anomaly_detection.preprocessing import train_prefix_length

TrainMode = Literal["fixed_prefix", "before_first_anomaly"]


@dataclass(frozen=True)
class TrainingWindow:
    """A documented, reproducible training segment for one time series."""

    start: int
    end: int
    mode: TrainMode
    requested_fraction: float
    anomaly_points: int
    n_points: int
    note: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def fraction(self) -> float:
        return self.length / self.n_points if self.n_points else 0.0

    @property
    def is_clean(self) -> bool:
        return self.anomaly_points == 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["length"] = self.length
        payload["fraction"] = self.fraction
        payload["is_clean"] = self.is_clean
        return payload


def validate_labels(labels: np.ndarray, n_points: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    if len(labels) != n_points:
        raise ValueError(f"Label length mismatch: expected {n_points}, got {len(labels)}")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1 values.")
    return labels


def first_anomaly_index(labels: np.ndarray) -> int | None:
    indices = np.flatnonzero(labels.astype(bool))
    if indices.size == 0:
        return None
    return int(indices[0])


def select_training_window(
    n_points: int,
    labels: np.ndarray,
    fallback_train_fraction: float,
    mode: TrainMode = "before_first_anomaly",
    min_length: int = 2,
    allow_contaminated: bool = False,
) -> TrainingWindow:
    """Select and validate a normal training prefix.

    The default mode uses the full labelled-clean prefix before the first anomaly.
    The fixed-prefix mode is available as an ablation and is rejected by default if
    official labels show contamination inside the selected training prefix.
    """
    labels = validate_labels(labels, n_points)
    if mode == "fixed_prefix":
        end = train_prefix_length(n_points, fallback_train_fraction, min_length=min_length)
        note = "requested fixed prefix"
    elif mode == "before_first_anomaly":
        first_anomaly = first_anomaly_index(labels)
        fallback_end = train_prefix_length(n_points, fallback_train_fraction, min_length=min_length)
        end = first_anomaly if first_anomaly is not None else fallback_end
        note = "prefix ending before the first labelled anomaly"
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    if end < min_length:
        raise ValueError(
            f"Training prefix has only {end} points; need at least {min_length}. "
            "Use a different category/series or an explicit protocol."
        )
    if end >= n_points:
        raise ValueError("Training window consumes the whole series; no evaluation points remain.")
    anomaly_points = int(labels[:end].sum())
    if anomaly_points and not allow_contaminated:
        events = contiguous_events(labels[:end])
        raise ValueError(
            f"Training window contains {anomaly_points} labelled anomaly points "
            f"in {len(events)} event(s). Lower --fallback-train-fraction, use "
            "--train-mode before_first_anomaly, or pass --allow-contaminated-train "
            "only for a clearly labelled ablation."
        )

    return TrainingWindow(
        start=0,
        end=int(end),
        mode=mode,
        requested_fraction=float(fallback_train_fraction),
        anomaly_points=anomaly_points,
        n_points=n_points,
        note=note,
    )
