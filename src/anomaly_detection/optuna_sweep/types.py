from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_detection.metrics import SeriesScores


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
