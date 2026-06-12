#!/usr/bin/env python3
"""Audit NAB labels and training-prefix cleanliness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from anomaly_detection.metrics import contiguous_events  # noqa: E402
from anomaly_detection.protocol import select_training_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check NAB anomaly positions and train prefixes.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--categories", nargs="+", default=["realKnownCause"])
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--fallback-train-fraction", type=float, default=0.05)
    parser.add_argument(
        "--train-mode",
        choices=("fixed_prefix", "before_first_anomaly"),
        default="before_first_anomaly",
    )
    parser.add_argument("--min-train-points", type=int, default=32)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_file = find_labels_file(args.labels_file, dataset_dir=args.dataset_dir)
    label_windows = load_label_windows(labels_file)
    dataset = load_dataset(args.dataset_dir, args.categories)

    rows = []
    for name, frame in dataset.items():
        labels = labels_for_series(label_windows, name, frame["timestamp"])
        anomaly_indices = np.flatnonzero(labels)
        first_anomaly = int(anomaly_indices[0]) if anomaly_indices.size else None
        first_anomaly_fraction = first_anomaly / len(labels) if first_anomaly is not None else None
        events = contiguous_events(labels)
        try:
            window = select_training_window(
                n_points=len(labels),
                labels=labels,
                fallback_train_fraction=args.fallback_train_fraction,
                mode=args.train_mode,
                min_length=args.min_train_points,
            )
            train_status = "clean"
            train_error = ""
        except ValueError as exc:
            window = None
            train_status = "invalid"
            train_error = str(exc)

        rows.append(
            {
                "series": name,
                "n_points": len(labels),
                "n_anomaly_points": int(labels.sum()),
                "n_anomaly_events": len(events),
                "first_anomaly_index": first_anomaly,
                "first_anomaly_fraction": first_anomaly_fraction,
                "train_mode": args.train_mode,
                "fallback_train_fraction": args.fallback_train_fraction,
                "train_end": None if window is None else window.end,
                "train_fraction_effective": None if window is None else window.fraction,
                "train_anomaly_points": None if window is None else window.anomaly_points,
                "train_status": train_status,
                "train_error": train_error,
            }
        )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_csv, index=False)
        print(f"\nSaved audit to {args.output_csv}.")


if __name__ == "__main__":
    main()
