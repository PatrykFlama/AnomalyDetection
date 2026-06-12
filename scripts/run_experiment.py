#!/usr/bin/env python3
"""Run the full anomaly-detection experiment and write one final summary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "nab"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baselines, TS2Vec and combined summary.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--categories", nargs="+", default=["realKnownCause"])
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ts2vec", action="store_true")
    parser.add_argument("--ts2vec-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.3)
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=[0.99, 0.995])
    parser.add_argument("--threshold-sweep-steps", type=int, default=200)
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--baseline-window", type=int, default=64)
    parser.add_argument("--baseline-batch-size", type=int, default=128)
    parser.add_argument("--ts2vec-iters", type=int, default=2000)
    parser.add_argument("--ts2vec-batch-size", type=int, default=8)
    parser.add_argument("--ts2vec-max-train-length", type=int, default=512)
    parser.add_argument("--save-scores", action="store_true")
    return parser.parse_args()


def make_output_dir(path: Path | None) -> Path:
    if path is None:
        path = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def add_common_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(["--dataset-dir", str(args.dataset_dir)])
    command.extend(["--categories", *args.categories])
    command.extend(["--seed", str(args.seed)])
    command.extend(["--split-seed", str(args.split_seed)])
    command.extend(["--validation-fraction", str(args.validation_fraction)])
    command.extend(["--threshold-sweep-steps", str(args.threshold_sweep_steps)])
    command.extend(["--threshold-quantiles", *[str(value) for value in args.threshold_quantiles]])
    if args.limit_series is not None:
        command.extend(["--limit-series", str(args.limit_series)])
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.save_scores:
        command.append("--save-scores")


def run_command(command: list[str]) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_baselines(args: argparse.Namespace, output_dir: Path) -> Path:
    baselines_dir = output_dir / "baselines"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_baselines.py"),
        "--output-dir",
        str(baselines_dir),
        "--methods",
        *args.methods,
        "--epochs",
        str(args.baseline_epochs),
        "--window",
        str(args.baseline_window),
        "--batch-size",
        str(args.baseline_batch_size),
    ]
    add_common_args(command, args)
    run_command(command)
    return baselines_dir / "summary.csv"


def run_ts2vec(args: argparse.Namespace, output_dir: Path) -> Path:
    train_dir = output_dir / "ts2vec" / "training"
    eval_dir = output_dir / "ts2vec" / "evaluation"

    train_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_ts2vec.py"),
        "--dataset-dir",
        str(args.dataset_dir),
        "--categories",
        *args.categories,
        "--output-dir",
        str(train_dir),
        "--iters",
        str(args.ts2vec_iters),
        "--batch-size",
        str(args.ts2vec_batch_size),
        "--max-train-length",
        str(args.ts2vec_max_train_length),
        "--seed",
        str(args.seed),
    ]
    if args.limit_series is not None:
        train_command.extend(["--limit-series", str(args.limit_series)])
    if args.device is not None:
        train_command.extend(["--device", args.device])
    if args.ts2vec_dir is not None:
        train_command.extend(["--ts2vec-dir", str(args.ts2vec_dir)])
    run_command(train_command)

    eval_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_ts2vec.py"),
        "--run-dir",
        str(train_dir),
        "--dataset-dir",
        str(args.dataset_dir),
        "--categories",
        *args.categories,
        "--output-dir",
        str(eval_dir),
        "--split-seed",
        str(args.split_seed),
        "--validation-fraction",
        str(args.validation_fraction),
        "--threshold-sweep-steps",
        str(args.threshold_sweep_steps),
        "--threshold-quantiles",
        *[str(value) for value in args.threshold_quantiles],
    ]
    if args.limit_series is not None:
        eval_command.extend(["--limit-series", str(args.limit_series)])
    if args.device is not None:
        eval_command.extend(["--device", args.device])
    if args.ts2vec_dir is not None:
        eval_command.extend(["--ts2vec-dir", str(args.ts2vec_dir)])
    if args.save_scores:
        eval_command.append("--save-scores")
    run_command(eval_command)
    return eval_dir / "summary_ts2vec.csv"


def write_combined_summary(summary_paths: list[Path], output_dir: Path) -> None:
    frames = []
    for path in summary_paths:
        summary = pd.read_csv(path)
        frames.append(summary[summary["split"] == "test"].copy())
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("event_f1", ascending=False)
    combined.to_csv(output_dir / "summary.csv", index=False)

    columns = [
        "method",
        "event_f1",
        "event_precision",
        "event_recall",
        "point_f1",
        "point_pr_auc",
        "threshold_quantile",
    ]
    print("\nFinal test summary:")
    print(combined[[column for column in columns if column in combined]].to_string(index=False))
    print(f"\nSaved combined summary to {output_dir / 'summary.csv'}.")


def main() -> None:
    args = parse_args()
    if args.skip_baselines and args.skip_ts2vec:
        raise SystemExit("Nothing to run: both --skip-baselines and --skip-ts2vec were passed.")

    output_dir = make_output_dir(args.output_dir)
    summary_paths = []
    if not args.skip_baselines:
        summary_paths.append(run_baselines(args, output_dir))
    if not args.skip_ts2vec:
        summary_paths.append(run_ts2vec(args, output_dir))
    write_combined_summary(summary_paths, output_dir)


if __name__ == "__main__":
    main()
