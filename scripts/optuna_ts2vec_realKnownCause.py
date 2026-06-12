#!/usr/bin/env python3
"""Optuna hyperparameter search for TS2Vec on NAB realKnownCause.

This script orchestrates the two helper scripts produced for this project:
  - train_ts2vec.py
  - evaluate_ts2vec.py

It stores Optuna trials in a persistent SQLite database by default and optionally
logs configs, metrics and artifacts to Weights & Biases.

Typical usage:
  export WANDB_API_KEY=...
  uv run scripts/optuna_ts2vec_realKnownCause.py --n-trials 30

Resume the same study:
  uv run scripts/optuna_ts2vec_realKnownCause.py --n-trials 30 \
    --storage sqlite:///optuna_ts2vec_realKnownCause.db \
    --study-name ts2vec-realKnownCause
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

try:
    import wandb
except ImportError:  # pragma: no cover - handled at runtime
    wandb = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "training" / "ts2vec_optuna" / "realKnownCause"
DEFAULT_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_ts2vec.py"
DEFAULT_EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_ts2vec.py"


MetricDict = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna HPO for TS2Vec on NAB realKnownCause, store trials in SQLite, "
            "and report metrics to Weights & Biases."
        )
    )
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=None, help="Optuna timeout in seconds.")
    parser.add_argument("--study-name", default="ts2vec-realKnownCause")
    parser.add_argument(
        "--storage",
        default="sqlite:///optuna_ts2vec_realKnownCause.db",
        help="Optuna storage URL. SQLite example: sqlite:///optuna_ts2vec_realKnownCause.db",
    )
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument(
        "--pruner",
        choices=("none", "median"),
        default="none",
        help=(
            "Pruning is limited because train/eval are subprocesses. Median pruning can "
            "only prune after a trial finishes and reports its objective."
        ),
    )
    parser.add_argument(
        "--objective-key",
        default="aggregate.eventwise.f1",
        help=(
            "Dot path inside metrics.json to maximize, e.g. aggregate.eventwise.f1, "
            "aggregate.pointwise.pr_auc, threshold_diagnostics.pointwise_best_f1.f1, "
            "threshold_diagnostics.eventwise_best_f1.f1."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["realKnownCause"],
        help="Defaults to realKnownCause. Keep as one category for this search.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--ts2vec-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument(
        "--include-mask-diff",
        action="store_true",
        help="Also allow the older mask-diff anomaly score in the Optuna search space.",
    )
    parser.add_argument(
        "--wandb-project",
        default="nab-ts2vec-optuna",
        help="Set to empty string together with --wandb-mode disabled to disable W&B.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use disabled for no W&B calls; offline keeps local W&B logs.",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        action="store_true",
        help="Log model, metadata and metrics files as a W&B artifact for each successful trial.",
    )
    parser.add_argument(
        "--keep-failed-dirs",
        action="store_true",
        help="Keep trial directories even when a subprocess fails. Logs are always kept.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one sampled command pair without running Optuna optimization.",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} not found: {resolved}")
    return resolved


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            output.update(flatten_dict(value, full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[full_key] = value
    return output


def get_by_dot_path(data: dict[str, Any], path: str) -> float:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric path '{path}' not found at '{part}'.")
        current = current[part]
    if current is None:
        raise ValueError(f"Metric path '{path}' is None.")
    value = float(current)
    if not (value == value):  # NaN check
        raise ValueError(f"Metric path '{path}' is NaN.")
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        out.write("$ " + " ".join(shlex.quote(part) for part in cmd) + "\n\n")
        out.flush()
        process = subprocess.run(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
    if process.returncode != 0:
        tail = ""
        try:
            tail = stderr_path.read_text(encoding="utf-8")[-4000:]
        except OSError:
            pass
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: "
            f"{' '.join(shlex.quote(part) for part in cmd)}\n"
            f"stderr tail:\n{tail}"
        )


def suggest_params(trial: optuna.Trial, include_mask_diff: bool) -> dict[str, Any]:
    score_methods = ["knn", "centroid"]
    if include_mask_diff:
        score_methods.append("mask-diff")

    return {
        # Training / representation learning.
        "train_fraction": trial.suggest_float("train_fraction", 0.10, 0.30, step=0.05),
        "normalization": trial.suggest_categorical("normalization", ["per-series", "global"]),
        "output_dims": trial.suggest_categorical("output_dims", [64, 160, 320]),
        "hidden_dims": trial.suggest_categorical("hidden_dims", [32, 64, 128]),
        "depth": trial.suggest_int("depth", 4, 10),
        "batch_size_train": trial.suggest_categorical("batch_size_train", [4, 8, 16]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "iters": trial.suggest_categorical("iters", [1000, 2000, 3000, 5000]),
        "max_train_length": trial.suggest_categorical("max_train_length", [256, 512, 1024, 2048]),
        "temporal_unit": trial.suggest_int("temporal_unit", 0, 2),
        # Evaluation / downstream anomaly scoring.
        "score_method": trial.suggest_categorical("score_method", score_methods),
        "knn_k": trial.suggest_categorical("knn_k", [1, 3, 5, 10, 20]),
        "batch_size_eval": trial.suggest_categorical("batch_size_eval", [128, 256, 512]),
        "sliding_length": trial.suggest_categorical("sliding_length", [16, 32, 64, 128]),
        "sliding_padding": trial.suggest_categorical("sliding_padding", [100, 200, 400]),
        "score_adjust_window": trial.suggest_categorical("score_adjust_window", [0, 21, 101, 501]),
        "threshold_quantile": trial.suggest_float("threshold_quantile", 0.90, 0.995),
        "event_min_true_overlap_fraction": trial.suggest_categorical(
            "event_min_true_overlap_fraction", [0.0, 0.01, 0.05, 0.10]
        ),
        "event_min_pred_overlap_fraction": trial.suggest_categorical(
            "event_min_pred_overlap_fraction", [0.0, 0.01, 0.05, 0.10]
        ),
    }


def build_train_command(args: argparse.Namespace, params: dict[str, Any], output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.train_script),
        "--categories",
        *args.categories,
        "--output-dir",
        str(output_dir),
        "--train-fraction",
        str(params["train_fraction"]),
        "--normalization",
        str(params["normalization"]),
        "--output-dims",
        str(params["output_dims"]),
        "--hidden-dims",
        str(params["hidden_dims"]),
        "--depth",
        str(params["depth"]),
        "--batch-size",
        str(params["batch_size_train"]),
        "--lr",
        str(params["lr"]),
        "--iters",
        str(params["iters"]),
        "--max-train-length",
        str(params["max_train_length"]),
        "--temporal-unit",
        str(params["temporal_unit"]),
        "--seed",
        str(10_000 + int(params.get("trial_number", 0))),
    ]
    if args.dataset_dir is not None:
        cmd += ["--dataset-dir", str(args.dataset_dir)]
    if args.ts2vec_dir is not None:
        cmd += ["--ts2vec-dir", str(args.ts2vec_dir)]
    if args.device is not None:
        cmd += ["--device", args.device]
    return cmd


def build_eval_command(args: argparse.Namespace, params: dict[str, Any], train_dir: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.eval_script),
        "--run-dir",
        str(train_dir),
        "--categories",
        *args.categories,
        "--output-dir",
        str(output_dir),
        "--score-method",
        str(params["score_method"]),
        "--knn-k",
        str(params["knn_k"]),
        "--batch-size",
        str(params["batch_size_eval"]),
        "--sliding-length",
        str(params["sliding_length"]),
        "--sliding-padding",
        str(params["sliding_padding"]),
        "--score-adjust-window",
        str(params["score_adjust_window"]),
        "--threshold-quantile",
        str(params["threshold_quantile"]),
        "--event-overlap-policy",
        "one-to-one",
        "--event-min-overlap-points",
        "1",
        "--event-min-true-overlap-fraction",
        str(params["event_min_true_overlap_fraction"]),
        "--event-min-pred-overlap-fraction",
        str(params["event_min_pred_overlap_fraction"]),
    ]
    if args.dataset_dir is not None:
        cmd += ["--dataset-dir", str(args.dataset_dir)]
    if args.labels_file is not None:
        cmd += ["--labels-file", str(args.labels_file)]
    if args.ts2vec_dir is not None:
        cmd += ["--ts2vec-dir", str(args.ts2vec_dir)]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.save_scores:
        cmd.append("--save-scores")
    return cmd


def maybe_start_wandb(args: argparse.Namespace, trial: optuna.Trial, params: dict[str, Any], trial_dir: Path):
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    if wandb is None:
        raise SystemExit("wandb is not installed. Install it or pass --wandb-mode disabled.")

    config = {
        **params,
        "study_name": args.study_name,
        "trial_number": trial.number,
        "objective_key": args.objective_key,
        "categories": args.categories,
        "optuna_storage": args.storage,
        "trial_dir": str(trial_dir),
        "train_script": str(args.train_script),
        "eval_script": str(args.eval_script),
    }
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=f"trial-{trial.number:04d}",
        group=args.study_name,
        job_type="optuna-trial",
        config=config,
        reinit=True,
    )


def log_wandb_artifact(run: Any, trial_dir: Path, trial_number: int, objective_value: float) -> None:
    if run is None or wandb is None:
        return
    artifact = wandb.Artifact(
        name=f"ts2vec-realKnownCause-trial-{trial_number:04d}",
        type="model-eval",
        metadata={"trial_number": trial_number, "objective_value": objective_value},
    )
    for rel in [
        "params.json",
        "commands.json",
        "train/metadata.json",
        "train/series.json",
        "train/loss_log.npy",
        "train/ts2vec_model.pt",
        "eval/metrics.json",
        "eval/per_series_metrics.jsonl",
        "logs/train_stdout.log",
        "logs/train_stderr.log",
        "logs/eval_stdout.log",
        "logs/eval_stderr.log",
    ]:
        path = trial_dir / rel
        if path.is_file():
            artifact.add_file(str(path), name=rel)
    run.log_artifact(artifact)


def make_objective(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.include_mask_diff)
        params["trial_number"] = trial.number

        trial_dir = args.output_root.expanduser().resolve() / f"trial_{trial.number:04d}"
        train_dir = trial_dir / "train"
        eval_dir = trial_dir / "eval"
        logs_dir = trial_dir / "logs"
        trial_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = build_train_command(args, params, train_dir)
        eval_cmd = build_eval_command(args, params, train_dir, eval_dir)
        save_json(trial_dir / "params.json", params)
        save_json(trial_dir / "commands.json", {"train": train_cmd, "eval": eval_cmd})

        run = None
        try:
            run = maybe_start_wandb(args, trial, params, trial_dir)
            if run is not None:
                run.log({"trial/status": "started"})

            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("train_dir", str(train_dir))
            trial.set_user_attr("eval_dir", str(eval_dir))
            trial.set_user_attr("params", params)

            run_command(
                train_cmd,
                cwd=PROJECT_ROOT,
                stdout_path=logs_dir / "train_stdout.log",
                stderr_path=logs_dir / "train_stderr.log",
            )
            run_command(
                eval_cmd,
                cwd=PROJECT_ROOT,
                stdout_path=logs_dir / "eval_stdout.log",
                stderr_path=logs_dir / "eval_stderr.log",
            )

            metrics_path = eval_dir / "metrics.json"
            metrics: MetricDict = json.loads(metrics_path.read_text(encoding="utf-8"))
            objective_value = get_by_dot_path(metrics, args.objective_key)

            flat_metrics = flatten_dict(metrics)
            trial.set_user_attr("objective_key", args.objective_key)
            trial.set_user_attr("objective_value", objective_value)
            trial.set_user_attr("metrics_path", str(metrics_path))
            trial.set_user_attr("model_path", str(train_dir / "ts2vec_model.pt"))

            # Store a compact but useful subset in Optuna attrs.
            for key, value in flat_metrics.items():
                if key.startswith("aggregate.") or key.startswith("threshold_diagnostics."):
                    trial.set_user_attr(key, value)

            trial.report(objective_value, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned after objective={objective_value:.6g}")

            if run is not None:
                run.log({
                    "objective": objective_value,
                    **{f"metrics/{k}": v for k, v in flat_metrics.items() if isinstance(v, (int, float))},
                    "trial/status": "completed",
                })
                if args.wandb_log_artifacts:
                    try:
                        log_wandb_artifact(run, trial_dir, trial.number, objective_value)
                    except Exception as exc:
                        print(f"[WARN] Failed to log W&B artifact for trial {trial.number}: {exc}", file=sys.stderr)

            return objective_value

        except optuna.TrialPruned:
            if run is not None:
                run.log({"trial/status": "pruned"})
            raise
        except Exception as exc:
            error_payload = {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "trial_number": trial.number,
                "trial_dir": str(trial_dir),
            }
            save_json(trial_dir / "error.json", error_payload)
            trial.set_user_attr("error", repr(exc))
            if run is not None:
                run.log({"trial/status": "failed"})
                try:
                    run.summary["error"] = repr(exc)
                except Exception:
                    pass
            raise
        finally:
            if run is not None:
                run.finish()

    return objective


def print_dry_run(args: argparse.Namespace) -> None:
    sampler = optuna.samplers.RandomSampler(seed=args.sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial = study.ask()
    params = suggest_params(trial, args.include_mask_diff)
    params["trial_number"] = trial.number
    trial_dir = args.output_root.expanduser().resolve() / "trial_0000"
    train_cmd = build_train_command(args, params, trial_dir / "train")
    eval_cmd = build_eval_command(args, params, trial_dir / "train", trial_dir / "eval")
    print("Sampled params:")
    print(json.dumps(params, indent=2, sort_keys=True))
    print("\nTrain command:")
    print(" ".join(shlex.quote(part) for part in train_cmd))
    print("\nEval command:")
    print(" ".join(shlex.quote(part) for part in eval_cmd))


def main() -> None:
    args = parse_args()
    args.train_script = ensure_file(args.train_script, "train script")
    args.eval_script = ensure_file(args.eval_script, "eval script")
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print_dry_run(args)
        return

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed, multivariate=True)
    if args.pruner == "median":
        pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    study.set_user_attr("created_or_resumed_at", datetime.now().isoformat(timespec="seconds"))
    study.set_user_attr("objective_key", args.objective_key)
    study.set_user_attr("categories", args.categories)
    study.set_user_attr("output_root", str(args.output_root))

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Objective: maximize {args.objective_key}")
    print(f"Output root: {args.output_root}")

    study.optimize(make_objective(args), n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    print("\nBest trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value:  {study.best_value:.6f}")
    print("  params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("  attrs:")
    for key in ["trial_dir", "metrics_path", "model_path"]:
        if key in study.best_trial.user_attrs:
            print(f"    {key}: {study.best_trial.user_attrs[key]}")


if __name__ == "__main__":
    # Avoid accidental nested tokenizers / BLAS oversubscription in subprocess-heavy HPO.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
