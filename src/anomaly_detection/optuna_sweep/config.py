from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

from anomaly_detection.data import DATASET_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optuna"
DEFAULT_FINAL_OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_from_optuna"
MODEL_CHOICES = (
    "robust_zscore",
    "rolling_residual",
    "isolation_forest",
    "mlp_autoencoder",
    "lstm",
    "gru",
    "ts2vec",
)
DEFAULT_OBJECTIVE = "macro_average.eventwise.f1"
MODEL_PARAM_KEYS: dict[str, set[str]] = {
    "robust_zscore": {"scale_estimator"},
    "rolling_residual": {"rolling_window", "baseline_statistic", "residual_transform"},
    "isolation_forest": {"window_size", "n_estimators", "contamination"},
    "mlp_autoencoder": {
        "window_size",
        "latent_dim",
        "hidden_dim",
        "learning_rate",
        "batch_size",
        "epochs",
    },
    "lstm": {"window_size", "hidden_dim", "learning_rate", "batch_size", "epochs"},
    "gru": {"window_size", "hidden_dim", "learning_rate", "batch_size", "epochs"},
    "ts2vec": {
        "output_dims",
        "hidden_dims",
        "depth",
        "batch_size_train",
        "learning_rate",
        "iters",
        "max_train_length",
        "temporal_unit",
        "score_method",
        "knn_k",
        "batch_size_eval",
        "sliding_length",
        "sliding_padding",
    },
}

LOCAL_BASELINE_MODELS = {"robust_zscore", "rolling_residual"}
GLOBAL_TRAINABLE_MODELS = {"isolation_forest", "mlp_autoencoder", "lstm", "gru", "ts2vec"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna sweeps and fixed-protocol final evaluation on NAB."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--mode",
        choices=("sweep", "final"),
        default="sweep",
        help="sweep runs Optuna; final evaluates exact --params-file with fixed protocol.",
    )
    parser.add_argument("--params-file", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default="sqlite:///optuna_nab_sweeps.db")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--wandb-project", default="anomaly-detection-nab-sweeps")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    parser.add_argument("--validation-subset-file", type=Path, default=None)
    parser.add_argument("--validation-size", type=int, default=12)
    parser.add_argument("--objective-key", default=DEFAULT_OBJECTIVE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--limit-series", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("all", "validation", "test", "validation_and_test"),
        default="all",
        help=(
            "Final-mode reporting split. 'all' evaluates all loaded series. "
            "'validation' uses the frozen validation subset. 'test' uses all loaded "
            "series except the frozen validation subset. 'validation_and_test' reports both."
        ),
    )
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument(
        "--load-models-dir",
        type=Path,
        default=None,
        help="Load fitted per-series models/checkpoints from this directory and skip fitting.",
    )
    parser.add_argument("--ts2vec-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--training-scope",
        choices=("auto", "per_series", "global"),
        default="auto",
        help=(
            "Training scope for model fitting. Default 'auto' uses per-series calibration "
            "for robust_zscore/rolling_residual and global pooled-prefix training for "
            "isolation_forest, mlp_autoencoder, lstm, gru and ts2vec. 'per_series' is "
            "available for ablations; 'global' is ignored for local statistical baselines."
        ),
    )
    parser.add_argument(
        "--ts2vec-training-scope",
        choices=("per_series", "global"),
        default=None,
        help=(
            "Deprecated compatibility flag. Prefer --training-scope. When provided for "
            "--model ts2vec, it overrides --training-scope."
        ),
    )

    parser.add_argument("--threshold-source", choices=("train", "all"), default="train")
    parser.add_argument("--threshold-scope", choices=("per_series", "global"), default="per_series")
    parser.add_argument("--threshold-sweep-steps", type=int, default=100)
    parser.add_argument("--protocol-threshold-quantile", type=float, default=0.99)
    parser.add_argument("--protocol-score-smoothing-window", type=int, default=1)
    parser.add_argument(
        "--protocol-normalization",
        choices=("none", "per-series", "robust"),
        default="robust",
        help="Common preprocessing fitted only on the training/calibration prefix.",
    )
    parser.add_argument("--protocol-min-overlap-points", type=int, default=1)
    parser.add_argument("--protocol-min-true-overlap-fraction", type=float, default=0.0)
    parser.add_argument("--protocol-min-pred-overlap-fraction", type=float, default=0.0)
    parser.add_argument(
        "--protocol-min-event-length",
        type=int,
        default=1,
        help="Drop predicted anomaly events shorter than this many points after thresholding.",
    )
    parser.add_argument(
        "--protocol-merge-gap",
        type=int,
        default=0,
        help="Merge predicted anomaly events separated by at most this many normal points.",
    )
    parser.add_argument(
        "--protocol-cooldown",
        type=int,
        default=0,
        help="Suppress new predicted events that start within this many points after a kept event.",
    )
    parser.add_argument(
        "--protocol-alert-mode",
        choices=("full_event", "peak_window"),
        default="full_event",
        help=(
            "How cleaned-up predicted events become point-wise alerts. "
            "'full_event' marks the whole event span; 'peak_window' marks only "
            "a fixed window around the highest score inside each event."
        ),
    )
    parser.add_argument(
        "--protocol-alert-window",
        type=int,
        default=1,
        help="Number of points to mark around each event peak when using peak-window alerts.",
    )
    parser.add_argument(
        "--protocol-max-alerts-per-event",
        type=int,
        default=1,
        help="Maximum number of separated peak windows to emit inside each cleaned event.",
    )
    parser.add_argument(
        "--protocol-peak-min-distance",
        type=int,
        default=None,
        help=(
            "Minimum distance between selected peaks inside one cleaned event. "
            "Defaults to --protocol-alert-window."
        ),
    )
    parser.add_argument("--min-train-points", type=int, default=32)
    parser.add_argument("--fallback-train-fraction", type=float, default=0.05)
    parser.add_argument(
        "--allow-contaminated-train",
        action="store_true",
        help=(
            "Allow fallback training prefixes containing anomaly labels. When this is not set, "
            "contamination is still recorded and warned about if no clean prefix can be formed."
        ),
    )
    parser.add_argument(
        "--strict-clean-train",
        action="store_true",
        help="Fail a series instead of using a contaminated fallback training prefix.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "sweep" and args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive.")
    if args.mode == "final" and args.params_file is None:
        raise SystemExit("--params-file is required when --mode final.")
    if args.params_file is not None and not args.params_file.expanduser().is_file():
        raise SystemExit(f"Params file does not exist: {args.params_file}")
    if args.limit_series is not None and args.limit_series <= 0:
        raise SystemExit("--limit-series must be positive.")
    if args.validation_size <= 0:
        raise SystemExit("--validation-size must be positive.")
    if args.threshold_sweep_steps <= 1:
        raise SystemExit("--threshold-sweep-steps must be greater than 1.")
    if not 0 <= args.protocol_threshold_quantile <= 1:
        raise SystemExit("--protocol-threshold-quantile must be in [0, 1].")
    if args.protocol_score_smoothing_window < 1:
        raise SystemExit("--protocol-score-smoothing-window must be at least 1.")
    if args.protocol_min_overlap_points < 1:
        raise SystemExit("--protocol-min-overlap-points must be at least 1.")
    if not 0 <= args.protocol_min_true_overlap_fraction <= 1:
        raise SystemExit("--protocol-min-true-overlap-fraction must be in [0, 1].")
    if not 0 <= args.protocol_min_pred_overlap_fraction <= 1:
        raise SystemExit("--protocol-min-pred-overlap-fraction must be in [0, 1].")
    if args.protocol_min_event_length < 1:
        raise SystemExit("--protocol-min-event-length must be at least 1.")
    if args.protocol_merge_gap < 0:
        raise SystemExit("--protocol-merge-gap must be non-negative.")
    if args.protocol_cooldown < 0:
        raise SystemExit("--protocol-cooldown must be non-negative.")
    if args.protocol_alert_window < 1:
        raise SystemExit("--protocol-alert-window must be at least 1.")
    if args.protocol_max_alerts_per_event < 1:
        raise SystemExit("--protocol-max-alerts-per-event must be at least 1.")
    if args.protocol_peak_min_distance is not None and args.protocol_peak_min_distance < 1:
        raise SystemExit("--protocol-peak-min-distance must be at least 1 when provided.")
    if not 0 < args.fallback_train_fraction <= 1:
        raise SystemExit("--fallback-train-fraction must be in the interval (0, 1].")
    if args.min_train_points <= 1:
        raise SystemExit("--min-train-points must be greater than 1.")


def effective_training_scope(args: argparse.Namespace, model: str) -> str:
    if model in LOCAL_BASELINE_MODELS:
        return "per_series"
    if model == "ts2vec" and getattr(args, "ts2vec_training_scope", None) is not None:
        return args.ts2vec_training_scope
    requested = getattr(args, "training_scope", "auto")
    if requested == "auto":
        return "global" if model in GLOBAL_TRAINABLE_MODELS else "per_series"
    if requested == "global" and model not in GLOBAL_TRAINABLE_MODELS:
        warnings.warn(
            f"Global training requested for local baseline {model}; using per_series instead.",
            stacklevel=2,
        )
        return "per_series"
    return requested


def protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training_window_mode": "before_first_anomaly_with_fallback",
        "fallback_train_fraction": args.fallback_train_fraction,
        "min_train_points": args.min_train_points,
        "allow_contaminated_train": args.allow_contaminated_train,
        "strict_clean_train": args.strict_clean_train,
        "normalization": args.protocol_normalization,
        "score_smoothing_window": args.protocol_score_smoothing_window,
        "threshold_source": args.threshold_source,
        "threshold_scope": args.threshold_scope,
        "threshold_quantile": args.protocol_threshold_quantile,
        "min_overlap_points": args.protocol_min_overlap_points,
        "min_true_overlap_fraction": args.protocol_min_true_overlap_fraction,
        "min_pred_overlap_fraction": args.protocol_min_pred_overlap_fraction,
        "min_event_length": args.protocol_min_event_length,
        "merge_gap": args.protocol_merge_gap,
        "cooldown": args.protocol_cooldown,
        "alert_mode": args.protocol_alert_mode,
        "alert_window": args.protocol_alert_window,
        "max_alerts_per_event": args.protocol_max_alerts_per_event,
        "peak_min_distance": args.protocol_peak_min_distance,
        "training_scope_cli": args.training_scope,
        "ts2vec_training_scope_override": args.ts2vec_training_scope,
        "evaluation_region": "after_train_end",
        "benchmark_training_protocol": (
            "hybrid: local statistical baselines are per-series; trainable models are global "
            "on pooled normalized normal-prefix data"
        ),
    }


def prediction_postprocessing_kwargs(args: argparse.Namespace) -> dict[str, int | str | None]:
    return {
        "min_event_length": args.protocol_min_event_length,
        "merge_gap": args.protocol_merge_gap,
        "cooldown": args.protocol_cooldown,
        "alert_mode": args.protocol_alert_mode,
        "alert_window": args.protocol_alert_window,
        "max_alerts_per_event": args.protocol_max_alerts_per_event,
        "peak_min_distance": args.protocol_peak_min_distance,
    }
