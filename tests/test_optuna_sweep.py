from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.metrics import SeriesScores  # noqa: E402
from anomaly_detection.optuna_sweep.config import (  # noqa: E402
    effective_training_scope,
)
from anomaly_detection.optuna_sweep.evaluation import evaluate_scores  # noqa: E402
from anomaly_detection.optuna_sweep.protocol import fit_normalization  # noqa: E402
from anomaly_detection.optuna_sweep.types import SweepContext  # noqa: E402


def protocol_args(**overrides: object) -> argparse.Namespace:
    values = {
        "allow_contaminated_train": False,
        "fallback_train_fraction": 0.05,
        "min_train_points": 2,
        "protocol_alert_mode": "full_event",
        "protocol_alert_window": 1,
        "protocol_cooldown": 0,
        "protocol_max_alerts_per_event": 1,
        "protocol_merge_gap": 0,
        "protocol_min_event_length": 1,
        "protocol_min_overlap_points": 1,
        "protocol_min_pred_overlap_fraction": 0.0,
        "protocol_min_true_overlap_fraction": 0.0,
        "protocol_normalization": "robust",
        "protocol_peak_min_distance": None,
        "protocol_score_smoothing_window": 1,
        "protocol_threshold_quantile": 1.0,
        "strict_clean_train": False,
        "threshold_scope": "per_series",
        "threshold_source": "train",
        "threshold_sweep_steps": 10,
        "training_scope": "auto",
        "ts2vec_training_scope": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SweepConfigTest(unittest.TestCase):
    def test_auto_scope_keeps_baselines_local_and_trainable_models_global(self) -> None:
        args = protocol_args()

        self.assertEqual(effective_training_scope(args, "robust_zscore"), "per_series")
        self.assertEqual(effective_training_scope(args, "isolation_forest"), "global")
        self.assertEqual(effective_training_scope(args, "ts2vec"), "global")


class SweepProtocolTest(unittest.TestCase):
    def test_normalization_is_fitted_only_on_training_prefix(self) -> None:
        values = np.array([0.0, 2.0, 1000.0], dtype=np.float32)

        normalized, details = fit_normalization(values, train_end=2, method="per-series")

        self.assertEqual(details["mean"], 1.0)
        self.assertEqual(details["std"], 1.0)
        np.testing.assert_allclose(normalized, [-1.0, 1.0, 999.0])

    def test_evaluation_uses_training_scores_for_threshold(self) -> None:
        args = protocol_args()
        context = SweepContext(
            args=args,
            dataset={},
            labels={},
            subset=["category/toy.csv"],
            subset_metadata=[],
            labels_file=None,
            output_root=PROJECT_ROOT,
            git_commit=None,
        )
        scores = SeriesScores(
            name="category/toy.csv",
            scores=np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32),
            labels=np.array([0, 0, 0, 1], dtype=np.int8),
            train_end=3,
        )

        metrics = evaluate_scores([scores], context)

        self.assertEqual(metrics["thresholds"]["category/toy.csv"], 0.0)
        self.assertEqual(metrics["aggregate"]["pointwise"]["tp"], 1)
        self.assertEqual(metrics["aggregate"]["pointwise"]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
