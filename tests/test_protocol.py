from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomaly_detection.metrics import SeriesScores, evaluate_series_scores  # noqa: E402
from anomaly_detection.protocol import select_training_window  # noqa: E402


class TrainingProtocolTest(unittest.TestCase):
    def test_fixed_prefix_records_clean_window(self) -> None:
        labels = np.zeros(20, dtype=np.int8)

        window = select_training_window(
            n_points=20,
            labels=labels,
            fallback_train_fraction=0.15,
            mode="fixed_prefix",
            min_length=2,
        )

        self.assertEqual(window.start, 0)
        self.assertEqual(window.end, 3)
        self.assertTrue(window.is_clean)

    def test_fixed_prefix_rejects_labelled_anomaly(self) -> None:
        labels = np.zeros(20, dtype=np.int8)
        labels[2] = 1

        with self.assertRaises(ValueError):
            select_training_window(
                n_points=20,
                labels=labels,
                fallback_train_fraction=0.15,
                mode="fixed_prefix",
                min_length=2,
            )

    def test_before_first_anomaly_is_explicitly_label_assisted(self) -> None:
        labels = np.zeros(20, dtype=np.int8)
        labels[5:8] = 1

        window = select_training_window(
            n_points=20,
            labels=labels,
            fallback_train_fraction=0.5,
            mode="before_first_anomaly",
            min_length=2,
        )

        self.assertEqual(window.end, 5)
        self.assertEqual(window.mode, "before_first_anomaly")
        self.assertTrue(window.is_clean)


class EvaluationProtocolTest(unittest.TestCase):
    def test_default_threshold_comes_from_training_scores(self) -> None:
        series = [
            SeriesScores(
                name="toy.csv",
                scores=np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float32),
                labels=np.array([0, 0, 0, 1], dtype=np.int8),
                train_end=3,
            )
        ]

        metrics = evaluate_series_scores(series, threshold_quantile=1.0)

        self.assertEqual(metrics["threshold_source"], "train")
        self.assertEqual(metrics["threshold_scope"], "per_series")
        self.assertIsNone(metrics["threshold"])
        self.assertEqual(metrics["thresholds"]["toy.csv"], 0.0)
        self.assertEqual(metrics["aggregate"]["pointwise"]["tp"], 1)
        self.assertEqual(metrics["aggregate"]["pointwise"]["fp"], 0)
        self.assertEqual(metrics["aggregate"]["pointwise"]["support"], 1)
        self.assertEqual(metrics["per_series"][0]["n_evaluation_points"], 1)


if __name__ == "__main__":
    unittest.main()
