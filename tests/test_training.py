import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.data import load_training_data, split_data
from model.evaluation import evaluate_model
from model.storage import load_model, save_model
from model.thresholds import find_draw_threshold, macro_f1_score, predict_results
from model.training import train_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "feature": list(range(30)),
            "result": [index % 3 for index in range(30)],
        })

    def test_chronological_splits_leave_final_test_period_untouched(self):
        x_train, y_train, x_calibration, y_calibration, x_test, y_test = split_data(self.data)

        self.assertEqual((len(x_train), len(x_calibration), len(x_test)), (18, 6, 6))
        self.assertLess(self.data.loc[y_train.index, "date"].max(), self.data.loc[y_calibration.index, "date"].min())
        self.assertLess(self.data.loc[y_calibration.index, "date"].max(), self.data.loc[y_test.index, "date"].min())
        self.assertEqual(set(y_test.index).intersection(y_train.index), set())
        self.assertEqual(set(y_test.index).intersection(y_calibration.index), set())

    def test_evaluation_metrics_and_saved_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            self.data.to_csv(path, index=False)
            data = load_training_data(path)
            x_train, y_train, x_calibration, y_calibration, _, _ = split_data(data)
            model = train_model(
                x_train,
                y_train,
                {"objective": "multi:softprob", "num_class": 3, "n_estimators": 3,
                 "max_depth": 1, "learning_rate": 0.1, "random_state": 1, "n_jobs": 1},
            )
            threshold = find_draw_threshold(model.predict_proba(x_calibration), y_calibration)
            metrics = evaluate_model(model, x_calibration, y_train, y_calibration, threshold)

            self.assertIn("macro_f1", metrics)
            self.assertIn("draw_brier_score", metrics)
            self.assertIn("draw_calibration_curve", metrics)
            self.assertGreaterEqual(metrics["draw_f1"], 0)
            self.assertLessEqual(metrics["draw_f1"], 1)

            model_path = Path(directory) / "model.joblib"
            save_model(
                model, x_train.columns, data, threshold, model_path,
                draw_weight=1.5,
                split_date_ranges={"train": ["2024-01-01", "2024-01-18"]},
                dataset_fingerprint="test-fingerprint",
            )
            artifact = load_model(model_path)
            self.assertEqual(artifact["draw_weight"], 1.5)
            self.assertEqual(artifact["selection_metric"], "macro_f1")
            self.assertEqual(artifact["dataset_fingerprint"], "test-fingerprint")
            assert_series_equal(
                pd.Series(artifact["model"].predict(x_calibration)),
                pd.Series(model.predict(x_calibration)),
            )

    def test_draw_threshold_maximizes_macro_f1(self):
        probabilities = np.array([
            *[[0.2, 0.7, 0.1]] * 6,
            *[[0.1, 0.8, 0.1]] * 4,
        ])
        results = pd.Series([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

        threshold = find_draw_threshold(probabilities, results, minimum=0.1, maximum=0.3, step=0.1)

        self.assertAlmostEqual(threshold, 0.2)
        self.assertEqual((predict_results(probabilities, threshold) == 0).sum(), 6)
        self.assertGreater(
            macro_f1_score(results, predict_results(probabilities, threshold)),
            macro_f1_score(results, predict_results(probabilities, 0.1)),
        )

    def test_rejects_non_positive_draw_weight(self):
        with self.assertRaises(ValueError):
            train_model(
                pd.DataFrame({"feature": [1, 2, 3]}),
                pd.Series([0, 1, 2]),
                {"objective": "multi:softprob", "num_class": 3},
                draw_weight=0,
            )
