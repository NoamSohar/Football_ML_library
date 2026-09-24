import sys
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model.predictor import FEATURE_COLUMNS, Predictor


class FeatureModel:
    feature_names_in_ = FEATURE_COLUMNS


def make_history():
    matches = []
    for day in range(5):
        date = pd.Timestamp("2024-08-01", tz="UTC") + pd.Timedelta(days=day)
        matches.extend([
            {"date": date, "round": "Regular Season - 1", "home_team": "Alpha", "home_team_id": 1,
             "away_team": "Gamma", "away_team_id": 3, "home_goals": 2, "away_goals": 1},
            {"date": date + pd.Timedelta(hours=12), "round": "Regular Season - 1", "home_team": "Gamma", "home_team_id": 3,
             "away_team": "Beta", "away_team_id": 2, "home_goals": 1, "away_goals": 0},
        ])
    return pd.DataFrame(matches)


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.history = make_history()
        self.fixture_date = "2024-08-10T12:00:00Z"

    def test_creates_complete_feature_row_in_training_order(self):
        features = Predictor(FeatureModel(), self.history)._create_features("Alpha", "Beta", self.fixture_date)
        self.assertEqual(list(features.columns), list(FEATURE_COLUMNS))
        self.assertEqual(features.shape, (1, len(FEATURE_COLUMNS)))
        self.assertFalse(features.isna().any().any())

    def test_future_matches_do_not_change_features(self):
        future = self.history.iloc[[-1]].assign(date=pd.Timestamp("2025-01-01", tz="UTC"), home_goals=99)
        expected = Predictor(FeatureModel(), self.history)._create_features("Alpha", "Beta", self.fixture_date)
        actual = Predictor(FeatureModel(), pd.concat([self.history, future]))._create_features("Alpha", "Beta", self.fixture_date)
        assert_frame_equal(actual, expected)

    def test_rejects_invalid_or_insufficient_history(self):
        predictor = Predictor(FeatureModel(), self.history)
        with self.assertRaises(ValueError):
            predictor._create_features("Alpha", "Alpha", self.fixture_date)
        with self.assertRaises(ValueError):
            predictor._create_features("Unknown", "Beta", self.fixture_date)
        with self.assertRaises(ValueError):
            Predictor(FeatureModel(), self.history.iloc[:4])._create_features("Alpha", "Beta", self.fixture_date)
