import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.validation import validate_scores
from src.features.elo import EloCalculator


class PipelineTests(unittest.TestCase):
    def test_rejects_negative_finished_scores(self):
        matches = pd.DataFrame({
            "status": ["FT"], "home_goals": [-1], "away_goals": [0]
        })
        with self.assertRaisesRegex(ValueError, "Home goals cannot be negative"):
            validate_scores(matches)

    def test_simultaneous_fixtures_receive_pre_kickoff_elo_ratings(self):
        kickoff = pd.Timestamp("2024-08-01T18:00:00Z")
        matches = pd.DataFrame([
            {"date": kickoff, "home_team_id": 1, "away_team_id": 2, "home_goals": 3, "away_goals": 0},
            {"date": kickoff, "home_team_id": 3, "away_team_id": 4, "home_goals": 0, "away_goals": 1},
            {"date": kickoff + pd.Timedelta(days=1), "home_team_id": 1, "away_team_id": 3, "home_goals": 1, "away_goals": 1},
        ])

        featured = EloCalculator(matches).add_features()

        self.assertEqual(featured.loc[0, "home_elo"], 1500)
        self.assertEqual(featured.loc[0, "away_elo"], 1500)
        self.assertEqual(featured.loc[1, "home_elo"], 1500)
        self.assertEqual(featured.loc[1, "away_elo"], 1500)
