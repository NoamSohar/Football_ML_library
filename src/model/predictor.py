import pandas as pd

from src.features.Goals import GoalCalculator
from src.features.elo import EloCalculator
from src.features.form import FormCalculator


FEATURE_COLUMNS = [
    "home_team_id",
    "away_team_id",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_5",
    "away_form_5",
    "home_home_form_5",
    "away_away_form_5",
    "home_season_form",
    "away_season_form",
    "home_goals_scored_5",
    "home_goals_against_5",
    "away_goals_scored_5",
    "away_goals_against_5",
    "home_season_goals_per_game",
    "away_season_goals_per_game",
]


class Predictor:
    def __init__(self, model, match_history: pd.DataFrame, draw_threshold=None, draw_weight=None):
        self.model = model
        self.draw_threshold = draw_threshold
        self.draw_weight = draw_weight
        self.match_history = match_history.copy()
        self.match_history["date"] = pd.to_datetime(self.match_history["date"], utc=True)
        self.match_history = self.match_history.sort_values("date").reset_index(drop=True)
        self.team_ids = {}
        duplicate_names = []

        for _, match in self.match_history.iterrows():
            home_team = match["home_team"]
            home_team_id = match["home_team_id"]

            if home_team in self.team_ids:
                if self.team_ids[home_team] != home_team_id:
                    duplicate_names.append(home_team)
            else:
                self.team_ids[home_team] = home_team_id

            away_team = match["away_team"]
            away_team_id = match["away_team_id"]

            if away_team in self.team_ids:
                if self.team_ids[away_team] != away_team_id:
                    duplicate_names.append(away_team)
            else:
                self.team_ids[away_team] = away_team_id

        if duplicate_names:
            duplicate_names = sorted(set(duplicate_names))
            raise ValueError(f"Team names map to multiple IDs: {', '.join(duplicate_names)}")

    @classmethod
    def from_artifact(cls, artifact, match_history: pd.DataFrame):
        return cls(
            artifact["model"],
            match_history,
            artifact.get("draw_threshold"),
            artifact.get("draw_weight"),
        )

    def predict(self, home_team, away_team, fixture_date, draw_threshold=None):
        features = self._create_features(home_team, away_team, fixture_date)
        probabilities = self.model.predict_proba(features)[0]

        threshold = draw_threshold if draw_threshold is not None else self.draw_threshold
        if threshold is None:
            threshold = 0.25

        if probabilities[0] >= threshold:
            return "draw"

        if probabilities[1] >= probabilities[2]:
            return "home_win"

        return "away_win"

    def predict_proba(self, home_team: str, away_team: str, fixture_date) -> dict[str, float]:
        features = self._create_features(home_team, away_team, fixture_date)
        probabilities = self.model.predict_proba(features)[0]

        return {
            "draw": probabilities[0],
            "home_win": probabilities[1],
            "away_win": probabilities[2],
        }

    def _create_features(self, home_team: str, away_team: str, fixture_date) -> pd.DataFrame:
        if home_team == away_team:
            raise ValueError("Home and away teams must differ.")

        if home_team not in self.team_ids:
            raise ValueError(f"Unknown team: {home_team}")

        if away_team not in self.team_ids:
            raise ValueError(f"Unknown team: {away_team}")

        home_team_id = self.team_ids[home_team]
        away_team_id = self.team_ids[away_team]

        fixture_date = pd.Timestamp(fixture_date)

        if fixture_date.tzinfo is None:
            fixture_date = fixture_date.tz_localize("UTC")
        else:
            fixture_date = fixture_date.tz_convert("UTC")

        history = self.match_history[self.match_history["date"] < fixture_date].copy()

        elo_calculator = EloCalculator(history)
        elo_calculator.add_features()

        form_calculator = FormCalculator(history)
        goal_calculator = GoalCalculator(history)

        home_elo = elo_calculator.get_elo(home_team_id)
        away_elo = elo_calculator.get_elo(away_team_id)

        home_form = form_calculator.get_team_form(home_team_id, fixture_date)
        away_form = form_calculator.get_team_form(away_team_id, fixture_date)

        home_home_form = form_calculator.get_location_specific_form(
            home_team_id, fixture_date, "home"
        )
        away_away_form = form_calculator.get_location_specific_form(
            away_team_id, fixture_date, "away"
        )

        home_season_form = form_calculator.get_seasonal_ppg(home_team_id, fixture_date)
        away_season_form = form_calculator.get_seasonal_ppg(away_team_id, fixture_date)

        home_goals = goal_calculator.get_team_goals(home_team_id, fixture_date)
        away_goals = goal_calculator.get_team_goals(away_team_id, fixture_date)

        if home_goals is None or away_goals is None:
            raise ValueError("Insufficient five-match history for one or both teams.")

        home_season_goals = goal_calculator.get_season_average_goals(
            home_team_id, fixture_date
        )
        away_season_goals = goal_calculator.get_season_average_goals(
            away_team_id, fixture_date
        )

        feature_values = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_form_5": home_form,
            "away_form_5": away_form,
            "home_home_form_5": home_home_form,
            "away_away_form_5": away_away_form,
            "home_season_form": home_season_form,
            "away_season_form": away_season_form,
            "home_goals_scored_5": home_goals[0],
            "home_goals_against_5": home_goals[1],
            "away_goals_scored_5": away_goals[0],
            "away_goals_against_5": away_goals[1],
            "home_season_goals_per_game": home_season_goals,
            "away_season_goals_per_game": away_season_goals,
        }
        features = pd.DataFrame([feature_values])

        if features.isna().any().any():
            raise ValueError("Insufficient historical data to create every feature.")

        feature_order = FEATURE_COLUMNS

        if hasattr(self.model, "feature_names_in_"):
            feature_order = self.model.feature_names_in_

        return features.reindex(columns=feature_order)
