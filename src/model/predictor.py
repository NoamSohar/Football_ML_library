import pandas as pd

from features.Goals import GoalCalculator
from features.elo import EloCalculator
from features.form import FormCalculator


FEATURE_COLUMNS = (
    "home_team_id", "away_team_id", "home_elo", "away_elo", "elo_diff",
    "home_form_5", "away_form_5", "home_home_form_5", "away_away_form_5",
    "home_season_form", "away_season_form", "home_goals_scored_5",
    "home_goals_against_5", "away_goals_scored_5", "away_goals_against_5",
    "home_season_goals_per_game", "away_season_goals_per_game",
)


class Predictor:
    def __init__(self, model, match_history: pd.DataFrame):
        self.model = model
        self.match_history = match_history.copy()

        self.match_history["date"] = pd.to_datetime(self.match_history["date"], utc=True)
        self.match_history = self.match_history.sort_values("date").reset_index(drop=True)

        teams = pd.concat([
            self.match_history[["home_team", "home_team_id"]].set_axis(["team", "id"], axis=1),
            self.match_history[["away_team", "away_team_id"]].set_axis(["team", "id"], axis=1),
        ])

        duplicates = teams.groupby("team").id.nunique()
        if (duplicates > 1).any():
            raise ValueError(f"Team names map to multiple IDs: {', '.join(duplicates[duplicates > 1].index)}")

        self.team_ids = teams.drop_duplicates("team").set_index("team").id.to_dict()

    def predict(self, home_team: str, away_team: str, fixture_date) -> str:
        return {0: "draw", 1: "home_win", 2: "away_win"}[self.model.predict(
            self._create_features(home_team, away_team, fixture_date)
        )[0]]

    def predict_proba(self, home_team: str, away_team: str, fixture_date) -> dict[str, float]:
        probabilities = self.model.predict_proba(self._create_features(home_team, away_team, fixture_date))[0]
        return dict(zip(("draw", "home_win", "away_win"), probabilities))

    def _create_features(self, home_team: str, away_team: str, fixture_date) -> pd.DataFrame:
        if home_team == away_team:
            raise ValueError("Home and away teams must differ.")
        try:
            home_id, away_id = self.team_ids[home_team], self.team_ids[away_team]
        except KeyError as error:
            raise ValueError(f"Unknown team: {error.args[0]}") from error

        fixture_date = pd.Timestamp(fixture_date)
        fixture_date = fixture_date.tz_localize("UTC") if fixture_date.tzinfo is None else fixture_date.tz_convert("UTC")

        history = self.match_history[self.match_history.date < fixture_date]

        elo = EloCalculator(history)
        elo.add_features()
        form, goals = FormCalculator(history), GoalCalculator(history)

        home = self._team_features(home_id, fixture_date, "home", form, goals)
        away = self._team_features(away_id, fixture_date, "away", form, goals)
        home_elo, away_elo = elo.get_elo(home_id), elo.get_elo(away_id)

        values = [home_id, away_id, home_elo, away_elo, home_elo - away_elo,
                  home[0], away[0], home[1], away[1], home[2], away[2],
                  home[3], home[4], away[3], away[4], home[5], away[5]]
        features = pd.DataFrame([values], columns=FEATURE_COLUMNS)

        if features.isna().any().any():
            raise ValueError("Insufficient historical data to create every feature.")

        return features.reindex(columns=getattr(self.model, "feature_names_in_", FEATURE_COLUMNS))

    @staticmethod
    def _team_features(team_id, fixture_date, location, form, goals):
        recent_goals = goals.get_team_goals(team_id, fixture_date)

        if recent_goals is None:
            raise ValueError("Insufficient five-match history for one or both teams.")

        return (
            form.get_team_form(team_id, fixture_date),
            form.get_location_specific_form(team_id, fixture_date, location),
            form.get_seasonal_ppg(team_id, fixture_date),
            *recent_goals,
            goals.get_season_average_goals(team_id, fixture_date),
        )
