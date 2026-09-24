import pandas as pd

class EloCalculator:
    def __init__(self, matches: pd.DataFrame, starting_elo: float=1500, k: int=20):
        self.starting_elo = starting_elo
        self.k = k
        self.ratings = {}
        self.matches = matches

    def get_elo(self, team_id: int) -> float:
        if team_id not in self.ratings:
            self.ratings[team_id] = self.starting_elo

        return self.ratings[team_id]

    def calculate_chance(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400)
        )

    def update_ratings(self, home_team_id: int, away_team_id: int, home_goals: int, away_goals: int) -> None:

        home_rating = self.get_elo(home_team_id)
        away_rating = self.get_elo(away_team_id)

        home_chance = self.calculate_chance(home_rating, away_rating)
        away_chance = self.calculate_chance(away_rating, home_rating)

        if home_goals > away_goals:
            home_score = 1
            away_score = 0

        elif home_goals < away_goals:
            home_score = 0
            away_score = 1

        else:
            home_score = 0.5
            away_score = 0.5

        self.ratings[home_team_id] = (
                home_rating
                + self.k
                * (home_score - home_chance)
        )

        self.ratings[away_team_id] = (
                away_rating
                + self.k
                * (away_score - away_chance)
        )

    def add_features(self) -> pd.DataFrame:
        df = self.matches.copy()
        df = df.sort_values("date").reset_index(drop=True)

        home_elos = []
        away_elos = []

        for _, fixtures_at_kickoff in df.groupby("date", sort=True):
            for _, match in fixtures_at_kickoff.iterrows():
                home_elos.append(self.get_elo(match["home_team_id"]))
                away_elos.append(self.get_elo(match["away_team_id"]))

            for _, match in fixtures_at_kickoff.iterrows():
                self.update_ratings(
                    match["home_team_id"],
                    match["away_team_id"],
                    match["home_goals"],
                    match["away_goals"],
                )

        df["home_elo"] = home_elos
        df["away_elo"] = away_elos
        df["elo_diff"] = (df["home_elo"] - df["away_elo"])

        return df
