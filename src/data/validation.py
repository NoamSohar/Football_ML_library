REQUIRED_COLUMNS = {
    "fixture_id",
    "date",
    "round",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "status"
}


def validate_columns(df):
    missing = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing.append(column)

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_fixture_ids(df):
    if df["fixture_id"].isna().any():
        raise ValueError("Missing fixture IDs found")

    if df["fixture_id"].duplicated().any():
        raise ValueError("Duplicate fixture IDs found")


def validate_teams(df):
    if (df["home_team_id"] == df["away_team_id"]).any():
        raise ValueError("Home team and away team cannot be the same")


def validate_scores(df):
    finished = df[df["status"] == "FT"]

    if finished["home_goals"].isna().any():
        raise ValueError("Home goals cannot be missing")

    if finished["away_goals"].isna().any():
        raise ValueError("Away goals cannot be missing")

    if (finished["home_goals"] < 0).any():
        raise ValueError("Home goals cannot be negative")

    if (finished["away_goals"] < 0).any():
        raise ValueError("Away goals cannot be negative")

def validate_matches(df):
    validate_columns(df)
    validate_fixture_ids(df)
    validate_teams(df)
    validate_scores(df)
