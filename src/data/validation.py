import pandas as pd

REQUIRED_MATCH_COLUMNS = frozenset({
    "fixture_id",
    "date",
    "round",
    "home_team_id",
    "home_team",
    "away_team_id",
    "away_team",
    "home_goals",
    "away_goals",
    "status",
})

def validate_columns(df: pd.DataFrame) -> None:
    """
    Validates that match data includes every required column.

    Args:
        df (pd.DataFrame): Match data to validate.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    missing_columns = REQUIRED_MATCH_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")


def validate_fixture_ids(df: pd.DataFrame) -> None:
    """
    Validates that every match has a unique fixture ID.

    Args:
        df (pd.DataFrame): Match data to validate.

    Raises:
        ValueError: If fixture IDs are missing or duplicated.
    """

    if df["fixture_id"].isna().any():
        raise ValueError("Missing fixture IDs found")

    if df["fixture_id"].duplicated().any():
        raise ValueError("Duplicate fixture IDs found")


def validate_teams(df: pd.DataFrame) -> None:
    """
    Validates that no match has the same home and away team.

    Args:
        df (pd.DataFrame): Match data to validate.

    Raises:
        ValueError: If a match has identical home and away team IDs.
    """

    if (df["home_team_id"] == df["away_team_id"]).any():
        raise ValueError("Home team and away team cannot be the same")


def validate_scores(df: pd.DataFrame) -> None:
    """
    Validates scores for finished matches.

    Args:
        df (pd.DataFrame): Match data to validate.

    Raises:
        ValueError: If a finished match has missing or negative scores.
    """

    finished = df[df["status"] == "FT"]

    if finished["home_goals"].isna().any():
        raise ValueError("Home goals cannot be missing")

    if finished["away_goals"].isna().any():
        raise ValueError("Away goals cannot be missing")

    if (finished["home_goals"] < 0).any():
        raise ValueError("Home goals cannot be negative")

    if (finished["away_goals"] < 0).any():
        raise ValueError("Away goals cannot be negative")

def validate_matches(df: pd.DataFrame) -> None:
    """
    Runs all validation checks for match data.

    Args:
        df (pd.DataFrame): Match data to validate.

    Raises:
        ValueError: If any match-data validation check fails.
    """

    validate_columns(df)
    validate_fixture_ids(df)
    validate_teams(df)
    validate_scores(df)
