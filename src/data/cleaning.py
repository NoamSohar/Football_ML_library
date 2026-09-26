import pandas as pd

def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned matches dataframe sorted chronologically,
    and using only finished matches. (status: FT)

    Args:
        df (pd.DataFrame): Raw matches dataframe.

    Returns:
        df (pd.DataFrame): Cleaned matches dataframe.
    """
    cleaned_df = df.copy()

    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], utc=True)
    cleaned_df = cleaned_df.loc[cleaned_df["status"] == "FT"]

    cleaned_df = cleaned_df.sort_values("date").reset_index(drop=True)

    return cleaned_df