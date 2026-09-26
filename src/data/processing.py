from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_matches
from src.data.load import RAW_DATA_DIR, load_all_seasons
from src.data.validation import validate_matches


# Path used to save processed match data.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def process_matches(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Loads, validates, and cleans raw match data.

    Returns:
        A validated DataFrame containing finished matches in chronological order.

    Raises:
        FileNotFoundError: If the raw match-data files cannot be found.
        ValueError: If the raw or cleaned match data fails validation.
    """
    df = load_all_seasons(raw_data_dir)
    validate_matches(df)

    df = clean_matches(df)
    validate_matches(df)

    return df


def save_processed_data(
    df: pd.DataFrame,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    file_name: str = "processed_data.csv",
) -> Path:
    """
    Saves processed match data to a CSV file.

    Args:
        df: Processed match data to save.
        processed_data_dir: Directory where the processed data is saved.
        file_name: Name of the processed data file.

    Returns:
        The path of the saved CSV file.
    """
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / file_name
    df.to_csv(output_path, index=False)

    return output_path
