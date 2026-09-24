from src.data.load import load_all_seasons
from src.data.validation import validate_matches
from src.data.cleaning import clean_matches

from pathlib import Path
import pandas as pd

def process_matches() -> pd.DataFrame:
    df = load_all_seasons()

    validate_matches(df)

    df = clean_matches(df)

    validate_matches(df)

    return df

def save_processed_data(df: pd.DataFrame):
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        processed_data_dir / "processed_data.csv",
        index=False
    )
