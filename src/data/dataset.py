from pathlib import Path
from data.processing import process_matches, save_processed_data
from features.Goals import GoalCalculator
from features.elo import EloCalculator
from features.form import FormCalculator

def build_dataset():
    project_root = Path(__file__).resolve().parents[2]
    dataset_data_dir = project_root / "data" / "dataset"

    processed_matches = process_matches()
    save_processed_data(processed_matches)

    df = EloCalculator(processed_matches).add_features()
    df = FormCalculator(df).add_features()
    df = GoalCalculator(df).add_features()

    for i, match in df.iterrows():
        if match["home_goals"] > match["away_goals"]:
            df.loc[i, "result"] = 1

        elif match["home_goals"] < match["away_goals"]:
            df.loc[i, "result"] = 2

        else:
            df.loc[i, "result"] = 0

    df = df.drop(columns=["fixture_id", "round", "home_goals", "away_goals", "home_team", "away_team", "status"])
    print("Matches count before cleaning:", len(df))

    df = df.dropna()
    print("Matches count after cleaning:", len(df))

    df.to_csv(
        dataset_data_dir / "training_dataset.csv",
        index=False
    )


