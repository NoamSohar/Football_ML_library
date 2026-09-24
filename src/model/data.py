import pandas as pd


TEAM_ID_COLUMNS = ["home_team_id", "away_team_id"]


def load_training_data(path):
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def split_data(data, train_ratio=0.6, calibration_ratio=0.2):
    if not 0 < train_ratio < 1 or not 0 < calibration_ratio < 1:
        raise ValueError("Split ratios must be between 0 and 1.")
    if train_ratio + calibration_ratio >= 1:
        raise ValueError("train_ratio + calibration_ratio must be less than 1.")

    train_end = int(len(data) * train_ratio)
    calibration_end = train_end + int(len(data) * calibration_ratio)
    if train_end == 0 or calibration_end == train_end or calibration_end == len(data):
        raise ValueError("Dataset is too small for train, calibration, and test splits.")

    train_data = data.iloc[:train_end]
    calibration_data = data.iloc[train_end:calibration_end]
    test_data = data.iloc[calibration_end:]
    columns_to_drop = [
        column for column in ["date", "result", *TEAM_ID_COLUMNS]
        if column in data.columns
    ]

    return (
        train_data.drop(columns=columns_to_drop),
        train_data["result"],
        calibration_data.drop(columns=columns_to_drop),
        calibration_data["result"],
        test_data.drop(columns=columns_to_drop),
        test_data["result"],
    )
