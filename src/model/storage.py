from pathlib import Path
import joblib


def save_model(
    model,
    feature_columns,
    training_data,
    draw_threshold,
    path,
    *,
    draw_weight=1.0,
    selection_metric="macro_f1",
    split_date_ranges=None,
    dataset_fingerprint=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": list(feature_columns),
        "model_params": model.get_params(),
        "train_date_range": [str(training_data["date"].min()), str(training_data["date"].max())],
        "draw_threshold": draw_threshold,
        "draw_weight": draw_weight,
        "selection_metric": selection_metric,
        "split_date_ranges": split_date_ranges or {},
        "dataset_fingerprint": dataset_fingerprint,
    }, path)


def load_model(path):
    return joblib.load(path)
