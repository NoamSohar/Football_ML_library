import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .data import load_training_data, split_data
from .evaluation import evaluate_model
from .thresholds import find_draw_threshold, macro_f1_score, predict_results
from .training import train_model


def dataset_fingerprint(data):
    values = pd.util.hash_pandas_object(data, index=True).values.tobytes()
    return hashlib.sha256(values).hexdigest()


def _split_calibration_and_scoring(indices):
    midpoint = len(indices) // 2
    if midpoint == 0 or midpoint == len(indices):
        raise ValueError("Validation window is too small to calibrate and score a threshold.")
    return indices[:midpoint], indices[midpoint:]


def select_configuration(x_train, y_train, model_configs, draw_weights, n_splits=3):
    """Choose settings using only chronologically earlier data."""
    candidates = []
    splits = TimeSeriesSplit(n_splits=n_splits)

    for model_params in model_configs:
        for draw_weight in draw_weights:
            fold_scores = []
            for fit_index, validation_index in splits.split(x_train):
                calibration_index, scoring_index = _split_calibration_and_scoring(validation_index)
                model = train_model(
                    x_train.iloc[fit_index], y_train.iloc[fit_index], model_params, draw_weight
                )
                calibration_probabilities = model.predict_proba(x_train.iloc[calibration_index])
                threshold = find_draw_threshold(
                    calibration_probabilities, y_train.iloc[calibration_index]
                )
                scoring_probabilities = model.predict_proba(x_train.iloc[scoring_index])
                scoring_predictions = predict_results(scoring_probabilities, threshold)
                fold_scores.append(
                    macro_f1_score(y_train.iloc[scoring_index], scoring_predictions)
                )

            candidates.append(
                {
                    "mean_macro_f1": float(np.mean(fold_scores)),
                    "fold_macro_f1": [float(score) for score in fold_scores],
                    "model_params": model_params,
                    "draw_weight": float(draw_weight),
                }
            )

    return max(candidates, key=lambda candidate: candidate["mean_macro_f1"])


def nested_time_series_evaluation(x_train, y_train, model_configs, draw_weights, n_splits=3):
    """Estimate tuning performance with outer chronological evaluation folds."""
    outer_scores = []
    outer_splits = TimeSeriesSplit(n_splits=n_splits)

    for outer_fit_index, outer_test_index in outer_splits.split(x_train):
        outer_features = x_train.iloc[outer_fit_index]
        outer_results = y_train.iloc[outer_fit_index]
        selected = select_configuration(
            outer_features, outer_results, model_configs, draw_weights, n_splits=2
        )
        threshold_fit_index, threshold_calibration_index = _split_calibration_and_scoring(
            np.arange(len(outer_features))
        )
        model = train_model(
            outer_features.iloc[threshold_fit_index],
            outer_results.iloc[threshold_fit_index],
            selected["model_params"],
            selected["draw_weight"],
        )
        probabilities = model.predict_proba(outer_features.iloc[threshold_calibration_index])
        threshold = find_draw_threshold(
            probabilities, outer_results.iloc[threshold_calibration_index]
        )
        outer_predictions = predict_results(model.predict_proba(x_train.iloc[outer_test_index]), threshold)
        outer_scores.append(
            {
                "macro_f1": float(macro_f1_score(y_train.iloc[outer_test_index], outer_predictions)),
                "draw_weight": selected["draw_weight"],
                "draw_threshold": threshold,
            }
        )

    return outer_scores


def run_training(
    dataset_path,
    model_configs,
    draw_weights,
    train_ratio=0.6,
    calibration_ratio=0.2,
):
    data = load_training_data(dataset_path)
    x_train, y_train, x_calibration, y_calibration, x_test, y_test = split_data(
        data, train_ratio, calibration_ratio
    )
    selection = select_configuration(x_train, y_train, model_configs, draw_weights)
    nested_scores = nested_time_series_evaluation(
        x_train, y_train, model_configs, draw_weights
    )
    model = train_model(
        x_train, y_train, selection["model_params"], selection["draw_weight"]
    )
    calibration_probabilities = model.predict_proba(x_calibration)
    draw_threshold = find_draw_threshold(calibration_probabilities, y_calibration)
    calibration_metrics = evaluate_model(
        model, x_calibration, y_train, y_calibration, draw_threshold
    )
    test_metrics = evaluate_model(model, x_test, y_train, y_test, draw_threshold)
    train_end = len(x_train)
    calibration_end = train_end + len(x_calibration)
    split_date_ranges = {
        "train": [str(data.iloc[:train_end]["date"].min()), str(data.iloc[:train_end]["date"].max())],
        "calibration": [str(data.iloc[train_end:calibration_end]["date"].min()), str(data.iloc[train_end:calibration_end]["date"].max())],
        "test": [str(data.iloc[calibration_end:]["date"].min()), str(data.iloc[calibration_end:]["date"].max())],
    }

    return {
        "data": data,
        "model": model,
        "x_train": x_train,
        "y_train": y_train,
        "x_calibration": x_calibration,
        "y_calibration": y_calibration,
        "x_test": x_test,
        "y_test": y_test,
        "draw_threshold": draw_threshold,
        "draw_weight": selection["draw_weight"],
        "model_params": selection["model_params"],
        "selection": selection,
        "nested_scores": nested_scores,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "split_date_ranges": split_date_ranges,
        "dataset_fingerprint": dataset_fingerprint(data),
    }
