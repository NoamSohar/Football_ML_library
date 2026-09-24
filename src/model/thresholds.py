import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_fscore_support


def predict_results(probabilities, draw_threshold=None):
    if draw_threshold is None:
        return probabilities.argmax(axis=1)

    return np.where(
        probabilities[:, 0] >= draw_threshold,
        0,
        np.where(probabilities[:, 1] >= probabilities[:, 2], 1, 2),
    )


def find_draw_threshold(probabilities, results, minimum=0.1, maximum=0.5, step=0.01):
    if not 0 < minimum <= maximum <= 1 or step <= 0:
        raise ValueError("Thresholds must be between 0 and 1, with a positive step.")

    thresholds = np.arange(minimum, maximum + step / 2, step)
    return float(max(
        thresholds,
        key=lambda threshold: macro_f1_score(
            results, predict_results(probabilities, threshold)
        ),
    ))


def draw_class_metrics(results, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        results, predictions, labels=[0], zero_division=0
    )
    return {
        "draw_precision": float(precision[0]),
        "draw_recall": float(recall[0]),
        "draw_f1": float(f1[0]),
    }


def macro_f1_score(results, predictions):
    return f1_score(
        results, predictions, labels=[0, 1, 2], average="macro", zero_division=0
    )


def draw_calibration_metrics(probabilities, results, bins=5):
    observed, predicted = calibration_curve(
        np.asarray(results) == 0, probabilities[:, 0], n_bins=bins, strategy="uniform"
    )
    return {
        "draw_brier_score": float(brier_score_loss(np.asarray(results) == 0, probabilities[:, 0])),
        "draw_calibration_curve": [
            {"mean_predicted_probability": float(pred), "draw_frequency": float(obs)}
            for pred, obs in zip(predicted, observed)
        ],
    }
