from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

from .thresholds import draw_calibration_metrics, draw_class_metrics, macro_f1_score, predict_results


def evaluate_probabilities(probabilities, y_reference, results, draw_threshold):
    predictions = predict_results(probabilities, draw_threshold)
    labels = [0, 1, 2]
    baseline = y_reference.mode().iloc[0]

    metrics = {
        "draw_threshold": draw_threshold,
        "accuracy": accuracy_score(results, predictions),
        "macro_f1": macro_f1_score(results, predictions),
        "baseline_accuracy": accuracy_score(results, [baseline] * len(results)),
        "log_loss": log_loss(results, probabilities, labels=labels),
        "confusion_matrix": confusion_matrix(results, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            results, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }
    metrics.update(draw_class_metrics(results, predictions))
    metrics.update(draw_calibration_metrics(probabilities, results))
    return metrics


def evaluate_model(model, x_evaluation, y_reference, y_evaluation, draw_threshold):
    probabilities = model.predict_proba(x_evaluation)
    return evaluate_probabilities(probabilities, y_reference, y_evaluation, draw_threshold)
