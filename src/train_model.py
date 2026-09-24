import argparse
import json
from pathlib import Path

from sklearn.model_selection import ParameterSampler

from src.data.dataset import build_dataset
from src.model.data import load_training_data, split_data
from src.model.evaluation import evaluate_model
from src.model.storage import load_model, save_model
from src.model.train import run_training


PARAMETER_DISTRIBUTIONS = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3],
    "min_child_weight": [1, 3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [1.0, 2.0],
}
DRAW_WEIGHTS = [1.0, 1.25, 1.5, 2.0, 3.0]
BASE_MODEL_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "random_state": 42,
    "n_jobs": 1,
}


def model_configurations(iterations):
    sampled = ParameterSampler(PARAMETER_DISTRIBUTIONS, n_iter=iterations, random_state=42)
    return [{**BASE_MODEL_PARAMS, **params} for params in sampled]


def print_metrics(title, metrics):
    print(f"{title} metrics")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"  Log loss: {metrics['log_loss']:.3f}")
    print(f"  Draw precision: {metrics['draw_precision']:.3f}")
    print(f"  Draw recall: {metrics['draw_recall']:.3f}")
    print(f"  Draw F1: {metrics['draw_f1']:.3f}")
    print(f"  Draw Brier score: {metrics['draw_brier_score']:.3f}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")


def train(args):
    result = run_training(
        args.dataset,
        model_configurations(args.search_iterations),
        DRAW_WEIGHTS,
        args.train_ratio,
        args.calibration_ratio,
    )
    save_model(
        result["model"],
        result["x_train"].columns,
        result["data"],
        result["draw_threshold"],
        args.model_path,
        draw_weight=result["draw_weight"],
        selection_metric="macro_f1",
        split_date_ranges=result["split_date_ranges"],
        dataset_fingerprint=result["dataset_fingerprint"],
    )
    print("Selected parameters:", result["model_params"])
    print(f"Selected draw weight: {result['draw_weight']:.2f}")
    print(f"Draw threshold calibrated on calibration period: {result['draw_threshold']:.2f}")
    print("Nested validation macro F1:", [round(score["macro_f1"], 3) for score in result["nested_scores"]])
    print_metrics("Calibration", result["calibration_metrics"])
    print_metrics("Final test", result["test_metrics"])


def evaluate(args):
    artifact = load_model(args.model_path)
    data = load_training_data(args.dataset)
    x_train, y_train, _, _, x_test, y_test = split_data(
        data, args.train_ratio, args.calibration_ratio
    )
    metrics = evaluate_model(
        artifact["model"], x_test, y_train, y_test, artifact["draw_threshold"]
    )
    print_metrics("Final test", metrics)
    print("Draw calibration curve:", json.dumps(metrics["draw_calibration_curve"], indent=2))


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train and evaluate the football outcome model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-dataset", help="Create processed and training datasets.")

    for name, help_text in [
        ("train", "Tune, calibrate, test, and save a model."),
        ("evaluate", "Evaluate a saved model on the untouched final test period."),
    ]:
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--dataset", type=Path, default=root / "data" / "dataset" / "training_dataset.csv"
        )
        command.add_argument(
            "--model-path", type=Path, default=root / "data" / "models" / "football_model.joblib"
        )
        command.add_argument("--train-ratio", type=float, default=0.6)
        command.add_argument("--calibration-ratio", type=float, default=0.2)
        if name == "train":
            command.add_argument("--search-iterations", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "build-dataset":
        build_dataset()
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
