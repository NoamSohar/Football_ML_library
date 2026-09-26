# Football Machine Learning Library⚽

An experimental machine-learning project that provides tools for building, training, and evaluating football match prediction models.

Note: The library is still evolving, so some parts aren't fully documented yet.

## What it does?
The library provides tools to:

- Load, validate and clean historical match data.
- Generate features such as Elo ratings, recent form, season form, and goals per-game.
- Build and train XGBoost models for match outcome prediction. (see why XGBoost is used below⬇️)
- Tune model parameters and class weights using chronological data splits.
- Find a custom draw prediction threshold.
- Evaluate models using accuracy, F1, log loss, confusion matrices and other metrics.
- Save and load trained models for later use.


## Why XGBoost?
XGBoost works well for football prediction (and generally all sport predictions) because the data is mostly structured, tabular features such as Elo ratings, form, and goals per game. It performs well even on relatively small datasets and can capture nonlinear relationships between these features. It also requires little preprocessing and provides class probabilities, which are useful when predicting home wins, draws, and away wins.


## How to handle data:
Place raw match CSV files in `data/raw/`. Each file should include:

`fixture_id`, `date`, `round`, `home_team_id`, `home_team`, `away_team_id`, `away_team`, `home_goals`, `away_goals`, and `status`.

I got my match data from API-Football, but you can use any data source as long as it follows the required format.

Only fixtures with `status == "FT"` are used. Every feature uses matches strictly before the fixture match.

## Setup
Clone the repository and create a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

Build processed and training datasets:

```powershell
python -m src.train_model build-dataset
```

Tune, calibrate, evaluate, and save a model:

```powershell
python -m src.train_model train
```

Evaluate a saved model on the untouched final test period:

```powershell
python -m src.train_model evaluate
```

Use `--search-iterations 20` to evaluate more sampled XGBoost configurations. The default evaluates four configurations and uses a 60% training, 20% calibration, and 20% final test split, in chronological order.

## Evaluation

The command reports accuracy, macro F1, log loss, draw precision, draw recall, draw F1, draw Brier score, a draw calibration curve, and the confusion matrix. Calibration metrics are development diagnostics; only the final test metrics should be used to describe model performance.

## Saved model metadata

Saved artifacts include the selected XGBoost parameters, draw weight, draw threshold, selection metric, feature columns, data date ranges, and a deterministic fingerprint of the training dataset.
