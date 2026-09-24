# Football Outcome Predictor

An experimental machine-learning project that predicts FootBall match outcomes: draw, home win, or away win.

## What it does

1. Validates and cleans historical finished fixtures.
2. Builds pre-kickoff Elo, five-match form, season form, and goals-per-game features.
3. Tunes an XGBoost classifier and draw-class weight with chronological nested validation.
4. Calibrates the draw decision threshold on a separate chronological period.
5. Reports performance once on an untouched final test period.

This is an educational portfolio project. It is not betting advice or a production prediction service.

## Data

Place raw match CSV files in `data/raw/`. Each file must include:

`fixture_id`, `date`, `round`, `home_team_id`, `home_team`, `away_team_id`, `away_team`, `home_goals`, `away_goals`, and `status`.

Only fixtures with `status == "FT"` are used. Every feature uses matches strictly before the fixture kickoff. Simultaneous kickoffs receive Elo ratings before any fixture in that kickoff group updates ratings.

## Setup

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
