import numpy as np

from xgboost import XGBClassifier


def train_model(
    x_train,
    y_train,
    model_params,
    draw_weight=1.0,
):
    if draw_weight <= 0:
        raise ValueError("draw_weight must be positive.")

    model = XGBClassifier(**model_params)
    sample_weight = np.where(y_train == 0, draw_weight, 1.0)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
    )
    return model
