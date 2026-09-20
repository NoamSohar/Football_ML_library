import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self.df = None
        self.model = None

        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None

    def load_data(self):
        self.df = pd.read_csv(self.dataset_path)

        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)


    def split_data(self, val_ratio: float = 0.2):
        split_index = int(len(self.df) * (1 - val_ratio))

        train_df = self.df.iloc[:split_index]
        val_df = self.df.iloc[split_index:]

        self.x_train = train_df.drop(columns=["date", "result"])
        self.y_train = train_df["result"]

        self.x_val = val_df.drop(columns=["date", "result"])
        self.y_val = val_df["result"]

    def train(self, model_params: dict):
        self.model = XGBClassifier(**model_params)

        self.model.fit(
            self.x_train,
            self.y_train,
            eval_set = [(self.x_val, self.y_val)],
            verbose = True
        )

    def evaluate(self):
        predictions = self.model.predict(self.x_val)

        accuracy = accuracy_score(
            self.y_val,
            predictions
        )

        return accuracy