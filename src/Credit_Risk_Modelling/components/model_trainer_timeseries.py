import pandas as pd
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

class TimeSeriesModelTrainer:
    def __init__(self, data_path: Path, model_path: Path):
        self.data_path = data_path
        self.model_path = model_path

    def train(self):
        df = pd.read_csv(self.data_path)

        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]

        model = LGBMClassifier(n_estimators=200, max_depth=6)
        model.fit(X, y)

        joblib.dump(model, self.model_path)
        return model
