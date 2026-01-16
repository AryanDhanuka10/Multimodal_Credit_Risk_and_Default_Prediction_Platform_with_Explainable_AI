import pandas as pd
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path

class TimeSeriesModelTrainer:
    def __init__(self, data_path: Path, model_path: Path, target_col: str):
        self.data_path = data_path
        self.model_path = model_path
        self.target_col = target_col


    def train(self):
        df = pd.read_csv(self.data_path)

        TARGET_COL = "default_flag"

        X = df.drop(TARGET_COL, axis=1)
        y = df[TARGET_COL]


        model = LGBMClassifier(n_estimators=200, max_depth=6)
        model.fit(X, y)

        joblib.dump(model, self.model_path)
        return model
