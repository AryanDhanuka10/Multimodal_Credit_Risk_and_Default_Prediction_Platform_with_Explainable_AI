import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

class TabularFeatureEngineering:
    def __init__(self, data_path: Path, output_path: Path):
        self.data_path = data_path
        self.output_path = output_path

    def transform(self):
        df = pd.read_excel(self.data_path, header=1)

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(".", "_")
        )

        y = df["default_payment_next_month"]
        X = df.drop("default_payment_next_month", axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        joblib.dump(scaler, self.output_path / "scaler.pkl")

        return X_scaled, y
