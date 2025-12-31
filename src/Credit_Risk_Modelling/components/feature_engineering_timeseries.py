import pandas as pd
import numpy as np
from Credit_Risk_Modelling.entity.feature_engineering_entity import TimeSeriesFeatureConfig

class TimeSeriesFeatureEngineering:
    def __init__(self, config: TimeSeriesFeatureConfig):
        self.config = config

    def transform(self):
        df = pd.read_csv(self.config.data_path)

        df = df.sort_values("TransactionDT")

        features = []
        for window in range(1, self.config.window_size + 1):
            df[f"amt_mean_{window}"] = (
                df["TransactionAmt"].rolling(window).mean()
            )
            df[f"amt_std_{window}"] = (
                df["TransactionAmt"].rolling(window).std()
            )

        df = df.dropna()

        df.to_csv(self.config.output_path / "timeseries_features.csv", index=False)
        return df
