import pandas as pd
from Credit_Risk_Modelling.entity.feature_engineering_entity import TimeSeriesFeatureConfig


class TimeSeriesFeatureEngineering:
    def __init__(self, config: TimeSeriesFeatureConfig):
        self.config = config

    def transform(self):
        df = pd.read_csv(self.config.data_path)

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Required columns check
        required_cols = {"customer_id", "month", "income", "expense", "balance"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for time-series FE: {missing}")

        # Sort by entity and time
        df = df.sort_values(["customer_id", "month"])

        # Rolling feature engineering PER CUSTOMER
        for window in range(1, self.config.window_size + 1):
            df[f"income_mean_{window}"] = (
                df.groupby("customer_id")["income"]
                  .rolling(window)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

            df[f"expense_mean_{window}"] = (
                df.groupby("customer_id")["expense"]
                  .rolling(window)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

            df[f"balance_mean_{window}"] = (
                df.groupby("customer_id")["balance"]
                  .rolling(window)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

        # Drop rows with insufficient history
        df = df.dropna()

        output_file = self.config.output_path / "timeseries_features.csv"
        df.to_csv(output_file, index=False)

        return df
