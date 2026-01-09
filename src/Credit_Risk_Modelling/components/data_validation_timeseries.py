import pandas as pd
import logging
from pathlib import Path

class TimeSeriesDataValidation:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.entity_col = None
        self.time_col = None

    def validate(self):
        logging.info("Validating time-series (panel) data")
        df = pd.read_csv(self.data_path)

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Detect entity identifier
        possible_entity_cols = ["customer_id", "user_id", "account_id"]
        for col in possible_entity_cols:
            if col in df.columns:
                self.entity_col = col
                break

        if self.entity_col is None:
            raise ValueError("No entity identifier column found")

        # Detect time index
        possible_time_cols = ["month", "time", "period"]
        for col in possible_time_cols:
            if col in df.columns:
                self.time_col = col
                break

        if self.time_col is None:
            raise ValueError("No time index column found")

        # Check ordering per entity
        if not df.sort_values([self.entity_col, self.time_col]).equals(df):
            logging.warning(
                "Time-series data is not ordered by entity and time index"
            )

        logging.info(
            f"Validated panel time-series with entity='{self.entity_col}' "
            f"and time='{self.time_col}'"
        )

        return True
