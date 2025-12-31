import pandas as pd
import logging
from pathlib import Path

class TimeSeriesDataValidation:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def validate(self):
        logging.info("Validating time-series data")
        df = pd.read_csv(self.data_path)

        if "TransactionDT" not in df.columns:
            raise ValueError("Missing timestamp column")

        if not df["TransactionDT"].is_monotonic_increasing:
            logging.warning("Time-series not strictly ordered")

        logging.info("Time-series validation passed")
        return True
