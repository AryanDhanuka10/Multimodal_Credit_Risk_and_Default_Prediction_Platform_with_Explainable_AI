import pandas as pd
import logging
from pathlib import Path

class TextDataValidation:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def validate(self):
        logging.info("Validating text dataset")
        df = pd.read_csv(self.data_path)

        if df.shape[0] < 1000:
            raise ValueError("Text dataset too small")

        if df.isnull().mean().max() > 0.5:
            raise ValueError("Too many nulls in text data")

        logging.info("Text data validation passed")
        return True
