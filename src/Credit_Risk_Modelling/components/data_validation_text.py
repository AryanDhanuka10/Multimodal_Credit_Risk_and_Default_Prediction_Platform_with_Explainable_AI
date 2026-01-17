import pandas as pd
import logging
from pathlib import Path

class TextDataValidation:
    def __init__(self, data_path: Path, text_column: str = "Consumer complaint narrative"):
        self.data_path = data_path
        self.text_column = text_column

    def validate(self):
        logging.info("Validating text dataset")

        df = pd.read_csv(self.data_path)

        if self.text_column not in df.columns:
            logging.warning(f"Text column '{self.text_column}' not found. Skipping text pipeline.")
            return False

        null_ratio = df[self.text_column].isnull().mean()

        logging.info(f"Text null ratio: {null_ratio:.2f}")

        # Allow high nulls but warn
        if null_ratio > 0.8:
            logging.warning(
                "High proportion of missing text data. "
                "Text model will be trained on filtered subset only."
            )

        # Drop null text rows for downstream NLP
        df = df.dropna(subset=[self.text_column])

        if df.empty:
            logging.warning("No valid text samples after cleaning. Skipping text pipeline.")
            return False

        logging.info(f"Text validation passed with {len(df)} usable samples")
        return True
