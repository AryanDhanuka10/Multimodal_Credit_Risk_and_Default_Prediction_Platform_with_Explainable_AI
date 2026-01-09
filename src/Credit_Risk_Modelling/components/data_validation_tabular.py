import pandas as pd
import logging
from Credit_Risk_Modelling.entity.data_validation_entity import DataValidationConfig

class TabularDataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self):
        logging.info("Validating tabular dataset")
        df = pd.read_excel(self.config.data_path, header=1)

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(".", "_")
        )

        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if df.isnull().mean().max() > 0.4:
            raise ValueError("Too many missing values in tabular data")

        logging.info("Tabular data validation passed")
        return True
