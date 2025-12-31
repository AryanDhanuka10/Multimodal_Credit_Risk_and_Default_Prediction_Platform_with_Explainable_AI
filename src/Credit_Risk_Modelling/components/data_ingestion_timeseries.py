import logging
from pathlib import Path
from Credit_Risk_Modelling.entity.data_ingestion_entity import DataIngestionConfig
from Credit_Risk_Modelling.utils.common import calculate_md5

class TimeSeriesDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self):
        logging.info("Validating time-series dataset")
        if not Path(self.config.local_path).exists():
            raise FileNotFoundError("Time-series data not found")
        checksum = calculate_md5(Path(self.config.local_path))
        logging.info(f"Time-series data checksum: {checksum}")
        return self.config.local_path
