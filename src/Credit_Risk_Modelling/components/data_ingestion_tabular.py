import logging
from Credit_Risk_Modelling.utils.common import download_file, calculate_md5
from Credit_Risk_Modelling.entity.data_ingestion_entity import DataIngestionConfig

class TabularDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self):
        logging.info("Starting tabular data ingestion")
        download_file(self.config.source_url, self.config.local_path)
        checksum = calculate_md5(self.config.local_path)
        logging.info(f"Tabular data checksum: {checksum}")
        return self.config.local_path
