from pathlib import Path
from Credit_Risk_Modelling.config.configuration import ConfigurationManager
from Credit_Risk_Modelling.components.data_ingestion_tabular import TabularDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_timeseries import TimeSeriesDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_documents import DocumentDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_text import TextDataIngestion
from Credit_Risk_Modelling.entity.data_ingestion_entity import DataIngestionConfig

class TrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager(Path("config/config.yaml"))

    def run_data_ingestion(self):
        di = self.config.get_data_ingestion_config()

        TabularDataIngestion(
            DataIngestionConfig(
                di.tabular.root_dir,
                di.tabular.source_url,
                Path(di.tabular.local_file)
            )
        ).ingest()

        TimeSeriesDataIngestion(
            DataIngestionConfig(
                di.timeseries.root_dir,
                di.timeseries.source_url,
                Path(di.timeseries.local_file)
            )
        ).ingest()

        DocumentDataIngestion(
            Path(di.documents.local_dir)
        ).ingest()

        TextDataIngestion(
            Path(di.text.local_file)
        ).ingest()
