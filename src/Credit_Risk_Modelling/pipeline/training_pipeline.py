from pathlib import Path
from Credit_Risk_Modelling.config.configuration import ConfigurationManager
from Credit_Risk_Modelling.components.data_ingestion_tabular import TabularDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_timeseries import TimeSeriesDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_documents import DocumentDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_text import TextDataIngestion
from Credit_Risk_Modelling.entity.data_ingestion_entity import DataIngestionConfig
from Credit_Risk_Modelling.components.data_validation_tabular import TabularDataValidation
from Credit_Risk_Modelling.components.feature_engineering_tabular import TabularFeatureEngineering
from Credit_Risk_Modelling.entity.data_validation_entity import DataValidationConfig
from Credit_Risk_Modelling.components.data_validation_documents import DocumentDataValidation
from Credit_Risk_Modelling.components.data_validation_timeseries import TimeSeriesDataValidation
from Credit_Risk_Modelling.components.data_validation_text import TextDataValidation



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

        TabularDataValidation(
            DataValidationConfig(
                required_columns=["default_payment_next_month"],
                data_path=Path("artifacts/data_ingestion/tabular/credit_default.xls")
            )
        ).validate()

        TimeSeriesDataValidation(
            Path("artifacts/data_ingestion/timeseries/transactions.csv")
        ).validate()

        DocumentDataValidation(
            Path("artifacts/data_ingestion/documents/images")
        ).validate()

        TextDataValidation(
            Path("artifacts/data_ingestion/text/complaints.csv")
        ).validate()


        TabularFeatureEngineering(
            data_path=Path("artifacts/data_ingestion/tabular/credit_default.xls"),
            output_path=Path("artifacts/feature_engineering/tabular")
        ).transform()
