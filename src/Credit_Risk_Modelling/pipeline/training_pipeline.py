import logging
from pathlib import Path

from Credit_Risk_Modelling.config.configuration import ConfigurationManager

# Ingestion
from Credit_Risk_Modelling.components.data_ingestion_tabular import TabularDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_timeseries import TimeSeriesDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_documents import DocumentDataIngestion
from Credit_Risk_Modelling.components.data_ingestion_text import TextDataIngestion
from Credit_Risk_Modelling.entity.data_ingestion_entity import DataIngestionConfig

# Validation
from Credit_Risk_Modelling.components.data_validation_tabular import TabularDataValidation
from Credit_Risk_Modelling.components.data_validation_timeseries import TimeSeriesDataValidation
from Credit_Risk_Modelling.components.data_validation_documents import DocumentDataValidation
from Credit_Risk_Modelling.components.data_validation_text import TextDataValidation
from Credit_Risk_Modelling.entity.data_validation_entity import DataValidationConfig

# Feature Engineering
from Credit_Risk_Modelling.components.feature_engineering_tabular import TabularFeatureEngineering
from Credit_Risk_Modelling.components.feature_engineering_timeseries import TimeSeriesFeatureEngineering
from Credit_Risk_Modelling.entity.feature_engineering_entity import TimeSeriesFeatureConfig
from Credit_Risk_Modelling.components.feature_engineering_documents import DocumentFeatureEngineering
from Credit_Risk_Modelling.components.feature_engineering_text import TextFeatureEngineering



# Training
from Credit_Risk_Modelling.components.model_trainer_timeseries import TimeSeriesModelTrainer
from Credit_Risk_Modelling.components.model_trainer_documents import DocumentRiskModelTrainer
from Credit_Risk_Modelling.components.model_trainer_text import TextRiskModelTrainer
from Credit_Risk_Modelling.components.topic_modeling_text import TextTopicModeler
from Credit_Risk_Modelling.utils.text_topic_risk_mapping import map_topics_to_risk





logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(levelname)s: %(message)s"
)


class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager(Path("config/config.yaml"))
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()

    # STAGE 1: DATA INGESTION
    def run_data_ingestion(self):
        logging.info("Starting data ingestion stage")

        di = self.data_ingestion_config

        TabularDataIngestion(
            DataIngestionConfig(
                root_dir=Path(di.tabular.root_dir),
                source_url=di.tabular.source_url,
                local_path=Path(di.tabular.local_file),
            )
        ).ingest()

        TimeSeriesDataIngestion(
            DataIngestionConfig(
                root_dir=Path(di.timeseries.root_dir),
                source_url=di.timeseries.source_url,
                local_path=Path(di.timeseries.local_file),
            )
        ).ingest()

        DocumentDataIngestion(
            Path(di.documents.local_dir)
        ).ingest()

        TextDataIngestion(
            Path(di.text.local_file)
        ).ingest()

        logging.info("Data ingestion stage completed")

    # STAGE 2: DATA VALIDATION
    def run_data_validation(self):
        logging.info("Starting data validation stage")

        di = self.data_ingestion_config

        TabularDataValidation(
            DataValidationConfig(
                required_columns=["default_payment_next_month"],
                data_path=Path(di.tabular.local_file),
            )
        ).validate()

        TimeSeriesDataValidation(
            Path(di.timeseries.local_file)
        ).validate()

        DocumentDataValidation(
            Path(di.documents.local_dir)
        ).validate()

        TextDataValidation(
            Path(di.text.local_file)
        ).validate()

        logging.info("Data validation stage completed")

    # STAGE 3: FEATURE ENGINEERING
    def run_feature_engineering(self):
        logging.info("Starting feature engineering stage")

        di = self.data_ingestion_config

        # ---- Tabular ----
        tabular_fe_path = Path("artifacts/feature_engineering/tabular")
        tabular_fe_path.mkdir(parents=True, exist_ok=True)

        TabularFeatureEngineering(
            data_path=Path(di.tabular.local_file),
            output_path=tabular_fe_path,
        ).transform()

        # ---- Time-Series ----
        ts_fe_path = Path("artifacts/feature_engineering/timeseries")
        ts_fe_path.mkdir(parents=True, exist_ok=True)

        TimeSeriesFeatureEngineering(
            TimeSeriesFeatureConfig(
                data_path=Path(di.timeseries.local_file),
                output_path=ts_fe_path,
                window_size=5,
            )
        ).transform()

        logging.info("Feature engineering stage completed")

    # STAGE 4: MODEL TRAINING
    def run_model_training(self):
        logging.info("Starting model training stage")

        ts_model_path = Path("artifacts/training/timeseries")
        ts_model_path.mkdir(parents=True, exist_ok=True)

        TimeSeriesModelTrainer(
            data_path=Path("artifacts/feature_engineering/timeseries/timeseries_features.csv"),
            model_path=Path("artifacts/training/timeseries/lightgbm.pkl"),
            target_col="default_flag"
        ).train()


        logging.info("Model training stage completed")

    def run_document_pipeline(self):
        logging.info("Starting document vision pipeline")

        image_dir = Path("artifacts/data_ingestion/documents/images")
        fe_output = Path("artifacts/feature_engineering/documents")

        fe = DocumentFeatureEngineering(image_dir, fe_output)
        fe.extract_embeddings()

        trainer = DocumentRiskModelTrainer(
            embedding_path=fe_output / "document_embeddings.pkl",
            model_path=Path("artifacts/training/documents/document_risk_model.pkl")
        )
        trainer.train()

        logging.info("Document vision pipeline completed")

    def run_text_pipeline(self):
        logging.info("Starting NLP text pipeline")

        fe = TextFeatureEngineering(
            data_path=Path("artifacts/data_ingestion/text/complaints.csv"),
            text_column="Consumer complaint narrative",
            output_dir=Path("artifacts/feature_engineering/text")
        )

        embeddings = fe.transform()
        if embeddings is None:
            logging.warning("Text pipeline skipped")
            return

        topic_modeler = TextTopicModeler(
            embedding_path=Path("artifacts/feature_engineering/text/text_embeddings.pkl"),
            output_dir=Path("artifacts/feature_engineering/text"),
            n_topics=10
        )

        topics = topic_modeler.fit()
        if topics is None:
            return

        risk_map = map_topics_to_risk(topics)

        # Save for downstream aggregation
        import joblib
        joblib.dump(
            risk_map,
            Path("artifacts/feature_engineering/text/text_topic_risk_map.pkl")
        )

        logging.info("Text topic modeling and risk mapping completed")



    # FULL PIPELINE
    def run_pipeline(self):
        self.run_data_ingestion()
        self.run_data_validation()
        self.run_feature_engineering()
        self.run_model_training()
        self.run_document_pipeline()
        self.run_text_pipeline()




# ENTRY POINT
if __name__ == "__main__":
    TrainingPipeline().run_pipeline()
