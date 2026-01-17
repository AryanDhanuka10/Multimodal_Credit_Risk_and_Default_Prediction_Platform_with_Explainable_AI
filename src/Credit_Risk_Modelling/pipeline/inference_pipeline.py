from Credit_Risk_Modelling.components.risk_aggregator import RiskAggregator
from Credit_Risk_Modelling.components.risk_adapter_tabular import TabularRiskAdapter
from Credit_Risk_Modelling.components.risk_adapter_timeseries import TimeSeriesRiskAdapter
from Credit_Risk_Modelling.components.risk_adapter_vision import VisionRiskAdapter
from Credit_Risk_Modelling.components.risk_adapter_text import TextRiskAdapter
from pathlib import Path


def run_inference(
    X_tabular,
    X_timeseries,
    tabular_adapter=None,
    timeseries_adapter=None,
    vision_adapter=None,
    text_adapter=None,
):
    """
    Run multimodal risk inference.

    Adapters can be injected for testing.
    If not provided, default production adapters are used.
    """

    signals = []

    # TABULAR
    if tabular_adapter is None:
        tabular_adapter = TabularRiskAdapter(
            Path("artifacts/training/tabular/lightgbm.pkl")
        )
    signals.append(tabular_adapter.predict(X_tabular))

    # TIME-SERIES
    if timeseries_adapter is None:
        timeseries_adapter = TimeSeriesRiskAdapter(
            Path("artifacts/training/timeseries/lightgbm.pkl")
        )
    signals.append(timeseries_adapter.predict(X_timeseries))

    # VISION
    if vision_adapter is None:
        vision_adapter = VisionRiskAdapter(
            embedding_path=Path(
                "artifacts/feature_engineering/documents/document_embeddings.pkl"
            ),
            model_path=Path(
                "artifacts/training/documents/document_risk_model.pkl"
            ),
        )
    signals.append(vision_adapter.predict())

    # TEXT / NLP
    if text_adapter is None:
        text_adapter = TextRiskAdapter(
            topic_path=Path(
                "artifacts/feature_engineering/text/text_topics.pkl"
            ),
            risk_map_path=Path(
                "artifacts/feature_engineering/text/text_topic_risk_map.pkl"
            ),
        )
    signals.append(text_adapter.predict())

    result = RiskAggregator().aggregate(signals)
    return result
