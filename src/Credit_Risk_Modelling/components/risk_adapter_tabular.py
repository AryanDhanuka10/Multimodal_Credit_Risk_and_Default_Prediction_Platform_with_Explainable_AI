import joblib
import numpy as np
from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal
from pathlib import Path


class TabularRiskAdapter:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)

    def predict(self, X):
        prob = self.model.predict_proba(X)[:, 1].mean()

        return RiskSignal(
            name="tabular",
            score=float(prob),
            confidence=0.9
        )
