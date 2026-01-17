import joblib
import numpy as np
from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal
from pathlib import Path


class TextRiskAdapter:
    def __init__(self, topic_path: Path, risk_map_path: Path):
        data = joblib.load(topic_path)
        self.topics = data["topics"]
        self.risk_map = joblib.load(risk_map_path)

    def predict(self):
        scores = [self.risk_map[t] for t in self.topics]
        score = float(np.mean(scores))

        return RiskSignal(
            name="text",
            score=score,
            confidence=0.6
        )
