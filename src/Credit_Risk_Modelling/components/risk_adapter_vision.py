import numpy as np
import joblib
from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal
from pathlib import Path


class VisionRiskAdapter:
    def __init__(self, embedding_path: Path, model_path: Path | None = None):
        self.embeddings = joblib.load(embedding_path)["embeddings"]
        self.model = joblib.load(model_path) if model_path and model_path.exists() else None

    def predict(self):
        if self.model:
            probs = self.model.predict_proba(self.embeddings)[:, 1]
            score = probs.mean()
            confidence = 0.7
        else:
            # proxy risk: embedding variance
            score = float(np.linalg.norm(self.embeddings, axis=1).mean())
            score = min(score / 50.0, 1.0)
            confidence = 0.4

        return RiskSignal(
            name="vision",
            score=score,
            confidence=confidence
        )
