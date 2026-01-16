import joblib
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from pathlib import Path

class DocumentRiskModelTrainer:
    def __init__(self, embedding_path: Path, model_path: Path):
        self.embedding_path = embedding_path
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        data = joblib.load(self.embedding_path)
        X = data["embeddings"]
        y = data["labels"]

        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            logging.warning(
                "Document risk model skipped: only one class present "
                f"({unique_classes}). Embeddings will be used without classifier."
            )
            return None

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        joblib.dump(model, self.model_path)
        return model
