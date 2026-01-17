import joblib
import logging
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path


class TextRiskModelTrainer:
    def __init__(self, embedding_path: Path, model_path: Path):
        self.embedding_path = embedding_path
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self):
        embeddings = joblib.load(self.embedding_path)

        if embeddings.shape[0] < 10:
            logging.warning("Not enough text samples to train text risk model.")
            return None

        model = KMeans(n_clusters=2, random_state=42)
        model.fit(embeddings)

        joblib.dump(model, self.model_path)
        return model
