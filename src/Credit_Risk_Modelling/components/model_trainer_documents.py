import joblib
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

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        joblib.dump(model, self.model_path)
        return model
