import joblib
import logging
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans


class TextTopicModeler:
    def __init__(
        self,
        embedding_path: Path,
        output_dir: Path,
        n_topics: int = 10
    ):
        self.embedding_path = embedding_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_topics = n_topics

    def fit(self):
        logging.info("Starting topic modeling on text embeddings")

        embeddings = joblib.load(self.embedding_path)

        if embeddings.shape[0] < self.n_topics:
            logging.warning("Not enough samples for topic modeling")
            return None

        model = KMeans(
            n_clusters=self.n_topics,
            random_state=42,
            n_init=10
        )

        topics = model.fit_predict(embeddings)

        joblib.dump(
            {
                "model": model,
                "topics": topics
            },
            self.output_dir / "text_topics.pkl"
        )

        logging.info(f"Generated {self.n_topics} text topics")
        return topics
