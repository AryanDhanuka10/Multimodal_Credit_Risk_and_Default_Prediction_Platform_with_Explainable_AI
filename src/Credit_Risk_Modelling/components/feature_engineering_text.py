import pandas as pd
import torch
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import joblib


class TextFeatureEngineering:
    def __init__(
        self,
        data_path: Path,
        text_column: str,
        output_dir: Path,
        max_length: int = 256
    ):
        self.data_path = data_path
        self.text_column = text_column
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to(self.device)

        self.max_length = max_length

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def transform(self):
        logging.info("Starting text feature engineering")

        df = pd.read_csv(self.data_path)

        if self.text_column not in df.columns:
            logging.warning("Text column not found. Skipping NLP feature engineering.")
            return None

        df = df.dropna(subset=[self.text_column])

        if df.empty:
            logging.warning("No usable text samples. Skipping NLP.")
            return None

        texts = df[self.text_column].astype(str).tolist()

        embeddings = []

        for text in texts[:5000]:  # cap for compute sanity
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded)

            sentence_embedding = self._mean_pooling(
                model_output, encoded["attention_mask"]
            )

            embeddings.append(sentence_embedding.squeeze().cpu().numpy())

        embeddings = np.vstack(embeddings)

        joblib.dump(
            embeddings,
            self.output_dir / "text_embeddings.pkl"
        )

        logging.info(f"Saved {embeddings.shape[0]} text embeddings")

        return embeddings
