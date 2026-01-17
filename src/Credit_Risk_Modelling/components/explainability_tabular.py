import joblib
import pandas as pd
from pathlib import Path


class TabularExplainer:
    def __init__(self, model_path: Path):
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "SHAP is required only for explainability. "
                "Install it via `pip install shap` if you need explanations."
            ) from e

        self.shap = shap
        self.model = joblib.load(model_path)
        self.explainer = self.shap.TreeExplainer(self.model)

    def explain(self, X: pd.DataFrame, top_k: int = 5):
        shap_values = self.explainer.shap_values(X)[1]

        mean_abs_shap = (
            pd.DataFrame(shap_values, columns=X.columns)
            .abs()
            .mean()
            .sort_values(ascending=False)
        )

        top_features = mean_abs_shap.head(top_k)

        return [
            {
                "feature": feature,
                "importance": float(value),
            }
            for feature, value in top_features.items()
        ]
