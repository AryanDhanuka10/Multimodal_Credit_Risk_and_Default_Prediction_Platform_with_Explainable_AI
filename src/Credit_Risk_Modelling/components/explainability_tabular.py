import shap
import joblib
import pandas as pd
from pathlib import Path


class TabularExplainer:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X: pd.DataFrame):
        shap_values = self.explainer.shap_values(X)
        return shap_values
