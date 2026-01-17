from fastapi import FastAPI, Depends
from pydantic import BaseModel
import pandas as pd

from Credit_Risk_Modelling.api.dependencies import get_inference_engine

app = FastAPI(title="Multimodal Credit Risk API")


class PredictRequest(BaseModel):
    tabular: dict
    timeseries: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(
    payload: PredictRequest,
    inference_engine=Depends(get_inference_engine),
):
    X_tabular = pd.DataFrame([payload.tabular["features"]])
    X_timeseries = pd.DataFrame(payload.timeseries["values"])

    return inference_engine(X_tabular, X_timeseries)
