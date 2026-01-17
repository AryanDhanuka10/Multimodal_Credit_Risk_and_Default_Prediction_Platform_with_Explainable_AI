from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from Credit_Risk_Modelling.pipeline.inference_pipeline import run_inference

app = FastAPI(title="Multimodal Credit Risk API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    tabular: dict
    timeseries: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest):
    """
    Predict credit risk using multimodal heuristic scoring.
    No trained models required.
    """
    try:
        # Convert to DataFrames
        X_tabular = pd.DataFrame([payload.tabular["features"]])
        X_timeseries = pd.DataFrame(payload.timeseries["values"])
        
        # Run inference
        result = run_inference(X_tabular, X_timeseries)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "final_risk_score": 0.5,
            "breakdown": {
                "tabular": {"score": 0.5, "confidence": 0.5, "percent_contribution": 0.25},
                "timeseries": {"score": 0.5, "confidence": 0.5, "percent_contribution": 0.25},
                "vision": {"score": 0.5, "confidence": 0.5, "percent_contribution": 0.25},
                "text": {"score": 0.5, "confidence": 0.5, "percent_contribution": 0.25}
            }
        }