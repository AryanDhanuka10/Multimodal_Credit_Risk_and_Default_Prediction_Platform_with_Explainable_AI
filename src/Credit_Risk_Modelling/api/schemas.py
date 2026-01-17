from pydantic import BaseModel
from typing import Optional, Dict, Any, List


class TabularInput(BaseModel):
    features: Dict[str, float]


class TimeSeriesInput(BaseModel):
    values: List[List[float]]  # shape: [T, F]


class InferenceRequest(BaseModel):
    tabular: TabularInput
    timeseries: TimeSeriesInput
    use_vision: Optional[bool] = True
    use_text: Optional[bool] = True


class InferenceResponse(BaseModel):
    final_risk_score: float
    breakdown: Dict[str, Any]
    explanations: Optional[Dict[str, Any]] = None
