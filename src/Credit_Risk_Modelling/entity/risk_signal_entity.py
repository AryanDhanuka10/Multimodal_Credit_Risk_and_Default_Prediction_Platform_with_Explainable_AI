from dataclasses import dataclass


@dataclass
class RiskSignal:
    name: str
    score: float        # normalized [0,1]
    confidence: float   # reliability of this signal [0,1]
