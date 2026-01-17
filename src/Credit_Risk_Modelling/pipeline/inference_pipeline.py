"""
Mock inference pipeline for production deployment.
Works without trained models using heuristic-based risk scoring.
"""

import pandas as pd
import numpy as np
from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal


def run_inference(X_tabular, X_timeseries, **adapters):
    """
    Run multimodal risk inference using heuristic scoring.
    No trained models required - perfect for demo/MVP.
    """
    
    signals = []
    
    # ============================================
    # 1. TABULAR RISK SCORING (Heuristic)
    # ============================================
    try:
        tabular_score = score_tabular_heuristic(X_tabular)
        signals.append(RiskSignal(
            name="tabular",
            score=tabular_score,
            confidence=0.85
        ))
    except Exception as e:
        print(f"Tabular scoring error: {e}")
        signals.append(RiskSignal(name="tabular", score=0.5, confidence=0.5))
    
    # ============================================
    # 2. TIME-SERIES RISK SCORING (Heuristic)
    # ============================================
    try:
        timeseries_score = score_timeseries_heuristic(X_timeseries)
        signals.append(RiskSignal(
            name="timeseries",
            score=timeseries_score,
            confidence=0.80
        ))
    except Exception as e:
        print(f"Time-series scoring error: {e}")
        signals.append(RiskSignal(name="timeseries", score=0.5, confidence=0.5))
    
    # ============================================
    # 3. VISION RISK SCORING (Mock)
    # ============================================
    # In production, use document embeddings
    vision_score = 0.3 + np.random.uniform(-0.1, 0.2)
    signals.append(RiskSignal(
        name="vision",
        score=np.clip(vision_score, 0, 1),
        confidence=0.65
    ))
    
    # ============================================
    # 4. NLP RISK SCORING (Mock)
    # ============================================
    # In production, use transformer embeddings + topic modeling
    text_score = 0.25 + np.random.uniform(-0.05, 0.15)
    signals.append(RiskSignal(
        name="text",
        score=np.clip(text_score, 0, 1),
        confidence=0.60
    ))
    
    # ============================================
    # 5. AGGREGATE SCORES
    # ============================================
    result = aggregate_signals(signals)
    return result


def score_tabular_heuristic(X_tabular):
    """
    Heuristic tabular risk scoring based on credit features.
    
    Features expected:
    - f0: credit_limit (normalized 0-1)
    - f1: monthly_income (normalized 0-1)
    - f2: monthly_bill (normalized 0-1)
    - f3: age (normalized 0-1)
    - f4: outstanding_balance (normalized 0-1)
    """
    try:
        # Extract features
        features = X_tabular.iloc[0] if len(X_tabular) > 0 else {}
        
        income = features.get('f1', 0.5) if hasattr(features, 'get') else 0.5
        bill = features.get('f2', 0.5) if hasattr(features, 'get') else 0.5
        balance = features.get('f4', 0.5) if hasattr(features, 'get') else 0.5
        
        # Risk scoring logic:
        # - High bill-to-income ratio = higher risk
        # - High outstanding balance = higher risk
        # - Low income = higher risk
        
        bill_to_income_ratio = bill / (income + 0.001)  # Avoid division by zero
        
        # Calculate risk components
        income_risk = 0.3 * (1 - income)  # Low income = high risk
        bill_risk = 0.4 * min(bill_to_income_ratio, 1.0)  # High bill/income = high risk
        balance_risk = 0.3 * balance  # High balance = higher risk
        
        # Combine
        risk_score = income_risk + bill_risk + balance_risk
        
        return float(np.clip(risk_score, 0, 1))
        
    except Exception as e:
        print(f"Tabular scoring error: {e}")
        return 0.5


def score_timeseries_heuristic(X_timeseries):
    """
    Heuristic time-series risk scoring based on transaction patterns.
    
    Features expected:
    - Transaction values over time (normalized 0-1)
    
    Risk indicators:
    - High volatility = higher risk (unpredictable spending)
    - Increasing trend = higher risk (unsustainable)
    """
    try:
        # Extract transaction values
        if isinstance(X_timeseries, pd.DataFrame):
            values = X_timeseries.iloc[0].values if len(X_timeseries) > 0 else [0.5]
        elif isinstance(X_timeseries, list):
            values = X_timeseries[0] if len(X_timeseries) > 0 else [0.5]
        else:
            values = [0.5]
        
        values = np.array(values, dtype=float)
        
        # Calculate volatility (standard deviation)
        volatility = float(np.std(values)) if len(values) > 1 else 0
        
        # Calculate trend (are transactions increasing?)
        if len(values) > 1:
            trend = float(np.polyfit(range(len(values)), values, 1)[0])
        else:
            trend = 0
        
        # Calculate average transaction level
        avg_level = float(np.mean(values))
        
        # Risk scoring
        volatility_risk = 0.4 * min(volatility * 2, 1.0)  # High volatility = risky
        trend_risk = 0.3 * max(trend, 0)  # Increasing transactions = risky
        level_risk = 0.3 * avg_level  # High spending = risky
        
        risk_score = volatility_risk + trend_risk + level_risk
        
        return float(np.clip(risk_score, 0, 1))
        
    except Exception as e:
        print(f"Time-series scoring error: {e}")
        return 0.5


def aggregate_signals(signals):
    """
    Aggregate multimodal risk signals using confidence-weighted fusion.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    
    breakdown = {}
    
    # First pass: compute weighted contributions
    for signal in signals:
        weighted_contribution = signal.score * signal.confidence
        weighted_sum += weighted_contribution
        total_weight += signal.confidence
        
        breakdown[signal.name] = {
            "score": signal.score,
            "confidence": signal.confidence,
            "weighted_contribution": weighted_contribution,
        }
    
    # Calculate final risk score
    final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    # Second pass: normalize contributions to percentages
    for key in breakdown:
        breakdown[key]["percent_contribution"] = (
            breakdown[key]["weighted_contribution"] / weighted_sum
            if weighted_sum > 0
            else 0.0
        )
    
    return {
        "final_risk_score": float(np.clip(final_risk, 0, 1)),
        "breakdown": breakdown,
    }


def run_explained_inference(X_tabular, X_timeseries, **adapters):
    """
    Run inference with explainability.
    """
    result = run_inference(X_tabular, X_timeseries, **adapters)
    
    # Add top features explanation
    result["explanations"] = {
        "tabular_top_features": [
            {"feature": "income_level", "importance": 0.35},
            {"feature": "bill_to_income_ratio", "importance": 0.40},
            {"feature": "outstanding_balance", "importance": 0.25},
        ]
    }
    
    return result