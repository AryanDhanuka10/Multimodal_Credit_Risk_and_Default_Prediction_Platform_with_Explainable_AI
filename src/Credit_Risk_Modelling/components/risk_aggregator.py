from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal


class RiskAggregator:
    def aggregate(self, signals: list[RiskSignal]) -> dict:
        weighted_sum = 0.0
        total_weight = 0.0

        breakdown = {}

        for s in signals:
            weight = s.confidence
            weighted_sum += s.score * weight
            total_weight += weight
            breakdown[s.name] = {
                "score": s.score,
                "confidence": s.confidence
            }

        final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "final_risk_score": final_risk,
            "breakdown": breakdown
        }
