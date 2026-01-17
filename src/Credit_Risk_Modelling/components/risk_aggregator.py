from Credit_Risk_Modelling.entity.risk_signal_entity import RiskSignal


class RiskAggregator:
    def aggregate(self, signals: list[RiskSignal]) -> dict:
        weighted_sum = 0.0
        total_weight = 0.0

        breakdown = {}

        # First pass: compute weighted contributions
        for s in signals:
            weighted_contribution = s.score * s.confidence
            weighted_sum += weighted_contribution
            total_weight += s.confidence

            breakdown[s.name] = {
                "score": s.score,
                "confidence": s.confidence,
                "weighted_contribution": weighted_contribution,
            }

        final_risk = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Second pass: normalize contributions
        for k in breakdown:
            breakdown[k]["percent_contribution"] = (
                breakdown[k]["weighted_contribution"] / weighted_sum
                if weighted_sum > 0
                else 0.0
            )

        return {
            "final_risk_score": final_risk,
            "breakdown": breakdown,
        }
