"""Stage-2 Bayesian confidence update (Eq. 6)."""
from __future__ import annotations

class BayesianConfidenceUpdate:
    def __init__(self, detection_probability=0.85, false_alarm_probability=0.15):
        if not 0.0 < detection_probability < 1.0:
            raise ValueError(f"detection_probability must be in (0,1)")
        if not 0.0 < false_alarm_probability < 1.0:
            raise ValueError(f"false_alarm_probability must be in (0,1)")
        self.p_det = detection_probability; self.p_fa = false_alarm_probability

    def update(self, stage1_confidence, verification_positive):
        p = max(1e-6, min(stage1_confidence, 1.0-1e-6))
        lf = self.p_det if verification_positive else (1.0 - self.p_det)
        lnf = self.p_fa if verification_positive else (1.0 - self.p_fa)
        denom = lf*p + lnf*(1.0-p)
        return float(lf*p/denom) if denom > 1e-12 else p

    def likelihood_ratio(self, verification_positive):
        return (self.p_det/self.p_fa) if verification_positive else ((1-self.p_det)/(1-self.p_fa))
