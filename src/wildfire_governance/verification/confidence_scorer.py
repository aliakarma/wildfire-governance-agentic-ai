"""Two-stage confidence scoring pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from wildfire_governance.verification.bayesian_update import BayesianConfidenceUpdate
from wildfire_governance.verification.fusion import CrossModalFusion

@dataclass
class VerificationResult:
    stage1_confidence: float; stage2_confidence: Optional[float]
    escalated_to_stage2: bool; escalated_to_hitl: bool; final_confidence: float

class TwoStageConfidenceScorer:
    def __init__(self, fusion=None, bayesian_update=None, tau1=0.60, tau2=0.80):
        if tau2 <= tau1: raise ValueError(f"tau2={tau2} must be > tau1={tau1}")
        self._fusion = fusion or CrossModalFusion()
        self._bayesian = bayesian_update or BayesianConfidenceUpdate()
        self.tau1 = tau1; self.tau2 = tau2

    def score(self, heat_anomaly_index, weather_index, verification_positive=None):
        conf1 = self._fusion.compute_stage1_confidence(heat_anomaly_index, weather_index)
        escalated_to_stage2 = conf1 > self.tau1
        conf2 = self._bayesian.update(conf1, verification_positive)                 if escalated_to_stage2 and verification_positive is not None else None
        final = conf2 if conf2 is not None else conf1
        return VerificationResult(conf1, conf2, escalated_to_stage2, final > self.tau2, final)

    def update_thresholds(self, tau1, tau2):
        if tau2 <= tau1: raise ValueError(f"tau2={tau2} must be > tau1={tau1}")
        self.tau1 = tau1; self.tau2 = tau2
