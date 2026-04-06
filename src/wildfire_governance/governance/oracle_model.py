"""Human operator oracle model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class OracleDecision:
    approved: bool; review_delay_steps: float; confidence_seen: float; reason: str

class HumanOperatorOracle:
    def __init__(self, mean_review_delay=3.0, std_review_delay=0.8,
                 rejection_rate=0.0, approval_threshold=0.75, rng=None):
        self._mean_delay = mean_review_delay; self._std_delay = std_review_delay
        self._rejection_rate = rejection_rate; self._approval_threshold = approval_threshold
        self._rng = rng or np.random.default_rng(42)
        self._n_approved = 0; self._n_rejected = 0

    def review(self, confidence):
        delay = max(0.5, float(self._rng.normal(self._mean_delay, self._std_delay)))
        if confidence < self._approval_threshold:
            self._n_rejected += 1
            return OracleDecision(False, delay, confidence, "rejected_low_conf")
        if self._rejection_rate > 0 and self._rng.random() < self._rejection_rate:
            self._n_rejected += 1
            return OracleDecision(False, delay, confidence, "rejected_oracle_risk")
        self._n_approved += 1
        return OracleDecision(True, delay, confidence, "approved")

    @property
    def approval_rate(self):
        total = self._n_approved + self._n_rejected
        return self._n_approved / total if total > 0 else 1.0
    def reset(self): self._n_approved = 0; self._n_rejected = 0
