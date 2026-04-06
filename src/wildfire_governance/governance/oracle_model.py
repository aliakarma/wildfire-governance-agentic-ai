"""Human operator oracle model — simulates HITL authorisation decisions.

Models the human review process as a stochastic oracle with:
- Response delay drawn from N(mean=3.0, std=0.8) timesteps (paper Section VI-A).
- Approval probability based on confidence score (higher confidence → more likely approved).
- Configurable rejection rate for oracle risk analysis.

The oracle is intentionally stylised; real operator response times depend on
cognitive load, fatigue, and workload. See Section VII (Discussion) in the paper
for the limitation discussion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OracleDecision:
    """Result of a simulated human operator review.

    Attributes:
        approved: True if the operator authorised the alert.
        review_delay_steps: Simulated review time in simulation timesteps.
        confidence_seen: Confidence score presented to the operator.
        reason: String explaining the decision ("approved", "rejected_low_conf",
                "rejected_oracle_risk").
    """

    approved: bool
    review_delay_steps: float
    confidence_seen: float
    reason: str


class HumanOperatorOracle:
    """Stochastic model of a human validator reviewing anomaly alerts.

    Args:
        mean_review_delay: Mean human response time (steps, paper default: 3.0).
        std_review_delay: Std deviation of response time (steps, paper default: 0.8).
        rejection_rate: Base probability of rejecting even a high-confidence alert
                        (oracle risk, paper default: 0.0).
        approval_threshold: Minimum confidence for operator to approve (default: 0.75).
        rng: Seeded NumPy Generator for reproducible simulation.
    """

    def __init__(
        self,
        mean_review_delay: float = 3.0,
        std_review_delay: float = 0.8,
        rejection_rate: float = 0.0,
        approval_threshold: float = 0.75,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._mean_delay = mean_review_delay
        self._std_delay = std_review_delay
        self._rejection_rate = rejection_rate
        self._approval_threshold = approval_threshold
        self._rng = rng or np.random.default_rng(42)
        self._n_approved: int = 0
        self._n_rejected: int = 0

    def review(self, confidence: float) -> OracleDecision:
        """Simulate a human operator reviewing an anomaly alert.

        The operator approves if:
        (1) confidence >= approval_threshold, AND
        (2) a random draw does not trigger the oracle rejection rate.

        Args:
            confidence: Final Conf^(2)_t score presented to the operator.

        Returns:
            OracleDecision with approval, delay, and reason.
        """
        delay = float(self._rng.normal(self._mean_delay, self._std_delay))
        delay = max(0.5, delay)

        if confidence < self._approval_threshold:
            self._n_rejected += 1
            return OracleDecision(
                approved=False,
                review_delay_steps=delay,
                confidence_seen=confidence,
                reason="rejected_low_conf",
            )

        # Oracle risk: operator may reject even high-confidence alerts
        if self._rejection_rate > 0 and self._rng.random() < self._rejection_rate:
            self._n_rejected += 1
            return OracleDecision(
                approved=False,
                review_delay_steps=delay,
                confidence_seen=confidence,
                reason="rejected_oracle_risk",
            )

        self._n_approved += 1
        return OracleDecision(
            approved=True,
            review_delay_steps=delay,
            confidence_seen=confidence,
            reason="approved",
        )

    @property
    def approval_rate(self) -> float:
        """Fraction of reviews that resulted in approval."""
        total = self._n_approved + self._n_rejected
        return self._n_approved / total if total > 0 else 1.0

    def reset(self) -> None:
        """Reset per-episode counters."""
        self._n_approved = 0
        self._n_rejected = 0
