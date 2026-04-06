"""Two-stage confidence scoring pipeline for anomaly verification.

Orchestrates the full two-stage verification process:
  Stage 1: CrossModalFusion  → Conf^(1)_t (Eq. 5)
  Stage 2: BayesianConfidenceUpdate → Conf^(2)_t (Eq. 6)

The pipeline outputs are used directly by the GOMDP governance predicate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wildfire_governance.verification.bayesian_update import BayesianConfidenceUpdate
from wildfire_governance.verification.fusion import CrossModalFusion


@dataclass
class VerificationResult:
    """Full result of two-stage anomaly verification.

    Attributes:
        stage1_confidence: Conf^(1)_t from cross-modal fusion.
        stage2_confidence: Conf^(2)_t after Bayesian update (None if stage 2 not triggered).
        escalated_to_stage2: Whether the stage-2 UAV verification was triggered.
        escalated_to_hitl: Whether the result was escalated to human review.
        final_confidence: stage2_confidence if available, else stage1_confidence.
    """

    stage1_confidence: float
    stage2_confidence: Optional[float]
    escalated_to_stage2: bool
    escalated_to_hitl: bool
    final_confidence: float


class TwoStageConfidenceScorer:
    """Orchestrates the full two-stage verification pipeline.

    Stage 1 computes Conf^(1)_t via weighted linear fusion (Eq. 5).
    If Conf^(1)_t > tau1, a secondary verification UAV is dispatched and
    Stage 2 applies a Bayesian update (Eq. 6) to obtain Conf^(2)_t.
    If Conf^(2)_t > tau2, the event is escalated to human review (HITL).

    Args:
        fusion: CrossModalFusion instance (Eq. 5).
        bayesian_update: BayesianConfidenceUpdate instance (Eq. 6).
        tau1: Stage-1 escalation threshold (paper default: 0.60).
        tau2: Stage-2 HITL escalation threshold (paper default: 0.80).
    """

    def __init__(
        self,
        fusion: Optional[CrossModalFusion] = None,
        bayesian_update: Optional[BayesianConfidenceUpdate] = None,
        tau1: float = 0.60,
        tau2: float = 0.80,
    ) -> None:
        if tau2 <= tau1:
            raise ValueError(f"tau2={tau2} must be > tau1={tau1}")
        self._fusion = fusion or CrossModalFusion()
        self._bayesian = bayesian_update or BayesianConfidenceUpdate()
        self.tau1 = tau1
        self.tau2 = tau2

    def score(
        self,
        heat_anomaly_index: float,
        weather_index: float,
        verification_positive: Optional[bool] = None,
    ) -> VerificationResult:
        """Run the full two-stage verification pipeline on a single anomaly.

        Args:
            heat_anomaly_index: Normalised heat anomaly hat_H_t in [0, 1].
            weather_index: Normalised weather index hat_W_t in [0, 1].
            verification_positive: Stage-2 UAV thermal observation result.
                - None: stage 2 has not been run yet (call will skip it).
                - True/False: UAV observed/did not observe fire at the site.

        Returns:
            VerificationResult with all confidence scores and escalation flags.
        """
        conf1 = self._fusion.compute_stage1_confidence(
            heat_anomaly_index, weather_index
        )
        escalated_to_stage2 = conf1 > self.tau1

        conf2: Optional[float] = None
        if escalated_to_stage2 and verification_positive is not None:
            conf2 = self._bayesian.update(conf1, verification_positive)

        final = conf2 if conf2 is not None else conf1
        escalated_to_hitl = final > self.tau2

        return VerificationResult(
            stage1_confidence=conf1,
            stage2_confidence=conf2,
            escalated_to_stage2=escalated_to_stage2,
            escalated_to_hitl=escalated_to_hitl,
            final_confidence=final,
        )

    def update_thresholds(self, tau1: float, tau2: float) -> None:
        """Update thresholds (called by the online learning module).

        Args:
            tau1: New stage-1 threshold. Must be < tau2.
            tau2: New stage-2 threshold.

        Raises:
            ValueError: If tau2 <= tau1.
        """
        if tau2 <= tau1:
            raise ValueError(f"tau2={tau2} must be > tau1={tau1}")
        self.tau1 = tau1
        self.tau2 = tau2
