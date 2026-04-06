"""Verification and confidence agent — executes multi-stage anomaly validation."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from wildfire_governance.verification.confidence_scorer import (
    TwoStageConfidenceScorer,
    VerificationResult,
)


class VerificationAndConfidenceAgent:
    """Runs the two-stage probabilistic verification pipeline on anomalies.

    Dispatches secondary UAV verification when stage-1 confidence exceeds
    tau1, then escalates to HITL when stage-2 confidence exceeds tau2.

    Args:
        scorer: TwoStageConfidenceScorer instance (Eqs. 5–6).
    """

    def __init__(self, scorer: Optional[TwoStageConfidenceScorer] = None) -> None:
        self._scorer = scorer or TwoStageConfidenceScorer()
        self._n_stage1_triggers: int = 0
        self._n_stage2_triggers: int = 0
        self._n_hitl_escalations: int = 0

    def verify_anomaly(
        self,
        heat_anomaly_index: float,
        weather_index: float,
        verification_positive: Optional[bool] = None,
    ) -> VerificationResult:
        """Run full two-stage verification on an anomaly.

        Args:
            heat_anomaly_index: Normalised heat anomaly in [0, 1].
            weather_index: Normalised weather index in [0, 1].
            verification_positive: UAV thermal observation result (None = no stage 2 yet).

        Returns:
            VerificationResult with all confidence scores and escalation flags.
        """
        result = self._scorer.score(heat_anomaly_index, weather_index, verification_positive)
        self._n_stage1_triggers += 1
        if result.escalated_to_stage2:
            self._n_stage2_triggers += 1
        if result.escalated_to_hitl:
            self._n_hitl_escalations += 1
        return result

    @property
    def stage1_trigger_count(self) -> int:
        """Total stage-1 triggers since last reset."""
        return self._n_stage1_triggers

    @property
    def hitl_escalation_count(self) -> int:
        """Total HITL escalations since last reset."""
        return self._n_hitl_escalations

    def reset_stats(self) -> None:
        """Reset per-episode counters."""
        self._n_stage1_triggers = 0
        self._n_stage2_triggers = 0
        self._n_hitl_escalations = 0
