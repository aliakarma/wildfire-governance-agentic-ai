"""Stage-2 Bayesian confidence update — Eq. (6) from the paper.

Conf^(2)_t = Conf^(1)_t * P(V_t | fire) / P(V_t)

where P(V_t) = P(V_t | fire) * Conf^(1)_t + P(V_t | no fire) * (1 - Conf^(1)_t)

This updates the stage-1 confidence using evidence from a secondary
verification UAV dispatched to the anomaly site.
"""
from __future__ import annotations


class BayesianConfidenceUpdate:
    """Stage-2 Bayesian confidence update g(·) from Eq. (6) in the paper.

    Uses the UAV thermal sensor likelihood model to update the prior
    confidence estimate from cross-modal fusion.

    Args:
        detection_probability: P(V_t | fire) — thermal sensor detection
            probability (paper default: 0.85).
        false_alarm_probability: P(V_t | no fire) — false positive rate
            (paper default: 0.15).
    """

    def __init__(
        self,
        detection_probability: float = 0.85,
        false_alarm_probability: float = 0.15,
    ) -> None:
        if not 0.0 < detection_probability < 1.0:
            raise ValueError(f"detection_probability must be in (0,1); got {detection_probability}")
        if not 0.0 < false_alarm_probability < 1.0:
            raise ValueError(f"false_alarm_probability must be in (0,1); got {false_alarm_probability}")
        self.p_det = detection_probability
        self.p_fa = false_alarm_probability

    def update(
        self,
        stage1_confidence: float,
        verification_positive: bool,
    ) -> float:
        """Compute Conf^(2)_t via Bayesian update (Eq. 6).

        Args:
            stage1_confidence: Conf^(1)_t from cross-modal fusion, in [0, 1].
                Used as the prior probability of fire.
            verification_positive: Whether the verification UAV detected fire.
                True → observation V_t consistent with fire.
                False → observation V_t inconsistent with fire.

        Returns:
            Updated confidence score Conf^(2)_t in (0, 1).

        Raises:
            ValueError: If stage1_confidence is outside (0, 1).
        """
        if not 0.0 < stage1_confidence < 1.0:
            stage1_confidence = max(1e-6, min(stage1_confidence, 1.0 - 1e-6))

        prior_fire = stage1_confidence
        prior_no_fire = 1.0 - prior_fire

        if verification_positive:
            likelihood_fire = self.p_det
            likelihood_no_fire = self.p_fa
        else:
            likelihood_fire = 1.0 - self.p_det
            likelihood_no_fire = 1.0 - self.p_fa

        numerator = likelihood_fire * prior_fire
        denominator = likelihood_fire * prior_fire + likelihood_no_fire * prior_no_fire

        if denominator < 1e-12:
            return stage1_confidence
        return float(numerator / denominator)

    def likelihood_ratio(self, verification_positive: bool) -> float:
        """Compute the Bayes factor (likelihood ratio) for the observation.

        Args:
            verification_positive: Verification UAV detection result.

        Returns:
            P(obs | fire) / P(obs | no fire).
        """
        if verification_positive:
            return self.p_det / self.p_fa
        return (1.0 - self.p_det) / (1.0 - self.p_fa)
