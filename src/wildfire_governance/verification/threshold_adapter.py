"""Online EMA-based threshold adaptation (learning module).

Adjusts tau1 and tau2 between episodes using exponential moving averages
of precision-recall measurements to track environmental drift without
full retraining.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThresholdHistory:
    """Record of threshold adaptation over episodes.

    Attributes:
        tau1_history: List of tau1 values per episode.
        tau2_history: List of tau2 values per episode.
        precision_history: List of episode precision values.
        recall_history: List of episode recall values.
    """

    tau1_history: list
    tau2_history: list
    precision_history: list
    recall_history: list


class OnlineThresholdAdapter:
    """EMA-based online adaptation of confidence thresholds tau1 and tau2.

    Implements the learning module described in Section III-A of the paper.
    Thresholds are updated between episodes to maintain a target F1-score.
    This keeps the verification pipeline calibrated as environmental
    conditions (fire frequency, sensor noise) change over deployment.

    The adaptation formula:
        tau_new = (1 - alpha_ema) * tau_old + alpha_ema * target_tau(precision, recall)

    Args:
        tau1_init: Initial stage-1 threshold (paper default: 0.60).
        tau2_init: Initial stage-2 threshold (paper default: 0.80).
        alpha_ema: EMA learning rate (paper default: 0.10).
        target_f1: F1-score target that drives threshold adjustment.
        tau1_min: Minimum allowed tau1.
        tau2_max: Maximum allowed tau2.
    """

    def __init__(
        self,
        tau1_init: float = 0.60,
        tau2_init: float = 0.80,
        alpha_ema: float = 0.10,
        target_f1: float = 0.85,
        tau1_min: float = 0.40,
        tau2_max: float = 0.95,
    ) -> None:
        self._tau1 = tau1_init
        self._tau2 = tau2_init
        self._alpha = alpha_ema
        self._target_f1 = target_f1
        self._tau1_min = tau1_min
        self._tau2_max = tau2_max
        self._history = ThresholdHistory([], [], [], [])

    def update(
        self,
        episode_precision: float,
        episode_recall: float,
    ) -> tuple[float, float]:
        """Update thresholds based on episode-level precision and recall.

        If precision is low (too many false alerts), increase tau1/tau2.
        If recall is low (too many missed detections), decrease tau1/tau2.

        Args:
            episode_precision: Fraction of alerts that were true fires in [0, 1].
            episode_recall: Fraction of fires that were detected in [0, 1].

        Returns:
            Tuple (tau1_new, tau2_new).
        """
        self._history.precision_history.append(episode_precision)
        self._history.recall_history.append(episode_recall)

        f1 = _compute_f1(episode_precision, episode_recall)
        gap = self._target_f1 - f1

        # Positive gap: below target → lower thresholds (more sensitive)
        # Negative gap: above target → raise thresholds (more specific)
        delta = -gap * 0.1  # Scale adjustment
        target_tau1 = float(min(max(self._tau1 + delta, self._tau1_min), self._tau2 - 0.05))
        target_tau2 = float(min(max(self._tau2 + delta, target_tau1 + 0.05), self._tau2_max))

        self._tau1 = (1.0 - self._alpha) * self._tau1 + self._alpha * target_tau1
        self._tau2 = (1.0 - self._alpha) * self._tau2 + self._alpha * target_tau2

        # Enforce ordering constraint
        if self._tau1 >= self._tau2:
            self._tau1 = self._tau2 - 0.05

        self._history.tau1_history.append(self._tau1)
        self._history.tau2_history.append(self._tau2)
        return self._tau1, self._tau2

    def get_thresholds(self) -> tuple[float, float]:
        """Return current (tau1, tau2).

        Returns:
            Tuple (tau1, tau2) with tau1 < tau2.
        """
        return self._tau1, self._tau2

    def get_history(self) -> ThresholdHistory:
        """Return the full adaptation history."""
        return self._history

    def reset(self, tau1: float = 0.60, tau2: float = 0.80) -> None:
        """Reset thresholds and history to initial values.

        Args:
            tau1: Initial tau1.
            tau2: Initial tau2.
        """
        self._tau1 = tau1
        self._tau2 = tau2
        self._history = ThresholdHistory([], [], [], [])


def _compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall < 1e-9:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
