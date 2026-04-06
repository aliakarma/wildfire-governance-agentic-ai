"""Bayesian belief state over wildfire ignition locations.

Maintains and updates the belief distribution b_t = P(s_t | o_{1:t}, a_{1:t-1})
using a grid-based Bayesian filter with multi-modal sensor fusion.
"""
from __future__ import annotations

from typing import List

import numpy as np

from wildfire_governance.simulation.sensor_models import SensorReading


class BeliefState:
    """Grid-based Bayesian belief state over fire ignition cells.

    Maintains a probability map over grid cells representing the current
    estimate of fire presence. Updated at each timestep using sensor
    observations via a Bayesian filter.

    Args:
        grid_size: Side length of the square grid.
        prior_fire_prob: Initial uniform prior for fire presence (default 0.01).
    """

    def __init__(
        self,
        grid_size: int = 100,
        prior_fire_prob: float = 0.01,
    ) -> None:
        self._grid_size = grid_size
        self._belief: np.ndarray = np.full(
            (grid_size, grid_size), prior_fire_prob, dtype=np.float64
        )
        self._prior = prior_fire_prob

    def update(
        self,
        observations: List[SensorReading],
        p_detect_fire: float = 0.85,
        p_detect_no_fire: float = 0.15,
    ) -> None:
        """Apply a Bayesian update using sensor observations.

        For each observation at (row, col):
            If detected:   P_new ∝ P_det_fire  * P_prior  (fire more likely)
            If not detected: P_new ∝ (1-P_det_fire) * P_prior  (fire less likely)

        Normalises the belief map after all observations are applied.

        Args:
            observations: List of SensorReading from any sensor type.
            p_detect_fire: P(obs | fire) — detection probability.
            p_detect_no_fire: P(obs | no fire) — false positive rate.
        """
        for obs in observations:
            row, col = obs.position
            if not (0 <= row < self._grid_size and 0 <= col < self._grid_size):
                continue
            prior = self._belief[row, col]
            if obs.is_fire_detected:
                likelihood_fire = p_detect_fire
                likelihood_no_fire = p_detect_no_fire
            else:
                likelihood_fire = 1.0 - p_detect_fire
                likelihood_no_fire = 1.0 - p_detect_no_fire

            numerator = likelihood_fire * prior
            denominator = (
                likelihood_fire * prior
                + likelihood_no_fire * (1.0 - prior)
            )
            self._belief[row, col] = numerator / denominator if denominator > 0 else prior

        # Temporal decay: beliefs drift toward prior (fire may spread)
        self._belief = 0.95 * self._belief + 0.05 * self._prior
        self._belief = np.clip(self._belief, 1e-8, 1.0 - 1e-8)

    def get_risk_map(self) -> np.ndarray:
        """Return the current risk estimate R_t derived from belief.

        Returns:
            Float64 array of shape (grid_size, grid_size) with values in [0, 1].
        """
        return self._belief.copy()

    def get_belief(self) -> np.ndarray:
        """Return the full belief distribution b_t.

        Returns:
            Float64 array of shape (grid_size, grid_size).
        """
        return self._belief.copy()

    def entropy(self) -> float:
        """Compute the Shannon entropy of the belief distribution.

        High entropy indicates high uncertainty about fire locations.

        Returns:
            Non-negative float (nats).
        """
        b = self._belief.ravel()
        b = np.clip(b, 1e-12, 1.0)
        return float(-np.sum(b * np.log(b)))

    def reset(self, prior_fire_prob: float | None = None) -> None:
        """Reset belief to uniform prior.

        Args:
            prior_fire_prob: Override prior; if None, uses the initial prior.
        """
        p = prior_fire_prob if prior_fire_prob is not None else self._prior
        self._belief = np.full(
            (self._grid_size, self._grid_size), p, dtype=np.float64
        )
