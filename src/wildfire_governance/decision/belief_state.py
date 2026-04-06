"""Bayesian belief state over wildfire ignition locations."""
from __future__ import annotations
from typing import List
import numpy as np
from wildfire_governance.simulation.sensor_models import SensorReading

class BeliefState:
    def __init__(self, grid_size=100, prior_fire_prob=0.01):
        self._grid_size = grid_size; self._prior = prior_fire_prob
        self._belief = np.full((grid_size, grid_size), prior_fire_prob, dtype=np.float64)

    def update(self, observations, p_detect_fire=0.85, p_detect_no_fire=0.15):
        for obs in observations:
            row, col = obs.position
            if not (0 <= row < self._grid_size and 0 <= col < self._grid_size): continue
            prior = self._belief[row, col]
            lf = p_detect_fire if obs.is_fire_detected else (1.0 - p_detect_fire)
            lnf = p_detect_no_fire if obs.is_fire_detected else (1.0 - p_detect_no_fire)
            denom = lf * prior + lnf * (1.0 - prior)
            self._belief[row, col] = (lf * prior / denom) if denom > 0 else prior
        self._belief = 0.95 * self._belief + 0.05 * self._prior
        self._belief = np.clip(self._belief, 1e-8, 1.0 - 1e-8)

    def get_risk_map(self): return self._belief.copy()
    def get_belief(self): return self._belief.copy()
    def entropy(self):
        b = np.clip(self._belief.ravel(), 1e-12, 1.0)
        return float(-np.sum(b * np.log(b)))
    def reset(self, prior_fire_prob=None):
        p = prior_fire_prob if prior_fire_prob is not None else self._prior
        self._belief = np.full((self._grid_size, self._grid_size), p, dtype=np.float64)
