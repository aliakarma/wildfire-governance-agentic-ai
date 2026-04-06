"""3-D wildfire risk digital twin for predictive simulation and risk forecasting.

The digital twin maintains a continuously updated spatial risk model of the
monitored region and supplies the coordination engine with risk priors and
fire-propagation estimates used for patrol reallocation and anomaly verification.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from wildfire_governance.simulation.fire_propagation import (
    FirePropagationConfig,
    compute_spread_probability,
)


class WildfireRiskDigitalTwin:
    """Dynamic 3-D wildfire-risk digital twin.

    Ingests multi-modal sensing data to perform predictive simulation,
    scenario analysis, and risk forecasting. Provides spatial risk priors
    to the coordination engine.

    Args:
        grid_size: Side length of the square monitored region.
        forecast_horizon: Number of steps to simulate ahead for risk prediction.
        fire_config: Fire propagation model parameters.
    """

    def __init__(
        self,
        grid_size: int = 100,
        forecast_horizon: int = 10,
        fire_config: Optional[FirePropagationConfig] = None,
    ) -> None:
        self._gs = grid_size
        self._horizon = forecast_horizon
        self._fire_config = fire_config or FirePropagationConfig()
        self._risk_map: np.ndarray = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._propagation_probs: np.ndarray = np.zeros_like(self._risk_map)
        self._wind_field: np.ndarray = np.zeros_like(self._risk_map)
        self._humidity_field: np.ndarray = np.ones_like(self._risk_map) * 0.5
        self._fuel_map: np.ndarray = np.ones_like(self._risk_map) * 0.5
        self._update_count: int = 0

    def update(
        self,
        heat_map: np.ndarray,
        wind_field: np.ndarray,
        humidity_field: np.ndarray,
        fuel_map: np.ndarray,
        belief_map: Optional[np.ndarray] = None,
    ) -> None:
        """Ingest multi-modal sensor data and update the risk model.

        Args:
            heat_map: Current heat distribution H_t, shape (H, W).
            wind_field: Wind speed magnitude field W_t, shape (H, W).
            humidity_field: Humidity field in [0, 1], shape (H, W).
            fuel_map: Fuel load map in [0, 1], shape (H, W).
            belief_map: Optional Bayesian belief state from the coordination engine.
        """
        self._wind_field = wind_field.astype(np.float32)
        self._humidity_field = humidity_field.astype(np.float32)
        self._fuel_map = fuel_map.astype(np.float32)

        # Compute instantaneous spread probability at each cell
        self._propagation_probs = compute_spread_probability(
            wind_field, fuel_map, humidity_field, self._fire_config
        )

        # Risk = blend of current heat, spread probability, and belief
        risk = 0.5 * heat_map + 0.3 * self._propagation_probs
        if belief_map is not None:
            risk += 0.2 * belief_map.astype(np.float32)
        self._risk_map = np.clip(risk, 0.0, 1.0).astype(np.float32)
        self._update_count += 1

    def forecast_risk(self, steps_ahead: int = 5) -> np.ndarray:
        """Forecast fire spread risk over a short horizon.

        Runs a deterministic (mean-field) propagation simulation for
        ``steps_ahead`` timesteps starting from the current risk map.

        Args:
            steps_ahead: Number of prediction steps (capped at forecast_horizon).

        Returns:
            Forecasted risk map of shape (H, W), values in [0, 1].
        """
        steps = min(steps_ahead, self._horizon)
        forecast = self._risk_map.copy()
        for _ in range(steps):
            spread = compute_spread_probability(
                self._wind_field, self._fuel_map, self._humidity_field, self._fire_config
            )
            # Mean-field update: risk propagates proportional to spread probability
            forecast = np.clip(forecast + 0.1 * spread * forecast, 0.0, 1.0)
        return forecast.astype(np.float32)

    def get_risk_map(self) -> np.ndarray:
        """Return the current integrated risk estimate R_t.

        Returns:
            Float32 array of shape (grid_size, grid_size) with values in [0, 1].
        """
        return self._risk_map.copy()

    def get_high_risk_sectors(self, threshold: float = 0.4, top_k: int = 10) -> list:
        """Return the top-k grid cells with the highest risk scores.

        Args:
            threshold: Minimum risk to be considered high-risk.
            top_k: Maximum number of cells to return.

        Returns:
            List of (row, col, risk_value) tuples sorted by descending risk.
        """
        above = np.argwhere(self._risk_map > threshold)
        if len(above) == 0:
            return []
        risks = [(int(r), int(c), float(self._risk_map[r, c])) for r, c in above]
        risks.sort(key=lambda x: x[2], reverse=True)
        return risks[:top_k]

    def get_summary(self) -> Dict[str, Any]:
        """Return a diagnostic summary of the digital twin state."""
        return {
            "update_count": self._update_count,
            "mean_risk": float(self._risk_map.mean()),
            "max_risk": float(self._risk_map.max()),
            "high_risk_cells": int((self._risk_map > 0.5).sum()),
            "mean_spread_prob": float(self._propagation_probs.mean()),
        }
