"""Fire risk prediction agent — maintains dynamic risk estimate R_t."""
from __future__ import annotations

import numpy as np

from wildfire_governance.simulation.digital_twin import WildfireRiskDigitalTwin


class FireRiskPredictionAgent:
    """Maintains and updates the dynamic wildfire risk estimate R_t.

    Wraps the digital twin to provide the coordination engine with a
    continuously updated risk map and short-term propagation forecasts.

    Args:
        grid_size: Environment grid side length.
        forecast_horizon: Steps ahead for risk forecasting.
    """

    def __init__(self, grid_size: int = 100, forecast_horizon: int = 10) -> None:
        self._twin = WildfireRiskDigitalTwin(
            grid_size=grid_size, forecast_horizon=forecast_horizon
        )

    def update(
        self,
        heat_map: np.ndarray,
        wind_field: np.ndarray,
        humidity_field: np.ndarray,
        fuel_map: np.ndarray,
        belief_map: np.ndarray | None = None,
    ) -> None:
        """Ingest sensor data and update the risk estimate.

        Args:
            heat_map: Current heat distribution H_t.
            wind_field: Wind field W_t.
            humidity_field: Humidity field.
            fuel_map: Fuel load map.
            belief_map: Optional Bayesian belief map.
        """
        self._twin.update(heat_map, wind_field, humidity_field, fuel_map, belief_map)

    def get_risk_map(self) -> np.ndarray:
        """Return the current integrated risk map R_t."""
        return self._twin.get_risk_map()

    def forecast(self, steps_ahead: int = 5) -> np.ndarray:
        """Return a short-horizon risk forecast.

        Args:
            steps_ahead: Number of steps to forecast.

        Returns:
            Forecasted risk map of shape (H, W).
        """
        return self._twin.forecast_risk(steps_ahead)
