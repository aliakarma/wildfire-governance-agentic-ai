"""100×100 Wildfire Grid Environment — main simulation entry-point.

Implements the stochastic cellular automaton environment described in the paper.
Compatible with the GOMDP framework and the Gym wrapper in ``rl/gomdp_env.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wildfire_governance.simulation.fire_propagation import (
    FirePropagationConfig,
    initialise_fire,
    propagate_fire,
)
from wildfire_governance.simulation.sensor_models import (
    GroundIoTSensor,
    SatelliteFeedSensor,
    SensorReading,
    ThermalUAVSensor,
)

ObservationDict = Dict[str, Any]
InfoDict = Dict[str, Any]


@dataclass
class EnvironmentConfig:
    """All parameters for the wildfire grid environment."""

    grid_size: int = 100
    n_timesteps: int = 3000
    uav_detection_probability: float = 0.85
    ground_iot_density: float = 0.05
    satellite_revisit: int = 6
    satellite_latency: int = 2
    n_ignition_points: int = 3
    anomaly_injection_rate: float = 0.02
    anomaly_intensity_range: Tuple[float, float] = (0.3, 0.7)
    fire_config: FirePropagationConfig = field(
        default_factory=FirePropagationConfig
    )


class WildfireGridEnvironment:
    """Stochastic wildfire grid environment for multi-agent UAV coordination.

    The environment simulates a 100×100 grid with heterogeneous vegetation,
    humidity, wind, and temperature fields. Fire propagation follows the
    sigmoid CA model from the paper. Synthetic heat anomalies can be injected
    to stress-test the false-alert suppression pipeline.

    Args:
        config: Environment configuration dataclass.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None) -> None:
        self.config = config or EnvironmentConfig()
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._timestep: int = 0
        self._fire_mask: np.ndarray = np.zeros(
            (self.config.grid_size, self.config.grid_size), dtype=np.float32
        )
        self._heat_map: np.ndarray = np.zeros_like(self._fire_mask)
        self._wind_field: np.ndarray = np.zeros_like(self._fire_mask)
        self._humidity_field: np.ndarray = np.zeros_like(self._fire_mask)
        self._fuel_map: np.ndarray = np.zeros_like(self._fire_mask)
        self._satellite = SatelliteFeedSensor(
            revisit_time_steps=self.config.satellite_revisit,
            latency_steps=self.config.satellite_latency,
        )
        self._uav_sensor = ThermalUAVSensor(
            detection_probability=self.config.uav_detection_probability
        )
        self._iot_positions: List[Tuple[int, int]] = []
        self._ignition_time: int = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int = 42) -> ObservationDict:
        """Reset the environment to a new episode.

        Args:
            seed: Random seed for this episode.

        Returns:
            Initial observation dictionary.
        """
        self._rng = np.random.default_rng(seed)
        self._timestep = 0
        gs = self.config.grid_size

        # Generate heterogeneous fields
        self._fuel_map = self._rng.uniform(0.3, 1.0, (gs, gs)).astype(np.float32)
        self._humidity_field = self._rng.uniform(0.2, 0.8, (gs, gs)).astype(np.float32)
        self._wind_field = self._rng.uniform(0.0, 0.6, (gs, gs)).astype(np.float32)

        # Place fire ignitions
        self._fire_mask = initialise_fire(
            gs, self.config.n_ignition_points, self._rng
        )
        self._ignition_time = 0
        self._heat_map = self._fire_mask.copy()

        # Place ground IoT sensors
        n_iot = max(1, int(gs * gs * self.config.ground_iot_density))
        rows = self._rng.integers(0, gs, size=n_iot)
        cols = self._rng.integers(0, gs, size=n_iot)
        self._iot_positions = list(zip(rows.tolist(), cols.tolist()))

        return self._build_observation()

    def step(
        self,
        uav_positions: List[Tuple[int, int]],
    ) -> Tuple[ObservationDict, bool, InfoDict]:
        """Advance the simulation by one timestep.

        Args:
            uav_positions: Current (row, col) positions of all UAVs.

        Returns:
            Tuple of (observation, done, info).
        """
        self._timestep += 1
        done = self._timestep >= self.config.n_timesteps

        # Fire propagation
        self._fire_mask = propagate_fire(
            self._fire_mask,
            self._wind_field,
            self._fuel_map,
            self._humidity_field,
            self.config.fire_config,
            self._rng,
        )

        # Update heat map: smooth blend of fire mask + noise
        noise = self._rng.normal(0, 0.02, self._fire_mask.shape)
        self._heat_map = np.clip(self._fire_mask + noise, 0.0, 1.0).astype(np.float32)

        # Inject synthetic anomalies (non-fire heat sources)
        if self._rng.random() < self.config.anomaly_injection_rate:
            lo, hi = self.config.anomaly_intensity_range
            intensity = self._rng.uniform(lo, hi)
            self.inject_synthetic_anomaly(
                location=(
                    int(self._rng.integers(0, self.config.grid_size)),
                    int(self._rng.integers(0, self.config.grid_size)),
                ),
                intensity=float(intensity),
            )

        # Satellite image capture
        self._satellite.update_image(self._heat_map, self._timestep)

        obs = self._build_observation(uav_positions)
        info: InfoDict = {
            "timestep": self._timestep,
            "fire_cells": int(self._fire_mask.sum()),
            "ignition_time": self._ignition_time,
        }
        return obs, done, info

    def inject_synthetic_anomaly(
        self,
        location: Tuple[int, int],
        intensity: float,
    ) -> None:
        """Inject a false heat source at *location* with given *intensity*.

        Args:
            location: (row, col) grid position.
            intensity: Heat intensity to add, in [0, 1].
        """
        row, col = location
        self._heat_map[row, col] = float(
            np.clip(self._heat_map[row, col] + intensity, 0.0, 1.0)
        )

    def get_observations(
        self, uav_positions: List[Tuple[int, int]]
    ) -> List[SensorReading]:
        """Generate sensor observations for all UAVs at their current positions.

        Args:
            uav_positions: List of (row, col) tuples for each UAV.

        Returns:
            List of SensorReading objects, one per UAV position.
        """
        return [
            self._uav_sensor.observe(self._heat_map, pos, self._rng)
            for pos in uav_positions
        ]

    def render(self) -> np.ndarray:
        """Return an RGB visualisation of the current grid state.

        Returns:
            uint8 array of shape (grid_size, grid_size, 3).
        """
        img = np.zeros((*self._heat_map.shape, 3), dtype=np.uint8)
        img[:, :, 0] = (self._heat_map * 255).astype(np.uint8)  # red = heat
        img[:, :, 2] = (self._fuel_map * 128).astype(np.uint8)  # blue = fuel
        return img

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def heat_map(self) -> np.ndarray:
        """Current heat distribution H_t, shape (H, W)."""
        return self._heat_map

    @property
    def fire_mask(self) -> np.ndarray:
        """Binary fire mask, shape (H, W)."""
        return self._fire_mask

    @property
    def timestep(self) -> int:
        """Current simulation timestep."""
        return self._timestep

    @property
    def grid_size(self) -> int:
        """Side length of the square grid."""
        return self.config.grid_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        uav_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> ObservationDict:
        return {
            "heat_map": self._heat_map.copy(),
            "fire_mask": self._fire_mask.copy(),
            "wind_field": self._wind_field.copy(),
            "humidity_field": self._humidity_field.copy(),
            "fuel_map": self._fuel_map.copy(),
            "uav_positions": uav_positions or [],
            "timestep": self._timestep,
        }
