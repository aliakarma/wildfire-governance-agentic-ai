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

# Heat above which a thermal sensor registers a cell as a candidate source.
# Burning cells sit near 1.0 and cold ground near 0, so this only changes the
# treatment of hot *non-fire* sources, which is what the two-stage verification
# pipeline exists to discriminate.
_HOT_SOURCE_THRESHOLD = 0.50


@dataclass
class EnvironmentConfig:
    """All parameters for the wildfire grid environment."""

    grid_size: int = 100
    n_timesteps: int = 3000
    uav_detection_probability: float = 0.85
    uav_footprint_radius: int = 4
    ground_iot_density: float = 0.05
    ground_iot_radius: int = 5
    satellite_revisit: int = 6
    satellite_latency: int = 2
    n_ignition_points: int = 3
    ignition_delay_range: Tuple[int, int] = (50, 150)
    anomaly_injection_rate: float = 0.02
    anomaly_decay: float = 0.90
    # Confusers must be able to reach the tau=0.80 confidence gate or the
    # false-alert-suppression claim is untestable: at 0.65*heat + 0.35*weather
    # with weather capped near 0.9, an anomaly needs heat > ~0.75 to escalate,
    # so the old (0.3, 0.7) range made F_p identically zero whatever the
    # pipeline did. Hot agricultural and industrial sources are genuinely
    # comparable to wildfire in thermal signature, so the upper range is
    # physical, and it is exactly those confusers that make suppression hard.
    anomaly_intensity_range: Tuple[float, float] = (0.55, 0.95)
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
        # Static IoT footprint, rebuilt on reset when sensors are re-placed.
        self._iot_cov_cache: Optional[np.ndarray] = None
        self._ignition_time: int = -1
        # Fire seeded at reset but withheld until _ignition_time; None once fired.
        self._pending_ignition: Optional[np.ndarray] = None
        # Persistent non-fire heat sources (decaying), kept separate from the
        # fire mask so ground truth for "is this a real fire?" stays clean.
        self._anomaly_field: np.ndarray = np.zeros_like(self._fire_mask)

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

        # Schedule ignition at a stochastic time after monitoring begins, rather
        # than at t=0. Igniting at t=0 alongside randomly-placed UAVs means a UAV
        # sometimes spawns on top of the fire and detects it instantly, which
        # collapses L_d to 0 and destroys its meaning as a search-performance
        # measure. A delayed ignition lets the fleet disperse into its patrol
        # pattern first, so L_d measures time-to-find a *new* fire.
        lo, hi = self.config.ignition_delay_range
        self._ignition_time = int(self._rng.integers(lo, hi + 1))
        self._pending_ignition = initialise_fire(
            gs, self.config.n_ignition_points, self._rng
        )
        self._fire_mask = np.zeros((gs, gs), dtype=np.float32)
        self._anomaly_field = np.zeros((gs, gs), dtype=np.float32)
        self._heat_map = self._fire_mask.copy()

        # Place ground IoT sensors
        n_iot = max(1, int(gs * gs * self.config.ground_iot_density))
        rows = self._rng.integers(0, gs, size=n_iot)
        cols = self._rng.integers(0, gs, size=n_iot)
        self._iot_positions = list(zip(rows.tolist(), cols.tolist()))
        self._iot_cov_cache = None  # sensors moved; footprint must be rebuilt

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

        # Ignition event: seed the scheduled fire once its time arrives.
        if self._pending_ignition is not None and self._timestep >= self._ignition_time:
            self._fire_mask = self._pending_ignition
            self._pending_ignition = None

        # Fire propagation
        self._fire_mask = propagate_fire(
            self._fire_mask,
            self._wind_field,
            self._fuel_map,
            self._humidity_field,
            self.config.fire_config,
            self._rng,
        )

        # Update heat map: fire mask + persistent anomalies + noise.
        # Anomalies decay geometrically (half-life ~7 steps) so a non-fire heat
        # source stays observable long enough for the fleet to encounter it.
        self._anomaly_field *= self.config.anomaly_decay
        self._anomaly_field[self._anomaly_field < 0.01] = 0.0
        noise = self._rng.normal(0, 0.02, self._fire_mask.shape)
        self._heat_map = np.clip(
            self._fire_mask + self._anomaly_field + noise, 0.0, 1.0
        ).astype(np.float32)

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
        # Persist the anomaly in its own field. Writing straight into the heat
        # map lets it survive exactly one step, because ``step`` rebuilds the
        # heat map from the fire mask each tick; a source that exists for a
        # single timestep is effectively undetectable and yields F_p == 0.
        # Real non-fire heat sources (agricultural burns, industrial plant,
        # solar-heated rock) persist over minutes, so the anomaly decays
        # geometrically instead of vanishing.
        self._anomaly_field[row, col] = float(
            np.clip(self._anomaly_field[row, col] + intensity, 0.0, 1.0)
        )
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

    def coverage_mask(
        self, uav_positions: List[Tuple[int, int]], radius: Optional[int] = None
    ) -> np.ndarray:
        """Return the boolean mask of cells within any UAV's sensor footprint.

        A UAV observes a square (Chebyshev) footprint of side ``2*radius+1``
        centred on its position, representing a downward-facing thermal camera's
        field of view at nominal altitude.

        Args:
            uav_positions: Current (row, col) of each UAV.
            radius: Footprint radius in cells; defaults to the configured value.

        Returns:
            Boolean array of shape (grid_size, grid_size), True where covered.
        """
        r = self.config.uav_footprint_radius if radius is None else radius
        gs = self.config.grid_size
        mask = np.zeros((gs, gs), dtype=bool)
        for row, col in uav_positions:
            r0, r1 = max(0, int(row) - r), min(gs, int(row) + r + 1)
            c0, c1 = max(0, int(col) - r), min(gs, int(col) + r + 1)
            mask[r0:r1, c0:c1] = True
        return mask

    def sense_fused(
        self, uav_positions: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """Fuse UAV, ground-IoT, and satellite observations into a detection field.

        This is the environment's detection interface. Crucially, it returns only
        what the sensing assets can actually observe: a fire is invisible to the
        system until a UAV footprint, an IoT sensor's coverage radius, or a
        satellite revisit covers it. Detection therefore depends on where the
        coordination policy has steered the fleet, which is the property that
        makes detection latency a meaningful measure of policy quality.

        Sensor parameters follow the manuscript: UAV thermal P(detect|fire)=0.85
        over a configurable footprint, ground IoT at density 0.05 with a 5-cell
        radius, and a satellite feed with 6-step revisit and 2-step latency.

        Args:
            uav_positions: Current (row, col) of each UAV.

        Returns:
            Dict with keys:
              ``observed_heat``: heat field masked to observed cells (unobserved = 0).
              ``coverage``: boolean mask of cells observed by any modality.
              ``max_heat``: maximum observed heat over detected cells (0 if none).
              ``argmax_cell``: (row, col) of that maximum, or None.
              ``detected``: whether any modality raised a detection this step.
              ``detected_fire``: whether a detected cell is *actually* burning.
                  This is the event that stops the detection-latency clock;
                  ``detected`` alone also fires on non-fire heat anomalies.
        """
        gs = self.config.grid_size
        truth = self._heat_map
        fire = self._fire_mask > 0.5

        # --- UAV thermal footprints (vectorised ThermalUAVSensor semantics) ---
        uav_cov = self.coverage_mask(uav_positions)
        noise = self._rng.normal(0.0, 0.05, (gs, gs))
        observed = np.clip(truth + noise, 0.0, 1.0).astype(np.float32)

        draw = self._rng.random((gs, gs))
        # A thermal camera responds to heat, not to ground truth. Keying the
        # detection probability on the fire mask made hot non-fire sources
        # (agricultural burns, industrial plant) all but invisible at 0.05,
        # so no anomaly ever reached the verification pipeline and the
        # false-alert rate was zero by construction rather than by suppression.
        # Deciding whether a hot cell is a fire is the pipeline's job, not the
        # sensor's.
        hot = truth > _HOT_SOURCE_THRESHOLD
        det_prob = np.where(hot, self.config.uav_detection_probability, 0.05)
        uav_detect = uav_cov & (draw < det_prob)

        # --- Ground IoT: fixed thermometers, local mean heat must exceed 0.3 ---
        # Matches GroundIoTSensor.observe: a point temperature sensor only fires
        # once a substantial fraction of its neighbourhood is burning, so it
        # cannot see a small new ignition. Without this threshold, IoT at
        # density 0.05 with radius 5 blankets the grid and UAVs become redundant.
        from scipy.ndimage import uniform_filter  # type: ignore[import]

        iot_r = self.config.ground_iot_radius
        local_mean = uniform_filter(truth, size=2 * iot_r + 1, mode="constant")
        # Ground sensors are fixed installations, so their footprint is an
        # episode constant. Rebuilding it per step meant a 500-iteration Python
        # loop on every one of 3000 steps — 64% of this method's runtime.
        if self._iot_cov_cache is None:
            self._iot_cov_cache = self.coverage_mask(
                self._iot_positions, radius=iot_r
            )
        iot_cov = self._iot_cov_cache
        iot_draw = self._rng.random((gs, gs))
        iot_detect = iot_cov & (local_mean > 0.30) & (iot_draw < 0.90)

        # --- Satellite: coarse resolution, revisit-limited, stale by design ---
        # GOES-16 pixels are ~2 km against ~250 m grid cells, so detection
        # requires a fire large enough to register in an aggregated block. A
        # three-cell ignition is invisible to it; this is why the UAV fleet is
        # the early-detection asset and why L_d depends on coordination.
        sat_detect = np.zeros((gs, gs), dtype=bool)
        if self._satellite._last_image is not None:  # noqa: SLF001 - internal by design
            stale = self._satellite._last_image  # noqa: SLF001
            age = self._timestep - self._satellite._last_capture_time  # noqa: SLF001
            if age >= self.config.satellite_latency:
                block_mean = uniform_filter(stale, size=8, mode="constant")
                sat_draw = self._rng.random((gs, gs))
                sat_detect = (block_mean > 0.30) & (sat_draw < 0.80)

        detect = uav_detect | iot_detect | sat_detect
        coverage = uav_cov | iot_cov

        observed_heat = np.where(detect, observed, 0.0).astype(np.float32)
        if detect.any():
            flat = int(observed_heat.argmax())
            row, col = np.unravel_index(flat, observed_heat.shape)
            max_heat = float(observed_heat[row, col])
            argmax_cell: Optional[Tuple[int, int]] = (int(row), int(col))
        else:
            max_heat = 0.0
            argmax_cell = None

        return {
            "observed_heat": observed_heat,
            "coverage": coverage,
            "max_heat": max_heat,
            "argmax_cell": argmax_cell,
            "detected": bool(detect.any()),
            "detected_fire": bool((detect & fire).any()),
        }

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
    def ignition_time(self) -> int:
        """Timestep at which the scheduled fire is seeded (-1 before reset)."""
        return self._ignition_time

    @property
    def wind_field(self) -> np.ndarray:
        """Current wind field W_t, shape (H, W)."""
        return self._wind_field

    @property
    def humidity_field(self) -> np.ndarray:
        """Current humidity field, shape (H, W)."""
        return self._humidity_field

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
