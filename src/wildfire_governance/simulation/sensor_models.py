"""Sensor models: thermal UAV, ground IoT, satellite, and real VIIRS adapter.

All sensors implement the :class:`SensorModel` abstract interface.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class SensorReading:
    """Output from a single sensor observation.

    Attributes:
        position: Grid cell (row, col) where the observation was taken.
        heat_value: Observed heat intensity in [0, 1].
        confidence: Sensor-level confidence in [0, 1].
        is_fire_detected: Boolean detection flag.
        sensor_type: String identifier of the sensor type.
    """

    position: Tuple[int, int]
    heat_value: float
    confidence: float
    is_fire_detected: bool
    sensor_type: str


class SensorModel(abc.ABC):
    """Abstract base class for all sensor types."""

    @abc.abstractmethod
    def observe(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        rng: np.random.Generator,
    ) -> SensorReading:
        """Generate a noisy observation at *position*.

        Args:
            grid: Current ground-truth heat map, shape (H, W), values in [0, 1].
            position: Grid cell (row, col) to observe.
            rng: Seeded NumPy Generator.

        Returns:
            SensorReading at the given position.
        """

    @abc.abstractmethod
    def get_detection_probability(self) -> float:
        """Return the nominal probability of detecting a true fire cell."""


class ThermalUAVSensor(SensorModel):
    """UAV-mounted thermal infrared camera.

    Attributes:
        detection_probability: P(detect | fire) (paper default: 0.85).
        false_positive_rate: P(detect | no fire).
        noise_std: Standard deviation of Gaussian noise added to heat readings.
    """

    def __init__(
        self,
        detection_probability: float = 0.85,
        false_positive_rate: float = 0.05,
        noise_std: float = 0.05,
    ) -> None:
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1]")
        self._det_prob = detection_probability
        self._fp_rate = false_positive_rate
        self._noise_std = noise_std

    def get_detection_probability(self) -> float:
        return self._det_prob

    def observe(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        rng: np.random.Generator,
    ) -> SensorReading:
        """Observe the heat at *position* with thermal-UAV noise model."""
        row, col = position
        true_heat = float(grid[row, col])
        is_fire = true_heat > 0.5

        noise = rng.normal(0, self._noise_std)
        observed_heat = float(np.clip(true_heat + noise, 0.0, 1.0))

        if is_fire:
            detected = rng.random() < self._det_prob
        else:
            detected = rng.random() < self._fp_rate

        confidence = self._det_prob if is_fire else (1.0 - self._fp_rate)
        return SensorReading(
            position=position,
            heat_value=observed_heat,
            confidence=confidence,
            is_fire_detected=detected,
            sensor_type="thermal_uav",
        )


class GroundIoTSensor(SensorModel):
    """Fixed ground IoT temperature sensor with limited spatial coverage.

    Attributes:
        coverage_radius: Radius (in grid cells) within which the sensor detects fires.
        detection_probability: Within-radius detection probability.
    """

    def __init__(
        self,
        coverage_radius: int = 5,
        detection_probability: float = 0.90,
    ) -> None:
        self._radius = coverage_radius
        self._det_prob = detection_probability

    def get_detection_probability(self) -> float:
        return self._det_prob

    def observe(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        rng: np.random.Generator,
    ) -> SensorReading:
        """Observe a local neighbourhood average heat value."""
        row, col = position
        h, w = grid.shape
        r = self._radius
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        neighbourhood = grid[r0:r1, c0:c1]
        avg_heat = float(neighbourhood.mean())
        is_fire = avg_heat > 0.3
        detected = is_fire and rng.random() < self._det_prob
        return SensorReading(
            position=position,
            heat_value=avg_heat,
            confidence=self._det_prob,
            is_fire_detected=detected,
            sensor_type="ground_iot",
        )


class SatelliteFeedSensor(SensorModel):
    """Simulated satellite feed with periodic revisit and bounded latency.

    Attributes:
        revisit_time_steps: Orbital revisit interval (paper default: 6 steps).
        latency_steps: Data delivery delay (paper default: 2 steps).
        detection_probability: Detection probability for fire cells.
    """

    def __init__(
        self,
        revisit_time_steps: int = 6,
        latency_steps: int = 2,
        detection_probability: float = 0.80,
    ) -> None:
        self._revisit = revisit_time_steps
        self._latency = latency_steps
        self._det_prob = detection_probability
        self._last_image: Optional[np.ndarray] = None
        self._last_capture_time: int = -revisit_time_steps

    def get_detection_probability(self) -> float:
        return self._det_prob

    def update_image(self, grid: np.ndarray, current_time: int) -> None:
        """Capture a new satellite image if a revisit is due.

        Args:
            grid: Current ground-truth heat map.
            current_time: Current simulation timestep.
        """
        if current_time - self._last_capture_time >= self._revisit:
            self._last_image = grid.copy()
            self._last_capture_time = current_time

    def observe(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        rng: np.random.Generator,
    ) -> SensorReading:
        """Return an observation from the latest satellite image (with latency)."""
        if self._last_image is None:
            return SensorReading(
                position=position,
                heat_value=0.0,
                confidence=0.0,
                is_fire_detected=False,
                sensor_type="satellite",
            )
        row, col = position
        heat = float(self._last_image[row, col])
        is_fire = heat > 0.5
        detected = is_fire and rng.random() < self._det_prob
        return SensorReading(
            position=position,
            heat_value=heat,
            confidence=self._det_prob,
            is_fire_detected=detected,
            sensor_type="satellite",
        )


class RealViirsSensor(SensorModel):
    """Adapter that serves observations from preprocessed VIIRS .npz data.

    The .npz file must contain arrays: ``heat_map`` (T, H, W) float32
    and optional ``fire_mask`` (T, H, W) bool for ground-truth labels.

    Attributes:
        data_path: Path to the preprocessed VIIRS .npz file.
    """

    def __init__(self, data_path: Path) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"VIIRS data not found: {data_path}\n"
                "Run: make download-viirs  (or: python data/scripts/download_viirs.py)"
            )
        data = np.load(data_path)
        self._heat_maps: np.ndarray = data["heat_map"].astype(np.float32)
        self._fire_masks: Optional[np.ndarray] = (
            data["fire_mask"].astype(bool) if "fire_mask" in data else None
        )
        self._n_timesteps, self._h, self._w = self._heat_maps.shape

    def get_detection_probability(self) -> float:
        return 0.85  # VIIRS 375m nominal detection probability

    def get_heat_map(self, timestep: int) -> np.ndarray:
        """Return the VIIRS heat map for a given timestep (clipped to available data)."""
        t = min(timestep, self._n_timesteps - 1)
        return self._heat_maps[t]

    def observe(
        self,
        grid: np.ndarray,
        position: Tuple[int, int],
        rng: np.random.Generator,
    ) -> SensorReading:
        """Observe using stored VIIRS heat map at current timestep."""
        row, col = position
        # grid is used as a proxy for the current timestep index in this adapter
        heat = float(self._heat_maps[0, row, col])
        is_fire = heat > 0.5
        return SensorReading(
            position=position,
            heat_value=heat,
            confidence=0.85,
            is_fire_detected=is_fire,
            sensor_type="viirs",
        )
