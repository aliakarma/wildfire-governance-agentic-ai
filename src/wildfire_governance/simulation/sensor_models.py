"""Sensor models: thermal UAV, ground IoT, satellite."""
from __future__ import annotations
import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

@dataclass
class SensorReading:
    position: Tuple[int, int]; heat_value: float; confidence: float
    is_fire_detected: bool; sensor_type: str

class SensorModel(abc.ABC):
    @abc.abstractmethod
    def observe(self, grid, position, rng) -> SensorReading: ...
    @abc.abstractmethod
    def get_detection_probability(self) -> float: ...

class ThermalUAVSensor(SensorModel):
    def __init__(self, detection_probability=0.85, false_positive_rate=0.05, noise_std=0.05):
        if not 0.0 <= detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1]")
        self._det_prob = detection_probability; self._fp_rate = false_positive_rate
        self._noise_std = noise_std

    def get_detection_probability(self): return self._det_prob

    def observe(self, grid, position, rng):
        row, col = position; true_heat = float(grid[row, col]); is_fire = true_heat > 0.5
        observed = float(np.clip(true_heat + rng.normal(0, self._noise_std), 0.0, 1.0))
        detected = rng.random() < (self._det_prob if is_fire else self._fp_rate)
        confidence = self._det_prob if is_fire else (1.0 - self._fp_rate)
        return SensorReading(position, observed, confidence, detected, "thermal_uav")

class GroundIoTSensor(SensorModel):
    def __init__(self, coverage_radius=5, detection_probability=0.90):
        self._radius = coverage_radius; self._det_prob = detection_probability
    def get_detection_probability(self): return self._det_prob
    def observe(self, grid, position, rng):
        row, col = position; h, w = grid.shape; r = self._radius
        neighbourhood = grid[max(0,row-r):min(h,row+r+1), max(0,col-r):min(w,col+r+1)]
        avg = float(neighbourhood.mean()); is_fire = avg > 0.3
        detected = is_fire and rng.random() < self._det_prob
        return SensorReading(position, avg, self._det_prob, detected, "ground_iot")

class SatelliteFeedSensor(SensorModel):
    def __init__(self, revisit_time_steps=6, latency_steps=2, detection_probability=0.80):
        self._revisit = revisit_time_steps; self._latency = latency_steps
        self._det_prob = detection_probability; self._last_image = None
        self._last_capture_time = -revisit_time_steps
    def get_detection_probability(self): return self._det_prob
    def update_image(self, grid, current_time):
        if current_time - self._last_capture_time >= self._revisit:
            self._last_image = grid.copy(); self._last_capture_time = current_time
    def observe(self, grid, position, rng):
        if self._last_image is None:
            return SensorReading(position, 0.0, 0.0, False, "satellite")
        row, col = position; heat = float(self._last_image[row, col])
        detected = heat > 0.5 and rng.random() < self._det_prob
        return SensorReading(position, heat, self._det_prob, detected, "satellite")
