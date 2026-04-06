"""100x100 Wildfire Grid Environment."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from wildfire_governance.simulation.fire_propagation import FirePropagationConfig, initialise_fire, propagate_fire
from wildfire_governance.simulation.sensor_models import SatelliteFeedSensor, SensorReading, ThermalUAVSensor

@dataclass
class EnvironmentConfig:
    grid_size: int = 100; n_timesteps: int = 3000; uav_detection_probability: float = 0.85; uav_noise_std: float = 0.05
    ground_iot_density: float = 0.05; satellite_revisit: int = 6; satellite_latency: int = 2
    n_ignition_points: int = 3; anomaly_injection_rate: float = 0.02
    anomaly_intensity_range: Tuple[float,float] = (0.3, 0.7)
    fire_config: FirePropagationConfig = field(default_factory=FirePropagationConfig)

class WildfireGridEnvironment:
    def __init__(self, config=None):
        self.config = config or EnvironmentConfig()
        self._rng = np.random.default_rng(42); self._timestep = 0
        gs = self.config.grid_size
        self._fire_mask = np.zeros((gs, gs), dtype=np.float32)
        self._heat_map = np.zeros((gs, gs), dtype=np.float32)
        self._wind_field = np.zeros((gs, gs), dtype=np.float32)
        self._humidity_field = np.zeros((gs, gs), dtype=np.float32)
        self._fuel_map = np.zeros((gs, gs), dtype=np.float32)
        self._satellite = SatelliteFeedSensor(self.config.satellite_revisit, self.config.satellite_latency)
        self._uav_sensor = ThermalUAVSensor(detection_probability=self.config.uav_detection_probability, noise_std=self.config.uav_noise_std)
        self._iot_positions: List[Tuple[int,int]] = []
        self._ignition_time = -1

    def reset(self, seed=42):
        self._rng = np.random.default_rng(seed); self._timestep = 0; gs = self.config.grid_size
        self._fuel_map = self._rng.uniform(0.3, 1.0, (gs, gs)).astype(np.float32)
        self._humidity_field = self._rng.uniform(0.2, 0.8, (gs, gs)).astype(np.float32)
        self._wind_field = self._rng.uniform(0.0, 0.6, (gs, gs)).astype(np.float32)
        self._fire_mask = initialise_fire(gs, self.config.n_ignition_points, self._rng)
        self._ignition_time = 0; self._heat_map = self._fire_mask.copy()
        n_iot = max(1, int(gs*gs*self.config.ground_iot_density))
        rows = self._rng.integers(0, gs, size=n_iot); cols = self._rng.integers(0, gs, size=n_iot)
        self._iot_positions = list(zip(rows.tolist(), cols.tolist()))
        return self._build_observation()

    def step(self, uav_positions):
        self._timestep += 1; done = self._timestep >= self.config.n_timesteps
        self._fire_mask = propagate_fire(self._fire_mask, self._wind_field, self._fuel_map,
                                         self._humidity_field, self.config.fire_config, self._rng)
        noise = self._rng.normal(0, 0.02, self._fire_mask.shape)
        self._heat_map = np.clip(self._fire_mask + noise, 0.0, 1.0).astype(np.float32)
        if self._rng.random() < self.config.anomaly_injection_rate:
            lo, hi = self.config.anomaly_intensity_range
            self.inject_synthetic_anomaly(
                (int(self._rng.integers(0,self.config.grid_size)), int(self._rng.integers(0,self.config.grid_size))),
                float(self._rng.uniform(lo, hi)))
        self._satellite.update_image(self._heat_map, self._timestep)
        return self._build_observation(uav_positions), done, {"timestep": self._timestep, "fire_cells": int(self._fire_mask.sum()), "ignition_time": self._ignition_time}

    def inject_synthetic_anomaly(self, location, intensity):
        row, col = location; self._heat_map[row,col] = float(np.clip(self._heat_map[row,col]+intensity,0.0,1.0))

    def get_observations(self, uav_positions):
        return [self._uav_sensor.observe(self._heat_map, pos, self._rng) for pos in uav_positions]

    def render(self):
        img = np.zeros((*self._heat_map.shape, 3), dtype=np.uint8)
        img[:,:,0] = (self._heat_map*255).astype(np.uint8)
        img[:,:,2] = (self._fuel_map*128).astype(np.uint8); return img

    @property
    def heat_map(self): return self._heat_map
    @property
    def fire_mask(self): return self._fire_mask
    @property
    def timestep(self): return self._timestep
    @property
    def grid_size(self): return self.config.grid_size

    def _build_observation(self, uav_positions=None):
        return {"heat_map": self._heat_map.copy(), "fire_mask": self._fire_mask.copy(),
                "wind_field": self._wind_field.copy(), "humidity_field": self._humidity_field.copy(),
                "fuel_map": self._fuel_map.copy(), "uav_positions": uav_positions or [],
                "timestep": self._timestep}
