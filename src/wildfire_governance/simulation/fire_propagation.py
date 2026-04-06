"""Stochastic CA fire propagation model."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

@dataclass
class FirePropagationConfig:
    alpha1: float = 0.3; alpha2: float = 0.5; alpha3: float = 0.4
    spread_model: str = "sigmoid_ca"

def _sigmoid(x):
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def compute_spread_probability(wind_field, fuel_map, humidity_field, config):
    if not (wind_field.shape == fuel_map.shape == humidity_field.shape):
        raise ValueError(f"Shape mismatch: wind={wind_field.shape}, fuel={fuel_map.shape}, humidity={humidity_field.shape}")
    linear = config.alpha1*wind_field + config.alpha2*fuel_map - config.alpha3*humidity_field
    return _sigmoid(linear)

def propagate_fire(fire_mask, wind_field, fuel_map, humidity_field, config, rng):
    from scipy.ndimage import binary_dilation
    p_spread = compute_spread_probability(wind_field, fuel_map, humidity_field, config)
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
    adjacent = binary_dilation(fire_mask.astype(bool), structure=kernel)
    candidates = adjacent & ~fire_mask.astype(bool) & (fuel_map > 0)
    new_ignitions = candidates & (rng.random(fire_mask.shape) < p_spread)
    return (fire_mask.astype(bool) | new_ignitions).astype(np.float32)

def initialise_fire(grid_size, n_ignition_points, rng):
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    rows = rng.integers(0, grid_size, size=n_ignition_points)
    cols = rng.integers(0, grid_size, size=n_ignition_points)
    mask[rows, cols] = 1.0
    return mask
