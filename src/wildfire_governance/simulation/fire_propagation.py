"""Stochastic cellular automaton fire propagation model.

Implements the sigmoid-CA model from the paper:
    P_spread = sigmoid(alpha1 * W + alpha2 * F - alpha3 * H)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class FirePropagationConfig:
    """Parameters for the stochastic CA fire model.

    Attributes:
        alpha1: Wind contribution coefficient (default 0.3).
        alpha2: Fuel contribution coefficient (default 0.5).
        alpha3: Humidity suppression coefficient (default 0.4).
        spread_model: Model type identifier (currently only "sigmoid_ca").
    """

    alpha1: float = 0.3
    alpha2: float = 0.5
    alpha3: float = 0.4
    spread_model: str = "sigmoid_ca"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_spread_probability(
    wind_field: np.ndarray,
    fuel_map: np.ndarray,
    humidity_field: np.ndarray,
    config: FirePropagationConfig,
) -> np.ndarray:
    """Compute per-cell fire spread probability for the current timestep.

    All input arrays must have the same shape (H, W).

    Args:
        wind_field: Local wind speed magnitude in [0, 1], shape (H, W).
        fuel_map: Fuel load in [0, 1], shape (H, W).
        humidity_field: Relative humidity in [0, 1], shape (H, W).
        config: Model coefficients.

    Returns:
        Array of spread probabilities in [0, 1], shape (H, W).

    Raises:
        ValueError: If input arrays have mismatched shapes.
    """
    if not (wind_field.shape == fuel_map.shape == humidity_field.shape):
        raise ValueError(
            f"Shape mismatch: wind={wind_field.shape}, "
            f"fuel={fuel_map.shape}, humidity={humidity_field.shape}"
        )
    linear = (
        config.alpha1 * wind_field
        + config.alpha2 * fuel_map
        - config.alpha3 * humidity_field
    )
    return _sigmoid(linear)


def propagate_fire(
    fire_mask: np.ndarray,
    wind_field: np.ndarray,
    fuel_map: np.ndarray,
    humidity_field: np.ndarray,
    config: FirePropagationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one step of stochastic CA fire propagation.

    For each non-burning cell adjacent (4-connected) to a burning cell,
    draw a Bernoulli sample with probability P_spread. Ignite if the sample
    is 1 and the cell has non-zero fuel.

    Args:
        fire_mask: Binary grid of currently burning cells, shape (H, W).
        wind_field: Wind speed magnitude in [0, 1], shape (H, W).
        fuel_map: Fuel load in [0, 1], shape (H, W). Cells with fuel=0 cannot ignite.
        humidity_field: Humidity in [0, 1], shape (H, W).
        config: Fire propagation parameters.
        rng: Seeded NumPy Generator for reproducibility.

    Returns:
        Updated binary fire mask, shape (H, W).
    """
    p_spread = compute_spread_probability(wind_field, fuel_map, humidity_field, config)

    # Find cells adjacent to burning cells (4-connected kernel)
    from scipy.ndimage import binary_dilation  # type: ignore[import]
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
    adjacent_to_fire = binary_dilation(fire_mask.astype(bool), structure=kernel)

    # Candidate cells: adjacent, not already burning, have fuel
    candidates = adjacent_to_fire & ~fire_mask.astype(bool) & (fuel_map > 0)

    # Stochastic ignition
    random_draw = rng.random(fire_mask.shape)
    new_ignitions = candidates & (random_draw < p_spread)

    return (fire_mask.astype(bool) | new_ignitions).astype(np.float32)


def initialise_fire(
    grid_size: int,
    n_ignition_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly place initial fire ignition points on the grid.

    Args:
        grid_size: Side length of the square grid.
        n_ignition_points: Number of initial fire cells.
        rng: Seeded NumPy Generator.

    Returns:
        Binary fire mask, shape (grid_size, grid_size).
    """
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    rows = rng.integers(0, grid_size, size=n_ignition_points)
    cols = rng.integers(0, grid_size, size=n_ignition_points)
    mask[rows, cols] = 1.0
    return mask
