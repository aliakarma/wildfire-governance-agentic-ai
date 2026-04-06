"""Unit tests for fire_propagation.py."""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.simulation.fire_propagation import (
    FirePropagationConfig,
    compute_spread_probability,
    initialise_fire,
    propagate_fire,
)


def test_spread_probability_range() -> None:
    """P_spread must always be in [0, 1]."""
    gs = 20
    rng = np.random.default_rng(0)
    wind = rng.uniform(0, 1, (gs, gs)).astype(np.float32)
    fuel = rng.uniform(0, 1, (gs, gs)).astype(np.float32)
    humidity = rng.uniform(0, 1, (gs, gs)).astype(np.float32)
    cfg = FirePropagationConfig()
    probs = compute_spread_probability(wind, fuel, humidity, cfg)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_no_spread_zero_fuel() -> None:
    """With fuel=0, fire cannot spread to adjacent cells."""
    gs = 10
    fire = np.zeros((gs, gs), dtype=np.float32)
    fire[5, 5] = 1.0
    wind = np.ones((gs, gs), dtype=np.float32) * 0.5
    fuel = np.zeros((gs, gs), dtype=np.float32)  # No fuel
    humidity = np.zeros((gs, gs), dtype=np.float32)
    cfg = FirePropagationConfig(alpha2=1.0)
    rng = np.random.default_rng(42)
    new_mask = propagate_fire(fire, wind, fuel, humidity, cfg, rng)
    # Only the original cell should be on fire
    assert int(new_mask.sum()) == 1


def test_spread_deterministic_with_seed() -> None:
    """Same seed must produce identical fire spread."""
    gs = 10
    fire = np.zeros((gs, gs), dtype=np.float32)
    fire[3, 3] = 1.0
    wind = np.ones((gs, gs), dtype=np.float32) * 0.6
    fuel = np.ones((gs, gs), dtype=np.float32) * 0.8
    humidity = np.ones((gs, gs), dtype=np.float32) * 0.2
    cfg = FirePropagationConfig()
    mask1 = propagate_fire(fire, wind, fuel, humidity, cfg, np.random.default_rng(99))
    mask2 = propagate_fire(fire, wind, fuel, humidity, cfg, np.random.default_rng(99))
    np.testing.assert_array_equal(mask1, mask2)


def test_spread_stochastic_between_seeds() -> None:
    """Different seeds should (with high probability) produce different results."""
    gs = 20
    fire = np.zeros((gs, gs), dtype=np.float32)
    fire[10, 10] = 1.0
    wind = np.ones((gs, gs), dtype=np.float32) * 0.7
    fuel = np.ones((gs, gs), dtype=np.float32) * 0.9
    humidity = np.ones((gs, gs), dtype=np.float32) * 0.1
    cfg = FirePropagationConfig()
    results = set()
    for s in range(10):
        m = propagate_fire(fire, wind, fuel, humidity, cfg, np.random.default_rng(s))
        results.add(int(m.sum()))
    assert len(results) > 1, "Expected some variation across seeds"


def test_shape_mismatch_raises() -> None:
    """Mismatched array shapes must raise ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_spread_probability(
            np.zeros((10, 10), dtype=np.float32),
            np.zeros((20, 20), dtype=np.float32),
            np.zeros((10, 10), dtype=np.float32),
            FirePropagationConfig(),
        )


def test_initialise_fire_count() -> None:
    """initialise_fire must place exactly n_ignition_points fire cells."""
    mask = initialise_fire(50, 3, np.random.default_rng(7))
    assert int(mask.sum()) == 3


def test_fire_spread_only_from_burning() -> None:
    """New ignitions can only occur adjacent to already-burning cells."""
    gs = 10
    fire = np.zeros((gs, gs), dtype=np.float32)
    fire[0, 0] = 1.0  # Single corner cell
    wind = np.ones_like(fire) * 0.8
    fuel = np.ones_like(fire)
    humidity = np.zeros_like(fire)
    cfg = FirePropagationConfig()
    new_fire = propagate_fire(fire, wind, fuel, humidity, cfg, np.random.default_rng(0))
    # Only (0,0) and its immediate neighbours can be burning
    expected_region = np.zeros_like(fire)
    expected_region[0:2, 0:2] = 1.0
    illegal = new_fire * (1.0 - expected_region)
    assert float(illegal.sum()) == 0.0
