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
        rng = np.random.default_rng(s)
        m = fire
        # The calibrated model spreads at ~0.005 per neighbour per step, so a
        # single step from one cell almost always yields zero new ignitions and
        # every seed returns the same count. Integrate over enough steps for the
        # stochastic difference between seeds to actually show up.
        for _ in range(200):
            m = propagate_fire(m, wind, fuel, humidity, cfg, rng)
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


def test_spread_rate_calibrated_to_manuscript() -> None:
    """Spread must match the manuscript's stated 1-4 cells per 10-step window.

    Regression guard for the missing logistic intercept: without ``alpha0`` the
    weighted term alone puts P_spread near 0.5, which saturates a 100x100 grid
    in ~300 steps (roughly 450 cells per 10-step window, two orders of magnitude
    off the stated calibration) and drives the false-alert rate to zero because
    every cell is genuinely on fire.
    """
    from wildfire_governance.simulation.grid_environment import (
        EnvironmentConfig,
        WildfireGridEnvironment,
    )

    rates = []
    for seed in range(5):
        env = WildfireGridEnvironment(
            EnvironmentConfig(grid_size=100, n_timesteps=400, ignition_delay_range=(1, 1))
        )
        env.reset(seed=seed)
        counts = []
        for _ in range(200):
            _, _, info = env.step([(0, 0)])
            counts.append(info["fire_cells"])
        windows = [counts[i + 10] - counts[i] for i in range(0, 100, 10)]
        rates.append(sum(windows) / len(windows))

    mean_rate = sum(rates) / len(rates)
    assert 1.0 <= mean_rate <= 4.0, (
        f"Mean spread rate {mean_rate:.2f} cells/10-step is outside the "
        f"manuscript's stated 1-4 band; per-seed rates: {rates}"
    )


def test_fire_does_not_saturate_grid() -> None:
    """A 100x100 grid must not be fully consumed within a standard episode."""
    from wildfire_governance.simulation.grid_environment import (
        EnvironmentConfig,
        WildfireGridEnvironment,
    )

    env = WildfireGridEnvironment(
        EnvironmentConfig(grid_size=100, n_timesteps=1500, ignition_delay_range=(1, 1))
    )
    env.reset(seed=0)
    for _ in range(1500):
        _, _, info = env.step([(0, 0)])
    assert info["fire_cells"] < 10000, "Grid fully saturated; F_p is meaningless"
