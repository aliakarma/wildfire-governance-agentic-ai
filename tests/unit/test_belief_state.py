"""Unit tests for decision/belief_state.py."""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.decision.belief_state import BeliefState
from wildfire_governance.simulation.sensor_models import SensorReading


def _make_reading(row: int, col: int, detected: bool) -> SensorReading:
    return SensorReading(
        position=(row, col),
        heat_value=0.9 if detected else 0.1,
        confidence=0.85,
        is_fire_detected=detected,
        sensor_type="thermal_uav",
    )


def test_initial_belief_uniform(belief_state_10x10: BeliefState) -> None:
    """Initial belief must be uniform (prior_fire_prob everywhere)."""
    b = belief_state_10x10.get_belief()
    assert b.shape == (10, 10)
    assert float(b.std()) < 1e-6


def test_positive_detection_increases_belief(belief_state_10x10: BeliefState) -> None:
    """A positive detection must increase the belief at that cell."""
    prior = float(belief_state_10x10.get_belief()[3, 3])
    belief_state_10x10.update([_make_reading(3, 3, True)])
    posterior = float(belief_state_10x10.get_belief()[3, 3])
    assert posterior > prior


def test_negative_detection_decreases_belief(belief_state_10x10: BeliefState) -> None:
    """A negative detection must decrease the belief at that cell."""
    # First raise the belief with a positive observation
    belief_state_10x10.update([_make_reading(2, 2, True)])
    elevated = float(belief_state_10x10.get_belief()[2, 2])
    belief_state_10x10.update([_make_reading(2, 2, False)])
    reduced = float(belief_state_10x10.get_belief()[2, 2])
    assert reduced < elevated


def test_belief_values_bounded(belief_state_10x10: BeliefState) -> None:
    """Belief values must stay in (0, 1) after multiple updates."""
    for i in range(10):
        belief_state_10x10.update([_make_reading(i % 10, i % 10, i % 2 == 0)])
    b = belief_state_10x10.get_belief()
    assert float(b.min()) > 0.0
    assert float(b.max()) < 1.0


def test_risk_map_shape(belief_state_10x10: BeliefState) -> None:
    """get_risk_map must return an array of the correct shape."""
    r = belief_state_10x10.get_risk_map()
    assert r.shape == (10, 10)


def test_entropy_non_negative(belief_state_10x10: BeliefState) -> None:
    """Shannon entropy must be non-negative."""
    assert belief_state_10x10.entropy() >= 0.0


def test_reset_restores_uniform(belief_state_10x10: BeliefState) -> None:
    """reset() must restore the belief to a uniform prior."""
    belief_state_10x10.update([_make_reading(5, 5, True)] * 5)
    belief_state_10x10.reset()
    b = belief_state_10x10.get_belief()
    assert float(b.std()) < 1e-3


def test_out_of_bounds_observation_ignored(belief_state_10x10: BeliefState) -> None:
    """Observations outside the grid must be silently ignored."""
    prior = belief_state_10x10.get_belief().copy()
    belief_state_10x10.update([_make_reading(50, 50, True)])  # Out of bounds
    posterior = belief_state_10x10.get_belief()
    # The belief should only differ due to temporal decay, not position update
    assert float(abs(posterior - prior).max()) < 0.1
