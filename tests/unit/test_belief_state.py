"""Unit tests for belief_state.py."""
import pytest
from wildfire_governance.decision.belief_state import BeliefState
from wildfire_governance.simulation.sensor_models import SensorReading

def _r(row, col, detected):
    return SensorReading((row,col), 0.9 if detected else 0.1, 0.85, detected, "thermal_uav")

def test_initial_belief_uniform(belief_state_10x10):
    b = belief_state_10x10.get_belief(); assert b.shape == (10,10)
    assert float(b.std()) < 1e-6

def test_positive_detection_increases_belief(belief_state_10x10):
    prior = float(belief_state_10x10.get_belief()[3,3])
    belief_state_10x10.update([_r(3,3,True)])
    assert float(belief_state_10x10.get_belief()[3,3]) > prior

def test_belief_values_bounded(belief_state_10x10):
    for i in range(10): belief_state_10x10.update([_r(i%10, i%10, i%2==0)])
    b = belief_state_10x10.get_belief()
    assert float(b.min()) > 0.0; assert float(b.max()) < 1.0

def test_entropy_non_negative(belief_state_10x10):
    assert belief_state_10x10.entropy() >= 0.0
