"""Unit tests for fire_propagation.py."""
import numpy as np
import pytest
from wildfire_governance.simulation.fire_propagation import FirePropagationConfig, compute_spread_probability, initialise_fire, propagate_fire

def test_spread_probability_range():
    rng = np.random.default_rng(0); gs = 20; cfg = FirePropagationConfig()
    w = rng.uniform(0,1,(gs,gs)).astype(np.float32); f = rng.uniform(0,1,(gs,gs)).astype(np.float32)
    h = rng.uniform(0,1,(gs,gs)).astype(np.float32); p = compute_spread_probability(w,f,h,cfg)
    assert p.min() >= 0.0; assert p.max() <= 1.0

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_spread_probability(np.zeros((10,10),dtype=np.float32), np.zeros((20,20),dtype=np.float32),
                                   np.zeros((10,10),dtype=np.float32), FirePropagationConfig())

def test_initialise_fire_count():
    mask = initialise_fire(50, 3, np.random.default_rng(7)); assert int(mask.sum()) == 3

def test_spread_deterministic_with_seed():
    gs = 10; fire = np.zeros((gs,gs), dtype=np.float32); fire[3,3] = 1.0
    wind = np.ones((gs,gs), dtype=np.float32)*0.6; fuel = np.ones_like(wind)*0.8
    hum = np.ones_like(wind)*0.2; cfg = FirePropagationConfig()
    m1 = propagate_fire(fire,wind,fuel,hum,cfg,np.random.default_rng(99))
    m2 = propagate_fire(fire,wind,fuel,hum,cfg,np.random.default_rng(99))
    np.testing.assert_array_equal(m1, m2)
