"""Integration tests for the end-to-end governance pipeline."""
import numpy as np, pytest
from experiments.utils.runner import run_episode

def test_governance_predicate_enforced_end_to_end():
    result = run_episode(seed=0, config_name="greedy_gomdp", grid_size=10, n_timesteps=100,
        n_uavs=5, enable_governance=True, enable_hitl=True, enable_blockchain=True,
        enable_verification=True, enable_coordination=True)
    assert result.governance_compliant is True

def test_adaptive_ai_no_governance():
    result = run_episode(seed=1, config_name="adaptive_ai", grid_size=10, n_timesteps=100,
        n_uavs=5, enable_governance=False, enable_hitl=False, enable_blockchain=False,
        enable_verification=True, enable_coordination=True)
    assert result.ld >= 0.0; assert 0.0 <= result.fp_pct <= 100.0

def test_adversarial_injection_blocked_in_full_pipeline():
    result = run_episode(seed=42, config_name="greedy_gomdp", grid_size=10, n_timesteps=200,
        n_uavs=5, enable_governance=True, enable_hitl=True, enable_blockchain=True,
        enable_verification=True, enable_coordination=True)
    if result.n_injections_attempted > 0:
        assert result.n_injections_blocked == result.n_injections_attempted
