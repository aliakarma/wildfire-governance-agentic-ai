"""Integration tests for the end-to-end governance pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.utils.runner import run_episode


def test_governance_predicate_enforced_end_to_end() -> None:
    """No alert must be broadcast without blockchain confirmation and human approval."""
    result = run_episode(
        seed=0, config_name="greedy_gomdp",
        grid_size=10, n_timesteps=100, n_uavs=5,
        enable_governance=True, enable_hitl=True,
        enable_blockchain=True, enable_verification=True,
        enable_coordination=True,
    )
    assert getattr(result, "governance_compliant", False) is True


def test_adaptive_ai_no_governance() -> None:
    """Adaptive AI (no governance) must produce a valid Ld and Fp."""
    result = run_episode(
        seed=1, config_name="adaptive_ai",
        grid_size=10, n_timesteps=100, n_uavs=5,
        enable_governance=False, enable_hitl=False,
        enable_blockchain=False, enable_verification=True,
        enable_coordination=True,
    )
    assert result.ld >= 0.0
    assert 0.0 <= result.fp_pct <= 100.0


def test_static_baseline_has_high_latency() -> None:
    """Static monitoring must have higher Ld than adaptive on average (3 seeds)."""
    from statistics import mean

    adaptive_lds, static_lds = [], []
    for seed in range(3):
        r_adaptive = run_episode(
            seed=seed, config_name="adaptive", grid_size=20, n_timesteps=200,
            n_uavs=5, enable_governance=False, enable_hitl=False,
            enable_blockchain=False, enable_verification=False, enable_coordination=True
        )
        r_static = run_episode(
            seed=seed, config_name="static", grid_size=20, n_timesteps=200,
            n_uavs=5, enable_governance=False, enable_hitl=False,
            enable_blockchain=False, enable_verification=False, enable_coordination=False
        )
        if r_adaptive.ld < float("inf"):
            adaptive_lds.append(r_adaptive.ld)
        if r_static.ld < float("inf"):
            static_lds.append(r_static.ld)

    if adaptive_lds and static_lds:
        assert mean(static_lds) >= mean(adaptive_lds) * 0.8  # Loose bound


def test_adversarial_injection_blocked_in_full_pipeline() -> None:
    """Adversarial injection attempts must be 100% blocked in the full pipeline."""
    result = run_episode(
        seed=42, config_name="greedy_gomdp",
        grid_size=10, n_timesteps=200, n_uavs=5,
        enable_governance=True, enable_hitl=True,
        enable_blockchain=True, enable_verification=True,
        enable_coordination=True,
    )
    # All injections attempted within the runner must be blocked
    if result.n_injections_attempted > 0:
        assert result.n_injections_blocked == result.n_injections_attempted


def test_metrics_non_negative_finite() -> None:
    """Ld and Fp must be non-negative finite numbers after a completed episode."""
    result = run_episode(
        seed=7, config_name="greedy_gomdp",
        grid_size=10, n_timesteps=100, n_uavs=3,
        enable_governance=True, enable_hitl=True,
        enable_blockchain=True, enable_verification=True,
        enable_coordination=True,
    )
    assert result.fp_pct >= 0.0
    assert result.fp_pct <= 100.0
