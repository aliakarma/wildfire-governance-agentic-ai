"""Smoke test — runs the full pipeline on a tiny grid.

Must complete in < 60 seconds on any machine.
Uses only synthetic data; no GPU, no real datasets, no network access.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


SMOKE_GRID = 10
SMOKE_STEPS = 20
SMOKE_UAVS = 3
MAX_SECONDS = 60


def test_smoke_full_pipeline_completes() -> None:
    """End-to-end pipeline on 10×10 grid, 20 steps must finish in 60s."""
    from experiments.utils.runner import run_episode

    start = time.time()
    result = run_episode(
        seed=42, config_name="greedy_gomdp",
        grid_size=SMOKE_GRID, n_timesteps=SMOKE_STEPS, n_uavs=SMOKE_UAVS,
        enable_governance=True, enable_hitl=True,
        enable_blockchain=True, enable_verification=True,
        enable_coordination=True,
    )
    elapsed = time.time() - start

    assert elapsed < MAX_SECONDS, f"Smoke test took {elapsed:.1f}s (limit {MAX_SECONDS}s)"
    assert result.fp_pct >= 0.0
    assert result.fp_pct <= 100.0
    assert result.governance_compliant is True


def test_smoke_crypto_operations() -> None:
    """Cryptographic primitives (keygen, sign, verify) must work correctly."""
    from wildfire_governance.blockchain.crypto_utils import (
        generate_key_pair, sign, verify_signature, sha3_256_hash
    )
    priv, pub = generate_key_pair()
    data = b"smoke test payload"
    sig = sign(data, priv)
    assert verify_signature(data, sig, pub) is True
    assert verify_signature(b"wrong", sig, pub) is False
    assert len(sha3_256_hash(data)) == 64


def test_smoke_gomdp_invariant() -> None:
    """GOMDP invariant: blocked attempts must not broadcast alerts."""
    from wildfire_governance.gomdp.definition import GovernanceInvariantMDP

    gomdp = GovernanceInvariantMDP(tau=0.80)
    for _ in range(10):
        result = gomdp.step_alert_action(
            confidence=0.50,  # Below tau
            human_approval=True,
            validator_signature_valid=True,
        )
        assert result.blocked is True
        assert result.governance_cert is None


def test_smoke_fire_propagation() -> None:
    """Fire propagation must produce valid output shapes."""
    from wildfire_governance.simulation.grid_environment import (
        EnvironmentConfig, WildfireGridEnvironment
    )
    env = WildfireGridEnvironment(EnvironmentConfig(
        grid_size=SMOKE_GRID, n_timesteps=SMOKE_STEPS
    ))
    obs = env.reset(seed=0)
    assert obs["heat_map"].shape == (SMOKE_GRID, SMOKE_GRID)
    obs2, done, info = env.step([])
    assert obs2["heat_map"].shape == (SMOKE_GRID, SMOKE_GRID)
    assert isinstance(done, bool)


def test_smoke_breach_probability() -> None:
    """Theorem 2: GOMDP breach probability must be less than centralized."""
    from wildfire_governance.gomdp.breach_probability import paper_numerical_check
    result = paper_numerical_check()
    assert result["paper_claim_satisfied"] is True
    assert result["p_breach_gomdp"] < result["p_breach_central_direct_injection"]


def test_smoke_results_paper_csvs_exist() -> None:
    """Pre-committed paper result CSVs must all be present."""
    paper_dir = Path("results/paper")
    required = [
        "table2_rl_comparison.csv",
        "table3_main_comparison.csv",
        "table4_ablation.csv",
        "table5_adversarial.csv",
    ]
    for filename in required:
        path = paper_dir / filename
        assert path.exists(), f"Missing paper result file: {path}"


def test_smoke_configs_loadable() -> None:
    """All experiment YAML configs must load without errors."""
    from wildfire_governance.utils.config import load_config
    config_dir = Path("configs/experiments")
    for yaml_file in config_dir.glob("*.yaml"):
        cfg = load_config(yaml_file)
        assert cfg is not None
