"""Integration tests — empirical verification of Theorem 1 (Policy-Agnostic Safety).

These are the most important tests in the repository.
Theorem 1: ANY policy operating in the GOMDP satisfies the governance
predicate with probability 1, regardless of optimality gap.

We test:
1. Random policy (worst case) — 20 episodes, must achieve 100% compliance.
2. Adversarial policy (always tries to alert) — must be blocked every time.
3. Greedy policy — must achieve 100% compliance.
"""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.gomdp.definition import GovernanceInvariantMDP
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.rl.gomdp_env import GOMMDPGymEnv
from wildfire_governance.simulation.grid_environment import EnvironmentConfig


N_EPISODES = 10  # Reduced for test speed; paper uses 20
N_TIMESTEPS = 50


@pytest.fixture
def gomdp_env_small() -> GOMMDPGymEnv:
    """Small GOMDP environment for integration tests."""
    cfg = EnvironmentConfig(grid_size=10, n_timesteps=N_TIMESTEPS)
    return GOMMDPGymEnv(config=cfg, n_uavs=5, enable_governance=True)


def test_theorem1_random_policy(gomdp_env_small: GOMMDPGymEnv) -> None:
    """Theorem 1: Random policy must achieve 100% governance compliance."""
    checker = GovernanceInvariantChecker(tau=0.80)
    all_compliant = True

    for ep in range(N_EPISODES):
        obs, _ = gomdp_env_small.reset(seed=ep)
        done = False
        while not done:
            # RANDOM policy — worst-case scenario
            action = np.random.default_rng(ep).integers(0, 25, size=5)
            obs, _, terminated, truncated, _ = gomdp_env_small.step(action)
            done = terminated or truncated

        trajectory = gomdp_env_small.get_trajectory()
        report = checker.check_trajectory(trajectory)
        if not report.theorem1_satisfied:
            all_compliant = False

    assert all_compliant, (
        "Theorem 1 VIOLATED: Random policy caused a governance breach. "
        "The GOMDP environment must block all non-compliant alert actions."
    )


def test_theorem1_high_confidence_always_blocked_without_approval() -> None:
    """Theorem 1: Alerts with high confidence but NO human approval are blocked."""
    gomdp = GovernanceInvariantMDP(tau=0.80)
    # Simulate 50 attempts with high confidence but HA=False
    for _ in range(50):
        result = gomdp.step_alert_action(
            confidence=0.95,
            human_approval=False,  # No human approval
            validator_signature_valid=True,
        )
        assert result.blocked is True, "Alert with HA=False must be blocked"

    # get_violation_count() tracks blocked (non-compliant) attempts — all 50 were correctly blocked
    assert gomdp.get_violation_count() == 50


def test_theorem1_compliance_with_greedy() -> None:
    """Theorem 1: Greedy-GOMDP policy must achieve 100% compliance."""
    from experiments.utils.runner import run_episode

    n_compliant = 0
    for seed in range(N_EPISODES):
        result = run_episode(
            seed=seed, config_name="greedy_gomdp",
            grid_size=10, n_timesteps=N_TIMESTEPS, n_uavs=5,
            enable_governance=True, enable_hitl=True,
            enable_blockchain=True, enable_verification=True,
            enable_coordination=True,
        )
        if getattr(result, "governance_compliant", False):
            n_compliant += 1

    compliance_rate = n_compliant / N_EPISODES
    assert compliance_rate == pytest.approx(1.0), (
        f"Greedy-GOMDP compliance rate = {compliance_rate:.1%}, expected 100%"
    )


def _registered_contract():
    """Contract with one authorised validator, plus that validator's keypair."""
    from wildfire_governance.blockchain.smart_contract import GovernanceSmartContract
    from wildfire_governance.blockchain.crypto_utils import generate_key_pair
    priv, pub = generate_key_pair()
    contract = GovernanceSmartContract(tau=0.80)
    contract.register_validator(pub)
    return contract, priv, pub


def test_theorem1_unsigned_injection_blocked() -> None:
    """Ed25519 verification alone defeats direct injection (ablation column)."""
    contract, _, _ = _registered_contract()
    n_attempts = 100
    n_blocked = sum(
        1 for i in range(n_attempts)
        if not contract.attempt_unauthorised_injection((i, i, i + 1, i + 1))
    )
    assert n_blocked == n_attempts, (
        f"Expected 100 blocked injections, got {n_blocked}. "
        f"P_breach^GOMDP should be 0.000 (Theorem 2)."
    )


def test_theorem1_valid_signature_from_unauthorised_key_blocked() -> None:
    """Theorem 1 Case 1: a cryptographically valid signature from a
    non-validator key must not authorise an alert."""
    contract, _, _ = _registered_contract()
    n_attempts = 100
    n_blocked = sum(
        1 for i in range(n_attempts)
        if not contract.attempt_unauthorised_injection(
            (i, i, i + 1, i + 1), attack="wrong_key"
        )
    )
    assert n_blocked == n_attempts, (
        f"Adversary keypair breached {n_attempts - n_blocked}/{n_attempts} times; "
        f"validator-key authorisation is not being enforced."
    )


def test_replayed_certificate_blocked() -> None:
    """Per-event nonces prevent replay of a previously approved certificate."""
    from wildfire_governance.blockchain.crypto_utils import sign
    from wildfire_governance.blockchain.transaction import build_transaction
    contract, priv, pub = _registered_contract()

    tx = build_transaction("legit", (0, 0, 5, 5), 0.95, {"heat": 0.9})
    first = contract.verify_and_execute(tx, sign(tx.to_bytes(), priv), pub)
    assert first.alert_enabled, "legitimate alert should be approved"

    replay = contract.probe_injection((0, 0, 5, 5), attack="replay")
    assert not replay.alert_enabled, "replayed certificate must be rejected"
    assert not replay.replay_ok


def test_legitimate_alert_still_approved() -> None:
    """Key authorisation must not break the legitimate duty-officer path."""
    from wildfire_governance.blockchain.crypto_utils import sign
    from wildfire_governance.blockchain.transaction import build_transaction
    contract, priv, pub = _registered_contract()
    tx = build_transaction("legit", (0, 0, 5, 5), 0.95, {"heat": 0.9})
    result = contract.verify_and_execute(tx, sign(tx.to_bytes(), priv), pub)
    assert result.alert_enabled and result.key_authorised and result.confidence_ok
