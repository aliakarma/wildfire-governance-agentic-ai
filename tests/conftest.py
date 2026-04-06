"""Shared pytest fixtures for unit, integration, and smoke tests.

All fixtures use synthetic data only — no real datasets required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wildfire_governance.agents.uav_agent import UAVAgent
from wildfire_governance.blockchain.audit_log import ImmutableAuditLog
from wildfire_governance.blockchain.consensus import PBFTConsensus
from wildfire_governance.blockchain.crypto_utils import generate_key_pair
from wildfire_governance.blockchain.smart_contract import GovernanceSmartContract
from wildfire_governance.blockchain.transaction import build_transaction
from wildfire_governance.decision.belief_state import BeliefState
from wildfire_governance.gomdp.definition import GovernanceInvariantMDP
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.governance.hitl_interface import HITLAuthorisationGate
from wildfire_governance.governance.oracle_model import HumanOperatorOracle
from wildfire_governance.simulation.fire_propagation import FirePropagationConfig
from wildfire_governance.simulation.grid_environment import (
    EnvironmentConfig,
    WildfireGridEnvironment,
)
from wildfire_governance.verification.confidence_scorer import TwoStageConfidenceScorer
from wildfire_governance.verification.fusion import CrossModalFusion


# ---- RNG ----------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible randomness in tests."""
    return np.random.default_rng(42)


# ---- Config -------------------------------------------------------------

@pytest.fixture
def small_grid_config() -> EnvironmentConfig:
    """10x10 grid, 100 timesteps — fast for unit tests."""
    return EnvironmentConfig(grid_size=10, n_timesteps=100, n_ignition_points=1)


@pytest.fixture
def medium_grid_config() -> EnvironmentConfig:
    """20x20 grid, 200 timesteps — for integration tests."""
    return EnvironmentConfig(grid_size=20, n_timesteps=200, n_ignition_points=2)


@pytest.fixture
def fire_config() -> FirePropagationConfig:
    """Default fire propagation config."""
    return FirePropagationConfig()


# ---- Simulation ---------------------------------------------------------

@pytest.fixture
def small_env(small_grid_config: EnvironmentConfig) -> WildfireGridEnvironment:
    """Small 10x10 wildfire environment, reset with seed 42."""
    env = WildfireGridEnvironment(small_grid_config)
    env.reset(seed=42)
    return env


@pytest.fixture
def small_heat_map(small_env: WildfireGridEnvironment) -> np.ndarray:
    """Current heat map from the small environment."""
    return small_env.heat_map.copy()


@pytest.fixture
def small_fire_mask(small_env: WildfireGridEnvironment) -> np.ndarray:
    """Current fire mask from the small environment."""
    return small_env.fire_mask.copy()


# ---- UAV agents ---------------------------------------------------------

@pytest.fixture
def single_uav() -> UAVAgent:
    """A single UAVAgent at position (5, 5) on a 10x10 grid."""
    return UAVAgent(agent_id="uav_0", initial_position=(5, 5), grid_size=10)


@pytest.fixture
def uav_fleet_5() -> list:
    """Fleet of 5 UAVs placed on a 10x10 grid."""
    return [
        UAVAgent(agent_id=f"uav_{i}", initial_position=(i * 2, i * 2), grid_size=10)
        for i in range(5)
    ]


# ---- Decision -----------------------------------------------------------

@pytest.fixture
def belief_state_10x10() -> BeliefState:
    """BeliefState for a 10x10 grid."""
    return BeliefState(grid_size=10)


# ---- Verification -------------------------------------------------------

@pytest.fixture
def fusion() -> CrossModalFusion:
    """Default cross-modal fusion with paper weights (w_H=0.65, w_W=0.35)."""
    return CrossModalFusion(w_h=0.65, w_w=0.35)


@pytest.fixture
def scorer() -> TwoStageConfidenceScorer:
    """Two-stage confidence scorer with paper thresholds."""
    return TwoStageConfidenceScorer(tau1=0.60, tau2=0.80)


# ---- Blockchain ---------------------------------------------------------

@pytest.fixture
def key_pair() -> tuple:
    """Ed25519 private/public key pair for signing tests."""
    return generate_key_pair()


@pytest.fixture
def consensus() -> PBFTConsensus:
    """PBFTConsensus with k=7, f=2, seeded for determinism."""
    return PBFTConsensus(
        n_validators=7, max_byzantine=2,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def audit_log() -> ImmutableAuditLog:
    """Empty immutable audit log."""
    return ImmutableAuditLog()


@pytest.fixture
def smart_contract(consensus: PBFTConsensus, audit_log: ImmutableAuditLog) -> GovernanceSmartContract:
    """GovernanceSmartContract with tau=0.80."""
    return GovernanceSmartContract(tau=0.80, consensus=consensus, audit_log=audit_log)


@pytest.fixture
def sample_transaction(key_pair: tuple):
    """Pre-built AnomalyTransaction with valid hash for smart contract tests."""
    return build_transaction(
        event_id="test_evt_001",
        geo_boundary=(10, 10, 11, 11),
        confidence_score=0.87,
        sensor_readings={"heat": 0.87, "weather": 0.65},
    )


# ---- GOMDP --------------------------------------------------------------

@pytest.fixture
def gomdp() -> GovernanceInvariantMDP:
    """GovernanceInvariantMDP with tau=0.80."""
    return GovernanceInvariantMDP(tau=0.80)


@pytest.fixture
def invariant_checker() -> GovernanceInvariantChecker:
    """GovernanceInvariantChecker with tau=0.80."""
    return GovernanceInvariantChecker(tau=0.80)


# ---- HITL ---------------------------------------------------------------

@pytest.fixture
def always_approve_oracle(rng: np.random.Generator) -> HumanOperatorOracle:
    """Oracle that always approves (rejection_rate=0, low threshold)."""
    return HumanOperatorOracle(
        rejection_rate=0.0, approval_threshold=0.0, rng=rng
    )


@pytest.fixture
def hitl_gate(always_approve_oracle: HumanOperatorOracle, rng: np.random.Generator) -> HITLAuthorisationGate:
    """HITL gate backed by the always-approve oracle."""
    return HITLAuthorisationGate(oracle=always_approve_oracle, rng=rng)
