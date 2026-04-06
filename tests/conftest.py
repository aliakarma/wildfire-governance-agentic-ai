"""Shared pytest fixtures."""
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
from wildfire_governance.simulation.grid_environment import EnvironmentConfig, WildfireGridEnvironment
from wildfire_governance.verification.confidence_scorer import TwoStageConfidenceScorer
from wildfire_governance.verification.fusion import CrossModalFusion

@pytest.fixture
def rng(): return np.random.default_rng(42)

@pytest.fixture
def small_grid_config(): return EnvironmentConfig(grid_size=10, n_timesteps=100, n_ignition_points=1)

@pytest.fixture
def small_env(small_grid_config):
    env = WildfireGridEnvironment(small_grid_config); env.reset(seed=42); return env

@pytest.fixture
def single_uav(): return UAVAgent(agent_id="uav_0", initial_position=(5,5), grid_size=10)

@pytest.fixture
def belief_state_10x10(): return BeliefState(grid_size=10)

@pytest.fixture
def fusion(): return CrossModalFusion(w_h=0.65, w_w=0.35)

@pytest.fixture
def scorer(): return TwoStageConfidenceScorer(tau1=0.60, tau2=0.80)

@pytest.fixture
def key_pair(): return generate_key_pair()

@pytest.fixture
def consensus(): return PBFTConsensus(n_validators=7, max_byzantine=2, rng=np.random.default_rng(42))

@pytest.fixture
def audit_log(): return ImmutableAuditLog()

@pytest.fixture
def smart_contract(consensus, audit_log): return GovernanceSmartContract(tau=0.80, consensus=consensus, audit_log=audit_log)

@pytest.fixture
def sample_transaction(): return build_transaction("test_evt_001", (10,10,11,11), 0.87, {"heat": 0.87})

@pytest.fixture
def gomdp(): return GovernanceInvariantMDP(tau=0.80)

@pytest.fixture
def invariant_checker(): return GovernanceInvariantChecker(tau=0.80)

@pytest.fixture
def always_approve_oracle(rng): return HumanOperatorOracle(rejection_rate=0.0, approval_threshold=0.0, rng=rng)

@pytest.fixture
def hitl_gate(always_approve_oracle, rng): return HITLAuthorisationGate(oracle=always_approve_oracle, rng=rng)
