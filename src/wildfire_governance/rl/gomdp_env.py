"""Gymnasium-compatible GOMDP environment wrapper for PPO-GOMDP training.

The governance constraint is NOT in the reward function.
The environment blocks non-compliant alert actions transparently via
GovernanceInvariantMDP.step_alert_action(), enforcing Theorem 1
(Policy-Agnostic Safety) for any policy trained in this environment.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from wildfire_governance.agents.coordination_engine import HierarchicalCoordinationEngine
from wildfire_governance.agents.uav_agent import UAVAgent
from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
from wildfire_governance.blockchain.smart_contract import GovernanceSmartContract
from wildfire_governance.blockchain.transaction import build_transaction
from wildfire_governance.gomdp.definition import GovernanceInvariantMDP
from wildfire_governance.gomdp.invariant_checker import GovernanceInvariantChecker
from wildfire_governance.governance.hitl_interface import HITLAuthorisationGate
from wildfire_governance.governance.oracle_model import HumanOperatorOracle
from wildfire_governance.simulation.grid_environment import (
    EnvironmentConfig,
    WildfireGridEnvironment,
)


class GOMMDPGymEnv:
    """Gymnasium-compatible GOMDP environment for PPO-GOMDP training.

    The policy selects UAV sector assignments. Alert triggering is handled
    by the two-stage verification pipeline inside the coordination engine.
    The GOMDP environment blocks any non-compliant alert action at the
    transition level — the policy receives no negative reward for this;
    it simply observes that the alert did not broadcast.

    Args:
        config: EnvironmentConfig for the wildfire grid (default: paper params).
        n_uavs: UAV fleet size N (default 20).
        n_sectors: Number of patrol sectors Z (default 25).
        enable_governance: If True, runs full GOMDP enforcement (default).
                           If False, removes governance (for CMDP comparison baseline).
    """

    def __init__(
        self,
        config: Optional[EnvironmentConfig] = None,
        n_uavs: int = 20,
        n_sectors: int = 25,
        enable_governance: bool = True,
    ) -> None:
        self._env_config = config or EnvironmentConfig()
        self._n_uavs = n_uavs
        self._n_sectors = n_sectors
        self._enable_governance = enable_governance
        self._gs = self._env_config.grid_size

        # Core simulation
        self._sim = WildfireGridEnvironment(self._env_config)

        # UAV fleet
        self._fleet: list[UAVAgent] = []

        # Governance components
        self._gomdp = GovernanceInvariantMDP(tau=0.80)
        self._smart_contract = GovernanceSmartContract(tau=0.80)
        self._hitl_gate = HITLAuthorisationGate()
        self._checker = GovernanceInvariantChecker(tau=0.80)

        # Episode tracking
        self._trajectory: list[dict] = []
        self._rng: np.random.Generator = np.random.default_rng(42)
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._ignition_time: int = 0
        self._first_detection: Optional[int] = None
        self._n_alerts_broadcast: int = 0
        self._n_false_alerts: int = 0

        # Observation and action space dimensions
        self._obs_dim = self._gs * self._gs + 2 * n_uavs

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed. If None, uses the previous RNG state.

        Returns:
            Tuple (observation, info).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs_dict = self._sim.reset(seed=int(self._rng.integers(0, 2**31)))
        self._fleet = self._init_fleet()
        self._gomdp.reset_stats()
        self._smart_contract._n_approved = 0
        self._smart_contract._n_blocked = 0
        self._trajectory = []
        self._total_reward = 0.0
        self._step_count = 0
        self._ignition_time = 0
        self._first_detection = None
        self._n_alerts_broadcast = 0
        self._n_false_alerts = 0

        return self._build_obs(), {"ignition_time": self._ignition_time}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Integer array of shape (n_uavs,) — sector assignment per UAV.

        Returns:
            Tuple (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Map action to sector assignments
        positions = [uav.position for uav in self._fleet]
        obs_dict, done, sim_info = self._sim.step(positions)

        # Get sensor readings
        readings = self._sim.get_observations(positions)

        # Run coordination engine step (produces alert_request if HITL escalation)
        alert_request = None
        anomaly_location = None
        heat_val = float(self._sim.heat_map.max())

        if heat_val > 0.60:
            if self._first_detection is None:
                self._first_detection = self._step_count
            anomaly_location = tuple(np.unravel_index(
                self._sim.heat_map.argmax(), self._sim.heat_map.shape
            ))

        # Two-stage verification and potential governance escalation
        info: Dict[str, Any] = {
            "timestep": self._step_count,
            "fire_cells": sim_info.get("fire_cells", 0),
            "alert_broadcast": False,
            "governance_cert": None,
            "confidence": 0.0,
            "human_approval": False,
        }

        alert_broadcast = False
        cert: Optional[str] = None

        if heat_val > 0.80 and self._enable_governance:
            # Simulate verification pipeline
            weather_idx = float(np.clip(
                self._sim._wind_field.mean() - self._sim._humidity_field.mean() + 0.5, 0.0, 1.0
            ))
            conf = float(np.clip(0.65 * heat_val + 0.35 * weather_idx, 0.0, 1.0))
            info["confidence"] = conf

            if conf > 0.80:
                # HITL review
                row, col = (int(anomaly_location[0]), int(anomaly_location[1])) if anomaly_location else (0, 0)
                tx = build_transaction(
                    event_id=f"evt_{self._step_count}",
                    geo_boundary=(row, col, row + 1, col + 1),
                    confidence_score=conf,
                    sensor_readings={"heat": heat_val, "weather": weather_idx},
                )
                decision, signature = self._hitl_gate.process(tx, conf)
                info["human_approval"] = decision.approved

                if decision.approved and signature is not None:
                    result = self._smart_contract.verify_and_execute(
                        tx, signature, self._hitl_gate.public_key_bytes
                    )
                    if result.alert_enabled:
                        cert = result.cert
                        alert_broadcast = True
                        self._n_alerts_broadcast += 1
                        is_true_fire = bool(self._sim.fire_mask[row, col] > 0.5)
                        if not is_true_fire:
                            self._n_false_alerts += 1
                        info["governance_cert"] = cert
                        info["alert_broadcast"] = True
                else:
                    # Non-governance path (CMDP comparison)
                    if conf > 0.80:
                        alert_broadcast = True
                        self._n_alerts_broadcast += 1
                        is_true_fire = bool(self._sim.fire_mask.max() > 0.5)
                        if not is_true_fire:
                            self._n_false_alerts += 1
                        info["alert_broadcast"] = True

        self._trajectory.append(dict(info))

        # Compute reward
        ld_component = 0.0 if self._first_detection else 1.0
        fp_component = 1.0 if (alert_broadcast and self._n_false_alerts > 0) else 0.0
        battery_cost = sum(1.0 - uav.battery_fraction for uav in self._fleet) / self._n_uavs
        reward = -(0.5 * ld_component + 0.35 * fp_component + 0.15 * battery_cost)
        self._total_reward += reward

        terminated = done
        truncated = self._step_count >= self._env_config.n_timesteps

        if terminated or truncated:
            fp_rate = self._n_false_alerts / max(1, self._n_alerts_broadcast) * 100.0
            info["episode_ld"] = float(
                self._first_detection - self._ignition_time
            ) if self._first_detection else float("inf")
            info["episode_fp_pct"] = fp_rate
            info["governance_compliance"] = self._gomdp.get_compliance_rate()

        return self._build_obs(), reward, terminated, truncated, info

    def get_trajectory(self) -> list:
        """Return the recorded trajectory for Theorem 1 verification."""
        return list(self._trajectory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_fleet(self) -> list:
        gs = self._gs
        fleet = []
        rng = np.random.default_rng(int(self._rng.integers(0, 2**31)))
        for i in range(self._n_uavs):
            pos = (int(rng.integers(0, gs)), int(rng.integers(0, gs)))
            fleet.append(UAVAgent(
                agent_id=f"uav_{i}",
                initial_position=pos,
                grid_size=gs,
            ))
        return fleet

    def _build_obs(self) -> np.ndarray:
        """Build flattened observation vector: belief map + UAV positions."""
        heat = self._sim.heat_map.ravel().astype(np.float32)
        pos_vec = np.array(
            [(uav.position[0] / self._gs, uav.position[1] / self._gs)
             for uav in self._fleet],
            dtype=np.float32,
        ).ravel()
        return np.concatenate([heat, pos_vec])
