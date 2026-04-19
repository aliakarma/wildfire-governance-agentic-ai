"""Shared single-seed episode runner used by all experiment scripts.

Provides a clean, consistent interface so every experiment script only
needs to call ``run_episode()`` rather than replicating the environment
setup loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from wildfire_governance.agents.uav_agent import UAVAgent
from wildfire_governance.blockchain.consensus import ByzantineFaultType, PBFTConsensus
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
from wildfire_governance.utils.reproducibility import set_global_seed


@dataclass
class EpisodeResult:
    """All metrics from one seed episode.

    Attributes:
        seed: Random seed used.
        config_name: Name of the configuration (e.g., ``"ppo_gomdp"``).
        ld: Detection latency (steps from ignition to detection).
        fp_pct: False public alert rate (%).
        bc_delay: Mean blockchain confirmation delay (steps).
        human_delay: Mean human review delay (steps).
        le2e: End-to-end latency.
        n_alerts: Total alerts broadcast.
        n_false: False alerts broadcast.
        governance_compliant: Whether all alerts had valid governance certs.
        n_injections_attempted: Adversarial injection attempts.
        n_injections_blocked: Injections successfully blocked.
        step_logs: Per-step trajectory records for post-hoc invariant checks.
        injection_success: 1 if any unauthorised injection breached, else 0.
    """

    seed: int
    config_name: str
    ld: float
    fp_pct: float
    bc_delay: float = 1.2
    human_delay: float = 3.0
    le2e: float = 0.0
    n_alerts: int = 0
    n_false: int = 0
    governance_compliant: bool = True
    n_injections_attempted: int = 0
    n_injections_blocked: int = 0
    step_logs: List[Dict[str, Any]] = field(default_factory=list)
    injection_success: int = 0

    def __post_init__(self) -> None:
        if self.le2e == 0.0:
            self.le2e = self.ld + self.bc_delay + self.human_delay


def run_episode(
    seed: int,
    config_name: str,
    grid_size: int = 100,
    n_timesteps: int = 3000,
    n_uavs: int = 20,
    enable_governance: bool = True,
    enable_hitl: bool = True,
    enable_blockchain: bool = True,
    enable_verification: bool = True,
    enable_coordination: bool = True,
    p_spoof: float = 0.0,
    n_byzantine: int = 0,
    p_drop: float = 0.0,
    sensor_failure_rate: float = 0.0,
    burst_mode: bool = False,
    policy: str = "greedy",
    attack_type: Optional[str] = None,
) -> EpisodeResult:
    """Run one episode and return all metrics.

    This is the single source of truth for episode execution across all
    experiment scripts. Every ablation, stress test, and comparison uses
    this function with different flag combinations.

    Args:
        seed: Random seed for this episode.
        config_name: Label for this configuration (used in output CSVs).
        grid_size: Grid side length.
        n_timesteps: Episode length in simulation steps.
        n_uavs: UAV fleet size N.
        enable_governance: Run full GOMDP smart contract enforcement.
        enable_hitl: Include human-in-the-loop authorisation.
        enable_blockchain: Include blockchain consensus.
        enable_verification: Include two-stage verification pipeline.
        enable_coordination: Use adaptive coordination (False = static patrol).
        p_spoof: Sensor spoofing attack probability per cell per step.
        n_byzantine: Number of Byzantine validators to inject.
        p_drop: Packet drop probability (communication disruption).
        sensor_failure_rate: Fraction of UAV sensors to disable.
        burst_mode: Apply burst multiplier to blockchain delay.
        policy: ``"greedy"`` or ``"ppo"``.
        attack_type: Optional attack label (e.g., ``"injection"``).

    Returns:
        EpisodeResult with all computed metrics.
    """
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    env_cfg = EnvironmentConfig(
        grid_size=grid_size,
        n_timesteps=n_timesteps,
    )
    env = WildfireGridEnvironment(env_cfg)
    env.reset(seed=seed)

    # Build UAV fleet
    fleet: List[UAVAgent] = [
        UAVAgent(
            agent_id=f"uav_{i}",
            initial_position=(
                int(rng.integers(0, grid_size)),
                int(rng.integers(0, grid_size)),
            ),
            grid_size=grid_size,
        )
        for i in range(n_uavs)
    ]

    # Apply sensor failure
    active_uavs = fleet
    if sensor_failure_rate > 0:
        n_failed = int(sensor_failure_rate * n_uavs)
        active_uavs = fleet[n_failed:]

    # Blockchain / governance setup
    consensus = PBFTConsensus(rng=rng)
    if n_byzantine > 0:
        for i in range(min(n_byzantine, consensus.n_validators)):
            try:
                consensus.inject_byzantine_fault(i, ByzantineFaultType.MALICIOUS)
            except ValueError:
                pass

    contract = GovernanceSmartContract(consensus=consensus) if enable_blockchain else None
    oracle = HumanOperatorOracle(rng=rng) if enable_hitl else None
    hitl_gate = HITLAuthorisationGate(oracle=oracle, rng=rng) if enable_hitl else None
    gomdp = GovernanceInvariantMDP(tau=0.80)
    checker = GovernanceInvariantChecker(tau=0.80)

    # Sensor spoofer
    spoofer = None
    if p_spoof > 0:
        from wildfire_governance.adversarial.sensor_spoofer import SensorSpoofer
        spoofer = SensorSpoofer(p_spoof=p_spoof, rng=rng)

    # Tracking
    ignition_time = 0
    first_detection: Optional[int] = None
    n_alerts = 0
    n_false = 0
    bc_delays: List[float] = []
    human_delays: List[float] = []
    trajectory: List[Dict] = []
    n_inject_attempted = 0
    n_inject_blocked = 0

    # Greedy policy setup
    greedy = None
    if enable_coordination and policy == "greedy":
        from wildfire_governance.decision.greedy_policy import RiskWeightedGreedyPolicy
        from wildfire_governance.decision.belief_state import BeliefState
        greedy = RiskWeightedGreedyPolicy(n_sectors=25, grid_size=grid_size)
        belief = BeliefState(grid_size=grid_size)

    # Static patrol pattern (when coordination disabled)
    sector_size = grid_size // max(1, int(np.sqrt(n_uavs)))

    for t in range(n_timesteps):
        positions = [u.position for u in active_uavs]

        # Communication disruption: skip some sensor readings
        if p_drop > 0 and rng.random() < p_drop:
            obs_dict, done, _ = env.step(positions)
            trajectory.append({"timestep": t, "alert_broadcast": False,
                                "governance_cert": None, "confidence": 0.0,
                                "human_approval": False})
            if done:
                break
            continue

        obs_dict, done, sim_info = env.step(positions)

        # Apply sensor spoofing
        heat_map = obs_dict["heat_map"].copy()
        if spoofer is not None:
            heat_map = spoofer.inject(heat_map, obs_dict["fire_mask"])

        # Detection check
        max_heat = float(heat_map.max())
        if max_heat > 0.60 and first_detection is None:
            first_detection = t

        # UAV movement (adaptive vs static)
        if enable_coordination and greedy is not None:
            readings = env.get_observations(positions)
            belief.update(readings)
            risk_map = belief.get_risk_map()
            allocation = greedy.select_actions(
                risk_map, positions, [u.battery_fraction for u in active_uavs]
            )
            for uav_idx, sector_id in allocation.items():
                if uav_idx < len(active_uavs):
                    centroid = greedy.sector_centroid(sector_id)
                    try:
                        active_uavs[uav_idx].move_to(centroid, rng)
                    except Exception:
                        active_uavs[uav_idx].recharge()
        else:
            # Static: move to fixed grid positions
            for i, uav in enumerate(active_uavs):
                target = (
                    (i * sector_size) % grid_size,
                    ((i * sector_size) // grid_size * sector_size) % grid_size,
                )
                try:
                    uav.move_to(target, rng)
                except Exception:
                    uav.recharge()

        # Governance pipeline
        step_info: Dict = {
            "timestep": t,
            "alert_broadcast": False,
            "governance_cert": None,
            "confidence": 0.0,
            "human_approval": False,
        }

        if max_heat > 0.80 and first_detection is not None:
            weather_idx = float(np.clip(
                obs_dict["wind_field"].mean() - obs_dict["humidity_field"].mean() + 0.5,
                0.0, 1.0,
            ))
            if enable_verification:
                conf = float(np.clip(0.65 * max_heat + 0.35 * weather_idx, 0.0, 1.0))
            else:
                conf = max_heat
            step_info["confidence"] = conf

            if conf > 0.80:
                row_idx, col_idx = np.unravel_index(heat_map.argmax(), heat_map.shape)
                is_true_fire = bool(obs_dict["fire_mask"][row_idx, col_idx] > 0.5)

                if enable_governance and enable_hitl and enable_blockchain and hitl_gate and contract:
                    tx = build_transaction(
                        event_id=f"evt_{seed}_{t}",
                        geo_boundary=(int(row_idx), int(col_idx), int(row_idx) + 1, int(col_idx) + 1),
                        confidence_score=conf,
                        sensor_readings={"heat": max_heat, "weather": weather_idx},
                    )
                    decision, sig = hitl_gate.process(tx, conf)
                    step_info["human_approval"] = decision.approved
                    human_delays.append(decision.review_delay_steps)

                    if decision.approved and sig is not None:
                        result = contract.verify_and_execute(
                            tx, sig, hitl_gate.public_key_bytes, burst_mode=burst_mode
                        )
                        if result.consensus_result:
                            bc_delays.append(result.consensus_result.delay_steps)
                        if result.alert_enabled:
                            step_info["alert_broadcast"] = True
                            step_info["governance_cert"] = result.cert
                            n_alerts += 1
                            if not is_true_fire:
                                n_false += 1
                elif enable_governance and enable_hitl and not enable_blockchain and hitl_gate:
                    tx = build_transaction(
                        event_id=f"evt_{seed}_{t}",
                        geo_boundary=(int(row_idx), int(col_idx), int(row_idx) + 1, int(col_idx) + 1),
                        confidence_score=conf,
                        sensor_readings={"heat": max_heat, "weather": weather_idx},
                    )
                    decision, _ = hitl_gate.process(tx, conf)
                    step_info["human_approval"] = decision.approved
                    human_delays.append(decision.review_delay_steps)

                    if decision.approved:
                        step_info["alert_broadcast"] = True
                        step_info["governance_cert"] = None

                        n_alerts += 1

                        if not is_true_fire:
                            n_false += 1
                elif not enable_governance:
                    # Ungoverned baseline: alert without any checks
                    step_info["alert_broadcast"] = True
                    n_alerts += 1
                    if not is_true_fire:
                        n_false += 1

                # Adversarial injection test (runs in background every 50 steps)
                if attack_type != "injection" and t % 50 == 0 and enable_blockchain and contract:
                    n_inject_attempted += 1
                    blocked = not contract.attempt_unauthorised_injection(
                        (int(row_idx), int(col_idx), int(row_idx) + 1, int(col_idx) + 1)
                    )
                    if blocked:
                        n_inject_blocked += 1

        if attack_type == "injection" and t % 50 == 0 and enable_blockchain and contract:
            attack_row, attack_col = np.unravel_index(heat_map.argmax(), heat_map.shape)
            n_inject_attempted += 1
            blocked = not contract.attempt_unauthorised_injection(
                (int(attack_row), int(attack_col), int(attack_row) + 1, int(attack_col) + 1)
            )
            if blocked:
                n_inject_blocked += 1

        trajectory.append(step_info)
        if done:
            break

    # Compute final metrics
    ld = float(first_detection - ignition_time) if first_detection is not None else float(n_timesteps)
    fp_pct = (n_false / max(1, n_alerts)) * 100.0
    mean_bc = float(np.mean(bc_delays)) if bc_delays else 1.2
    mean_hv = float(np.mean(human_delays)) if human_delays else 3.0
    n_inject_success = max(0, n_inject_attempted - n_inject_blocked)

    report = checker.check_trajectory(trajectory)

    return EpisodeResult(
        seed=seed,
        config_name=config_name,
        ld=ld,
        fp_pct=fp_pct,
        bc_delay=mean_bc,
        human_delay=mean_hv,
        n_alerts=n_alerts,
        n_false=n_false,
        governance_compliant=report.theorem1_satisfied,
        n_injections_attempted=n_inject_attempted,
        n_injections_blocked=n_inject_blocked,
        step_logs=trajectory,
        injection_success=int(n_inject_success > 0),
    )
