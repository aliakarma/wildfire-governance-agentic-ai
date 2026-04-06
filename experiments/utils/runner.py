"""Shared single-seed episode runner."""
from __future__ import annotations
from dataclasses import dataclass
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
from wildfire_governance.simulation.grid_environment import EnvironmentConfig, WildfireGridEnvironment
from wildfire_governance.utils.reproducibility import set_global_seed

@dataclass
class EpisodeResult:
    seed: int; config_name: str; ld: float; fp_pct: float
    bc_delay: float = 1.2; human_delay: float = 3.0; le2e: float = 0.0
    n_alerts: int = 0; n_false: int = 0; governance_compliant: bool = True
    n_injections_attempted: int = 0; n_injections_blocked: int = 0

    def __post_init__(self):
        if self.le2e == 0.0: self.le2e = self.ld + self.bc_delay + self.human_delay

def run_episode(seed, config_name, grid_size=100, n_timesteps=3000, n_uavs=20,
                enable_governance=True, enable_hitl=True, enable_blockchain=True,
                enable_verification=True, enable_coordination=True, p_spoof=0.0,
                n_byzantine=0, p_drop=0.0, sensor_failure_rate=0.0, uav_noise_std=0.05,
                burst_mode=False, policy="greedy"):
    set_global_seed(seed); rng = np.random.default_rng(seed)
    env = WildfireGridEnvironment(EnvironmentConfig(grid_size=grid_size, n_timesteps=n_timesteps, uav_noise_std=uav_noise_std))
    env.reset(seed=seed)
    fleet = [UAVAgent(f"uav_{i}", (int(rng.integers(0,grid_size)), int(rng.integers(0,grid_size))), grid_size=grid_size) for i in range(n_uavs)]
    active_uavs = fleet[int(sensor_failure_rate*n_uavs):] if sensor_failure_rate > 0 else fleet
    consensus = PBFTConsensus(rng=rng)
    for i in range(min(n_byzantine, consensus.n_validators)):
        try: consensus.inject_byzantine_fault(i, ByzantineFaultType.MALICIOUS)
        except ValueError: pass
    contract = GovernanceSmartContract(consensus=consensus) if enable_blockchain else None
    oracle = HumanOperatorOracle(rng=rng) if enable_hitl else None
    hitl_gate = HITLAuthorisationGate(oracle=oracle, rng=rng) if enable_hitl else None
    gomdp = GovernanceInvariantMDP(tau=0.80); checker = GovernanceInvariantChecker(tau=0.80)
    spoofer = None
    if p_spoof > 0:
        from wildfire_governance.adversarial.sensor_spoofer import SensorSpoofer
        spoofer = SensorSpoofer(p_spoof=p_spoof, rng=rng)
    from wildfire_governance.decision.greedy_policy import RiskWeightedGreedyPolicy
    from wildfire_governance.decision.belief_state import BeliefState
    greedy = RiskWeightedGreedyPolicy(n_sectors=25, grid_size=grid_size) if enable_coordination else None
    belief = BeliefState(grid_size=grid_size) if enable_coordination else None
    sector_size = grid_size // max(1, int(np.sqrt(n_uavs)))
    ignition_time = 0; first_detection = None; n_alerts = 0; n_false = 0
    bc_delays: List[float] = []; human_delays: List[float] = []; trajectory = []
    n_inject_attempted = 0; n_inject_blocked = 0

    for t in range(n_timesteps):
        positions = [u.position for u in active_uavs]
        if p_drop > 0 and rng.random() < p_drop:
            obs_dict, done, _ = env.step(positions)
            trajectory.append({"timestep": t, "alert_broadcast": False, "governance_cert": None, "confidence": 0.0, "human_approval": False})
            if done: break
            continue
        obs_dict, done, sim_info = env.step(positions)
        heat_map = obs_dict["heat_map"].copy()
        if spoofer is not None: heat_map = spoofer.inject(heat_map, obs_dict["fire_mask"])
        
        readings = env.get_observations(positions)
        if len(readings) > 0:
            max_heat = max([float(r.heat_value) for r in readings])
            best_r = max(readings, key=lambda r: r.heat_value)
            best_pos = best_r.position
        else:
            max_heat = 0.0
            best_pos = (0, 0)
            
        if max_heat > 0.60 and first_detection is None: first_detection = t
        if enable_coordination and greedy is not None:
            belief.update(readings)
            allocation = greedy.select_actions(belief.get_risk_map(), positions, [u.battery_fraction for u in active_uavs])
            for idx, sid in allocation.items():
                if idx < len(active_uavs):
                    try: active_uavs[idx].move_to(greedy.sector_centroid(sid), rng)
                    except Exception: active_uavs[idx].recharge()
        step_info = {"timestep": t, "alert_broadcast": False, "governance_cert": None, "confidence": 0.0, "human_approval": False}
        if max_heat > 0.80 and first_detection is not None:
            weather_idx = float(np.clip(obs_dict["wind_field"].mean()-obs_dict["humidity_field"].mean()+0.5, 0.0, 1.0))
            conf = float(np.clip(0.65*max_heat+0.35*weather_idx, 0.0, 1.0)) if enable_verification else max_heat
            step_info["confidence"] = conf
            if conf > 0.80:
                row_idx, col_idx = best_pos
                is_true_fire = bool(obs_dict["fire_mask"][row_idx, col_idx] > 0.5)
                if enable_governance and enable_hitl and enable_blockchain and hitl_gate and contract:
                    tx = build_transaction(f"evt_{seed}_{t}", (row_idx,col_idx,row_idx+1,col_idx+1), conf, {"heat": max_heat})
                    decision, sig = hitl_gate.process(tx, conf)
                    step_info["human_approval"] = decision.approved; human_delays.append(decision.review_delay_steps)
                    gomdp_result = gomdp.step_alert_action(conf, decision.approved, sig is not None)
                    if not gomdp_result.blocked and sig is not None:
                        result = contract.verify_and_execute(tx, sig, hitl_gate.public_key_bytes, burst_mode=burst_mode)
                        if result.consensus_result: bc_delays.append(result.consensus_result.delay_steps)
                        if result.alert_enabled:
                            step_info["alert_broadcast"] = True; step_info["governance_cert"] = result.cert
                            n_alerts += 1
                            if not is_true_fire: n_false += 1
                elif not enable_governance:
                    step_info["alert_broadcast"] = True; n_alerts += 1
                    if not is_true_fire: n_false += 1
                if t % 50 == 0 and enable_blockchain and contract:
                    n_inject_attempted += 1
                    if not contract.attempt_unauthorised_injection((row_idx,col_idx,row_idx+1,col_idx+1)): n_inject_blocked += 1
        trajectory.append(step_info)
        if done: break
    ld = float(first_detection - ignition_time) if first_detection is not None else float(n_timesteps)
    fp_pct = (n_false / max(1, n_alerts)) * 100.0
    report = checker.check_trajectory(trajectory)
    return EpisodeResult(seed=seed, config_name=config_name, ld=ld, fp_pct=fp_pct,
        bc_delay=float(np.mean(bc_delays)) if bc_delays else 1.2,
        human_delay=float(np.mean(human_delays)) if human_delays else 3.0,
        n_alerts=n_alerts, n_false=n_false, governance_compliant=report.theorem1_satisfied,
        n_injections_attempted=n_inject_attempted, n_injections_blocked=n_inject_blocked)
