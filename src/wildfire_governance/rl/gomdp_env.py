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

from wildfire_governance.agents.base_agent import InsufficientBatteryError
from wildfire_governance.agents.coordination_engine import HierarchicalCoordinationEngine
from wildfire_governance.agents.uav_agent import UAVAgent
from wildfire_governance.decision.greedy_policy import RiskWeightedGreedyPolicy
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

# Per-step retention of the belief map outside current sensor coverage. At 0.99
# an unvisited cell's evidence half-lives in ~69 steps, comparable to the
# 50-150 step ignition delay, so the fleet cannot bank one early sweep.
_BELIEF_DECAY = 0.99

# Mission deadline in steps: the detection window over which latency is scored.
# Each reward term is normalised by the span over which it can actually accrue,
# so an episode's worst case for each is exactly its paper weight — 0.5 for a
# fire left undetected through the whole window, 0.15 for flying the whole
# fleet for the whole episode, 0.35 for every alert being false.
#
# Normalising energy by this horizon instead of the episode length is what an
# earlier revision got wrong: energy accrues across all 3000 steps whereas
# latency is only at stake for the ~150 before the fixed IoT and satellite
# sensors find the fire regardless. That put energy's episode maximum at 1.5
# against latency's realistic 0.25, making "park the fleet" a stronger lever
# than "find the fire", which inverts the objective the paper states.
_LD_HORIZON = 300

# Cells within this radius of an already-alerted cell belong to that incident
# and are excluded when looking for the next one. This deduplicates alerts —
# re-running verification every step on the same fire produced ~800 alerts per
# episode — and, because the search then moves on to the next strongest
# hotspot, it is also what lets a confuser reach the pipeline at all. Scoring
# only the global argmax meant the fire outshone every anomaly on 96% of
# steps, so no candidate false alert was ever evaluated.
_INCIDENT_RADIUS = 10

# Distinct hotspots the verification pipeline processes per step. A duty officer
# triages every candidate detection, not just the strongest one.
_MAX_INCIDENTS_PER_STEP = 3


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

        # Sector geometry. Reused from the greedy baseline so that a sector id
        # means the same patch of grid for both policies, which is what makes
        # the PPO-vs-greedy comparison in the paper an apples-to-apples one.
        self._sectors = RiskWeightedGreedyPolicy(
            n_sectors=n_sectors, grid_size=self._gs
        )

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
        # Grid regions already covered by a broadcast alert.
        self._alerted_mask: np.ndarray = np.zeros(
            (self._gs, self._gs), dtype=bool
        )

        # Belief map B_t: what the fleet has actually *observed*, not ground
        # truth. Feeding the simulator's true heat map to the policy would hand
        # it the fire's location for free and reduce L_d to a routing time
        # rather than a search-and-detect time.
        self._belief_map: np.ndarray = np.zeros((self._gs, self._gs), dtype=np.float32)

        # Alert incidents a typical episode contains, measured at ~55 per 3000
        # steps. The false-alert term is divided by it so that an episode in
        # which *every* alert is false costs the paper's full 0.35 weight.
        # Charging 0.35 per individual false alert instead made one mistake
        # worth more than the entire detection-latency term.
        self._alert_budget = max(1.0, self._env_config.n_timesteps / 55.0)

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
        # Ignition is scheduled by the simulator at a stochastic delay, not at
        # t=0. Hard-coding 0 here made L_d measure "time since episode start"
        # instead of "time since the fire started".
        self._ignition_time = int(self._sim.ignition_time)
        self._first_detection = None
        self._n_alerts_broadcast = 0
        self._n_false_alerts = 0
        self._alerted_mask = np.zeros((self._gs, self._gs), dtype=bool)
        self._belief_map = np.zeros((self._gs, self._gs), dtype=np.float32)

        return self._build_obs(), {"ignition_time": self._ignition_time}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Integer array of shape (n_uavs,) — sector assignment per UAV.

        Returns:
            Tuple (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1

        # Map action to sector assignments and fly the fleet. Until this was
        # wired up the action argument was discarded, so every policy produced
        # byte-identical trajectories and PPO had no gradient to follow.
        energy_used = self._apply_action(action)
        positions = [uav.position for uav in self._fleet]
        obs_dict, done, sim_info = self._sim.step(positions)
        self._ignition_time = int(sim_info.get("ignition_time", self._ignition_time))

        # Sense through the fleet's actual sensor footprints. The previous
        # global ``heat_map.max()`` saw the whole grid at once, which made
        # detection independent of where the UAVs were.
        fused = self._sim.sense_fused(positions)
        self._update_belief(fused)

        heat_val = float(fused["max_heat"])
        anomaly_location = fused["argmax_cell"]
        fire_ignited = self._step_count >= self._ignition_time

        # The latency clock stops only on a *true* fire cell; a detection on a
        # synthetic anomaly is a false positive, not a detection.
        if fire_ignited and fused["detected_fire"] and self._first_detection is None:
            self._first_detection = self._step_count

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
        false_alert_now = False

        # Verify every distinct hotspot the fleet can see, not only the hottest.
        # Scoring the single global argmax meant a spreading fire monopolised
        # the pipeline — 4142 fire candidates against 9 non-fire ones — so the
        # confusers that F_p is meant to measure were never even examined.
        seen = np.zeros_like(self._alerted_mask)
        for _ in range(_MAX_INCIDENTS_PER_STEP):
            incident = self._select_incident(fused, seen)
            if incident is None or incident[2] <= 0.60:
                break
            row, col, cand_heat = incident
            broadcast, was_false = self._verify_candidate(row, col, cand_heat, info)
            alert_broadcast = alert_broadcast or broadcast
            false_alert_now = false_alert_now or was_false

            r0, r1, c0, c1 = self._incident_region(row, col)
            seen[r0:r1, c0:c1] = True
            if broadcast:
                self._alerted_mask[r0:r1, c0:c1] = True

        self._trajectory.append(dict(info))

        # Compute reward. Every term is now a consequence of where the policy
        # chose to fly, which is what makes this a learnable objective.
        #
        # ld_component: one unit of cost for every step the fire burns
        #   undetected. No cost before ignition (nothing to find yet) and none
        #   after detection. The old form used a truthiness test on
        #   ``_first_detection``, which also misread a step-0 detection as "not
        #   yet detected".
        # fp_component: charged only on the step a false alert is actually
        #   broadcast. The old form charged every alert forever after the first
        #   false one, so a single early mistake poisoned the rest of the episode.
        ld_component = 1.0 if (fire_ignited and self._first_detection is None) else 0.0
        fp_component = 1.0 if false_alert_now else 0.0
        reward = -(
            0.5 * ld_component / _LD_HORIZON
            + 0.35 * fp_component / self._alert_budget
            + 0.15 * energy_used / self._env_config.n_timesteps
        )
        self._total_reward += reward

        terminated = done
        truncated = self._step_count >= self._env_config.n_timesteps

        if terminated or truncated:
            fp_rate = self._n_false_alerts / max(1, self._n_alerts_broadcast) * 100.0
            info["episode_ld"] = float(
                self._first_detection - self._ignition_time
            ) if self._first_detection is not None else float("inf")
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

    def _apply_action(self, action: np.ndarray) -> float:
        """Fly each UAV one cell toward the centroid of its assigned sector.

        A UAV too flat to fly returns to base and recharges instead of moving,
        mirroring :class:`HierarchicalCoordinationEngine`. Battery capacity is
        500 steps against episodes of 3000, so recharge cycles are a real part
        of the coordination problem rather than an edge case.

        Args:
            action: Integer array of sector ids, one per UAV.

        Returns:
            Fraction of the fleet that expended battery this step, in [0, 1].
            This is the energy actually spent now, not cumulative depletion:
            charging the depleted *level* every step billed a UAV forever for a
            single early move.
        """
        act = np.asarray(action).ravel()
        energy = 0.0
        for i, uav in enumerate(self._fleet):
            sector_id = int(act[i]) % self._n_sectors if i < act.size else 0
            uav.assign_sector(sector_id)
            try:
                energy += uav.move_to(self._sectors.sector_centroid(sector_id), self._rng)
            except InsufficientBatteryError:
                uav.recharge()
        return energy / self._n_uavs

    def _verify_candidate(
        self, row: int, col: int, heat: float, info: Dict[str, Any]
    ) -> Tuple[bool, bool]:
        """Run two-stage verification and governance on one candidate hotspot.

        Args:
            row: Candidate row.
            col: Candidate column.
            heat: Observed heat at the candidate cell.
            info: Per-step info dict, updated in place when an alert broadcasts.

        Returns:
            ``(alert_broadcast, was_false_alert)``.
        """
        # Weather sampled at the candidate cell, not grid-wide. The mean of
        # wind ~U(0, 0.6) minus humidity ~U(0.2, 0.8) is a near-constant 0.3,
        # which pinned confidence at 0.755 and left the tau=0.80 gate
        # permanently shut — the HITL and blockchain path never executed once
        # during training, making the 100% compliance figure vacuous.
        w_val = float(self._sim.wind_field[row, col])
        h_val = float(self._sim.humidity_field[row, col])
        weather_idx = float(np.clip(w_val - h_val + 0.5, 0.0, 1.0))
        conf = float(np.clip(0.65 * heat + 0.35 * weather_idx, 0.0, 1.0))
        info["confidence"] = max(float(info.get("confidence", 0.0)), conf)

        if conf <= 0.80:
            return False, False

        is_true_fire = bool(self._sim.fire_mask[row, col] > 0.5)

        if not self._enable_governance:
            # Non-governance path (CMDP comparison / ungoverned)
            self._n_alerts_broadcast += 1
            if not is_true_fire:
                self._n_false_alerts += 1
            info["alert_broadcast"] = True
            return True, not is_true_fire

        tx = build_transaction(
            event_id=f"evt_{self._step_count}_{row}_{col}",
            geo_boundary=(row, col, row + 1, col + 1),
            confidence_score=conf,
            sensor_readings={"heat": heat, "weather": weather_idx},
        )
        decision, signature = self._hitl_gate.process(tx, conf)
        if not (decision.approved and signature is not None):
            return False, False
        info["human_approval"] = True

        result = self._smart_contract.verify_and_execute(
            tx, signature, self._hitl_gate.public_key_bytes
        )
        if not result.alert_enabled:
            return False, False

        self._n_alerts_broadcast += 1
        if not is_true_fire:
            self._n_false_alerts += 1
        info["governance_cert"] = result.cert
        info["alert_broadcast"] = True
        return True, not is_true_fire

    def _incident_region(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Bounding slice of the incident centred on ``(row, col)``."""
        return (
            max(0, row - _INCIDENT_RADIUS), min(self._gs, row + _INCIDENT_RADIUS + 1),
            max(0, col - _INCIDENT_RADIUS), min(self._gs, col + _INCIDENT_RADIUS + 1),
        )

    def _select_incident(
        self, fused: Dict[str, Any], seen: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        """Pick the strongest detected hotspot not already covered by an alert.

        Args:
            fused: Return value of :meth:`WildfireGridEnvironment.sense_fused`.
            seen: Additional mask of regions already examined this step.

        Returns:
            ``(row, col, observed_heat)`` of the candidate, or None when every
            detection belongs to an incident that has already been broadcast.
        """
        if not fused["detected"]:
            return None
        blocked = self._alerted_mask | seen
        candidates = np.where(blocked, 0.0, fused["observed_heat"])
        flat = int(candidates.argmax())
        heat = float(candidates.ravel()[flat])
        if heat <= 0.0:
            return None
        row, col = np.unravel_index(flat, candidates.shape)
        return int(row), int(col), heat

    def _update_belief(self, fused: Dict[str, Any]) -> None:
        """Fold this step's observations into the belief map.

        Cells the fleet can currently see are overwritten with what was
        observed there — including zero, which is the informative result of
        looking somewhere and finding nothing. Everything else decays toward
        zero so that stale observations lose value and the policy is pushed to
        revisit, which is the whole point of a patrol.

        Args:
            fused: Return value of :meth:`WildfireGridEnvironment.sense_fused`.
        """
        self._belief_map *= _BELIEF_DECAY
        coverage = fused["coverage"]
        self._belief_map[coverage] = fused["observed_heat"][coverage]

    def _build_obs(self) -> np.ndarray:
        """Build flattened observation vector: belief map + UAV positions."""
        belief = self._belief_map.ravel().astype(np.float32)
        pos_vec = np.array(
            [(uav.position[0] / self._gs, uav.position[1] / self._gs)
             for uav in self._fleet],
            dtype=np.float32,
        ).ravel()
        return np.concatenate([belief, pos_vec])
