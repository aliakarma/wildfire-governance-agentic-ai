"""Hierarchical multi-agent coordination engine.

The meta-controller maintains the global belief state and drives
risk-weighted allocation of UAVs. Supports both Greedy-GOMDP and
PPO-GOMDP policy backends interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wildfire_governance.agents.uav_agent import UAVAgent
from wildfire_governance.decision.belief_state import BeliefState
from wildfire_governance.decision.greedy_policy import RiskWeightedGreedyPolicy
from wildfire_governance.simulation.sensor_models import SensorReading
from wildfire_governance.verification.confidence_scorer import (
    TwoStageConfidenceScorer,
    VerificationResult,
)
from wildfire_governance.verification.threshold_adapter import OnlineThresholdAdapter


@dataclass
class CoordinationOutput:
    """Output of one coordination step.

    Attributes:
        allocation: Dict mapping uav_index → assigned sector_id.
        alert_request: VerificationResult triggering HITL if escalated; None otherwise.
        updated_belief: Updated risk map after sensor fusion.
        anomaly_location: Grid cell (row, col) of the detected anomaly, if any.
    """

    allocation: Dict[int, int]
    alert_request: Optional[VerificationResult]
    updated_belief: np.ndarray
    anomaly_location: Optional[Tuple[int, int]]


class HierarchicalCoordinationEngine:
    """Hierarchical coordination engine for wildfire UAV monitoring.

    Drives the perception → belief update → allocation → verification → alert loop.
    The governance predicate (GOMDP) is evaluated downstream by
    :class:`wildfire_governance.gomdp.definition.GovernanceInvariantMDP`.

    Args:
        uav_fleet: List of UAVAgent instances.
        grid_size: Environment grid side length.
        n_sectors: Number of patrol sectors for the greedy policy.
        policy_backend: ``"greedy"`` (default) or ``"ppo"`` (loads PPO-GOMDP checkpoint).
    """

    def __init__(
        self,
        uav_fleet: List[UAVAgent],
        grid_size: int = 100,
        n_sectors: int = 25,
        policy_backend: str = "greedy",
    ) -> None:
        self._fleet = uav_fleet
        self._grid_size = grid_size
        self._belief = BeliefState(grid_size=grid_size)
        self._greedy = RiskWeightedGreedyPolicy(n_sectors=n_sectors, grid_size=grid_size)
        self._scorer = TwoStageConfidenceScorer()
        self._threshold_adapter = OnlineThresholdAdapter()
        self._policy_backend = policy_backend
        self._ppo_agent: Any = None
        if policy_backend == "ppo":
            self._load_ppo_agent()

    def _load_ppo_agent(self) -> None:
        """Lazily load the PPO-GOMDP agent from the pre-trained checkpoint."""
        try:
            from wildfire_governance.rl.ppo_agent import PPOGOMDPAgent  # type: ignore[import]
            from pathlib import Path
            ckpt = Path(__file__).parent.parent / "rl" / "checkpoints" / "ppo_gomdp_best.pt"
            agent = PPOGOMDPAgent(grid_size=self._grid_size, n_uavs=len(self._fleet))
            if ckpt.exists():
                agent.load_checkpoint(ckpt)
            self._ppo_agent = agent
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(
                f"Could not load PPO-GOMDP checkpoint: {exc}. Falling back to greedy.",
                stacklevel=2,
            )
            self._policy_backend = "greedy"

    def step(
        self,
        observations: List[SensorReading],
        heat_map: np.ndarray,
        wind_field: np.ndarray,
        humidity_field: np.ndarray,
        timestep: int,
        rng: np.random.Generator,
    ) -> CoordinationOutput:
        """Execute one full coordination cycle.

        1. Update belief state via Bayesian filtering.
        2. Compute risk-weighted allocation (greedy or PPO).
        3. Execute UAV moves toward assigned sectors.
        4. Run two-stage verification on high-confidence anomalies.
        5. Return CoordinationOutput for downstream governance evaluation.

        Args:
            observations: Sensor readings from all UAVs.
            heat_map: Current heat map H_t.
            wind_field: Current wind field W_t.
            humidity_field: Current humidity field.
            timestep: Current simulation timestep.
            rng: Seeded NumPy Generator.

        Returns:
            CoordinationOutput with allocation, alert request, and updated belief.
        """
        # 1. Update belief state
        self._belief.update(
            observations,
            p_detect_fire=0.85,
            p_detect_no_fire=0.15,
        )
        risk_map = self._belief.get_risk_map()

        # 2. Allocate UAVs
        positions = [uav.position for uav in self._fleet]
        batteries = [uav.battery_fraction for uav in self._fleet]

        if self._policy_backend == "ppo" and self._ppo_agent is not None:
            belief_vec = np.concatenate(
                [risk_map.ravel(),
                 np.array([(p[0] / self._grid_size, p[1] / self._grid_size) for p in positions]).ravel()]
            )
            allocation = self._ppo_agent.select_actions(belief_vec, self._fleet)
        else:
            allocation = self._greedy.select_actions(risk_map, positions, batteries)

        # 3. Move UAVs toward assigned sector centroids
        for uav_idx, sector_id in allocation.items():
            uav = self._fleet[uav_idx]
            centroid = self._greedy.sector_centroid(sector_id)
            uav.assign_sector(sector_id)
            try:
                uav.move_to(centroid, rng)
            except Exception:  # noqa: BLE001
                uav.recharge()

        # 4. Two-stage verification on peak-risk anomaly
        alert_request: Optional[VerificationResult] = None
        anomaly_location: Optional[Tuple[int, int]] = None

        if observations:
            best_obs = max(observations, key=lambda o: o.heat_value)
            row, col = best_obs.position
            heat_idx = float(best_obs.heat_value)
            # Derive weather index from wind and humidity at the anomaly cell
            w_val = float(wind_field[row, col]) if wind_field.shape == heat_map.shape else 0.5
            h_val = float(humidity_field[row, col]) if humidity_field.shape == heat_map.shape else 0.5
            weather_idx = float(np.clip(w_val - h_val + 0.5, 0.0, 1.0))

            # Stage-1 score (no UAV verification yet)
            stage1_result = self._scorer.score(heat_idx, weather_idx)

            if stage1_result.escalated_to_stage2:
                anomaly_location = (row, col)
                # Stage-2: simulate verification UAV observation
                verification_pos = best_obs.is_fire_detected
                result = self._scorer.score(heat_idx, weather_idx, verification_pos)
                if result.escalated_to_hitl:
                    alert_request = result

        return CoordinationOutput(
            allocation=allocation,
            alert_request=alert_request,
            updated_belief=risk_map,
            anomaly_location=anomaly_location,
        )

    def adapt_thresholds(self, precision: float, recall: float) -> None:
        """Update verification thresholds via EMA adaptation (learning module).

        Args:
            precision: Episode-level precision.
            recall: Episode-level recall.
        """
        tau1, tau2 = self._threshold_adapter.update(precision, recall)
        self._scorer.update_thresholds(tau1, tau2)

    def reset(self) -> None:
        """Reset belief state and threshold adapter for a new episode."""
        self._belief.reset()
        self._threshold_adapter.reset()
