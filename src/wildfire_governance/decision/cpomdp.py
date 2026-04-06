"""CPOMDP theoretical formulation for wildfire governance.

This module defines the CPOMDP theoretical target (Eqs. 1–4 in the paper).
The exact CPOMDP is computationally intractable at operational scale.

The practical approximation is implemented in ``greedy_policy.py``.
The blockchain layer enforces the governance predicate EXACTLY at the
environment transition level, independently of approximation quality.

See also: ``gomdp/definition.py`` for the GOMDP framework that provides
the per-trajectory safety guarantee (Theorem 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class CPOMDPNotSolvedException(NotImplementedError):
    """Raised when caller attempts to solve the CPOMDP exactly.

    The CPOMDP formulation is a theoretical target, not a solvable model
    at operational scale. Use ``greedy_policy.RiskWeightedGreedyPolicy``
    or ``rl.ppo_agent.PPOGOMDPAgent`` as practical approximations.
    """


@dataclass
class CPOMDPCostWeights:
    """Weighted cost function coefficients (Eq. 2, paper).

    Attributes:
        alpha: Detection latency weight (default 0.50).
        beta: False alert weight (default 0.35).
        gamma: Resource cost weight (default 0.15).
    """

    alpha: float = 0.50
    beta: float = 0.35
    gamma: float = 0.15

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Cost weights must sum to 1.0; got alpha+beta+gamma={total:.4f}"
            )


class WildfireCPOMDP:
    """Theoretical CPOMDP formulation for wildfire monitoring.

    Defines the state space, action space, cost function, governance predicate,
    and Proposition 1 latency bound. Does NOT implement a solver.

    Args:
        cost_weights: Weighted cost coefficients.
        tau: Alert confidence threshold (governance predicate parameter).
    """

    def __init__(
        self,
        cost_weights: CPOMDPCostWeights | None = None,
        tau: float = 0.80,
    ) -> None:
        self.cost_weights = cost_weights or CPOMDPCostWeights()
        self.tau = tau

    def compute_cost(
        self,
        ld: float,
        fp: float,
        cr: float,
    ) -> float:
        """Compute the weighted CPOMDP cost (Eq. 2 in paper).

        Args:
            ld: Detection latency (steps).
            fp: False public alert rate (fraction, not percentage).
            cr: Resource cost (normalised to [0, 1]).

        Returns:
            Scalar cost value.
        """
        w = self.cost_weights
        return w.alpha * ld + w.beta * fp + w.gamma * cr

    def check_governance_predicate(
        self,
        confidence: float,
        human_approval: bool,
    ) -> bool:
        """Evaluate the governance predicate G(s, a) (Eq. 4 in paper).

        G(s, a) = [Conf^(2)_t > tau] AND [HA_t = 1]

        Args:
            confidence: Stage-2 confidence score Conf^(2)_t.
            human_approval: Binary human authorisation HA_t.

        Returns:
            True if and only if both conditions are satisfied.
        """
        return confidence > self.tau and human_approval

    def latency_bound(
        self,
        area: float,
        velocity: float,
        n_uavs: int,
        delta: float,
    ) -> float:
        """Compute the Proposition 1 detection latency upper bound.

        E[L_d] <= A / (v * N) + Delta

        Args:
            area: Monitored area (grid cells squared).
            velocity: UAV velocity (grid cells per step).
            n_uavs: Fleet size N.
            delta: Communication overhead bound (steps).

        Returns:
            Latency bound in timesteps.

        Raises:
            ValueError: If n_uavs <= 0 or velocity <= 0.
        """
        if n_uavs <= 0:
            raise ValueError(f"n_uavs must be > 0; got {n_uavs}")
        if velocity <= 0:
            raise ValueError(f"velocity must be > 0; got {velocity}")
        return area / (velocity * n_uavs) + delta

    def solve(self) -> None:
        """Raise CPOMDPNotSolvedException — exact CPOMDP is intractable.

        See ``greedy_policy.RiskWeightedGreedyPolicy`` or
        ``rl.ppo_agent.PPOGOMDPAgent`` for practical approximations.
        """
        raise CPOMDPNotSolvedException(
            "The CPOMDP is computationally intractable at operational scale. "
            "Use RiskWeightedGreedyPolicy (O(NZ) per step) or PPOGOMDPAgent "
            "as practical approximations. "
            "The governance predicate is enforced exactly by the blockchain layer "
            "regardless of approximation quality (Theorem 1, GOMDP framework)."
        )
