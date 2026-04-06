"""CPOMDP theoretical formulation."""
from __future__ import annotations
from dataclasses import dataclass

class CPOMDPNotSolvedException(NotImplementedError): pass

@dataclass
class CPOMDPCostWeights:
    alpha: float = 0.50; beta: float = 0.35; gamma: float = 0.15
    def __post_init__(self):
        if abs(self.alpha+self.beta+self.gamma-1.0) > 1e-6:
            raise ValueError(f"Cost weights must sum to 1.0")

class WildfireCPOMDP:
    def __init__(self, cost_weights=None, tau=0.80):
        self.cost_weights = cost_weights or CPOMDPCostWeights(); self.tau = tau
    def compute_cost(self, ld, fp, cr):
        w = self.cost_weights; return w.alpha*ld + w.beta*fp + w.gamma*cr
    def check_governance_predicate(self, confidence, human_approval):
        return confidence > self.tau and human_approval
    def latency_bound(self, area, velocity, n_uavs, delta):
        if n_uavs <= 0: raise ValueError(f"n_uavs must be > 0")
        if velocity <= 0: raise ValueError(f"velocity must be > 0")
        return area / (velocity * n_uavs) + delta
    def solve(self): raise CPOMDPNotSolvedException("CPOMDP is intractable. Use greedy or PPO.")
