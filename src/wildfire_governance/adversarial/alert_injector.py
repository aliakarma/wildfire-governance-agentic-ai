"""Alert injection attack simulator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class InjectionResult:
    success: bool; system_type: str; reason: str

class AlertInjector:
    def __init__(self, p_attack=1.0):
        if not 0.0 <= p_attack <= 1.0: raise ValueError(f"p_attack must be in [0,1]")
        self.p_attack = p_attack; self._n_attempted = 0; self._n_succeeded = 0

    def attempt_injection_gomdp(self, smart_contract, geo_boundary=(0,0,10,10)):
        self._n_attempted += 1
        success = smart_contract.attempt_unauthorised_injection(geo_boundary) if hasattr(smart_contract, "attempt_unauthorised_injection") else False
        return InjectionResult(success, "gomdp", "blocked_by_smart_contract" if not success else "breach")

    def attempt_injection_centralized(self, geo_boundary=(0,0,10,10)):
        self._n_attempted += 1; self._n_succeeded += 1
        return InjectionResult(True, "centralized", "no_cryptographic_enforcement")

    @property
    def success_rate(self): return self._n_succeeded / max(1, self._n_attempted)
    def reset(self): self._n_attempted = 0; self._n_succeeded = 0
