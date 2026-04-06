"""GOMDP — Definition 1."""
from __future__ import annotations
import hashlib, time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional
from wildfire_governance.utils.logging import get_structured_logger

logger = get_structured_logger(__name__)

class ContractState(Enum):
    PENDING = auto(); APPROVED = auto(); REJECTED = auto(); BLOCKED = auto()

@dataclass
class GovernanceTransitionResult:
    next_state: Any; contract_state: ContractState; blocked: bool
    governance_cert: Optional[str]; event_id: str

class GovernanceInvariantMDP:
    def __init__(self, tau=0.80, n_validators=7, max_byzantine=2):
        self.tau = tau; self.n_validators = n_validators; self.max_byzantine = max_byzantine
        self._violation_count = 0; self._total_alert_attempts = 0; self._compliance_log = []

    def evaluate_governance_predicate(self, confidence, human_approval, validator_signature_valid):
        return confidence > self.tau and human_approval and validator_signature_valid

    def step_alert_action(self, confidence, human_approval, validator_signature_valid, metadata=None):
        self._total_alert_attempts += 1
        event_id = _generate_event_id()
        satisfied = self.evaluate_governance_predicate(confidence, human_approval, validator_signature_valid)
        log_entry = {"event_id": event_id, "timestamp": time.time(), "confidence": confidence,
                     "human_approval": human_approval, "signature_valid": validator_signature_valid,
                     "predicate_satisfied": satisfied, "metadata": metadata or {}}
        if not satisfied:
            self._violation_count += 1; log_entry["outcome"] = "BLOCKED"
            self._compliance_log.append(log_entry)
            return GovernanceTransitionResult(None, ContractState.BLOCKED, True, None, event_id)
        cert = _compute_certificate(event_id, confidence)
        log_entry["outcome"] = "APPROVED"; log_entry["cert"] = cert
        self._compliance_log.append(log_entry)
        return GovernanceTransitionResult({"alert": True, "confidence": confidence},
                                          ContractState.APPROVED, False, cert, event_id)

    def get_compliance_rate(self):
        if self._total_alert_attempts == 0: return 1.0
        return (self._total_alert_attempts - self._violation_count) / self._total_alert_attempts

    def get_violation_count(self): return self._violation_count
    def reset_stats(self): self._violation_count = 0; self._total_alert_attempts = 0; self._compliance_log.clear()

def _generate_event_id():
    import os
    raw = f"{time.time_ns()}_{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _compute_certificate(event_id, confidence):
    data = f"{event_id}|{confidence:.6f}|{time.time_ns()}"
    return hashlib.sha3_256(data.encode()).hexdigest()
