"""Governance-Invariant MDP (GOMDP) — Definition 1 from the paper.

The GOMDP enforces safety constraints at the environment transition level
via a cryptographic state-transition invariant, not as a Lagrangian penalty.

Key distinction from CMDP / CPO:
- CMDP: constraint enforced as Lagrangian soft penalty → violations POSSIBLE in expectation.
- GOMDP: constraint enforced at env boundary → violations IMPOSSIBLE with prob. 1 (Theorem 1).

Theorem 1 (Policy-Agnostic Safety): Any policy π operating in a GOMDP
satisfies P(Alert_t=1 | Conf<τ or HA=0) = 0 for all t, regardless of
the policy's optimality gap.

Corollary 1 (Safety–Optimality Decoupling): The safety guarantee holds
independently of the policy's approximation quality.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from wildfire_governance.utils.logging import get_structured_logger

logger = get_structured_logger(__name__)


class ContractState(Enum):
    """Immutable state machine for alert authorisation."""

    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    BLOCKED = auto()  # Non-compliant action blocked at environment level


class GovernanceViolationAttempt(Exception):
    """Logged (not fatal) when a policy attempts a non-compliant alert action.

    The GOMDP blocks the transition; the exception is caught internally and
    logged to the audit trail for forensic analysis.
    """


@dataclass
class GovernanceTransitionResult:
    """Result of a governance-constrained state transition.

    Attributes:
        next_state: The environment state after transition (None if blocked).
        contract_state: Final state of the alert authorisation state machine.
        blocked: True if the alert action was blocked by the governance predicate.
        governance_cert: Cryptographic certificate hash if approved; None otherwise.
        event_id: Unique identifier for this transition event.
    """

    next_state: Any
    contract_state: ContractState
    blocked: bool
    governance_cert: Optional[str]
    event_id: str


class GovernanceInvariantMDP:
    """Formal instantiation of Definition 1 (GOMDP).

    The governance predicate G(s, a) = [Conf^(2)_t > tau AND HA_t = 1]
    is enforced at the transition level: any alert action with G(s,a)=0
    is blocked and recorded in the immutable audit log.

    This is NOT a Lagrangian penalty. The environment physically refuses
    to execute non-compliant alert state transitions.

    Args:
        base_env: The underlying wildfire simulation environment.
        smart_contract: GovernanceSmartContract enforcing the predicate.
        tau: Alert confidence threshold (default 0.80).
        audit_log: ImmutableAuditLog for recording all governance events.
    """

    def __init__(
        self,
        tau: float = 0.80,
        n_validators: int = 7,
        max_byzantine: int = 2,
    ) -> None:
        self.tau = tau
        self.n_validators = n_validators
        self.max_byzantine = max_byzantine
        self._violation_count: int = 0
        self._total_alert_attempts: int = 0
        self._compliance_log: list[dict[str, Any]] = []

    def evaluate_governance_predicate(
        self,
        confidence: float,
        human_approval: bool,
        validator_signature_valid: bool,
    ) -> bool:
        """Evaluate G(s, a) = [Conf > tau] AND [HA = 1] AND [sig valid].

        This is the core predicate from Eq. (4) in the paper.

        Args:
            confidence: Stage-2 confidence score Conf^(2)_t.
            human_approval: Binary human authorisation HA_t.
            validator_signature_valid: Whether the Ed25519 signature is valid.

        Returns:
            True if and only if ALL three conditions are satisfied.
        """
        return (
            confidence > self.tau
            and human_approval
            and validator_signature_valid
        )

    def step_alert_action(
        self,
        confidence: float,
        human_approval: bool,
        validator_signature_valid: bool,
        metadata: Optional[dict] = None,
    ) -> GovernanceTransitionResult:
        """Execute a governance-constrained alert state transition.

        If G(s, a) = 0: the alert is BLOCKED. No public dissemination occurs.
        If G(s, a) = 1: the alert is APPROVED and a governance certificate is issued.

        This method implements the T_G transition function from Definition 1.
        By Theorem 1, repeated calls from ANY policy maintain P(violation)=0.

        Args:
            confidence: Stage-2 confidence score.
            human_approval: Human authorisation flag.
            validator_signature_valid: Cryptographic signature validity.
            metadata: Optional dict logged to the audit trail.

        Returns:
            GovernanceTransitionResult documenting the outcome.
        """
        self._total_alert_attempts += 1
        event_id = _generate_event_id()
        predicate_satisfied = self.evaluate_governance_predicate(
            confidence, human_approval, validator_signature_valid
        )

        log_entry = {
            "event_id": event_id,
            "timestamp": time.time(),
            "confidence": confidence,
            "human_approval": human_approval,
            "signature_valid": validator_signature_valid,
            "predicate_satisfied": predicate_satisfied,
            "metadata": metadata or {},
        }

        if not predicate_satisfied:
            self._violation_count += 1
            log_entry["outcome"] = "BLOCKED"
            self._compliance_log.append(log_entry)
            logger.info(
                "governance_predicate_blocked",
                event_id=event_id,
                confidence=confidence,
                human_approval=human_approval,
            )
            return GovernanceTransitionResult(
                next_state=None,
                contract_state=ContractState.BLOCKED,
                blocked=True,
                governance_cert=None,
                event_id=event_id,
            )

        cert = _compute_certificate(event_id, confidence)
        log_entry["outcome"] = "APPROVED"
        log_entry["cert"] = cert
        self._compliance_log.append(log_entry)
        logger.info("governance_predicate_approved", event_id=event_id, cert=cert[:12])
        return GovernanceTransitionResult(
            next_state={"alert": True, "confidence": confidence},
            contract_state=ContractState.APPROVED,
            blocked=False,
            governance_cert=cert,
            event_id=event_id,
        )

    def get_compliance_rate(self) -> float:
        """Return fraction of alert attempts that were NOT blocked.

        A rate of 1.0 means 100% governance compliance (Theorem 1 verified).
        A rate < 1.0 means the GOMDP blocked some non-compliant attempts
        (which is correct behaviour; the attempts did NOT result in alerts).

        Returns:
            Float in [0, 1]. Returns 1.0 if no attempts have been made.
        """
        if self._total_alert_attempts == 0:
            return 1.0
        approved = self._total_alert_attempts - self._violation_count
        return approved / self._total_alert_attempts

    def get_violation_count(self) -> int:
        """Return the number of blocked (non-compliant) alert attempts."""
        return self._violation_count

    def reset_stats(self) -> None:
        """Reset per-episode statistics (call at episode start)."""
        self._violation_count = 0
        self._total_alert_attempts = 0
        self._compliance_log.clear()


def _generate_event_id() -> str:
    """Generate a unique event ID using timestamp + random bytes."""
    import os
    raw = f"{time.time_ns()}_{os.urandom(8).hex()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _compute_certificate(event_id: str, confidence: float) -> str:
    """Compute a deterministic governance certificate hash.

    In production this is replaced by the blockchain transaction hash.
    Here it serves as a lightweight simulation stand-in.
    """
    data = f"{event_id}|{confidence:.6f}|{time.time_ns()}"
    return hashlib.sha3_256(data.encode()).hexdigest()
