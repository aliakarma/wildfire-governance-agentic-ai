"""Governance smart contract — cryptographic enforcement of the GOMDP."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from wildfire_governance.blockchain.audit_log import ImmutableAuditLog
from wildfire_governance.blockchain.consensus import ConsensusResult, PBFTConsensus
from wildfire_governance.blockchain.crypto_utils import verify_signature
from wildfire_governance.blockchain.transaction import AnomalyTransaction
from wildfire_governance.utils.logging import get_structured_logger

logger = get_structured_logger(__name__)

class ContractState(Enum):
    PENDING = auto(); APPROVED = auto(); REJECTED = auto(); BLOCKED = auto()

@dataclass
class ContractVerificationResult:
    transaction_hash: str; contract_state: ContractState; alert_enabled: bool
    confidence_ok: bool; signature_ok: bool
    consensus_result: Optional[ConsensusResult]; cert: Optional[str]

class GovernanceSmartContract:
    def __init__(self, tau: float = 0.80, consensus: Optional[PBFTConsensus] = None,
                 audit_log: Optional[ImmutableAuditLog] = None,
                 validator_public_keys: Optional[list] = None) -> None:
        self.tau = tau
        self._consensus = consensus or PBFTConsensus()
        self._audit_log = audit_log or ImmutableAuditLog()
        self._validator_public_keys = validator_public_keys or []
        self._n_approved = 0; self._n_rejected = 0; self._n_blocked = 0

    def verify_and_execute(self, transaction: AnomalyTransaction, validator_signature: bytes,
                           validator_public_key: bytes, burst_mode: bool = False) -> ContractVerificationResult:
        tx_hash = transaction.transaction_hash
        confidence_ok = transaction.confidence_score > self.tau
        signature_ok = verify_signature(transaction.to_bytes(), validator_signature, validator_public_key)
        consensus_result = None
        if confidence_ok and signature_ok:
            consensus_result = self._consensus.propose(transaction, burst_mode)
        if confidence_ok and signature_ok and consensus_result and consensus_result.approved:
            state = ContractState.APPROVED; alert_enabled = True; cert = consensus_result.transaction_hash
            self._n_approved += 1
            self._audit_log.append("APPROVED", tx_hash, {"confidence": transaction.confidence_score, "cert": cert})
        elif not confidence_ok:
            state = ContractState.BLOCKED; alert_enabled = False; cert = None; self._n_blocked += 1
            self._audit_log.append("BLOCKED_LOW_CONFIDENCE", tx_hash, {"confidence": transaction.confidence_score})
        elif not signature_ok:
            state = ContractState.BLOCKED; alert_enabled = False; cert = None; self._n_blocked += 1
            self._audit_log.append("BLOCKED_INVALID_SIGNATURE", tx_hash, {})
        else:
            state = ContractState.REJECTED; alert_enabled = False; cert = None; self._n_rejected += 1
            self._audit_log.append("REJECTED_CONSENSUS_FAILED", tx_hash, {})
        return ContractVerificationResult(tx_hash, state, alert_enabled, confidence_ok, signature_ok, consensus_result, cert)

    def attempt_unauthorised_injection(self, geo_boundary: tuple, severity: str = "critical") -> bool:
        self._audit_log.append("ADVERSARIAL_INJECTION_ATTEMPT", "UNAUTHORISED",
                               {"geo_boundary": list(geo_boundary), "severity": severity})
        return False  # Always blocked

    @property
    def n_approved(self) -> int: return self._n_approved
    @property
    def n_blocked(self) -> int: return self._n_blocked
    @property
    def audit_log(self) -> ImmutableAuditLog: return self._audit_log
