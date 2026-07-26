"""Governance smart contract — cryptographic enforcement of Eq. (9) in the paper.

Implements the T_G transition function from Definition 1 (GOMDP):
    Alert_t <- 1  iff  Conf^(2)_t > tau  AND  sigma_validator is valid (Ed25519)

This is the environment-level enforcement mechanism that makes Theorem 1
(Policy-Agnostic Safety) hold: no alert can be broadcast without passing
this atomic verification, regardless of which policy requested it.
"""
from __future__ import annotations

import time
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
    """Immutable alert authorisation state machine."""

    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    BLOCKED = auto()


@dataclass
class ContractVerificationResult:
    """Result of smart contract verification.

    Attributes:
        transaction_hash: Hash of the evaluated transaction.
        contract_state: Final state of the state machine.
        alert_enabled: True if the alert may be publicly broadcast.
        confidence_ok: Whether confidence threshold was satisfied.
        signature_ok: Whether the Ed25519 signature was valid.
        key_authorised: Whether the presenting public key is a registered validator.
        replay_ok: Whether the transaction nonce was previously unseen.
        consensus_result: Result of the PBFT consensus round.
        cert: Governance certificate hash (non-None iff APPROVED).
    """

    transaction_hash: str
    contract_state: ContractState
    alert_enabled: bool
    confidence_ok: bool
    signature_ok: bool
    consensus_result: Optional[ConsensusResult]
    cert: Optional[str]
    key_authorised: bool = True
    replay_ok: bool = True


class GovernanceSmartContract:
    """Hyperledger Fabric chaincode implementing the GOMDP governance predicate.

    Atomically verifies BOTH the confidence threshold AND the Ed25519 validator
    signature before enabling public alert dissemination (Eq. 9 in paper).

    This is the core cryptographic enforcement mechanism. By Definition 1,
    any alert action that does not satisfy this contract is blocked at the
    environment transition level, providing the per-trajectory safety guarantee
    of Theorem 1 (Policy-Agnostic Safety) for any policy operating in the GOMDP.

    Args:
        tau: Alert confidence threshold (paper default: 0.80).
        consensus: PBFTConsensus instance for validator agreement.
        audit_log: ImmutableAuditLog for non-repudiation.
        validator_public_keys: List of authorised validator public key bytes.
    """

    def __init__(
        self,
        tau: float = 0.80,
        consensus: Optional[PBFTConsensus] = None,
        audit_log: Optional[ImmutableAuditLog] = None,
        validator_public_keys: Optional[list[bytes]] = None,
    ) -> None:
        self.tau = tau
        self._consensus = consensus or PBFTConsensus()
        self._audit_log = audit_log or ImmutableAuditLog()
        self._validator_public_keys: list[bytes] = list(validator_public_keys or [])
        self._n_approved: int = 0
        self._n_rejected: int = 0
        self._n_blocked: int = 0
        # Per-event nonces already committed, for replay resistance (paper
        # Section "Blockchain-Enforced Governance Invariant").
        self._seen_nonces: set[str] = set()
        # Last approved (tx, signature, key) triple, retained so adversarial
        # probes can attempt a genuine replay against real credentials.
        self._last_approved_credential: Optional[tuple] = None

    def register_validator(self, public_key: bytes) -> None:
        """Add a public key to the authorised validator set.

        While the set is empty the contract runs in *open* mode: any
        well-formed signature verifies. Registering at least one key activates
        key-authorisation enforcement, which is what Theorem 1's Case 1
        ("forge a certificate without an authorised validator key") requires.

        Args:
            public_key: Ed25519 public key bytes of an authorised validator.
        """
        if public_key not in self._validator_public_keys:
            self._validator_public_keys.append(public_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_and_execute(
        self,
        transaction: AnomalyTransaction,
        validator_signature: bytes,
        validator_public_key: bytes,
        burst_mode: bool = False,
    ) -> ContractVerificationResult:
        """Atomically verify the governance predicate and execute state transition.

        Implements Eq. (9):
            Alert_t <- 1  iff  Conf^(2)_t > tau  AND  sig is valid

        Both conditions are checked atomically. If either fails, the alert
        is BLOCKED and logged. No alert is broadcast. This method cannot
        be bypassed by any policy — it is called by the GOMDP environment,
        not by the policy itself.

        Args:
            transaction: The anomaly transaction to evaluate.
            validator_signature: Ed25519 signature of the transaction payload.
            validator_public_key: Corresponding public key bytes.
            burst_mode: If True, applies burst delay multiplier to consensus.

        Returns:
            ContractVerificationResult documenting the full verification outcome.
        """
        tx_hash = transaction.transaction_hash

        # Step 1: Check confidence threshold
        confidence_ok = transaction.confidence_score > self.tau

        # Step 2: Verify Ed25519 signature
        signature_ok = verify_signature(
            transaction.to_bytes(), validator_signature, validator_public_key
        )

        # Step 3: Check the presenting key is an authorised validator. Without
        # this an adversary can generate its own keypair, sign its own forged
        # transaction, and present its own public key — a valid signature over
        # an unauthorised certificate. Enforced only when a validator set has
        # been registered; see register_validator().
        key_authorised = (
            validator_public_key in self._validator_public_keys
            if self._validator_public_keys
            else True
        )

        # Step 4: Replay resistance — a per-event nonce may be committed once.
        replay_ok = transaction.nonce not in self._seen_nonces

        # Step 5: Run PBFT consensus (simulated)
        consensus_result: Optional[ConsensusResult] = None
        if confidence_ok and signature_ok and key_authorised and replay_ok:
            consensus_result = self._consensus.propose(transaction, burst_mode)

        # Step 6: Determine final state
        if (
            confidence_ok and signature_ok and key_authorised and replay_ok
            and consensus_result and consensus_result.approved
        ):
            state = ContractState.APPROVED
            alert_enabled = True
            cert = consensus_result.transaction_hash
            self._n_approved += 1
            self._seen_nonces.add(transaction.nonce)
            self._last_approved_credential = (
                transaction, validator_signature, validator_public_key,
            )
            self._audit_log.append(
                "APPROVED", tx_hash,
                {"confidence": transaction.confidence_score, "cert": cert},
            )
            logger.info("smart_contract_approved", tx_hash=tx_hash[:12], cert=cert[:12])
        elif not confidence_ok:
            state = ContractState.BLOCKED
            alert_enabled = False
            cert = None
            self._n_blocked += 1
            self._audit_log.append(
                "BLOCKED_LOW_CONFIDENCE", tx_hash,
                {"confidence": transaction.confidence_score, "tau": self.tau},
            )
        elif not signature_ok:
            state = ContractState.BLOCKED
            alert_enabled = False
            cert = None
            self._n_blocked += 1
            self._audit_log.append(
                "BLOCKED_INVALID_SIGNATURE", tx_hash, {}
            )
        elif not key_authorised:
            state = ContractState.BLOCKED
            alert_enabled = False
            cert = None
            self._n_blocked += 1
            self._audit_log.append(
                "BLOCKED_UNAUTHORISED_VALIDATOR_KEY", tx_hash, {}
            )
        elif not replay_ok:
            state = ContractState.BLOCKED
            alert_enabled = False
            cert = None
            self._n_blocked += 1
            self._audit_log.append(
                "BLOCKED_REPLAYED_NONCE", tx_hash, {"nonce": transaction.nonce},
            )
        else:
            state = ContractState.REJECTED
            alert_enabled = False
            cert = None
            self._n_rejected += 1
            self._audit_log.append(
                "REJECTED_CONSENSUS_FAILED", tx_hash,
                {"n_byzantine": self._consensus.n_byzantine},
            )

        return ContractVerificationResult(
            transaction_hash=tx_hash,
            contract_state=state,
            alert_enabled=alert_enabled,
            confidence_ok=confidence_ok,
            signature_ok=signature_ok,
            consensus_result=consensus_result,
            cert=cert,
            key_authorised=key_authorised,
            replay_ok=replay_ok,
        )

    def probe_injection(
        self,
        geo_boundary: tuple,
        severity: str = "critical",
        attack: str = "unsigned",
        confidence: float = 0.99,
    ) -> ContractVerificationResult:
        """Mount a real alert-injection attack against the contract.

        The attempt is executed through :meth:`verify_and_execute`, the same
        code path a legitimate alert traverses — nothing is short-circuited, so
        the outcome is a measurement of the enforcement mechanism rather than a
        restatement of it. The adversary controls the payload and may set an
        arbitrarily high confidence score; what it lacks is an authorised
        validator key.

        Attack variants:
            ``"unsigned"``   garbage signature bytes (defeated by Ed25519
                             verification alone — the claim the ablation's
                             injection column tests).
            ``"wrong_key"``  adversary generates its own Ed25519 keypair and
                             produces a *cryptographically valid* signature
                             over its forged transaction, presenting its own
                             public key. Defeated only by validator-key
                             authorisation; requires a registered validator set.
            ``"replay"``     resubmits the last approved (transaction,
                             signature, key) triple. Defeated by the nonce
                             ledger.

        Args:
            geo_boundary: Target geographic boundary for the injected alert.
            severity: Severity string recorded in the forged evidence payload.
            attack: One of ``"unsigned"``, ``"wrong_key"``, ``"replay"``.
            confidence: Confidence score the adversary claims.

        Returns:
            The full ContractVerificationResult. ``alert_enabled`` is True only
            if the injection actually breached the governance predicate.
        """
        from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
        from wildfire_governance.blockchain.transaction import build_transaction

        self._audit_log.append(
            "ADVERSARIAL_INJECTION_ATTEMPT",
            "UNAUTHORISED",
            {"geo_boundary": list(geo_boundary), "severity": severity, "attack": attack},
        )

        if attack == "replay":
            if self._last_approved_credential is None:
                # Nothing legitimate has been approved yet, so there is no
                # credential to replay; treat as a blocked no-op.
                return ContractVerificationResult(
                    transaction_hash="", contract_state=ContractState.BLOCKED,
                    alert_enabled=False, confidence_ok=False, signature_ok=False,
                    consensus_result=None, cert=None, key_authorised=False,
                    replay_ok=False,
                )
            tx, sig, pub = self._last_approved_credential
            return self.verify_and_execute(tx, sig, pub)

        forged_tx = build_transaction(
            event_id=f"forged_{geo_boundary}",
            geo_boundary=geo_boundary,
            confidence_score=confidence,
            sensor_readings={"forged": True, "severity": severity},
        )

        if attack == "wrong_key":
            adversary_priv, adversary_pub = generate_key_pair()
            forged_sig = sign(forged_tx.to_bytes(), adversary_priv)
            return self.verify_and_execute(forged_tx, forged_sig, adversary_pub)

        # "unsigned": well-formed but meaningless signature bytes, presented
        # against a registered validator key (or a decoy when none is set).
        garbage_sig = b"\x00" * 64
        target_key = (
            self._validator_public_keys[0]
            if self._validator_public_keys
            else generate_key_pair()[1]
        )
        return self.verify_and_execute(forged_tx, garbage_sig, target_key)

    def attempt_unauthorised_injection(
        self,
        geo_boundary: tuple,
        severity: str = "critical",
        attack: str = "unsigned",
    ) -> bool:
        """Attempt an unauthorised alert injection.

        Thin boolean wrapper over :meth:`probe_injection` for call sites that
        only need the outcome.

        Args:
            geo_boundary: Target geographic boundary for the injected alert.
            severity: Severity string for the injected alert.
            attack: Attack variant; see :meth:`probe_injection`.

        Returns:
            True iff the injection succeeded in enabling an alert (a breach).
            False iff the contract blocked it.
        """
        result = self.probe_injection(geo_boundary, severity=severity, attack=attack)
        if not result.alert_enabled:
            logger.info(
                "adversarial_injection_blocked",
                geo_boundary=str(geo_boundary),
                reason=result.contract_state.name,
            )
        else:
            logger.warning("adversarial_injection_BREACH", geo_boundary=str(geo_boundary))
        return result.alert_enabled

    @property
    def n_approved(self) -> int:
        """Total alerts approved since initialisation."""
        return self._n_approved

    @property
    def n_blocked(self) -> int:
        """Total alert attempts blocked by the governance predicate."""
        return self._n_blocked

    @property
    def audit_log(self) -> ImmutableAuditLog:
        """Reference to the immutable audit log."""
        return self._audit_log
