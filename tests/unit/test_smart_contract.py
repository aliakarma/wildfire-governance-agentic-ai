"""Unit tests for blockchain/smart_contract.py — the core GOMDP enforcement."""
from __future__ import annotations

import pytest

from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
from wildfire_governance.blockchain.smart_contract import (
    ContractState,
    GovernanceSmartContract,
)
from wildfire_governance.blockchain.transaction import build_transaction


def _make_tx(confidence: float = 0.87):
    return build_transaction(
        event_id="test_evt",
        geo_boundary=(5, 5, 6, 6),
        confidence_score=confidence,
        sensor_readings={"heat": confidence, "weather": 0.6},
    )


def test_approval_when_both_conditions_met(smart_contract: GovernanceSmartContract) -> None:
    """Alert is approved when confidence > tau AND signature is valid."""
    priv, pub = generate_key_pair()
    tx = _make_tx(confidence=0.90)
    sig = sign(tx.to_bytes(), priv)
    result = smart_contract.verify_and_execute(tx, sig, pub)
    assert result.alert_enabled is True
    assert result.contract_state == ContractState.APPROVED
    assert result.cert is not None


def test_blocked_when_confidence_below_tau(smart_contract: GovernanceSmartContract) -> None:
    """Alert is BLOCKED when confidence <= tau, even with valid signature."""
    priv, pub = generate_key_pair()
    tx = _make_tx(confidence=0.75)  # Below tau=0.80
    sig = sign(tx.to_bytes(), priv)
    result = smart_contract.verify_and_execute(tx, sig, pub)
    assert result.alert_enabled is False
    assert result.contract_state == ContractState.BLOCKED
    assert result.confidence_ok is False


def test_blocked_when_signature_invalid(smart_contract: GovernanceSmartContract) -> None:
    """Alert is BLOCKED when signature is invalid, even with high confidence."""
    _, pub = generate_key_pair()
    tx = _make_tx(confidence=0.92)
    invalid_sig = b"\x00" * 64  # Garbage signature
    result = smart_contract.verify_and_execute(tx, invalid_sig, pub)
    assert result.alert_enabled is False
    assert result.contract_state == ContractState.BLOCKED
    assert result.signature_ok is False


def test_audit_log_updated_on_approval(smart_contract: GovernanceSmartContract) -> None:
    """Audit log must have exactly one entry after one approval."""
    priv, pub = generate_key_pair()
    tx = _make_tx(confidence=0.95)
    sig = sign(tx.to_bytes(), priv)
    smart_contract.verify_and_execute(tx, sig, pub)
    assert len(smart_contract.audit_log) >= 1


def test_audit_log_updated_on_block(smart_contract: GovernanceSmartContract) -> None:
    """Audit log must record blocked attempts."""
    priv, pub = generate_key_pair()
    tx = _make_tx(confidence=0.50)  # Below tau
    sig = sign(tx.to_bytes(), priv)
    smart_contract.verify_and_execute(tx, sig, pub)
    assert len(smart_contract.audit_log) >= 1


def test_adversarial_injection_always_blocked(smart_contract: GovernanceSmartContract) -> None:
    """Unauthorised injection attempt must always return False (Theorem 2)."""
    result = smart_contract.attempt_unauthorised_injection((0, 0, 5, 5))
    assert result is False
    # Injection attempt is logged
    assert len(smart_contract.audit_log) >= 1


def test_n_approved_counter(smart_contract: GovernanceSmartContract) -> None:
    """n_approved must increment only on successful approvals."""
    priv, pub = generate_key_pair()
    assert smart_contract.n_approved == 0
    tx = _make_tx(confidence=0.91)
    sig = sign(tx.to_bytes(), priv)
    smart_contract.verify_and_execute(tx, sig, pub)
    assert smart_contract.n_approved == 1


def test_n_blocked_counter(smart_contract: GovernanceSmartContract) -> None:
    """n_blocked must count low-confidence AND invalid-signature blocks."""
    priv, pub = generate_key_pair()
    # Low confidence
    tx = _make_tx(confidence=0.60)
    sig = sign(tx.to_bytes(), priv)
    smart_contract.verify_and_execute(tx, sig, pub)
    assert smart_contract.n_blocked >= 1
