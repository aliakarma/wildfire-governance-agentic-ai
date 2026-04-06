"""Unit tests for smart_contract.py."""
import pytest
from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
from wildfire_governance.blockchain.smart_contract import ContractState, GovernanceSmartContract
from wildfire_governance.blockchain.transaction import build_transaction

def _make_tx(confidence=0.87):
    return build_transaction("test_evt", (5,5,6,6), confidence, {"heat": confidence})

def test_approval_when_both_conditions_met(smart_contract):
    priv, pub = generate_key_pair(); tx = _make_tx(0.90)
    result = smart_contract.verify_and_execute(tx, sign(tx.to_bytes(), priv), pub)
    assert result.alert_enabled is True; assert result.contract_state == ContractState.APPROVED
    assert result.cert is not None

def test_blocked_when_confidence_below_tau(smart_contract):
    priv, pub = generate_key_pair(); tx = _make_tx(0.75)
    result = smart_contract.verify_and_execute(tx, sign(tx.to_bytes(), priv), pub)
    assert result.alert_enabled is False; assert result.contract_state == ContractState.BLOCKED

def test_blocked_when_signature_invalid(smart_contract):
    _, pub = generate_key_pair(); tx = _make_tx(0.92)
    result = smart_contract.verify_and_execute(tx, b"\x00"*64, pub)
    assert result.alert_enabled is False; assert result.signature_ok is False

def test_adversarial_injection_always_blocked(smart_contract):
    assert smart_contract.attempt_unauthorised_injection((0,0,5,5)) is False
