"""Unit tests for crypto_utils.py."""
import pytest
from wildfire_governance.blockchain.crypto_utils import (
    compute_evidence_hash, generate_key_pair, generate_nonce, sha3_256_hash, sign, verify_signature)

def test_sha3_deterministic():
    assert sha3_256_hash(b"wildfire") == sha3_256_hash(b"wildfire")

def test_sha3_length():
    assert len(sha3_256_hash(b"test")) == 64

def test_sign_verify_roundtrip():
    priv, pub = generate_key_pair(); data = b"governance certificate"
    assert verify_signature(data, sign(data, priv), pub) is True

def test_tampered_data_rejected():
    priv, pub = generate_key_pair()
    assert verify_signature(b"tampered", sign(b"original", priv), pub) is False

def test_wrong_key_rejected():
    priv1, pub1 = generate_key_pair(); _, pub2 = generate_key_pair()
    assert verify_signature(b"data", sign(b"data", priv1), pub2) is False

def test_nonce_unique():
    nonces = {generate_nonce() for _ in range(100)}; assert len(nonces) == 100

def test_evidence_hash_order_invariant():
    assert compute_evidence_hash({"a":1,"b":2}) == compute_evidence_hash({"b":2,"a":1})
