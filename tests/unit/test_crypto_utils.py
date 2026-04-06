"""Unit tests for blockchain/crypto_utils.py."""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.blockchain.crypto_utils import (
    compute_evidence_hash,
    generate_key_pair,
    generate_nonce,
    sha3_256_hash,
    sign,
    verify_signature,
)


def test_sha3_deterministic() -> None:
    """Same input must always produce the same hash."""
    h1 = sha3_256_hash(b"wildfire")
    h2 = sha3_256_hash(b"wildfire")
    assert h1 == h2


def test_sha3_length() -> None:
    """SHA-3 256-bit hash must be 64 hex characters."""
    assert len(sha3_256_hash(b"test")) == 64


def test_sha3_avalanche() -> None:
    """One-bit change in input must produce a completely different hash."""
    h1 = sha3_256_hash(b"\x00")
    h2 = sha3_256_hash(b"\x01")
    assert h1 != h2


def test_sign_verify_roundtrip() -> None:
    """sign() → verify_signature() must return True for the same data."""
    priv, pub = generate_key_pair()
    data = b"governance certificate"
    sig = sign(data, priv)
    assert verify_signature(data, sig, pub) is True


def test_tampered_data_rejected() -> None:
    """Modifying data after signing must cause verify to return False."""
    priv, pub = generate_key_pair()
    data = b"original data"
    sig = sign(data, priv)
    tampered = b"tampered data"
    assert verify_signature(tampered, sig, pub) is False


def test_wrong_key_rejected() -> None:
    """Verifying with a different public key must return False."""
    priv1, pub1 = generate_key_pair()
    _, pub2 = generate_key_pair()
    sig = sign(b"data", priv1)
    assert verify_signature(b"data", sig, pub2) is False


def test_tampered_signature_rejected() -> None:
    """Flipping a bit in the signature must cause verify to return False (never raises)."""
    priv, pub = generate_key_pair()
    data = b"hello"
    sig = bytearray(sign(data, priv))
    sig[0] ^= 0xFF  # Flip all bits of first byte
    assert verify_signature(data, bytes(sig), pub) is False


def test_nonce_unique() -> None:
    """1000 generated nonces must all be distinct."""
    nonces = {generate_nonce() for _ in range(1000)}
    assert len(nonces) == 1000


def test_evidence_hash_deterministic() -> None:
    """Same sensor readings dict must always produce the same hash."""
    readings = {"heat": 0.85, "weather": 0.72, "sensor_id": "uav_0"}
    h1 = compute_evidence_hash(readings)
    h2 = compute_evidence_hash(readings)
    assert h1 == h2


def test_evidence_hash_order_invariant() -> None:
    """Dict key order must not affect the evidence hash (uses sort_keys=True)."""
    r1 = {"a": 1, "b": 2}
    r2 = {"b": 2, "a": 1}
    assert compute_evidence_hash(r1) == compute_evidence_hash(r2)
