"""Cryptographic primitives for the GOMDP governance layer.

Uses the ``cryptography`` library for Ed25519 signatures and SHA-3 hashing.
All functions return False (never raise) on verification failure so that
the caller can treat a failed signature check as a governance predicate failure
rather than an unhandled exception.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from typing import Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def sha3_256_hash(data: bytes) -> str:
    """Compute a SHA-3 256-bit hash of *data*.

    Args:
        data: Raw bytes to hash.

    Returns:
        Lowercase hex-encoded hash string (64 characters).
    """
    return hashlib.sha3_256(data).hexdigest()


def generate_key_pair() -> Tuple[bytes, bytes]:
    """Generate an Ed25519 private/public key pair.

    Returns:
        Tuple (private_key_bytes, public_key_bytes) in raw DER/PKCS8 format.
    """
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=Encoding.DER,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.DER,
        format=PublicFormat.SubjectPublicKeyInfo,
    )
    return private_bytes, public_bytes


def sign(data: bytes, private_key_bytes: bytes) -> bytes:
    """Sign *data* with an Ed25519 private key.

    Args:
        data: Payload to sign.
        private_key_bytes: DER-encoded PKCS8 private key bytes.

    Returns:
        Ed25519 signature bytes (64 bytes).
    """
    from cryptography.hazmat.primitives.serialization import load_der_private_key

    private_key = load_der_private_key(private_key_bytes, password=None)
    assert isinstance(private_key, Ed25519PrivateKey)
    return private_key.sign(data)


def verify_signature(data: bytes, signature: bytes, public_key_bytes: bytes) -> bool:
    """Verify an Ed25519 signature.

    Args:
        data: Original signed payload.
        signature: Signature bytes to verify.
        public_key_bytes: DER-encoded SubjectPublicKeyInfo public key bytes.

    Returns:
        True if the signature is valid; False on any failure (never raises).
    """
    from cryptography.hazmat.primitives.serialization import load_der_public_key

    try:
        public_key = load_der_public_key(public_key_bytes)
        assert isinstance(public_key, Ed25519PublicKey)
        public_key.verify(signature, data)
        return True
    except (InvalidSignature, Exception):  # noqa: BLE001
        return False


def generate_nonce() -> str:
    """Generate a cryptographically secure unique nonce.

    Returns:
        UUID4 string (36 characters, e.g. ``"550e8400-e29b-41d4-a716-446655440000"``).
    """
    return str(uuid.uuid4())


def compute_evidence_hash(sensor_readings: dict) -> str:
    """Compute a deterministic SHA-3 hash of sensor evidence.

    Serialises *sensor_readings* to a canonical JSON string (sorted keys)
    before hashing to ensure the same dict always produces the same hash.

    Args:
        sensor_readings: Dict of sensor evidence (must be JSON-serialisable).

    Returns:
        64-character lowercase hex hash string.
    """
    import json

    canonical = json.dumps(sensor_readings, sort_keys=True, default=str)
    return sha3_256_hash(canonical.encode("utf-8"))
