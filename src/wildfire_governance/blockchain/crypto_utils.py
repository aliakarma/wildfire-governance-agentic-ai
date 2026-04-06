"""Cryptographic primitives for the GOMDP governance layer."""
from __future__ import annotations
import hashlib, os, uuid
from typing import Tuple
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat, PublicFormat

def sha3_256_hash(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()

def generate_key_pair() -> Tuple[bytes, bytes]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    public_bytes = private_key.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    return private_bytes, public_bytes

def sign(data: bytes, private_key_bytes: bytes) -> bytes:
    from cryptography.hazmat.primitives.serialization import load_der_private_key
    private_key = load_der_private_key(private_key_bytes, password=None)
    assert isinstance(private_key, Ed25519PrivateKey)
    return private_key.sign(data)

def verify_signature(data: bytes, signature: bytes, public_key_bytes: bytes) -> bool:
    from cryptography.hazmat.primitives.serialization import load_der_public_key
    try:
        public_key = load_der_public_key(public_key_bytes)
        assert isinstance(public_key, Ed25519PublicKey)
        public_key.verify(signature, data)
        return True
    except Exception:
        return False

def generate_nonce() -> str:
    return str(uuid.uuid4())

def compute_evidence_hash(sensor_readings: dict) -> str:
    import json
    canonical = json.dumps(sensor_readings, sort_keys=True, default=str)
    return sha3_256_hash(canonical.encode("utf-8"))
