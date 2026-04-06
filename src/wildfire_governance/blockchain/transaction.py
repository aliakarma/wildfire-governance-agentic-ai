"""Anomaly transaction builder for the Hyperledger Fabric governance layer.

Each anomaly event is serialised as a transaction containing:
event_id, UTC timestamp, geospatial coordinates, confidence score,
SHA-3 hash of sensor evidence, and a per-event nonce for replay resistance.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from wildfire_governance.blockchain.crypto_utils import (
    compute_evidence_hash,
    generate_nonce,
    sha3_256_hash,
)


@dataclass
class AnomalyTransaction:
    """Serialisable anomaly event transaction.

    Attributes:
        event_id: Unique event identifier (hex string).
        timestamp_utc: Unix timestamp when the anomaly was detected.
        geo_boundary: (min_row, min_col, max_row, max_col) bounding box.
        confidence_score: Final Conf^(2)_t value.
        evidence_hash: SHA-3 hash of supporting sensor readings.
        nonce: Per-event UUID4 nonce for replay resistance.
        transaction_hash: SHA-3 hash of the full transaction payload (computed on init).
    """

    event_id: str
    timestamp_utc: float
    geo_boundary: Tuple[int, int, int, int]
    confidence_score: float
    evidence_hash: str
    nonce: str = field(default_factory=generate_nonce)
    transaction_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.transaction_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a deterministic hash of the transaction payload."""
        payload = json.dumps(
            {
                "event_id": self.event_id,
                "timestamp_utc": self.timestamp_utc,
                "geo_boundary": list(self.geo_boundary),
                "confidence_score": round(self.confidence_score, 8),
                "evidence_hash": self.evidence_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
        )
        return sha3_256_hash(payload.encode("utf-8"))

    def to_bytes(self) -> bytes:
        """Serialise the transaction to bytes for signing.

        Returns:
            UTF-8 encoded canonical JSON of the transaction payload.
        """
        return json.dumps(
            {
                "event_id": self.event_id,
                "timestamp_utc": self.timestamp_utc,
                "geo_boundary": list(self.geo_boundary),
                "confidence_score": round(self.confidence_score, 8),
                "evidence_hash": self.evidence_hash,
                "nonce": self.nonce,
            },
            sort_keys=True,
        ).encode("utf-8")

    def to_dict(self) -> Dict:
        """Return the transaction as a plain dict for logging."""
        d = asdict(self)
        d["geo_boundary"] = list(self.geo_boundary)
        return d


def build_transaction(
    event_id: str,
    geo_boundary: Tuple[int, int, int, int],
    confidence_score: float,
    sensor_readings: Dict,
) -> AnomalyTransaction:
    """Convenience factory: build and hash an AnomalyTransaction.

    Args:
        event_id: Unique event identifier string.
        geo_boundary: (min_row, min_col, max_row, max_col) bounding box.
        confidence_score: Stage-2 confidence score Conf^(2)_t.
        sensor_readings: Raw sensor evidence dict for hashing.

    Returns:
        Fully initialised AnomalyTransaction with computed hashes.
    """
    evidence_hash = compute_evidence_hash(sensor_readings)
    return AnomalyTransaction(
        event_id=event_id,
        timestamp_utc=time.time(),
        geo_boundary=geo_boundary,
        confidence_score=confidence_score,
        evidence_hash=evidence_hash,
    )
