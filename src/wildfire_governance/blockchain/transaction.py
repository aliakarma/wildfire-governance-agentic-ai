"""Anomaly transaction builder."""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from typing import Dict, Tuple
from wildfire_governance.blockchain.crypto_utils import compute_evidence_hash, generate_nonce, sha3_256_hash

@dataclass
class AnomalyTransaction:
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
        payload = json.dumps({"event_id": self.event_id, "timestamp_utc": self.timestamp_utc,
            "geo_boundary": list(self.geo_boundary), "confidence_score": round(self.confidence_score, 8),
            "evidence_hash": self.evidence_hash, "nonce": self.nonce}, sort_keys=True)
        return sha3_256_hash(payload.encode("utf-8"))

    def to_bytes(self) -> bytes:
        return json.dumps({"event_id": self.event_id, "timestamp_utc": self.timestamp_utc,
            "geo_boundary": list(self.geo_boundary), "confidence_score": round(self.confidence_score, 8),
            "evidence_hash": self.evidence_hash, "nonce": self.nonce}, sort_keys=True).encode("utf-8")

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        d = asdict(self); d["geo_boundary"] = list(self.geo_boundary); return d

def build_transaction(event_id: str, geo_boundary: Tuple[int,int,int,int],
                      confidence_score: float, sensor_readings: Dict) -> AnomalyTransaction:
    return AnomalyTransaction(event_id=event_id, timestamp_utc=time.time(),
        geo_boundary=geo_boundary, confidence_score=confidence_score,
        evidence_hash=compute_evidence_hash(sensor_readings))
