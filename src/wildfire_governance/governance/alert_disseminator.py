"""Alert disseminator."""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class AlertPayload:
    event_id: str; geo_boundary: Tuple[int,int,int,int]; severity: str
    advisory_text: str; governance_cert: str; broadcast_timestamp: float; channels: List[str]

class AlertDisseminator:
    def __init__(self, channels=None):
        self._channels = channels or ["primary", "backup"]; self._broadcast_history = []

    def disseminate(self, event_id, geo_boundary, confidence, governance_cert):
        if not governance_cert: raise ValueError("Cannot disseminate without governance certificate.")
        sev = "CRITICAL" if confidence >= 0.90 else "HIGH" if confidence >= 0.80 else "MODERATE"
        r0,c0,r1,c1 = geo_boundary
        payload = AlertPayload(event_id, geo_boundary, sev,
            f"[{sev}] Wildfire alert for ({r0},{c0})-({r1},{c1}). Evacuate immediately.",
            governance_cert, time.time(), self._channels)
        self._broadcast_history.append(payload); return payload

    @property
    def broadcast_count(self): return len(self._broadcast_history)
