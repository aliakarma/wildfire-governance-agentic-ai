"""Geo-fenced alert payload generation and dissemination.

After the smart contract approves an alert, this module generates and
broadcasts the geo-fenced payload over redundant communication channels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AlertPayload:
    """Approved public alert payload.

    Attributes:
        event_id: Anomaly event identifier.
        geo_boundary: (min_row, min_col, max_row, max_col) bounding box.
        severity: AI-estimated severity level (confirmed by human).
        advisory_text: Public advisory message.
        governance_cert: Blockchain governance certificate hash.
        broadcast_timestamp: Unix time of broadcast.
        channels: Communication channels used for dissemination.
    """

    event_id: str
    geo_boundary: Tuple[int, int, int, int]
    severity: str
    advisory_text: str
    governance_cert: str
    broadcast_timestamp: float
    channels: List[str]


class AlertDisseminator:
    """Generates and broadcasts geo-fenced alert payloads.

    Only called after successful smart contract verification.
    Broadcasts over ``channels`` for resilience against partial failure.

    Args:
        channels: Communication channel identifiers (default: primary + backup).
    """

    def __init__(self, channels: Optional[List[str]] = None) -> None:
        self._channels = channels or ["primary", "backup"]
        self._broadcast_history: List[AlertPayload] = []

    def disseminate(
        self,
        event_id: str,
        geo_boundary: Tuple[int, int, int, int],
        confidence: float,
        governance_cert: str,
    ) -> AlertPayload:
        """Generate and broadcast a geo-fenced alert payload.

        Severity is determined from the confidence score:
        - confidence >= 0.90: CRITICAL
        - confidence >= 0.80: HIGH
        - confidence >= 0.70: MODERATE
        - otherwise: LOW

        Args:
            event_id: Anomaly event identifier.
            geo_boundary: Alert geographic bounding box.
            confidence: Final confidence score (for severity derivation).
            governance_cert: Blockchain certificate hash (required).

        Returns:
            The broadcast AlertPayload.

        Raises:
            ValueError: If governance_cert is empty (alert without governance).
        """
        if not governance_cert:
            raise ValueError(
                "Cannot disseminate alert without a governance certificate. "
                "All alerts must pass the smart contract verification."
            )

        severity = self._classify_severity(confidence)
        advisory = self._generate_advisory(severity, geo_boundary)

        payload = AlertPayload(
            event_id=event_id,
            geo_boundary=geo_boundary,
            severity=severity,
            advisory_text=advisory,
            governance_cert=governance_cert,
            broadcast_timestamp=time.time(),
            channels=self._channels,
        )
        self._broadcast_history.append(payload)
        return payload

    @property
    def broadcast_count(self) -> int:
        """Total alerts broadcast since initialisation."""
        return len(self._broadcast_history)

    @property
    def history(self) -> List[AlertPayload]:
        """List of all broadcast AlertPayload objects."""
        return list(self._broadcast_history)

    @staticmethod
    def _classify_severity(confidence: float) -> str:
        if confidence >= 0.90:
            return "CRITICAL"
        if confidence >= 0.80:
            return "HIGH"
        if confidence >= 0.70:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _generate_advisory(
        severity: str, geo_boundary: Tuple[int, int, int, int]
    ) -> str:
        r0, c0, r1, c1 = geo_boundary
        return (
            f"[{severity}] Wildfire alert issued for grid region "
            f"({r0},{c0})–({r1},{c1}). "
            "Evacuate immediately if in the affected area. "
            "Follow official guidance from emergency services."
        )
