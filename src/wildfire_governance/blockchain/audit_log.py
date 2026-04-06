"""Immutable hash-chain audit log for governance event recording.

Simulates the on-chain immutable ledger. Every governance event — including
approvals, rejections, blocks, and adversarial injection attempts — is
appended as a signed, hashed entry. The hash chain ensures tamper-evidence:
any modification of a past entry breaks the chain and is detected by
verify_integrity().
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from wildfire_governance.blockchain.crypto_utils import sha3_256_hash


_GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    """A single immutable audit log entry.

    Attributes:
        entry_id: Sequential integer ID.
        timestamp_utc: Unix timestamp of the event.
        event_type: Type string (e.g. "APPROVED", "BLOCKED", "REJECTED").
        event_id: Reference to the anomaly event ID.
        details: Arbitrary metadata dict.
        prev_hash: Hash of the preceding entry (genesis entry has 64 zeros).
        entry_hash: SHA-3 hash of this entry's canonical payload (computed on init).
    """

    entry_id: int
    timestamp_utc: float
    event_type: str
    event_id: str
    details: Dict[str, Any]
    prev_hash: str
    entry_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps(
            {
                "entry_id": self.entry_id,
                "timestamp_utc": self.timestamp_utc,
                "event_type": self.event_type,
                "event_id": self.event_id,
                "details": self.details,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            default=str,
        )
        return sha3_256_hash(payload.encode("utf-8"))


class AuditTamperException(Exception):
    """Raised when the audit log hash chain is found to be broken."""


class ImmutableAuditLog:
    """Append-only hash-chained audit log simulating the on-chain ledger.

    Each entry commits to the hash of the previous entry. Any post-hoc
    modification of any entry breaks the chain and is detected by
    ``verify_integrity()``.

    This provides the non-repudiation guarantee (Guarantee 3 in the paper):
    no party can deny having authorised or submitted an alert.
    """

    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._index: Dict[str, AuditEntry] = {}  # entry_hash -> entry

    def append(
        self,
        event_type: str,
        event_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append a new entry to the audit log.

        Args:
            event_type: Event classification string (e.g. ``"APPROVED"``,
                ``"BLOCKED"``, ``"ADVERSARIAL_INJECTION_ATTEMPT"``).
            event_id: Reference to the anomaly event or transaction ID.
            details: Optional metadata dict (must be JSON-serialisable).

        Returns:
            Entry hash of the newly appended record.

        Raises:
            AuditTamperException: If the existing chain is found to be broken
                before appending (integrity check on append).
        """
        if self._entries:
            self._spot_check_last()

        prev_hash = self._entries[-1].entry_hash if self._entries else _GENESIS_HASH
        entry = AuditEntry(
            entry_id=len(self._entries),
            timestamp_utc=time.time(),
            event_type=event_type,
            event_id=event_id,
            details=details or {},
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        self._index[entry.entry_hash] = entry
        return entry.entry_hash

    def verify_integrity(self) -> bool:
        """Verify the full hash chain integrity.

        Recomputes each entry's hash and checks the chain linkage.

        Returns:
            True if the chain is intact; False if any entry has been tampered.
        """
        expected_prev = _GENESIS_HASH
        for entry in self._entries:
            recomputed = entry._compute_hash()
            if recomputed != entry.entry_hash:
                return False
            if entry.prev_hash != expected_prev:
                return False
            expected_prev = entry.entry_hash
        return True

    def get_entry(self, entry_hash: str) -> AuditEntry:
        """Retrieve an entry by its hash.

        Args:
            entry_hash: 64-character hex hash string.

        Returns:
            The corresponding AuditEntry.

        Raises:
            KeyError: If the hash is not found in the log.
        """
        if entry_hash not in self._index:
            raise KeyError(f"Entry hash not found: {entry_hash[:16]}...")
        return self._index[entry_hash]

    def export_to_json(self, path: Path) -> None:
        """Export the full audit log to a JSON file.

        Args:
            path: Output file path (will be created or overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for e in self._entries:
            d = asdict(e)
            d["details"] = dict(e.details)
            records.append(d)
        with open(path, "w") as fh:
            json.dump(records, fh, indent=2, default=str)

    def __len__(self) -> int:
        return len(self._entries)

    def _spot_check_last(self) -> None:
        """Verify the most recent entry has not been tampered."""
        last = self._entries[-1]
        if last._compute_hash() != last.entry_hash:
            raise AuditTamperException(
                f"Audit log tampered at entry {last.entry_id}. "
                "Chain integrity violated."
            )
