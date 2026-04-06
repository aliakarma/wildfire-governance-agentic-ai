"""Immutable hash-chain audit log."""
from __future__ import annotations
import json, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from wildfire_governance.blockchain.crypto_utils import sha3_256_hash

_GENESIS_HASH = "0" * 64

@dataclass
class AuditEntry:
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
        payload = json.dumps({"entry_id": self.entry_id, "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type, "event_id": self.event_id,
            "details": self.details, "prev_hash": self.prev_hash}, sort_keys=True, default=str)
        return sha3_256_hash(payload.encode("utf-8"))

class AuditTamperException(Exception):
    pass

class ImmutableAuditLog:
    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []
        self._index: Dict[str, AuditEntry] = {}

    def append(self, event_type: str, event_id: str, details: Optional[Dict[str, Any]] = None) -> str:
        if self._entries:
            last = self._entries[-1]
            if last._compute_hash() != last.entry_hash:
                raise AuditTamperException(f"Audit log tampered at entry {last.entry_id}.")
        prev_hash = self._entries[-1].entry_hash if self._entries else _GENESIS_HASH
        entry = AuditEntry(entry_id=len(self._entries), timestamp_utc=time.time(),
            event_type=event_type, event_id=event_id, details=details or {}, prev_hash=prev_hash)
        self._entries.append(entry)
        self._index[entry.entry_hash] = entry
        return entry.entry_hash

    def verify_integrity(self) -> bool:
        expected_prev = _GENESIS_HASH
        for entry in self._entries:
            if entry._compute_hash() != entry.entry_hash: return False
            if entry.prev_hash != expected_prev: return False
            expected_prev = entry.entry_hash
        return True

    def get_entry(self, entry_hash: str) -> AuditEntry:
        if entry_hash not in self._index:
            raise KeyError(f"Entry hash not found: {entry_hash[:16]}...")
        return self._index[entry_hash]

    def export_to_json(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump([dict(e.__dict__) for e in self._entries], fh, indent=2, default=str)

    def __len__(self) -> int:
        return len(self._entries)
