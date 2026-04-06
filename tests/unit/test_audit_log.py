"""Unit tests for blockchain/audit_log.py."""
from __future__ import annotations

import pytest

from wildfire_governance.blockchain.audit_log import AuditTamperException, ImmutableAuditLog


def test_append_returns_hash(audit_log: ImmutableAuditLog) -> None:
    """append() must return a 64-character hex hash string."""
    h = audit_log.append("APPROVED", "evt_001", {"confidence": 0.9})
    assert isinstance(h, str)
    assert len(h) == 64


def test_integrity_empty_log(audit_log: ImmutableAuditLog) -> None:
    """Empty log must report integrity as True."""
    assert audit_log.verify_integrity() is True


def test_integrity_after_appends(audit_log: ImmutableAuditLog) -> None:
    """Log with multiple entries must report integrity as True."""
    for i in range(5):
        audit_log.append("APPROVED", f"evt_{i}")
    assert audit_log.verify_integrity() is True


def test_tampered_hash_detected(audit_log: ImmutableAuditLog) -> None:
    """Modifying an entry hash must cause verify_integrity to return False."""
    audit_log.append("APPROVED", "evt_001")
    audit_log.append("BLOCKED", "evt_002")
    # Tamper with the first entry's hash
    audit_log._entries[0].entry_hash = "00" * 32
    assert audit_log.verify_integrity() is False


def test_get_entry_by_hash(audit_log: ImmutableAuditLog) -> None:
    """get_entry must return the correct entry by hash."""
    h = audit_log.append("APPROVED", "evt_42", {"cert": "abc123"})
    entry = audit_log.get_entry(h)
    assert entry.event_id == "evt_42"
    assert entry.event_type == "APPROVED"


def test_get_entry_not_found_raises(audit_log: ImmutableAuditLog) -> None:
    """get_entry with unknown hash must raise KeyError."""
    with pytest.raises(KeyError):
        audit_log.get_entry("0" * 64)


def test_log_length(audit_log: ImmutableAuditLog) -> None:
    """len() must equal number of appended entries."""
    for i in range(7):
        audit_log.append("APPROVED", f"evt_{i}")
    assert len(audit_log) == 7


def test_chain_linkage(audit_log: ImmutableAuditLog) -> None:
    """Each entry's prev_hash must equal the previous entry's entry_hash."""
    h1 = audit_log.append("APPROVED", "e1")
    audit_log.append("BLOCKED", "e2")
    assert audit_log._entries[1].prev_hash == h1
