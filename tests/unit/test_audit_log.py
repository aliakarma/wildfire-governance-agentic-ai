"""Unit tests for audit_log.py."""
import pytest
from wildfire_governance.blockchain.audit_log import ImmutableAuditLog

def test_append_returns_hash(audit_log):
    h = audit_log.append("APPROVED", "evt_001"); assert isinstance(h, str); assert len(h) == 64

def test_integrity_empty_log(audit_log):
    assert audit_log.verify_integrity() is True

def test_integrity_after_appends(audit_log):
    for i in range(5): audit_log.append("APPROVED", f"evt_{i}")
    assert audit_log.verify_integrity() is True

def test_tampered_hash_detected(audit_log):
    audit_log.append("APPROVED", "evt_001"); audit_log.append("BLOCKED", "evt_002")
    audit_log._entries[0].entry_hash = "00"*32
    assert audit_log.verify_integrity() is False

def test_get_entry_by_hash(audit_log):
    h = audit_log.append("APPROVED", "evt_42", {"cert": "abc123"})
    entry = audit_log.get_entry(h); assert entry.event_id == "evt_42"

def test_chain_linkage(audit_log):
    h1 = audit_log.append("APPROVED", "e1"); audit_log.append("BLOCKED", "e2")
    assert audit_log._entries[1].prev_hash == h1
