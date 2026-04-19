"""Governance-layer invariant checker compatibility wrapper.

Provides an episode-level API expected by experiment scripts while
reusing the canonical GOMDP trajectory checker implementation.
"""
from __future__ import annotations

from typing import Any

from wildfire_governance.gomdp.invariant_checker import (
    GovernanceInvariantChecker as _GOMDPInvariantChecker,
)


class GovernanceInvariantChecker(_GOMDPInvariantChecker):
    """Governance invariant checker with an episode-log convenience API."""

    def check_episode(self, step_logs: list[dict[str, Any]]) -> bool:
        """Return True when an episode contains zero governance violations."""
        report = self.check_trajectory(step_logs or [])
        return report.theorem1_satisfied
