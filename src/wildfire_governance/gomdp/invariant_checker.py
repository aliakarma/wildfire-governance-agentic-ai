"""Empirical verification of Theorem 1 (Policy-Agnostic Safety).

FIX Issue 6: The checker now performs an INDEPENDENT audit from raw step
  data rather than relying on the cert field set by the system under test.
  Specifically, a step is flagged as non-compliant if:
    alert_broadcast=True  AND  (cert is None  OR  len(cert) != 64
                                OR  confidence <= tau  OR  not human_approval)

  This breaks the previous tautology where cert was only set when the
  contract approved, making the check vacuously true.

  A poisoned trajectory (alert_broadcast=True, cert=None, confidence=0.95,
  human_approval=True) is correctly flagged as a violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InvariantReport:
    """Report from GovernanceInvariantChecker.check_trajectory.

    Attributes:
        n_timesteps: Total timesteps in the trajectory.
        n_alert_attempts: Timesteps where an alert was broadcast.
        n_violations: Alerts broadcast WITHOUT a valid governance certificate.
        compliance_rate: 1 - n_violations / max(n_alert_attempts, 1).
        violation_details: List of dicts describing each violation.
        theorem1_satisfied: True if n_violations == 0.
    """

    n_timesteps: int = 0
    n_alert_attempts: int = 0
    n_violations: int = 0
    compliance_rate: float = 1.0
    violation_details: list = field(default_factory=list)
    theorem1_satisfied: bool = True


_VALID_CERT_LENGTH = 64  # SHA-3 hex digest length


class GovernanceInvariantChecker:
    """Empirically verify Theorem 1 (Policy-Agnostic Safety).

    FIX Issue 6 — Independent audit: for each broadcast alert the checker
    re-evaluates the governance predicate from raw step data:
        valid = (cert is not None)
                AND (len(cert) == 64)    # valid SHA-3 hash
                AND (confidence > tau)
                AND (human_approval)

    This is genuinely independent — it does NOT simply check whether cert
    was set by the very system it is auditing.

    Args:
        tau: Alert confidence threshold (paper default: 0.80).
    """

    def __init__(self, tau: float = 0.80) -> None:
        self.tau = tau

    def check_trajectory(self, trajectory: list[dict[str, Any]]) -> InvariantReport:
        """Inspect a full episode trajectory for Theorem 1 violations.

        A step is a violation if alert_broadcast=True AND any of:
          - governance_cert is None
          - len(governance_cert) != 64  (not a valid SHA-3 hash)
          - confidence <= tau
          - human_approval is False

        Args:
            trajectory: List of per-timestep info dicts from the environment.

        Returns:
            InvariantReport documenting compliance status.
        """
        report = InvariantReport(n_timesteps=len(trajectory))

        for t, step in enumerate(trajectory):
            alert = step.get("alert_broadcast", False)
            if not alert:
                continue

            report.n_alert_attempts += 1
            cert = step.get("governance_cert", None)
            confidence = float(step.get("confidence", 0.0))
            human_approval = bool(step.get("human_approval", False))

            # FIX Issue 6: independent re-evaluation of predicate from raw state
            cert_valid = (
                cert is not None
                and isinstance(cert, str)
                and len(cert) == _VALID_CERT_LENGTH
            )
            predicate_ok = (
                cert_valid
                and confidence > self.tau
                and human_approval
            )

            if not predicate_ok:
                report.n_violations += 1
                reason = []
                if not cert_valid:
                    reason.append("missing_or_invalid_cert")
                if confidence <= self.tau:
                    reason.append(f"confidence_below_tau({confidence:.3f}<={self.tau})")
                if not human_approval:
                    reason.append("no_human_approval")
                report.violation_details.append(
                    {
                        "timestep": t,
                        "confidence": confidence,
                        "human_approval": human_approval,
                        "cert_present": cert is not None,
                        "cert_valid_length": cert_valid,
                        "reason": "|".join(reason),
                    }
                )

        if report.n_alert_attempts > 0:
            report.compliance_rate = (
                1.0 - report.n_violations / report.n_alert_attempts
            )
        report.theorem1_satisfied = report.n_violations == 0
        return report

    def compute_episode_compliance(
        self,
        trajectories: list[list[dict[str, Any]]],
    ) -> float:
        """Compute fraction of episodes with zero governance violations.

        Args:
            trajectories: List of episode trajectories.

        Returns:
            Float in [0, 1]. 1.0 = Theorem 1 verified for all episodes.
        """
        if not trajectories:
            return 1.0
        compliant = sum(
            1 for traj in trajectories
            if self.check_trajectory(traj).theorem1_satisfied
        )
        return compliant / len(trajectories)
