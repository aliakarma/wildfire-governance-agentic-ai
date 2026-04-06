"""Empirical verification of Theorem 1 (Policy-Agnostic Safety).

Inspects episode trajectories to confirm that no alert was broadcast
without a valid governance certificate — regardless of which policy
(random, greedy, PPO) generated the actions.
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
        n_alert_attempts: Number of timesteps where an alert was requested.
        n_violations: Number of alerts broadcast WITHOUT governance certificate.
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


class GovernanceInvariantChecker:
    """Empirically verify Theorem 1 (Policy-Agnostic Safety).

    For each episode trajectory, checks that no alert was broadcast without
    both (a) confidence > tau and (b) a valid on-chain governance certificate.

    A trajectory is a list of dicts; each dict must contain:
        - ``alert_broadcast`` (bool): whether a public alert was sent
        - ``governance_cert`` (str | None): certificate hash if approved
        - ``confidence`` (float): stage-2 confidence score at this step
        - ``human_approval`` (bool): human authorisation flag

    Args:
        tau: Alert confidence threshold (paper default: 0.80).
    """

    def __init__(self, tau: float = 0.80) -> None:
        self.tau = tau

    def check_trajectory(self, trajectory: list[dict[str, Any]]) -> InvariantReport:
        """Inspect a full episode trajectory for Theorem 1 violations.

        Args:
            trajectory: List of per-timestep info dicts from the environment.

        Returns:
            InvariantReport documenting compliance status.
        """
        report = InvariantReport(n_timesteps=len(trajectory))
        for t, step in enumerate(trajectory):
            alert = step.get("alert_broadcast", False)
            cert = step.get("governance_cert", None)
            confidence = step.get("confidence", 0.0)
            human_approval = step.get("human_approval", False)
            if not alert:
                continue
            report.n_alert_attempts += 1
            # A valid governance certificate must exist and confidence must exceed tau
            valid = cert is not None and confidence > self.tau and human_approval
            if not valid:
                report.n_violations += 1
                report.violation_details.append(
                    {
                        "timestep": t,
                        "confidence": confidence,
                        "human_approval": human_approval,
                        "cert_present": cert is not None,
                        "reason": (
                            "confidence_below_tau" if confidence <= self.tau
                            else "no_human_approval" if not human_approval
                            else "missing_cert"
                        ),
                    }
                )

        if report.n_alert_attempts > 0:
            report.compliance_rate = 1.0 - report.n_violations / report.n_alert_attempts
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
            1 for traj in trajectories if self.check_trajectory(traj).theorem1_satisfied
        )
        return compliant / len(trajectories)
