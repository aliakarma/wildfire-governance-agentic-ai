"""CMDP vs. GOMDP violation rate comparison module.

Provides a programmatic interface for comparing governance compliance
between CMDP (Lagrangian relaxation) and GOMDP (cryptographic enforcement)
frameworks. Used in Experiment 12 and the Discussion section.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViolationStudyResult:
    """Result of a CMDP vs. GOMDP violation rate study.

    Attributes:
        framework: ``"GOMDP"`` or ``"CMDP"``.
        n_episodes: Number of evaluation episodes.
        n_violations: Episodes with at least one governance violation.
        compliance_rate: 1.0 - n_violations / n_episodes.
        theorem1_holds: True only for GOMDP (by Theorem 1 proof).
        note: Explanatory note about the enforcement mechanism.
    """

    framework: str
    n_episodes: int
    n_violations: int
    compliance_rate: float
    theorem1_holds: bool
    note: str


class CMDPViolationStudy:
    """Empirical comparison of GOMDP vs. CMDP governance violation rates.

    Key distinction (Remark 1 in paper):
    - CMDP enforces constraints via Lagrangian penalty in the reward signal.
      The policy may take non-compliant actions, especially during training.
      Empirical violation rate: ~7.2% (Table II, PPO-CMDP column).
    - GOMDP enforces constraints at the environment boundary via cryptographic
      smart contract. ANY policy satisfies the governance predicate with
      probability one (Theorem 1). Violation rate: 0.0% by construction.

    Args:
        n_episodes: Number of evaluation episodes to run.
    """

    def __init__(self, n_episodes: int = 20) -> None:
        self.n_episodes = n_episodes

    def run_gomdp_evaluation(
        self,
        grid_size: int = 10,
        n_uavs: int = 5,
        n_timesteps: int = 100,
    ) -> ViolationStudyResult:
        """Evaluate GOMDP governance compliance.

        Args:
            grid_size: Simulation grid side length.
            n_uavs: UAV fleet size.
            n_timesteps: Episode length.

        Returns:
            ViolationStudyResult with compliance_rate=1.0 by Theorem 1.
        """
        from experiments.utils.runner import run_episode

        violations = 0
        for seed in range(self.n_episodes):
            result = run_episode(
                seed=seed, config_name="gomdp",
                grid_size=grid_size, n_timesteps=n_timesteps, n_uavs=n_uavs,
                enable_governance=True, enable_hitl=True,
                enable_blockchain=True, enable_verification=True,
                enable_coordination=True,
            )
            if not getattr(result, "governance_compliant", False):
                violations += 1

        compliance = 1.0 - violations / self.n_episodes
        return ViolationStudyResult(
            framework="GOMDP",
            n_episodes=self.n_episodes,
            n_violations=violations,
            compliance_rate=compliance,
            theorem1_holds=(violations == 0),
            note=(
                "Blockchain smart contract enforces governance at environment level. "
                "Theorem 1 guarantees 0 violations for any policy."
            ),
        )

    def run_cmdp_surrogate_evaluation(
        self,
        grid_size: int = 10,
        n_uavs: int = 5,
        n_timesteps: int = 100,
    ) -> ViolationStudyResult:
        """Evaluate a CMDP-surrogate (no blockchain enforcement).

        Without blockchain enforcement, the system relies on procedural HITL
        only. This surrogate models the CMDP regime where constraints are soft
        and violations can occur.

        Returns:
            ViolationStudyResult showing non-zero violation rate.
        """
        from experiments.utils.runner import run_episode

        violations = 0
        for seed in range(self.n_episodes):
            result = run_episode(
                seed=seed, config_name="cmdp_surrogate",
                grid_size=grid_size, n_timesteps=n_timesteps, n_uavs=n_uavs,
                enable_governance=False, enable_hitl=True,
                enable_blockchain=False, enable_verification=True,
                enable_coordination=True,
            )
            if not getattr(result, "governance_compliant", False):
                violations += 1

        compliance = 1.0 - violations / self.n_episodes
        return ViolationStudyResult(
            framework="CMDP (surrogate)",
            n_episodes=self.n_episodes,
            n_violations=violations,
            compliance_rate=compliance,
            theorem1_holds=False,
            note=(
                "No blockchain enforcement. HITL approval alone does not provide "
                "cryptographic non-repudiation. Theorem 1 cannot hold."
            ),
        )
