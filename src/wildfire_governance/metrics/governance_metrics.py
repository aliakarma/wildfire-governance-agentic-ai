"""Governance overhead metrics: blockchain delay, human review, overhead percentage.

Implements the metrics reported in Sections VI-C4 and Table III of the paper.
Governance overhead is defined as:
    (Ld_proposed - Ld_adaptive) / Ld_adaptive × 100%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GovernanceOverheadMetrics:
    """Governance-layer latency decomposition.

    Attributes:
        ld_proposed: Detection latency of the governance-constrained system.
        ld_adaptive: Detection latency of the ungoverned adaptive baseline.
        bc_delay_mean: Mean blockchain confirmation delay (steps).
        human_review_delay_mean: Mean human review delay (steps).
        le2e: End-to-end latency = ld + bc_delay + human_review_delay.
        governance_overhead_pct: (ld_proposed - ld_adaptive) / ld_adaptive × 100%.
        bc_fraction_of_le2e: Blockchain delay as fraction of Le2e (%).
        human_fraction_of_le2e: Human review delay as fraction of Le2e (%).
        detection_fraction_of_le2e: Detection latency as fraction of Le2e (%).
    """

    ld_proposed: float
    ld_adaptive: float
    bc_delay_mean: float
    human_review_delay_mean: float
    le2e: float = 0.0
    governance_overhead_pct: float = 0.0
    bc_fraction_of_le2e: float = 0.0
    human_fraction_of_le2e: float = 0.0
    detection_fraction_of_le2e: float = 0.0

    def __post_init__(self) -> None:
        self.le2e = self.ld_proposed + self.bc_delay_mean + self.human_review_delay_mean
        if self.ld_adaptive > 0:
            self.governance_overhead_pct = (
                (self.ld_proposed - self.ld_adaptive) / self.ld_adaptive * 100.0
            )
        if self.le2e > 0:
            self.bc_fraction_of_le2e = self.bc_delay_mean / self.le2e * 100.0
            self.human_fraction_of_le2e = self.human_review_delay_mean / self.le2e * 100.0
            self.detection_fraction_of_le2e = self.ld_proposed / self.le2e * 100.0


def compute_governance_overhead(
    ld_proposed: float,
    ld_adaptive: float,
    bc_delay_mean: float = 1.2,
    human_review_delay_mean: float = 3.0,
) -> GovernanceOverheadMetrics:
    """Compute the full governance overhead decomposition.

    Replicates the analysis in Section VI-C4 of the paper.

    At N=20 UAVs (paper values):
        ld_proposed=18.3, ld_adaptive=16.2
        → governance_overhead_pct = (18.3-16.2)/16.2 × 100 = 13.0%
        → Le2e = 18.3 + 1.2 + 3.0 = 22.5 steps
        → BC fraction = 1.2/22.5 × 100 = 5.3%
        → Human fraction = 3.0/22.5 × 100 = 13.3%

    Args:
        ld_proposed: Detection latency of the proposed system.
        ld_adaptive: Detection latency of the ungoverned adaptive baseline.
        bc_delay_mean: Mean blockchain confirmation delay.
        human_review_delay_mean: Mean human review delay.

    Returns:
        GovernanceOverheadMetrics with all decomposition values.
    """
    return GovernanceOverheadMetrics(
        ld_proposed=ld_proposed,
        ld_adaptive=ld_adaptive,
        bc_delay_mean=bc_delay_mean,
        human_review_delay_mean=human_review_delay_mean,
    )
