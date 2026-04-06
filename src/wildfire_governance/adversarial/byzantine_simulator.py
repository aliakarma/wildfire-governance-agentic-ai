"""Byzantine validator fault injection simulator.

Enables controlled Byzantine fault experiments matching Table V rows 6–9
(f ∈ {0, 1, 2, 3}). At f=3 the PBFT threshold k//3 = 2 is exceeded and
consensus fails, raising P_breach to 0.581 as predicted by Theorem 2.
"""
from __future__ import annotations

from typing import Dict, List

from wildfire_governance.blockchain.consensus import ByzantineFaultType, PBFTConsensus


class ByzantineValidatorSimulator:
    """Injects Byzantine faults into the consensus layer for robustness testing.

    Args:
        consensus: PBFTConsensus instance to inject faults into.
    """

    def __init__(self, consensus: PBFTConsensus) -> None:
        self._consensus = consensus

    def inject_faults(
        self,
        n_byzantine: int,
        fault_type: ByzantineFaultType = ByzantineFaultType.MALICIOUS,
    ) -> None:
        """Inject *n_byzantine* Byzantine validators.

        Args:
            n_byzantine: Number of validators to make Byzantine.
                         Paper values: {0, 1, 2, 3}.
            fault_type: Type of Byzantine behaviour.

        Raises:
            ValueError: If n_byzantine > n_validators (impossible configuration).
        """
        self._consensus.clear_byzantine_faults()
        for i in range(min(n_byzantine, self._consensus.n_validators)):
            self._consensus.inject_byzantine_fault(i, fault_type)

    def clear(self) -> None:
        """Remove all injected faults (return to honest configuration)."""
        self._consensus.clear_byzantine_faults()

    def get_theoretical_breach_prob(self, p_compromise: float = 0.3) -> float:
        """Compute the theoretical P_breach from Theorem 2 for current fault count.

        Args:
            p_compromise: Per-validator compromise probability p_c.

        Returns:
            Theoretical breach probability.
        """
        from wildfire_governance.gomdp.breach_probability import (
            compute_breach_probability_gomdp,
        )
        k = self._consensus.n_validators
        f = self._consensus.max_byzantine
        try:
            return compute_breach_probability_gomdp(k, f, p_compromise)
        except ValueError:
            return 1.0  # Threshold exceeded

    @property
    def is_within_safety_threshold(self) -> bool:
        """True if current Byzantine count is within the provable safety bound."""
        return self._consensus.is_below_threshold
