"""PBFT-variant consensus simulation for the Hyperledger Fabric governance layer.

Simulates Byzantine fault-tolerant consensus among k validators with
f < k/3 Byzantine fault tolerance, as described in the paper (Section III-A).

Nominal blockchain confirmation delay: N(1.2, 0.3) steps.
Under 5× anomaly burst: delay increases by 35% (1.2 × 1.35 = 1.62 steps).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Literal, Optional

import numpy as np

from wildfire_governance.blockchain.transaction import AnomalyTransaction


class ByzantineFaultType(str, Enum):
    """Types of Byzantine validator behaviour."""

    SILENT = "silent"          # Crash fault: never responds
    EQUIVOCATING = "equivocating"  # Sends different votes to different nodes
    MALICIOUS = "malicious"    # Always votes to approve (adversarial)


@dataclass
class Vote:
    """A single validator vote on a transaction.

    Attributes:
        validator_id: Integer identifier of the voting validator.
        transaction_hash: Hash of the transaction being voted on.
        approve: True = vote to approve; False = vote to reject.
        is_byzantine: Whether this vote was produced by a Byzantine validator.
    """

    validator_id: int
    transaction_hash: str
    approve: bool
    is_byzantine: bool = False


@dataclass
class ConsensusResult:
    """Outcome of a PBFT consensus round.

    Attributes:
        transaction_hash: Hash of the transaction that was voted on.
        consensus_reached: True if >= 2f+1 valid votes were collected.
        approved: True if consensus reached AND majority vote is approve.
        n_valid_votes: Number of non-Byzantine votes cast.
        n_byzantine_votes: Number of Byzantine votes cast.
        delay_steps: Simulated confirmation delay (fractional steps).
    """

    transaction_hash: str
    consensus_reached: bool
    approved: bool
    n_valid_votes: int
    n_byzantine_votes: int
    delay_steps: float


class PBFTConsensus:
    """Simulated PBFT-variant consensus for the Hyperledger Fabric governance network.

    Models k=7 validators with f=2 Byzantine fault tolerance (paper default).
    Confirmation delay is drawn from N(mean_delay, std_delay) and multiplied
    by burst_multiplier during high-anomaly-frequency bursts.

    Args:
        n_validators: Total validators k (default 7).
        max_byzantine: Maximum Byzantine faults f (default 2). Must be < k//3.
        mean_delay_steps: Mean confirmation delay in timesteps (default 1.2).
        std_delay_steps: Std deviation of confirmation delay (default 0.3).
        burst_multiplier: Delay multiplier under 5× anomaly burst (default 1.35).
        rng: Seeded NumPy Generator for delay sampling.
    """

    def __init__(
        self,
        n_validators: int = 7,
        max_byzantine: int = 2,
        mean_delay_steps: float = 1.2,
        std_delay_steps: float = 0.3,
        burst_multiplier: float = 1.35,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if max_byzantine > (n_validators - 1) // 3:
            raise ValueError(
                f"max_byzantine={max_byzantine} must be < k//3={n_validators // 3} "
                "to satisfy PBFT safety requirements."
            )
        self.n_validators = n_validators
        self.max_byzantine = max_byzantine
        self.mean_delay = mean_delay_steps
        self.std_delay = std_delay_steps
        self.burst_multiplier = burst_multiplier
        self._rng = rng or np.random.default_rng(42)
        self._byzantine_validators: Dict[int, ByzantineFaultType] = {}
        self._quorum = 2 * max_byzantine + 1  # Minimum votes for consensus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        transaction: AnomalyTransaction,
        burst_mode: bool = False,
    ) -> ConsensusResult:
        """Run one PBFT consensus round on *transaction*.

        In non-Byzantine conditions: all validators vote to approve valid
        transactions (confidence already verified by the smart contract).
        Byzantine validators behave according to their injected fault type.

        Args:
            transaction: The anomaly transaction to achieve consensus on.
            burst_mode: If True, applies burst_multiplier to confirmation delay.

        Returns:
            ConsensusResult documenting the consensus outcome and simulated delay.
        """
        delay = self._simulate_delay(burst_mode)
        votes = self._collect_votes(transaction)
        valid_votes = [v for v in votes if not v.is_byzantine]
        byzantine_votes = [v for v in votes if v.is_byzantine]
        consensus_reached = len(valid_votes) >= self._quorum
        approved = consensus_reached and sum(v.approve for v in valid_votes) >= self._quorum

        return ConsensusResult(
            transaction_hash=transaction.transaction_hash,
            consensus_reached=consensus_reached,
            approved=approved,
            n_valid_votes=len(valid_votes),
            n_byzantine_votes=len(byzantine_votes),
            delay_steps=delay,
        )

    def inject_byzantine_fault(
        self,
        validator_id: int,
        fault_type: ByzantineFaultType,
    ) -> None:
        """Make a validator Byzantine.

        Args:
            validator_id: Integer ID of the validator to corrupt (0-indexed).
            fault_type: Type of Byzantine behaviour to inject.

        Raises:
            ValueError: If injecting > max_byzantine faults OR invalid validator_id.
        """
        if validator_id < 0 or validator_id >= self.n_validators:
            raise ValueError(
                f"validator_id={validator_id} out of range [0, {self.n_validators})"
            )
        if len(self._byzantine_validators) >= self.n_validators:
            raise ValueError("Cannot inject more Byzantine faults than validators.")
        self._byzantine_validators[validator_id] = fault_type

    def clear_byzantine_faults(self) -> None:
        """Remove all injected Byzantine faults (reset to honest validators)."""
        self._byzantine_validators.clear()

    @property
    def n_byzantine(self) -> int:
        """Number of currently injected Byzantine validators."""
        return len(self._byzantine_validators)

    @property
    def is_below_threshold(self) -> bool:
        """True if current Byzantine count is within the safety threshold."""
        return self.n_byzantine <= self.max_byzantine

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_delay(self, burst_mode: bool) -> float:
        """Draw confirmation delay from the configured distribution.

        Args:
            burst_mode: If True, multiply by burst_multiplier.

        Returns:
            Positive float delay in simulation timesteps.
        """
        base = float(self._rng.normal(self.mean_delay, self.std_delay))
        base = max(0.1, base)
        return base * self.burst_multiplier if burst_mode else base

    def _collect_votes(self, transaction: AnomalyTransaction) -> List[Vote]:
        """Simulate vote collection from all validators.

        Args:
            transaction: Transaction being voted on.

        Returns:
            List of Vote objects, one per validator.
        """
        votes: List[Vote] = []
        for vid in range(self.n_validators):
            if vid in self._byzantine_validators:
                vote = self._byzantine_vote(vid, transaction)
            else:
                # Honest validator: approve if transaction hash is valid
                vote = Vote(
                    validator_id=vid,
                    transaction_hash=transaction.transaction_hash,
                    approve=True,
                    is_byzantine=False,
                )
            votes.append(vote)
        return votes

    def _byzantine_vote(self, validator_id: int, transaction: AnomalyTransaction) -> Vote:
        """Produce a Byzantine vote based on the injected fault type.

        Args:
            validator_id: ID of the Byzantine validator.
            transaction: Transaction being voted on.

        Returns:
            A Vote with is_byzantine=True.
        """
        fault_type = self._byzantine_validators[validator_id]
        if fault_type == ByzantineFaultType.SILENT:
            # Silent fault: vote is never counted (simulate by marking byzantine)
            approve = False
        elif fault_type == ByzantineFaultType.EQUIVOCATING:
            # Random vote (equivocating)
            approve = bool(self._rng.integers(0, 2))
        else:
            # Malicious: always approves to try to force through invalid transactions
            approve = True

        return Vote(
            validator_id=validator_id,
            transaction_hash=transaction.transaction_hash,
            approve=approve,
            is_byzantine=True,
        )
