"""PBFT-variant consensus simulation."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from wildfire_governance.blockchain.transaction import AnomalyTransaction

class ByzantineFaultType(str, Enum):
    SILENT = "silent"; EQUIVOCATING = "equivocating"; MALICIOUS = "malicious"

@dataclass
class Vote:
    validator_id: int; transaction_hash: str; approve: bool; is_byzantine: bool = False

@dataclass
class ConsensusResult:
    transaction_hash: str; consensus_reached: bool; approved: bool
    n_valid_votes: int; n_byzantine_votes: int; delay_steps: float

class PBFTConsensus:
    def __init__(self, n_validators: int = 7, max_byzantine: int = 2,
                 mean_delay_steps: float = 1.2, std_delay_steps: float = 0.3,
                 burst_multiplier: float = 1.35, rng: Optional[np.random.Generator] = None) -> None:
        if max_byzantine > (n_validators - 1) // 3:
            raise ValueError(f"max_byzantine={max_byzantine} must be <= (k-1)//3")
        self.n_validators = n_validators; self.max_byzantine = max_byzantine
        self.mean_delay = mean_delay_steps; self.std_delay = std_delay_steps
        self.burst_multiplier = burst_multiplier
        self._rng = rng or np.random.default_rng(42)
        self._byzantine_validators: Dict[int, ByzantineFaultType] = {}
        self._quorum = 2 * max_byzantine + 1

    def propose(self, transaction: AnomalyTransaction, burst_mode: bool = False) -> ConsensusResult:
        delay = max(0.1, float(self._rng.normal(self.mean_delay, self.std_delay)))
        if burst_mode: delay *= self.burst_multiplier
        votes = [self._byzantine_vote(v, transaction) if v in self._byzantine_validators
                 else Vote(validator_id=v, transaction_hash=transaction.transaction_hash, approve=True)
                 for v in range(self.n_validators)]
        valid = [v for v in votes if not v.is_byzantine]
        byz = [v for v in votes if v.is_byzantine]
        reached = len(valid) >= self._quorum
        approved = reached and sum(v.approve for v in valid) >= self._quorum
        return ConsensusResult(transaction.transaction_hash, reached, approved, len(valid), len(byz), delay)

    def inject_byzantine_fault(self, validator_id: int, fault_type: ByzantineFaultType) -> None:
        if validator_id < 0 or validator_id >= self.n_validators:
            raise ValueError(f"validator_id={validator_id} out of range")
        self._byzantine_validators[validator_id] = fault_type

    def clear_byzantine_faults(self) -> None:
        self._byzantine_validators.clear()

    @property
    def n_byzantine(self) -> int: return len(self._byzantine_validators)
    @property
    def is_below_threshold(self) -> bool: return self.n_byzantine <= self.max_byzantine

    def _byzantine_vote(self, vid: int, tx: AnomalyTransaction) -> Vote:
        ft = self._byzantine_validators[vid]
        approve = (ft == ByzantineFaultType.MALICIOUS) or                   (ft == ByzantineFaultType.EQUIVOCATING and bool(self._rng.integers(0, 2)))
        return Vote(validator_id=vid, transaction_hash=tx.transaction_hash, approve=approve, is_byzantine=True)
