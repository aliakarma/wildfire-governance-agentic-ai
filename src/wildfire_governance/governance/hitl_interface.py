"""Human-in-the-loop authorisation gate."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
from wildfire_governance.blockchain.transaction import AnomalyTransaction
from wildfire_governance.governance.oracle_model import HumanOperatorOracle, OracleDecision

class HITLAuthorisationGate:
    def __init__(self, oracle=None, rng=None):
        self._oracle = oracle or HumanOperatorOracle()
        self._rng = rng or np.random.default_rng(42)
        self._private_key, self.public_key = generate_key_pair()

    def process(self, transaction, confidence):
        decision = self._oracle.review(confidence)
        if not decision.approved: return decision, None
        signature = sign(transaction.to_bytes(), self._private_key)
        return decision, signature

    @property
    def public_key_bytes(self): return self.public_key
