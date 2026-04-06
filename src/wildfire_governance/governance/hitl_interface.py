"""Human-in-the-loop (HITL) authorisation gate.

Mediates all override and authorisation decisions before alerts reach the
blockchain smart contract. Integrates the operator oracle model with the
cryptographic signing step required by the governance predicate.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from wildfire_governance.blockchain.crypto_utils import generate_key_pair, sign
from wildfire_governance.blockchain.transaction import AnomalyTransaction
from wildfire_governance.governance.oracle_model import HumanOperatorOracle, OracleDecision


class HITLAuthorisationGate:
    """HITL gate integrating human review with cryptographic authorisation.

    After the operator reviews an anomaly and approves it, this gate signs
    the corresponding transaction with the operator's private key. The signed
    transaction is then submitted to the smart contract for final verification.

    In the GOMDP, this gate provides the HA_t = 1 component of the governance
    predicate G(s,a) = [Conf > tau AND HA = 1].

    Args:
        oracle: HumanOperatorOracle simulating operator decisions.
        rng: Seeded NumPy Generator.
    """

    def __init__(
        self,
        oracle: Optional[HumanOperatorOracle] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._oracle = oracle or HumanOperatorOracle()
        self._rng = rng or np.random.default_rng(42)
        # Generate a validator key pair for this gate instance
        self._private_key, self.public_key = generate_key_pair()

    def process(
        self,
        transaction: AnomalyTransaction,
        confidence: float,
    ) -> Tuple[OracleDecision, Optional[bytes]]:
        """Submit an anomaly for human review and optionally sign it.

        Args:
            transaction: The anomaly transaction to review.
            confidence: The Conf^(2)_t score to present to the operator.

        Returns:
            Tuple (OracleDecision, signature_bytes).
            ``signature_bytes`` is the Ed25519 signature if approved;
            ``None`` if the operator rejected the alert.
        """
        decision = self._oracle.review(confidence)
        if not decision.approved:
            return decision, None
        # Operator approved: sign the transaction
        signature = sign(transaction.to_bytes(), self._private_key)
        return decision, signature

    @property
    def public_key_bytes(self) -> bytes:
        """Public key bytes for smart contract signature verification."""
        return self.public_key
