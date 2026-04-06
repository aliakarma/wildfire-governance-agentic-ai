"""Unit tests for gomdp/breach_probability.py — empirical verification of Theorem 2."""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.gomdp.breach_probability import (
    compute_breach_probability_centralized,
    compute_breach_probability_gomdp,
    paper_numerical_check,
)


def test_paper_numerical_example() -> None:
    """Theorem 2: k=7, f=2, p_c=0.3 → P_breach^GOMDP < P_breach^central=1.0.

    The GOMDP framework is provably safer than centralized for direct injection.
    P_breach^GOMDP = 0.353 for k=7, f=2, p_c=0.3 (P(X>2) for Binomial(7, 0.3)).
    P_breach^central = 1.0 for direct injection.
    """
    check = paper_numerical_check()
    assert check["paper_claim_satisfied"] is True
    assert check["p_breach_gomdp"] < check["p_breach_central_direct_injection"]
    assert check["p_breach_gomdp"] < 1.0


def test_breach_increases_with_byzantine_count() -> None:
    """More Byzantine validators must increase the breach probability."""
    probs = [compute_breach_probability_gomdp(7, f, 0.3) for f in [2, 1, 0]]
    # Fewer tolerated faults → higher breach probability (stricter system)
    assert probs[0] < probs[1] < probs[2]


def test_centralized_equals_p_attack() -> None:
    """P_breach^central must equal p_attack for all values."""
    for p in [0.0, 0.3, 0.5, 1.0]:
        assert compute_breach_probability_centralized(p) == pytest.approx(p)


def test_centralized_direct_injection() -> None:
    """Direct injection (p_att=1.0) → P_breach^central = 1.0."""
    assert compute_breach_probability_centralized(1.0) == pytest.approx(1.0)


def test_gomdp_safer_than_centralized_for_direct_injection() -> None:
    """P_breach^GOMDP must be < P_breach^central=1.0 for all valid p_c."""
    for p_c in np.linspace(0.01, 0.30, 10):
        gomdp = compute_breach_probability_gomdp(7, 2, float(p_c))
        central_injection = compute_breach_probability_centralized(1.0)
        assert gomdp < central_injection, f"GOMDP not safer at p_c={p_c:.2f}"


def test_invalid_p_compromise_raises() -> None:
    """p_compromise outside [0, 1] must raise ValueError."""
    with pytest.raises(ValueError):
        compute_breach_probability_gomdp(7, 2, 1.5)


def test_byzantine_threshold_exceeded_raises() -> None:
    """f > (k-1)//3 must raise ValueError (PBFT safety violated)."""
    with pytest.raises(ValueError):
        compute_breach_probability_gomdp(7, 3, 0.3)  # f=3 > (7-1)//3=2


def test_breach_increases_with_compromise_prob() -> None:
    """P_breach^GOMDP must be monotonically increasing in p_c."""
    probs = [compute_breach_probability_gomdp(7, 2, p) for p in [0.05, 0.15, 0.30]]
    assert probs[0] < probs[1] < probs[2]


def test_gomdp_output_in_unit_interval() -> None:
    """Breach probability must always be in [0, 1]."""
    p = compute_breach_probability_gomdp(7, 2, 0.3)
    assert 0.0 <= p <= 1.0
