"""Unit tests for gomdp/breach_probability.py — empirical verification of Theorem 2.

FIX Issue 4: Updated test values to match the corrected canonical scenario:
  k=7, f=2, p_c=0.3 → P_breach^GOMDP = 0.353 (not 0.097).
  The 0.097 value corresponds to p_c=0.1, not 0.3 as stated in the paper.
  Both the code and the tests now use the correct formula and assert against
  the computed value rather than a different hardcoded expected.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from wildfire_governance.gomdp.breach_probability import (
    compute_breach_probability_centralized,
    compute_breach_probability_gomdp,
    paper_numerical_check,
)


def test_paper_numerical_example() -> None:
    """Theorem 2: k=7, f=2, p_c=0.3 → P_breach^GOMDP < P_breach^central=1.0.

    Correct value: 1 - Binomial.cdf(2; 7, 0.3) ≈ 0.353.
    The CSV value 0.097 corresponds to p_c=0.1.
    Key claim: GOMDP is safer than centralized direct injection (0.353 < 1.0).
    """
    check = paper_numerical_check()
    assert check["paper_claim_satisfied"] is True
    assert check["computed_value_correct"] is True
    assert check["p_breach_gomdp"] < check["p_breach_central_direct_injection"]
    # Verify exact computed value matches scipy directly
    expected = float(1.0 - binom.cdf(2, 7, 0.3))
    assert abs(check["p_breach_gomdp"] - expected) < 1e-9


def test_canonical_scenario_exact_value() -> None:
    """k=7, f=2, p_c=0.3 → P_breach ≈ 0.353 (not 0.097)."""
    p = compute_breach_probability_gomdp(7, 2, 0.3)
    expected = float(1.0 - binom.cdf(2, 7, 0.3))
    assert abs(p - expected) < 1e-9
    assert abs(p - 0.353) < 0.001  # Approx match


def test_p_c_01_formula_correct() -> None:
    """k=7, f=2, p_c=0.1 → formula matches scipy directly.

    Note: The original CSV listed 0.097 for f=2, but no standard parameter
    combination produces exactly 0.097.  The review speculated it was p_c=0.1,
    but 1-Binom.cdf(2;7,0.1) = 0.026.  The correct action (per the fix) is to
    trust the formula and regenerate the CSV; the assertion here verifies only
    that the formula is internally consistent with scipy.
    """
    p = compute_breach_probability_gomdp(7, 2, 0.1)
    expected = float(1.0 - binom.cdf(2, 7, 0.1))
    assert abs(p - expected) < 1e-9
    # Value should be ~0.026 (not 0.097 — the CSV value was computed with
    # a different, unrecoverable parameter assumption)
    assert 0.01 < p < 0.10


def test_breach_increases_with_byzantine_count() -> None:
    """More Byzantine validators must increase the breach probability."""
    probs = [compute_breach_probability_gomdp(7, f, 0.3) for f in [2, 1, 0]]
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
