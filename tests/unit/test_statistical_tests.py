"""Unit tests for metrics/statistical_tests.py."""
from __future__ import annotations

import numpy as np
import pytest

from wildfire_governance.metrics.statistical_tests import (
    paired_ttest_holm_bonferroni,
    summarise_tests,
)


def test_holm_bonferroni_correction_applied() -> None:
    """Corrected p-values must be >= raw p-values."""
    group_a = list(np.random.default_rng(0).normal(10, 1, 20))
    group_b = list(np.random.default_rng(1).normal(12, 1, 20))
    group_c = list(np.random.default_rng(2).normal(14, 1, 20))
    comparisons = [
        ("A", "B", "ld", group_a, group_b),
        ("A", "C", "ld", group_a, group_c),
        ("B", "C", "ld", group_b, group_c),
    ]
    results = paired_ttest_holm_bonferroni(comparisons)
    for r in results:
        assert r.p_value_corrected >= r.p_value_raw - 1e-12


def test_significant_result_preserved() -> None:
    """A clearly significant difference must remain significant after correction."""
    rng = np.random.default_rng(42)
    a = list(rng.normal(5.0, 0.1, 20))
    b = list(rng.normal(20.0, 0.1, 20))  # Large effect
    results = paired_ttest_holm_bonferroni([("A", "B", "ld", a, b)])
    assert results[0].significant is True


def test_paired_ttest_symmetric() -> None:
    """Swapping groups must not change the p-value."""
    a = list(np.random.default_rng(5).normal(10, 1, 20))
    b = list(np.random.default_rng(6).normal(12, 1, 20))
    r1 = paired_ttest_holm_bonferroni([("A", "B", "m", a, b)])[0]
    r2 = paired_ttest_holm_bonferroni([("B", "A", "m", b, a)])[0]
    assert abs(r1.p_value_raw - r2.p_value_raw) < 1e-10


def test_length_mismatch_raises() -> None:
    """Mismatched group lengths must raise ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        paired_ttest_holm_bonferroni([("A", "B", "ld", [1.0, 2.0], [1.0, 2.0, 3.0])])


def test_summarise_tests() -> None:
    """summarise_tests must return the correct n_tests and n_significant."""
    a = list(np.random.default_rng(0).normal(5, 0.1, 20))
    b = list(np.random.default_rng(1).normal(15, 0.1, 20))  # Very different
    c = list(np.random.default_rng(2).normal(5.01, 0.1, 20))  # Nearly same
    comparisons = [
        ("A", "B", "ld", a, b),
        ("A", "C", "ld", a, c),
    ]
    results = paired_ttest_holm_bonferroni(comparisons)
    summary = summarise_tests(results)
    assert summary["n_tests"] == 2
    assert "n_significant" in summary
    assert "all_significant" in summary
