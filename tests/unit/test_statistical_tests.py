"""Unit tests for statistical_tests.py."""
import numpy as np
import pytest
from wildfire_governance.metrics.statistical_tests import paired_ttest_holm_bonferroni, summarise_tests

def test_significant_result_preserved():
    rng = np.random.default_rng(42)
    a = list(rng.normal(5.0,0.1,20)); b = list(rng.normal(20.0,0.1,20))
    results = paired_ttest_holm_bonferroni([("A","B","ld",a,b)]); assert results[0].significant is True

def test_length_mismatch_raises():
    with pytest.raises(ValueError): paired_ttest_holm_bonferroni([("A","B","ld",[1.0,2.0],[1.0,2.0,3.0])])

def test_summarise_tests():
    a = list(np.random.default_rng(0).normal(5,0.1,20)); b = list(np.random.default_rng(1).normal(15,0.1,20))
    results = paired_ttest_holm_bonferroni([("A","B","ld",a,b)])
    summary = summarise_tests(results); assert summary["n_tests"] == 1
