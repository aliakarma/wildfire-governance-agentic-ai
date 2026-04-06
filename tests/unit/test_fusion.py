"""Unit tests for verification/fusion.py (Eq. 5) and bayesian_update.py (Eq. 6)."""
from __future__ import annotations

import pytest

from wildfire_governance.verification.bayesian_update import BayesianConfidenceUpdate
from wildfire_governance.verification.fusion import CrossModalFusion


# ---- CrossModalFusion (Eq. 5) -------------------------------------------

def test_fusion_output_range(fusion: CrossModalFusion) -> None:
    """Confidence must always be in [0, 1]."""
    conf = fusion.compute_stage1_confidence(0.7, 0.5)
    assert 0.0 <= conf <= 1.0


def test_fusion_weights_must_sum_to_one() -> None:
    """CrossModalFusion must raise ValueError if weights don't sum to 1."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        CrossModalFusion(w_h=0.6, w_w=0.6)


def test_fusion_pure_heat_dominance() -> None:
    """With w_h=1.0, output equals heat_anomaly_index."""
    f = CrossModalFusion(w_h=1.0, w_w=0.0)
    heat = 0.75
    assert f.compute_stage1_confidence(heat, 0.3) == pytest.approx(heat)


def test_fusion_pure_weather_dominance() -> None:
    """With w_w=1.0, output equals weather_index."""
    f = CrossModalFusion(w_h=0.0, w_w=1.0)
    weather = 0.65
    assert f.compute_stage1_confidence(0.1, weather) == pytest.approx(weather)


def test_fusion_input_out_of_range_raises(fusion: CrossModalFusion) -> None:
    """Inputs outside [0, 1] must raise ValueError."""
    with pytest.raises(ValueError):
        fusion.compute_stage1_confidence(1.5, 0.5)


def test_fusion_paper_weights() -> None:
    """Reproduce the paper equation: 0.65*0.8 + 0.35*0.6 = 0.73."""
    f = CrossModalFusion(w_h=0.65, w_w=0.35)
    result = f.compute_stage1_confidence(0.8, 0.6)
    assert result == pytest.approx(0.65 * 0.8 + 0.35 * 0.6, rel=1e-5)


# ---- BayesianConfidenceUpdate (Eq. 6) ------------------------------------

def test_bayesian_update_output_range() -> None:
    """Conf^(2) must be in (0, 1)."""
    b = BayesianConfidenceUpdate()
    conf2 = b.update(0.7, True)
    assert 0.0 < conf2 < 1.0


def test_bayesian_positive_increases_confidence() -> None:
    """A positive verification UAV observation must increase confidence."""
    b = BayesianConfidenceUpdate(detection_probability=0.85, false_alarm_probability=0.15)
    prior = 0.65
    posterior_pos = b.update(prior, True)
    posterior_neg = b.update(prior, False)
    assert posterior_pos > prior
    assert posterior_neg < prior


def test_bayesian_likelihood_ratio_positive() -> None:
    """Likelihood ratio for positive observation must be > 1."""
    b = BayesianConfidenceUpdate(detection_probability=0.85, false_alarm_probability=0.15)
    assert b.likelihood_ratio(True) == pytest.approx(0.85 / 0.15, rel=1e-5)


def test_bayesian_invalid_probs_raise() -> None:
    """Probabilities outside (0, 1) must raise ValueError."""
    with pytest.raises(ValueError):
        BayesianConfidenceUpdate(detection_probability=1.1)


def test_bayesian_extreme_prior_handled() -> None:
    """Values at 0.0 or 1.0 must be clipped, not cause division by zero."""
    b = BayesianConfidenceUpdate()
    result = b.update(0.0, True)   # Will be clipped to 1e-6
    assert 0.0 <= result <= 1.0
    result2 = b.update(1.0, True)  # Will be clipped to 1-1e-6
    assert 0.0 <= result2 <= 1.0
