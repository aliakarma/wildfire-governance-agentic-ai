"""Unit tests for fusion.py and bayesian_update.py."""
import pytest
from wildfire_governance.verification.bayesian_update import BayesianConfidenceUpdate
from wildfire_governance.verification.fusion import CrossModalFusion

def test_fusion_output_range(fusion):
    conf = fusion.compute_stage1_confidence(0.7, 0.5); assert 0.0 <= conf <= 1.0

def test_fusion_weights_must_sum_to_one():
    with pytest.raises(ValueError): CrossModalFusion(w_h=0.6, w_w=0.6)

def test_fusion_paper_weights():
    f = CrossModalFusion(w_h=0.65, w_w=0.35)
    assert f.compute_stage1_confidence(0.8, 0.6) == pytest.approx(0.65*0.8+0.35*0.6, rel=1e-5)

def test_bayesian_positive_increases_confidence():
    b = BayesianConfidenceUpdate(); prior = 0.65
    assert b.update(prior, True) > prior; assert b.update(prior, False) < prior

def test_bayesian_invalid_probs_raise():
    with pytest.raises(ValueError): BayesianConfidenceUpdate(detection_probability=1.1)
