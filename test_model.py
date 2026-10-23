"""Unit tests for `model.py` (XGBoost + TreeSHAP + Cautious Arbitrator)."""
import numpy as np
import pytest

from cohort import generate_cohort
from model import (
    BAND_ORDER,
    RiskModel,
    evaluate_cohort_performance,
    intake_to_features,
    prob_to_band,
)
from test_triage import VIGNETTES, make
from triage import Band, Triage


@pytest.fixture(scope="module")
def trained_model() -> RiskModel:
    cohort = generate_cohort(n=1500, seed=20261023)
    model = RiskModel(seed=20261023)
    model.fit(cohort[:1100])
    return model


def test_feature_encoding_handles_missing_as_nan():
    x = make(age=32, bmi=None, c4_alcohol=None, b5_varicocele=None)
    arr = intake_to_features(x)
    assert arr[0] == 32.0
    assert np.isnan(arr[4])   # b5_varicocele
    assert np.isnan(arr[8])   # bmi
    assert np.isnan(arr[11])  # c4_alcohol_heavy


def test_b1_and_red_flags_are_excluded_from_ml_features(trained_model: RiskModel):
    """Changing B1 or D1 must not change the ML upstream biological probability."""
    base_p = trained_model.predict_proba_one(make())
    b1_p = trained_model.predict_proba_one(make(b1_abnormal_semen_test=True))
    d1_p = trained_model.predict_proba_one(make(d1_sudden_severe_pain=True))
    assert base_p == b1_p == d1_p


def test_ml_probability_increases_with_major_risk_factors(trained_model: RiskModel):
    p_healthy = trained_model.predict_proba_one(make(age=25, bmi=22.0))
    p_varicocele = trained_model.predict_proba_one(make(age=25, bmi=22.0, b5_varicocele=True))
    p_multi = trained_model.predict_proba_one(
        make(age=38, bmi=35.0, b5_varicocele=True, c5_testosterone_steroids=True)
    )
    assert p_healthy < p_varicocele < p_multi


def test_shap_explanations_identify_true_risk_drivers(trained_model: RiskModel):
    x = make(age=30, c5_testosterone_steroids=True, b2_undescended_testis=True, c3_tobacco=True)
    shap_top = trained_model.explain_one(x, top_k=3)
    top_ids = {s.code_id for s in shap_top}
    assert "C5" in top_ids or "B2" in top_ids
    for s in shap_top:
        if s.code_id in ("C5", "C3"):
            assert s.modifiable is True


def test_cautious_arbitration_never_downgrades_band(trained_model: RiskModel):
    """Across all 13 clinical vignettes, final_band must always be >= rule_band and >= ml_band."""
    for vid, intake, expected_triage, expected_rule in VIGNETTES:
        res = trained_model.evaluate_hybrid(intake, log_disagreement=False)
        assert res.final_band is not None
        assert BAND_ORDER[res.final_band] >= BAND_ORDER[res.rule_result.band]
        assert BAND_ORDER[res.final_band] >= BAND_ORDER[res.ml_band]
        # Urgent / specialist / fertility rules (R1-R4) must never be overridden by ML
        if expected_rule in ("R1", "R2", "R3", "R4"):
            assert res.final_triage == expected_triage
            assert res.final_rule_id == expected_rule


def test_ml_upgrade_promotes_self_care_to_clinical_care():
    """When ML band is higher than Rule band on an R7 case, arbitrator upgrades triage."""
    cohort = generate_cohort(n=800, seed=20261023)
    # Use a low medium_cutoff to force an ML upgrade on an older man with BMI 29.5 + minor factor
    custom_model = RiskModel(medium_cutoff=0.10, high_cutoff=0.25, seed=20261023).fit(cohort)
    # Age 48 + BMI 29.5 + fever (C9=1 pt) -> Rule score = 1 (Band.LOW -> R7 Self-care)
    # But ML sees Age 48 + overweight BMI + fever -> higher probability -> Band.MEDIUM
    borderline_man = make(age=48, bmi=29.5, c9_recent_fever_illness=True)
    hybrid = custom_model.evaluate_hybrid(borderline_man, person_id=999, log_disagreement=True)
    assert hybrid.rule_result.band == Band.LOW
    assert hybrid.ml_band in (Band.MEDIUM, Band.HIGH)
    assert hybrid.disagreed is True
    assert hybrid.disagreement_direction == "ml_higher"
    assert hybrid.final_triage in (Triage.PRIMARY_CARE, Triage.ANDROLOGY_UROLOGY)
    assert len(custom_model.disagreement_log) == 1


def test_r0_stop_bypasses_ml_and_logging(trained_model: RiskModel):
    before_len = len(trained_model.disagreement_log)
    res = trained_model.evaluate_hybrid(make(consent=False), log_disagreement=True)
    assert res.final_triage == Triage.STOP
    assert len(trained_model.disagreement_log) == before_len


def test_cohort_evaluation_metrics_and_fairness():
    cohort = generate_cohort(n=1600, seed=20261023)
    metrics = evaluate_cohort_performance(cohort[:1100], cohort[1100:])
    assert metrics["ml_roc_auc"] >= 0.65
    # Hybrid sensitivity must be >= rules-only sensitivity because arbitration only upgrades
    assert metrics["impaired_sensitivity_hybrid"] >= metrics["impaired_sensitivity_rules_only"]
    assert metrics["disagreement_rate"] > 0.0
    assert metrics["subgroup_auc_gap"] < 0.15