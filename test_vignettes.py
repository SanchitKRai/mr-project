"""Unit and benchmark tests for the 60-case clinical vignette suite (`vignettes.py`)."""
import pytest

from cohort import generate_cohort
from model import BAND_ORDER, RiskModel
from triage import Triage, evaluate
from vignettes import (
    ALL_VIGNETTES,
    DEV_VIGNETTES,
    HOLDOUT_VIGNETTES,
    Vignette,
    evaluate_vignette_suite,
)


def test_vignette_counts_and_two_thirds_split():
    assert len(ALL_VIGNETTES) == 60
    assert len(DEV_VIGNETTES) == 40
    assert len(HOLDOUT_VIGNETTES) == 20
    ids = [v.vid for v in ALL_VIGNETTES]
    assert len(set(ids)) == 60, "Duplicate vignette IDs found"


def test_every_rule_r0_through_r7_is_covered_in_both_splits():
    dev_rules = {v.expected_rule for v in DEV_VIGNETTES}
    hold_rules = {v.expected_rule for v in HOLDOUT_VIGNETTES}
    expected = {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    assert dev_rules == expected
    assert hold_rules == expected


def test_every_vignette_has_non_empty_summary_and_clinical_rationale():
    for v in ALL_VIGNETTES:
        assert len(v.summary.strip()) >= 15, f"{v.vid} summary too short"
        assert len(v.rationale.strip()) >= 25, f"{v.vid} rationale too short"


@pytest.mark.parametrize("v", DEV_VIGNETTES, ids=[v.vid for v in DEV_VIGNETTES])
def test_dev_vignettes_exact_match(v: Vignette):
    res = evaluate(v.intake)
    assert (res.triage, res.rule) == (v.expected_triage, v.expected_rule), (
        f"{v.vid} ({v.summary}): expected {v.expected_triage}/{v.expected_rule}, "
        f"got {res.triage}/{res.rule}"
    )


@pytest.mark.parametrize("v", HOLDOUT_VIGNETTES, ids=[v.vid for v in HOLDOUT_VIGNETTES])
def test_holdout_vignettes_exact_match(v: Vignette):
    res = evaluate(v.intake)
    assert (res.triage, res.rule) == (v.expected_triage, v.expected_rule), (
        f"{v.vid} ({v.summary}): expected {v.expected_triage}/{v.expected_rule}, "
        f"got {res.triage}/{res.rule}"
    )


def test_hybrid_ml_arbitrator_preserves_safety_on_all_60_vignettes():
    """Verify the hybrid Rule + XGBoost arbitrator never downgrades any of the 60 vignettes."""
    cohort = generate_cohort(n=1500, seed=20261023)
    model = RiskModel(seed=20261023).fit(cohort)

    for v in ALL_VIGNETTES:
        h = model.evaluate_hybrid(v.intake, log_disagreement=False)
        if v.expected_rule == "R0":
            assert h.final_triage == Triage.STOP
            continue
        assert h.final_band is not None and h.rule_result.band is not None
        assert BAND_ORDER[h.final_band] >= BAND_ORDER[h.rule_result.band]
        # High-priority rules R1-R4 must never be diverted by ML
        if v.expected_rule in ("R1", "R2", "R3", "R4"):
            assert h.final_triage == v.expected_triage
            assert h.final_rule_id == v.expected_rule


def test_suite_summary_helper_reports_100_percent():
    rep = evaluate_vignette_suite(ALL_VIGNETTES)
    assert rep["total"] == 60
    assert rep["passed"] == 60
    assert rep["accuracy"] == 1.0
    assert rep["failures"] == []