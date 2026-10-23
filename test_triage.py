"""Unit tests for triage.py (spec v0.2). Run: pytest -q"""
import pytest

from triage import Band, Intake, Triage, evaluate


def make(**kw) -> Intake:
    """A complete, low-risk adult who is not planning children. Override as needed."""
    base = dict(
        consent=True, age=30, trying="not_planning", months_trying=None, partner_35_plus=None,
        b1_abnormal_semen_test=False, b2_undescended_testis=False, b3_mumps_orchitis=False,
        b4_testis_groin_surgery_injury=False, b5_varicocele=False, b6_cancer_treatment=False,
        b7_genetic_condition=False, b8_untreated_infection=False,
        bmi=22.0, c2_diabetes_metabolic=False, c3_tobacco=False, c4_alcohol="none",
        c5_testosterone_steroids=False, c6_recreational_drugs=False, c7_heat_exposure=False,
        c8_work_chemicals=False, c9_recent_fever_illness=False, c10_listed_medicine=False,
        d1_sudden_severe_pain=False, d2_sudden_swelling=False, d3_fever_with_pain_swelling=False,
        d4_prolonged_erection=False, d5_lump_or_size_change=False,
        d6_blood_in_semen_or_persistent_ache=False, d7_sexual_function_issue=False,
        d8_recent_groin_injury_swelling=False,
    )
    base.update(kw)
    return Intake(**base)


# ---- The 13 vignettes from the spec: (id, intake, expected triage, expected rule) ----
VIGNETTES = [
    ("V1", make(age=24), Triage.SELF_CARE, "R7"),
    ("V2", make(age=29, c3_tobacco=True, bmi=32.0), Triage.PRIMARY_CARE, "R6"),
    ("V3", make(age=33, trying="now", months_trying=14), Triage.FERTILITY_SERVICES, "R4"),
    ("V4", make(age=36, trying="now", months_trying=7, partner_35_plus=True),
     Triage.FERTILITY_SERVICES, "R4"),
    ("V5", make(age=31, trying="now", months_trying=5, partner_35_plus=False),
     Triage.SELF_CARE, "R7"),
    ("V6", make(age=22, d1_sudden_severe_pain=True), Triage.URGENT_EMERGENCY, "R1"),
    ("V7", make(age=40, d5_lump_or_size_change=True), Triage.URGENT_ASSESSMENT, "R2"),
    ("V8", make(age=27, b2_undescended_testis=True), Triage.ANDROLOGY_UROLOGY, "R5"),
    ("V9", make(age=35, b1_abnormal_semen_test=True), Triage.ANDROLOGY_UROLOGY, "R3"),
    ("V10", make(age=30, trying="within_year", c5_testosterone_steroids=True),
     Triage.ANDROLOGY_UROLOGY, "R5"),
    ("V11", make(age=28, d2_sudden_swelling=True, d3_fever_with_pain_swelling=True),
     Triage.URGENT_EMERGENCY, "R1"),
    ("V12", make(age=45, c2_diabetes_metabolic=True, d7_sexual_function_issue=True),
     Triage.PRIMARY_CARE, "R6"),
    ("V13", make(age=26, b8_untreated_infection=True), Triage.PRIMARY_CARE, "R6"),
]


@pytest.mark.parametrize("vid,intake,triage,rule", VIGNETTES, ids=[v[0] for v in VIGNETTES])
def test_vignettes(vid, intake, triage, rule):
    r = evaluate(intake)
    assert (r.triage, r.rule) == (triage, rule), f"{vid}: got {r.triage} via {r.rule}"


# ---- Score and band checks ----
def test_v2_score_is_three_medium():
    r = evaluate(VIGNETTES[1][1])
    assert r.score == 3 and r.band == Band.MEDIUM


def test_v10_score_alone_is_only_medium_so_c5_override_matters():
    r = evaluate(VIGNETTES[9][1])
    assert r.score == 3 and r.band == Band.MEDIUM and r.rule == "R5"


def test_v13_score_two_band_low_but_routes_to_primary_care():
    r = evaluate(VIGNETTES[12][1])
    assert r.score == 2 and r.band == Band.LOW and r.rule == "R6"


def test_single_moderate_factors_stay_in_self_care():
    for kw in (dict(b3_mumps_orchitis=True), dict(b4_testis_groin_surgery_injury=True), dict(bmi=30.5)):
        r = evaluate(make(**kw))
        assert (r.band, r.triage) == (Band.LOW, Triage.SELF_CARE), kw


def test_bmi_boundary():
    assert evaluate(make(bmi=29.9)).score == 0
    assert evaluate(make(bmi=30.0)).score == 2


def test_alcohol_only_heavy_scores():
    assert evaluate(make(c4_alcohol="regular")).score == 0
    assert evaluate(make(c4_alcohol="heavy")).score == 1


def test_major_plus_d7_forces_high_band():
    r = evaluate(make(b5_varicocele=True, d7_sexual_function_issue=True))
    assert r.score == 5 and r.band == Band.HIGH and r.rule == "R5"


def test_high_band_by_score_routes_r5():
    r = evaluate(make(b3_mumps_orchitis=True, b4_testis_groin_surgery_injury=True,
                      c2_diabetes_metabolic=True))
    assert r.score == 6 and r.band == Band.HIGH and r.rule == "R5"


# ---- Precedence ----
def test_red_flag_beats_everything():
    r = evaluate(make(d1_sudden_severe_pain=True, b1_abnormal_semen_test=True, trying="now",
                      months_trying=24))
    assert r.rule == "R1"


def test_r2_beats_r3_and_r4():
    r = evaluate(make(d6_blood_in_semen_or_persistent_ache=True, b1_abnormal_semen_test=True))
    assert r.rule == "R2"


def test_r3_beats_r4():
    r = evaluate(make(b6_cancer_treatment=True, trying="now", months_trying=20))
    assert r.rule == "R3"


# ---- Time-to-evaluation rule ----
def test_six_months_needs_partner_35_plus():
    assert evaluate(make(trying="now", months_trying=6, partner_35_plus=True)).rule == "R4"
    assert evaluate(make(trying="now", months_trying=6, partner_35_plus=False)).rule == "R7"
    assert evaluate(make(trying="now", months_trying=11)).rule == "R7"
    assert evaluate(make(trying="now", months_trying=12)).rule == "R4"


def test_not_trying_never_triggers_r4():
    assert evaluate(make(trying="not_planning", months_trying=30)).rule == "R7"


# ---- Gates ----
def test_no_consent_stops():
    r = evaluate(make(consent=False))
    assert r.triage == Triage.STOP and r.rule == "R0" and r.score == 0


@pytest.mark.parametrize("age", [None, 17])
def test_minor_or_missing_age_stops(age):
    assert evaluate(make(age=age)).triage == Triage.STOP


# ---- Medicines ----
def test_listed_medicine_routes_primary_care_with_safety_note():
    r = evaluate(make(c10_listed_medicine=True))
    assert r.rule == "R6" and any("Do not stop" in n for n in r.notes)


# ---- Skipped answers ----
def test_prefer_not_to_say_on_red_flag_is_not_a_trigger_but_shows_safety_net():
    r = evaluate(make(d1_sudden_severe_pain=None))
    assert r.rule == "R7" and r.safety_net is True


def test_complete_answers_have_no_safety_net_or_low_confidence():
    r = evaluate(make())
    assert r.safety_net is False and r.low_confidence is False


def test_many_skipped_answers_flag_low_confidence_and_bump_band():
    skipped = dict(b3_mumps_orchitis=None, b4_testis_groin_surgery_injury=None, b8_untreated_infection=None,
                   c3_tobacco=None, c4_alcohol=None, c6_recreational_drugs=None)  # 6 of 19 = 31.6%
    r = evaluate(make(**skipped))
    assert r.low_confidence is True
    assert r.score == 0 and r.band == Band.MEDIUM and r.rule == "R6"


def test_exactly_thirty_percent_is_not_low_confidence():
    # 5 of 19 = 26.3%, below the threshold
    skipped = dict(b3_mumps_orchitis=None, b4_testis_groin_surgery_injury=None, b8_untreated_infection=None,
                   c3_tobacco=None, c4_alcohol=None)
    assert evaluate(make(**skipped)).low_confidence is False


# ---- Explainability ----
def test_top_factors_sorted_by_points_and_flag_modifiable():
    r = evaluate(make(c5_testosterone_steroids=True, c3_tobacco=True, bmi=33.0, b5_varicocele=True))
    assert [f.id for f in r.top_factors][:2] == ["B5", "C5"]
    assert len(r.top_factors) == 3
    c5 = next(f for f in r.top_factors if f.id == "C5")
    assert c5.modifiable is True
