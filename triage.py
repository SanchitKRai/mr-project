"""Male reproductive health screening: rule engine (spec v0.2).

Screening and triage support only. This is NOT a diagnosis.
All thresholds are design defaults awaiting clinician and guideline review.

Convention: every yes/no field is Optional[bool].
    True  = yes
    False = no
    None  = "prefer not to say" / skipped (never treated as risky by itself)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

DISCLAIMER = (
    "This is screening support, not a diagnosis. If you ever have sudden "
    "testicular pain or swelling, get care immediately."
)
MEDICINE_NOTE = "Do not stop any prescribed medicine without asking your doctor."


class Triage(str, Enum):
    STOP = "stop"                      # R0
    URGENT_EMERGENCY = "urgent_emergency"   # R1
    URGENT_ASSESSMENT = "urgent_assessment" # R2
    ANDROLOGY_UROLOGY = "andrology_urology" # R3 / R5
    FERTILITY_SERVICES = "fertility_services"  # R4
    PRIMARY_CARE = "primary_care"      # R6
    SELF_CARE = "self_care"            # R7


class Band(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class Intake:
    # Consent and context (A)
    consent: bool = False
    age: Optional[int] = None
    trying: Optional[str] = None          # "now" | "within_year" | "not_planning" | None
    months_trying: Optional[int] = None
    partner_35_plus: Optional[bool] = None
    # History (B)
    b1_abnormal_semen_test: Optional[bool] = None
    b2_undescended_testis: Optional[bool] = None
    b3_mumps_orchitis: Optional[bool] = None
    b4_testis_groin_surgery_injury: Optional[bool] = None
    b5_varicocele: Optional[bool] = None
    b6_cancer_treatment: Optional[bool] = None
    b7_genetic_condition: Optional[bool] = None
    b8_untreated_infection: Optional[bool] = None
    # Metabolic and lifestyle (C)
    bmi: Optional[float] = None
    c2_diabetes_metabolic: Optional[bool] = None
    c3_tobacco: Optional[bool] = None
    c4_alcohol: Optional[str] = None      # "none"|"occasional"|"regular"|"heavy"|None
    c5_testosterone_steroids: Optional[bool] = None
    c6_recreational_drugs: Optional[bool] = None
    c7_heat_exposure: Optional[bool] = None
    c8_work_chemicals: Optional[bool] = None
    c9_recent_fever_illness: Optional[bool] = None
    c10_listed_medicine: Optional[bool] = None
    # Symptoms (D)
    d1_sudden_severe_pain: Optional[bool] = None
    d2_sudden_swelling: Optional[bool] = None
    d3_fever_with_pain_swelling: Optional[bool] = None
    d4_prolonged_erection: Optional[bool] = None
    d5_lump_or_size_change: Optional[bool] = None
    d6_blood_in_semen_or_persistent_ache: Optional[bool] = None
    d7_sexual_function_issue: Optional[bool] = None
    d8_recent_groin_injury_swelling: Optional[bool] = None


@dataclass
class Factor:
    id: str
    label: str
    points: int
    modifiable: bool


@dataclass
class Result:
    triage: Triage
    rule: str
    band: Optional[Band] = None
    score: int = 0
    low_confidence: bool = False
    safety_net: bool = False
    top_factors: List[Factor] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Question groups used by the safety net and the low-confidence check
RED_FLAG_FIELDS = (
    "d1_sudden_severe_pain", "d2_sudden_swelling", "d3_fever_with_pain_swelling",
    "d4_prolonged_erection", "d8_recent_groin_injury_swelling",
)
URGENT_FIELDS = ("d5_lump_or_size_change", "d6_blood_in_semen_or_persistent_ache")
# 19 non-safety triage questions: B1-B8, C1-C10, D7
TRIAGE_QUESTIONS = (
    "b1_abnormal_semen_test", "b2_undescended_testis", "b3_mumps_orchitis",
    "b4_testis_groin_surgery_injury", "b5_varicocele", "b6_cancer_treatment",
    "b7_genetic_condition", "b8_untreated_infection",
    "bmi", "c2_diabetes_metabolic", "c3_tobacco", "c4_alcohol",
    "c5_testosterone_steroids", "c6_recreational_drugs", "c7_heat_exposure",
    "c8_work_chemicals", "c9_recent_fever_illness", "c10_listed_medicine",
    "d7_sexual_function_issue",
)
LOW_CONFIDENCE_SKIP_SHARE = 0.30


def compute_factors(x: Intake) -> List[Factor]:
    """Scored factors: Major = 3, Moderate = 2, Minor = 1."""
    f: List[Factor] = []

    def add(cond, fid, label, pts, modifiable=False):
        if cond is True:
            f.append(Factor(fid, label, pts, modifiable))

    add(x.b1_abnormal_semen_test, "B1", "Previous abnormal semen test", 3)
    add(x.b2_undescended_testis, "B2", "Undescended testis in childhood", 3)
    add(x.b3_mumps_orchitis, "B3", "Mumps with testicular involvement", 2)
    add(x.b4_testis_groin_surgery_injury, "B4", "Testis, groin or pelvic surgery or injury", 2)
    add(x.b5_varicocele, "B5", "Varicocele", 3)
    add(x.b6_cancer_treatment, "B6", "Chemotherapy, radiotherapy or cancer treatment", 3)
    add(x.b7_genetic_condition, "B7", "Genetic or chromosomal condition", 3)
    add(x.b8_untreated_infection, "B8", "Untreated or incompletely treated infection", 2)
    add(x.bmi is not None and x.bmi >= 30, "C1", "BMI of 30 or above", 2, True)
    add(x.c2_diabetes_metabolic, "C2", "Diabetes or metabolic condition", 2)
    add(x.c3_tobacco, "C3", "Tobacco or vaping", 1, True)
    add(x.c4_alcohol == "heavy", "C4", "Heavy alcohol use (15 or more drinks a week)", 1, True)
    add(x.c5_testosterone_steroids, "C5", "Testosterone or anabolic steroid use", 3, True)
    add(x.c6_recreational_drugs, "C6", "Recreational drug use", 1, True)
    add(x.c7_heat_exposure, "C7", "Regular heat exposure to the groin", 1, True)
    add(x.c8_work_chemicals, "C8", "Work exposure to chemicals or radiation", 1, True)
    add(x.c9_recent_fever_illness, "C9", "High fever or serious illness in last 3 months (temporary)", 1)
    add(x.d7_sexual_function_issue, "D7", "Sexual function concern", 2)
    return f


def _bump(band: Band) -> Band:
    return {Band.LOW: Band.MEDIUM, Band.MEDIUM: Band.HIGH, Band.HIGH: Band.HIGH}[band]


def _skipped(x: Intake, name: str) -> bool:
    return getattr(x, name) is None


def evaluate(x: Intake) -> Result:
    # R0: consent and age gate. Nothing is scored or stored.
    if not x.consent or x.age is None or x.age < 18:
        return Result(Triage.STOP, "R0", notes=["Consent and an adult age are required to continue."])

    factors = compute_factors(x)
    score = sum(p.points for p in factors)
    major_ids = {"B1", "B2", "B5", "B6", "B7", "C5"}
    has_major = any(p.id in major_ids for p in factors)

    band = Band.LOW if score <= 2 else Band.MEDIUM if score <= 5 else Band.HIGH
    if has_major and x.d7_sexual_function_issue is True:
        band = Band.HIGH

    # Conservative defaults for skipped answers
    skipped = sum(_skipped(x, q) for q in TRIAGE_QUESTIONS)
    low_conf = skipped / len(TRIAGE_QUESTIONS) > LOW_CONFIDENCE_SKIP_SHARE
    if low_conf:
        band = _bump(band)
    safety_net = any(_skipped(x, q) for q in RED_FLAG_FIELDS + URGENT_FIELDS)

    top = sorted(factors, key=lambda p: -p.points)[:3]
    notes = [DISCLAIMER]
    if low_conf:
        notes.append("Low confidence: based on limited answers.")
    if x.c10_listed_medicine is True:
        notes.append(MEDICINE_NOTE)

    def out(triage: Triage, rule: str) -> Result:
        return Result(triage, rule, band, score, low_conf, safety_net, top, notes)

    # R1: emergency red flags (answers of None never trigger, never downgrade)
    if any(getattr(x, q) is True for q in RED_FLAG_FIELDS):
        return out(Triage.URGENT_EMERGENCY, "R1")
    # R2: urgent assessment
    if any(getattr(x, q) is True for q in URGENT_FIELDS):
        return out(Triage.URGENT_ASSESSMENT, "R2")
    # R3: specialist referral for established history
    if (x.b1_abnormal_semen_test or x.b6_cancer_treatment or x.b7_genetic_condition):
        return out(Triage.ANDROLOGY_UROLOGY, "R3")
    # R4: time-to-evaluation rule
    if x.trying == "now" and x.months_trying is not None:
        if x.months_trying >= 12 or (x.months_trying >= 6 and x.partner_35_plus is True):
            return out(Triage.FERTILITY_SERVICES, "R4")
    # R5: routine andrology / urology
    if (band == Band.HIGH or x.b5_varicocele or x.b2_undescended_testis
            or x.c5_testosterone_steroids):
        return out(Triage.ANDROLOGY_UROLOGY, "R5")
    # R6: primary care
    if (band == Band.MEDIUM or x.d7_sexual_function_issue or x.c2_diabetes_metabolic
            or x.b8_untreated_infection or x.c10_listed_medicine):
        return out(Triage.PRIMARY_CARE, "R6")
    # R7: self-care and prevention
    return out(Triage.SELF_CARE, "R7")
