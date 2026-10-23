"""60-case clinical vignette benchmark for Male Reproductive Health Screening.

Implements Section 7 of the Spec (v0.2):
- 60 realistic clinical vignettes (V01-V60) covering every rule (R0-R7),
  boundary condition, rule precedence, safety net, and missingness bump.
- Each vignette records: id, split ("dev" vs "holdout"), clinical summary,
  Intake input, expected Triage, expected Rule ID, and a one-line clinical
  rationale anchored to EAU / AUA / WHO guidelines.
- 40 cases (two-thirds) form the development set (`DEV_VIGNETTES`).
- 20 cases (one-third) form the locked hold-out set (`HOLDOUT_VIGNETTES`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from triage import Intake, Triage, evaluate


@dataclass(frozen=True)
class Vignette:
    vid: str
    split: str               # "dev" (40) or "holdout" (20)
    summary: str
    intake: Intake
    expected_triage: Triage
    expected_rule: str
    rationale: str


def _make(**kw) -> Intake:
    """Baseline consenting 30yo healthy adult not currently planning children."""
    base = dict(
        consent=True,
        age=30,
        trying="not_planning",
        months_trying=None,
        partner_35_plus=None,
        b1_abnormal_semen_test=False,
        b2_undescended_testis=False,
        b3_mumps_orchitis=False,
        b4_testis_groin_surgery_injury=False,
        b5_varicocele=False,
        b6_cancer_treatment=False,
        b7_genetic_condition=False,
        b8_untreated_infection=False,
        bmi=22.0,
        c2_diabetes_metabolic=False,
        c3_tobacco=False,
        c4_alcohol="none",
        c5_testosterone_steroids=False,
        c6_recreational_drugs=False,
        c7_heat_exposure=False,
        c8_work_chemicals=False,
        c9_recent_fever_illness=False,
        c10_listed_medicine=False,
        d1_sudden_severe_pain=False,
        d2_sudden_swelling=False,
        d3_fever_with_pain_swelling=False,
        d4_prolonged_erection=False,
        d5_lump_or_size_change=False,
        d6_blood_in_semen_or_persistent_ache=False,
        d7_sexual_function_issue=False,
        d8_recent_groin_injury_swelling=False,
    )
    base.update(kw)
    return Intake(**base)


# ---------------------------------------------------------------------------
# Full 60-Vignette Suite (40 Dev: V01-V40 | 20 Holdout: V41-V60)
# ---------------------------------------------------------------------------
ALL_VIGNETTES: List[Vignette] = [
    # ---- Starter 13 Vignettes from Spec v0.2 (Dev Set: V01-V13) ----
    Vignette(
        "V01", "dev",
        "24yo, no symptoms, no history, non-smoker, BMI 22, not planning children",
        _make(age=24),
        Triage.SELF_CARE, "R7",
        "Zero risk factors and no conception delay warrant routine preventive self-care.",
    ),
    Vignette(
        "V02", "dev",
        "29yo, smoker, BMI 32, not planning children",
        _make(age=29, c3_tobacco=True, bmi=32.0),
        Triage.PRIMARY_CARE, "R6",
        "Combined obesity (2 pts) and tobacco use (1 pt) reach Medium band (3 pts) for primary care metabolic/lifestyle review.",
    ),
    Vignette(
        "V03", "dev",
        "33yo, trying to conceive for 14 months, no other history",
        _make(age=33, trying="now", months_trying=14, partner_35_plus=False),
        Triage.FERTILITY_SERVICES, "R4",
        "EAU/ASRM guidelines recommend couple fertility evaluation and semen analysis after >=12 months of unprotected intercourse.",
    ),
    Vignette(
        "V04", "dev",
        "36yo, trying 7 months, partner aged 37",
        _make(age=36, trying="now", months_trying=7, partner_35_plus=True),
        Triage.FERTILITY_SERVICES, "R4",
        "When the female partner is >=35 years old, ASRM/EAU guidelines lower the evaluation threshold to >=6 months of trying.",
    ),
    Vignette(
        "V05", "dev",
        "31yo, trying 5 months, partner under 35, no history",
        _make(age=31, trying="now", months_trying=5, partner_35_plus=False),
        Triage.SELF_CARE, "R7",
        "Trying <6 months with no risk factors is within normal conception timelines; provide preconception self-care advice.",
    ),
    Vignette(
        "V06", "dev",
        "22yo, sudden severe left testicular pain since this morning",
        _make(age=22, d1_sudden_severe_pain=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Acute severe testicular pain is suspected testicular torsion until proven otherwise and requires emergency surgical evaluation within hours.",
    ),
    Vignette(
        "V07", "dev",
        "40yo, new hard painless lump found on testicle while showering",
        _make(age=40, d5_lump_or_size_change=True),
        Triage.URGENT_ASSESSMENT, "R2",
        "A new intrascrotal mass or hardness requires prompt urological assessment and scrotal ultrasound within days to rule out testicular malignancy.",
    ),
    Vignette(
        "V08", "dev",
        "27yo, childhood history of undescended testis (orchidopexy), no symptoms",
        _make(age=27, b2_undescended_testis=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Prior cryptorchidism carries a major long-term risk of impaired spermatogenesis and warrants baseline specialist andrology review.",
    ),
    Vignette(
        "V09", "dev",
        "35yo, previous abnormal semen analysis report",
        _make(age=35, b1_abnormal_semen_test=True),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "Established abnormal semen parameters require targeted andrological workup and repeat confirmatory WHO laboratory testing.",
    ),
    Vignette(
        "V10", "dev",
        "30yo, exogenous testosterone and anabolic steroid use for 2 years, planning children within a year",
        _make(age=30, trying="within_year", c5_testosterone_steroids=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Exogenous androgens suppress the hypothalamic-pituitary-gonadal axis causing severe oligo/azoospermia requiring specialist cessation management.",
    ),
    Vignette(
        "V11", "dev",
        "28yo, fever with painful swollen scrotum for 2 days",
        _make(age=28, d2_sudden_swelling=True, d3_fever_with_pain_swelling=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Acute febrile scrotal swelling suggests severe epididymo-orchitis or abscess requiring immediate emergency evaluation.",
    ),
    Vignette(
        "V12", "dev",
        "45yo, type 2 diabetes and erectile difficulty, not planning children",
        _make(age=45, c2_diabetes_metabolic=True, d7_sexual_function_issue=True),
        Triage.PRIMARY_CARE, "R6",
        "Co-existing diabetes and erectile dysfunction warrant primary care cardiovascular, glycemic, and hormonal evaluation.",
    ),
    Vignette(
        "V13", "dev",
        "26yo, history of incompletely treated urogenital STI, no current acute pain",
        _make(age=26, b8_untreated_infection=True),
        Triage.PRIMARY_CARE, "R6",
        "Untreated or incompletely treated urogenital infection requires primary care NAAT screening and antimicrobial clearance to prevent obstruction.",
    ),

    # ---- Additional Dev Set Vignettes (V14-V40) ----
    Vignette(
        "V14", "dev",
        "17yo adolescent attempting intake with consent checked",
        _make(age=17, consent=True),
        Triage.STOP, "R0",
        "Tool scope is restricted to consenting adults aged 18 and over; minors stop at gate R0 with zero data stored.",
    ),
    Vignette(
        "V15", "dev",
        "34yo adult who declines the initial privacy consent gate",
        _make(age=34, consent=False),
        Triage.STOP, "R0",
        "Explicit consent is mandatory before processing any reproductive health inputs.",
    ),
    Vignette(
        "V16", "dev",
        "31yo with painful erection lasting >4 hours (priapism)",
        _make(age=31, d4_prolonged_erection=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Ischemic priapism (>4 hours) is a urological emergency requiring immediate corporal aspiration to prevent permanent fibrosis.",
    ),
    Vignette(
        "V17", "dev",
        "25yo with blunt groin trauma during football followed by acute scrotal swelling",
        _make(age=25, d8_recent_groin_injury_swelling=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Traumatic scrotal swelling requires emergency ultrasound to exclude testicular rupture or intratesticular hematoma.",
    ),
    Vignette(
        "V18", "dev",
        "38yo with acute scrotal swelling (D2) AND prior abnormal semen test (B1) and trying 18m",
        _make(age=38, d2_sudden_swelling=True, b1_abnormal_semen_test=True, trying="now", months_trying=18),
        Triage.URGENT_EMERGENCY, "R1",
        "Emergency red flags (R1) always override chronic specialist (R3) and fertility (R4) rules.",
    ),
    Vignette(
        "V19", "dev",
        "29yo with sudden severe testicular pain (D1) and new lump (D5)",
        _make(age=29, d1_sudden_severe_pain=True, d5_lump_or_size_change=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Immediate emergency care (R1) takes strict precedence over fast-track urgent urology (R2).",
    ),
    Vignette(
        "V20", "dev",
        "37yo noticing blood in semen (hematospermia) and persistent scrotal ache for 3 weeks",
        _make(age=37, d6_blood_in_semen_or_persistent_ache=True),
        Triage.URGENT_ASSESSMENT, "R2",
        "Persistent scrotal pain >2 weeks or hematospermia warrants prompt outpatient urological workup.",
    ),
    Vignette(
        "V21", "dev",
        "34yo with new testicular lump (D5) who has also been trying to conceive for 16 months",
        _make(age=34, d5_lump_or_size_change=True, trying="now", months_trying=16),
        Triage.URGENT_ASSESSMENT, "R2",
        "Ruling out testicular neoplasia (R2) takes priority over routine fertility clinic referral (R4).",
    ),
    Vignette(
        "V22", "dev",
        "42yo with persistent scrotal heaviness (D6) and known varicocele (B5)",
        _make(age=42, d6_blood_in_semen_or_persistent_ache=True, b5_varicocele=True),
        Triage.URGENT_ASSESSMENT, "R2",
        "Symptomatic ache >2 weeks (R2) upgrades a routine varicocele (R5) to prompt urological assessment.",
    ),
    Vignette(
        "V23", "dev",
        "29yo lymphoma survivor treated with chemotherapy 3 years ago, planning children soon",
        _make(age=29, trying="within_year", b6_cancer_treatment=True),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "Prior gonadotoxic chemotherapy requires specialist andrology evaluation for post-treatment spermatogenic recovery.",
    ),
    Vignette(
        "V24", "dev",
        "32yo diagnosed with Klinefelter syndrome / Y-chromosome microdeletion (B7)",
        _make(age=32, b7_genetic_condition=True),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "Known genetic or chromosomal conditions affecting fertility require specialist andrology and reproductive genetics care.",
    ),
    Vignette(
        "V25", "dev",
        "36yo with prior chemotherapy (B6) AND trying to conceive for 15 months",
        _make(age=36, b6_cancer_treatment=True, trying="now", months_trying=15),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "Established gonadotoxic history (R3) routes directly to specialist andrology ahead of general fertility intake (R4).",
    ),
    Vignette(
        "V26", "dev",
        "30yo trying to conceive for exactly 12 months, partner aged 28, no other risk factors",
        _make(age=30, trying="now", months_trying=12, partner_35_plus=False),
        Triage.FERTILITY_SERVICES, "R4",
        "Exact 12-month boundary triggers standard couple fertility and semen evaluation.",
    ),
    Vignette(
        "V27", "dev",
        "34yo trying to conceive for exactly 6 months, partner aged 36, no other risk factors",
        _make(age=34, trying="now", months_trying=6, partner_35_plus=True),
        Triage.FERTILITY_SERVICES, "R4",
        "Exact 6-month boundary with partner >=35 triggers earlier fertility workup.",
    ),
    Vignette(
        "V28", "dev",
        "34yo trying to conceive for 6 months, partner aged 31 (under 35), no other risk factors",
        _make(age=34, trying="now", months_trying=6, partner_35_plus=False),
        Triage.SELF_CARE, "R7",
        "At 6 months with a partner under 35 and zero risk factors, premature fertility referral is avoided.",
    ),
    Vignette(
        "V29", "dev",
        "28yo with clinically diagnosed varicocele (B5), asymptomatic, planning children within a year",
        _make(age=28, trying="within_year", b5_varicocele=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Known varicocele in a man planning conception warrants routine urology/andrology assessment and baseline semen analysis.",
    ),
    Vignette(
        "V30", "dev",
        "39yo with mumps orchitis history (B3), groin surgery (B4), and diabetes (C2) -> Score 6 (High)",
        _make(age=39, b3_mumps_orchitis=True, b4_testis_groin_surgery_injury=True, c2_diabetes_metabolic=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Cumulative multi-factor burden of 3 Moderate factors (6 points = High band) warrants specialist andrology review.",
    ),
    Vignette(
        "V31", "dev",
        "33yo with varicocele (B5 = Major) and erectile difficulty (D7 = 2 pts) -> Score 5 bumped to High band",
        _make(age=33, b5_varicocele=True, d7_sexual_function_issue=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "A Major factor combined with sexual dysfunction forces High band and specialist routing.",
    ),
    Vignette(
        "V32", "dev",
        "35yo with childhood groin surgery (B4=2), BMI 33 (C1=2), tobacco (C3=1), heavy alcohol (C4=1) -> Score 6",
        _make(age=35, b4_testis_groin_surgery_injury=True, bmi=33.0, c3_tobacco=True, c4_alcohol="heavy"),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Surgical history plus severe modifiable metabolic/lifestyle clustering (6 pts = High) routes to specialist care.",
    ),
    Vignette(
        "V33", "dev",
        "27yo taking finasteride for hair loss (C10=True), no other risk factors (Score 0)",
        _make(age=27, c10_listed_medicine=True),
        Triage.PRIMARY_CARE, "R6",
        "Fertility-relevant medications require primary care review with an explicit warning never to stop medicine without doctor advice.",
    ),
    Vignette(
        "V34", "dev",
        "41yo with well-controlled diabetes (C2=True), BMI 24, non-smoker (Score 2, Band Low)",
        _make(age=41, c2_diabetes_metabolic=True, bmi=24.0),
        Triage.PRIMARY_CARE, "R6",
        "Diabetes alone (even in Low score band) explicitly routes to primary care for metabolic and reproductive check-up.",
    ),
    Vignette(
        "V35", "dev",
        "36yo with isolated erectile/ejaculatory concern (D7=True), no other history (Score 2, Band Low)",
        _make(age=36, d7_sexual_function_issue=True),
        Triage.PRIMARY_CARE, "R6",
        "Sexual function concerns route to primary care for morning testosterone, metabolic screening, and targeted support.",
    ),
    Vignette(
        "V36", "dev",
        "31yo baker with groin heat exposure (C7=1), chemical exposure (C8=1), and recent high fever (C9=1) -> Score 3",
        _make(age=31, c7_heat_exposure=True, c8_work_chemicals=True, c9_recent_fever_illness=True),
        Triage.PRIMARY_CARE, "R6",
        "Three minor occupational/transient exposures sum to 3 points (Medium band), warranting primary care guidance and repeat check after 74 days.",
    ),
    Vignette(
        "V37", "dev",
        "29yo who skips 7 of 19 triage questions (>30% missing) with zero positive answers",
        _make(
            age=29,
            b3_mumps_orchitis=None, b4_testis_groin_surgery_injury=None, b8_untreated_infection=None,
            c3_tobacco=None, c4_alcohol=None, c5_testosterone_steroids=None, c6_recreational_drugs=None,
        ),
        Triage.PRIMARY_CARE, "R6",
        "Skipping >30% of non-safety triage questions triggers the low-confidence flag and bumps Low band to Medium (R6).",
    ),
    Vignette(
        "V38", "dev",
        "26yo smoker (C3=1) who drinks regularly (8-14/wk, C4='regular'=0), BMI 24 -> Score 1",
        _make(age=26, c3_tobacco=True, c4_alcohol="regular", bmi=24.0),
        Triage.SELF_CARE, "R7",
        "A single minor modifiable factor (Score 1, Low band) receives targeted smoking cessation and preconception self-care advice.",
    ),
    Vignette(
        "V39", "dev",
        "32yo with BMI 30.5 (C1=2) and zero other risk factors -> Score 2 (Low band)",
        _make(age=32, bmi=30.5),
        Triage.SELF_CARE, "R7",
        "Isolated mild obesity (2 pts, Low band) without diabetes or subfertility receives actionable weight and metabolic self-care guidance.",
    ),
    Vignette(
        "V40", "dev",
        "25yo healthy man who selects 'prefer not to say' (None) on acute pain D1",
        _make(age=25, d1_sudden_severe_pain=None),
        Triage.SELF_CARE, "R7",
        "Prefer-not-to-say on a red flag never triggers a false emergency, but activates the explicit safety-net warning on the result screen.",
    ),

    # ===========================================================================
    # LOCKED HOLD-OUT TEST SET (20 Vignettes: V41-V60 — never tune rules against)
    # ===========================================================================
    Vignette(
        "V41", "holdout",
        "Adult user with missing age (age=None) despite checking consent",
        _make(age=None, consent=True),
        Triage.STOP, "R0",
        "Age verification is required to confirm adult eligibility before screening.",
    ),
    Vignette(
        "V42", "holdout",
        "16yo user with acute testicular pain who attempts to use adult tool",
        _make(age=16, consent=True, d1_sudden_severe_pain=True),
        Triage.STOP, "R0",
        "Hard consent/age gate R0 executes first and directs underage users out of adult intake with emergency disclaimer.",
    ),
    Vignette(
        "V43", "holdout",
        "21yo with sudden scrotal swelling and redness (D2) starting 6 hours ago",
        _make(age=21, d2_sudden_swelling=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Isolated sudden scrotal swelling/inflammation requires immediate emergency evaluation.",
    ),
    Vignette(
        "V44", "holdout",
        "30yo with fever plus testicular pain (D3) and untreated STI history (B8)",
        _make(age=30, d3_fever_with_pain_swelling=True, b8_untreated_infection=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Acute febrile orchitis/epididymitis (R1) overrides routine STI primary care (R6).",
    ),
    Vignette(
        "V45", "holdout",
        "27yo bodybuilder on steroids (C5) presenting with prolonged erection >4h (D4)",
        _make(age=27, c5_testosterone_steroids=True, d4_prolonged_erection=True),
        Triage.URGENT_EMERGENCY, "R1",
        "Acute priapism emergency (R1) overrides routine andrology steroid referral (R5).",
    ),
    Vignette(
        "V46", "holdout",
        "33yo with painless testicular size change (D5) and childhood undescended testis (B2)",
        _make(age=33, d5_lump_or_size_change=True, b2_undescended_testis=True),
        Triage.URGENT_ASSESSMENT, "R2",
        "Cryptorchidism increases testicular cancer risk; a new size change requires fast-track R2 urgent urology over routine R5.",
    ),
    Vignette(
        "V47", "holdout",
        "44yo with blood in semen (D6) taking fertility-relevant prescription medication (C10)",
        _make(age=44, d6_blood_in_semen_or_persistent_ache=True, c10_listed_medicine=True),
        Triage.URGENT_ASSESSMENT, "R2",
        "Urgent assessment for hematospermia (R2) takes priority over primary care medication review (R6) while preserving the medicine note.",
    ),
    Vignette(
        "V48", "holdout",
        "31yo with prior abnormal semen test (B1) and trying for 8 months with 36yo partner (R4 also true)",
        _make(age=31, b1_abnormal_semen_test=True, trying="now", months_trying=8, partner_35_plus=True),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "When a man already has a known abnormal semen test (R3), direct andrology/urology evaluation supersedes general fertility intake (R4).",
    ),
    Vignette(
        "V49", "holdout",
        "28yo with chromosomal microdeletion (B7) and varicocele (B5)",
        _make(age=28, b7_genetic_condition=True, b5_varicocele=True),
        Triage.ANDROLOGY_UROLOGY, "R3",
        "Established genetic etiology (R3) takes precedence over routine varicocele rule (R5).",
    ),
    Vignette(
        "V50", "holdout",
        "37yo trying for 11 months with 32yo partner (partner_35_plus=False), no other risk factors",
        _make(age=37, trying="now", months_trying=11, partner_35_plus=False),
        Triage.SELF_CARE, "R7",
        "At 11 months with a partner under 35 and zero risk factors, the 12-month R4 threshold is not yet met.",
    ),
    Vignette(
        "V51", "holdout",
        "40yo trying for 15 months with partner under 35, also has diabetes (C2) and smokes (C3)",
        _make(age=40, trying="now", months_trying=15, partner_35_plus=False, c2_diabetes_metabolic=True, c3_tobacco=True),
        Triage.FERTILITY_SERVICES, "R4",
        "Couple conception delay >=12 months (R4) takes precedence over primary care metabolic routing (R6).",
    ),
    Vignette(
        "V52", "holdout",
        "23yo using anabolic steroids (C5) and recreational drugs (C6), not planning children",
        _make(age=23, trying="not_planning", c5_testosterone_steroids=True, c6_recreational_drugs=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Anabolic steroid use routes to specialist care (R5) regardless of current conception plans to prevent irreversible hypogonadism.",
    ),
    Vignette(
        "V53", "holdout",
        "35yo with childhood undescended testis (B2=3) and erectile difficulty (D7=2) -> High band",
        _make(age=35, b2_undescended_testis=True, d7_sexual_function_issue=True),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "Undescended testis plus sexual dysfunction triggers High band and specialist andrology evaluation.",
    ),
    Vignette(
        "V54", "holdout",
        "43yo with BMI 34 (C1=2), mumps orchitis (B3=2), tobacco (C3=1), heavy alcohol (C4=1) -> Score 6",
        _make(age=43, bmi=34.0, b3_mumps_orchitis=True, c3_tobacco=True, c4_alcohol="heavy"),
        Triage.ANDROLOGY_UROLOGY, "R5",
        "High cumulative score (6 pts) from historical plus modifiable factors routes to routine andrology/urology.",
    ),
    Vignette(
        "V55", "holdout",
        "34yo with mumps orchitis (B3=2) and regular sauna/groin heat exposure (C7=1) -> Score 3",
        _make(age=34, b3_mumps_orchitis=True, c7_heat_exposure=True),
        Triage.PRIMARY_CARE, "R6",
        "Moderate history (2 pts) plus minor modifiable heat exposure (1 pt) reaches Medium band (3 pts) for primary care evaluation.",
    ),
    Vignette(
        "V56", "holdout",
        "38yo with prior inguinal hernia repair (B4=2) and occupational chemical exposure (C8=1) -> Score 3",
        _make(age=38, b4_testis_groin_surgery_injury=True, c8_work_chemicals=True),
        Triage.PRIMARY_CARE, "R6",
        "Prior groin surgery combined with occupational toxin exposure reaches Medium band (R6) to order baseline semen analysis.",
    ),
    Vignette(
        "V57", "holdout",
        "31yo with BMI 31 (C1=2, Low band) who skips 6 of 19 triage questions (>30% missing) -> bumped to Medium",
        _make(
            age=31, bmi=31.0,
            b2_undescended_testis=None, b3_mumps_orchitis=None, b5_varicocele=None,
            c4_alcohol=None, c6_recreational_drugs=None, c8_work_chemicals=None,
        ),
        Triage.PRIMARY_CARE, "R6",
        "Low-confidence missingness (>30% skipped) bumps a borderline Low score (2 pts) into Medium band (R6).",
    ),
    Vignette(
        "V58", "holdout",
        "29yo with childhood mumps orchitis (B3=2) only, no other risk factors, not trying",
        _make(age=29, b3_mumps_orchitis=True),
        Triage.SELF_CARE, "R7",
        "A single isolated Moderate history factor (2 pts = Low band) without trying delay stays in preventive self-care.",
    ),
    Vignette(
        "V59", "holdout",
        "33yo with BMI 29.9 (just below obesity cutoff) and occasional alcohol (1-7/wk) -> Score 0",
        _make(age=33, bmi=29.9, c4_alcohol="occasional"),
        Triage.SELF_CARE, "R7",
        "Sub-threshold overweight BMI (<30) and occasional alcohol score 0 points and receive preventive lifestyle tips.",
    ),
    Vignette(
        "V60", "holdout",
        "27yo who skips exactly 5 of 19 triage questions (26.3% < 30% threshold) with high fever 2 months ago (C9=1)",
        _make(
            age=27, c9_recent_fever_illness=True,
            c4_alcohol=None, c6_recreational_drugs=None, b8_untreated_infection=None,
            d7_sexual_function_issue=None, c5_testosterone_steroids=None,
        ),
        Triage.SELF_CARE, "R7",
        "Skipping <=30% of questions does not trigger a low-confidence bump; transient recent fever (1 pt) receives self-care & re-test timing note.",
    ),
]

DEV_VIGNETTES: List[Vignette] = [v for v in ALL_VIGNETTES if v.split == "dev"]
HOLDOUT_VIGNETTES: List[Vignette] = [v for v in ALL_VIGNETTES if v.split == "holdout"]


def evaluate_vignette_suite(vignettes: List[Vignette]) -> Dict[str, object]:
    """Run deterministic rule engine across a vignette list and summarize accuracy."""
    passed = 0
    by_rule: Dict[str, int] = {}
    failures: List[str] = []

    for v in vignettes:
        res = evaluate(v.intake)
        by_rule[v.expected_rule] = by_rule.get(v.expected_rule, 0) + 1
        if res.triage == v.expected_triage and res.rule == v.expected_rule:
            passed += 1
        else:
            failures.append(
                f"{v.vid}: expected ({v.expected_triage.value}, {v.expected_rule}), "
                f"got ({res.triage.value}, {res.rule})"
            )

    return {
        "total": len(vignettes),
        "passed": passed,
        "accuracy": round(passed / max(1, len(vignettes)), 4),
        "rule_coverage": dict(sorted(by_rule.items())),
        "failures": failures,
    }


if __name__ == "__main__":
    dev_rep = evaluate_vignette_suite(DEV_VIGNETTES)
    hold_rep = evaluate_vignette_suite(HOLDOUT_VIGNETTES)
    print(f"DEV set ({dev_rep['passed']}/{dev_rep['total']}): {dev_rep['rule_coverage']}")
    print(f"HOLDOUT set ({hold_rep['passed']}/{hold_rep['total']}): {hold_rep['rule_coverage']}")