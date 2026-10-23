"""Synthetic cohort generator for Male Reproductive Health Screening (Days 6-7).

Generates realistic adult male Intake profiles anchored to UAE/Abu Dhabi
demographic distributions and EAU/WHO/meta-analysis prevalence & odds ratios.

Crucial evaluation design:
- The hidden biological outcome (`impaired_semen_quality`) is drawn from an
  epidemiological logistic model using published Odds Ratios (ORs), NOT from
  the triage rules in `triage.py`.
- `b1_abnormal_semen_test` is generated downstream of the latent outcome and
  prior testing likelihood (e.g., trying to conceive), avoiding causal leakage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from triage import Band, Intake, Triage, evaluate


# ---------------------------------------------------------------------------
# Published Odds Ratios (OR) for impaired semen quality (WHO 6th ed criteria)
# Every figure is mapped to the literature table in the project documentation.
# ---------------------------------------------------------------------------
LITERATURE_ORS: Dict[str, float] = {
    "age_per_year_over_35": 1.03,       # Johnson et al. 2015
    "b2_undescended_testis": 4.20,      # EAU Guidelines 2024 / StatPearls
    "b3_mumps_orchitis": 2.80,          # Davis et al. 2010
    "b4_testis_groin_surgery_injury": 2.10,  # EAU Guidelines 2024
    "b5_varicocele": 2.60,              # Global Andrology Forum 2024 / EAU
    "b6_cancer_treatment": 6.50,        # ASRM / EAU gonadotoxicity guidelines
    "b7_genetic_condition": 8.00,       # EAU Guidelines 2024
    "b8_untreated_infection": 1.90,     # EAU Guidelines 2024
    "bmi_overweight_25_30": 1.11,       # Sermondade et al. 2013 (Hum Reprod Update)
    "bmi_obese_30_40": 1.28,            # Sermondade et al. 2013 (Hum Reprod Update)
    "bmi_morbid_40_plus": 2.04,         # Sermondade et al. 2013 (Hum Reprod Update)
    "c2_diabetes_metabolic": 1.60,      # Pergialiotis et al. 2016 / UAEDIAB
    "c3_tobacco": 1.55,                 # Sharma et al. 2016 (Eur Urol)
    "c4_alcohol_heavy": 1.45,           # Ricci et al. 2017 / Amor et al. 2022
    "c5_testosterone_steroids": 5.50,   # McBride & Coward 2016
    "c6_recreational_drugs": 1.40,      # Payne et al. 2019
    "c7_heat_exposure": 1.35,           # Mieusset & Bujan / EAU
    "c8_work_chemicals": 1.45,          # Jurewicz et al. 2009
    "c9_recent_fever_illness": 1.80,    # Carlsen et al. 2003
    "c10_listed_medicine": 1.50,        # Drobnis & Nangia 2017
    "d7_sexual_function_issue": 1.30,   # Lotti & Maggi 2018 (shared hypogonadal pathway)
}

# Baseline probability of impaired semen quality in a 25yo man with zero risk factors
BASELINE_IMPAIRED_PROB = 0.075


@dataclass
class SyntheticRecord:
    """One synthetic man: his Intake, his true latent biological state, and rule triage."""
    person_id: int
    intake: Intake
    latent_impaired_prob: float
    latent_impaired_semen: bool
    had_prior_semen_test: bool
    rule_triage: Triage
    rule_id: str
    rule_band: Optional[Band]
    rule_score: int
    rule_low_confidence: bool


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def compute_latent_impairment_prob(x: Intake) -> float:
    """Epidemiological logistic model for hidden outcome `impaired_semen_quality`.

    Note: `b1_abnormal_semen_test` is intentionally EXCLUDED here because a prior
    lab test is a downstream clinical measurement of the latent state, not a cause.
    """
    log_odds = _logit(BASELINE_IMPAIRED_PROB)
    age = x.age if x.age is not None else 30

    if age > 35:
        log_odds += (age - 35) * math.log(LITERATURE_ORS["age_per_year_over_35"])

    # Binary history & lifestyle factors
    binary_map = [
        (x.b2_undescended_testis, "b2_undescended_testis"),
        (x.b3_mumps_orchitis, "b3_mumps_orchitis"),
        (x.b4_testis_groin_surgery_injury, "b4_testis_groin_surgery_injury"),
        (x.b5_varicocele, "b5_varicocele"),
        (x.b6_cancer_treatment, "b6_cancer_treatment"),
        (x.b7_genetic_condition, "b7_genetic_condition"),
        (x.b8_untreated_infection, "b8_untreated_infection"),
        (x.c2_diabetes_metabolic, "c2_diabetes_metabolic"),
        (x.c3_tobacco, "c3_tobacco"),
        (x.c5_testosterone_steroids, "c5_testosterone_steroids"),
        (x.c6_recreational_drugs, "c6_recreational_drugs"),
        (x.c7_heat_exposure, "c7_heat_exposure"),
        (x.c8_work_chemicals, "c8_work_chemicals"),
        (x.c9_recent_fever_illness, "c9_recent_fever_illness"),
        (x.c10_listed_medicine, "c10_listed_medicine"),
        (x.d7_sexual_function_issue, "d7_sexual_function_issue"),
    ]
    for val, key in binary_map:
        if val is True:
            log_odds += math.log(LITERATURE_ORS[key])

    # BMI tiers from Sermondade et al. (2013)
    if x.bmi is not None:
        if x.bmi >= 40.0:
            log_odds += math.log(LITERATURE_ORS["bmi_morbid_40_plus"])
        elif x.bmi >= 30.0:
            log_odds += math.log(LITERATURE_ORS["bmi_obese_30_40"])
        elif x.bmi >= 25.0:
            log_odds += math.log(LITERATURE_ORS["bmi_overweight_25_30"])

    # Heavy alcohol
    if x.c4_alcohol == "heavy":
        log_odds += math.log(LITERATURE_ORS["c4_alcohol_heavy"])

    return _sigmoid(log_odds)


def generate_cohort(
    n: int = 2000,
    seed: int = 20261023,
    include_missingness: bool = True,
) -> List[SyntheticRecord]:
    """Generate `n` realistic synthetic adult men with documented correlations."""
    rng = np.random.default_rng(seed)
    records: List[SyntheticRecord] = []

    for i in range(n):
        # 1. Age (18-60, skewed toward young/middle adult male demography in UAE)
        age = int(np.clip(rng.normal(loc=33.5, scale=8.5), 18, 60))

        # 2. Family / conception context (correlated with age)
        if 25 <= age <= 42:
            trying = str(rng.choice(
                ["now", "within_year", "not_planning"], p=[0.30, 0.22, 0.48]
            ))
        else:
            trying = str(rng.choice(
                ["now", "within_year", "not_planning"], p=[0.12, 0.13, 0.75]
            ))

        # 3. Correlated Metabolic Profile: Age -> BMI -> Diabetes (C1, C2)
        bmi_mean = 26.8 + 0.11 * (age - 30)
        bmi = round(float(np.clip(rng.normal(loc=bmi_mean, scale=4.6), 17.5, 48.0)), 1)

        dm_logit = _logit(0.06) + 0.055 * (age - 30) + 0.095 * max(0.0, bmi - 25.0)
        c2_diabetes = bool(rng.random() < _sigmoid(dm_logit))

        # 4. Correlated Lifestyle Cluster (C3-C9)
        c3_tobacco = bool(rng.random() < 0.24)
        # Alcohol in UAE context: high abstinence (~72%), higher drinking if smoker
        if c3_tobacco:
            c4_alcohol = str(rng.choice(
                ["none", "occasional", "regular", "heavy"], p=[0.58, 0.24, 0.12, 0.06]
            ))
        else:
            c4_alcohol = str(rng.choice(
                ["none", "occasional", "regular", "heavy"], p=[0.77, 0.15, 0.055, 0.025]
            ))

        # Gym/anabolic steroid use (C5): clustered in ages 20-38
        steroid_prob = 0.042 if 20 <= age <= 38 else 0.010
        c5_steroids = bool(rng.random() < steroid_prob)

        # Recreational drugs (C6): correlated with tobacco and heavy alcohol
        drug_prob = 0.006 + (0.022 if c3_tobacco else 0.0) + (0.04 if c4_alcohol == "heavy" else 0.0)
        c6_drugs = bool(rng.random() < drug_prob)

        c7_heat = bool(rng.random() < 0.10)
        c8_chem = bool(rng.random() < (0.14 if c7_heat else 0.05))
        c9_fever = bool(rng.random() < 0.05)

        med_prob = 0.025 + (0.03 if age >= 40 else 0.0) + (0.03 if c2_diabetes else 0.0)
        c10_meds = bool(rng.random() < med_prob)

        # 5. Medical & Congenital History (B2-B8)
        b2_udt = bool(rng.random() < 0.015)
        b3_mumps = bool(rng.random() < 0.008)
        b4_surg = bool(rng.random() < (0.18 if b2_udt else 0.032))
        b5_varicocele = bool(rng.random() < 0.15)
        b6_cancer = bool(rng.random() < (0.003 if age < 40 else 0.009))
        b7_genetic = bool(rng.random() < 0.004)
        sti_prob = 0.028 + (0.035 if c6_drugs or c4_alcohol == "heavy" else 0.0)
        b8_sti = bool(rng.random() < sti_prob)

        # 6. Sexual function (D7) correlated with Age, Diabetes, Obesity, Steroids
        ed_logit = (
            _logit(0.05)
            + 0.05 * max(0, age - 30)
            + (1.10 if c2_diabetes else 0.0)
            + (0.45 if bmi >= 30.0 else 0.0)
            + (0.75 if c5_steroids else 0.0)
        )
        d7_ed = bool(rng.random() < _sigmoid(ed_logit))

        # 7. Acute Red Flags & Urgent Symptoms (D1-D6, D8) - low prevalence in screening
        d1_pain = bool(rng.random() < 0.003)
        d2_swell = bool(rng.random() < (0.35 if d1_pain else 0.003))
        d3_fev_pain = bool(rng.random() < (0.25 if d2_swell else 0.002))
        d4_priapism = bool(rng.random() < 0.001)
        d5_lump = bool(rng.random() < 0.008)
        d6_blood_ache = bool(rng.random() < (0.035 if b5_varicocele or b8_sti else 0.008))
        d8_injury = bool(rng.random() < 0.002)

        # Build preliminary Intake (before B1 and trying duration) to compute latent state
        intake = Intake(
            consent=True,
            age=age,
            trying=trying,
            b1_abnormal_semen_test=False,
            b2_undescended_testis=b2_udt,
            b3_mumps_orchitis=b3_mumps,
            b4_testis_groin_surgery_injury=b4_surg,
            b5_varicocele=b5_varicocele,
            b6_cancer_treatment=b6_cancer,
            b7_genetic_condition=b7_genetic,
            b8_untreated_infection=b8_sti,
            bmi=bmi,
            c2_diabetes_metabolic=c2_diabetes,
            c3_tobacco=c3_tobacco,
            c4_alcohol=c4_alcohol,
            c5_testosterone_steroids=c5_steroids,
            c6_recreational_drugs=c6_drugs,
            c7_heat_exposure=c7_heat,
            c8_work_chemicals=c8_chem,
            c9_recent_fever_illness=c9_fever,
            c10_listed_medicine=c10_meds,
            d1_sudden_severe_pain=d1_pain,
            d2_sudden_swelling=d2_swell,
            d3_fever_with_pain_swelling=d3_fev_pain,
            d4_prolonged_erection=d4_priapism,
            d5_lump_or_size_change=d5_lump,
            d6_blood_in_semen_or_persistent_ache=d6_blood_ache,
            d7_sexual_function_issue=d7_ed,
            d8_recent_groin_injury_swelling=d8_injury,
        )

        # 8. Draw hidden biological outcome (`latent_impaired_semen`)
        impaired_prob = compute_latent_impairment_prob(intake)
        latent_impaired = bool(rng.random() < impaired_prob)

        # 9. Trying duration & partner age (correlated with latent impairment & male age)
        if trying == "now":
            # Men with impaired semen take longer to conceive, so in a cross-sectional
            # intake they have longer `months_trying` on average.
            mean_months = 13.0 if latent_impaired else 5.5
            months_trying = int(np.clip(rng.exponential(scale=mean_months), 1, 60))
            partner_35_plus = bool(rng.random() < _sigmoid(-3.2 + 0.18 * (age - 30)))
        else:
            months_trying = None
            partner_35_plus = None

        intake.months_trying = months_trying
        intake.partner_35_plus = partner_35_plus

        # 10. Downstream clinical history: B1 (prior abnormal semen test)
        # Only men who have ever had a test can answer True; testing is much more
        # common if trying >= 6 months or with major history (B2, B5, B6).
        prior_test_prob = 0.03
        if trying == "now" and months_trying is not None:
            prior_test_prob += 0.30 if months_trying >= 12 else (0.12 if months_trying >= 6 else 0.03)
        if b2_udt or b5_varicocele or b6_cancer:
            prior_test_prob += 0.15
        had_prior_test = bool(rng.random() < min(prior_test_prob, 0.85))

        if had_prior_test:
            # Lab semen analysis has ~88% sensitivity and ~92% specificity for true state
            b1_abn = bool(rng.random() < (0.88 if latent_impaired else 0.08))
        else:
            b1_abn = False
        intake.b1_abnormal_semen_test = b1_abn

        # 11. Realistic missingness ("prefer not to say" -> None)
        if include_missingness:
            # ~5% of users skip many questions (triggering low-confidence >30% check)
            heavy_skipper = bool(rng.random() < 0.05)
            sensitive_fields = [
                ("c4_alcohol", 0.08),
                ("c6_recreational_drugs", 0.08),
                ("b8_untreated_infection", 0.06),
                ("d7_sexual_function_issue", 0.07),
                ("c5_testosterone_steroids", 0.04),
            ]
            for fname, base_p in sensitive_fields:
                if rng.random() < (0.65 if heavy_skipper else base_p):
                    setattr(intake, fname, None)

            if heavy_skipper:
                extra_skip_pool = [
                    "b3_mumps_orchitis", "b4_testis_groin_surgery_injury",
                    "c3_tobacco", "c7_heat_exposure", "c8_work_chemicals",
                    "c9_recent_fever_illness",
                ]
                for fname in extra_skip_pool:
                    if rng.random() < 0.70:
                        setattr(intake, fname, None)

        res = evaluate(intake)
        records.append(
            SyntheticRecord(
                person_id=i,
                intake=intake,
                latent_impaired_prob=round(impaired_prob, 4),
                latent_impaired_semen=latent_impaired,
                had_prior_semen_test=had_prior_test,
                rule_triage=res.triage,
                rule_id=res.rule,
                rule_band=res.band,
                rule_score=res.score,
                rule_low_confidence=res.low_confidence,
            )
        )

    return records


def summarize_cohort(records: List[SyntheticRecord]) -> Dict[str, object]:
    """Return summary metrics for slide reporting and sanity verification."""
    n = len(records)
    impaired_rate = sum(r.latent_impaired_semen for r in records) / n
    obesity_rate = sum(
        (r.intake.bmi is not None and r.intake.bmi >= 30.0) for r in records
    ) / n
    diabetes_rate = sum(r.intake.c2_diabetes_metabolic is True for r in records) / n
    varicocele_rate = sum(r.intake.b5_varicocele is True for r in records) / n

    # Check Age-BMI and BMI-Diabetes correlation
    young_dm = [r.intake.c2_diabetes_metabolic is True for r in records if (r.intake.age or 30) < 30]
    older_dm = [r.intake.c2_diabetes_metabolic is True for r in records if (r.intake.age or 30) >= 40]

    rule_counts: Dict[str, int] = {}
    for r in records:
        rule_counts[r.rule_id] = rule_counts.get(r.rule_id, 0) + 1

    return {
        "n": n,
        "latent_impaired_rate": round(impaired_rate, 4),
        "obesity_rate": round(obesity_rate, 4),
        "diabetes_rate": round(diabetes_rate, 4),
        "varicocele_rate": round(varicocele_rate, 4),
        "dm_rate_under_30": round(sum(young_dm) / max(1, len(young_dm)), 4),
        "dm_rate_40_plus": round(sum(older_dm) / max(1, len(older_dm)), 4),
        "rule_distribution": dict(sorted(rule_counts.items())),
    }


def uci_sanity_check_summary(records: List[SyntheticRecord]) -> Dict[str, float]:
    """Compare overlapping feature directions against the UCI Fertility dataset (Gil et al., 2012).

    UCI Fertility (N=100) has 12% 'Altered' diagnosis and measures overlapping
    proxies: surgical intervention (B4), recent high fever (C9), smoking (C3),
    and alcohol (C4). We verify that each factor increases impairment rate in
    our synthetic cohort just as in the literature/UCI reference.
    """
    def rate_when(pred) -> float:
        sub = [r for r in records if pred(r.intake)]
        return sum(r.latent_impaired_semen for r in sub) / max(1, len(sub))

    return {
        "overall_impaired_rate": round(rate_when(lambda _: True), 4),
        "surgery_true_vs_false_ratio": round(
            rate_when(lambda x: x.b4_testis_groin_surgery_injury is True)
            / rate_when(lambda x: x.b4_testis_groin_surgery_injury is False), 2
        ),
        "fever_true_vs_false_ratio": round(
            rate_when(lambda x: x.c9_recent_fever_illness is True)
            / rate_when(lambda x: x.c9_recent_fever_illness is False), 2
        ),
        "tobacco_true_vs_false_ratio": round(
            rate_when(lambda x: x.c3_tobacco is True)
            / rate_when(lambda x: x.c3_tobacco is False), 2
        ),
    }


if __name__ == "__main__":
    cohort = generate_cohort(n=2000, seed=20261023)
    print("Cohort summary:", summarize_cohort(cohort))
    print("UCI direction sanity check:", uci_sanity_check_summary(cohort))