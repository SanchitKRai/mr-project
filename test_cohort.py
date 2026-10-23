"""Unit tests for the Days 6-7 synthetic cohort generator (`cohort.py`)."""
from cohort import (
     compute_latent_impairment_prob,
    generate_cohort,
    summarize_cohort,
    uci_sanity_check_summary,
)
from test_triage import make


def test_reproducibility_with_documented_seed():
    c1 = generate_cohort(n=250, seed=20261023)
    c2 = generate_cohort(n=250, seed=20261023)
    assert [r.latent_impaired_prob for r in c1] == [r.latent_impaired_prob for r in c2]
    assert [r.rule_id for r in c1] == [r.rule_id for r in c2]


def test_b1_does_not_leak_into_latent_biological_probability():
    """B1 is a downstream lab result; changing B1 must not alter the biological prior."""
    p_false = compute_latent_impairment_prob(make(b1_abnormal_semen_test=False))
    p_true = compute_latent_impairment_prob(make(b1_abnormal_semen_test=True))
    assert p_false == p_true


def test_published_odds_ratios_increase_impairment_monotonically():
    p_base = compute_latent_impairment_prob(make(bmi=22.0))
    p_over = compute_latent_impairment_prob(make(bmi=27.0))
    p_obese = compute_latent_impairment_prob(make(bmi=33.0))
    p_morbid = compute_latent_impairment_prob(make(bmi=42.0))
    p_steroids = compute_latent_impairment_prob(make(c5_testosterone_steroids=True))
    assert p_base < p_over < p_obese < p_morbid < p_steroids


def test_prevalences_and_correlations_in_expected_ranges():
    cohort = generate_cohort(n=2000, seed=20261023)
    summary = summarize_cohort(cohort)
    # Overall impaired semen quality should sit in the 14% - 26% window
    assert 0.14 <= summary["latent_impaired_rate"] <= 0.26
    # Varicocele ~15%, Obesity ~18-32%
    assert 0.11 <= summary["varicocele_rate"] <= 0.19
    assert 0.16 <= summary["obesity_rate"] <= 0.34
    # Older men must have higher diabetes prevalence than men under 30
    assert summary["dm_rate_40_plus"] > summary["dm_rate_under_30"] * 1.8
    # Every triage rule R1-R7 should be represented in a 2,000-man cohort
    for rule in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert summary["rule_distribution"].get(rule, 0) > 0, f"Missing {rule}"


def test_uci_sanity_check_directions_are_all_above_one():
    cohort = generate_cohort(n=2000, seed=20261023)
    sanity = uci_sanity_check_summary(cohort)
    assert sanity["surgery_true_vs_false_ratio"] > 1.2
    assert sanity["fever_true_vs_false_ratio"] > 1.2
    assert sanity["tobacco_true_vs_false_ratio"] > 1.15