# Future Health Digital Challenge — Evaluation & Fairness Report

> **Disclosure:** Cohort metrics below are evaluated on a reproducible synthetic population
> ($N = 2,500$, `seed = 20261023`) parameterized from published EAU/AUA/WHO guidelines and UAE
> epidemiological prevalence studies. Clinical rules are verified against 60 expert-crafted vignettes
> (40 development + 20 locked hold-out cases). Simulated data does not replace prospective clinical trials.

## 1. Executive KPI Summary (Held-Out Test Cohort, $N = 750$)

| Metric / Evaluation Pillar | Result | Clinical & Operational Meaning |
| :--- | :--- | :--- |
| **Clinical Vignette Accuracy (Dev / Hold-Out)** | **40/40 (100%) / 20/20 (100%)** | Zero safety or routing errors across all `R0–R7` rules and boundary cases |
| **ML Second-Opinion ROC AUC (`xgboost_native`)** | **0.7251** | Matches idealized upstream rule score (0.7259) without `B1` lab leakage |
| **Probabilistic Calibration (Brier Score)** | **0.1142** | Well-calibrated risk probabilities across the ~17.1% baseline prevalence |
| **Impaired Semen Clinical Capture (Sensitivity)** | **84.5%** | Share of truly impaired men routed to clinical care (`R1–R6`) |
| **Upstream Detection Rate (Pre-Infertility Diagnosis)** | **81.9%** (77/94) | Men with latent impairment caught *before* 12 months of trying |
| **Mean Time-to-Evaluation Saved (Trying <12m)** | **9.0 months earlier** | Earlier action vs. waiting for the standard 12-month infertility threshold |
| **Unnecessary Clinic Visit Deflection (`R7`)** | **41.7%** | Low-risk men safely guided to preventive self-care without a clinic visit |
| **Age Subgroup Fairness Gap ($|\Delta \text{AUC}|$)** | **0.0171** | Near-zero discrimination gap between men $<35$ (0.734) and $\ge 35$ (0.717) |

## 2. Final Hybrid Triage Pathway Distribution ($N = 750$)

| Triage Destination | Count | Share (%) | Primary Clinical Action |
| :--- | :---: | :---: | :--- |
| **R1 — Urgent Emergency Care** | 6 | 0.8% | Immediate ED / surgical evaluation (torsion, acute infection, priapism) |
| **R2 — Fast-Track Urology (Days)** | 14 | 1.9% | Prompt scrotal ultrasound and urologist exam (new mass, hematospermia) |
| **R3 / R5 — Andrology & Urology Specialist** | 157 | 20.9% | Specialist workup (varicocele, cryptorchidism, steroid cessation, abnormal semen) |
| **R4 — Couple Fertility Services** | 29 | 3.9% | WHO laboratory semen analysis and coordinated couple evaluation |
| **R6 — Primary Care & Metabolic Review** | 231 | 30.8% | Glycemic/BMI optimization, STI NAAT panel, medication review, baseline lab order |
| **R7 — Preventive Self-Care & Guidance** | 313 | 41.7% | Targeted lifestyle adjustments, heat/smoking reduction, and re-screen reminder |

## 3. Rule Band vs. ML Second-Opinion Band Confusion Matrix ($N = 750$)

| Deterministic Rule Band \ ML Band | ML: Low | ML: Medium | ML: High |
| :--- | :---: | :---: | :---: |
| **Rule: Low** | 353 | 83 | 1 |
| **Rule: Medium** | 36 | 176 | 29 |
| **Rule: High** | 0 | 19 | 53 |

- **Total Band Disagreements Logged:** 168 / 750 (22.4%)
- **ML Higher (`ml_higher`):** 113 cases (cautiously upgraded to higher band)
- **Rules Higher (`rules_higher`):** 55 cases (retained higher rule band; never downgraded)

### Concrete Examples of Logged Disagreements (`ml_higher` Upgrades)

| Person ID | Rule Band | ML Band ($\hat{p}$) | Final Band | Triage Upgrade | Top TreeSHAP Risk Drivers |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `#1758` | Low | Medium (`0.24`) | **Medium** | `self_care` $\rightarrow$ **`primary_care`** | `A1, C3, C1` |
| `#1763` | Low | Medium (`0.21`) | **Medium** | `urgent_assessment` $\rightarrow$ **`urgent_assessment`** | `C2, A1` |
| `#1779` | Low | Medium (`0.19`) | **Medium** | `urgent_emergency` $\rightarrow$ **`urgent_emergency`** | `C2, A1` |
| `#1790` | Low | Medium (`0.17`) | **Medium** | `primary_care` $\rightarrow$ **`primary_care`** | `C2, C1` |
| `#1797` | Low | Medium (`0.21`) | **Medium** | `self_care` $\rightarrow$ **`primary_care`** | `C7, A1, C1` |

## 4. Subgroup Fairness & Privacy-Skipper Robustness

| Population Subgroup | Test $N$ | Impaired Prevalence | ROC AUC | Brier Score | Clinical Capture (Sensitivity) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Younger Adults (18–34 yrs)** | 429 | 14.0% | 0.734 | 0.107 | 81.7% |
| **Older Adults (35–60 yrs)** | 321 | 15.6% | 0.716 | 0.124 | 88.0% |
| **Not Yet Trying / Preconception** | 561 | 13.5% | 0.718 | 0.109 | 84.2% |
| **Actively Trying to Conceive** | 189 | 18.0% | 0.743 | 0.128 | 85.3% |
| **Complete / Low-Skip Intake** | 725 | 14.6% | 0.726 | 0.114 | 84.0% |
| **High-Privacy Skippers (>30% skipped)** | 25 | 16.0% | 0.714 | 0.129 | 100.0% |

## 5. Epidemiological Sanity Check vs. Public UCI Fertility Reference

- **Cohort Overall Impaired Semen Rate:** `17.1%` (matches WHO/EAU 15–20% reference range)
- **Prior Groin/Testicular Surgery (`B4`) Risk Ratio:** `2.29x` higher impairment rate
- **Recent High Fever in Last 3 Months (`C9`) Risk Ratio:** `1.19x` higher impairment rate
- **Tobacco / Vaping (`C3`) Risk Ratio:** `1.43x` higher impairment rate
