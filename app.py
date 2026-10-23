"""Private men's health check: Streamlit prototype (English).

Run:  streamlit run app.py
Screening and triage support only. Not a diagnosis. Uses simulated data for the
second-opinion model; nothing a user enters is written to disk.
"""
from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from triage import Band, Intake, Triage, compute_factors, evaluate

try:
    import numpy as np
    import model as model_mod
    from cohort import generate_cohort
    from model import HAS_NATIVE_XGB, RiskModel
    HAVE_MODEL = True
except Exception:  # keeps the app usable in rules-only mode
    HAVE_MODEL = False
    HAS_NATIVE_XGB = False

SEED = 20261023
HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Question wording (plain language, no assumptions about partners or family plans)
# ---------------------------------------------------------------------------
OPTS = ["No", "Yes", "Prefer not to say"]

URGENT_QS = [
    ("d1_sudden_severe_pain", "Sudden, severe pain in one or both testicles that started within the last day"),
    ("d2_sudden_swelling", "Sudden swelling, redness or warmth of the scrotum"),
    ("d3_fever_with_pain_swelling", "Fever together with testicular pain or swelling"),
    ("d4_prolonged_erection", "A painful erection lasting more than 4 hours"),
    ("d8_recent_groin_injury_swelling", "A recent injury to the groin, with swelling"),
    ("d5_lump_or_size_change", "A new lump, hardness or change in size of a testicle"),
    ("d6_blood_in_semen_or_persistent_ache", "Blood in semen, or ongoing pain or heaviness in the scrotum for more than 2 weeks"),
]
URGENT_KEYS = [k for k, _ in URGENT_QS]

HISTORY_QS = [
    ("b1_abnormal_semen_test", "Have you ever been told a semen test result was abnormal?"),
    ("b2_undescended_testis", "Was one or both testes undescended in childhood (needed treatment or surgery)?"),
    ("b3_mumps_orchitis", "Have you ever had mumps with testicular pain or swelling?"),
    ("b4_testis_groin_surgery_injury", "Have you had surgery, torsion or a serious injury involving the testes, groin or pelvis?"),
    ("b5_varicocele", "Have you been told you have a varicocele (enlarged veins in the scrotum)?"),
    ("b6_cancer_treatment", "Have you had chemotherapy, radiotherapy or other cancer treatment?"),
    ("b7_genetic_condition", "Have you been told you have a genetic or chromosomal condition that affects fertility?"),
    ("b8_untreated_infection", "Have you had a sexually transmitted infection, or a genital or urinary infection that was not fully treated?"),
]

LIFESTYLE_QS = [
    ("c2_diabetes_metabolic", "Do you have diabetes or another metabolic condition?"),
    ("c3_tobacco", "Do you smoke or use tobacco or vaping products?"),
    ("c5_testosterone_steroids", "Have you used testosterone, anabolic steroids or muscle-building supplements?"),
    ("c6_recreational_drugs", "Do you regularly use recreational drugs?"),
    ("c7_heat_exposure", "Regular heat on the groin, such as long hot baths or saunas, a laptop on your lap, or a hot work environment"),
    ("c8_work_chemicals", "Work exposure to chemicals, pesticides, heavy metals or radiation"),
    ("c9_recent_fever_illness", "A high fever or serious illness in the last 3 months"),
]
D7_Q = ("d7_sexual_function_issue", "Difficulty getting or keeping an erection, low desire, or trouble with ejaculation")

ALCOHOL = {
    "I don't drink alcohol": "none",
    "Occasionally (1 to 7 drinks a week)": "occasional",
    "Regularly (8 to 14 drinks a week)": "regular",
    "Heavily (15 or more drinks a week)": "heavy",
    "Prefer not to say": None,
}
TRYING = {
    "Yes, we are trying now": "now",
    "Planning to within the next year": "within_year",
    "Not planning to": "not_planning",
    "Prefer not to say": None,
}
MEDS = [
    "Finasteride or dutasteride",
    "Sulfasalazine",
    "Methotrexate",
    "Opioid painkillers",
    "Long-term antidepressants (SSRIs)",
    "Alpha-blockers (for prostate or blood pressure)",
]
MEDS_NONE = "None of these"
MEDS_SKIP = "Prefer not to say"

LABELS = {k: q for k, q in URGENT_QS + HISTORY_QS + LIFESTYLE_QS + [D7_Q]}

# ---------------------------------------------------------------------------
# Result copy
# ---------------------------------------------------------------------------
# Semantic card palettes. Every card sets its own background AND its own text colors, so nothing
# depends on the colour Streamlit's active theme would otherwise inherit.
# keys: bg, fg (body text), head (title), edge (left border + badge outline), badge (small label)
TONES = {
    "red": dict(bg="#B3261E", fg="#FFFFFF", head="#FFFFFF", edge="#FFD9D4", badge="Emergency"),
    "amber": dict(bg="#FBF0D9", fg="#2A1D00", head="#6B3F00", edge="#C9851A", badge="See a doctor soon"),
    "teal": dict(bg="#E2F1EE", fg="#112725", head="#0D514B", edge="#2E8F86", badge="Planned follow-up"),
    "green": dict(bg="#E4F1E7", fg="#0F2418", head="#1B5A3B", edge="#3B8F63", badge="Routine"),
}

CONTENT = {
    Triage.URGENT_EMERGENCY: dict(
        tone="red",
        headline="Get urgent medical care now",
        when="Today, without waiting",
        intro="Sudden severe pain, swelling or a long painful erection can need emergency treatment within hours. "
              "A delay can cause permanent damage.",
        steps=[
            "Go to the nearest emergency department, or call your local emergency number.",
            "Do not wait for an online booking or a call back.",
            "Tell the team exactly when the symptoms started.",
        ],
    ),
    Triage.URGENT_ASSESSMENT: dict(
        tone="amber",
        headline="See a doctor within the next few days",
        when="This week",
        intro="A new lump, blood in semen or an ache that lasts more than two weeks should be examined soon. "
              "Many causes are treatable, and checking early is the right call.",
        steps=[
            "Book an appointment with a urologist, or a primary care doctor who can refer you quickly.",
            "Expect a physical exam and often a scrotal ultrasound.",
            "If the pain or swelling becomes sudden or severe, go to an emergency department instead.",
        ],
    ),
    Triage.ANDROLOGY_UROLOGY: dict(
        tone="teal",
        headline="Book a visit with a men's health specialist",
        when="In the next few weeks",
        intro="Your answers include factors that a specialist in andrology or urology should look at. "
              "This is a screening result, not a diagnosis.",
        steps=[
            "Ask for an andrology or urology appointment.",
            "Ask about a semen analysis, which is a simple and private test.",
            "Bring any earlier test results or medicine lists you have.",
        ],
    ),
    Triage.FERTILITY_SERVICES: dict(
        tone="teal",
        headline="Start a fertility assessment",
        when="In the next few weeks",
        intro="Based on how long you have been trying, an assessment is now the standard next step. "
              "It usually includes a semen analysis, and it is not a sign that anything is wrong with you.",
        steps=[
            "Book a fertility clinic or a doctor who offers a semen analysis.",
            "Assessment normally looks at both partners, so plan it together if you can.",
            "A male assessment is quick and private.",
        ],
    ),
    Triage.PRIMARY_CARE: dict(
        tone="teal",
        headline="See your primary care doctor",
        when="In the next month or so",
        intro="Your answers point to things a routine check-up can sort out first. "
              "This is a screening result, not a diagnosis.",
        steps=[
            "Book a general check-up and mention your answers here.",
            "Ask about a metabolic review, infection screening where relevant, and a medicine review.",
            "Ask whether a semen analysis makes sense for you.",
        ],
    ),
    Triage.SELF_CARE: dict(
        tone="green",
        headline="No clinic visit needed right now",
        when="Check again in 6 to 12 months, or sooner if anything changes",
        intro="Your answers did not show signals that need a visit today. Healthy habits protect reproductive health "
              "at every stage of life, whether or not you plan to have children.",
        steps=[
            "Keep a healthy weight, stay active and avoid tobacco.",
            "Limit heat on the groin and avoid anabolic steroids.",
            "Come back and re-check if your health changes or you start trying to conceive.",
        ],
    ),
}

RED_FLAG_LIST = (
    "Get care immediately for sudden severe testicular pain, sudden swelling, fever with testicular "
    "pain, a painful erection lasting more than 4 hours, or a groin injury with swelling. "
    "These override any result on this page."
)


# ---------------------------------------------------------------------------
# State and model helpers
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("A", {})


def go(step: int) -> None:
    st.session_state.step = step
    st.rerun()


def _install_neutral_missing() -> None:
    """Make skipped answers neutral for the ML second opinion (inference only).

    The model never saw missing BMI or medicine answers in training, so XGBoost routed them
    arbitrarily and a healthy man who skipped those two questions was sent to primary care.
    The product promise is that "prefer not to say" is never risky by itself, so at inference
    a skipped yes/no answer counts as "no" and a skipped BMI as 24. Training is unchanged.
    The lasting fix belongs in cohort.py (simulate missingness on every field).
    """
    orig = model_mod.intake_to_features
    if getattr(orig, "_neutral", False):
        return
    bmi_i = model_mod.FEATURE_NAMES.index("bmi")

    def neutral(x):
        row = orig(x)
        gaps = np.isnan(row)
        row[gaps] = 0.0
        if gaps[bmi_i]:
            row[bmi_i] = 24.0
        return row

    neutral._neutral = True
    model_mod.intake_to_features = neutral


@st.cache_resource(show_spinner="Preparing the second-opinion model...")
def load_model():
    if not HAVE_MODEL:
        return None
    try:
        cohort = generate_cohort(n=2500, seed=SEED)
        fitted = RiskModel(seed=SEED).fit(cohort[:1750])
        _install_neutral_missing()  # after fitting, so training is untouched
        return fitted
    except Exception:
        return None


def tri(value):
    return True if value == "Yes" else False if value == "No" else None


def build_intake(A: dict) -> Intake:
    h, w = A.get("height_cm"), A.get("weight_kg")
    bmi = round(w / ((h / 100) ** 2), 1) if h and w and h > 0 else None

    meds = A.get("meds")
    if not meds or MEDS_SKIP in meds:
        c10 = None
    else:
        c10 = any(m in MEDS for m in meds)

    kw = {k: tri(A.get(k)) for k in LABELS}
    kw.update(
        consent=bool(A.get("consent")),
        age=A.get("age"),
        trying=TRYING.get(A.get("trying")),
        months_trying=A.get("months_trying") if TRYING.get(A.get("trying")) == "now" else None,
        partner_35_plus=tri(A.get("partner_35_plus")) if TRYING.get(A.get("trying")) == "now" else None,
        b1_abnormal_semen_test=tri(A.get("b1_abnormal_semen_test")),
        bmi=bmi,
        c4_alcohol=ALCOHOL.get(A.get("alcohol")),
        c10_listed_medicine=c10,
    )
    return Intake(**kw)


def yn(key: str, label: str, A: dict):
    cur = A.get(key)
    idx = OPTS.index(cur) if cur in OPTS else None
    return st.radio(label, OPTS, index=idx, horizontal=True, key=f"w_{key}")


# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# All colours, sizes and radii live in the :root tokens below. Components only read tokens.
# Contrast (WCAG AA, checked with a script): text on bg >= 14:1, text-2 on bg >= 7:1, card text on every
# card background >= 7:1, primary-button text on accent >= 7:1.
# "paper" cards (.note, .result-card) set their own background and every text colour explicitly,
# so they read the same in Streamlit's dark or light theme.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
:root {
  --bg: #101A18;
  --surface: #17241F;
  --surface-2: #1E2E28;
  --border: #2E423A;
  --border-strong: #6F8C80;
  --text: #EEF2EF;
  --text-2: #A9BCB4;
  --accent: #5CC2B6;
  --accent-hover: #7AD3C8;
  --accent-soft: #1F4A45;
  --on-accent: #072320;
  --focus: #F2C14E;
  --danger: #FF9C92;
  --paper: #F3EEE2;
  --paper-edge: #D8CFBA;
  --paper-text: #1F2A26;
  --paper-text-2: #44534D;
  --paper-head: #0D514B;
  --font-body: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-head: 'Newsreader', Georgia, 'Times New Roman', serif;
  --fs-body: 1.0625rem;
  --fs-small: 0.9375rem;
  --lh-body: 1.6;
  --space-1: .5rem; --space-2: 1rem; --space-3: 1.5rem; --space-4: 2.25rem;
  --radius-card: 12px;
  --radius-control: 10px;
  --control-h: 3rem;
  --card-pad: 1.25rem 1.4rem;
}
html, body, .stApp, [class*="css"] { font-family: var(--font-body); }
.stApp { background: var(--bg); color: var(--text); font-size: var(--fs-body); line-height: var(--lh-body); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }
.block-container { max-width: 760px; padding: 4rem 1.25rem 4rem; }
h1, h2, h3, h4, .serif { font-family: var(--font-head) !important; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
h1 { font-size: clamp(2.1rem, 6vw, 3rem) !important; line-height: 1.1 !important; padding: .25rem 0 .5rem !important; }
h2 { font-size: clamp(1.5rem, 4vw, 1.9rem) !important; }
h3 { font-size: 1.4rem !important; margin-top: var(--space-2); }
a { color: var(--accent); text-underline-offset: 3px; }
a:hover { color: var(--accent-hover); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--text-2); font-size: var(--fs-small); }
hr { border-color: var(--border); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; scrollbar-width: thin; }
.stTabs button[data-baseweb="tab"] { min-height: 2.75rem; padding: 0 .9rem; border-radius: 8px 8px 0 0; color: var(--text-2); }
.stTabs button[data-baseweb="tab"] p { color: inherit; font-size: 1rem; font-weight: 500; }
.stTabs button[data-baseweb="tab"]:hover { color: var(--text); background: var(--surface); }
.stTabs button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
.stTabs button[data-baseweb="tab"][aria-selected="true"] p { font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 3px; }
.stTabs [data-baseweb="tab-border"] { background: var(--border); }

/* Progress */
[data-testid="stProgress"] p { color: var(--text); font-weight: 600; font-size: 1rem; }
[data-testid="stProgressBarTrack"] { background: var(--surface-2); border-radius: 999px; }
[data-testid="stProgressBarTrack"] > div { background: var(--accent); border-radius: 999px; }

/* Labels and form controls */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color: var(--text); font-size: var(--fs-body); line-height: 1.45; }
[data-testid="stElementContainer"]:has([data-testid="stRadio"]) { width: 100% !important; }
[data-testid="stRadio"] { width: 100% !important; padding: .9rem 0 1rem; border-bottom: 1px solid var(--border); }
[data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; gap: .25rem 1.5rem; }
[data-testid="stRadio"] label { min-height: 2.75rem; align-items: center; cursor: pointer; }
[data-testid="stRadio"] label p { color: var(--text); font-size: var(--fs-body); }
[data-testid="stRadio"] label:has(input:checked) p { color: var(--accent); font-weight: 600; }
[data-testid="stCheckbox"] label { min-height: 2.75rem; align-items: center; cursor: pointer; }
[data-testid="stCheckbox"] label p { color: var(--text); font-size: var(--fs-body); }
[data-baseweb="input"], [data-baseweb="select"] > div { background: var(--surface-2) !important; border: 1px solid var(--border-strong) !important; border-radius: var(--radius-control) !important; min-height: var(--control-h); }
[data-baseweb="base-input"] { background: transparent !important; }
[data-baseweb="input"] input, [data-baseweb="select"] input, [data-baseweb="select"] [class*="singleValue"] { color: var(--text) !important; -webkit-text-fill-color: var(--text); font-size: var(--fs-body); }
[data-baseweb="input"] input::placeholder { color: var(--text-2) !important; -webkit-text-fill-color: var(--text-2); opacity: 1; }
[data-baseweb="input"][aria-invalid="true"], [data-baseweb="select"] [aria-invalid="true"] { border-color: var(--danger) !important; }
[data-baseweb="tag"] { background: var(--accent-soft) !important; border: 1px solid var(--accent); border-radius: 8px !important; }
[data-baseweb="tag"], [data-baseweb="tag"] span { color: var(--text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--surface-2) !important; }
[data-baseweb="popover"] li { color: var(--text) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--accent-soft) !important; }

/* Buttons */
.stButton > button, .stDownloadButton > button { min-height: var(--control-h); padding: 0 1.4rem; border-radius: var(--radius-control); font-family: var(--font-body); font-size: var(--fs-body); font-weight: 600; background: transparent; color: var(--text); border: 1px solid var(--border-strong); }
.stButton > button p, .stDownloadButton > button p { color: inherit; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--surface-2); border-color: var(--accent); color: var(--text); }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background: var(--accent-hover); border-color: var(--accent-hover); color: var(--on-accent); }
.stButton > button:disabled, .stDownloadButton > button:disabled { background: var(--surface) !important; color: var(--text-2) !important; border: 1px dashed var(--border-strong) !important; cursor: not-allowed; opacity: 1; }
.st-key-btn_delete button, .st-key-btn_restart button { border-color: var(--border-strong); color: var(--danger); }
.st-key-btn_delete button:hover, .st-key-btn_restart button:hover { border-color: var(--danger); background: #2A1E1D; color: var(--danger); }

/* Keyboard focus: one visible, consistent ring */
button:focus-visible, [role="tab"]:focus-visible, a:focus-visible, summary:focus-visible, [data-baseweb="select"]:focus-within > div, [data-baseweb="input"]:focus-within { outline: 3px solid var(--focus) !important; outline-offset: 2px; }
[data-testid="stRadio"] label:has(input:focus-visible), [data-testid="stCheckbox"] label:has(input:focus-visible) { outline: 3px solid var(--focus); outline-offset: 2px; border-radius: 8px; }

/* Expanders */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-card); }
[data-testid="stExpander"] summary { min-height: var(--control-h); }
[data-testid="stExpander"] summary p { color: var(--text); font-weight: 600; font-size: var(--fs-body); }
[data-testid="stExpander"] summary:hover p { color: var(--accent); }

/* Markdown content (How it works, Evidence) */
.stMarkdown table { border-collapse: collapse; width: 100%; font-size: var(--fs-small); }
.stMarkdown th, .stMarkdown td { border: 1px solid var(--border); padding: .45rem .6rem; color: var(--text); }
.stMarkdown th { background: var(--surface-2); }
.stMarkdown code { background: var(--surface-2); color: var(--text); border-radius: 6px; }
.stMarkdown li { margin-bottom: .35rem; }

/* Custom components. Every one of these sets its own text colours. */
.pm-eyebrow { color: var(--accent); font-size: .8125rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; margin: 0 0 .25rem; }
.pm-lede { color: var(--text); font-size: 1.15rem; line-height: 1.65; max-width: 62ch; margin: 0 0 var(--space-3); }
.note { background: var(--paper); color: var(--paper-text); border: 1px solid var(--paper-edge); border-radius: var(--radius-card); padding: var(--card-pad); font-size: var(--fs-body); line-height: 1.6; margin: var(--space-2) 0; }
.note p, .note li, .note span, .note div { color: var(--paper-text); }
.note b, .note strong { color: var(--paper-text); font-weight: 600; }
.note .note-title { color: var(--paper-head); font-family: var(--font-head); font-size: 1.2rem; font-weight: 600; margin: 0 0 .3rem; }
.note .note-row + .note-row { margin-top: .35rem; }
.note.note--quiet { background: var(--surface); color: var(--text); border-color: var(--border); }
.note.note--quiet p, .note.note--quiet b, .note.note--quiet strong, .note.note--quiet span, .note.note--quiet div { color: var(--text); }
.note.note--quiet .note-title { color: var(--text); }
.note.note--warn { border-left: 6px solid #C9851A; }
.note.note--ok { border-left: 6px solid #3B8F63; }
.note.note--info { border-left: 6px solid #2E8F86; }
.note.note--safety { border-left: 6px solid #A62A21; }
.small { font-size: var(--fs-small); }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-card); padding: var(--card-pad); height: 100%; }
.panel h4 { font-family: var(--font-body) !important; font-size: .8125rem; letter-spacing: .08em; text-transform: uppercase; color: var(--text-2); margin: 0 0 .6rem; }
.panel ul { margin: 0; padding-left: 1.1rem; color: var(--text); }
.panel li { margin-bottom: .45rem; color: var(--text); }
.panel p { color: var(--text-2); margin: 0; }
.pill { display: inline-block; margin-left: .35rem; padding: 0 .5rem; border: 1px solid var(--border-strong); border-radius: 999px; font-size: .8125rem; color: var(--text-2); white-space: nowrap; }
.result-card { background: var(--tint); color: var(--fg); border-left: 8px solid var(--edge); border-radius: var(--radius-card); padding: 1.4rem 1.5rem; margin: .75rem 0 1.25rem; }
.result-card .badge { display: inline-block; padding: .1rem .6rem; border: 1.5px solid var(--edge); border-radius: 999px; font-size: .8125rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--head); margin-bottom: .6rem; }
.result-card h2 { margin: 0 0 .35rem !important; padding: 0; font-size: clamp(1.6rem, 5vw, 2.1rem) !important; line-height: 1.15 !important; color: var(--head) !important; }
.result-card .when { font-weight: 600; font-size: 1.05rem; color: var(--fg); margin-bottom: .6rem; }
.result-card p { margin: 0; line-height: 1.6; color: var(--fg); }
.steps { margin: 0 0 var(--space-2); padding-left: 1.2rem; }
.steps li { margin-bottom: .55rem; color: var(--text); }
.pm-footer { margin-top: var(--space-4); padding-top: var(--space-2); border-top: 1px solid var(--border); color: var(--text-2); font-size: var(--fs-small); }
.pm-spacer { height: var(--space-2); }

/* Responsive */
@media (max-width: 640px) {
  .block-container { padding: 3.75rem 1rem 3rem; }
  .note, .panel { padding: 1rem; }
  .result-card { padding: 1.1rem 1.1rem; }
  [data-testid="stRadio"] [role="radiogroup"] { gap: .1rem 1.1rem; }
  .stButton > button, .stDownloadButton > button { width: 100%; }
}
@media (prefers-reduced-motion: no-preference) {
  .stButton > button, .stTabs button[data-baseweb="tab"], [data-baseweb="input"] { transition: background-color .15s ease, border-color .15s ease, color .15s ease; }
}
</style>
"""


def inject_css() -> None:
    # Blank lines would end the HTML block and turn the rest of the CSS into visible markdown text.
    st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)


def card(tone: str, headline: str, when: str, intro: str) -> None:
    t = TONES[tone]
    st.markdown(
        f'<div class="result-card" role="region" aria-label="Your result" '
        f'style="--tint:{t["bg"]};--fg:{t["fg"]};--head:{t["head"]};--edge:{t["edge"]}">'
        f'<div class="badge">{t["badge"]}</div>'
        f"<h2>{headline}</h2><div class='when'>{when}</div><p>{intro}</p></div>",
        unsafe_allow_html=True,
    )


def notice(kind: str, label: str, text: str) -> None:
    """Replacement for st.info / st.warning / st.success: fixed colours, plus a text label so that
    the meaning never depends on colour alone. kind: info | warn | ok | safety."""
    role = "alert" if kind in ("warn", "safety") else "status"
    st.markdown(
        f'<div class="note note--{kind}" role="{role}"><div class="note-title">{html.escape(label)}</div>'
        f"<div>{html.escape(text)}</div></div>",
        unsafe_allow_html=True,
    )


def bullet_panel(title: str, items: list[tuple[str, bool]], empty: str) -> str:
    if items:
        lis = "".join(
            f'<li>{html.escape(label)}<span class="pill">{"can be changed" if mod else "cannot be changed"}</span></li>'
            for label, mod in items
        )
        body = f"<ul>{lis}</ul>"
    else:
        body = f"<p>{html.escape(empty)}</p>"
    return f'<div class="panel"><h4>{html.escape(title)}</h4>{body}</div>'


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
STEP_TITLES = {1: "Urgent symptoms", 2: "About you", 3: "Your history", 4: "Health and lifestyle"}


def header(step: int) -> None:
    st.progress(step / 4, text=f"Step {step} of 4: {STEP_TITLES[step]}")
    st.caption("Every question can be skipped. Choose \"Prefer not to say\" or leave it blank.")


def step_welcome(A: dict) -> None:
    st.markdown('<div class="pm-eyebrow">Private screening &middot; 4 short steps</div>', unsafe_allow_html=True)
    st.title("A private health check for men")
    st.markdown(
        '<p class="pm-lede">Answer a few questions in a few minutes and find out whether a check-up, a specialist '
        "visit or no visit at all makes sense for you. This works for any stage of life, with or without plans "
        "to have children.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='note'>"
        "<div class='note-row'><b>What this is:</b> a screening tool that suggests a next step.</div>"
        "<div class='note-row'><b>What it is not:</b> a diagnosis, a fertility prediction or a replacement "
        "for seeing a doctor.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='note note--quiet'><div class='note-title'>Your privacy</div>"
        "<div>Your answers stay in this browser session. Nothing is saved to a database, and closing the page "
        "or pressing \"Delete my answers\" clears everything. You choose what, if anything, to share with a clinic."
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Before you start")
    with st.container(border=True):
        age = st.number_input("Your age", min_value=18, max_value=90, value=A.get("age"), step=1,
                              placeholder="Enter your age (18 or over)")
        c1 = st.checkbox("I am 18 or older.", value=bool(A.get("c_adult")))
        c2 = st.checkbox("I understand this is screening support, not a diagnosis.", value=bool(A.get("c_screen")))
        c3 = st.checkbox("I agree to my answers being used in this session to produce a result.",
                         value=bool(A.get("c_use")))
        ready = bool(age) and c1 and c2 and c3
        if st.button("Start the check", type="primary", disabled=not ready):
            A.update(age=int(age), c_adult=c1, c_screen=c2, c_use=c3, consent=True)
            go(1)
        if not ready:
            st.caption("To start, enter your age (18 or over) and tick all three boxes.")


def step_urgent(A: dict) -> None:
    header(1)
    st.subheader("Do you have any of these right now?")
    vals = {k: yn(k, q, A) for k, q in URGENT_QS}
    with st.container():
        c_back, c_next = st.columns([1, 3])
    if c_back.button("Back"):
        go(0)
    if c_next.button("Continue", type="primary"):
        A.update({k: v for k, v in vals.items() if v})
        if any(A.get(k) == "Yes" for k in URGENT_KEYS):
            go(5)
        go(2)


def step_about(A: dict) -> None:
    header(2)
    st.subheader("About you")
    trying_opts = list(TRYING)
    cur = A.get("trying")
    trying = st.radio("Are you, or is anyone you are with, trying to conceive?", trying_opts,
                      index=trying_opts.index(cur) if cur in trying_opts else None, key="w_trying")
    months = st.number_input("If trying now: how many months so far?", min_value=0, max_value=240,
                             value=A.get("months_trying"), step=1, placeholder="Months")
    partner = yn("partner_35_plus", "If trying now: is your partner aged 35 or over?", A)
    c1, c2 = st.columns(2)
    height = c1.number_input("Height (cm)", min_value=120, max_value=230, value=A.get("height_cm"),
                             step=1, placeholder="cm")
    weight = c2.number_input("Weight (kg)", min_value=35, max_value=250, value=A.get("weight_kg"),
                             step=1, placeholder="kg")
    st.caption("Height and weight are used to work out BMI and are not stored.")
    c_back, c_next = st.columns([1, 3])
    if c_back.button("Back"):
        go(1)
    if c_next.button("Continue", type="primary"):
        A.update(trying=trying, months_trying=months, partner_35_plus=partner,
                 height_cm=height, weight_kg=weight)
        go(3)


def step_history(A: dict) -> None:
    header(3)
    st.subheader("Your medical history")
    vals = {k: yn(k, q, A) for k, q in HISTORY_QS}
    c_back, c_next = st.columns([1, 3])
    if c_back.button("Back"):
        go(2)
    if c_next.button("Continue", type="primary"):
        A.update({k: v for k, v in vals.items() if v})
        go(4)


def step_lifestyle(A: dict) -> None:
    header(4)
    st.subheader("Health and lifestyle")
    vals = {k: yn(k, q, A) for k, q in LIFESTYLE_QS}
    alc_opts = list(ALCOHOL)
    cur = A.get("alcohol")
    alcohol = st.radio("How much alcohol do you drink?", alc_opts,
                       index=alc_opts.index(cur) if cur in alc_opts else None, key="w_alcohol")
    meds = st.multiselect("Do you regularly take any of these medicines?", MEDS + [MEDS_NONE, MEDS_SKIP],
                          default=A.get("meds", []), key="w_meds",
                          help="Please keep taking any prescribed medicine unless your doctor tells you to stop.")
    st.subheader("Sexual health (optional)")
    d7 = yn(D7_Q[0], D7_Q[1], A)
    c_back, c_next = st.columns([1, 3])
    if c_back.button("Back"):
        go(3)
    if c_next.button("See my result", type="primary"):
        A.update({k: v for k, v in vals.items() if v})
        A.update(alcohol=alcohol, meds=meds)
        if d7:
            A[D7_Q[0]] = d7
        go(5)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
def summary_text(intake: Intake, triage: Triage, band, rr, A: dict, sections: list[str]) -> str:
    lines = ["Private health check: summary for a clinician",
             "Screening support only. This is not a diagnosis.", ""]
    if "Result and next step" in sections:
        c = CONTENT[triage]
        lines += [f"Suggested next step: {c['headline']} ({c['when']})",
                  f"Risk band: {band.value if band else 'n/a'}", ""]
    if "Factors I reported" in sections:
        lines.append("Factors reported:")
        lines += [f"- {f.label}" for f in compute_factors(intake)] or ["- None reported"]
        lines.append("")
    if "Symptoms I reported" in sections:
        yes = [LABELS[k] for k in URGENT_KEYS if A.get(k) == "Yes"]
        lines.append("Symptoms reported:")
        lines += [f"- {s}" for s in yes] or ["- None reported"]
    return "\n".join(lines)


def step_result(A: dict) -> None:
    intake = build_intake(A)
    model = load_model()
    if model is not None:
        h = model.evaluate_hybrid(intake, log_disagreement=False)
        triage, rule_id, band = h.final_triage, h.final_rule_id, h.final_band
        rr, shap = h.rule_result, h.top_shap_factors
    else:
        h, shap = None, []
        rr = evaluate(intake)
        triage, rule_id, band = rr.triage, rr.rule, rr.band

    if triage == Triage.STOP:
        st.title("We can't continue yet")
        st.write("An adult age and your consent are needed to continue.")
        if st.button("Start again", key="btn_restart"):
            st.session_state.clear()
            st.rerun()
        return

    c = CONTENT[triage]
    emergency = triage == Triage.URGENT_EMERGENCY

    # 1. Result and recommended next step
    st.title("Your result")
    card(c["tone"], c["headline"], c["when"], c["intro"])

    # 2. Actions to take
    st.subheader("What to do")
    st.markdown(
        '<ol class="steps">' + "".join(f"<li>{html.escape(s)}</li>" for s in c["steps"]) + "</ol>"
        if emergency else
        '<ul class="steps">' + "".join(f"<li>{html.escape(s)}</li>" for s in c["steps"]) + "</ul>",
        unsafe_allow_html=True,
    )

    # Notices that qualify the result sit directly under the actions so they are not missed.
    if rr.low_confidence:
        notice("info", "Limited answers",
               "This result is based on limited answers, so we leaned cautious. Answering more questions can sharpen it.")
    if rr.safety_net and not emergency:
        notice("warn", "Some symptom questions were skipped", "You skipped some symptom questions. " + RED_FLAG_LIST)
    if intake.c10_listed_medicine:
        notice("info", "About your medicines", "Do not stop any prescribed medicine without asking your doctor.")
    if h is not None and h.disagreed and h.disagreement_direction == "ml_higher":
        st.caption("The second-opinion model rated your risk a level higher than the rules, so we used the more cautious level.")

    # 3. Factors that influenced the result
    if not emergency:
        st.subheader("What shaped this result")
        a, b = st.columns(2)
        a.markdown(bullet_panel("From your answers",
                                [(f.label, f.modifiable) for f in (rr.top_factors or [])],
                                "No single risk factor stood out."), unsafe_allow_html=True)
        b.markdown(bullet_panel("Second-opinion model",
                                [(s.label, s.modifiable) for s in (shap or [])],
                                "No extra drivers found." if h is not None else "Not available in this mode."),
                   unsafe_allow_html=True)

    # 4. Follow-up options
    if not emergency:
        st.subheader("Next steps you control")
        with st.expander("Get a summary to show a clinician"):
            secs = st.multiselect("Include", ["Result and next step", "Factors I reported", "Symptoms I reported"],
                                  default=["Result and next step"], key="w_sections")
            st.download_button("Download summary (text)", summary_text(intake, triage, band, rr, A, secs),
                               file_name="health_check_summary.txt")
        if triage != Triage.SELF_CARE:
            with st.expander("Request a private teleconsultation (demo)"):
                st.selectbox("Clinician preference", ["No preference", "Male clinician", "Female clinician"])
                st.selectbox("Preferred time", ["Morning", "Afternoon", "Evening"])
                st.checkbox("Share my summary with the clinician")
                if st.button("Request consultation"):
                    notice("ok", "Demo only", "Demo only: no request was sent and no data left this page.")

    # 5. Safety information (always visible, full size)
    st.markdown("<div class='pm-spacer'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='note note--safety' role='note'><div class='note-title'>Safety information</div>"
        f"<div>{html.escape(RED_FLAG_LIST)}</div></div>",
        unsafe_allow_html=True,
    )

    # 6. Technical details
    with st.expander("Technical details"):
        st.write({
            "rule fired": rule_id,
            "rule score": rr.score,
            "rule band": rr.band.value if rr.band else None,
            "final band": band.value if band else None,
            "second-opinion band": h.ml_band.value if h else "n/a",
            "model backend": ("xgboost" if HAS_NATIVE_XGB else "numpy boosted trees") if h else "rules only",
        })
        st.caption("The model is trained on a simulated population, so its probabilities are not shown to users.")

    st.markdown("<div class='pm-spacer'></div>", unsafe_allow_html=True)
    if st.button("Delete my answers and start again", key="btn_delete"):
        st.session_state.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Other tabs
# ---------------------------------------------------------------------------
def tab_how() -> None:
    st.header("How it works")
    st.markdown(
        """
1. **Safety first.** Urgent symptoms are checked before anything else and always override the rest.
2. **Transparent rules.** A written scoring protocol weighs history, health and lifestyle factors and picks one of five next steps.
3. **Second opinion.** A small machine-learning model, trained on simulated data, rates the risk too. If it rates higher than the rules, the more cautious level is used. It can never lower a result or overrule an urgent, specialist or fertility-timing rule.
4. **You decide.** Every question can be skipped, and sharing with a clinic is optional.
"""
    )
    st.header("Privacy")
    st.markdown(
        """
- Answers live only in this browser session and are not written to disk or a database.
- Only the questions that change the result are asked.
- Height and weight are used for BMI and then discarded.
- The summary file contains only the sections you tick.
"""
    )
    st.header("Limits")
    st.markdown(
        """
- This is screening support, not a diagnosis, and it is not a clinician.
- The prototype's thresholds are design defaults awaiting clinical review.
- The second-opinion model is trained on a simulated population, not real patients.
"""
    )


def tab_evidence() -> None:
    st.header("Evidence")
    st.markdown(
        "<div class='note'><b>How to read these numbers:</b> the cohort is simulated from published estimates, "
        "and the 60 vignettes were written against the same written protocol. Vignette results therefore show "
        "<i>agreement with the protocol</i>, not clinical accuracy, and the simulated outcome is not real patient data. "
        "Sensitivity and fairness figures show how the pipeline behaves, not real-world equity.</div>",
        unsafe_allow_html=True,
    )
    rep = HERE / "EVALUATION_REPORT.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.write("Run `python evaluate_report.py` to generate EVALUATION_REPORT.md.")
    app = HERE / "vignettes_appendix.csv"
    if app.exists():
        import pandas as pd
        st.subheader("Vignette appendix")
        st.dataframe(pd.read_csv(app), hide_index=True)


# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Health Check", page_icon=None, layout="centered")
    inject_css()
    init_state()
    t1, t2, t3 = st.tabs(["Check", "How it works", "Evidence"])
    A = st.session_state.A
    with t1:
        step = st.session_state.step
        {0: step_welcome, 1: step_urgent, 2: step_about, 3: step_history,
         4: step_lifestyle, 5: step_result}[step](A)
    with t2:
        tab_how()
    with t3:
        tab_evidence()
    st.markdown(
        "<div class='pm-footer'>Prototype for the Future Health Digital Innovation Challenge. "
        "Simulated data. Not medical advice.</div>", unsafe_allow_html=True,
    )


main()
