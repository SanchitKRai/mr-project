"""Second-opinion ML layer (XGBoost + exact TreeSHAP) and cautious arbitrator.

Zero-dependency on `scikit-learn` (avoids Windows Application Control DLL blocks
on `_radius_neighbors.pyd`). Uses native `xgboost.train` (`xgb.Booster`) when
permitted by OS policy, and includes a pure-NumPy gradient boosted ensemble
fallback if `xgboost.dll` is also blocked by Windows Smart App Control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import xgboost as xgb
    # Smoke-test native DMatrix creation to ensure xgboost.dll isn't blocked by OS
    _test_dm = xgb.DMatrix(np.zeros((2, 2), dtype=np.float32))
    HAS_NATIVE_XGB = True
except Exception:
    xgb = None
    HAS_NATIVE_XGB = False

from cohort import SyntheticRecord, generate_cohort
from triage import Band, Intake, Result, Triage, evaluate


# ---------------------------------------------------------------------------
# Pure-NumPy metrics (replaces sklearn.metrics to avoid WDAC DLL block)
# ---------------------------------------------------------------------------
def roc_auc_score_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC AUC via Mann-Whitney U statistic using pure NumPy."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Pairwise comparison: count(pos > neg) + 0.5 * count(pos == neg)
    diff = pos[:, None] - neg[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(wins / (len(pos) * len(neg)))


def brier_score_loss_np(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_true - y_prob) ** 2))


# ---------------------------------------------------------------------------
# Feature specification (19 upstream predictors)
# ---------------------------------------------------------------------------
FEATURE_SPECS: List[Tuple[str, str, str, bool]] = [
    ("age", "A1", "Age", False),
    ("b2_undescended_testis", "B2", "Undescended testis in childhood", False),
    ("b3_mumps_orchitis", "B3", "Mumps with testicular involvement", False),
    ("b4_testis_groin_surgery_injury", "B4", "Testis, groin or pelvic surgery/injury", False),
    ("b5_varicocele", "B5", "Varicocele", False),
    ("b6_cancer_treatment", "B6", "Chemotherapy, radiotherapy or cancer treatment", False),
    ("b7_genetic_condition", "B7", "Genetic or chromosomal condition", False),
    ("b8_untreated_infection", "B8", "Untreated or incompletely treated infection", False),
    ("bmi", "C1", "Body Mass Index (BMI)", True),
    ("c2_diabetes_metabolic", "C2", "Diabetes or metabolic condition", False),
    ("c3_tobacco", "C3", "Tobacco or vaping", True),
    ("c4_alcohol_heavy", "C4", "Heavy alcohol use (15+ drinks/week)", True),
    ("c5_testosterone_steroids", "C5", "Testosterone or anabolic steroid use", True),
    ("c6_recreational_drugs", "C6", "Recreational drug use", True),
    ("c7_heat_exposure", "C7", "Regular heat exposure to the groin", True),
    ("c8_work_chemicals", "C8", "Work exposure to chemicals or radiation", True),
    ("c9_recent_fever_illness", "C9", "High fever or serious illness in last 3 months", False),
    ("c10_listed_medicine", "C10", "Fertility-relevant prescription medicine", False),
    ("d7_sexual_function_issue", "D7", "Sexual function concern", False),
]

FEATURE_NAMES: List[str] = [f[0] for f in FEATURE_SPECS]
BAND_ORDER: Dict[Band, int] = {Band.LOW: 0, Band.MEDIUM: 1, Band.HIGH: 2}
INV_BAND_ORDER: Dict[int, Band] = {0: Band.LOW, 1: Band.MEDIUM, 2: Band.HIGH}

DEFAULT_MEDIUM_PROB_CUTOFF = 0.16
DEFAULT_HIGH_PROB_CUTOFF = 0.30


@dataclass
class ShapContribution:
    feature_name: str
    code_id: str
    label: str
    shap_value: float      # log-odds contribution
    feature_value: float   # encoded input value (or NaN)
    modifiable: bool


@dataclass
class DisagreementEvent:
    person_id: Optional[int]
    rule_band: Band
    ml_band: Band
    final_band: Band
    rule_triage: Triage
    final_triage: Triage
    ml_prob: float
    direction: str         # "ml_higher" | "rules_higher"
    band_gap: int
    top_shap_ids: List[str]


@dataclass
class HybridResult:
    rule_result: Result
    ml_prob: float
    ml_band: Band
    final_band: Optional[Band]
    final_triage: Triage
    final_rule_id: str
    disagreed: bool
    disagreement_direction: Optional[str] = None
    top_shap_factors: List[ShapContribution] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def intake_to_features(x: Intake) -> np.ndarray:
    """Convert a single Intake into a 1D float32 array with np.nan for None."""
    def b2f(val: Optional[bool]) -> float:
        if val is None:
            return np.nan
        return 1.0 if val else 0.0

    if x.c4_alcohol is None:
        alc_heavy = np.nan
    else:
        alc_heavy = 1.0 if x.c4_alcohol == "heavy" else 0.0

    row = [
        float(x.age) if x.age is not None else np.nan,
        b2f(x.b2_undescended_testis),
        b2f(x.b3_mumps_orchitis),
        b2f(x.b4_testis_groin_surgery_injury),
        b2f(x.b5_varicocele),
        b2f(x.b6_cancer_treatment),
        b2f(x.b7_genetic_condition),
        b2f(x.b8_untreated_infection),
        float(x.bmi) if x.bmi is not None else np.nan,
        b2f(x.c2_diabetes_metabolic),
        b2f(x.c3_tobacco),
        alc_heavy,
        b2f(x.c5_testosterone_steroids),
        b2f(x.c6_recreational_drugs),
        b2f(x.c7_heat_exposure),
        b2f(x.c8_work_chemicals),
        b2f(x.c9_recent_fever_illness),
        b2f(x.c10_listed_medicine),
        b2f(x.d7_sexual_function_issue),
    ]
    return np.asarray(row, dtype=np.float32)


def records_to_xy(records: List[SyntheticRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from synthetic cohort records."""
    X = np.vstack([intake_to_features(r.intake) for r in records])
    y = np.asarray([1 if r.latent_impaired_semen else 0 for r in records], dtype=np.int32)
    return X, y


def prob_to_band(
    prob: float,
    medium_cutoff: float = DEFAULT_MEDIUM_PROB_CUTOFF,
    high_cutoff: float = DEFAULT_HIGH_PROB_CUTOFF,
) -> Band:
    if prob >= high_cutoff:
        return Band.HIGH
    if prob >= medium_cutoff:
        return Band.MEDIUM
    return Band.LOW


class _NumpyBoostedTrees:
    """Pure-NumPy Gradient Boosted Decision Stumps with exact additive TreeSHAP.

    Used automatically if Windows Application Control blocks xgboost/sklearn DLLs.
    Because each boosting round is a single-feature split (with learned default
    routing for NaN), the exact Shapley value for feature j is simply the sum of
    log-odds leaf outputs across all trees that split on feature j, centered by
    their expected value on the training data.
    """

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.08) -> None:
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.base_log_odds: float = 0.0
        # Each tree: (feat_idx, threshold, nan_go_left, left_val, right_val, mean_val)
        self.trees: List[Tuple[int, float, bool, float, float, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyBoostedTrees":
        y_f = y.astype(np.float64)
        p_mean = float(np.clip(np.mean(y_f), 1e-4, 1.0 - 1e-4))
        self.base_log_odds = float(np.log(p_mean / (1.0 - p_mean)))
        f_pred = np.full(len(y_f), self.base_log_odds, dtype=np.float64)
        self.trees = []

        # Precompute candidate thresholds per feature
        n_samples, n_features = X.shape
        thresholds: List[np.ndarray] = []
        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            uniq = np.unique(valid)
            if len(uniq) <= 2:
                thresholds.append(np.array([0.5], dtype=np.float64))
            else:
                qs = np.quantile(valid, [0.2, 0.4, 0.5, 0.6, 0.75, 0.85, 0.92])
                thresholds.append(np.unique(qs))

        for _ in range(self.n_estimators):
            prob = 1.0 / (1.0 + np.exp(-f_pred))
            grad = y_f - prob                  # negative gradient of log-loss
            hess = prob * (1.0 - prob) + 1e-6  # second derivative

            best_gain = -1e18
            best_tree = None

            for j in range(n_features):
                col = X[:, j]
                is_nan = np.isnan(col)
                for thr in thresholds[j]:
                    for nan_left in (True, False):
                        left_mask = (col <= thr) | is_nan if nan_left else (col <= thr) & (~is_nan)
                        right_mask = ~left_mask
                        if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                            continue
                        g_l, h_l = float(np.sum(grad[left_mask])), float(np.sum(hess[left_mask]))
                        g_r, h_r = float(np.sum(grad[right_mask])), float(np.sum(hess[right_mask]))
                        gain = (g_l * g_l) / (h_l + 2.0) + (g_r * g_r) / (h_r + 2.0)
                        if gain > best_gain:
                            best_gain = gain
                            l_val = self.lr * (g_l / (h_l + 2.0))
                            r_val = self.lr * (g_r / (h_r + 2.0))
                            mean_v = (l_val * np.sum(left_mask) + r_val * np.sum(right_mask)) / n_samples
                            best_tree = (j, float(thr), nan_left, float(l_val), float(r_val), float(mean_v))

            if best_tree is None:
                break
            j, thr, nan_left, l_val, r_val, _ = best_tree
            col = X[:, j]
            is_nan = np.isnan(col)
            left_mask = (col <= thr) | is_nan if nan_left else (col <= thr) & (~is_nan)
            f_pred += np.where(left_mask, l_val, r_val)
            self.trees.append(best_tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        f_pred = np.full(X.shape[0], self.base_log_odds, dtype=np.float64)
        for j, thr, nan_left, l_val, r_val, _ in self.trees:
            col = X[:, j]
            is_nan = np.isnan(col)
            left_mask = (col <= thr) | is_nan if nan_left else (col <= thr) & (~is_nan)
            f_pred += np.where(left_mask, l_val, r_val)
        p1 = 1.0 / (1.0 + np.exp(-f_pred))
        return np.column_stack([1.0 - p1, p1])

    def pred_contribs_one(self, x_row: np.ndarray) -> np.ndarray:
        """Exact additive feature attribution (TreeSHAP equivalent for depth-1 trees)."""
        contribs = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        for j, thr, nan_left, l_val, r_val, mean_v in self.trees:
            val = x_row[j]
            if np.isnan(val):
                go_left = nan_left
            else:
                go_left = val <= thr
            leaf = l_val if go_left else r_val
            contribs[j] += (leaf - mean_v)
        return contribs


class RiskModel:
    """XGBoost + exact TreeSHAP second-opinion model and cautious arbitrator."""

    def __init__(
        self,
        medium_cutoff: float = DEFAULT_MEDIUM_PROB_CUTOFF,
        high_cutoff: float = DEFAULT_HIGH_PROB_CUTOFF,
        seed: int = 20261023,
    ) -> None:
        self.medium_cutoff = medium_cutoff
        self.high_cutoff = high_cutoff
        self.seed = seed
        self._booster = None
        self._np_fallback: Optional[_NumpyBoostedTrees] = None
        self._is_fitted = False
        self.disagreement_log: List[DisagreementEvent] = []

    def fit(self, records: List[SyntheticRecord]) -> "RiskModel":
        X, y = records_to_xy(records)
        if HAS_NATIVE_XGB:
            dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
            params = {
                "max_depth": 1,
                "eta": 0.08,
                "subsample": 0.90,
                "colsample_bytree": 0.50,
                "min_child_weight": 0.5,
                "alpha": 0.0,
                "lambda": 1.0,
                "monotone_constraints": "(" + ",".join(["1"] * len(FEATURE_NAMES)) + ")",
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": self.seed,
                "nthread": 1,
            }
            self._booster = xgb.train(params, dtrain, num_boost_round=160)
        else:
            self._np_fallback = _NumpyBoostedTrees(n_estimators=120, learning_rate=0.08).fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("RiskModel must be fitted before calling predict_proba_batch.")
        if HAS_NATIVE_XGB and self._booster is not None:
            dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            return np.asarray(self._booster.predict(dmat), dtype=np.float64)
        assert self._np_fallback is not None
        return self._np_fallback.predict_proba(X)[:, 1]

    def predict_proba_one(self, x: Intake) -> float:
        feats = intake_to_features(x).reshape(1, -1)
        return float(self.predict_proba_batch(feats)[0])

    def explain_one(self, x: Intake, top_k: int = 3) -> List[ShapContribution]:
        """Compute exact TreeSHAP values for a single Intake."""
        if not self._is_fitted:
            raise RuntimeError("RiskModel must be fitted before calling explain_one.")
        feats = intake_to_features(x).reshape(1, -1)
        if HAS_NATIVE_XGB and self._booster is not None:
            dmat = xgb.DMatrix(feats, feature_names=FEATURE_NAMES)
            contribs = self._booster.predict(dmat, pred_contribs=True)[0][:-1]
        else:
            assert self._np_fallback is not None
            contribs = self._np_fallback.pred_contribs_one(feats[0])

        items: List[ShapContribution] = []
        for idx, (fname, cid, label, mod) in enumerate(FEATURE_SPECS):
            s_val = float(contribs[idx])
            f_val = float(feats[0, idx])
            if s_val > 0.02 and not np.isnan(f_val) and f_val > 0:
                if fname == "age" and f_val <= 35:
                    continue
                if fname == "bmi" and f_val < 25.0:
                    continue
                items.append(
                    ShapContribution(
                        feature_name=fname,
                        code_id=cid,
                        label=label,
                        shap_value=round(s_val, 4),
                        feature_value=f_val,
                        modifiable=mod,
                    )
                )

        items.sort(key=lambda c: -c.shap_value)
        return items[:top_k]

    def evaluate_hybrid(
        self,
        x: Intake,
        person_id: Optional[int] = None,
        log_disagreement: bool = True,
    ) -> HybridResult:
        """Run both the deterministic rule engine and ML second opinion."""
        rule_res = evaluate(x)

        if rule_res.triage == Triage.STOP or rule_res.band is None:
            return HybridResult(
                rule_result=rule_res,
                ml_prob=0.0,
                ml_band=Band.LOW,
                final_band=None,
                final_triage=Triage.STOP,
                final_rule_id="R0",
                disagreed=False,
                notes=list(rule_res.notes),
            )

        ml_prob = self.predict_proba_one(x)
        ml_band = prob_to_band(ml_prob, self.medium_cutoff, self.high_cutoff)
        top_shap = self.explain_one(x, top_k=3)

        r_ord = BAND_ORDER[rule_res.band]
        m_ord = BAND_ORDER[ml_band]
        gap = abs(m_ord - r_ord)
        disagreed = gap >= 1

        final_band = INV_BAND_ORDER[max(r_ord, m_ord)]
        final_triage = rule_res.triage
        final_rule_id = rule_res.rule
        notes = list(rule_res.notes)
        direction: Optional[str] = None

        if disagreed:
            direction = "ml_higher" if m_ord > r_ord else "rules_higher"
            notes.append(
                f"Second-opinion arbitration: rules={rule_res.band.value}, "
                f"ML={ml_band.value} (p={ml_prob:.2f}) -> using {final_band.value}."
            )

            if direction == "ml_higher" and rule_res.rule in ("R5", "R6", "R7"):
                if final_band == Band.HIGH and rule_res.rule in ("R6", "R7"):
                    final_triage = Triage.ANDROLOGY_UROLOGY
                    final_rule_id = "R5_ML"
                elif final_band == Band.MEDIUM and rule_res.rule == "R7":
                    final_triage = Triage.PRIMARY_CARE
                    final_rule_id = "R6_ML"

            if log_disagreement:
                self.disagreement_log.append(
                    DisagreementEvent(
                        person_id=person_id,
                        rule_band=rule_res.band,
                        ml_band=ml_band,
                        final_band=final_band,
                        rule_triage=rule_res.triage,
                        final_triage=final_triage,
                        ml_prob=round(ml_prob, 4),
                        direction=direction,
                        band_gap=gap,
                        top_shap_ids=[s.code_id for s in top_shap],
                    )
                )

        return HybridResult(
            rule_result=rule_res,
            ml_prob=round(ml_prob, 4),
            ml_band=ml_band,
            final_band=final_band,
            final_triage=final_triage,
            final_rule_id=final_rule_id,
            disagreed=disagreed,
            disagreement_direction=direction,
            top_shap_factors=top_shap,
            notes=notes,
        )


def evaluate_cohort_performance(
    train_records: List[SyntheticRecord],
    test_records: List[SyntheticRecord],
    seed: int = 20261023,
) -> Dict[str, object]:
    """Train on `train_records`, evaluate on held-out `test_records`."""
    model = RiskModel(seed=seed).fit(train_records)
    X_test, y_test = records_to_xy(test_records)
    probs = model.predict_proba_batch(X_test)

    upstream_rule_scores = np.asarray(
        [r.rule_score - (3 if r.intake.b1_abnormal_semen_test is True else 0) for r in test_records],
        dtype=np.float64,
    )

    ml_auc = roc_auc_score_np(y_test, probs)
    rule_auc = roc_auc_score_np(y_test, upstream_rule_scores)
    brier = brier_score_loss_np(y_test, probs)

    hybrid_results = [
        model.evaluate_hybrid(r.intake, person_id=r.person_id, log_disagreement=True)
        for r in test_records
    ]

    impaired_indices = [i for i, r in enumerate(test_records) if r.latent_impaired_semen]
    rules_caught = sum(
        test_records[i].rule_triage != Triage.SELF_CARE for i in impaired_indices
    )
    hybrid_caught = sum(
        hybrid_results[i].final_triage != Triage.SELF_CARE for i in impaired_indices
    )

    young_mask = np.asarray([(r.intake.age or 30) < 35 for r in test_records])
    older_mask = ~young_mask
    auc_young = roc_auc_score_np(y_test[young_mask], probs[young_mask])
    auc_older = roc_auc_score_np(y_test[older_mask], probs[older_mask])

    ml_higher_count = sum(d.direction == "ml_higher" for d in model.disagreement_log)
    rules_higher_count = sum(d.direction == "rules_higher" for d in model.disagreement_log)

    return {
        "backend": "xgboost_native" if HAS_NATIVE_XGB else "numpy_boosted_trees",
        "n_train": len(train_records),
        "n_test": len(test_records),
        "ml_roc_auc": round(ml_auc, 4),
        "rule_score_roc_auc": round(rule_auc, 4),
        "ml_brier_score": round(brier, 4),
        "impaired_sensitivity_rules_only": round(rules_caught / max(1, len(impaired_indices)), 4),
        "impaired_sensitivity_hybrid": round(hybrid_caught / max(1, len(impaired_indices)), 4),
        "disagreement_rate": round(len(model.disagreement_log) / len(test_records), 4),
        "disagreements_ml_higher": ml_higher_count,
        "disagreements_rules_higher": rules_higher_count,
        "subgroup_auc_under_35": round(auc_young, 4),
        "subgroup_auc_35_plus": round(auc_older, 4),
        "subgroup_auc_gap": round(abs(auc_young - auc_older), 4),
    }


if __name__ == "__main__":
    cohort = generate_cohort(n=2500, seed=20261023)
    train_split, test_split = cohort[:1750], cohort[1750:]
    metrics = evaluate_cohort_performance(train_split, test_split)
    for k, v in metrics.items():
        print(f"{k}: {v}")