"""
Prediction engine for the Hypertension Risk Screening System.

Everything here follows Appendix G.3 (03_streamlit_app.py) of the project report:

* The SCORE comes from bundle['pipeline'] - the Platt-calibrated SVM
  (CalibratedClassifierCV, sigmoid, 5 folds).
* The EXPLANATION comes from bundle['explain_pipeline'] - the same SVM pipeline
  refitted on all development data, without the calibrator - using
  shap.KernelExplainer over bundle['background'] with nsamples=200.
* Predictors the form does not ask for are passed as NaN and filled by the
  imputer already fitted inside the pipeline ("Nothing is invented", G.3 l.163).
* Categories: Higher if score >= threshold; Moderate if score >= 0.6 x threshold;
  otherwise Lower (G.3 l.180-185). The threshold is the frozen value stored in
  the bundle (chosen on development data for >= 85% recall, section 3.8.3).

No input is written to disk, logged or transmitted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT / "model" / "model_bundle.joblib"

# Outcome-defining and outcome-adjacent columns. The pipeline excluded these by
# construction (run_summary.json -> leakage_columns_excluded). The app checks
# at load time that none of them is in the model's input schema.
LEAKAGE_COLUMNS = frozenset({
    "sbp_1", "dbp_1", "sbp_2", "dbp_2", "sbp_3", "dbp_3", "sbp_4", "dbp_4",
    "mean_sbp", "mean_dbp", "on_bp_medication", "prior_htn_diagnosis",
    "prior_bp_check", "elevated_bp",
})

# Variables the user is asked for. Waist and hip are collected only to derive
# WHR (the model uses WHR, not waist or hip on their own).
FORM_FIELDS = ("age", "waist_cm", "hip_cm", "cigarettes_per_day", "alcohol_frequency")

ALCOHOL_OPTIONS = [
    "Never",
    "Less than once a month",
    "1–3 times a month",
    "1–2 times a week",
    "3–4 times a week",
    "Daily or almost daily",
]

# Validation limits. Age follows the study eligibility (18-40). The measurement
# ranges are the plausibility limits used by the Appendix G.3 form.
LIMITS = {
    "age": (18, 40),
    "waist_cm": (50.0, 150.0),
    "hip_cm": (60.0, 180.0),
    "cigarettes_per_day": (0, 60),
}
WHR_PLAUSIBLE = (0.55, 1.40)

MODERATE_FACTOR = 0.6  # Appendix G.3 line 182

FRIENDLY = {
    "age": "Age",
    "whr": "Waist-to-hip ratio",
    "cigarettes_per_day": "Cigarettes smoked per day",
    "alcohol_frequency": "Drinks alcohol daily or almost daily",
}
MODIFIABLE = {"whr", "cigarettes_per_day", "alcohol_frequency"}


class ValidationError(ValueError):
    """Raised with a list of user-facing messages."""

    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


@dataclass
class Assessment:
    inputs: dict
    whr: float
    score: float
    threshold: float
    moderate_cutoff: float
    category: str
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap_base_value: float | None = None
    shap_model_output: float | None = None
    shap_error: str | None = None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict:
    p = Path(path) if path else BUNDLE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found at {p}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(p)
    check_bundle(bundle)
    return bundle


def check_bundle(bundle: dict) -> None:
    """Fail loudly if the bundle is not what the app was built for."""
    required = {"pipeline", "explain_pipeline", "threshold", "numeric",
                "categorical", "selected_variables", "background", "is_tree"}
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"Model bundle is missing keys: {sorted(missing)}")
    schema = set(bundle["numeric"]) | set(bundle["categorical"])
    leaked = schema & LEAKAGE_COLUMNS
    if leaked:
        raise RuntimeError(f"Leakage columns found in model schema: {sorted(leaked)}")
    if set(bundle["selected_variables"]) != {"age", "whr", "cigarettes_per_day", "alcohol_frequency"}:
        raise RuntimeError("Unexpected selected_variables in bundle; app form would not match model.")


def input_schema(bundle: dict) -> list[str]:
    return list(bundle["numeric"]) + list(bundle["categorical"])


def selected_encoded_features(bundle: dict) -> list[str]:
    steps = bundle["explain_pipeline"].named_steps
    names = np.array(steps["prep"].get_feature_names_out())
    return list(names[steps["select"].get_support()])


def fold_feature_sets(bundle: dict) -> list[list[str]]:
    """Encoded features kept by SelectKBest inside each of the 5 calibration folds."""
    out = []
    for cc in bundle["pipeline"].calibrated_classifiers_:
        steps = cc.estimator.named_steps
        names = np.array(steps["prep"].get_feature_names_out())
        out.append(list(names[steps["select"].get_support()]))
    return out


def imputed_values(bundle: dict) -> dict:
    """Median / most-frequent values the fitted imputers use for unasked inputs."""
    prep = bundle["explain_pipeline"].named_steps["prep"]
    num = prep.named_transformers_["num"].named_steps["impute"].statistics_
    cat = prep.named_transformers_["cat"].named_steps["impute"].statistics_
    vals = dict(zip(bundle["numeric"], [float(v) for v in num]))
    vals.update(dict(zip(bundle["categorical"], list(cat))))
    return vals


# --------------------------------------------------------------------------
# Validation and derived variables
# --------------------------------------------------------------------------
def _to_number(value, name: str, errors: list[str], integer: bool = False):
    if value is None or (isinstance(value, str) and value.strip() == ""):
        errors.append(f"{name} is required.")
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number.")
        return None
    if not np.isfinite(num):
        errors.append(f"{name} must be a finite number.")
        return None
    if integer and not float(num).is_integer():
        errors.append(f"{name} must be a whole number.")
        return None
    return int(num) if integer else num


def validate_inputs(raw: dict) -> dict:
    """Return cleaned inputs or raise ValidationError with readable messages."""
    errors: list[str] = []
    age = _to_number(raw.get("age"), "Age", errors, integer=True)
    waist = _to_number(raw.get("waist_cm"), "Waist circumference", errors)
    hip = _to_number(raw.get("hip_cm"), "Hip circumference", errors)
    cigs = _to_number(raw.get("cigarettes_per_day"), "Cigarettes per day", errors, integer=True)
    alcohol = raw.get("alcohol_frequency")

    def rng(val, key, label, unit=""):
        lo, hi = LIMITS[key]
        if val is not None and not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo:g} and {hi:g}{unit}.")

    rng(age, "age", "Age", " years (the model was developed for adults aged 18–40)")
    rng(waist, "waist_cm", "Waist circumference", " cm")
    rng(hip, "hip_cm", "Hip circumference", " cm")
    rng(cigs, "cigarettes_per_day", "Cigarettes per day", " (enter 0 if you do not smoke)")

    if alcohol in (None, ""):
        errors.append("Alcohol consumption frequency is required.")
    elif alcohol not in ALCOHOL_OPTIONS:
        errors.append("Alcohol consumption frequency must be one of the listed options.")

    whr = None
    if waist is not None and hip is not None and not errors:
        whr = calculate_whr(waist, hip)
        lo, hi = WHR_PLAUSIBLE
        if not (lo <= whr <= hi):
            errors.append(
                f"The waist-to-hip ratio from these measurements is {whr:.2f}, which is outside the "
                f"plausible range {lo:.2f}–{hi:.2f}. Please check that waist and hip were not swapped "
                f"and are both in centimetres.")
    if errors:
        raise ValidationError(errors)
    return {"age": age, "waist_cm": waist, "hip_cm": hip,
            "cigarettes_per_day": cigs, "alcohol_frequency": alcohol, "whr": whr}


def calculate_whr(waist_cm: float, hip_cm: float) -> float:
    if hip_cm is None or hip_cm <= 0:
        raise ValueError("Hip circumference must be greater than zero.")
    return float(waist_cm) / float(hip_cm)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def build_row(bundle: dict, clean: dict) -> pd.DataFrame:
    """One-row frame in the model's full input schema; unasked inputs are NaN."""
    row = {c: np.nan for c in input_schema(bundle)}
    row.update({
        "age": clean["age"],
        "waist_cm": clean["waist_cm"],
        "hip_cm": clean["hip_cm"],
        "whr": clean["whr"],
        "cigarettes_per_day": clean["cigarettes_per_day"],
        "alcohol_frequency": clean["alcohol_frequency"],
    })
    X = pd.DataFrame([row])[input_schema(bundle)]
    # Categorical columns must be object dtype for the fitted imputer/encoder.
    for c in bundle["categorical"]:
        X[c] = X[c].astype(object)
    return X


def categorise(score: float, threshold: float) -> str:
    if score >= threshold:
        return "Higher Risk"
    if score >= threshold * MODERATE_FACTOR:
        return "Moderate Risk"
    return "Lower Risk"


def score_row(bundle: dict, X: pd.DataFrame) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(bundle["pipeline"].predict_proba(X)[0, 1])


def explain_row(bundle: dict, X: pd.DataFrame):
    """KernelSHAP on the uncalibrated explain pipeline, as in Appendix G.3."""
    import shap

    steps = bundle["explain_pipeline"].named_steps
    clf = steps["clf"]
    Xt = steps["prep"].transform(X)
    names = np.array(steps["prep"].get_feature_names_out())
    Xt = steps["select"].transform(Xt)
    names = list(names[steps["select"].get_support()])

    def f(d):
        return clf.predict_proba(d)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(f, bundle["background"])
        sv = explainer.shap_values(Xt, nsamples=200, silent=True)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    contrib = sv[0]
    base = float(np.ravel(explainer.expected_value)[0])
    fx = float(f(Xt)[0])

    variables = [n.split("__", 1)[-1] for n in names]
    variables = ["alcohol_frequency" if v.startswith("alcohol_frequency") else v for v in variables]
    df = pd.DataFrame({
        "encoded_feature": names,
        "variable": variables,
        "label": [FRIENDLY.get(v, v) for v in variables],
        "shap_value": contrib,
        "modifiable": [v in MODIFIABLE for v in variables],
    })
    df["direction"] = np.where(df["shap_value"] > 0, "Increases", np.where(df["shap_value"] < 0, "Decreases", "No effect"))
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    return df, base, fx


def assess(raw: dict, bundle: dict | None = None, explain: bool = True) -> Assessment:
    bundle = bundle or load_bundle()
    clean = validate_inputs(raw)
    X = build_row(bundle, clean)
    score = score_row(bundle, X)
    thr = float(bundle["threshold"])
    result = Assessment(
        inputs={k: clean[k] for k in FORM_FIELDS},
        whr=clean["whr"], score=score, threshold=thr,
        moderate_cutoff=thr * MODERATE_FACTOR, category=categorise(score, thr),
    )
    if explain:
        try:
            df, base, fx = explain_row(bundle, X)
            result.contributions, result.shap_base_value, result.shap_model_output = df, base, fx
        except Exception as exc:  # noqa: BLE001 - mirror G.3: never block the result
            result.shap_error = f"{type(exc).__name__}: {exc}"
    return result
