"""Unit tests for the prediction engine. Run: python -m pytest -q"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from htn import model as M
from htn.profiles import DEMO_PROFILES

ROOT = Path(__file__).resolve().parent.parent
RS = json.loads((ROOT / "research_results" / "run_summary.json").read_text())


@pytest.fixture(scope="module")
def bundle():
    return M.load_bundle()


def good(**kw):
    base = dict(age=30, waist_cm=90.0, hip_cm=100.0, cigarettes_per_day=0, alcohol_frequency="Never")
    base.update(kw)
    return base


# ---------------------------------------------------------------- loading
def test_bundle_loads_with_expected_metadata(bundle):
    assert bundle["model_name"] == RS["selected_model"] == "Support Vector Machine"
    assert bundle["threshold"] == pytest.approx(RS["decision_threshold"])
    assert bundle["calibrated"] is True
    assert type(bundle["pipeline"]).__name__ == "CalibratedClassifierCV"


def test_sklearn_version_matches_run_summary():
    import sklearn
    assert sklearn.__version__ == RS["library_versions"]["scikit-learn"]


def test_feature_compatibility(bundle):
    assert M.selected_encoded_features(bundle) == RS["selected_encoded_features"]
    assert sorted(bundle["selected_variables"]) == sorted(RS["selected_variables"])
    assert len(M.input_schema(bundle)) == RS["n_features_numeric"] + RS["n_features_categorical"]


def test_no_leakage_columns_in_schema(bundle):
    schema = set(M.input_schema(bundle))
    assert not schema & set(RS["leakage_columns_excluded"])
    assert not schema & M.LEAKAGE_COLUMNS
    for fs in M.fold_feature_sets(bundle):
        names = {f.split("__", 1)[1] for f in fs}
        assert not names & M.LEAKAGE_COLUMNS


def test_check_bundle_rejects_leaky_schema(bundle):
    bad = dict(bundle)
    bad["numeric"] = list(bundle["numeric"]) + ["mean_sbp"]
    with pytest.raises(RuntimeError, match="Leakage"):
        M.check_bundle(bad)


def test_form_does_not_collect_leakage_variables():
    assert not set(M.FORM_FIELDS) & M.LEAKAGE_COLUMNS


# ---------------------------------------------------------------- WHR
def test_whr_calculation():
    assert M.calculate_whr(90, 100) == pytest.approx(0.9)
    assert M.calculate_whr(85.5, 102.3) == pytest.approx(85.5 / 102.3)
    with pytest.raises(ValueError):
        M.calculate_whr(80, 0)


def test_whr_passed_to_model(bundle):
    X = M.build_row(bundle, M.validate_inputs(good(waist_cm=81, hip_cm=90)))
    assert X.loc[0, "whr"] == pytest.approx(0.9)
    assert np.isnan(X.loc[0, "weight_kg"]) and pd.isna(X.loc[0, "sex"])


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("field", ["age", "waist_cm", "hip_cm", "cigarettes_per_day", "alcohol_frequency"])
def test_missing_inputs_rejected(field):
    raw = good()
    raw[field] = None
    with pytest.raises(M.ValidationError) as e:
        M.validate_inputs(raw)
    assert any("required" in m for m in e.value.messages)


@pytest.mark.parametrize("raw,needle", [
    (good(age="abc"), "must be a number"),
    (good(age=25.5), "whole number"),
    (good(cigarettes_per_day=-1), "between 0 and 60"),
    (good(cigarettes_per_day=2.5), "whole number"),
    (good(waist_cm=20), "Waist circumference must be between"),
    (good(hip_cm=500), "Hip circumference must be between"),
    (good(alcohol_frequency="Weekly"), "listed options"),
    (good(waist_cm=140, hip_cm=70), "waist-to-hip ratio"),
    (good(age=float("nan")), "finite"),
])
def test_invalid_inputs_rejected(raw, needle):
    with pytest.raises(M.ValidationError) as e:
        M.validate_inputs(raw)
    assert any(needle in m for m in e.value.messages), e.value.messages


@pytest.mark.parametrize("age,ok", [(17, False), (18, True), (40, True), (41, False), (0, False), (120, False)])
def test_age_boundaries(age, ok):
    if ok:
        assert M.validate_inputs(good(age=age))["age"] == age
    else:
        with pytest.raises(M.ValidationError, match="between 18 and 40"):
            M.validate_inputs(good(age=age))


# ---------------------------------------------------------------- prediction
def test_prediction_matches_direct_pipeline_call(bundle):
    raw = good(age=33, waist_cm=95, hip_cm=101, cigarettes_per_day=5, alcohol_frequency="Daily or almost daily")
    a = M.assess(raw, bundle, explain=False)
    row = {c: np.nan for c in M.input_schema(bundle)}
    row.update(age=33, waist_cm=95, hip_cm=101, whr=95 / 101, cigarettes_per_day=5,
               alcohol_frequency="Daily or almost daily")
    direct = bundle["pipeline"].predict_proba(pd.DataFrame([row])[M.input_schema(bundle)])[0, 1]
    assert a.score == pytest.approx(direct, abs=1e-12)
    assert 0 < a.score < 1


def test_prediction_is_deterministic(bundle):
    raw = good(age=27)
    assert M.assess(raw, bundle, explain=False).score == M.assess(raw, bundle, explain=False).score


def test_categorisation_logic_matches_appendix_g():
    t = 0.04340401747306985
    assert M.categorise(t, t) == "Higher Risk"
    assert M.categorise(t + 1e-9, t) == "Higher Risk"
    assert M.categorise(t - 1e-9, t) == "Moderate Risk"
    assert M.categorise(0.6 * t, t) == "Moderate Risk"
    assert M.categorise(0.6 * t - 1e-9, t) == "Lower Risk"
    assert M.categorise(0.0, t) == "Lower Risk"


def test_assessment_uses_bundle_threshold(bundle):
    a = M.assess(good(), bundle, explain=False)
    assert a.threshold == bundle["threshold"]
    assert a.moderate_cutoff == pytest.approx(0.6 * bundle["threshold"])
    assert a.category == M.categorise(a.score, a.threshold)


def test_score_floor_and_unasked_input_sensitivity(bundle):
    """Backs two statements in the results page's transparency note."""
    cols = M.input_schema(bundle)
    rows = []
    for age, whr, c, al in itertools.product(range(18, 41, 2), np.arange(0.55, 1.41, 0.05),
                                             [0, 2, 5, 10, 20, 40, 60],
                                             ["Never", "Daily or almost daily"]):
        r = {k: np.nan for k in cols}
        r.update(age=age, whr=whr, cigarettes_per_day=c, alcohol_frequency=al)
        rows.append(r)
    X = pd.DataFrame(rows)[cols]
    p = bundle["pipeline"].predict_proba(X)[:, 1]
    assert p.min() >= 0.035 - 5e-4          # "about 0.035 or above"
    assert p.min() > 0.6 * bundle["threshold"]  # Lower Risk not reached
    # Weight, pulse and education (used by some calibration folds) move the score by "up to about 0.02"
    sub = X.sample(150, random_state=0).reset_index(drop=True)
    base = bundle["pipeline"].predict_proba(sub)[:, 1]
    worst = 0
    for col, vals in {"weight_kg": [45, 60, 100, 140, 180], "pulse_bpm": [45, 60, 90, 120],
                      "education": ["Less than secondary", "Secondary", "Bachelor's degree / HND"]}.items():
        for v in vals:
            s = sub.copy()
            s[col] = v
            if col == "education":
                s[col] = s[col].astype(object)
            worst = max(worst, np.abs(bundle["pipeline"].predict_proba(s)[:, 1] - base).max())
    assert worst < 0.02, worst


def test_alcohol_only_daily_level_matters(bundle):
    scores = {al: M.assess(good(alcohol_frequency=al), bundle, explain=False).score for al in M.ALCOHOL_OPTIONS}
    non_daily = [v for k, v in scores.items() if k != "Daily or almost daily"]
    assert max(non_daily) - min(non_daily) < 1e-12
    assert scores["Daily or almost daily"] != non_daily[0]


def test_demo_profiles_run_through_model(bundle):
    cats = {}
    for name, raw in DEMO_PROFILES.items():
        a = M.assess(raw, bundle)
        assert a.shap_error is None
        cats[name] = a.category
    # Recorded so a change in model behaviour is noticed; not assigned by the app.
    assert list(cats.values()) == ["Moderate Risk", "Higher Risk", "Higher Risk"]


# ---------------------------------------------------------------- SHAP
def test_shap_explanation_is_additive_and_complete(bundle):
    a = M.assess(good(age=36, waist_cm=100, hip_cm=104, cigarettes_per_day=8), bundle)
    df = a.contributions
    assert a.shap_error is None
    assert set(df["variable"]) == {"age", "whr", "cigarettes_per_day", "alcohol_frequency"}
    assert a.shap_base_value + df["shap_value"].sum() == pytest.approx(a.shap_model_output, abs=1e-6)
    assert list(df["shap_value"].abs()) == sorted(df["shap_value"].abs(), reverse=True)
    assert set(df["direction"]) <= {"Increases", "Decreases", "No effect"}
    # Explained output is the uncalibrated explain pipeline, as in Appendix G.3
    X = M.build_row(bundle, M.validate_inputs(good(age=36, waist_cm=100, hip_cm=104, cigarettes_per_day=8)))
    assert a.shap_model_output == pytest.approx(bundle["explain_pipeline"].predict_proba(X)[0, 1], abs=1e-9)


def test_shap_failure_does_not_block_result(bundle, monkeypatch):
    monkeypatch.setattr(M, "explain_row", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    a = M.assess(good(), bundle)
    assert a.category in {"Lower Risk", "Moderate Risk", "Higher Risk"}
    assert "boom" in a.shap_error


def test_no_files_written(bundle, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(ROOT.rglob("*"))
    M.assess(good(), bundle)
    after = set(ROOT.rglob("*"))
    assert not [p for p in after - before if "__pycache__" not in str(p) and ".pytest_cache" not in str(p)]
    assert not list(tmp_path.iterdir())
