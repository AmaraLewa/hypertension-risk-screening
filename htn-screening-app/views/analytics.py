import pandas as pd
import streamlit as st

from htn.research import (decision_curve, fig_brier_compare, fig_calibration_compare, fig_confusion, fig_decision_curve,
                          fig_nested_metric, fig_shap_global, fig_subgroups, leakage_tests, model_comparison,
                          run_summary, shap_global, subgroups)
from htn.ui import PLOTLY_CONFIG, POPULATION, disclaimer, footer, kpi, page_header


def _chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


def render():
    rs = run_summary()
    t = rs["test_metrics"]
    page_header("Research analytics",
                "Results from the study's actual pipeline run, loaded directly from the supplied CSV and JSON outputs. "
                "No value on this page has been recomputed, rounded upwards or altered.")
    st.markdown(
        f"<span class='hrs-pill'>👥 {POPULATION}</span>"
        f"<span class='hrs-pill'>Analytical cohort n = {rs['n_total']}</span>"
        f"<span class='hrs-pill'>Events = {rs['n_events']} ({rs['prevalence']:.1%})</span>"
        "<span class='hrs-pill'>80% development / 20% sealed test</span>"
        "<span class='hrs-pill'>Test n = 163 · 9 events</span>",
        unsafe_allow_html=True)
    st.caption("These are findings about a US survey population. They are not Lagos or Nigerian population findings.")

    tabs = st.tabs(["Model comparison", "Classification", "Calibration", "Decision curve",
                    "SHAP importance", "Subgroups", "Leakage guard"])

    # ---------------------------------------------------------------- 1, 2
    with tabs[0]:
        mc = model_comparison()
        base = mc[mc.is_baseline].iloc[0]
        sel = mc[mc.model == rs["selected_model"]].iloc[0]
        k = st.columns(4)
        with k[0]:
            kpi("Selected model", "SVM", "Highest nested PR-AUC")
        with k[1]:
            kpi("Nested PR-AUC", f"{sel.nested_pr_auc:.3f}", f"± {sel.nested_pr_auc_sd:.3f} SD")
        with k[2]:
            kpi("Baseline nested PR-AUC", f"{base.nested_pr_auc:.3f}", "Age, sex, BMI")
        with k[3]:
            gain = sel.nested_pr_auc - base.nested_pr_auc
            kpi("Gain over baseline", f"+{gain:.3f}", f"{gain / sel.nested_pr_auc_sd:.2f} SD (below 1 SD)")
        c1, c2 = st.columns(2)
        with c1:
            _chart(fig_nested_metric("nested_pr_auc", "nested_pr_auc_sd", "Nested CV PR-AUC", rs["prevalence"],
                                     f"No-skill PR-AUC = prevalence ({rs['prevalence']:.3f})"), "cmp_pr")
        with c2:
            _chart(fig_nested_metric("nested_roc_auc", None, "Nested CV ROC-AUC", 0.5, "Chance = 0.5"), "cmp_roc")
        st.caption("Teal = selected model · blue = clinical baseline · grey = other candidates. Error bars = SD across "
                   "outer folds. The baseline had the highest nested ROC-AUC; the SVM was selected on nested PR-AUC by the "
                   "pre-specified rule. The intervals overlap widely, so the candidates cannot be clearly told apart at "
                   "this sample size.")
        c3, c4 = st.columns(2)
        with c3:
            _chart(fig_nested_metric("roc_auc", None, "Test ROC-AUC (before calibration)", 0.5, "Chance = 0.5"), "cmp_troc")
        with c4:
            _chart(fig_nested_metric("pr_auc", None, "Test PR-AUC (before calibration)", rs["prevalence"],
                                     "No-skill = prevalence"), "cmp_tpr")
        st.markdown("**Full comparison table** (`model_comparison.csv`, sealed test partition, each model evaluated "
                    "before calibration at its own recall-based threshold)")
        show = mc.rename(columns={"model": "Model", "is_baseline": "Baseline", "nested_pr_auc": "Nested PR-AUC",
                                  "nested_pr_auc_sd": "SD", "nested_roc_auc": "Nested ROC-AUC",
                                  "threshold": "Threshold", "roc_auc": "Test ROC-AUC", "pr_auc": "Test PR-AUC",
                                  "precision": "Precision", "recall": "Recall", "f1": "F1",
                                  "specificity": "Specificity", "brier": "Brier", "accuracy": "Accuracy"})
        cols = ["Model", "Baseline", "Nested PR-AUC", "SD", "Nested ROC-AUC", "Test ROC-AUC", "Test PR-AUC",
                "Threshold", "Precision", "Recall", "F1", "Specificity", "Brier", "tp", "fp", "tn", "fn"]
        st.dataframe(show[cols].style.format({c: "{:.3f}" for c in cols[2:13]}), hide_index=True,
                     use_container_width=True)
        st.caption("ROC and PR curves are not part of the supplied outputs, so only the area values are shown; no "
                   "curve has been reconstructed.")

    # ---------------------------------------------------------------- 3, 4
    with tabs[1]:
        st.markdown("#### Selected model at its operating point (calibrated SVM, threshold "
                    f"{t['threshold']:.4f}, sealed test partition)")
        k = st.columns(5)
        for col, (lab, key) in zip(k, [("Recall / sensitivity", "recall"), ("Specificity", "specificity"),
                                         ("Precision / PPV", "precision"), ("F1-score", "f1"), ("Accuracy", "accuracy")]):
            with col:
                kpi(lab, f"{t[key]:.3f}")
        c1, c2 = st.columns([1, 1])
        with c1:
            _chart(fig_confusion(t["tp"], t["fp"], t["tn"], t["fn"], "Confusion matrix (n = 163)"), "cm")
        with c2:
            k2 = st.columns(2)
            with k2[0]:
                kpi("Test ROC-AUC", f"{t['roc_auc']:.3f}", "Calibrated model")
            with k2[1]:
                kpi("Test PR-AUC", f"{t['pr_auc']:.3f}", f"No-skill = {rs['prevalence']:.3f}")
            st.write("")
            k3 = st.columns(2)
            with k3[0]:
                kpi("Fraction flagged", f"{t['fraction_flagged']:.1%}", "Of the test partition")
            with k3[1]:
                kpi("Net benefit", f"{t['net_benefit_at_threshold']:.4f}", "At the threshold")
            st.markdown(
                f"\nAll **{t['tp']}** people with elevated blood pressure in the test partition were flagged "
                f"(no false negatives). The cost is low specificity: **{t['fp']}** of {t['fp'] + t['tn']} people without "
                "elevated BP were also flagged. With only 9 test events, recall of 1.00 has a wide confidence "
                "interval.")

    # ---------------------------------------------------------------- 5
    with tabs[2]:
        k = st.columns(4)
        with k[0]:
            kpi("Slope before", f"{t['uncalibrated_slope']:.2f}", f"Intercept {t['uncalibrated_intercept']:.2f}")
        with k[1]:
            kpi("Slope after", f"{t['calibration_slope']:.2f}", f"Intercept {t['calibration_intercept']:.2f}")
        with k[2]:
            kpi("Brier before", f"{t['uncalibrated_brier']:.3f}")
        with k[3]:
            kpi("Brier after", f"{t['brier']:.3f}")
        c1, c2 = st.columns(2)
        with c1:
            _chart(fig_calibration_compare(), "cal_slope")
        with c2:
            _chart(fig_brier_compare(), "cal_brier")
        st.markdown(
            "**Interpretation.** Platt scaling (5-fold, sigmoid) brought the predicted values down to the 5.3% event "
            "rate, which cut the Brier score from 0.207 to 0.051. It also compressed them into a narrow band: the test "
            "calibration slope rose from 1.19 to 5.95, where 1.0 is ideal. A slope well above 1 means the calibrated "
            "scores vary too little between people. The calibrated score should therefore be treated as a ranking "
            "signal, not a reliable absolute probability. This is why the app labels it *experimental*.")
        st.caption("A binned reliability diagram needs per-participant predictions, which are not in the supplied "
                   "outputs; only the summary statistics are reported.")

    # ---------------------------------------------------------------- 6
    with tabs[3]:
        _chart(fig_decision_curve(), "dca")
        lo, hi = t["benefit_band"]
        dc = decision_curve()
        row = dc.iloc[(dc.threshold - 0.04).abs().argmin()]
        st.markdown(
            f"The model offers positive net benefit over both default strategies between thresholds **{lo:.2f} and "
            f"{hi:.2f}**. The operating threshold ({t['threshold']:.4f}) lies inside this band "
            f"(`threshold_in_benefit_band = {t['threshold_in_benefit_band']}`). At that point the model's net benefit is "
            f"**{t['net_benefit_at_threshold']:.4f}**, against {row.net_benefit_screen_all:.4f} for screening everyone. "
            "The band is narrow, and the gain is small.")

    # ---------------------------------------------------------------- 7
    with tabs[4]:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            _chart(fig_shap_global(), "shap_glob")
        with c2:
            sg = shap_global().sort_values("mean_abs_shap", ascending=False)
            st.dataframe(sg[["label", "mean_abs_shap"]].rename(columns={"label": "Variable", "mean_abs_shap": "Mean |SHAP|"})
                         .style.format({"Mean |SHAP|": "{:.4f}"}), hide_index=True, use_container_width=True)
            cv = rs["cv_results"]
            st.caption(f"Top-variable agreement across bootstraps: {cv['_shap_top1_agreement']:.0%}; top-k overlap: "
                       f"{cv['_shap_stability_top10_overlap']:.0%} (20 bootstraps, as reported in `run_summary.json`). "
                       "Section 5.11 of the report notes that an earlier version of this stability check was "
                       "uninformative for non-tree models (defect 18), so treat these figures with caution.")
        st.info("SHAP values describe how the model uses each input. They do not show that any factor causes "
                "elevated blood pressure.")

    # ---------------------------------------------------------------- 8
    with tabs[5]:
        sgp = subgroups()
        c1, c2 = st.columns(2)
        with c1:
            _chart(fig_subgroups("roc_auc", "ROC-AUC"), "sub_auc")
        with c2:
            _chart(fig_subgroups("specificity", "Specificity"), "sub_spec")
        tbl = sgp.assign(Subgroup=sgp["group"].map({"sex": "Sex", "age_band": "Age band"}) + ": " + sgp["level"])
        cols = ["Subgroup", "n", "tp", "fn", "fp", "tn", "recall", "specificity", "precision", "roc_auc", "pr_auc", "brier"]
        st.dataframe(tbl[cols].rename(columns={"recall": "Recall", "specificity": "Specificity", "precision": "Precision",
                                               "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier": "Brier"})
                     .style.format({c: "{:.3f}" for c in ["Recall", "Specificity", "Precision", "ROC-AUC", "PR-AUC", "Brier"]}),
                     hide_index=True, use_container_width=True)
        st.warning("Subgroups contain between 2 and 7 events each, so these estimates are very imprecise. Recall was 1.00 "
                   "in every subgroup. Specificity was very low for ages 30–40 (0.017): nearly everyone in that band "
                   "was flagged.")

    # ---------------------------------------------------------------- 9
    with tabs[6]:
        lt = leakage_tests()
        with st.container(border=True):
            st.markdown("**Columns excluded by construction**")
            st.markdown(" ".join(f"<span class='hrs-pill'>🚫 {c}</span>" for c in rs["leakage_columns_excluded"]),
                        unsafe_allow_html=True)
            st.caption("Blood-pressure readings, their means, current BP medication, prior diagnosis and prior BP "
                       "checks. None can enter the predictors. The app re-checks the model's input schema against "
                       "this list every time the model loads.")
        st.markdown("**Leakage-guard verification tests** (`leakage_test_results.json`)")
        names = {"0_control": "Control: clean cohort", "A_unknown_name": "A: outcome-derived column under an unknown name",
                 "B_contaminated_predictor": "B: approved predictor contaminated with the label",
                 "C_noisy_contamination": "C: noisy proxy of the label"}
        rows = []
        for key, v in lt["tests"].items():
            rows.append({"Test": names.get(key, key), "Result": v.get("result", ""), "Expected": v.get("expected", ""),
                         "Correlation with outcome": v.get("correlation_with_outcome")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"Correlation with outcome": st.column_config.NumberColumn(format="%.3f")})
        st.caption(f"Run on a copy of the cohort (n = {lt['n']}, {lt['events']} events, seed {lt['seed']}). "
                   "Test C records a known limit: a noisy proxy at r = 0.818 passed the 0.95 alarm but would be "
                   "caught at 0.80, the threshold the report recommends.")
    disclaimer(compact=True)
    footer()
