import streamlit as st

from htn import nav
from htn.research import run_summary
from htn.ui import POPULATION, card, disclaimer, footer, kpi


def render():
    rs = run_summary()
    t = rs["test_metrics"]
    st.markdown(
        """
<div class="hrs-hero">
  <span class="hrs-eyebrow">Explainable AI · Research prototype</span>
  <h1>Hypertension Risk Screening System</h1>
  <p><b>An Explainable AI Approach to Early Elevated Blood-Pressure Screening.</b>
  A machine-learning screening demonstrator that estimates whether a young adult should be
  prioritised for a blood-pressure measurement, and shows which inputs drove that assessment.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1.1, 1.1, 2.2])
    with c1:
        if st.button("Start Screening  →", type="primary", use_container_width=True, key="home_start"):
            nav.go("screen")
    with c2:
        if st.button("View research results", use_container_width=True, key="home_research"):
            nav.go("analytics")

    st.write("")
    disclaimer()

    st.markdown("### Project overview")
    st.markdown(
        "This application accompanies the Professional Master's research project "
        "*Development and Evaluation of a SHAP-Enabled Machine-Learning Model for Elevated Blood-Pressure "
        "Screening in Young Black Adults: A Predictive Study on National Survey Data with Application to the "
        "Lagos Context*. It runs the **actual trained model** from the study: a support vector machine with "
        "Platt calibration, selected by nested cross-validation from eight candidates. It asks only for the "
        "predictors that model uses, and never for blood-pressure readings, diagnoses or medication."
    )

    k = st.columns(4)
    with k[0]:
        kpi("Analytical cohort", f"{rs['n_total']:,}", f"{rs['n_events']} with elevated BP ({rs['prevalence']:.1%})")
    with k[1]:
        kpi("Selected model", "SVM", "RBF kernel, Platt-calibrated")
    with k[2]:
        kpi("Test ROC-AUC", f"{t['roc_auc']:.3f}", "Sealed partition, 163 people, 9 events")
    with k[3]:
        kpi("Model inputs", "4", "Age · WHR · cigarettes · alcohol")
    st.caption(f"Population: {POPULATION}. These are findings about that US population, not about Lagos.")

    st.markdown("### How it works")
    cols = st.columns(4)
    items = [
        ("📝", "1 · Enter five answers", "Age, waist and hip measurements, cigarettes per day and how often you drink alcohol. "
         "Waist-to-hip ratio is calculated for you."),
        ("🧠", "2 · The model scores", "Your answers pass through the study's own fitted preprocessing pipeline and the "
         "calibrated SVM. Nothing is invented or simulated."),
        ("🚦", "3 · A screening category", "The frozen threshold from the study places the score in Lower, Moderate or "
         "Higher priority for a blood-pressure check."),
        ("🔍", "4 · An explanation", "SHAP values show which inputs pushed the model's output up or down for you, "
         "alongside the model's global feature importance."),
    ]
    for col, (ico, title, body) in zip(cols, items):
        with col:
            card(ico, title, body)

    st.markdown("### About the research")
    a, b = st.columns([1.3, 1])
    with a:
        st.markdown(
            "Hypertension in young adults in Lagos is largely undetected, and many published risk models rely on "
            "laboratory tests or admit variables that define the outcome. This project built a **leakage-controlled** "
            "pipeline on measured US national survey data (NHANES 2021–2023) for Non-Hispanic Black adults aged 18–40. "
            "It compares eight algorithms against a clinical baseline (age, sex, BMI) using nested cross-validation, "
            "calibrates the selected model, sets a recall-based threshold, runs decision-curve and subgroup analysis, "
            "and explains predictions with SHAP. External validation in a Nigerian cohort is the necessary next step."
        )
        if st.button("Read about the project", key="home_about"):
            nav.go("about")
    with b:
        st.markdown("**Technology and methodology**")
        st.markdown(
            "- **Python · Streamlit** web application\n"
            "- **scikit-learn 1.6.1 · imbalanced-learn** for the model pipeline (imputation, scaling, one-hot "
            "encoding, SelectKBest, SMOTE, SVM)\n"
            "- **Nested cross-validation** for model selection\n"
            "- **Platt scaling** for calibration\n"
            "- **SHAP KernelExplainer** for explanations\n"
            "- **Plotly** for interactive charts\n"
            "- **No database, no accounts, no stored inputs**"
        )
    footer()
