import streamlit as st

from htn.research import run_summary
from htn.ui import POPULATION, disclaimer, footer, page_header


def render():
    rs = run_summary()
    page_header("About the project",
                "Development and Evaluation of a SHAP-Enabled Machine-Learning Model for Elevated Blood-Pressure "
                "Screening in Young Black Adults: A Predictive Study on National Survey Data with Application to the "
                "Lagos Context")
    st.caption("Professional Master of Information Technology (AI and Machine Learning) · Department of Information "
               "Technology, School of Computing · Miva Open University, Abuja")

    t1, t2, t3, t4, t5 = st.tabs(["Aim and objectives", "Data and population", "Methodology",
                                  "Explainability", "Limitations and next steps"])
    with t1:
        st.markdown("#### Aim")
        st.markdown(
            "To develop and evaluate a SHAP-enabled, leakage-controlled machine-learning model for current elevated "
            "blood-pressure screening among young Black adults, using measured national survey data, and to assess its "
            "discrimination, calibration, decision-analytic utility and subgroup performance as a basis for subsequent "
            "external validation in a Nigerian population.")
        st.markdown("#### Objectives")
        st.markdown(
            "1. **Specify the screening task and assemble the analytic cohort:** target population, eligibility, outcome "
            "definition and non-laboratory candidate predictors, mapped to survey variables, with a published data "
            "dictionary and an explicit leakage-control specification.\n"
            "2. **Prepare the data under reproducible, leakage-controlled procedures:** exclude people with a prior "
            "diagnosis or current antihypertensive treatment, derive the outcome from measured readings, and apply the "
            "allow-list leakage guard, imputation and encoding within cross-validation folds.\n"
            "3. **Develop and compare the prescribed algorithms** within a pipeline with nested cross-validation, "
            "probability calibration and pre-specified threshold selection, selecting one model by a rule fixed before "
            "testing.\n"
            "4. **Evaluate the selected model on a sealed test partition** across discrimination, calibration, "
            "decision-analytic and subgroup criteria, following TRIPOD+AI.\n"
            "5. **Integrate SHAP-based attribution**, assess its stability, implement a non-diagnostic proof-of-concept "
            "interface, and carry out software, integration, safety and interface testing.")
    with t2:
        st.markdown("#### Data source: NHANES")
        st.markdown(
            "The US National Health and Nutrition Examination Survey (NHANES), August 2021–August 2023 cycle, run by the "
            "CDC National Center for Health Statistics. It combines interviews with standardised physical examinations, "
            "including oscillometric blood-pressure measurement.")
        st.markdown("#### Population and eligibility")
        st.markdown(
            f"- **Population:** {POPULATION}.\n"
            "- **Eligible:** age 18–40 at interview (RIDAGEYR) and race/ethnicity Non-Hispanic Black (RIDRETH3 = 4), "
            "with valid examination readings.\n"
            "- **Excluded from the analytical cohort:** anyone reporting a prior diagnosis of hypertension or current "
            "antihypertensive medication. The tool is for people not yet diagnosed, so 151 of 963 eligible records were "
            f"excluded, leaving **{rs['n_total']}** for modelling.\n"
            "- **Outcome:** current elevated blood pressure, i.e. a mean of readings 2 and 3 of systolic ≥ 140 mmHg "
            "and/or diastolic ≥ 90 mmHg (ESH 2023). This is a single-session screening status, **not a diagnosis** of "
            f"hypertension. Prevalence in the analytical cohort: **{rs['prevalence']:.1%}** ({rs['n_events']} events).\n"
            "- **Predictors offered to the pipeline:** 21 allow-listed non-laboratory variables (13 numeric, 8 "
            "categorical). Fourteen instrument variables, including salt intake, perceived stress and family history, "
            "have no equivalent in this NHANES cycle.")
    with t3:
        st.markdown("#### Model development")
        st.markdown(
            "- **Partition:** 80% development (649 people, 34 events) / 20% sealed test (163 people, 9 events), "
            "stratified.\n"
            "- **Candidates:** a clinical baseline (age, sex, BMI) and logistic regression, elastic-net logistic "
            "regression, random forest, XGBoost, LightGBM, support vector machine and k-nearest neighbours.\n"
            "- **Preprocessing inside every fold:** median imputation and standardisation (numeric); most-frequent "
            "imputation and one-hot encoding (categorical); **SelectKBest** capped at **4 encoded parameters**, the "
            "number 34 events can support; **SMOTE** oversampling, because the event rate was below 10%.\n"
            "- **Selection rule:** highest nested cross-validated PR-AUC, fixed in advance. The selected model is an "
            f"RBF-kernel SVM (C = 10, γ = 0.01, balanced class weights) using `{'`, `'.join(rs['selected_variables'])}`.")
        st.markdown("#### Nested cross-validation")
        st.markdown(
            "An outer loop estimates performance and an inner loop tunes hyperparameters, so no data used to choose "
            "settings is used to score them. This avoids the optimistic bias of tuning and evaluating on the same folds.")
        st.markdown("#### Target-leakage prevention")
        st.markdown(
            "An **allow-list guard** admits only the approved predictors. Thirteen outcome-derived columns (all "
            "blood-pressure readings and their means, BP medication, prior diagnosis and prior BP check) are excluded by "
            "construction, and the run stops if any admitted feature correlates with the outcome at r ≥ 0.95. The guard "
            "was tested with planted faults; see *Research Analytics → Leakage guard*.")
        st.markdown("#### Calibration and decision analysis")
        st.markdown(
            "The selected SVM was calibrated with **Platt scaling** (sigmoid, 5 folds). The operating threshold "
            f"(**{rs['decision_threshold']:.4f}**) was chosen on development data to give at least 85% recall, then "
            "frozen. **Decision-curve analysis** (Vickers & Elkin, 2006) checks whether the model gives more net "
            "benefit than screening everyone or no one at that threshold.")
    with t4:
        st.markdown("#### SHAP explainability")
        st.markdown(
            "**SHAP** (Lundberg & Lee, 2017) assigns each input a contribution to an individual prediction, based on "
            "Shapley values from cooperative game theory. For the SVM, a model-agnostic **KernelExplainer** is used with "
            "a 50-row background sample stored in the model bundle. The app shows both the **individual** explanation "
            "for each screening and the **global** mean absolute SHAP importance from the study.\n\n"
            "SHAP describes **model attribution, not causation**. The data are cross-sectional, and an input that "
            "raises the model's output is not thereby shown to raise blood pressure.")
    with t5:
        st.markdown("#### Study limitations")
        st.markdown(
            "- **The cohort is not Nigerian.** Results describe Non-Hispanic Black adults in the US and cannot be "
            "taken as Lagos population findings.\n"
            "- **Small number of events** (43 in total, 9 in the test partition): wide confidence intervals and limited "
            "power to separate candidate models. The SVM's gain over the clinical baseline was less than one standard "
            "deviation.\n"
            "- **Calibration compression:** after Platt scaling the test calibration slope was 5.95, and 78.5% of test "
            "participants were flagged (specificity 0.23).\n"
            "- **Fourteen specified predictors are missing** from NHANES 2021–2023 (e.g. salt intake, stress, family "
            "history).\n"
            "- **Cross-sectional design:** the model classifies current status and supports no causal claims.\n"
            "- **Internal validation only**, within one survey cycle.\n"
            "- **Usability not studied:** whether users understand the category and the caveat has not been evaluated.")
        st.markdown("#### Future validation in Nigeria")
        st.markdown(
            "The report specifies the next step: **external validation in a Lagos cohort** using the instrument, data "
            "dictionary and NHANES-matched measurement protocol in Part B, after ethical approval. The pipeline runs on "
            "any dataset in the same schema, so validation requires collecting data, not rebuilding software. Further "
            "recommendations include restoring the missing predictors, pooling NHANES cycles, lowering the leakage alarm "
            "to r ≈ 0.80 and evaluating comprehension with real users.")
        st.markdown("#### Key references")
        st.markdown(
            "- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in "
            "Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates. "
            "https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html\n"
            "- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction "
            "models. *Medical Decision Making, 26*(6), 565–574. https://doi.org/10.1177/0272989X06295361\n"
            "- Mancia, G., Kreutz, R., Brunström, M., et al. (2023). 2023 ESH guidelines for the management of arterial hypertension. *Journal of "
            "Hypertension, 41*(12), 1874–2071. https://doi.org/10.1097/HJH.0000000000003480\n"
            "- Collins, G. S., Moons, K. G. M., Dhiman, P., et al. (2024). TRIPOD+AI statement: Updated guidance for "
            "reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. "
            "https://doi.org/10.1136/bmj-2023-078378")
    disclaimer(compact=True)
    footer()
