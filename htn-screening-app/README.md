# Hypertension Risk Screening System

**An Explainable AI Approach to Early Elevated Blood-Pressure Screening**

A Streamlit web application that runs the trained model from the Professional Master's project
*Development and Evaluation of a SHAP-Enabled Machine-Learning Model for Elevated Blood-Pressure Screening in
Young Black Adults: A Predictive Study on National Survey Data with Application to the Lagos Context*
(Miva Open University, Abuja).

> **This application is a research prototype, not a diagnostic tool.** It has been developed using US NHANES
> survey data and has not been externally validated in a Nigerian population. Its output must not be used to
> diagnose hypertension, prescribe medication or make treatment decisions. Confirmatory blood-pressure
> measurement by a qualified health professional is required.

## What it does

| Page | Contents |
|---|---|
| **Home** | Project overview, key figures, method and technology summary |
| **Screen for Risk** | Five-question form (age, waist, hip, cigarettes/day, alcohol frequency); WHR calculated live; validation; three synthetic demonstration profiles |
| **Assessment Results** | Screening category, experimental model score, gauge, individual SHAP contributions, global SHAP importance, next steps, model-transparency note |
| **Research Analytics** | Model comparison, ROC-AUC / PR-AUC, confusion matrix, precision/recall/F1/specificity, calibration, decision curve, SHAP importance, subgroups, leakage-guard tests, all read from the supplied result files |
| **About the Project** | Aim, objectives, data, population, methodology, limitations, Nigerian validation plan |

## The model

* `model/model_bundle.joblib`: the study's actual bundle (SHA-256 `3921faad…aa82f`), unchanged.
* **Scoring:** `bundle['pipeline']`, an RBF SVM (C = 10, γ = 0.01) inside a 5-fold Platt-calibrated
  `CalibratedClassifierCV`, with median/most-frequent imputation, scaling, one-hot encoding,
  `SelectKBest(k=4)` and SMOTE.
* **Threshold and categories (Appendix G.3):** Higher if score ≥ 0.043404; Moderate if score ≥ 0.6 × threshold
  (0.026042); otherwise Lower.
* **Explanation (Appendix G.3):** SHAP `KernelExplainer` on `bundle['explain_pipeline']` (the uncalibrated
  SVM refitted on all development data) with the 50-row background stored in the bundle.
* **Inputs used:** age, waist-to-hip ratio, cigarettes per day, alcohol "Daily or almost daily".
  No blood-pressure reading, diagnosis or medication variable is collected or accepted. The app checks the
  model's input schema against the leakage list every time it loads.

### Known model behaviours (documented in the app)

1. **Lower Risk is not reached.** After Platt calibration every valid input scores ≥ ~0.035, which is above the
   Moderate cut-off of 0.026 (test calibration slope 5.95; 78.5% of the test partition flagged).
2. **Fold-level feature differences.** `SelectKBest` was refitted inside each calibration fold. Two folds kept
   `weight_kg`, one kept `pulse_bpm` and one kept `education`. The form does not ask for these, so, as in
   Appendix G.3, they are passed as missing and imputed with the cohort median/mode (80.3 kg, 70 bpm,
   Post-secondary). A person far from those values could see a score up to ~0.02 different.

## Run locally

Requires **Python 3.13** (scikit-learn 1.6.1 must match the pickled model).

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

There are 55 tests covering model loading, version match, feature compatibility, leakage exclusion, WHR, missing and
invalid inputs, age boundaries, determinism, Appendix G categorisation, SHAP additivity, demonstration profiles,
disclaimer presence on every page, and the end-to-end screening journey (Streamlit `AppTest`).

## Privacy and data

* No database, no accounts, no logging, no files written. Inputs live only in the browser session.
* `browser.gatherUsageStats = false`.
* **Not included in this repository:** participant-level cohort CSVs, training checkpoints and logs.
  `research_results/` contains only aggregate results. Note that the model bundle necessarily contains the SVM's
  support vectors and a 50-row SHAP background sample: standardised values of age, WHR, cigarettes/day and
  daily-alcohol for de-identified public-use NHANES records, with no identifiers.

## Project structure

```
app.py                 Entry point and navigation
htn/model.py           Loading, validation, WHR, scoring, categories, SHAP
htn/research.py        Loaders and Plotly charts for the supplied results
htn/ui.py              Theme, CSS, disclaimer and shared components
htn/profiles.py        Synthetic demonstration profiles
views/                 One module per page
model/                 model_bundle.joblib
research_results/      Aggregate CSV/JSON outputs from the pipeline run
tests/                 pytest + Streamlit AppTest suites
.streamlit/config.toml Theme and privacy settings
```

Deployment instructions are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.
