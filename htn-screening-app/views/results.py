import html

import plotly.graph_objects as go
import streamlit as st

from htn import nav
from htn.model import fold_feature_sets, imputed_values, load_bundle
from htn.research import fig_shap_global
from htn.ui import (CATEGORY_STYLE, CORAL, INK_2, NAVY, PLOTLY_CONFIG, POPULATION, TEAL, disclaimer, footer,
                    kpi, page_header, plotly_layout)

AXIS_MAX = 0.10

EXPLAIN = {
    "Higher Risk": "The model's output is at or above the study's frozen operating threshold. In the study's terms "
                   "this means you would be <b>prioritised for a blood-pressure measurement</b>. It does not mean you have "
                   "high blood pressure.",
    "Moderate Risk": "The model's output is below the operating threshold but within 60% of it, the band the original "
                     "prototype labels Moderate. A blood-pressure check is still the only way to know your reading.",
    "Lower Risk": "The model's output is below 60% of the operating threshold. This does not rule out high blood "
                  "pressure, which often has no symptoms; a routine blood-pressure check is still recommended.",
}

NEXT_STEPS = {
    "Higher Risk": ["**Have your blood pressure measured soon** by a qualified health worker at a pharmacy, clinic "
                    "or primary health centre. It takes a few minutes.",
                    "A single high reading is not a diagnosis. Confirmation needs repeated readings on separate occasions.",
                    "Do not start, stop or change any medicine based on this result."],
    "Moderate Risk": ["**Arrange a blood-pressure measurement** with a qualified health worker; it is the only way to "
                      "know your actual blood pressure.",
                      "Do not start, stop or change any medicine based on this result."],
    "Lower Risk": ["**Keep up routine blood-pressure checks**: adults should know their numbers even when they feel well.",
                   "Do not start, stop or change any medicine based on this result."],
}


def _gauge(score: float, thr: float, mod: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(valueformat=".3f", font=dict(family="Manrope, Inter", size=40, color=NAVY)),
        gauge=dict(
            axis=dict(range=[0, AXIS_MAX], tickformat=".2f", tickcolor="#94A3B8", tickfont=dict(color=INK_2, size=11)),
            bar=dict(color=NAVY, thickness=0.28),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, mod], color="#D8F0E4"),
                   dict(range=[mod, thr], color="#FBE7C2"),
                   dict(range=[thr, AXIS_MAX], color="#F8D3D7")],
            threshold=dict(line=dict(color=CORAL, width=3), thickness=0.9, value=thr),
        ),
    ))
    fig = plotly_layout(fig, height=250)
    fig.update_layout(margin=dict(l=45, r=45, t=25, b=5))
    return fig


def _shap_bar(df):
    d = df.iloc[::-1]
    colors = [CORAL if v > 0 else TEAL for v in d["shap_value"]]
    fig = go.Figure(go.Bar(
        x=d["shap_value"], y=d["label"], orientation="h", marker=dict(color=colors, cornerradius=4), width=0.55,
        text=[f"{v:+.3f}" for v in d["shap_value"]], textposition="outside",
        customdata=d["direction"],
        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:+.4f}<br>%{customdata} the model output<extra></extra>",
    ))
    m = max(0.05, float(d["shap_value"].abs().max()) * 1.6)
    fig.add_vline(x=0, line_color="#94A3B8", line_width=1)
    fig.update_xaxes(title="← decreases · increases →", range=[-m, m], zeroline=False)
    return plotly_layout(fig, height=300, title="Your individual SHAP contributions", showlegend=False)


def _value_text(var: str, inputs: dict, whr: float) -> str:
    return {"age": f"{inputs['age']} years", "whr": f"{whr:.3f}",
            "cigarettes_per_day": f"{inputs['cigarettes_per_day']} per day",
            "alcohol_frequency": inputs["alcohol_frequency"]}.get(var, "")


def render():
    page_header("Assessment results")
    disclaimer()

    a = st.session_state.get("assessment")
    if a is None:
        st.info("No assessment yet. Complete the screening form, or run one of the synthetic demonstration profiles.")
        if st.button("Go to the screening form →", type="primary", key="res_go_screen"):
            nav.go("screen")
        footer()
        return

    style = CATEGORY_STYLE[a.category]
    st.caption(f"Input source: {st.session_state.get('assessment_source', 'Form')}")

    top_l, top_r = st.columns([1.15, 1])
    with top_l:
        st.markdown(
            f'<div class="hrs-cat" style="background:{style["bg"]};border-color:{style["color"]}55">'
            f'<div class="k" style="color:{style["color"]}">Screening category</div>'
            f'<div class="v" style="color:{style["color"]}">{style["icon"]} {a.category}</div>'
            f'<p>{EXPLAIN[a.category]}</p></div>', unsafe_allow_html=True)
        st.write("")
        k1, k2, k3 = st.columns(3)
        with k1:
            kpi("Model output score", f"{a.score:.3f}", "Experimental")
        with k2:
            kpi("Threshold", f"{a.threshold:.4f}", "Higher if ≥ this")
        with k3:
            kpi("Moderate cut-off", f"{a.moderate_cutoff:.4f}", "0.6 × threshold")
    with top_r:
        with st.container(border=True):
            st.markdown("**Visual risk indicator**")
            st.plotly_chart(_gauge(a.score, a.threshold, a.moderate_cutoff), use_container_width=True,
                            config=PLOTLY_CONFIG, key="gauge")
            st.markdown(
                "<div style='font-size:.82rem;color:#475569;text-align:center'>"
                "<span style='color:#1F8A5B'>● Lower</span> &nbsp; <span style='color:#B7791F'>▲ Moderate</span> &nbsp; "
                "<span style='color:#C8323E'>◆ Higher</span> &nbsp; · &nbsp; red marker = operating threshold</div>",
                unsafe_allow_html=True)
    st.warning(
        "**About the score:** this is the experimental output of a Platt-calibrated research model. "
        "**It is not a clinically validated probability** and should not be read as your chance of having high "
        "blood pressure. On the study's test data the calibrated scores were compressed into a narrow band (calibration "
        "slope 5.95, where 1.0 is ideal), so the screening category matters more than the exact number.")

    # ------------------------------------------------------------ factors
    st.markdown("### What drove this assessment")
    if a.shap_error or a.contributions.empty:
        st.error("The SHAP explanation could not be computed for this input. The screening category above is unaffected. "
                 f"(Detail: {a.shap_error})")
    else:
        c1, c2 = st.columns([1, 1.25])
        with c1:
            with st.container(border=True):
                st.markdown("**Top contributing factors**")
                rows = []
                for _, r in a.contributions.iterrows():
                    up = r["shap_value"] > 0
                    arrow = ('<span class="hrs-up">▲ increases</span>' if up else
                             '<span class="hrs-down">▼ decreases</span>')
                    tag = "modifiable" if r["modifiable"] else "not modifiable"
                    rows.append(
                        f'<div class="hrs-factor"><div><div class="name">{html.escape(r["label"])}</div>'
                        f'<div class="meta">Your value: {html.escape(_value_text(r["variable"], a.inputs, a.whr))} · {tag}</div></div>'
                        f'<div>{arrow}</div></div>')
                st.markdown("".join(rows), unsafe_allow_html=True)
                st.caption("Ordered by size of influence. 'Increases' and 'decreases' describe the model's output, "
                           "not your body.")
        with c2:
            st.plotly_chart(_shap_bar(a.contributions), use_container_width=True, config=PLOTLY_CONFIG, key="shap_local")
            st.caption(f"Model output before calibration f(x) = {a.shap_model_output:.3f} = base value "
                       f"{a.shap_base_value:.3f} + sum of contributions {a.contributions['shap_value'].sum():+.3f}.")

    with st.expander("ℹ️  What SHAP means (and what it does not)", expanded=False):
        st.markdown(
            "**SHAP (SHapley Additive exPlanations)** splits one model output into a contribution from each input, "
            "measured against the model's average output over a reference sample (the *base value*). A positive value "
            "means that input pushed the model's output up for this person; a negative value means it pushed it down. "
            "The contributions add up exactly to the output.\n\n"
            "**SHAP explains the model's behaviour, not medicine.** A factor that increases the output is not shown to "
            "cause high blood pressure, and a factor that decreases it is not protective. The model was trained on "
            "cross-sectional survey data, which cannot support causal claims. Some patterns reflect the SVM's shape; "
            "for example, cigarettes can pull the output down at some ages, which is a property of the model, not "
            "evidence that smoking lowers blood pressure.\n\n"
            "**Technical note.** As in the study's prototype (Appendix G.3), the category comes from the calibrated "
            "model and the explanation from the same SVM pipeline before calibration, using SHAP KernelExplainer with "
            "the 50-row background sample stored in the model bundle.")

    st.markdown("### Global feature importance")
    g1, g2 = st.columns([1.3, 1])
    with g1:
        st.plotly_chart(fig_shap_global(), use_container_width=True, config=PLOTLY_CONFIG, key="shap_global_res")
    with g2:
        st.markdown(
            "Across the study data, **age** and **waist-to-hip ratio** had by far the largest average influence on the "
            "model, followed by cigarettes per day. Daily or almost-daily alcohol had very little. These values come "
            f"from the supplied study results ({POPULATION}).")

    # ------------------------------------------------------------ next steps
    st.markdown("### Recommended next steps")
    with st.container(border=True):
        for line in NEXT_STEPS[a.category]:
            st.markdown(f"- {line}")
        st.markdown("- High blood pressure usually has **no symptoms**. Seek urgent care for severe headache, chest pain, "
                    "shortness of breath, weakness on one side or vision changes, whatever this tool says.")

    with st.expander("🔬  Model transparency: how this result was produced"):
        b = load_bundle()
        imp = imputed_values(b)
        folds = fold_feature_sets(b)
        st.markdown(
            f"- **Your inputs:** age {a.inputs['age']}, waist {a.inputs['waist_cm']:g} cm, hip {a.inputs['hip_cm']:g} cm "
            f"→ WHR {a.whr:.3f}, {a.inputs['cigarettes_per_day']} cigarettes/day, alcohol: {a.inputs['alcohol_frequency']}.\n"
            f"- **Scoring model:** {b['model_name']} wrapped in `CalibratedClassifierCV` (sigmoid / Platt, 5 folds). "
            f"The score is the average of the five fold models.\n"
            f"- **Threshold:** {a.threshold:.6f}, frozen on development data for ≥ 85% recall. Moderate band: "
            f"≥ {a.moderate_cutoff:.6f} (0.6 × threshold), as in Appendix G.3.\n"
            "- **Inputs not asked for** are passed as missing and filled by the imputer fitted inside the pipeline. "
            "In the full-data model these inputs are dropped by SelectKBest. However, SelectKBest was refitted inside "
            "each calibration fold, and some folds kept a different fourth or third feature:")
        for i, fs in enumerate(folds, 1):
            st.markdown(f"    - Fold {i}: `{'`, `'.join(f.split('__', 1)[1] for f in fs)}`")
        st.markdown(
            f"  For those folds the score uses the cohort's typical value: weight {imp['weight_kg']:.1f} kg, "
            f"resting pulse {imp['pulse_bpm']:.0f} bpm and education '{imp['education']}'. None of these is a "
            "blood-pressure measurement. This is a real limitation: if someone's actual weight or education differs "
            "markedly from these typical values, the score could be up to about 0.02 different. Near the threshold, "
            "that is enough to change the category, so treat results close to a cut-off with extra caution.\n"
            "- **Lower Risk in practice:** with this frozen model the calibrated score stays at about 0.035 or "
            f"above for every valid input in this app, above the Moderate cut-off of {a.moderate_cutoff:.3f}. The Lower Risk "
            "category is defined but not reached. This is a consequence of calibration compression, also seen in the "
            "study's test results, where 78.5% of participants were flagged.\n"
            "- **Privacy:** nothing you entered is stored, logged or sent anywhere. It exists only in this browser "
            "session.")

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("← Screen again", key="res_again", use_container_width=True):
            nav.go("screen")
    with c2:
        if st.button("Research analytics", key="res_analytics", use_container_width=True):
            nav.go("analytics")
    disclaimer(compact=True)
    footer()
