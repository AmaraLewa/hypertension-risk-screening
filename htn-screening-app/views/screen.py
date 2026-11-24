import streamlit as st

from htn import nav
from htn.model import ALCOHOL_OPTIONS, ValidationError, assess, calculate_whr, load_bundle
from htn.profiles import DEMO_PROFILES
from htn.ui import disclaimer, footer, page_header

KEYS = {"age": "in_age", "waist_cm": "in_waist", "hip_cm": "in_hip",
        "cigarettes_per_day": "in_cigs", "alcohol_frequency": "in_alcohol"}


def _load_profile(values: dict, name: str) -> None:
    for field, key in KEYS.items():
        st.session_state[key] = values[field]
    st.session_state["demo_profile"] = name


def _clear() -> None:
    for key in KEYS.values():
        st.session_state[key] = None
    st.session_state.pop("demo_profile", None)
    st.session_state.pop("assessment", None)


def _run(raw: dict, source: str) -> bool:
    try:
        with st.spinner("Running the trained model and computing SHAP values…"):
            result = assess(raw, load_bundle())
    except ValidationError as err:
        st.session_state["form_errors"] = err.messages
        return False
    st.session_state.pop("form_errors", None)
    st.session_state["assessment"] = result
    st.session_state["assessment_source"] = source
    return True


def render():
    page_header("Screen for risk",
                "Answer five questions. The trained model uses age, waist-to-hip ratio, cigarettes per day and "
                "whether you drink alcohol daily or almost daily.")
    disclaimer()

    for key in KEYS.values():
        st.session_state.setdefault(key, None)

    # ---------------------------------------------------------- demo mode
    with st.expander("🎬  Demonstration mode — three synthetic sample profiles", expanded=False):
        st.markdown(
            "These profiles are **invented, illustrative inputs**. They are not NHANES participants and were not "
            "taken from the study data. Choosing one fills the form; the category and explanation are then "
            "produced by the trained model, not assigned by hand.")
        for i, (name, values) in enumerate(DEMO_PROFILES.items()):
            c1, c2, c3 = st.columns([3.2, 1, 1])
            with c1:
                st.markdown(
                    f"**{name}**  \n"
                    f"<span class='hrs-pill'>Age {values['age']}</span>"
                    f"<span class='hrs-pill'>Waist {values['waist_cm']:g} cm</span>"
                    f"<span class='hrs-pill'>Hip {values['hip_cm']:g} cm</span>"
                    f"<span class='hrs-pill'>{values['cigarettes_per_day']} cig/day</span>"
                    f"<span class='hrs-pill'>Alcohol: {values['alcohol_frequency']}</span>",
                    unsafe_allow_html=True)
            with c2:
                st.button("Fill form", key=f"demo_fill_{i}", on_click=_load_profile, args=(values, name),
                          use_container_width=True)
            with c3:
                if st.button("Run model", key=f"demo_run_{i}", type="primary", use_container_width=True):
                    _load_profile(values, name)
                    if _run(values, f"Synthetic demonstration profile: {name}"):
                        nav.go("results")

    if st.session_state.get("demo_profile"):
        st.info(f"Form filled with **{st.session_state['demo_profile']}** (synthetic, illustrative only). "
                "You can edit any value before running the assessment.")

    # ---------------------------------------------------------- form
    with st.container(border=True):
        st.markdown("#### About you")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Age (years)", min_value=0, max_value=120, step=1, key=KEYS["age"],
                            placeholder="18–40", help="The model was developed for adults aged 18 to 40.")
        with c2:
            st.caption(" ")
            st.caption("Eligible range: **18–40 years**, the age range the model was developed for.")

        st.markdown("#### Body measurements")
        c3, c4, c5 = st.columns(3)
        with c3:
            st.number_input("Waist circumference (cm)", min_value=0.0, max_value=300.0, step=0.5,
                            key=KEYS["waist_cm"], placeholder="e.g. 85",
                            help="Measured midway between the lowest rib and the top of the hip bone, after breathing out.")
        with c4:
            st.number_input("Hip circumference (cm)", min_value=0.0, max_value=300.0, step=0.5,
                            key=KEYS["hip_cm"], placeholder="e.g. 100",
                            help="Measured around the widest part of the buttocks.")
        with c5:
            w, h = st.session_state.get(KEYS["waist_cm"]), st.session_state.get(KEYS["hip_cm"])
            if w and h:
                st.metric("Waist-to-hip ratio (calculated)", f"{calculate_whr(w, h):.3f}")
            else:
                st.metric("Waist-to-hip ratio (calculated)", "—")
                st.caption("Enter waist and hip to calculate.")

        st.markdown("#### Lifestyle")
        c6, c7 = st.columns(2)
        with c6:
            st.number_input("Cigarettes smoked per day", min_value=0, max_value=200, step=1,
                            key=KEYS["cigarettes_per_day"], placeholder="0 if you do not smoke",
                            help="Average number of cigarettes on the days you smoke. Enter 0 if you do not smoke.")
        with c7:
            st.selectbox("How often do you drink alcohol?", ALCOHOL_OPTIONS, index=None,
                         key=KEYS["alcohol_frequency"], placeholder="Choose an option")
            st.caption("The model distinguishes only *daily or almost daily* drinking from all other answers; "
                       "the other options do not change its output.")

        run = st.button("Run screening assessment", type="primary", use_container_width=True, key="run_assessment")

    if run:
        raw = {field: st.session_state.get(key) for field, key in KEYS.items()}
        source = (f"Synthetic demonstration profile: {st.session_state['demo_profile']}"
                  if st.session_state.get("demo_profile") else "Values entered in the form")
        if _run(raw, source):
            nav.go("results")

    if st.session_state.get("form_errors"):
        st.error("**Please correct the following before running the assessment:**\n\n" +
                 "\n".join(f"- {m}" for m in st.session_state["form_errors"]))

    st.button("Clear form", on_click=_clear, key="clear_form")

    with st.expander("Why are only these questions asked?"):
        st.markdown(
            "The study's pipeline caps the model at four encoded parameters, the number its 34 development events "
            "can support. SelectKBest kept **age, waist-to-hip ratio, cigarettes per day and daily-or-almost-daily "
            "alcohol**, so those are the only answers that drive the explanation. Waist and hip are collected "
            "only to calculate WHR. (The calibrated scorer averages five fold models, and a few of them kept weight, "
            "resting pulse or education instead; those are held at the cohort's typical value. See *Model "
            "transparency* on the results page.)\n\n"
            "**Blood-pressure readings, previous diagnoses and blood-pressure medication are never asked for.** "
            "They define the outcome, so the study's leakage guard excluded them from every model by design.")
    footer()
