"""End-to-end UI tests using Streamlit's AppTest. Run: python -m pytest -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from htn.ui import DISCLAIMER

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DISC_START = "This application is a research prototype, not a diagnostic tool."


def html_text(at):
    return " ".join(m.value for m in at.markdown)


def new(page=None):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, at.exception
    return at


def test_disclaimer_text_exact():
    assert DISCLAIMER.startswith(DISC_START)
    assert DISCLAIMER.endswith("Confirmatory blood-pressure measurement by a qualified health professional is required.")


@pytest.mark.parametrize("page", [None, "views/screen.py", "views/results.py", "views/analytics.py", "views/about.py"])
def test_every_page_renders_with_disclaimer(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert DISC_START in html_text(at)


def _goto(at, title):
    at.switch_page({"Screen for Risk": "screen", "Assessment Results": "results", "Research Analytics": "analytics",
                    "About the Project": "about"}[title])


def run_nav(title):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    # Navigate through the page's url_path
    at.query_params["page"] = title
    at.run()
    return at


def screen_app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.button(key="home_start").click().run()
    assert not at.exception, at.exception
    return at


def test_home_has_title_and_start_button():
    at = AppTest.from_file(APP, default_timeout=60).run()
    text = html_text(at)
    assert "Hypertension Risk Screening System" in text
    assert "An Explainable AI Approach to Early Elevated Blood-Pressure Screening" in text
    assert at.button(key="home_start").label.startswith("Start Screening")


def test_screen_form_fields_are_only_model_inputs():
    at = screen_app()
    labels = [w.label for w in at.number_input] + [w.label for w in at.selectbox]
    assert labels == ["Age (years)", "Waist circumference (cm)", "Hip circumference (cm)",
                      "Cigarettes smoked per day", "How often do you drink alcohol?"]
    lowered = " ".join(labels).lower()
    for banned in ["systolic", "diastolic", "blood pressure", "medication", "diagnos"]:
        assert banned not in lowered
    assert DISC_START in html_text(at)


def test_empty_form_shows_errors():
    at = screen_app()
    at.button(key="run_assessment").click().run()
    assert not at.exception
    assert at.error and "Age is required" in at.error[0].value
    assert "assessment" not in at.session_state


def test_out_of_range_age_shows_error():
    at = screen_app()
    at.number_input(key="in_age").set_value(45)
    at.number_input(key="in_waist").set_value(90.0)
    at.number_input(key="in_hip").set_value(100.0)
    at.number_input(key="in_cigs").set_value(0)
    at.selectbox(key="in_alcohol").select("Never")
    at.button(key="run_assessment").click().run()
    assert "between 18 and 40" in at.error[0].value


def test_whr_displayed_live():
    at = screen_app()
    at.number_input(key="in_waist").set_value(81.0)
    at.number_input(key="in_hip").set_value(90.0).run()
    assert at.metric[0].value == "0.900"


def test_end_to_end_manual_screening():
    at = screen_app()
    at.number_input(key="in_age").set_value(34)
    at.number_input(key="in_waist").set_value(96.0)
    at.number_input(key="in_hip").set_value(102.0)
    at.number_input(key="in_cigs").set_value(4)
    at.selectbox(key="in_alcohol").select("1–2 times a week")
    at.button(key="run_assessment").click().run()
    assert not at.exception, at.exception
    a = at.session_state["assessment"]
    assert a.category in {"Lower Risk", "Moderate Risk", "Higher Risk"}
    text = html_text(at)
    assert "Screening category" in text and a.category in text
    assert DISC_START in text                                   # disclaimer on results
    assert any("not a clinically validated probability" in w.value for w in at.warning)
    assert "What drove this assessment" in text
    assert "Recommended next steps" in text
    assert len(at.get("plotly_chart")) >= 3                     # gauge, local SHAP, global SHAP


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_demo_profiles_run_model_end_to_end(idx):
    from htn.model import assess
    from htn.profiles import DEMO_PROFILES

    at = screen_app()
    at.button(key=f"demo_run_{idx}").click().run()
    assert not at.exception, at.exception
    a = at.session_state["assessment"]
    raw = list(DEMO_PROFILES.values())[idx]
    assert a.score == pytest.approx(assess(raw, explain=False).score)
    assert "Synthetic demonstration profile" in at.session_state["assessment_source"]
    assert DISC_START in html_text(at)


def test_demo_fill_populates_form():
    at = screen_app()
    at.button(key="demo_fill_2").click().run()
    assert at.number_input(key="in_age").value == 37
    assert at.selectbox(key="in_alcohol").value == "Daily or almost daily"


def test_research_analytics_renders_all_sections():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="home_research").click().run()
    assert not at.exception, at.exception
    tabs = [t.label for t in at.tabs]
    for expected in ["Model comparison", "Classification", "Calibration", "Decision curve", "SHAP importance",
                     "Subgroups", "Leakage guard"]:
        assert expected in tabs
    assert len(at.get("plotly_chart")) >= 10
    text = html_text(at)
    assert "Non-Hispanic Black adults aged 18–40" in text
    assert DISC_START in text
