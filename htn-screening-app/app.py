"""
Hypertension Risk Screening System
An Explainable AI Approach to Early Elevated Blood-Pressure Screening

Streamlit entry point. Run locally with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Hypertension Risk Screening System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from htn import nav  # noqa: E402
from htn.ui import inject_css  # noqa: E402
from views import about, analytics, home, results, screen  # noqa: E402

inject_css()

nav.PAGES["home"] = st.Page(home.render, title="Home", icon=":material/home:", default=True)
nav.PAGES["screen"] = st.Page(screen.render, title="Screen for Risk", icon=":material/assignment:", url_path="screen")
nav.PAGES["results"] = st.Page(results.render, title="Assessment Results", icon=":material/monitor_heart:", url_path="results")
nav.PAGES["analytics"] = st.Page(analytics.render, title="Research Analytics", icon=":material/insights:", url_path="analytics")
nav.PAGES["about"] = st.Page(about.render, title="About the Project", icon=":material/school:", url_path="about")

pg = st.navigation(list(nav.PAGES.values()), position="top")
pg.run()
