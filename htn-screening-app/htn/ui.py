"""Shared styling, colours and UI components."""
from __future__ import annotations

import html

import streamlit as st

NAVY = "#0B1F3A"
NAVY_2 = "#16325C"
TEAL = "#0F9D9A"
TEAL_LIGHT = "#E6F6F5"
BLUE = "#3B5BDB"
CORAL = "#E4572E"
INK = "#0B1F3A"
INK_2 = "#475569"
MUTED = "#64748B"
GRID = "#E2E8F0"
SURFACE = "#FFFFFF"

# Status colours are reserved for the screening category and always ship with
# an icon and a text label (never colour alone).
CATEGORY_STYLE = {
    "Lower Risk": {"color": "#1F8A5B", "bg": "#E8F6EF", "icon": "●", "word": "Lower"},
    "Moderate Risk": {"color": "#B7791F", "bg": "#FDF4E3", "icon": "▲", "word": "Moderate"},
    "Higher Risk": {"color": "#C8323E", "bg": "#FCECEE", "icon": "◆", "word": "Higher"},
}

DISCLAIMER = (
    "This application is a research prototype, not a diagnostic tool. It has been developed "
    "using US NHANES survey data and has not been externally validated in a Nigerian population. "
    "Its output must not be used to diagnose hypertension, prescribe medication or make treatment "
    "decisions. Confirmatory blood-pressure measurement by a qualified health professional is required."
)

POPULATION = "Non-Hispanic Black adults aged 18–40, US NHANES August 2021–August 2023"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label, input, button, textarea, select {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
h1, h2, h3, h4, .hrs-display {{ font-family: 'Manrope', 'Inter', sans-serif !important; color: {NAVY}; letter-spacing: -0.01em; }}
.stApp {{ background: linear-gradient(180deg, #F5F8FC 0%, #FFFFFF 340px); }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1180px; }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: {NAVY}; }}
[data-testid="stHeader"] a, [data-testid="stHeader"] span, [data-testid="stHeader"] p {{ color: #E2E8F0 !important; }}
[data-testid="stHeader"] [aria-current="page"] p, [data-testid="stHeader"] [aria-current="page"] span {{ color: #FFFFFF !important; font-weight: 700; }}

/* Buttons */
.stButton > button, .stFormSubmitButton > button, [data-testid="stPageLink"] a {{
  border-radius: 12px; font-weight: 600; padding: 0.55rem 1.1rem; transition: all .15s ease;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background: {TEAL}; border: 1px solid {TEAL}; color: #fff;
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
  background: #0B827F; border-color: #0B827F;
}}
.stButton > button[kind="secondary"] {{ border: 1px solid #CBD5E1; color: {NAVY}; background:#fff; }}
.stButton > button[kind="secondary"]:hover {{ border-color: {TEAL}; color: {TEAL}; }}

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, .stNumberInput input {{ border-radius: 10px !important; }}

/* Bordered containers as cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 16px; }}

/* Hero */
.hrs-hero {{
  background: radial-gradient(1200px 400px at 85% -10%, rgba(15,157,154,.35), transparent 60%),
              linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 100%);
  color: #fff; border-radius: 22px; padding: 2.6rem 2.4rem; margin-bottom: 1.4rem;
  box-shadow: 0 20px 40px rgba(11,31,58,.18);
}}
.hrs-hero h1 {{ color: #fff !important; font-size: clamp(1.8rem, 3.6vw, 2.7rem); line-height: 1.12; margin: .4rem 0 .6rem; }}
.hrs-hero p {{ color: #CFE3F5; font-size: 1.06rem; max-width: 720px; margin: 0; }}
.hrs-eyebrow {{ display:inline-block; background: rgba(15,157,154,.22); color:#9FF0EC; border:1px solid rgba(159,240,236,.35);
  padding: .25rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }}

/* Cards */
.hrs-card {{ background:#fff; border:1px solid {GRID}; border-radius:16px; padding:1.15rem 1.2rem; height:100%;
  box-shadow: 0 1px 2px rgba(11,31,58,.04), 0 6px 18px rgba(11,31,58,.04); }}
.hrs-card h4 {{ margin: .35rem 0 .35rem; font-size: 1.02rem; }}
.hrs-card p {{ color:{INK_2}; font-size:.93rem; margin:0; line-height:1.5; }}
.hrs-icon {{ width:38px; height:38px; border-radius:11px; display:flex; align-items:center; justify-content:center;
  background:{TEAL_LIGHT}; color:{TEAL}; font-size:1.15rem; }}

/* KPI tiles */
.hrs-kpi {{ background:#fff; border:1px solid {GRID}; border-radius:14px; padding: .9rem 1rem; height:100%; }}
.hrs-kpi .lab {{ color:{MUTED}; font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.hrs-kpi .val {{ font-family:'Manrope',sans-serif; color:{NAVY}; font-size:1.55rem; font-weight:800; line-height:1.2; margin-top:.15rem; }}
.hrs-kpi .sub {{ color:{MUTED}; font-size:.8rem; margin-top:.15rem; }}

/* Disclaimer */
.hrs-disclaimer {{ display:flex; gap:.8rem; align-items:flex-start; background:#FFF8E6; border:1px solid #F2D38A;
  border-left: 5px solid #D99A00; border-radius:14px; padding: .9rem 1.05rem; margin: .2rem 0 1.1rem; color:#5B4300; font-size:.92rem; line-height:1.5; }}
.hrs-disclaimer b {{ color:#3D2C00; }}
.hrs-disclaimer .ico {{ font-size:1.2rem; line-height:1.2; }}

/* Category banner */
.hrs-cat {{ border-radius:18px; padding:1.3rem 1.4rem; border:1px solid; }}
.hrs-cat .k {{ font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; opacity:.85; }}
.hrs-cat .v {{ font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; line-height:1.15; margin:.15rem 0 .35rem; }}
.hrs-cat p {{ margin:0; color:{INK_2}; font-size:.95rem; }}

.hrs-pill {{ display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid {GRID}; color:{INK_2}; background:#F8FAFC; margin-right:.3rem; }}
.hrs-section-title {{ margin: 1.6rem 0 .2rem; }}
.hrs-lead {{ color:{INK_2}; font-size:1rem; margin-bottom:.8rem; }}
.hrs-footer {{ color:{MUTED}; font-size:.8rem; border-top:1px solid {GRID}; padding-top:.9rem; margin-top:2.2rem; }}
.hrs-factor {{ display:flex; justify-content:space-between; align-items:center; gap:.6rem; padding:.55rem .2rem; border-bottom:1px dashed {GRID}; }}
.hrs-factor:last-child {{ border-bottom:none; }}
.hrs-factor .name {{ font-weight:600; color:{NAVY}; }}
.hrs-factor .meta {{ color:{MUTED}; font-size:.82rem; }}
.hrs-up {{ color:#B23A17; font-weight:700; white-space:nowrap; }}
.hrs-down {{ color:#0B7471; font-weight:700; white-space:nowrap; }}

[data-testid="stTabs"] button p {{ font-weight:600; }}
[data-testid="stMetricValue"] {{ font-family:'Manrope',sans-serif; color:{NAVY}; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: .9rem; padding-right: .9rem; padding-top: 3.8rem; }}
  .hrs-hero {{ padding: 1.6rem 1.25rem; border-radius: 18px; }}
  .hrs-cat .v {{ font-size: 1.6rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def disclaimer(compact: bool = False) -> None:
    title = "Research prototype — not a diagnostic tool." if not compact else "Not a diagnostic tool."
    st.markdown(
        f'<div class="hrs-disclaimer" role="note" aria-label="Medical disclaimer">'
        f'<span class="ico">⚠️</span><div><b>{title}</b> {html.escape(DISCLAIMER)}</div></div>',
        unsafe_allow_html=True,
    )


def card(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="hrs-card"><div class="hrs-icon">{icon}</div>'
        f'<h4>{html.escape(title)}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="hrs-kpi"><div class="lab">{html.escape(label)}</div>'
        f'<div class="val">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, lead: str = "") -> None:
    st.markdown(f"## {title}")
    if lead:
        st.markdown(f'<div class="hrs-lead">{lead}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        '<div class="hrs-footer">Research prototype developed for the Professional Master of Information '
        "Technology project, Department of Information Technology, School of Computing, Miva Open "
        "University, Abuja. Not a medical device. No information entered here is stored, logged or "
        "transmitted; inputs exist only in your browser session and are cleared when you close the tab.</div>",
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height: int = 360, **kw):
    kw.setdefault("margin", dict(l=10, r=10, t=40, b=10))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        title_font=dict(family="Manrope, Inter, sans-serif", size=15, color=NAVY),
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#CBD5E1", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#CBD5E1", linecolor="#CBD5E1")
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
