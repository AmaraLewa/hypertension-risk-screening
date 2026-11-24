"""Loaders and charts for the supplied research results (no values are recomputed or altered)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .ui import BLUE, CORAL, GRID, INK_2, MUTED, NAVY, TEAL, plotly_layout

RESULTS = Path(__file__).resolve().parent.parent / "research_results"


@lru_cache(maxsize=None)
def run_summary() -> dict:
    return json.loads((RESULTS / "run_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def leakage_tests() -> dict:
    return json.loads((RESULTS / "leakage_test_results.json").read_text(encoding="utf-8"))


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@lru_cache(maxsize=None)
def model_comparison() -> pd.DataFrame:
    df = _csv("model_comparison.csv")
    df["is_baseline"] = df["is_baseline"].astype(str).str.lower().eq("true")
    return df


@lru_cache(maxsize=None)
def decision_curve() -> pd.DataFrame:
    return _csv("decision_curve.csv")


@lru_cache(maxsize=None)
def subgroups() -> pd.DataFrame:
    return _csv("fairness_by_subgroup.csv")


@lru_cache(maxsize=None)
def shap_global() -> pd.DataFrame:
    df = _csv("shap_importance_by_variable.csv")
    labels = {"age": "Age", "whr": "Waist-to-hip ratio", "cigarettes_per_day": "Cigarettes per day",
              "alcohol_frequency": "Alcohol: daily or almost daily"}
    df["label"] = df["variable"].map(labels).fillna(df["variable"])
    return df.sort_values("mean_abs_shap", ascending=True)


# ------------------------------------------------------------------ charts
def _short(name: str) -> str:
    return {"Clinical baseline (age, sex, BMI)": "Clinical baseline<br>(age, sex, BMI)",
            "Support Vector Machine": "Support Vector<br>Machine",
            "Elastic Net Logistic": "Elastic Net<br>Logistic",
            "k-Nearest Neighbours": "k-Nearest<br>Neighbours",
            "Logistic Regression": "Logistic<br>Regression"}.get(name, name)


def fig_nested_metric(metric: str, sd: str | None, title: str, ref: float | None = None, ref_label: str = ""):
    df = model_comparison().sort_values(metric, ascending=True)
    selected = run_summary()["selected_model"]
    colors = [TEAL if m == selected else (BLUE if b else "#94A3B8") for m, b in zip(df["model"], df["is_baseline"])]
    fig = go.Figure(go.Bar(
        y=df["model"], x=df[metric], orientation="h",
        error_x=dict(type="data", array=df[sd], color="#64748B", thickness=1.2, width=4) if sd else None,
        marker=dict(color=colors, cornerradius=4), width=0.6,
        text=None if sd else [f"{v:.3f}" for v in df[metric]], textposition="inside", insidetextanchor="end",
        textfont=dict(color="white", size=11),
        customdata=df[["model"] + ([sd] if sd else [])].values,
        hovertemplate="<b>%{customdata[0]}</b><br>" + title + ": %{x:.3f}" +
                      ("<br>SD: %{customdata[1]:.3f}" if sd else "") + "<extra></extra>",
    ))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dot", line_color=CORAL, line_width=2)
        fig.add_annotation(x=ref, y=1.0, yref="paper", yanchor="bottom", xanchor="left", text=" " + ref_label,
                           showarrow=False, font=dict(color=INK_2, size=11))
    fig.update_xaxes(rangemode="tozero")
    return plotly_layout(fig, height=380, title=title, showlegend=False, margin=dict(l=10, r=20, t=60, b=10))


def fig_confusion(tp: int, fp: int, tn: int, fn: int, title: str):
    z = [[tp, fn], [fp, tn]]
    text = [[f"TP<br><b>{tp}</b>", f"FN<br><b>{fn}</b>"], [f"FP<br><b>{fp}</b>", f"TN<br><b>{tn}</b>"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Flagged (predicted elevated)", "Not flagged"], y=["Elevated BP (actual)", "Not elevated (actual)"],
        colorscale=[[0, "#EEF6F6"], [1, "#0F9D9A"]], showscale=False, text=text, texttemplate="%{text}",
        textfont=dict(size=15, color=NAVY), xgap=3, ygap=3,
        hovertemplate="%{y} · %{x}: %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return plotly_layout(fig, height=320, title=title)


def fig_calibration_compare():
    t = run_summary()["test_metrics"]
    cats = ["Before calibration (SVM)", "After Platt calibration"]
    fig = go.Figure()
    fig.add_bar(name="Calibration slope", x=cats, y=[t["uncalibrated_slope"], t["calibration_slope"]],
                marker=dict(color=TEAL, cornerradius=4), width=0.45,
                text=[f"{t['uncalibrated_slope']:.2f}", f"{t['calibration_slope']:.2f}"], textposition="outside",
                hovertemplate="%{x}<br>Slope: %{y:.3f}<extra></extra>")
    fig.add_hline(y=1, line_dash="dot", line_color=CORAL, line_width=2, annotation_text="Ideal slope = 1.0", annotation_position="top left",
                  annotation_font=dict(color=INK_2, size=11))
    fig.update_yaxes(title="Calibration slope (test partition)", rangemode="tozero")
    return plotly_layout(fig, height=330, title="Calibration slope, sealed test partition", showlegend=False)


def fig_brier_compare():
    t = run_summary()["test_metrics"]
    cats = ["Before calibration (SVM)", "After Platt calibration"]
    fig = go.Figure(go.Bar(x=cats, y=[t["uncalibrated_brier"], t["brier"]], marker=dict(color=BLUE, cornerradius=4),
                           width=0.45, text=[f"{t['uncalibrated_brier']:.3f}", f"{t['brier']:.3f}"],
                           textposition="outside", hovertemplate="%{x}<br>Brier: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title="Brier score (lower is better)", rangemode="tozero")
    return plotly_layout(fig, height=330, title="Brier score, sealed test partition", showlegend=False)


def fig_decision_curve():
    dc = decision_curve()
    rs = run_summary()["test_metrics"]
    lo, hi = rs["benefit_band"]
    fig = go.Figure()
    fig.add_vrect(x0=lo, x1=hi, fillcolor=TEAL, opacity=0.08, line_width=0,
                  annotation_text="Positive net benefit band", annotation_position="top left",
                  annotation_font=dict(size=11, color=INK_2))
    fig.add_scatter(x=dc["threshold"], y=dc["net_benefit_model"], name="Selected model (SVM, calibrated)",
                    mode="lines", line=dict(color=TEAL, width=2.5),
                    hovertemplate="Threshold %{x:.2f}<br>Model: %{y:.4f}<extra></extra>")
    fig.add_scatter(x=dc["threshold"], y=dc["net_benefit_screen_all"], name="Screen everyone",
                    mode="lines", line=dict(color=BLUE, width=2, dash="dash"),
                    hovertemplate="Threshold %{x:.2f}<br>Screen all: %{y:.4f}<extra></extra>")
    fig.add_scatter(x=dc["threshold"], y=dc["net_benefit_screen_none"], name="Screen no one",
                    mode="lines", line=dict(color="#94A3B8", width=2, dash="dot"),
                    hovertemplate="Threshold %{x:.2f}<br>Screen none: %{y:.4f}<extra></extra>")
    thr = rs["threshold"]
    fig.add_vline(x=thr, line_color=CORAL, line_width=2, line_dash="dot",
                  annotation_text=f"Operating threshold {thr:.4f}", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=INK_2))
    fig.update_xaxes(title="Threshold probability", range=[0, 0.2])
    fig.update_yaxes(title="Net benefit", range=[-0.02, 0.05])
    fig = plotly_layout(fig, height=460, title="Decision curve analysis (sealed test partition)")
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig


def fig_shap_global():
    df = shap_global()
    fig = go.Figure(go.Bar(x=df["mean_abs_shap"], y=df["label"], orientation="h",
                           marker=dict(color=TEAL, cornerradius=4), width=0.55,
                           text=[f"{v:.4f}" for v in df["mean_abs_shap"]], textposition="outside",
                           hovertemplate="%{y}<br>Mean |SHAP|: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Mean |SHAP value|", range=[0, df["mean_abs_shap"].max() * 1.45])
    return plotly_layout(fig, height=300, title="Global feature importance (mean absolute SHAP)", showlegend=False)


def fig_subgroups(metric: str, label: str):
    df = subgroups().copy()
    df["name"] = df["group"].map({"sex": "Sex", "age_band": "Age band"}) + ": " + df["level"].astype(str)
    colors = [TEAL if g == "sex" else BLUE for g in df["group"]]
    fig = go.Figure(go.Bar(x=df["name"], y=df[metric], marker=dict(color=colors, cornerradius=4), width=0.5,
                           customdata=df[["n", "tp", "fn"]].values,
                           text=[f"{v:.3f}" for v in df[metric]], textposition="outside",
                           hovertemplate="%{x}<br>" + label + ": %{y:.3f}<br>n = %{customdata[0]} "
                                         "(events = %{customdata[1]})<extra></extra>"))
    fig.update_yaxes(title=label, range=[0, 1.12])
    return plotly_layout(fig, height=340, title=f"{label} by subgroup (test partition)", showlegend=False)
