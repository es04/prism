"""
PRISM v2 – Dashboard Overview
Full redesign with new design system, dark mode, student drill-down shortcut.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
import theme


def _fig_base(fig: go.Figure, height: int = 310, **extra) -> go.Figure:
    dark = theme.is_dark()
    layout = theme.plotly_layout(dark)
    layout["height"] = height
    fig.update_layout(**layout)
    if extra:
        fig.update_layout(**extra)
    grid_col = "#1E2D4A" if dark else "#F1F5F9"
    fig.update_xaxes(gridcolor=grid_col, gridwidth=1, showline=False, zeroline=False)
    fig.update_yaxes(gridcolor=grid_col, gridwidth=1, showline=False, zeroline=False)
    return fig


def render() -> None:
    if not get_or_init_state():
        return

    pred_df = st.session_state["pred_df"]
    xm = st.session_state["xgb_metrics"]
    lm = st.session_state["lr_metrics"]
    feature_names = st.session_state["feature_names"]
    shap_values = st.session_state["shap_values"]
    dark = theme.is_dark()

    theme.page_header(
        "🏠",
        "Dashboard Overview",
        "High-level summary of student risk classifications and model performance.",
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    total = len(pred_df)
    high = int((pred_df["risk_label"] == "High Risk").sum())
    med = int((pred_df["risk_label"] == "Medium Risk").sum())
    low = int((pred_df["risk_label"] == "Low Risk").sum())
    avg_c = float(pred_df["xgb_prob"].mean())
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feat = feature_names[int(np.argmax(mean_abs))]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        theme.kpi_card("Total Students", f"{total:,}", accent="blue")
    with k2:
        theme.kpi_card(
            "High Risk",
            f"{high:,}",
            sub=f"{high/total*100:.1f}% of cohort",
            accent="red",
        )
    with k3:
        theme.kpi_card(
            "Medium Risk",
            f"{med:,}",
            sub=f"{med/total*100:.1f}% of cohort",
            accent="amber",
        )
    with k4:
        theme.kpi_card(
            "Low Risk", f"{low:,}", sub=f"{low/total*100:.1f}% of cohort", accent="teal"
        )
    with k5:
        theme.kpi_card("Avg Risk Score", f"{avg_c:.1f}%", accent="violet")
    with k6:
        theme.kpi_card(
            "Top Risk Factor",
            top_feat[:16],
            sub="Most predictive feature",
            accent="blue",
        )

    theme.section_div()

    # ── Row 1: Pie + bar ──────────────────────────────────────────────────────
    ca, cb = st.columns(2)

    with ca:
        counts = pred_df["risk_label"].value_counts().reset_index()
        counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(
            counts,
            names="Risk Level",
            values="Count",
            color="Risk Level",
            color_discrete_map=theme.RISK_COLORS,
            hole=0.5,
        )
        fig_pie.update_traces(
            textposition="outside",
            textinfo="percent+label",
            pull=[0.04, 0, 0],
            marker=dict(line=dict(color="#000", width=1)),
        )
        _fig_base(fig_pie, height=300, showlegend=False)
        theme.chart_card_header(
            "Risk Distribution", "Share of students in each risk tier"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with cb:
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        xv = [
            xm["accuracy"],
            xm["precision"],
            xm["recall"],
            xm["f1_score"],
            xm["roc_auc"],
        ]
        lv = [
            lm["accuracy"],
            lm["precision"],
            lm["recall"],
            lm["f1_score"],
            lm["roc_auc"],
        ]
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    name="XGBoost",
                    x=metrics,
                    y=xv,
                    marker_color=theme.LIGHT["blue"],
                    text=[f"{v*100:.1f}%" for v in xv],
                    textposition="outside",
                    marker_line_width=0,
                ),
                go.Bar(
                    name="Logistic Regression",
                    x=metrics,
                    y=lv,
                    marker_color="#93C5FD",
                    text=[f"{v*100:.1f}%" for v in lv],
                    textposition="outside",
                    marker_line_width=0,
                ),
            ]
        )
        _fig_base(
            fig_bar,
            height=300,
            barmode="group",
            yaxis=dict(range=[0, 1.18], tickformat=".0%"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        theme.chart_card_header(
            "Model Performance Comparison", "XGBoost vs Logistic Regression"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 2: stacked bar + histogram ───────────────────────────────────────
    cc, cd = st.columns(2)

    with cc:
        mr = (
            pred_df.groupby(["code_module", "risk_label"])
            .size()
            .reset_index(name="Count")
        )
        fig_mod = px.bar(
            mr,
            x="code_module",
            y="Count",
            color="risk_label",
            color_discrete_map=theme.RISK_COLORS,
            barmode="stack",
            labels={"code_module": "Module", "risk_label": "Risk Level"},
        )
        fig_mod.update_traces(marker_line_width=0)
        _fig_base(
            fig_mod,
            height=290,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        theme.chart_card_header("Risk by Module", "At-risk distribution across courses")
        st.plotly_chart(fig_mod, use_container_width=True)

    with cd:
        fig_hist = px.histogram(
            pred_df,
            x="xgb_prob",
            nbins=30,
            color_discrete_sequence=[theme.LIGHT["blue"]],
            labels={"xgb_prob": "Risk Score (%)"},
        )
        fig_hist.add_vline(
            x=65,
            line_color=theme.RISK_COLORS["High Risk"],
            line_dash="dash",
            line_width=1.5,
            annotation_text="High ≥65%",
            annotation_font_color=theme.RISK_COLORS["High Risk"],
        )
        fig_hist.add_vline(
            x=40,
            line_color=theme.RISK_COLORS["Medium Risk"],
            line_dash="dash",
            line_width=1.5,
            annotation_text="Medium ≥40%",
            annotation_font_color=theme.RISK_COLORS["Medium Risk"],
        )
        fig_hist.update_traces(marker_line_width=0)
        _fig_base(fig_hist, height=290, bargap=0.05, yaxis=dict(title="Students"))
        theme.chart_card_header(
            "Risk Score Distribution", "Confidence distribution across all students"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter ───────────────────────────────────────────────────────────────
    if "avg_score" in pred_df.columns and "total_clicks" in pred_df.columns:
        theme.section_div()
        scat_df = pred_df.dropna(subset=["avg_score", "total_clicks"])
        scat_sample = scat_df.sample(min(2000, len(scat_df)), random_state=42)
        fig_scat = px.scatter(
            scat_sample,
            x="avg_score",
            y="total_clicks",
            color="risk_label",
            color_discrete_map=theme.RISK_COLORS,
            opacity=0.5,
            labels={
                "avg_score": "Average Assessment Score",
                "total_clicks": "Total VLE Clicks",
            },
        )
        fig_scat.update_traces(marker=dict(size=4, line=dict(width=0)))
        _fig_base(
            fig_scat,
            height=340,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        theme.chart_card_header(
            "Assessment Score vs Online Engagement",
            "Each dot is one student — lower scores + fewer clicks often predict higher risk",
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── Prior attempts ────────────────────────────────────────────────────────
    if "num_of_prev_attempts" in pred_df.columns:
        attempt_risk = (
            pred_df.groupby("num_of_prev_attempts")
            .apply(lambda x: (x["risk_label"] == "High Risk").mean() * 100)
            .reset_index()
        )
        attempt_risk.columns = ["Previous Attempts", "High Risk Rate (%)"]
        fig_att = px.bar(
            attempt_risk,
            x="Previous Attempts",
            y="High Risk Rate (%)",
            color="High Risk Rate (%)",
            color_continuous_scale=[[0, "#FEF2F2"], [0.5, "#F87171"], [1, "#B91C1C"]],
        )
        fig_att.update_traces(marker_line_width=0)
        _fig_base(
            fig_att,
            height=260,
            coloraxis_showscale=False,
            yaxis=dict(ticksuffix="%"),
            xaxis=dict(tickmode="linear"),
        )
        theme.chart_card_header(
            "High-Risk Rate by Previous Attempts",
            "Repeat-course students tend to face higher dropout risk",
        )
        st.plotly_chart(fig_att, use_container_width=True)

    # ── Model performance summary ─────────────────────────────────────────────
    theme.section_div()
    st.markdown(
        f'<div style="font-size:1rem;font-weight:700;color:{"#F1F5F9" if dark else "#0F172A"};margin-bottom:0.75rem;">📌 XGBoost Performance vs Logistic Regression</div>',
        unsafe_allow_html=True,
    )

    perf_cols = st.columns(5)
    perf_data = [
        ("Accuracy", xm["accuracy"], lm["accuracy"]),
        ("Precision", xm["precision"], lm["precision"]),
        ("Recall", xm["recall"], lm["recall"]),
        ("F1 Score", xm["f1_score"], lm["f1_score"]),
        ("ROC-AUC", xm["roc_auc"], lm["roc_auc"]),
    ]
    for col, (label, xval, lval) in zip(perf_cols, perf_data):
        diff = (xval - lval) * 100
        sign = "+" if diff >= 0 else ""
        clr = "#10B981" if diff >= 0 else "#EF4444"
        with col:
            theme.kpi_card(
                label,
                f"{xval*100:.1f}%",
                sub=f'<span style="color:{clr};font-weight:700;">{sign}{diff:.1f}%</span> vs LR',
                accent="blue",
            )

    # ── Top 10 at-risk students ───────────────────────────────────────────────
    theme.section_div()
    st.markdown(
        f'<div style="font-size:1rem;font-weight:700;color:{"#F1F5F9" if dark else "#0F172A"};margin-bottom:0.6rem;">🚨 Top 10 Highest-Risk Students</div>',
        unsafe_allow_html=True,
    )

    top10 = pred_df.nlargest(10, "xgb_prob")[
        [
            "id_student",
            "code_module",
            "gender",
            "age_band",
            "risk_label",
            "xgb_prob",
            "final_result",
        ]
    ].copy()

    headers = [
        "Student ID",
        "Module",
        "Gender",
        "Age Band",
        "Risk Level",
        "Risk Score",
        "Actual Result",
        "",
    ]
    th = "padding:10px 14px;text-align:left;font-weight:700;color:var(--text2);font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;"
    td_base = "padding:10px 14px;"

    rows_html = ""
    for _, r in top10.iterrows():
        sid = int(r["id_student"])
        rows_html += f"""<tr style="border-bottom:1px solid var(--border);">
            <td style="{td_base}font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--text2);">{sid}</td>
            <td style="{td_base}">{r['code_module']}</td>
            <td style="{td_base}">{r['gender']}</td>
            <td style="{td_base}">{r['age_band']}</td>
            <td style="{td_base}">{theme.badge_html(r['risk_label'])}</td>
            <td style="{td_base}font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--red);">{r['xgb_prob']:.1f}%</td>
            <td style="{td_base}">{r['final_result']}</td>
            <td style="{td_base}"><span style="font-size:0.72rem;color:var(--blue);cursor:pointer;"
                onclick="window.parent.postMessage({{type:'streamlit:componentCommunication',key:'profile_student',value:{sid}}}, '*')">View →</span></td>
        </tr>"""

    hdr_html = "".join(f'<th style="{th}">{h}</th>' for h in headers)
    st.markdown(
        f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);
            overflow:auto;box-shadow:var(--shadow);">
    <table class="prism-table">
        <thead><tr style="background:var(--surface2);">{hdr_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
<div style="margin-top:0.5rem;font-size:0.78rem;color:var(--text3);">
    💡 Tip: Go to <strong>Predictions</strong> and click a student row to see their full profile.
</div>
""",
        unsafe_allow_html=True,
    )
