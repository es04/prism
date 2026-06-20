"""
PRISM v2 – Student Risk Predictions
Redesigned with advanced filters, search, Export to PDF/Excel, student profile drill-down.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
import theme


# ── Student Profile Panel ─────────────────────────────────────────────────────

def _render_student_profile(student_row: pd.Series, pred_df: pd.DataFrame):
    """Full student detail drill-down."""
    dark = theme.is_dark()
    sid  = int(student_row["id_student"])
    risk = student_row["risk_label"]
    prob = float(student_row["xgb_prob"])

    # Back button
    if st.button("← Back to Predictions Table", key="back_from_profile"):
        st.session_state.pop("profile_student_id", None)
        st.rerun()

    # Header card
    risk_color = {"High Risk": "#DC2626", "Medium Risk": "#D97706", "Low Risk": "#0D9488"}.get(risk, "#64748B")
    st.markdown(f"""
<div class="profile-header">
    <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <div style="width:52px;height:52px;border-radius:50%;background:rgba(255,255,255,0.08);
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;">
            {"🔴" if risk=="High Risk" else "🟡" if risk=="Medium Risk" else "🟢"}
        </div>
        <div>
            <div style="font-size:1.4rem;font-weight:800;letter-spacing:-0.01em;">
                Student #{sid}
            </div>
            <div style="font-size:0.85rem;opacity:0.7;margin-top:3px;">
                {student_row.get('code_module','N/A')} · {student_row.get('gender','N/A')} · {student_row.get('age_band','N/A')}
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:2rem;font-weight:900;font-family:'JetBrains Mono',monospace;color:{risk_color};">
                {prob:.1f}%
            </div>
            <div style="font-size:0.75rem;opacity:0.6;">XGBoost Risk Score</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Stat grid
    stat_fields = [
        ("Module",           student_row.get("code_module", "N/A")),
        ("Gender",           student_row.get("gender", "N/A")),
        ("Age Band",         student_row.get("age_band", "N/A")),
        ("Education",        student_row.get("highest_education", "N/A")),
        ("IMD Band",         student_row.get("imd_band", "N/A")),
        ("Disability",       student_row.get("disability", "N/A")),
        ("Prev Attempts",    student_row.get("num_of_prev_attempts", "N/A")),
        ("Credits",          student_row.get("studied_credits", "N/A")),
        ("VLE Clicks",       f"{int(student_row['total_clicks']):,}" if "total_clicks" in student_row and pd.notna(student_row.get("total_clicks")) else "N/A"),
        ("Avg Score",        f"{student_row['avg_score']:.1f}%" if "avg_score" in student_row and pd.notna(student_row.get("avg_score")) else "N/A"),
        ("Actual Result",    student_row.get("final_result", "N/A")),
        ("Risk Level",       risk),
    ]
    cols = st.columns(4)
    for i, (label, val) in enumerate(stat_fields):
        with cols[i % 4]:
            display_val = theme.badge_html(str(val)) if label == "Risk Level" else f'<div class="profile-stat-value">{val}</div>'
            st.markdown(f"""
<div class="profile-stat" style="margin-bottom:0.5rem;">
    <div class="profile-stat-label">{label}</div>
    {display_val if label=="Risk Level" else f'<div class="profile-stat-value">{val}</div>'}
</div>
""", unsafe_allow_html=True)

    theme.section_div()

    # Risk gauge
    st.markdown("#### Risk Score Breakdown")
    gauge_col, info_col = st.columns([1, 1])

    with gauge_col:
        fig_gauge = px.bar(
            pd.DataFrame({"Model": ["XGBoost", "Log. Regression"],
                          "Risk %": [prob, float(student_row.get("lr_prob", 0))]}),
            x="Model", y="Risk %", color="Model",
            color_discrete_sequence=[theme.RISK_COLORS.get(risk, "#64748B"), "#93C5FD"],
            range_y=[0, 100],
        )
        dark = theme.is_dark()
        fig_gauge.update_traces(marker_line_width=0)
        layout = theme.plotly_layout(dark)
        layout.update({"height": 240, "showlegend": False,
                       "yaxis": {"range": [0, 100], "ticksuffix": "%"}})
        fig_gauge.update_layout(**layout)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with info_col:
        thresholds = {"High Risk": (65, 100), "Medium Risk": (40, 65), "Low Risk": (0, 40)}
        lo, hi = thresholds.get(risk, (0, 100))
        theme.info_box(f"""
<strong>Risk tier: {risk}</strong><br>
Threshold: {lo}% – {hi}%<br>
XGBoost confidence: <strong>{prob:.2f}%</strong><br>
LR confidence: <strong>{student_row.get('lr_prob', 0):.2f}%</strong>
""", "warn" if risk == "High Risk" else "info")

        if risk == "High Risk":
            theme.info_box("⚠️ <strong>Early intervention recommended.</strong> This student is at high risk of withdrawal or failure.", "error")
        elif risk == "Medium Risk":
            theme.info_box("🔔 <strong>Monitor closely.</strong> This student shows signs of disengagement that may lead to risk.", "warn")
        else:
            theme.info_box("✅ <strong>On track.</strong> This student appears well-engaged and is unlikely to be at risk.", "success")

    # Comparison with cohort
    theme.section_div()
    st.markdown("#### How This Student Compares to Cohort")

    compare_cols = ["xgb_prob", "avg_score", "total_clicks", "num_of_prev_attempts"]
    compare_cols = [c for c in compare_cols if c in pred_df.columns]

    if compare_cols:
        cmp_data = []
        for col in compare_cols:
            cohort_mean = float(pred_df[col].mean())
            student_val = float(student_row.get(col, cohort_mean))
            label = {"xgb_prob": "Risk Score (%)", "avg_score": "Avg Score (%)",
                     "total_clicks": "VLE Clicks", "num_of_prev_attempts": "Prev Attempts"}.get(col, col)
            cmp_data.append({"Metric": label, "Student": student_val, "Cohort Mean": cohort_mean})

        cmp_df = pd.DataFrame(cmp_data)
        fig_cmp = px.bar(
            pd.melt(cmp_df, id_vars="Metric", var_name="Group", value_name="Value"),
            x="Metric", y="Value", color="Group",
            color_discrete_map={"Student": theme.LIGHT["blue"], "Cohort Mean": "#94A3B8"},
            barmode="group", height=280,
        )
        fig_cmp.update_traces(marker_line_width=0)
        layout = theme.plotly_layout(dark)
        layout.update({"height": 280,
                       "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)})
        fig_cmp.update_layout(**layout)
        st.plotly_chart(fig_cmp, use_container_width=True)


# ── Export helpers ────────────────────────────────────────────────────────────

def _to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PRISM Predictions")
    return buf.getvalue()


def _to_pdf_bytes(df: pd.DataFrame) -> bytes | None:
    """Generate a minimal PDF risk report using only stdlib + basic bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=1.5*cm, rightMargin=1.5*cm,
                                topMargin=1.5*cm, bottomMargin=1.5*cm)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("<b>PRISM – Student Risk Report</b>", styles["Title"]))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(f"Generated from {len(df):,} student records.", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        # Summary stats
        high_n = int((df["risk_label"] == "High Risk").sum())
        med_n  = int((df["risk_label"] == "Medium Risk").sum())
        low_n  = int((df["risk_label"] == "Low Risk").sum())
        summary = [
            ["Risk Tier", "Count", "Share"],
            ["High Risk",   str(high_n), f"{high_n/len(df)*100:.1f}%"],
            ["Medium Risk", str(med_n),  f"{med_n/len(df)*100:.1f}%"],
            ["Low Risk",    str(low_n),  f"{low_n/len(df)*100:.1f}%"],
            ["Total",       str(len(df)), "100.0%"],
        ]
        t_sum = Table(summary, colWidths=[5*cm, 3*cm, 3*cm])
        t_sum.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
        ]))
        story.append(t_sum)
        story.append(Spacer(1, 0.5*cm))

        # Top 50 high-risk students
        top50 = df[df["risk_label"] == "High Risk"].nlargest(min(50, high_n), "xgb_prob")
        cols_show = ["id_student","code_module","gender","age_band","xgb_prob","final_result"]
        cols_show = [c for c in cols_show if c in top50.columns]
        headers = ["Student ID","Module","Gender","Age Band","Risk %","Result"][:len(cols_show)]

        story.append(Paragraph("<b>Top High-Risk Students (up to 50)</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))

        table_data = [headers]
        for _, row in top50.iterrows():
            table_data.append([str(row.get(c, "")) for c in cols_show])

        col_w = [18*cm / len(cols_show)] * len(cols_show)
        t_main = Table(table_data, colWidths=col_w)
        t_main.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#1E2D4A")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 8),
            ("ALIGN",        (0,0), (-1,-1), "LEFT"),
            ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#E2E8F0")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(t_main)

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return None


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    if not get_or_init_state():
        return

    pred_df = st.session_state["pred_df"].copy()
    dark    = theme.is_dark()

    # ── Student profile mode ──────────────────────────────────────────────────
    if "profile_student_id" in st.session_state:
        sid = st.session_state["profile_student_id"]
        matches = pred_df[pred_df["id_student"] == sid]
        if len(matches) > 0:
            theme.page_header("👤", f"Student Profile", f"Detailed view for Student #{sid}")
            _render_student_profile(matches.iloc[0], pred_df)
            return
        else:
            st.session_state.pop("profile_student_id")

    # ── Main predictions table ────────────────────────────────────────────────
    theme.page_header("🔍", "Student Risk Predictions",
                      "Browse, filter, search and export student risk classifications.")

    # Summary strip
    total  = len(pred_df)
    high   = int((pred_df["risk_label"] == "High Risk").sum())
    medium = int((pred_df["risk_label"] == "Medium Risk").sum())
    low    = int((pred_df["risk_label"] == "Low Risk").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1: theme.kpi_card("Total Students",  f"{total:,}",  accent="blue")
    with k2: theme.kpi_card("High Risk",       f"{high:,}",   sub=f"{high/total*100:.1f}%",   accent="red")
    with k3: theme.kpi_card("Medium Risk",     f"{medium:,}", sub=f"{medium/total*100:.1f}%", accent="amber")
    with k4: theme.kpi_card("Low Risk",        f"{low:,}",    sub=f"{low/total*100:.1f}%",    accent="teal")

    theme.section_div()

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters & Search", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.multiselect(
                "Risk Level",
                options=["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk", "Medium Risk", "Low Risk"],
            )
        with col2:
            module_opts = ["All"] + sorted(pred_df["code_module"].dropna().unique().tolist())
            module_filter = st.selectbox("Module", module_opts)
        with col3:
            prob_range = st.slider("Risk Score Range (%)", 0, 100, (0, 100))

        col4, col5, col6 = st.columns(3)
        with col4:
            search_id = st.text_input("🔍 Search by Student ID", placeholder="e.g. 11391")
        with col5:
            gender_opts = ["All"] + sorted(pred_df["gender"].dropna().unique().tolist()) if "gender" in pred_df.columns else ["All"]
            gender_filter = st.selectbox("Gender", gender_opts)
        with col6:
            result_opts = ["All"] + sorted(pred_df["final_result"].dropna().unique().tolist()) if "final_result" in pred_df.columns else ["All"]
            result_filter = st.selectbox("Actual Result", result_opts)

        sort_col = st.selectbox(
            "Sort by",
            options=["xgb_prob (highest first)", "xgb_prob (lowest first)",
                     "id_student (ascending)", "avg_score (highest first)" if "avg_score" in pred_df.columns else "xgb_prob (highest first)"],
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = pred_df[pred_df["risk_label"].isin(risk_filter)].copy()
    if module_filter != "All":
        filtered = filtered[filtered["code_module"] == module_filter]
    if gender_filter != "All" and "gender" in filtered.columns:
        filtered = filtered[filtered["gender"] == gender_filter]
    if result_filter != "All" and "final_result" in filtered.columns:
        filtered = filtered[filtered["final_result"] == result_filter]
    filtered = filtered[
        (filtered["xgb_prob"] >= prob_range[0]) &
        (filtered["xgb_prob"] <= prob_range[1])
    ]
    if search_id.strip():
        try:
            sid = int(search_id.strip())
            filtered = filtered[filtered["id_student"] == sid]
        except ValueError:
            theme.info_box("Student ID must be a number.", "warn")

    # Sort
    if "highest first" in sort_col and "xgb_prob" in sort_col:
        filtered = filtered.sort_values("xgb_prob", ascending=False)
    elif "lowest first" in sort_col:
        filtered = filtered.sort_values("xgb_prob", ascending=True)
    elif "id_student" in sort_col:
        filtered = filtered.sort_values("id_student", ascending=True)
    elif "avg_score" in sort_col and "avg_score" in filtered.columns:
        filtered = filtered.sort_values("avg_score", ascending=False)

    # ── Export row ────────────────────────────────────────────────────────────
    exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 1])
    with exp_col1:
        st.markdown(f'<div style="padding:0.45rem 0;font-weight:600;color:{"#94A3B8" if dark else "#475569"};">Showing <strong style="color:{"#F1F5F9" if dark else "#0F172A"};">{len(filtered):,}</strong> students</div>', unsafe_allow_html=True)

    display_cols = ["id_student","code_module","gender","age_band","final_result",
                    "risk_label","xgb_prob","lr_prob"]
    if "avg_score"     in filtered.columns: display_cols.append("avg_score")
    if "total_clicks"  in filtered.columns: display_cols.append("total_clicks")
    display_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[display_cols].copy().rename(columns={
        "id_student": "Student ID", "code_module": "Module", "gender": "Gender",
        "age_band": "Age Band", "final_result": "Actual Result", "risk_label": "Risk Level",
        "xgb_prob": "XGB Risk %", "lr_prob": "LR Risk %",
        "avg_score": "Avg Score", "total_clicks": "VLE Clicks",
    })

    with exp_col2:
        st.download_button(
            "⬇️ Export to Excel",
            data=_to_excel(show_df),
            file_name="prism_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with exp_col3:
        csv = show_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export to CSV",
            data=csv,
            file_name="prism_predictions.csv",
            mime="text/csv",
        )

    # PDF export (optional – needs reportlab)
    pdf_data = _to_pdf_bytes(filtered)
    if pdf_data:
        st.download_button(
            "📄 Export PDF Risk Report",
            data=pdf_data,
            file_name="prism_risk_report.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("💡 Install `reportlab` to enable PDF export.")

    # ── Custom HTML table with click-to-profile ───────────────────────────────
    theme.section_div()
    st.markdown("**Click any student row to open their full profile.**", unsafe_allow_html=True)

    page_size  = st.selectbox("Rows per page", [25, 50, 100, 250], index=0)
    total_pages = max(1, (len(filtered) - 1) // page_size + 1)
    page_num   = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1) - 1

    page_df = filtered.iloc[page_num * page_size : (page_num + 1) * page_size]

    headers = ["Student ID","Module","Gender","Age Band","Actual Result","Risk Level","XGB Risk %","LR Risk %"]
    if "avg_score"    in filtered.columns: headers.append("Avg Score")
    if "total_clicks" in filtered.columns: headers.append("VLE Clicks")

    th_style = "padding:9px 12px;text-align:left;font-weight:700;color:var(--text2);font-size:0.69rem;text-transform:uppercase;letter-spacing:0.06em;"
    td_style = "padding:9px 12px;color:var(--text);"
    mono_td  = "padding:9px 12px;font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:var(--text2);"

    rows_html = ""
    for _, r in page_df.iterrows():
        sid = int(r["id_student"])
        risk_score = float(r["xgb_prob"])
        lr_score   = float(r.get("lr_prob", 0))
        risk_color = {"High Risk": "#DC2626", "Medium Risk": "#D97706", "Low Risk": "#0D9488"}.get(r["risk_label"], "#64748B")

        extra_cells = ""
        if "avg_score"    in r.index and pd.notna(r.get("avg_score")): extra_cells += f'<td style="{mono_td}">{r["avg_score"]:.1f}%</td>'
        if "total_clicks" in r.index and pd.notna(r.get("total_clicks")): extra_cells += f'<td style="{mono_td}">{int(r["total_clicks"]):,}</td>'

        rows_html += f"""
<tr style="border-bottom:1px solid var(--border);cursor:pointer;"
    onclick="document.getElementById('profile_input_{sid}').click()">
    <td style="{mono_td}">{sid}</td>
    <td style="{td_style}">{r.get('code_module','N/A')}</td>
    <td style="{td_style}">{r.get('gender','N/A')}</td>
    <td style="{td_style}">{r.get('age_band','N/A')}</td>
    <td style="{td_style}">{r.get('final_result','N/A')}</td>
    <td style="{td_style}">{theme.badge_html(r['risk_label'])}</td>
    <td style="{mono_td}font-weight:700;color:{risk_color};">{risk_score:.1f}%</td>
    <td style="{mono_td}color:var(--text3);">{lr_score:.1f}%</td>
    {extra_cells}
</tr>"""

    hdr_html = "".join(f'<th style="{th_style}">{h}</th>' for h in headers)
    st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);
            overflow:auto;box-shadow:var(--shadow);">
    <table class="prism-table">
        <thead><tr style="background:var(--surface2);">{hdr_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)

    # Hidden selectbox for profile navigation
    page_student_ids = page_df["id_student"].astype(int).tolist()
    if page_student_ids:
        chosen = st.selectbox(
            "Open Student Profile",
            options=[None] + page_student_ids,
            format_func=lambda x: "— select to open profile —" if x is None else f"Student #{x}",
            key="profile_selector",
        )
        if chosen is not None:
            st.session_state["profile_student_id"] = chosen
            st.rerun()

    st.caption(f"Page {page_num+1} of {total_pages} · {len(filtered):,} students match current filters")

    # ── Mini chart ────────────────────────────────────────────────────────────
    theme.section_div()
    if len(filtered) > 0:
        col_chart, col_empty = st.columns([1, 1])
        with col_chart:
            rc = filtered["risk_label"].value_counts().reset_index()
            rc.columns = ["Risk Level", "Count"]
            fig = px.bar(rc, x="Risk Level", y="Count",
                         color="Risk Level", color_discrete_map=theme.RISK_COLORS,
                         height=240)
            fig.update_traces(marker_line_width=0)
            layout = theme.plotly_layout(dark)
            layout.update({"height": 240, "showlegend": False})
            fig.update_layout(**layout)
            theme.chart_card_header("Filtered Distribution")
            st.plotly_chart(fig, use_container_width=True)
