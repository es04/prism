import streamlit as st
import pandas as pd
import plotly.express as px

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state


def render():
    if not get_or_init_state():
        return

    pred_df = st.session_state["pred_df"].copy()

    st.markdown("""
    <div class="page-header">
        <h2>📋 Student Risk Predictions</h2>
        <p>Browse, filter and search individual student risk classifications.</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary strip 
    total  = len(pred_df)
    high   = int((pred_df["risk_label"] == "High Risk").sum())
    medium = int((pred_df["risk_label"] == "Medium Risk").sum())
    low    = int((pred_df["risk_label"] == "Low Risk").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students",    f"{total:,}")
    c2.metric("🔴 High Risk",      f"{high:,}")
    c3.metric("🟡 Medium Risk",    f"{medium:,}")
    c4.metric("🟢 Low Risk",       f"{low:,}")

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # Filter controls 
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"],
        )
    with fc2:
        module_options = ["All"] + sorted(pred_df["code_module"].dropna().unique().tolist())
        module_filter  = st.selectbox("Filter by Module", module_options)
    with fc3:
        prob_range = st.slider("Risk Probability Range (%)", 0, 100, (0, 100))

    # Search bar
    search_id = st.text_input("🔍 Search by Student ID", placeholder="e.g. 11391")

    # Apply filters 
    filtered = pred_df[pred_df["risk_label"].isin(risk_filter)].copy()
    if module_filter != "All":
        filtered = filtered[filtered["code_module"] == module_filter]
    filtered = filtered[
        (filtered["xgb_prob"] >= prob_range[0]) &
        (filtered["xgb_prob"] <= prob_range[1])
    ]
    if search_id.strip():
        try:
            sid = int(search_id.strip())
            filtered = filtered[filtered["id_student"] == sid]
        except ValueError:
            st.warning("Student ID must be a number.")

    filtered = filtered.sort_values("xgb_prob", ascending=False)

    st.markdown(f"**Showing {len(filtered):,} student records**")

    # Risk badge helper 
    def badge(label):
        cls = {"High Risk":"badge-high","Medium Risk":"badge-medium","Low Risk":"badge-low"}.get(label,"")
        return f'<span class="{cls}">{label}</span>'

    # Columns to show 
    display_cols = ["id_student","code_module","gender","age_band","final_result",
                    "risk_label","xgb_prob","lr_prob"]
    if "avg_score" in filtered.columns:
        display_cols.append("avg_score")
    if "total_clicks" in filtered.columns:
        display_cols.append("total_clicks")

    display_cols = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[display_cols].copy()

    rename_map = {
        "id_student":     "Student ID",
        "code_module":    "Module",
        "gender":         "Gender",
        "age_band":       "Age Band",
        "final_result":   "Actual Result",
        "risk_label":     "Risk Level",
        "xgb_prob":       "XGB Risk % ",
        "lr_prob":        "LR Risk %",
        "avg_score":      "Avg Score",
        "total_clicks":   "VLE Clicks",
    }
    show_df.rename(columns=rename_map, inplace=True)
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True, height=460)

    # Download button
    csv = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download filtered predictions as CSV",
        data=csv,
        file_name="prism_predictions.csv",
        mime="text/csv",
    )

    # Mini charts for filtered data
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    if len(filtered) > 0:
        st.subheader("Risk Level Distribution (Filtered)")
        rc = filtered["risk_label"].value_counts().reset_index()
        rc.columns = ["Risk Level","Count"]
        fig = px.bar(
            rc, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
            height=280,
        )
        fig.update_layout(showlegend=False, margin=dict(t=10,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
