"""
PRISM v2 – Dataset Overview (redesigned)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
import theme


def _fig_base(fig, height=290, **extra):
    dark   = theme.is_dark()
    layout = theme.plotly_layout(dark)
    layout["height"] = height
    fig.update_layout(**layout)
    if extra:
        fig.update_layout(**extra)
    gc = "#1E2D4A" if dark else "#F1F5F9"
    fig.update_xaxes(gridcolor=gc, zeroline=False)
    fig.update_yaxes(gridcolor=gc, zeroline=False)
    return fig


def render():
    if not get_or_init_state():
        return

    dfs       = st.session_state["raw_dfs"]
    master_df = st.session_state["master_df"]
    dark      = theme.is_dark()

    theme.page_header("🗂️", "Dataset Overview",
                      "OULAD dataset statistics, quality metrics, and feature distributions.")

    si = dfs["studentInfo"]
    sa = dfs["studentAssessment"]
    sv = dfs["studentVle"]
    modules = si["code_module"].nunique() if not si.empty else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: theme.kpi_card("Student Records",      f"{len(si):,}",          accent="blue")
    with k2: theme.kpi_card("Assessment Records",   f"{len(sa):,}",          accent="teal")
    with k3: theme.kpi_card("VLE Interactions",     f"{len(sv):,}",          accent="amber")
    with k4: theme.kpi_card("Unique Modules",       f"{modules}",            accent="violet")
    with k5: theme.kpi_card("Master Features",      f"{master_df.shape[1]}", accent="red")

    theme.section_div()

    tab_names = ["studentInfo","studentAssessment","studentVle","assessments","courses","Distributions","Data Quality"]
    tabs = st.tabs(tab_names)

    def show_df_preview(df, name):
        if df.empty:
            theme.info_box(f"{name} not loaded.", "warn")
            return
        st.markdown(f'**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns')
        st.dataframe(df.head(20), use_container_width=True)
        with st.expander("Summary Statistics"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tabs[0]:
        show_df_preview(si, "studentInfo")
        if not si.empty and "final_result" in si.columns:
            st.subheader("Final Result Distribution")
            fr = si["final_result"].value_counts().reset_index()
            fr.columns = ["Result","Count"]
            color_map = {"Pass":"#0D9488","Distinction":"#2563EB","Fail":"#DC2626","Withdrawn":"#D97706"}
            fig = px.bar(fr, x="Result", y="Count", color="Result",
                         color_discrete_map=color_map)
            fig.update_traces(marker_line_width=0)
            _fig_base(fig)
            st.plotly_chart(fig, use_container_width=True)

    for i, (key, label) in enumerate([("studentAssessment","studentAssessment"),
                                       ("studentVle","studentVle"),
                                       ("assessments","assessments"),
                                       ("courses","courses")]):
        with tabs[i+1]:
            show_df_preview(dfs.get(key, pd.DataFrame()), label)

    with tabs[5]:
        st.subheader("Feature Distributions")
        if not master_df.empty:
            num_cols = master_df.select_dtypes(include=np.number).columns.tolist()
            feat = st.selectbox("Select feature to plot", num_cols)
            if feat:
                fig_d = px.histogram(master_df, x=feat, nbins=40,
                                     color_discrete_sequence=["#2563EB"])
                fig_d.update_traces(marker_line_width=0)
                _fig_base(fig_d)
                st.plotly_chart(fig_d, use_container_width=True)

            if "code_module" in master_df.columns and len(num_cols) >= 1:
                fig_box = px.box(master_df, x="code_module", y=feat, color="code_module",
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                _fig_base(fig_box, height=300, showlegend=False,
                          xaxis_title="Module", yaxis_title=feat)
                theme.chart_card_header(f"{feat} by Module")
                st.plotly_chart(fig_box, use_container_width=True)

    with tabs[6]:
        st.subheader("Data Quality Report")
        if not master_df.empty:
            null_counts = master_df.isnull().sum()
            null_pct    = (null_counts / len(master_df) * 100).round(2)
            dtypes      = master_df.dtypes.astype(str)
            quality_df  = pd.DataFrame({
                "Column":    null_counts.index,
                "Dtype":     dtypes.values,
                "Null Count":null_counts.values,
                "Null %":    null_pct.values,
            }).sort_values("Null %", ascending=False).reset_index(drop=True)
            st.dataframe(quality_df, use_container_width=True, hide_index=True)

            missing = quality_df[quality_df["Null %"] > 0]
            if len(missing) > 0:
                fig_miss = px.bar(missing.head(20), x="Column", y="Null %",
                                  color="Null %",
                                  color_continuous_scale=[[0,"#FEF9C3"],[0.5,"#FDE68A"],[1,"#DC2626"]])
                fig_miss.update_traces(marker_line_width=0)
                _fig_base(fig_miss, height=280, coloraxis_showscale=False,
                          yaxis=dict(ticksuffix="%"))
                theme.chart_card_header("Missing Values by Feature")
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                theme.info_box("✅ No missing values detected in the master dataset.", "success")
