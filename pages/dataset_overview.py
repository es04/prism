import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state


def render():
    if not get_or_init_state():
        return

    dfs       = st.session_state["raw_dfs"]
    master_df = st.session_state["master_df"]

    st.markdown("""
    <div class="page-header">
        <h2>🗂️ Dataset Overview</h2>
        <p>OULAD dataset statistics, quality metrics, and feature distributions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top metric cards
    si   = dfs["studentInfo"]
    sa   = dfs["studentAssessment"]
    sv   = dfs["studentVle"]
    modules = si["code_module"].nunique() if not si.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Student Records",      f"{len(si):,}")
    c2.metric("Assessment Records",   f"{len(sa):,}")
    c3.metric("VLE Interaction Rows", f"{len(sv):,}")
    c4.metric("Unique Modules",        f"{modules}")
    c5.metric("Master Features",       f"{master_df.shape[1]}")

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # Tabs per dataset
    tab_names = ["studentInfo","studentAssessment","studentVle",
                 "assessments","courses","Distributions","Data Quality"]
    tabs = st.tabs(tab_names)

    def show_df_preview(df, name):
        if df.empty:
            st.warning(f"{name} not loaded.")
            return
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True)
        with st.expander("Summary Statistics"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    with tabs[0]:
        show_df_preview(si, "studentInfo")

        st.subheader("Final Result Distribution")
        fr = si["final_result"].value_counts().reset_index()
        fr.columns = ["Result","Count"]
        color_map = {"Pass":"#22c55e","Distinction":"#3b82f6","Fail":"#ef4444","Withdrawn":"#f59e0b"}
        fig = px.bar(fr, x="Result", y="Count", color="Result",
                     color_discrete_map=color_map, height=300)
        fig.update_layout(showlegend=False, margin=dict(t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        show_df_preview(sa, "studentAssessment")
        if not sa.empty:
            sa2 = sa.copy()
            sa2["score"] = pd.to_numeric(sa2["score"], errors="coerce")
            st.subheader("Score Distribution")
            fig_s = px.histogram(sa2, x="score", nbins=40,
                                 color_discrete_sequence=["#2563eb"], height=280)
            fig_s.update_layout(margin=dict(t=10,b=20))
            st.plotly_chart(fig_s, use_container_width=True)

    with tabs[2]:
        st.markdown(f"**Shape:** {sv.shape[0]:,} rows × {sv.shape[1]} columns")
        st.dataframe(sv.head(20), use_container_width=True)
        if not sv.empty:
            st.subheader("Sum-Click Distribution (log scale)")
            sv2 = sv.copy()
            sv2["sum_click"] = pd.to_numeric(sv2["sum_click"], errors="coerce")
            sv_agg = sv2.groupby("id_student")["sum_click"].sum().reset_index()
            sv_agg["sum_click"] = np.log1p(sv_agg["sum_click"])
            fig_v = px.histogram(sv_agg, x="sum_click", nbins=40,
                                  color_discrete_sequence=["#0891b2"], height=280,
                                  labels={"sum_click":"log(1+total_clicks)"})
            fig_v.update_layout(margin=dict(t=10,b=20))
            st.plotly_chart(fig_v, use_container_width=True)

    with tabs[3]:
        show_df_preview(dfs["assessments"], "assessments")
        a = dfs["assessments"]
        if not a.empty:
            st.subheader("Assessment Types")
            at = a["assessment_type"].value_counts().reset_index()
            at.columns = ["Type","Count"]
            fig_at = px.pie(at, names="Type", values="Count", hole=0.4, height=280)
            st.plotly_chart(fig_at, use_container_width=True)

    with tabs[4]:
        show_df_preview(dfs["courses"], "courses")

    # Distributions of master features
    with tabs[5]:
        st.subheader("Student Demographic & Academic Distributions")

        col1, col2 = st.columns(2)

        with col1:
            if "gender" in si.columns:
                g = si["gender"].value_counts().reset_index()
                g.columns = ["Gender","Count"]
                fig_g = px.pie(g, names="Gender", values="Count", hole=0.4,
                                title="Gender Distribution", height=280)
                st.plotly_chart(fig_g, use_container_width=True)

            if "age_band" in si.columns:
                ab = si["age_band"].value_counts().reset_index()
                ab.columns = ["Age Band","Count"]
                fig_ab = px.bar(ab, x="Age Band", y="Count",
                                 color_discrete_sequence=["#2563eb"],
                                 title="Age Band Distribution", height=280)
                st.plotly_chart(fig_ab, use_container_width=True)

        with col2:
            if "highest_education" in si.columns:
                he = si["highest_education"].value_counts().reset_index()
                he.columns = ["Education","Count"]
                fig_he = px.bar(he, x="Count", y="Education", orientation="h",
                                 color_discrete_sequence=["#7c3aed"],
                                 title="Highest Education", height=280)
                fig_he.update_layout(yaxis_title="")
                st.plotly_chart(fig_he, use_container_width=True)

            if "imd_band" in si.columns:
                imd = si["imd_band"].value_counts().reset_index()
                imd.columns = ["IMD Band","Count"]
                fig_imd = px.bar(imd, x="IMD Band", y="Count",
                                  color_discrete_sequence=["#059669"],
                                  title="IMD Band Distribution", height=280)
                st.plotly_chart(fig_imd, use_container_width=True)

        # Risk by demographics
        st.subheader("At-Risk Rate by Demographic Group")
        pred_df = st.session_state["pred_df"]
        si_pred = si.merge(pred_df[["id_student","risk_label","xgb_prob"]],
                           on="id_student", how="left")

        demo_col = st.selectbox("Demographic Feature",
                                 ["age_band","gender","highest_education","imd_band","disability"])
        if demo_col in si_pred.columns:
            grp = si_pred.groupby(demo_col)["risk_label"].apply(
                lambda x: (x == "High Risk").mean() * 100
            ).reset_index()
            grp.columns = [demo_col, "High Risk Rate (%)"]
            fig_demo = px.bar(
                grp, x=demo_col, y="High Risk Rate (%)",
                color="High Risk Rate (%)", color_continuous_scale="Reds",
                height=300,
            )
            fig_demo.update_layout(margin=dict(t=10,b=30), coloraxis_showscale=False)
            st.plotly_chart(fig_demo, use_container_width=True)

    # Data Quality
    with tabs[6]:
        st.subheader("Data Quality Report")
        quality_rows = []
        for name, df in dfs.items():
            if df.empty:
                continue
            missing = df.isnull().sum().sum()
            total   = df.shape[0] * df.shape[1]
            pct_missing = missing / total * 100 if total > 0 else 0
            quality_rows.append({
                "Dataset": name,
                "Rows": f"{df.shape[0]:,}",
                "Columns": df.shape[1],
                "Missing Values": f"{missing:,}",
                "Missing %": f"{pct_missing:.2f}%",
                "Duplicates": f"{df.duplicated().sum():,}",
            })
        q_df = pd.DataFrame(quality_rows)
        st.dataframe(q_df, use_container_width=True, hide_index=True)

        # Master dataset quality
        st.subheader("Master Dataset Missing Values per Feature")
        miss_per_col = master_df.isnull().sum()
        miss_per_col = miss_per_col[miss_per_col > 0].sort_values(ascending=False).head(20)
        if len(miss_per_col) > 0:
            fig_miss = px.bar(
                x=miss_per_col.index, y=miss_per_col.values,
                labels={"x":"Feature","y":"Missing Count"},
                color=miss_per_col.values, color_continuous_scale="Reds",
                height=300,
            )
            fig_miss.update_layout(margin=dict(t=10,b=30), coloraxis_showscale=False)
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("✅ No missing values in the master dataset after preprocessing!")
