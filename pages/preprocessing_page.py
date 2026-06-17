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

    master   = st.session_state["master_df"]
    raw_dfs  = st.session_state["raw_dfs"]
    feat_nms = st.session_state["feature_names"]
    X_all    = st.session_state["X_all"]
    y_all    = st.session_state["y_all"]

    st.markdown("""
    <div class="page-header">
        <h2>🔬 Preprocessing Pipeline</h2>
        <p>Step-by-step breakdown of the data preparation pipeline with before/after comparisons.</p>
    </div>
    """, unsafe_allow_html=True)

    steps = st.tabs([
        "1️⃣ Raw Data",
        "2️⃣ Feature Engineering",
        "3️⃣ Target Encoding",
        "4️⃣ Missing Values",
        "5️⃣ Label Encoding",
        "6️⃣ Scaling",
        "7️⃣ Class Balance",
        "8️⃣ Final Dataset",
    ])

    # Step 1: Raw Data
    with steps[0]:
        st.subheader("Step 1: Raw OULAD Dataset Files")
        st.markdown("""
        <div class="info-box">
            Seven CSV files are loaded from the OULAD dataset. Each provides a different
            dimension of student behaviour and academic history.
        </div>
        """, unsafe_allow_html=True)

        for name, df in raw_dfs.items():
            if df.empty: continue
            with st.expander(f"📄 {name}.csv — {len(df):,} rows × {len(df.columns)} columns"):
                c1, c2 = st.columns([2,1])
                with c1:
                    st.dataframe(df.head(5), use_container_width=True)
                with c2:
                    st.markdown("**Column Types**")
                    dtype_df = pd.DataFrame({
                        "Column": df.columns,
                        "Dtype":  df.dtypes.astype(str).values,
                        "Nulls":  df.isnull().sum().values,
                        "Nulls%": (df.isnull().mean()*100).round(1).astype(str) + "%",
                    })
                    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    # Step 2: Feature Engineering
    with steps[1]:
        st.subheader("Step 2: Feature Engineering")
        st.markdown("""
        <div class="info-box">
            Raw records from assessment, VLE, and registration tables are aggregated
            per student to produce predictive behavioural indicators.
        </div>
        """, unsafe_allow_html=True)

        eng_feats = {
            "Assessment Aggregates": {
                "source": "studentAssessment + assessments",
                "features": {
                    "avg_score":             "Mean score across all submitted assessments",
                    "min_score":             "Minimum score (worst performance)",
                    "max_score":             "Maximum score (best performance)",
                    "std_score":             "Standard deviation — score consistency",
                    "num_assessments":       "Total count of assessments attempted",
                    "num_late_submissions":  "Count of assessments submitted after due date",
                    "avg_submission_timing": "Mean days before/after due date (positive = early)",
                    "num_banked":            "Count of banked/credited assessments",
                }
            },
            "VLE Interaction Aggregates": {
                "source": "studentVle + vle",
                "features": {
                    "total_clicks":           "Sum of all VLE interactions",
                    "avg_clicks_per_day":     "Mean daily interaction count",
                    "num_active_days":        "Distinct days with any VLE activity",
                    "num_resources_accessed": "Distinct learning resources visited",
                }
            },
            "Registration Features": {
                "source": "studentRegistration",
                "features": {
                    "registered_early":   "1 if registered before course start date",
                    "withdrew_early":     "1 if student later unregistered",
                    "days_enrolled":      "Duration between registration and unregistration",
                    "num_registrations":  "Total number of course registrations",
                }
            },
        }

        for group, info in eng_feats.items():
            with st.expander(f"🔧 {group} (from: {info['source']})"):
                feat_df = pd.DataFrame([
                    {"Feature": k, "Description": v,
                     "In Master": "✅" if k in master.columns else "❌"}
                    for k, v in info["features"].items()
                ])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

                # Distribution plots for available features
                avail = [k for k in info["features"] if k in master.columns]
                if avail:
                    sel = st.selectbox(f"Plot distribution for {group}", avail,
                                       key=f"fe_{group}")
                    col_data = master[sel].dropna()
                    if col_data.dtype == object or col_data.nunique() < 15:
                        vc = col_data.value_counts().reset_index()
                        vc.columns = [sel, "Count"]
                        fig = px.bar(vc, x=sel, y="Count",
                                     color_discrete_sequence=["#2563eb"], height=250)
                    else:
                        fig = px.histogram(col_data, nbins=40, height=250,
                                           color_discrete_sequence=["#2563eb"])
                    fig.update_layout(margin=dict(t=5,b=20,l=5,r=5))
                    st.plotly_chart(fig, use_container_width=True)

    # Step 3: Target Encoding
    with steps[2]:
        st.subheader("Step 3: Target Variable Encoding")
        st.markdown("""
        <div class="info-box">
            The <code>final_result</code> column (4-class) is mapped to a binary
            <code>is_at_risk</code> label for supervised classification.
        </div>
        """, unsafe_allow_html=True)

        mapping = pd.DataFrame({
            "final_result": ["Pass","Distinction","Fail","Withdrawn"],
            "is_at_risk":   [0, 0, 1, 1],
            "Risk Class":   ["Not At Risk","Not At Risk","At Risk","At Risk"],
            "Reasoning":    [
                "Successfully completed the module",
                "Excellent performance — clearly not at risk",
                "Failed the module — requires intervention",
                "Withdrew from course — high risk indicator",
            ],
        })
        st.dataframe(mapping, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original 4-Class Distribution**")
            fr = raw_dfs["studentInfo"]["final_result"].value_counts().reset_index()
            fr.columns = ["Result","Count"]
            color_map = {"Pass":"#22c55e","Distinction":"#3b82f6",
                         "Fail":"#ef4444","Withdrawn":"#f59e0b"}
            fig1 = px.bar(fr, x="Result", y="Count", color="Result",
                          color_discrete_map=color_map, height=280)
            fig1.update_layout(showlegend=False, margin=dict(t=5,b=20,l=5,r=5))
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            st.markdown("**Binary Risk Label Distribution**")
            bin_counts = {"Not At Risk": int((y_all==0).sum()),
                          "At Risk":     int((y_all==1).sum())}
            fig2 = px.pie(
                names=list(bin_counts.keys()), values=list(bin_counts.values()),
                color=list(bin_counts.keys()),
                color_discrete_map={"Not At Risk":"#3b82f6","At Risk":"#ef4444"},
                hole=0.4, height=280,
            )
            fig2.update_traces(textinfo="percent+label")
            fig2.update_layout(margin=dict(t=5,b=5), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # Step 4: Missing Values
    with steps[3]:
        st.subheader("Step 4: Missing Value Analysis & Imputation")
        st.markdown("""
        <div class="info-box">
            Missing values are imputed using the <b>median strategy</b> (robust to outliers).
            Features with &gt;50% missing are excluded.
        </div>
        """, unsafe_allow_html=True)

        feat_df_raw = master[feat_nms].copy() if all(f in master.columns for f in feat_nms) else master

        miss = feat_df_raw.isnull().sum().sort_values(ascending=False)
        miss_pct = (feat_df_raw.isnull().mean() * 100).sort_values(ascending=False)
        miss_df = pd.DataFrame({
            "Feature": miss.index,
            "Missing Count": miss.values,
            "Missing %": miss_pct.values.round(2),
            "Strategy": ["Median Imputation" if v > 0 else "No imputation needed"
                         for v in miss.values],
        })
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        top_miss = miss_pct[miss_pct > 0].head(15)
        if len(top_miss) > 0:
            fig = px.bar(x=top_miss.index, y=top_miss.values,
                         labels={"x":"Feature","y":"Missing %"},
                         color=top_miss.values, color_continuous_scale="Reds",
                         height=300)
            fig.update_layout(margin=dict(t=5,b=30,l=5,r=5), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values in the final feature set after preprocessing!")

    # Step 5: Label Encoding
    with steps[4]:
        st.subheader("Step 5: Categorical Label Encoding")
        st.markdown("""
        <div class="info-box">
            <code>sklearn.LabelEncoder</code> converts string category values to integers.
            An "Unknown" sentinel class is added to each encoder to handle unseen categories gracefully.
        </div>
        """, unsafe_allow_html=True)

        encoders = st.session_state.get("encoders", {})
        cat_cols_used = [c for c in ["gender","age_band","highest_education",
                                      "imd_band","disability"] if c in feat_nms]

        if encoders:
            for col in cat_cols_used:
                if col not in encoders: continue
                le = encoders[col]
                with st.expander(f"🔤 {col} encoder"):
                    enc_df = pd.DataFrame({
                        "Original Category": le.classes_,
                        "Encoded Integer":   range(len(le.classes_)),
                    })
                    c1, c2 = st.columns([1,2])
                    with c1:
                        st.dataframe(enc_df, use_container_width=True, hide_index=True)
                    with c2:
                        if col in master.columns:
                            vc = master[col].astype(str).value_counts().reset_index()
                            vc.columns = [col, "Count"]
                            fig = px.bar(vc, x=col, y="Count",
                                         color_discrete_sequence=["#7c3aed"], height=250)
                            fig.update_layout(margin=dict(t=5,b=20))
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Encoders not yet available. Run the pipeline first.")

    # Step 6: Scaling
    with steps[5]:
        st.subheader("Step 6: StandardScaler Normalisation")
        st.markdown("""
        <div class="info-box">
            <code>StandardScaler</code> transforms each numeric feature to zero mean and unit variance:
            <code>z = (x − μ) / σ</code>. This ensures no single feature dominates due to scale differences.
        </div>
        """, unsafe_allow_html=True)

        scaler = st.session_state.get("scaler")
        if scaler is not None and hasattr(scaler, "mean_"):
            scaler_df = pd.DataFrame({
                "Feature": feat_nms[:len(scaler.mean_)],
                "Mean (μ)": np.round(scaler.mean_, 4),
                "Std (σ)":  np.round(scaler.scale_, 4),
                "Min (before)": np.round(scaler.mean_ - 2*scaler.scale_, 2),
                "Max (before)": np.round(scaler.mean_ + 2*scaler.scale_, 2),
            })
            st.dataframe(scaler_df, use_container_width=True, hide_index=True)

            # Before/After scaling for a selected feature
            sel = st.selectbox("Visualise scaling for feature", feat_nms[:len(scaler.mean_)])
            idx = feat_nms.index(sel)
            if idx < X_all.shape[1]:
                raw_vals = master[sel].dropna() if sel in master.columns else None
                scaled_vals = X_all[:, idx]
                if raw_vals is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Before Scaling**")
                        fig_b = px.histogram(raw_vals, nbins=30, height=220,
                                              color_discrete_sequence=["#94a3b8"])
                        fig_b.update_layout(margin=dict(t=5,b=20))
                        st.plotly_chart(fig_b, use_container_width=True)
                    with c2:
                        st.markdown("**After Scaling (z-score)**")
                        fig_a = px.histogram(scaled_vals, nbins=30, height=220,
                                              color_discrete_sequence=["#2563eb"])
                        fig_a.update_layout(margin=dict(t=5,b=20))
                        st.plotly_chart(fig_a, use_container_width=True)

    # Step 7: Class Balance
    with steps[6]:
        st.subheader("Step 7: SMOTE Class Balancing")
        sb = st.session_state.get("smote_before", {"Not At Risk": int((y_all==0).sum()), "At Risk": int((y_all==1).sum())})
        sa = st.session_state.get("smote_after",  {"Not At Risk": sb["Not At Risk"], "At Risk": sb["Not At Risk"]})

        st.markdown("""
        <div class="info-box">
            SMOTE is applied <b>only on the training split</b> to prevent data leakage into the test set.
        </div>
        """, unsafe_allow_html=True)

        rows = [
            {"Stage": "Before SMOTE (train split)", "Not At Risk": sb["Not At Risk"],
             "At Risk": sb["At Risk"], "Ratio": f"{sb['Not At Risk']/max(sb['At Risk'],1):.2f}:1"},
            {"Stage": "After SMOTE (train split)",  "Not At Risk": sa["Not At Risk"],
             "At Risk": sa["At Risk"], "Ratio": "1.00:1"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        fig = go.Figure(data=[
            go.Bar(name="Not At Risk", x=["Before SMOTE","After SMOTE"],
                   y=[sb["Not At Risk"], sa["Not At Risk"]], marker_color="#3b82f6"),
            go.Bar(name="At Risk",     x=["Before SMOTE","After SMOTE"],
                   y=[sb["At Risk"],   sa["At Risk"]],     marker_color="#ef4444"),
        ])
        fig.update_layout(barmode="group", height=300, margin=dict(t=5,b=20,l=5,r=5))
        st.plotly_chart(fig, use_container_width=True)

    # Step 8: Final Dataset
    with steps[7]:
        st.subheader("Step 8: Final Processed Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Samples",  f"{X_all.shape[0]:,}")
        c2.metric("Features",       f"{X_all.shape[1]}")
        c3.metric("At-Risk (1)",    f"{int(y_all.sum()):,}")
        c4.metric("Not At-Risk (0)",f"{int((y_all==0).sum()):,}")

        st.markdown("**Final Feature Matrix (first 10 rows, scaled)**")
        final_df = pd.DataFrame(X_all[:10], columns=feat_nms)
        st.dataframe(final_df.round(3), use_container_width=True)

        st.markdown("**Feature Matrix Statistics (scaled)**")
        stats = pd.DataFrame(X_all, columns=feat_nms).describe().T
        st.dataframe(stats.round(4), use_container_width=True)
