import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
import theme


def render():
    theme.inject_theme_css()
    if not get_or_init_state():
        return

    xgb_model     = st.session_state["xgb_model"]
    shap_values   = st.session_state["shap_values"]
    feature_names = st.session_state["feature_names"]
    X_all         = st.session_state["X_all"]
    master_df     = st.session_state["master_df"]

    st.markdown("""
    <div class="page-header">
        <h2>📈 Feature Importance Analysis</h2>
        <p>XGBoost intrinsic importance, SHAP-based importance, and feature correlations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Top summary
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]
    top15_feat    = [feature_names[i] for i in sorted_idx[:15]]
    top15_shap    = mean_abs_shap[sorted_idx[:15]]

    st.markdown("""
    <div class="info-box">
        Feature importance is measured in two complementary ways:<br>
        <b>1. XGBoost Built-in</b> – based on the number of times a feature is used for splitting.<br>
        <b>2. SHAP (Mean |SHAP|)</b> – measures the average absolute impact on model output (more reliable).
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["SHAP Importance", "XGBoost Built-in", "Correlation Analysis"])

    # ── SHAP importance ────────────────────────────────────────────────────
    with tab1:
        st.subheader("Top 15 Features by Mean |SHAP Value|")

        # Colour bars by category
        cat_map = {
            "avg_score":"Academic","min_score":"Academic","max_score":"Academic",
            "std_score":"Academic","num_assessments":"Academic",
            "num_late_submissions":"Academic","avg_submission_timing":"Academic",
            "num_banked":"Academic",
            "total_clicks":"Behavioural","avg_clicks_per_day":"Behavioural",
            "num_active_days":"Behavioural","num_resources_accessed":"Behavioural",
            "gender":"Demographic","age_band":"Demographic","highest_education":"Demographic",
            "imd_band":"Demographic","disability":"Demographic",
            "num_of_prev_attempts":"Demographic","studied_credits":"Demographic",
            "registered_early":"Registration","withdrew_early":"Registration",
            "days_enrolled":"Registration","num_registrations":"Registration",
            "module_presentation_length":"Course",
        }
        colors_map = {
            "Academic":"#1d4ed8","Behavioural":"#0891b2",
            "Demographic":"#7c3aed","Registration":"#059669","Course":"#d97706",
        }
        bar_colors = [colors_map.get(cat_map.get(f,"Other"),"#6b7280") for f in top15_feat]

        shap_imp_df = pd.DataFrame({
            "Feature":      top15_feat,
            "Mean |SHAP|":  np.round(top15_shap, 5),
            "Category":     [cat_map.get(f,"Other") for f in top15_feat],
        })

        fig_shap = px.bar(
            shap_imp_df.sort_values("Mean |SHAP|"),
            x="Mean |SHAP|", y="Feature",
            color="Category",
            color_discrete_map=colors_map,
            orientation="h", height=480,
            labels={"Mean |SHAP|":"Mean Absolute SHAP Value"},
        )
        fig_shap.update_layout(margin=dict(t=10,b=30,l=10,r=10), yaxis_title="")
        st.plotly_chart(fig_shap, use_container_width=True)

        # Percentage contribution
        total_imp = top15_shap.sum()
        shap_imp_df["% Contribution"] = (top15_shap / total_imp * 100).round(2)
        shap_imp_df["% Contribution"] = shap_imp_df["% Contribution"].apply(lambda x: f"{x:.2f}%")
        shap_imp_df["Mean |SHAP|"]    = shap_imp_df["Mean |SHAP|"].apply(lambda x: f"{x:.5f}")
        st.dataframe(shap_imp_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # XGBoost built-in
    with tab2:
        st.subheader("XGBoost Built-in Feature Importance")

        imp_types = ["weight","gain","cover"]
        sel_type  = st.radio("Importance Type", imp_types, horizontal=True)

        scores = xgb_model.get_booster().get_score(importance_type=sel_type)
        if scores:
            imp_df = pd.DataFrame(list(scores.items()), columns=["Feature","Importance"])
            imp_df = imp_df.sort_values("Importance", ascending=False).head(20)
            fig_xgb = px.bar(
                imp_df.sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                height=500, labels={"Importance": f"Importance ({sel_type})"},
            )
            fig_xgb.update_layout(margin=dict(t=10,b=30,l=10,r=10),
                                  coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig_xgb, use_container_width=True)
        else:
            st.info("Built-in importance scores not available for this model configuration.")

    # Correlation
    with tab3:
        st.subheader("Feature Correlation with At-Risk Classification")

        feat_df = pd.DataFrame(X_all, columns=feature_names)
        feat_df["is_at_risk"] = st.session_state["y_all"]

        # Correlation with target
        corr_with_target = feat_df.corr()["is_at_risk"].drop("is_at_risk").sort_values()
        fig_corr_bar = px.bar(
            x=corr_with_target.values,
            y=corr_with_target.index,
            orientation="h",
            color=corr_with_target.values,
            color_continuous_scale="RdBu_r",
            labels={"x":"Pearson Correlation","y":"Feature"},
            height=500,
        )
        fig_corr_bar.update_layout(margin=dict(t=10,b=30,l=10,r=10),
                                   coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_corr_bar, use_container_width=True)

        # Correlation heatmap (top features only)
        st.subheader("Correlation Heatmap (Top 10 Features + Target)")
        top10_f = [feature_names[i] for i in sorted_idx[:10]]
        corr_matrix = feat_df[top10_f + ["is_at_risk"]].corr()

        fig_heat, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax, annot_kws={"size":8})
        ax.set_title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True)
        plt.close("all")

        # Feature descriptions
        st.markdown("<hr class='section'>", unsafe_allow_html=True)
        st.subheader("Feature Descriptions")
        feat_desc = {
            "avg_score":              "Average assessment score across all submissions",
            "min_score":              "Minimum assessment score recorded",
            "max_score":              "Maximum assessment score recorded",
            "std_score":              "Standard deviation of assessment scores (consistency indicator)",
            "num_assessments":        "Total number of assessments attempted",
            "num_late_submissions":   "Number of assessments submitted after the due date",
            "avg_submission_timing":  "Average days before/after deadline assessments were submitted",
            "num_banked":             "Number of banked/carried-over assessments",
            "total_clicks":           "Total VLE interaction clicks (engagement volume)",
            "avg_clicks_per_day":     "Average daily VLE interactions",
            "num_active_days":        "Number of unique days the student accessed the VLE",
            "num_resources_accessed": "Number of distinct learning resources accessed",
            "gender":                 "Student gender (encoded)",
            "age_band":               "Age group band of the student",
            "highest_education":      "Highest prior educational qualification",
            "imd_band":               "Index of Multiple Deprivation band (socioeconomic indicator)",
            "disability":             "Whether the student declared a disability",
            "num_of_prev_attempts":   "Number of previous module attempts",
            "studied_credits":        "Total credits being studied in the presentation",
            "registered_early":       "Whether the student registered before the course start",
            "withdrew_early":         "Whether the student withdrew from the course",
            "days_enrolled":          "Days between registration and unregistration (if applicable)",
            "num_registrations":      "Total number of course registrations",
            "module_presentation_length": "Length of the module presentation in days",
        }
        rows = [{"Feature": k, "Description": v}
                for k, v in feat_desc.items() if k in feature_names]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
