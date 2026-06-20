import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
import theme
from utils.model_utils   import (shap_summary_figure, shap_bar_figure,
                                  shap_local_figure, assign_risk_label)


def render():
    theme.inject_theme_css()
    if not get_or_init_state():
        return

    shap_values   = st.session_state["shap_values"]
    X_sample      = st.session_state["X_sample"]
    explainer     = st.session_state["shap_explainer"]
    feature_names = st.session_state["feature_names"]
    pred_df       = st.session_state["pred_df"]
    X_all         = st.session_state["X_all"]

    st.markdown("""
    <div class="page-header">
        <h2>🔍 Explainability Insights</h2>
        <p>Global and local SHAP explanations to interpret model predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🌐 Global SHAP Explanations", "👤 Local Student Explanation"])

    # Global
    with tab1:
        st.subheader("Global Feature Impact (SHAP Summary)")
        st.markdown("""
        <div class="info-box">
            The beeswarm plot below shows how each feature pushes predictions
            toward risk (positive SHAP) or away from risk (negative SHAP).
            Each dot = one student. Colour = feature value (red = high, blue = low).
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            with st.spinner("Rendering SHAP beeswarm …"):
                fig_bee = shap_summary_figure(shap_values, X_sample, feature_names)
                st.pyplot(fig_bee, use_container_width=True)
                plt.close("all")

        with col_b:
            with st.spinner("Rendering SHAP bar …"):
                fig_bar = shap_bar_figure(shap_values, feature_names, top_n=15)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close("all")

        st.markdown("<hr class='section'>", unsafe_allow_html=True)

        # Mean |SHAP| table
        st.subheader("Feature Importance Table (Mean |SHAP Value|)")
        mean_abs = np.abs(shap_values).mean(axis=0)
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": np.round(mean_abs, 5),
            "Importance Rank": np.argsort(np.argsort(-mean_abs)) + 1,
        }).sort_values("Importance Rank")
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

        # SHAP dependence for top feature
        st.markdown("<hr class='section'>", unsafe_allow_html=True)
        st.subheader("SHAP Value Distribution per Feature")
        top_n = min(10, len(feature_names))
        top_feats = list(
            pd.DataFrame({"feat": feature_names, "imp": mean_abs})
            .nlargest(top_n, "imp")["feat"]
        )
        sel_feat = st.selectbox("Select Feature", top_feats)
        feat_idx = feature_names.index(sel_feat)

        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=X_sample[:, feat_idx],
            y=shap_values[:, feat_idx],
            mode="markers",
            marker=dict(
                color=shap_values[:, feat_idx],
                colorscale="RdBu_r",
                size=4, opacity=0.7,
                colorbar=dict(title="SHAP value"),
            ),
        ))
        fig_dep.update_layout(
            xaxis_title=sel_feat,
            yaxis_title="SHAP value",
            height=350, margin=dict(t=10,b=30,l=10,r=10),
        )
        st.plotly_chart(fig_dep, use_container_width=True)

    # Local
    with tab2:
        st.subheader("Individual Student SHAP Explanation")
        st.markdown("""
        <div class="info-box">
            Select a student to see which features drove their risk prediction.
            Red bars push toward <em>At Risk</em>; green bars push away from risk.
        </div>
        """, unsafe_allow_html=True)

        # Student selector
        student_ids = sorted(pred_df["id_student"].unique().tolist())
        selected_id = st.selectbox("Select Student ID", student_ids)

        row = pred_df[pred_df["id_student"] == selected_id].iloc[0]
        student_idx = pred_df.index[pred_df["id_student"] == selected_id][0]

        # Map row position to X_all index
        try:
            X_row = X_all[student_idx]
        except IndexError:
            X_row = X_all[0]

        col_info, col_shap = st.columns([1, 2])

        with col_info:
            risk_color_map = {"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"}
            rc = risk_color_map.get(row["risk_label"], "#6b7280")
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.2rem;
                        border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,.06);">
                <h4 style="margin:0 0 .8rem;color:#1e40af;">👤 Student Profile</h4>
                <p><b>Student ID:</b> {row['id_student']}</p>
                <p><b>Module:</b> {row.get('code_module','N/A')}</p>
                <p><b>Gender:</b> {row.get('gender','N/A')}</p>
                <p><b>Age Band:</b> {row.get('age_band','N/A')}</p>
                <p><b>Actual Result:</b> {row.get('final_result','N/A')}</p>
                <hr style="border-color:#e2e8f0;margin:.5rem 0;">
                <p><b>Predicted Risk:</b>
                    <span style="color:{rc};font-weight:700;">{row['risk_label']}</span></p>
                <p><b>Risk Probability:</b>
                    <span style="color:{rc};font-weight:700;">{row['xgb_prob']:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)

            # Risk meter
            st.markdown("**Risk Confidence Meter**")
            prob = row["xgb_prob"] / 100
            bar_color = "#ef4444" if prob >= 0.65 else "#f59e0b" if prob >= 0.40 else "#22c55e"
            st.markdown(f"""
            <div style="background:#f1f5f9;border-radius:999px;height:18px;overflow:hidden;margin-top:.4rem;">
              <div style="width:{prob*100:.1f}%;background:{bar_color};height:100%;border-radius:999px;
                          transition:width .3s ease;display:flex;align-items:center;justify-content:center;">
                <span style="color:white;font-size:.7rem;font-weight:600;">{prob*100:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_shap:
            with st.spinner("Computing local SHAP …"):
                fig_local = shap_local_figure(explainer, X_row, feature_names,
                                              student_id=selected_id)
                st.pyplot(fig_local, use_container_width=True)
                plt.close("all")

        # Intervention suggestions
        st.markdown("<hr class='section'>", unsafe_allow_html=True)
        st.subheader("💡 Suggested Interventions")

        risk = row["risk_label"]
        if risk == "High Risk":
            st.markdown("""
            <div class="warn-box">
            🚨 <b>High Risk – Immediate Intervention Recommended</b><br>
            • Schedule an urgent academic counselling session.<br>
            • Assign a peer mentor or academic buddy.<br>
            • Check engagement with learning materials on VLE.<br>
            • Review assessment submission history and provide targeted feedback.<br>
            • Consider referral to student support services.
            </div>
            """, unsafe_allow_html=True)
        elif risk == "Medium Risk":
            st.markdown("""
            <div class="info-box">
            ⚠️ <b>Medium Risk – Proactive Monitoring Advised</b><br>
            • Schedule a check-in meeting within the next two weeks.<br>
            • Encourage participation in online VLE resources.<br>
            • Monitor upcoming assessment submission deadlines.<br>
            • Provide optional academic enrichment materials.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
            ✅ <b>Low Risk – Continue Standard Monitoring</b><br>
            • Student is performing well – continue regular progress reviews.<br>
            • Encourage participation in advanced or enrichment modules.<br>
            • Maintain positive engagement with VLE resources.
            </div>
            """, unsafe_allow_html=True)
