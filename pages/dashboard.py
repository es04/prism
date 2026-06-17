import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state


def render():
    if not get_or_init_state():
        return

    pred_df       = st.session_state["pred_df"]
    xm            = st.session_state["xgb_metrics"]
    lm            = st.session_state["lr_metrics"]
    feature_names = st.session_state["feature_names"]
    shap_values   = st.session_state["shap_values"]
    master        = st.session_state["master_df"]

    st.markdown("""
    <div class="page-header">
        <h2>🏠 Dashboard Overview</h2>
        <p>High-level summary of student risk classifications and model performance.</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary Metrics Row
    total   = len(pred_df)
    high    = int((pred_df["risk_label"]=="High Risk").sum())
    med     = int((pred_df["risk_label"]=="Medium Risk").sum())
    low     = int((pred_df["risk_label"]=="Low Risk").sum())
    avg_c   = float(pred_df["xgb_prob"].mean())
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_feat = feature_names[int(np.argmax(mean_abs))]

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("👥 Total Students",    f"{total:,}")
    c2.metric("🔴 High Risk",         f"{high:,}",  f"{high/total*100:.1f}%")
    c3.metric("🟡 Medium Risk",       f"{med:,}",   f"{med/total*100:.1f}%")
    c4.metric("🟢 Low Risk",          f"{low:,}",   f"{low/total*100:.1f}%")
    c5.metric("🎯 Avg Confidence",    f"{avg_c:.1f}%")
    c6.metric("🔑 Top Feature",       top_feat[:14])

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # Row 1: Risk distribution pie + performance comparison bar 
    ca, cb = st.columns(2)
    with ca:
        st.subheader("Risk Distribution")
        counts = pred_df["risk_label"].value_counts().reset_index()
        counts.columns = ["Risk Level","Count"]
        fig = px.pie(counts, names="Risk Level", values="Count",
                     color="Risk Level",
                     color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
                     hole=0.45, height=320)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=True, margin=dict(t=20,b=20,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.subheader("Model Performance Comparison")
        metrics = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        xv = [xm["accuracy"],xm["precision"],xm["recall"],xm["f1_score"],xm["roc_auc"]]
        lv = [lm["accuracy"],lm["precision"],lm["recall"],lm["f1_score"],lm["roc_auc"]]
        fig2 = go.Figure(data=[
            go.Bar(name="XGBoost",             x=metrics, y=xv, marker_color="#1d4ed8",
                   text=[f"{v*100:.1f}%" for v in xv], textposition="outside"),
            go.Bar(name="Logistic Regression", x=metrics, y=lv, marker_color="#93c5fd",
                   text=[f"{v*100:.1f}%" for v in lv], textposition="outside"),
        ])
        fig2.update_layout(barmode="group", height=320,
                           yaxis=dict(range=[0,1.15],tickformat=".0%"),
                           legend=dict(orientation="h",yanchor="bottom",y=1),
                           margin=dict(t=30,b=20,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Module-wise risk distribution + confidence histogram
    cc, cd = st.columns(2)
    with cc:
        st.subheader("Risk Distribution by Module")
        mr = pred_df.groupby(["code_module","risk_label"]).size().reset_index(name="Count")
        fig3 = px.bar(mr, x="code_module", y="Count", color="risk_label",
                      color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
                      barmode="stack", height=320,
                      labels={"code_module":"Module","risk_label":"Risk Level"})
        fig3.update_layout(margin=dict(t=10,b=30,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with cd:
        st.subheader("Prediction Confidence Distribution")
        fig4 = px.histogram(pred_df, x="xgb_prob", nbins=30,
                             color_discrete_sequence=["#2563eb"], height=320,
                             labels={"xgb_prob":"Risk Probability (%)"})
        fig4.add_vline(x=65, line_color="#ef4444", line_dash="dash",
                       annotation_text="High (65%)", annotation_position="top right")
        fig4.add_vline(x=40, line_color="#f59e0b", line_dash="dash",
                       annotation_text="Medium (40%)", annotation_position="top right")
        fig4.update_layout(margin=dict(t=10,b=30,l=10,r=10))
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: score vs clicks scatter 
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    st.subheader("Score vs VLE Engagement by Risk Level")
    if "avg_score" in pred_df.columns and "total_clicks" in pred_df.columns:
        scat_df = pred_df.dropna(subset=["avg_score","total_clicks"])
        scat_sample = scat_df.sample(min(2000, len(scat_df)), random_state=42)
        fig5 = px.scatter(
            scat_sample, x="avg_score", y="total_clicks", color="risk_label",
            color_discrete_map={"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"},
            opacity=0.5, height=350, size_max=5,
            labels={"avg_score":"Average Assessment Score","total_clicks":"Total VLE Clicks"},
        )
        fig5.update_layout(margin=dict(t=10,b=30,l=10,r=10))
        st.plotly_chart(fig5, use_container_width=True)

    # Row 4: Trend over time if data has date 
    st.subheader("At-Risk Rate by Prior Attempts")
    if "num_of_prev_attempts" in pred_df.columns:
        attempt_risk = pred_df.groupby("num_of_prev_attempts").apply(
            lambda x: (x["risk_label"]=="High Risk").mean()*100
        ).reset_index()
        attempt_risk.columns = ["Prev Attempts","High Risk Rate (%)"]
        fig6 = px.bar(attempt_risk, x="Prev Attempts", y="High Risk Rate (%)",
                      color="High Risk Rate (%)", color_continuous_scale="Reds",
                      height=280, labels={"Prev Attempts":"Number of Previous Attempts"})
        fig6.update_layout(margin=dict(t=10,b=30), coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

    # Summary metrics table 
    st.markdown("<hr class='section'>", unsafe_allow_html=True)
    st.subheader("📌 Model Performance Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{xm['accuracy']*100:.2f}%",  f"+{(xm['accuracy']-lm['accuracy'])*100:.2f}% vs LR")
    c2.metric("Precision", f"{xm['precision']*100:.2f}%", f"+{(xm['precision']-lm['precision'])*100:.2f}% vs LR")
    c3.metric("Recall",    f"{xm['recall']*100:.2f}%",    f"+{(xm['recall']-lm['recall'])*100:.2f}% vs LR")
    c4.metric("F1 Score",  f"{xm['f1_score']*100:.2f}%",  f"+{(xm['f1_score']-lm['f1_score'])*100:.2f}% vs LR")
    c5.metric("ROC-AUC",   f"{xm['roc_auc']*100:.2f}%",   f"+{(xm['roc_auc']-lm['roc_auc'])*100:.2f}% vs LR")

    st.subheader("🚨 Top 10 Highest-Risk Students")
    top10 = pred_df.nlargest(10,"xgb_prob")[
        ["id_student","code_module","gender","age_band","risk_label","xgb_prob","final_result"]
    ].copy()
    top10.columns = ["Student ID","Module","Gender","Age Band","Risk Level","Risk Prob (%)","Actual Result"]
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)
