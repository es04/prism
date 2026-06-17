import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state


def render():
    if not get_or_init_state():
        return

    xm = st.session_state["xgb_metrics"]
    lm = st.session_state["lr_metrics"]
    X  = st.session_state["X_all"]
    y  = st.session_state["y_all"]
    y_te = st.session_state["y_te"]

    st.markdown("""
    <div class="page-header">
        <h2>📊 Prediction Summary</h2>
        <p>Detailed evaluation: confusion matrix, ROC/PR curves, threshold analysis, cross-validation.</p>
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.radio("Select Model", ["XGBoost (Primary)", "Logistic Regression (Baseline)"],
                             horizontal=True)
    m = xm if "XGBoost" in model_choice else lm

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    # KPIs 
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall",    f"{m['recall']*100:.2f}%")
    c4.metric("F1 Score",  f"{m['f1_score']*100:.2f}%")
    c5.metric("ROC-AUC",   f"{m['roc_auc']*100:.2f}%")

    st.markdown("<hr class='section'>", unsafe_allow_html=True)

    tab_cm, tab_roc, tab_pr, tab_thresh, tab_cv = st.tabs([
        "Confusion Matrix", "ROC Curve", "PR Curve",
        "Threshold Analysis", "Cross-Validation",
    ])

    # Confusion Matrix 
    with tab_cm:
        c1, c2 = st.columns([1,1])
        with c1:
            cm = m["conf_matrix"]
            labels = ["Not At Risk","At Risk"]
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                               color_continuous_scale="Blues",
                               labels=dict(x="Predicted",y="Actual",color="Count"),
                               aspect="auto", height=350)
            fig_cm.update_layout(margin=dict(t=10,b=30,l=10,r=10))
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown("**Confusion Matrix Breakdown**")
            st.markdown(f"""
            <div class="tech-box">
            True Negatives  (TN): {tn:,}  — Correctly predicted Not At Risk<br>
            False Positives (FP): {fp:,}  — Incorrectly flagged as At Risk<br>
            False Negatives (FN): {fn:,}  — Missed at-risk students ⚠️<br>
            True Positives  (TP): {tp:,}  — Correctly identified At Risk<br><br>
            Sensitivity (Recall):   {tp/(tp+fn)*100:.2f}%<br>
            Specificity:            {tn/(tn+fp)*100:.2f}%<br>
            PPV (Precision):        {tp/(tp+fp)*100:.2f}%<br>
            NPV:                    {tn/(tn+fn)*100:.2f}%<br>
            False Negative Rate:    {fn/(fn+tp)*100:.2f}%  (missed at-risk)<br>
            False Positive Rate:    {fp/(fp+tn)*100:.2f}%
            </div>
            """, unsafe_allow_html=True)
        st.subheader("Classification Report")
        st.code(m["class_report"], language="text")

    # ROC Curve 
    with tab_roc:
        fpr_x, tpr_x, thresh_x = roc_curve(y_te, xm["y_proba"])
        fpr_l, tpr_l, _        = roc_curve(y_te, lm["y_proba"])
        auc_x = auc(fpr_x, tpr_x)
        auc_l = auc(fpr_l, tpr_l)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_x, y=tpr_x, mode="lines",
            name=f"XGBoost (AUC={auc_x:.4f})",
            line=dict(color="#1d4ed8", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode="lines",
            name=f"Logistic Regression (AUC={auc_l:.4f})",
            line=dict(color="#93c5fd", width=2, dash="dash")))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            name="Random (AUC=0.5)", line=dict(color="gray", dash="dot")))
        # Mark operating point
        fig_roc.add_trace(go.Scatter(x=[fpr_x[np.argmin(np.abs(thresh_x-0.5))]],
                                     y=[tpr_x[np.argmin(np.abs(thresh_x-0.5))]],
                                     mode="markers", name="Threshold=0.5",
                                     marker=dict(color="#ef4444", size=10, symbol="star")))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(x=0.5,y=0.05), height=420, margin=dict(t=10,b=30,l=10,r=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown(f"""
        <div class="info-box">
            <b>AUC Interpretation:</b> XGBoost achieves AUC = {auc_x:.4f}, meaning the model
            correctly ranks a randomly chosen at-risk student above a not-at-risk student
            {auc_x*100:.1f}% of the time.
        </div>
        """, unsafe_allow_html=True)

    # PR Curve 
    with tab_pr:
        prec_x, rec_x, _ = precision_recall_curve(y_te, xm["y_proba"])
        prec_l, rec_l, _ = precision_recall_curve(y_te, lm["y_proba"])
        baseline = y_te.mean()
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec_x, y=prec_x, mode="lines",
            name="XGBoost", line=dict(color="#1d4ed8", width=2.5)))
        fig_pr.add_trace(go.Scatter(x=rec_l, y=prec_l, mode="lines",
            name="Logistic Regression", line=dict(color="#93c5fd", width=2, dash="dash")))
        fig_pr.add_hline(y=baseline, line_dash="dot", line_color="gray",
                         annotation_text=f"Baseline ({baseline:.2f})")
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            height=400, margin=dict(t=10,b=30,l=10,r=10),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.markdown("""
        <div class="info-box">
            The <b>Precision-Recall curve</b> is particularly informative under class imbalance.
            High area under the PR curve means the model is good at identifying at-risk students
            without too many false alarms.
        </div>
        """, unsafe_allow_html=True)

    # Threshold Analysis 
    with tab_thresh:
        st.subheader("Decision Threshold Analysis")
        st.markdown("""
        <div class="info-box">
            By default PRISM uses probability thresholds: High ≥65%, Medium 40–65%, Low &lt;40%.
            This tool shows how precision, recall, and F1 change as the binary threshold varies.
        </div>
        """, unsafe_allow_html=True)

        thresholds = np.arange(0.1, 0.95, 0.05)
        from sklearn.metrics import precision_score, recall_score, f1_score as f1s
        rows_t = []
        for t in thresholds:
            preds = (xm["y_proba"] >= t).astype(int)
            rows_t.append({
                "Threshold": round(t,2),
                "Precision": round(float(precision_score(y_te, preds, zero_division=0)),4),
                "Recall":    round(float(recall_score(y_te, preds, zero_division=0)),4),
                "F1":        round(float(f1s(y_te, preds, zero_division=0)),4),
                "At-Risk Flagged": int(preds.sum()),
            })
        thresh_df = pd.DataFrame(rows_t)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Precision"],
                                    mode="lines+markers", name="Precision", line=dict(color="#3b82f6")))
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Recall"],
                                    mode="lines+markers", name="Recall", line=dict(color="#ef4444")))
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["F1"],
                                    mode="lines+markers", name="F1 Score", line=dict(color="#22c55e")))
        fig_t.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Default 0.5")
        fig_t.add_vline(x=0.65, line_dash="dash", line_color="#ef4444", annotation_text="High Risk 0.65")
        fig_t.update_layout(height=380, yaxis=dict(range=[0,1.05]),
                             xaxis_title="Decision Threshold",
                             margin=dict(t=10,b=30,l=10,r=10))
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)

    # Cross-Validation
    with tab_cv:
        st.subheader("5-Fold Stratified Cross-Validation")
        if "cv_results" not in st.session_state:
            with st.spinner("Running cross-validation …"):
                xgb = st.session_state["xgb_model"]
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv  = cross_validate(xgb, X, y, cv=skf,
                                     scoring=["accuracy","precision","recall","f1","roc_auc"])
                st.session_state["cv_results"] = cv

        cv = st.session_state["cv_results"]
        cv_rows = []
        for mn in ["accuracy","precision","recall","f1","roc_auc"]:
            vals = cv[f"test_{mn}"]
            cv_rows.append({
                "Metric": mn.replace("_"," ").title(),
                **{f"Fold {i+1}": f"{v*100:.2f}%" for i,v in enumerate(vals)},
                "Mean": f"{vals.mean()*100:.2f}%",
                "Std":  f"±{vals.std()*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        # Box plot of CV scores
        box_data = []
        for mn in ["accuracy","precision","recall","f1","roc_auc"]:
            for v in cv[f"test_{mn}"]:
                box_data.append({"Metric": mn.replace("_"," ").title(), "Score": v})
        fig_box = px.box(pd.DataFrame(box_data), x="Metric", y="Score",
                          color="Metric", height=350, points="all",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(showlegend=False, yaxis=dict(range=[0.5,1.05]),
                               margin=dict(t=10,b=20))
        st.plotly_chart(fig_box, use_container_width=True)

        # Model comparison table
        st.subheader("Side-by-Side Comparison")
        cmp = pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            "XGBoost": [f"{xm[m]*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
            "Logistic Regression": [f"{lm[m]*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
            "Δ (XGB − LR)": [f"+{(xm[m]-lm[m])*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
        })
        st.dataframe(cmp, use_container_width=True, hide_index=True)
