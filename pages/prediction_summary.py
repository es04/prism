"""
PRISM v2 – Prediction Summary
Redesigned with theme system.
"""

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
import theme


def _fig_base(fig, height=380, **extra):
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

    xm   = st.session_state["xgb_metrics"]
    lm   = st.session_state["lr_metrics"]
    X    = st.session_state["X_all"]
    y    = st.session_state["y_all"]
    y_te = st.session_state["y_te"]
    dark = theme.is_dark()

    theme.page_header("📊", "Prediction Summary",
                      "Confusion matrix, ROC/PR curves, threshold analysis, and cross-validation.")

    model_choice = st.radio("Model", ["XGBoost (Primary)", "Logistic Regression (Baseline)"],
                            horizontal=True)
    m = xm if "XGBoost" in model_choice else lm

    theme.section_div()

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: theme.kpi_card("Accuracy",  f"{m['accuracy']*100:.2f}%",  accent="blue")
    with k2: theme.kpi_card("Precision", f"{m['precision']*100:.2f}%", accent="teal")
    with k3: theme.kpi_card("Recall",    f"{m['recall']*100:.2f}%",    accent="amber")
    with k4: theme.kpi_card("F1 Score",  f"{m['f1_score']*100:.2f}%",  accent="violet")
    with k5: theme.kpi_card("ROC-AUC",   f"{m['roc_auc']*100:.2f}%",   accent="red")

    theme.section_div()

    tab_cm, tab_roc, tab_pr, tab_thresh, tab_cv = st.tabs([
        "Confusion Matrix", "ROC Curve", "PR Curve", "Threshold Analysis", "Cross-Validation",
    ])

    # Confusion Matrix
    with tab_cm:
        c1, c2 = st.columns([1, 1])
        with c1:
            cm = m["conf_matrix"]
            labels = ["Not At Risk", "At Risk"]
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels,
                               color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               aspect="auto", height=340)
            _fig_base(fig_cm, height=340, margin=dict(t=10, b=30, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            tn, fp, fn, tp = cm.ravel()
            theme.info_box(f"""
<strong>Confusion Matrix Breakdown</strong><br><br>
True Negatives  (TN): <strong>{tn:,}</strong> — Correct Not At Risk<br>
False Positives (FP): <strong>{fp:,}</strong> — Incorrectly flagged<br>
False Negatives (FN): <strong>{fn:,}</strong> — Missed at-risk ⚠️<br>
True Positives  (TP): <strong>{tp:,}</strong> — Correct At Risk<br><br>
Sensitivity (Recall):  {tp/(tp+fn)*100:.2f}%<br>
Specificity:           {tn/(tn+fp)*100:.2f}%<br>
Precision (PPV):       {tp/(tp+fp)*100:.2f}%<br>
NPV:                   {tn/(tn+fn)*100:.2f}%<br>
False Negative Rate:   {fn/(fn+tp)*100:.2f}%<br>
False Positive Rate:   {fp/(fp+tn)*100:.2f}%
""", "info")
        st.subheader("Classification Report")
        st.code(m["class_report"], language="text")

    # ROC Curve
    with tab_roc:
        fpr_x, tpr_x, thresh_x = roc_curve(y_te, xm["y_proba"])
        fpr_l, tpr_l, _         = roc_curve(y_te, lm["y_proba"])
        auc_x = auc(fpr_x, tpr_x)
        auc_l = auc(fpr_l, tpr_l)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_x, y=tpr_x, mode="lines",
            name=f"XGBoost (AUC={auc_x:.4f})", line=dict(color="#2563EB", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=fpr_l, y=tpr_l, mode="lines",
            name=f"Logistic Regression (AUC={auc_l:.4f})", line=dict(color="#93C5FD", width=2, dash="dash")))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
            name="Random (0.5)", line=dict(color="gray", dash="dot")))
        idx = np.argmin(np.abs(thresh_x - 0.5))
        fig_roc.add_trace(go.Scatter(x=[fpr_x[idx]], y=[tpr_x[idx]], mode="markers",
            name="Threshold=0.5", marker=dict(color="#DC2626", size=10, symbol="star")))
        _fig_base(fig_roc, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                  legend=dict(x=0.5, y=0.05))
        st.plotly_chart(fig_roc, use_container_width=True)
        theme.info_box(f"<b>AUC Interpretation:</b> XGBoost achieves AUC = {auc_x:.4f}, meaning the model correctly ranks a randomly chosen at-risk student above a not-at-risk student {auc_x*100:.1f}% of the time.")

    # PR Curve
    with tab_pr:
        prec_x, rec_x, _ = precision_recall_curve(y_te, xm["y_proba"])
        prec_l, rec_l, _ = precision_recall_curve(y_te, lm["y_proba"])
        baseline = y_te.mean()
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec_x, y=prec_x, mode="lines",
            name="XGBoost", line=dict(color="#2563EB", width=2.5)))
        fig_pr.add_trace(go.Scatter(x=rec_l, y=prec_l, mode="lines",
            name="Logistic Regression", line=dict(color="#93C5FD", width=2, dash="dash")))
        fig_pr.add_hline(y=baseline, line_dash="dot", line_color="gray",
                         annotation_text=f"Baseline ({baseline:.2f})")
        _fig_base(fig_pr, xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True)
        theme.info_box("The <b>Precision-Recall curve</b> is particularly informative under class imbalance. High area under the PR curve means the model finds at-risk students with fewer false alarms.")

    # Threshold Analysis
    with tab_thresh:
        theme.info_box("PRISM uses probability thresholds: High ≥65%, Medium 40–65%, Low <40%. This shows how precision, recall, and F1 shift as the binary threshold changes.", "info")

        thresholds = np.arange(0.1, 0.95, 0.05)
        from sklearn.metrics import precision_score, recall_score, f1_score as f1s
        rows_t = []
        for t in thresholds:
            preds = (xm["y_proba"] >= t).astype(int)
            rows_t.append({
                "Threshold": round(t, 2),
                "Precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
                "Recall":    round(float(recall_score(y_te, preds, zero_division=0)), 4),
                "F1":        round(float(f1s(y_te, preds, zero_division=0)), 4),
                "Flagged":   int(preds.sum()),
            })
        thresh_df = pd.DataFrame(rows_t)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Precision"],
                                    mode="lines+markers", name="Precision", line=dict(color="#2563EB")))
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Recall"],
                                    mode="lines+markers", name="Recall", line=dict(color="#DC2626")))
        fig_t.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["F1"],
                                    mode="lines+markers", name="F1 Score", line=dict(color="#0D9488")))
        fig_t.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Default 0.5")
        fig_t.add_vline(x=0.65, line_dash="dash", line_color="#DC2626", annotation_text="High Risk 0.65")
        _fig_base(fig_t, yaxis=dict(range=[0, 1.05]), xaxis_title="Decision Threshold")
        st.plotly_chart(fig_t, use_container_width=True)
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)

    # Cross-Validation
    with tab_cv:
        if "cv_results" not in st.session_state:
            with st.spinner("Running 5-fold cross-validation …"):
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
                **{f"Fold {i+1}": f"{v*100:.2f}%" for i, v in enumerate(vals)},
                "Mean": f"{vals.mean()*100:.2f}%",
                "Std":  f"±{vals.std()*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        box_data = []
        for mn in ["accuracy","precision","recall","f1","roc_auc"]:
            for v in cv[f"test_{mn}"]:
                box_data.append({"Metric": mn.replace("_"," ").title(), "Score": v})
        fig_box = px.box(pd.DataFrame(box_data), x="Metric", y="Score",
                          color="Metric", points="all",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        _fig_base(fig_box, height=320, showlegend=False, yaxis=dict(range=[0.5, 1.05]))
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Side-by-Side Comparison")
        cmp = pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            "XGBoost": [f"{xm[m]*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
            "Logistic Regression": [f"{lm[m]*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
            "Δ (XGB − LR)": [f"+{(xm[m]-lm[m])*100:.2f}%" for m in ["accuracy","precision","recall","f1_score","roc_auc"]],
        })
        st.dataframe(cmp, use_container_width=True, hide_index=True)
