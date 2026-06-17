"""
PRISM – Page: Model Training & Tuning
Live hyperparameter control panel, SMOTE visualisation, training diagnostics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from pages.state_manager import get_or_init_state
from utils.model_utils import (
    DEFAULT_XGB_PARAMS,
    cross_validate_model,
    evaluate_model,
    assign_risk_label,
)


def render():
    st.markdown(
        """
    <div class="page-header">
        <h2>⚙️ Model Training & Tuning</h2>
        <p>Configure XGBoost hyperparameters, inspect SMOTE balancing, and re-train interactively.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Need data first ──────────────────────────────────────────────────────
    if not st.session_state.get("prism_ready"):
        st.warning("⚠️ Models not yet trained. Running default pipeline …")
        if not get_or_init_state():
            return

    tab_hp, tab_smote, tab_cv, tab_compare, tab_log = st.tabs(
        [
            "🎛️ Hyperparameters",
            "⚖️ SMOTE Balancing",
            "🔁 Cross-Validation",
            "📊 Model Comparison",
            "📝 Training Log",
        ]
    )

    # ════════════════════════════════════════════════════════
    # TAB 1 – Hyperparameters
    # ════════════════════════════════════════════════════════
    with tab_hp:
        st.subheader("XGBoost Hyperparameter Control Panel")
        st.markdown(
            """
        <div class="info-box">
            Adjust hyperparameters and click <b>Re-train Model</b> to see the effect on performance.
            Changes are applied immediately to the XGBoost classifier.
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Tree Structure**")
            n_est = st.slider(
                "n_estimators (trees)",
                50,
                500,
                300,
                50,
                help="Number of boosting rounds. More trees = better fit but slower.",
            )
            max_d = st.slider(
                "max_depth",
                2,
                12,
                6,
                1,
                help="Maximum depth of each tree. Deeper = more complex.",
            )
            min_cw = st.slider(
                "min_child_weight",
                1,
                10,
                3,
                1,
                help="Minimum sum of instance weight in a leaf. Higher = more conservative.",
            )

        with col2:
            st.markdown("**Learning & Sampling**")
            lr = st.slider(
                "learning_rate (eta)",
                0.005,
                0.3,
                0.05,
                0.005,
                format="%.3f",
                help="Step size shrinkage. Lower = slower but more robust.",
            )
            subsamp = st.slider(
                "subsample",
                0.4,
                1.0,
                0.8,
                0.05,
                help="Fraction of rows sampled per tree. Lower = less overfitting.",
            )
            col_samp = st.slider(
                "colsample_bytree",
                0.4,
                1.0,
                0.8,
                0.05,
                help="Fraction of features used per tree.",
            )

        with col3:
            st.markdown("**Regularisation**")
            gamma = st.slider(
                "gamma (min split loss)",
                0.0,
                2.0,
                0.1,
                0.05,
                help="Minimum loss reduction to split a node. Higher = fewer splits.",
            )
            reg_a = st.slider(
                "reg_alpha (L1)",
                0.0,
                2.0,
                0.1,
                0.05,
                help="L1 regularisation on leaf weights.",
            )
            reg_l = st.slider(
                "reg_lambda (L2)",
                0.0,
                5.0,
                1.0,
                0.1,
                help="L2 regularisation on leaf weights.",
            )

        test_sz = (
            st.slider(
                "Test set size (%)",
                10,
                35,
                20,
                5,
                help="Proportion of data held out for evaluation.",
            )
            / 100
        )

        st.markdown("---")
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            retrain = st.button(
                "🔄 Re-train with These Parameters",
                type="primary",
                use_container_width=True,
            )
        with col_info:
            st.markdown(
                f"""
            <div class="tech-box">
            XGBClassifier(<br>
            &nbsp;&nbsp;n_estimators={n_est}, max_depth={max_d},<br>
            &nbsp;&nbsp;learning_rate={lr}, subsample={subsamp},<br>
            &nbsp;&nbsp;colsample_bytree={col_samp},<br>
            &nbsp;&nbsp;min_child_weight={min_cw},<br>
            &nbsp;&nbsp;gamma={gamma}, reg_alpha={reg_a},<br>
            &nbsp;&nbsp;reg_lambda={reg_l},<br>
            &nbsp;&nbsp;eval_metric="logloss", random_state=42<br>
            )
            </div>
            """,
                unsafe_allow_html=True,
            )

        if retrain:
            custom = {
                "n_estimators": n_est,
                "max_depth": max_d,
                "learning_rate": lr,
                "subsample": subsamp,
                "colsample_bytree": col_samp,
                "min_child_weight": min_cw,
                "gamma": gamma,
                "reg_alpha": reg_a,
                "reg_lambda": reg_l,
            }
            st.session_state["prism_ready"] = False
            with st.spinner("Re-training …"):
                get_or_init_state(
                    force_retrain=True, xgb_params=custom, test_size=test_sz
                )
            st.success("✅ Re-training complete! All pages now reflect the new model.")
            st.rerun()

        # Current vs default comparison
        st.markdown("<hr class='section'>", unsafe_allow_html=True)
        st.subheader("Parameter Comparison: Current vs Default")
        default = DEFAULT_XGB_PARAMS
        current = {
            "n_estimators": n_est,
            "max_depth": max_d,
            "learning_rate": lr,
            "subsample": subsamp,
            "colsample_bytree": col_samp,
            "min_child_weight": min_cw,
            "gamma": gamma,
            "reg_alpha": reg_a,
            "reg_lambda": reg_l,
        }
        rows = []
        for k, dv in {k: v for k, v in default.items() if k in current}.items():
            cv = current[k]
            rows.append(
                {
                    "Parameter": k,
                    "Default": dv,
                    "Current": cv,
                    "Changed": "🔴 Yes" if cv != dv else "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 – SMOTE
    # ════════════════════════════════════════════════════════
    with tab_smote:
        st.subheader("SMOTE Class Balancing Analysis")
        st.markdown(
            """
        <div class="info-box">
            <b>SMOTE</b> (Synthetic Minority Oversampling Technique) generates synthetic
            samples for the minority class (At Risk) to address class imbalance in the
            training set, improving Recall for at-risk students.
        </div>
        """,
            unsafe_allow_html=True,
        )

        y_all = st.session_state.get("y_all")
        if y_all is not None:
            # Original distribution
            orig = {
                "Not At Risk": int((y_all == 0).sum()),
                "At Risk": int((y_all == 1).sum()),
            }

            sb = st.session_state.get("smote_before", orig)
            sa = st.session_state.get(
                "smote_after",
                {
                    "Not At Risk": sb["Not At Risk"],
                    "At Risk": sb["Not At Risk"],  # perfectly balanced after SMOTE
                },
            )

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Before SMOTE (Training split)**")
                tot_b = sum(sb.values())
                fig_b = px.pie(
                    names=list(sb.keys()),
                    values=list(sb.values()),
                    color=list(sb.keys()),
                    color_discrete_map={"Not At Risk": "#3b82f6", "At Risk": "#ef4444"},
                    hole=0.4,
                    height=280,
                )
                fig_b.update_traces(textinfo="percent+label")
                fig_b.update_layout(margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
                st.plotly_chart(fig_b, use_container_width=True)
                st.metric("Not At Risk", f"{sb['Not At Risk']:,}")
                st.metric("At Risk", f"{sb['At Risk']:,}")
                ir = sb["Not At Risk"] / max(sb["At Risk"], 1)
                st.metric("Imbalance Ratio", f"{ir:.2f}:1")

            with c2:
                st.markdown("**After SMOTE (Training split)**")
                fig_a = px.pie(
                    names=list(sa.keys()),
                    values=list(sa.values()),
                    color=list(sa.keys()),
                    color_discrete_map={"Not At Risk": "#3b82f6", "At Risk": "#ef4444"},
                    hole=0.4,
                    height=280,
                )
                fig_a.update_traces(textinfo="percent+label")
                fig_a.update_layout(margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
                st.plotly_chart(fig_a, use_container_width=True)
                st.metric("Not At Risk", f"{sa['Not At Risk']:,}")
                st.metric("At Risk", f"{sa['At Risk']:,}")
                st.metric("Imbalance Ratio", "1.00:1 (balanced)")

            # Before vs After bar
            st.markdown("**Side-by-Side Comparison**")
            comp_df = pd.DataFrame(
                {
                    "Class": ["Not At Risk", "At Risk", "Not At Risk", "At Risk"],
                    "Count": [
                        sb["Not At Risk"],
                        sb["At Risk"],
                        sa["Not At Risk"],
                        sa["At Risk"],
                    ],
                    "Stage": [
                        "Before SMOTE",
                        "Before SMOTE",
                        "After SMOTE",
                        "After SMOTE",
                    ],
                }
            )
            fig_comp = px.bar(
                comp_df,
                x="Class",
                y="Count",
                color="Stage",
                barmode="group",
                color_discrete_map={
                    "Before SMOTE": "#94a3b8",
                    "After SMOTE": "#2563eb",
                },
                height=300,
            )
            fig_comp.update_layout(margin=dict(t=5, b=20, l=5, r=5))
            st.plotly_chart(fig_comp, use_container_width=True)

            # Technical explanation
            with st.expander("🔬 How SMOTE Works"):
                st.markdown("""
                SMOTE works by:
                1. Selecting a minority class sample
                2. Finding its **k nearest neighbours** (default k=5) in feature space
                3. Generating a synthetic sample along the line segment between the
                   original sample and a randomly selected neighbour

                This creates realistic synthetic samples rather than simple duplication,
                reducing overfitting while improving minority class recall.

                **PRISM Configuration:** `SMOTE(random_state=42)` — applied only to the
                training split (never the test set) to prevent data leakage.
                """)

    # ════════════════════════════════════════════════════════
    # TAB 3 – Cross-Validation
    # ════════════════════════════════════════════════════════
    with tab_cv:
        st.subheader("Stratified K-Fold Cross-Validation")
        st.markdown(
            """
        <div class="info-box">
            Cross-validation evaluates model generalisation by training/testing on
            different data splits. Stratified folds preserve class distribution across splits.
        </div>
        """,
            unsafe_allow_html=True,
        )

        n_folds = st.slider("Number of CV folds", 3, 10, 5)

        if st.button("▶️ Run Cross-Validation", type="primary"):
            xgb = st.session_state["xgb_model"]
            X_all = st.session_state["X_all"]
            y_all = st.session_state["y_all"]
            with st.spinner(f"Running {n_folds}-fold stratified CV …"):
                cv_res = cross_validate_model(xgb, X_all, y_all, cv=n_folds)
                st.session_state["cv_results_detailed"] = cv_res
                st.session_state["cv_n_folds"] = n_folds

        cv_res = st.session_state.get("cv_results_detailed")
        if cv_res:
            n = st.session_state.get("cv_n_folds", 5)
            rows = []
            for metric, data in cv_res.items():
                row = {
                    "Metric": metric.replace("_", " ").title(),
                    "Mean": f"{data['mean']*100:.2f}%",
                    "Std": f"±{data['std']*100:.2f}%",
                }
                for i, v in enumerate(data["folds"]):
                    row[f"Fold {i+1}"] = f"{v*100:.2f}%"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Fold performance chart
            st.markdown("**Per-Fold Performance Chart**")
            fold_data = []
            for metric, data in cv_res.items():
                for i, v in enumerate(data["folds"]):
                    fold_data.append(
                        {
                            "Metric": metric.replace("_", " ").title(),
                            "Fold": f"Fold {i+1}",
                            "Score": v,
                        }
                    )
            fig_cv = px.line(
                pd.DataFrame(fold_data),
                x="Fold",
                y="Score",
                color="Metric",
                markers=True,
                height=350,
                labels={"Score": "Score (0–1)"},
            )
            fig_cv.update_layout(
                margin=dict(t=5, b=20, l=5, r=5), yaxis_range=[0.5, 1.05]
            )
            st.plotly_chart(fig_cv, use_container_width=True)

            # Radar chart of mean metrics
            st.markdown("**Mean Metric Radar Chart**")
            labels = [m.replace("_", " ").title() for m in cv_res.keys()]
            vals = [cv_res[m]["mean"] for m in cv_res.keys()]
            fig_r = go.Figure(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    fillcolor="rgba(37,99,235,0.15)",
                    line=dict(color="#2563eb"),
                    name="XGBoost",
                )
            )
            fig_r.update_layout(
                polar=dict(radialaxis=dict(range=[0.5, 1.0])),
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Click **Run Cross-Validation** above to compute fold results.")

    # ════════════════════════════════════════════════════════
    # TAB 4 – Model Comparison
    # ════════════════════════════════════════════════════════
    with tab_compare:
        st.subheader("XGBoost vs Logistic Regression — Detailed Comparison")
        xm = st.session_state["xgb_metrics"]
        lm = st.session_state["lr_metrics"]

        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        xv = [xm[m] for m in metrics]
        lv = [lm[m] for m in metrics]

        # Grouped bar
        fig_cmp = go.Figure(
            data=[
                go.Bar(
                    name="XGBoost",
                    x=labels,
                    y=xv,
                    marker_color="#1d4ed8",
                    text=[f"{v*100:.1f}%" for v in xv],
                    textposition="outside",
                ),
                go.Bar(
                    name="Logistic Regression",
                    x=labels,
                    y=lv,
                    marker_color="#93c5fd",
                    text=[f"{v*100:.1f}%" for v in lv],
                    textposition="outside",
                ),
            ]
        )
        fig_cmp.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
            legend=dict(orientation="h", yanchor="bottom", y=1),
            height=380,
            margin=dict(t=30, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Delta table
        st.markdown("**Performance Delta (XGBoost − Logistic Regression)**")
        delta_rows = []
        for m, l in zip(metrics, labels):
            diff = xm[m] - lm[m]
            delta_rows.append(
                {
                    "Metric": l,
                    "XGBoost": f"{xm[m]*100:.2f}%",
                    "Logistic Regression": f"{lm[m]*100:.2f}%",
                    "Delta": f"{'+'if diff>=0 else ''}{diff*100:.2f}%",
                    "Winner": "XGBoost 🏆" if diff > 0 else "Logistic Regression 🏆",
                }
            )
        st.dataframe(
            pd.DataFrame(delta_rows), use_container_width=True, hide_index=True
        )

        # Radar
        st.markdown("**Radar Comparison**")
        fig_rad = go.Figure()
        fig_rad.add_trace(
            go.Scatterpolar(
                r=xv + [xv[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(29,78,216,0.15)",
                line=dict(color="#1d4ed8"),
                name="XGBoost",
            )
        )
        fig_rad.add_trace(
            go.Scatterpolar(
                r=lv + [lv[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(147,197,253,0.15)",
                line=dict(color="#93c5fd", dash="dash"),
                name="Logistic Regression",
            )
        )
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(range=[0.5, 1.05])),
            height=380,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_rad, use_container_width=True)

        # Side-by-side classification reports
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**XGBoost Classification Report**")
            st.code(xm["class_report"], language="text")
        with c2:
            st.markdown("**Logistic Regression Classification Report**")
            st.code(lm["class_report"], language="text")

    # ════════════════════════════════════════════════════════
    # TAB 5 – Training Log
    # ════════════════════════════════════════════════════════
    with tab_log:
        st.subheader("Training Configuration Log")
        xm = st.session_state["xgb_metrics"]
        lm = st.session_state["lr_metrics"]
        master = st.session_state["master_df"]
        fn = st.session_state["feature_names"]

        st.markdown("**Dataset Statistics**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(master):,}")
        c2.metric("Features Used", f"{len(fn)}")
        c3.metric(
            "Train Samples",
            (
                f"{st.session_state.get('train_size','N/A'):,}"
                if st.session_state.get("train_size")
                else "N/A"
            ),
        )
        c4.metric(
            "Test Samples",
            (
                f"{st.session_state.get('test_size_n','N/A'):,}"
                if st.session_state.get("test_size_n")
                else "N/A"
            ),
        )

        st.markdown("**Feature List Used for Training**")
        feat_df = pd.DataFrame(
            {
                "Index": range(1, len(fn) + 1),
                "Feature Name": fn,
                "Category": [
                    (
                        "Demographic"
                        if f
                        in [
                            "gender",
                            "age_band",
                            "highest_education",
                            "imd_band",
                            "disability",
                            "num_of_prev_attempts",
                            "studied_credits",
                        ]
                        else (
                            "Academic"
                            if f
                            in [
                                "avg_score",
                                "min_score",
                                "max_score",
                                "std_score",
                                "num_assessments",
                                "num_late_submissions",
                                "avg_submission_timing",
                                "num_banked",
                            ]
                            else (
                                "Behavioural"
                                if f
                                in [
                                    "total_clicks",
                                    "avg_clicks_per_day",
                                    "num_active_days",
                                    "num_resources_accessed",
                                ]
                                else (
                                    "Registration"
                                    if f
                                    in [
                                        "registered_early",
                                        "withdrew_early",
                                        "days_enrolled",
                                        "num_registrations",
                                    ]
                                    else "Course"
                                )
                            )
                        )
                    )
                    for f in fn
                ],
            }
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("**Final Model Configuration**")
        xgb_model = st.session_state["xgb_model"]
        params = xgb_model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        st.markdown("**Pipeline Steps**")
        steps = [
            ("1. Data Loading", "Load 7 OULAD CSV files from data/ or uploaded files"),
            (
                "2. Feature Engineering",
                "Aggregate assessment, VLE, and registration features per student",
            ),
            (
                "3. Target Encoding",
                "Map final_result → binary risk label (Fail/Withdrawn=1, Pass/Distinction=0)",
            ),
            ("4. Label Encoding", "LabelEncoder applied to categorical columns"),
            (
                "5. Missing Imputation",
                "SimpleImputer(strategy='median') on all numeric features",
            ),
            ("6. Standard Scaling", "StandardScaler applied to all numeric features"),
            (
                "7. Train-Test Split",
                f"StratifiedShuffleSplit(test_size={st.session_state.get('test_size_n','N/A')})",
            ),
            ("8. SMOTE", "SMOTE(random_state=42) applied to training split only"),
            ("9. XGBoost Training", "XGBClassifier with tuned hyperparameters"),
            ("10. LR Training", "LogisticRegression(C=1.0, class_weight='balanced')"),
            ("11. SHAP", "TreeExplainer on XGBoost for global + local explanations"),
        ]
        step_df = pd.DataFrame(steps, columns=["Step", "Description"])
        st.dataframe(step_df, use_container_width=True, hide_index=True)
