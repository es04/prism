import streamlit as st


def render():
    st.markdown("""
    <div class="page-header">
        <h2>ℹ️ About PRISM</h2>
        <p>Predictive Risk Identification for Student Monitoring — System Overview</p>
    </div>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e3a8a 0%,#1d4ed8 60%,#0ea5e9 100%);
                border-radius:14px;padding:2rem;color:white;margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;font-weight:800;letter-spacing:3px;margin-bottom:.5rem;">
            🔮 PRISM
        </div>
        <div style="font-size:1.1rem;opacity:.9;margin-bottom:1rem;">
            Predictive Risk Identification for Student Monitoring
        </div>
        <p style="opacity:.85;max-width:700px;">
            PRISM is an explainable AI system for early identification of academically
            at-risk students. It combines XGBoost machine learning, SHAP explainability,
            and an interactive Streamlit dashboard, built on the Open University Learning
            Analytics Dataset (OULAD).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How It Works Section
    st.subheader("⚙️ How PRISM Works")
    col1, col2, col3, col4 = st.columns(4)

    step_style = """background:white;border-radius:10px;padding:1rem;text-align:center;
                    border:1px solid #e2e8f0;box-shadow:0 2px 6px rgba(0,0,0,.05);height:160px;"""

    with col1:
        st.markdown(f"""
        <div style="{step_style}">
            <div style="font-size:2rem">📂</div>
            <b style="color:#1e40af;">1. Data Collection</b>
            <p style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                OULAD datasets merged into a master dataset with 32,593 student records.
            </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{step_style}">
            <div style="font-size:2rem">⚗️</div>
            <b style="color:#1e40af;">2. Feature Engineering</b>
            <p style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                Behavioral, academic and registration features engineered from raw data.
            </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="{step_style}">
            <div style="font-size:2rem">🤖</div>
            <b style="color:#1e40af;">3. ML Prediction</b>
            <p style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                XGBoost classifier trained with SMOTE balancing and stratified K-fold CV.
            </p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="{step_style}">
            <div style="font-size:2rem">💡</div>
            <b style="color:#1e40af;">4. SHAP Explanation</b>
            <p style="font-size:.8rem;color:#64748b;margin-top:.4rem;">
                Global & local SHAP values explain each prediction to educators.
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs: Technical Specs, Dataset, Methodology, Project Info
    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Technical Specifications",
                                       "📚 Dataset",
                                       "📐 Methodology",
                                       "🏫 Project Information"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Machine Learning**
            | Component | Detail |
            |---|---|
            | Primary Model | XGBoost (Extreme Gradient Boosting) |
            | Baseline Model | Logistic Regression |
            | Class Balancing | SMOTE (Synthetic Minority Oversampling) |
            | Validation | Stratified 5-Fold Cross-Validation |
            | Explainability | SHAP (TreeExplainer) |
            | Risk Thresholds | High ≥65%, Medium 40–65%, Low <40% |
            """)

        with col_b:
            st.markdown("""
            **Technology Stack**
            | Component | Technology |
            |---|---|
            | Language | Python 3.x |
            | Dashboard | Streamlit |
            | ML Library | XGBoost, scikit-learn |
            | Class Balancing | imbalanced-learn |
            | Explainability | SHAP |
            | Visualization | Plotly, Matplotlib, Seaborn |
            | Data Processing | Pandas, NumPy |
            """)

        st.subheader("XGBoost Hyperparameters")
        st.code("""
XGBClassifier(
    n_estimators      = 300,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    eval_metric       = "logloss",
    random_state      = 42,
)
        """, language="python")

    with tab2:
        st.markdown("""
        ### Open University Learning Analytics Dataset (OULAD)

        OULAD is a publicly available educational dataset from the Open University (UK),
        widely used in Educational Data Mining and Learning Analytics research.

        | Dataset | Records | Description |
        |---|---|---|
        | studentInfo.csv | 32,593 | Demographics, background, final results |
        | studentAssessment.csv | 173,912 | Assessment submission scores and timing |
        | studentVle.csv | 10.6M+ | Virtual Learning Environment interactions |
        | studentRegistration.csv | 32,593 | Course registration/withdrawal dates |
        | assessments.csv | 206 | Assessment metadata (type, weight, date) |
        | courses.csv | 22 | Module presentation lengths |
        | vle.csv | ~6,000 | VLE resource/activity metadata |

        ### Feature Categories Used in PRISM

        | Category | Features |
        |---|---|
        | Demographic | gender, age_band, highest_education, imd_band, disability, num_of_prev_attempts, studied_credits |
        | Academic | avg_score, min_score, max_score, std_score, num_assessments, num_late_submissions, num_banked |
        | Behavioural | total_clicks, avg_clicks_per_day, num_active_days, num_resources_accessed |
        | Registration | registered_early, withdrew_early, days_enrolled, num_registrations |
        | Course | module_presentation_length |

        ### Target Variable
        - **At Risk (1):** final_result = "Withdrawn" or "Fail"
        - **Not At Risk (0):** final_result = "Pass" or "Distinction"
        """)

    with tab3:
        st.markdown("""
        ### CRISP-DM Analytical Methodology

        PRISM follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**:

        1. **Business Understanding** — Define the academic monitoring problem and intervention goals.
        2. **Data Understanding** — Explore OULAD datasets, identify quality issues and relevant features.
        3. **Data Preparation** — Clean, encode, impute, engineer features, apply SMOTE balancing.
        4. **Modeling** — Train XGBoost (primary) and Logistic Regression (baseline) classifiers.
        5. **Evaluation** — Assess models on Accuracy, Precision, Recall, F1, ROC-AUC; run SHAP analysis.
        6. **Deployment** — Interactive Streamlit dashboard for educators and academic advisors.

        ### Design Thinking Alignment

        | Phase | PRISM Implementation |
        |---|---|
        | Empathize | Identified educator needs for transparent, actionable predictions |
        | Define | Defined risk classification problem and XAI requirements |
        | Ideate | Explored ML algorithms; selected XGBoost + SHAP |
        | Prototype | Built low-fidelity wireframes and high-fidelity Figma mockups |
        | Test | Functional testing of prediction, explainability, and dashboard modules |

        ### Evaluation Metrics

        Recall is prioritised (minimise false negatives = missing at-risk students).
        F1-score balances precision and recall under class imbalance.
        """)

    with tab4:
        st.markdown("""
        ### Project Information

        | Field | Detail |
        |---|---|
        | System Name | PRISM (Predictive Risk Identification for Student Monitoring) |
        | Programme | Master in Applied Computing (MAC) |
        | Institution | Taylor's University, Malaysia |
        | Project Type | Capstone I Research Prototype |
        | Supervisor | Assoc. Prof. Dr. Shakeel Ahmed |
        | Module Leader | Prof. Dr. Noor Zaman Jhanjhi |

        ### System Modules

        | Module | Description |
        |---|---|
        | Dashboard Overview | High-level KPI cards, risk distribution, model performance |
        | Student Risk Predictions | Searchable student-level prediction table |
        | Prediction Summary | Confusion matrix, ROC curve, CV results |
        | Explainability Insights | Global SHAP beeswarm, local waterfall per student |
        | Feature Importance | SHAP importance, XGBoost built-in, correlation heatmap |
        | Dataset Overview | Raw data previews, distributions, data quality report |
        | About PRISM | System documentation and technical specifications |

        ### Mission Statement
        > *PRISM enables transparent, data-informed early intervention for at-risk students
        > by combining explainable machine learning with an interactive analytics dashboard —
        > empowering educators to act proactively, not reactively.*
        """)
