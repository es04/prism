"""
PRISM v2 – About PRISM (redesigned)
"""

import streamlit as st
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import theme


def render():
    dark = theme.is_dark()

    theme.page_header("ℹ️", "About PRISM",
                      "Predictive Risk Identification for Student Monitoring — System Overview")

    # Hero
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0F172A 0%,#1E2D4A 60%,#1E3A5F 100%);
            border-radius:var(--radius-lg);padding:2rem 2.2rem;color:white;
            margin-bottom:1.5rem;border:1px solid rgba(255,255,255,0.06);
            position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;
                background:var(--accent-bar);"></div>
    <div style="font-size:2.2rem;font-weight:800;letter-spacing:4px;
                margin-bottom:.4rem;font-family:'JetBrains Mono',monospace;">🔮 PRISM</div>
    <div style="font-size:1.05rem;opacity:.85;margin-bottom:.9rem;">
        Predictive Risk Identification for Student Monitoring
    </div>
    <p style="opacity:.75;max-width:680px;line-height:1.65;font-size:0.9rem;">
        PRISM is an explainable AI system for early identification of academically at-risk students.
        It combines XGBoost machine learning, SHAP explainability, and an interactive dashboard,
        built on the Open University Learning Analytics Dataset (OULAD) with 32,593 student records.
    </p>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:1.1rem;">
        <span style="background:rgba(37,99,235,0.25);border:1px solid rgba(37,99,235,0.4);
                     border-radius:999px;padding:4px 14px;font-size:0.78rem;font-weight:600;">
            93.05% Accuracy
        </span>
        <span style="background:rgba(13,148,136,0.25);border:1px solid rgba(13,148,136,0.4);
                     border-radius:999px;padding:4px 14px;font-size:0.78rem;font-weight:600;">
            97.94% ROC-AUC
        </span>
        <span style="background:rgba(139,92,246,0.25);border:1px solid rgba(139,92,246,0.4);
                     border-radius:999px;padding:4px 14px;font-size:0.78rem;font-weight:600;">
            32,593 Students
        </span>
        <span style="background:rgba(217,119,6,0.25);border:1px solid rgba(217,119,6,0.4);
                     border-radius:999px;padding:4px 14px;font-size:0.78rem;font-weight:600;">
            10.6M LMS Interactions
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

    # How it works
    st.subheader("⚙️ How PRISM Works")
    steps = [
        ("📂", "1. Data Collection",     "OULAD datasets merged into a master dataset with 32,593 student records and 10.6M VLE interactions."),
        ("⚗️", "2. Feature Engineering", "Behavioural, academic, and registration features engineered from raw CSV files."),
        ("⚖️", "3. SMOTE Balancing",    "Synthetic Minority Oversampling addresses class imbalance before model training."),
        ("🤖", "4. ML Prediction",       "XGBoost classifier trained with stratified K-fold cross-validation."),
        ("💡", "5. SHAP Explanation",    "Global & local SHAP values explain each prediction in plain terms for educators."),
    ]
    cols = st.columns(len(steps))
    text_col = "#E2E8F0" if dark else "#1E293B"
    muted    = "#94A3B8" if dark else "#64748B"
    bg_step  = "#1A2234" if dark else "#FFFFFF"
    border_s = "#1E2D4A" if dark else "#E2E8F0"

    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
<div style="background:{bg_step};border:1px solid {border_s};border-radius:var(--radius-md);
            padding:1rem;text-align:center;height:160px;box-shadow:var(--shadow);">
    <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
    <div style="font-weight:700;color:{text_col};font-size:0.85rem;margin-bottom:0.4rem;">{title}</div>
    <div style="font-size:0.77rem;color:{muted};line-height:1.5;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    theme.section_div()

    tab1, tab2, tab3, tab4 = st.tabs(["🛠️ Technical Specs", "📚 Dataset", "📐 Methodology", "🏫 Project Info"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Machine Learning**

| Component | Detail |
|---|---|
| Primary Model | XGBoost (Extreme Gradient Boosting) |
| Baseline | Logistic Regression |
| Balancing | SMOTE |
| Validation | Stratified 5-Fold CV |
| Explainability | SHAP (TreeExplainer) |
| High Risk Threshold | ≥ 65% probability |
| Medium Risk Threshold | 40–65% probability |
""")
        with col_b:
            st.markdown("""
**Technology Stack**

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| ML Library | XGBoost, scikit-learn |
| Data | pandas, numpy |
| Visualisation | Plotly, Matplotlib |
| Explainability | SHAP |
| Database | Supabase (PostgreSQL) |
| Language | Python 3.11 |
""")

    with tab2:
        st.markdown("""
**Open University Learning Analytics Dataset (OULAD)**

The OULAD dataset was released by The Open University (UK) for learning analytics research.

| File | Description | Rows |
|---|---|---|
| studentInfo | Demographics & final results | 32,593 |
| studentAssessment | Assessment submissions | ~173,000 |
| studentVle | VLE click interactions | ~10.6M |
| studentRegistration | Registration dates | ~32,593 |
| assessments | Assessment metadata | ~206 |
| courses | Module metadata | ~22 |
| vle | VLE activity types | ~6,364 |

**Target Variable:** Binary classification — *At Risk* (Withdrawn/Fail) vs *Not At Risk* (Pass/Distinction).
""")

    with tab3:
        st.markdown("""
**Pipeline Architecture**

1. **Data Merging** — Seven OULAD CSVs joined on `id_student`, `code_module`, `code_presentation`
2. **Feature Engineering** — 24 features across 5 categories: Academic, Behavioural, Demographic, Registration, Course
3. **Preprocessing** — Label encoding for categoricals, StandardScaler for numerics, median imputation
4. **Class Balancing** — SMOTE applied on training split only (test leakage prevention)
5. **Model Training** — XGBoost with tunable hyperparameters; Logistic Regression as baseline
6. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix
7. **Explainability** — SHAP TreeExplainer for global and local feature attribution
8. **Risk Classification** — 3-tier: High (≥65%), Medium (40–65%), Low (<40%)

**Key Design Decisions**
- SMOTE on training set only ensures no data leakage
- 80/20 stratified train-test split preserves class ratios
- SHAP over feature importance scores for causal interpretability
""")

    with tab4:
        st.markdown(f"""
<div style="background:{'#1A2234' if dark else '#F8FAFC'};border:1px solid {'#1E2D4A' if dark else '#E2E8F0'};
            border-radius:var(--radius-md);padding:1.3rem;font-size:0.9rem;line-height:1.8;">
    <strong>Project Information</strong><br><br>
    <b>System:</b> PRISM – Predictive Risk Identification for Student Monitoring<br>
    <b>Type:</b> Master of Applied Computing (MAC) Capstone Project<br>
    <b>Institution:</b> Taylor's University, Malaysia<br>
    <b>Module Leader:</b> Prof. Dr. Noor Zaman Jhanjhi<br>
    <b>Supervisor:</b> Assoc. Prof. Dr. Shakeel Ahmed<br>
    <b>Dataset:</b> OULAD (Open University Learning Analytics Dataset)<br>
    <b>Key Metrics:</b> 93.05% Accuracy · 97.94% ROC-AUC<br>
    <b>Built with:</b> Python 3.11 · Streamlit · XGBoost · SHAP · Supabase
</div>
""", unsafe_allow_html=True)
