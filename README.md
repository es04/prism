# 🔮 PRISM — Predictive Risk Identification for Student Monitoring

An explainable AI system for early identification of academically at-risk students,
built on the OULAD dataset with XGBoost, SHAP, and Streamlit.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

> **First launch**: The app trains XGBoost and Logistic Regression models, applies SMOTE,
> computes SHAP values, and caches the results as `.pkl` files in `models/`.
> This takes ~1–3 minutes depending on your machine. Subsequent launches load from cache instantly.

---

## 📁 Project Structure

```
prism/
├── app.py                        # Main Streamlit entry point + sidebar navigation
├── requirements.txt
├── data/                         # OULAD CSV datasets (7 files)
│   ├── studentInfo.csv
│   ├── studentAssessment.csv
│   ├── studentVle.csv
│   ├── studentRegistration.csv
│   ├── assessments.csv
│   ├── courses.csv
│   └── vle.csv
├── models/                       # Auto-generated trained model artefacts
│   ├── xgb_model.pkl
│   ├── lr_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── utils/
│   ├── preprocessing.py          # Data loading, feature engineering, encoding
│   └── model_utils.py            # Model training, evaluation, SHAP
└── pages/
    ├── state_manager.py          # Shared session state (trains once, caches)
    ├── dashboard.py              # Page 1: Overview KPIs and charts
    ├── predictions.py            # Page 2: Student risk table with filters
    ├── prediction_summary.py     # Page 3: Confusion matrix, ROC, CV results
    ├── explainability.py         # Page 4: Global + local SHAP explanations
    ├── feature_importance.py     # Page 5: SHAP & XGBoost importance + correlations
    ├── dataset_overview.py       # Page 6: Raw data preview + distributions
    └── about.py                  # Page 7: System documentation
```

---

## 🛠️ System Architecture

```
OULAD CSVs → Feature Engineering → Preprocessing (encode + scale + SMOTE)
          ↓
    XGBoost Classifier ←→ Logistic Regression (baseline)
          ↓
    SHAP TreeExplainer (global + local)
          ↓
    Streamlit Dashboard (7 modules)
```

---

## 📊 Model Performance (held-out test set, 20%)

| Metric    | XGBoost | Logistic Regression |
|-----------|---------|---------------------|
| Accuracy  | ~91%    | ~87%                |
| Precision | ~91%    | ~87%                |
| Recall    | ~92%    | ~87%                |
| F1 Score  | ~91%    | ~87%                |
| ROC-AUC   | ~97%    | ~94%                |

---

## 🔑 Risk Classification Thresholds

| Risk Level  | Probability |
|-------------|-------------|
| 🔴 High Risk   | ≥ 65%       |
| 🟡 Medium Risk | 40% – 65%   |
| 🟢 Low Risk    | < 40%       |

---

## 📚 Dataset

**OULAD (Open University Learning Analytics Dataset)**
- 32,593 student records across 7 modules
- 173,912 assessment submissions
- 10.6M+ VLE interactions

Target variable: `final_result`
- At Risk (1) = Withdrawn or Fail
- Not At Risk (0) = Pass or Distinction

---

## 🎓 Project Info

| Field | Detail |
|-------|--------|
| Programme | Master in Applied Computing (MAC) |
| Institution | Taylor's University, Malaysia |
| Supervisor | Assoc. Prof. Dr. Shakeel Ahmed |
| Type | Capstone I Research Prototype |
