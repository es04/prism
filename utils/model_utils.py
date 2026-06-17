import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import cast

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix,
                                     classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "models"

DEFAULT_XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "use_label_encoder": False,
    "eval_metric":       "logloss",
    "random_state":      42,
    "n_jobs":            -1,
}

def apply_smote(X, y, random_state=42) -> tuple[np.ndarray, np.ndarray]:
    sm = SMOTE(random_state=random_state)
    result = cast(tuple[np.ndarray, np.ndarray], sm.fit_resample(X, y))
    X_res, y_res = result
    return X_res, y_res


def train_xgboost(X_train, y_train, custom_params=None):
    params = DEFAULT_XGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    return {
        "model_name":   model_name,
        "accuracy":     round(accuracy_score(y_test, y_pred), 4),
        "precision":    round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":       round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score":     round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":      round(roc_auc_score(y_test, y_proba), 4),
        "conf_matrix":  confusion_matrix(y_test, y_pred),
        "y_pred":       y_pred,
        "y_proba":      y_proba,
        "class_report": classification_report(
            y_test, y_pred, target_names=["Not At Risk", "At Risk"]
        ),
    }


def cross_validate_model(model, X, y, cv=5):
    skf     = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy","precision","recall","f1","roc_auc"]
    res     = cross_validate(model, X, y, cv=skf, scoring=scoring)
    return {m: {"mean": round(float(res[f"test_{m}"].mean()), 4),
                "std":  round(float(res[f"test_{m}"].std()), 4),
                "folds": res[f"test_{m}"].tolist()}
            for m in scoring}


def compute_shap_values(model, X, feature_names, max_samples=500):
    X_s = X[np.random.choice(len(X), min(max_samples, len(X)), replace=False)]
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X_s)
    if isinstance(sv, list):
        sv = sv[1]
    return sv, X_s, exp


def shap_summary_figure(shap_values, X_sample, feature_names):
    fig, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def shap_bar_figure(shap_values, feature_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx      = np.argsort(mean_abs)[::-1][:top_n]
    feats    = [feature_names[i] for i in idx]
    vals     = mean_abs[idx]
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.barh(feats[::-1], vals[::-1], color="#2563eb", edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def shap_local_figure(explainer, X_row, feature_names, student_id=None):
    sv = explainer.shap_values(X_row.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]
    order  = np.argsort(np.abs(sv))[::-1][:12]
    feats  = [feature_names[i] for i in order]
    vals   = sv[order]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
    title = f"Local SHAP Explanation — Student {student_id}" if student_id else "Local SHAP Explanation"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    return fig


def save_artifacts(xgb, lr, encoders, scaler, feature_names, model_dir=MODEL_DIR):
    os.makedirs(model_dir, exist_ok=True)
    for name, obj in [("xgb_model",xgb),("lr_model",lr),("encoders",encoders),
                       ("scaler",scaler),("feature_names",feature_names)]:
        with open(f"{model_dir}/{name}.pkl","wb") as f:
            pickle.dump(obj, f)


def load_artifacts(model_dir=MODEL_DIR):
    required = ["xgb_model","lr_model","encoders","scaler","feature_names"]
    if not all(os.path.exists(f"{model_dir}/{n}.pkl") for n in required):
        return None
    arts = {}
    for name, key in [("xgb_model","xgb"),("lr_model","lr"),("encoders","encoders"),
                       ("scaler","scaler"),("feature_names","features")]:
        with open(f"{model_dir}/{name}.pkl","rb") as f:
            arts[key] = pickle.load(f)
    return arts


def assign_risk_label(prob):
    if prob >= 0.65: return "High Risk"
    if prob >= 0.40: return "Medium Risk"
    return "Low Risk"


def risk_color(label):
    return {"High Risk":"#ef4444","Medium Risk":"#f59e0b","Low Risk":"#22c55e"}.get(label,"#6b7280")
