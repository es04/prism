import os, io
from typing import Optional
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.preprocessing import (
    load_raw_data, build_master_dataset, preprocess, get_display_df
)
from utils.model_utils import (
    apply_smote, train_xgboost, train_logistic_regression,
    evaluate_model, compute_shap_values,
    save_artifacts, load_artifacts,
    assign_risk_label,
)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _load_uploaded_or_default(upload_dict: dict) -> dict:
    """
    If user has uploaded files via the Upload page, use those.
    Otherwise fall back to data/ directory CSVs — downloading from
    Google Drive first if any are missing.
    """
    expected = ["studentInfo","studentAssessment","studentVle",
                "studentRegistration","assessments","courses","vle"]

    has_uploads = all(
        k in upload_dict and upload_dict[k] is not None for k in expected
    )
    if not has_uploads:
        from utils.gdrive_loader import ensure_datasets_available
        ensure_datasets_available(os.path.abspath(DATA_DIR))

    dfs = {}
    for key in expected:
        if key in upload_dict and upload_dict[key] is not None:
            dfs[key] = upload_dict[key].copy()
        else:
            path = os.path.join(DATA_DIR, f"{key}.csv")
            try:
                dfs[key] = pd.read_csv(path)
            except FileNotFoundError:
                dfs[key] = pd.DataFrame()
    return dfs


def get_or_init_state(force_retrain: bool = False,
                      xgb_params: Optional[dict] = None,
                      test_size: float = 0.2):
    """
    Idempotent state init. Pass force_retrain=True to re-run pipeline
    (e.g. after new uploads or hyperparameter changes).
    """
    if st.session_state.get("prism_ready") and not force_retrain:
        return True

    upload_dict = st.session_state.get("uploaded_dfs", {})

    # 1. Load data
    with st.spinner("📂 Loading datasets …"):
        dfs = _load_uploaded_or_default(upload_dict)
        missing_files = [k for k, v in dfs.items() if v.empty]
        if "studentInfo" in missing_files:
            st.error("❌  studentInfo.csv not found. Use Upload page or add to data/.")
            return False
        if missing_files:
            st.warning(f"⚠️ The following files are empty/missing: {missing_files}. "
                       "Results may be incomplete.")
        st.session_state["raw_dfs"] = dfs

    # 2. Feature engineering
    with st.spinner("⚗️ Engineering features …"):
        master = build_master_dataset(dfs)
        st.session_state["master_df"]  = master
        st.session_state["display_df"] = get_display_df(master)

    # 3. Preprocess (+ load cached encoders/scaler, or fit fresh ones)
    arts = None if force_retrain else load_artifacts(MODEL_DIR)
    use_cached = arts is not None and xgb_params is None

    if use_cached:
        assert arts is not None 
        encoders      = arts["encoders"]
        scaler        = arts["scaler"]
        feature_names = arts["features"]
        xgb_model     = arts["xgb"]
        lr_model      = arts["lr"]
        X_all, y_all, _, _, _ = preprocess(master, encoders, scaler, fit=False)
    else:
        with st.spinner("🔧 Preprocessing data …"):
            X_all, y_all, encoders, scaler, feature_names = preprocess(master, fit=True)

    # Always split deterministically so train/test stats are available
    # whether or not we retrained this run.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=test_size, random_state=42, stratify=np.asarray(y_all)
    )

    if use_cached:
        train_size = len(X_tr)
        smote_after = {
            "Not At Risk": int((y_tr == 0).sum()),
            "At Risk":     int((y_tr == 1).sum()),
        }
    else:
        with st.spinner("🤖 Training models (XGBoost + Logistic Regression) …"):
            X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)
            xgb_model = train_xgboost(X_tr_sm, y_tr_sm, custom_params=xgb_params)
            lr_model  = train_logistic_regression(X_tr_sm, y_tr_sm)
            save_artifacts(xgb_model, lr_model, encoders, scaler, feature_names, MODEL_DIR)
            train_size = len(X_tr_sm)
            smote_after = {
                "Not At Risk": int((y_tr_sm == 0).sum()),
                "At Risk":     int((y_tr_sm == 1).sum()),
            }

    # These are now always set, regardless of which branch ran above.
    st.session_state["smote_before"] = {
        "Not At Risk": int((y_tr == 0).sum()),
        "At Risk":     int((y_tr == 1).sum()),
    }
    st.session_state["smote_after"]  = smote_after
    st.session_state["train_size"]   = train_size
    st.session_state["test_size_n"]  = len(X_te)
    st.session_state["X_te"] = X_te
    st.session_state["y_te"] = y_te

    # 4. Full-dataset predictions
    with st.spinner("🔮 Generating predictions …"):
        xgb_proba = xgb_model.predict_proba(X_all)[:, 1]
        lr_proba  = lr_model.predict_proba(X_all)[:, 1]
        risk_labels = [assign_risk_label(p) for p in xgb_proba]

        pred_df = master[["id_student","code_module","code_presentation",
                          "gender","age_band","final_result"]].copy()
        pred_df["xgb_prob"]   = np.round(xgb_proba * 100, 1)
        pred_df["lr_prob"]    = np.round(lr_proba  * 100, 1)
        pred_df["risk_label"] = risk_labels
        pred_df["is_at_risk"] = master["is_at_risk"].values
        for col in ["avg_score","total_clicks","num_active_days",
                    "studied_credits","num_of_prev_attempts","num_late_submissions",
                    "num_active_days","withdrew_early"]:
            if col in master.columns:
                pred_df[col] = master[col].values
        st.session_state["pred_df"] = pred_df

    # 5. Evaluation
    with st.spinner("📊 Evaluating models …"):
        xgb_metrics = evaluate_model(xgb_model, X_te, y_te, "XGBoost")
        lr_metrics  = evaluate_model(lr_model,  X_te, y_te, "Logistic Regression")
        st.session_state["xgb_metrics"] = xgb_metrics
        st.session_state["lr_metrics"]  = lr_metrics

    # 6. SHAP
    with st.spinner("💡 Computing SHAP explanations …"):
        shap_vals, X_sample, explainer = compute_shap_values(
            xgb_model, X_all, feature_names, max_samples=600
        )
        st.session_state["shap_values"]    = shap_vals
        st.session_state["X_sample"]       = X_sample
        st.session_state["shap_explainer"] = explainer

    # 7. Store artefacts
    st.session_state["xgb_model"]     = xgb_model
    st.session_state["lr_model"]      = lr_model
    st.session_state["encoders"]      = encoders
    st.session_state["scaler"]        = scaler
    st.session_state["feature_names"] = feature_names
    st.session_state["X_all"]         = X_all
    st.session_state["y_all"]         = y_all
    st.session_state["prism_ready"]   = True

    # 8. Sync students + predictions to Supabase
    from utils.supabase_client import is_connected
    if is_connected():
        with st.spinner("☁️ Syncing results to Supabase …"):
            import uuid
            from utils.db_sync import save_run_to_supabase
            run_id = str(uuid.uuid4())
            ok, msg = save_run_to_supabase(st.session_state["display_df"], pred_df, run_id)
            st.session_state["last_sync_run_id"] = run_id if ok else None
            st.session_state["last_sync_message"] = msg
            st.session_state["last_sync_ok"] = ok
            if ok:
                st.toast(f"☁️ {msg}", icon="✅")
            else:
                st.toast(f"⚠️ Supabase sync skipped: {msg}", icon="⚠️")

    return True