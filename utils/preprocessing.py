"""
PRISM – Predictive Risk Identification for Student Monitoring
Data Preprocessing & Feature Engineering Pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ─── Risk label mapping ──────────────────────────────────────────────────────
#   final_result → binary at-risk flag
#   Withdrawn / Fail  = 1 (At Risk)
#   Pass / Distinction = 0 (Not At Risk)
RISK_MAP = {"Withdrawn": 1, "Fail": 1, "Pass": 0, "Distinction": 0}


def load_raw_data(data_dir: str = None) -> dict:
    """Load all seven OULAD CSV files, downloading from Google Drive if needed."""
    if data_dir is None:
        data_dir = os.path.abspath(DATA_DIR)

    # ── Auto-download from Google Drive if any file is missing ───────────────
    from utils.gdrive_loader import ensure_datasets_available
    ensure_datasets_available(data_dir)

    files = {
        "studentInfo":        "studentInfo.csv",
        "studentAssessment":  "studentAssessment.csv",
        "studentVle":         "studentVle.csv",
        "studentRegistration":"studentRegistration.csv",
        "assessments":        "assessments.csv",
        "courses":            "courses.csv",
        "vle":                "vle.csv",
    }
    dfs = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        try:
            dfs[key] = pd.read_csv(path)
        except FileNotFoundError:
            dfs[key] = pd.DataFrame()
    return dfs


def engineer_assessment_features(dfs: dict) -> pd.DataFrame:
    """Aggregate assessment-level features per student × module × presentation."""
    sa = dfs["studentAssessment"].copy()
    a  = dfs["assessments"].copy()
    if sa.empty or a.empty:
        return pd.DataFrame()

    merged = sa.merge(a[["id_assessment", "assessment_type", "weight", "date"]],
                      on="id_assessment", how="left")

    merged["score"] = pd.to_numeric(merged["score"], errors="coerce")
    merged["date_submitted"] = pd.to_numeric(merged["date_submitted"], errors="coerce")
    merged["date"] = pd.to_numeric(merged["date"], errors="coerce")
    merged["submission_timeliness"] = merged["date"] - merged["date_submitted"]
    merged["late_submission"] = (merged["submission_timeliness"] < 0).astype(int)

    agg = merged.groupby("id_student").agg(
        avg_score              = ("score", "mean"),
        min_score              = ("score", "min"),
        max_score              = ("score", "max"),
        std_score              = ("score", "std"),
        num_assessments        = ("score", "count"),
        num_late_submissions   = ("late_submission", "sum"),
        avg_submission_timing  = ("submission_timeliness", "mean"),
        num_banked             = ("is_banked", "sum"),
    ).reset_index()

    agg["std_score"] = agg["std_score"].fillna(0)
    return agg


def engineer_vle_features(dfs: dict) -> pd.DataFrame:
    """Aggregate VLE interaction features per student."""
    sv  = dfs["studentVle"].copy()
    vle = dfs["vle"].copy()
    if sv.empty:
        return pd.DataFrame()

    if not vle.empty:
        sv = sv.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    else:
        sv["activity_type"] = "unknown"

    sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0)

    base = sv.groupby("id_student").agg(
        total_clicks           = ("sum_click", "sum"),
        avg_clicks_per_day     = ("sum_click", "mean"),
        num_active_days        = ("date", "nunique"),
        num_resources_accessed = ("id_site", "nunique"),
    ).reset_index()

    if "activity_type" in sv.columns:
        pivot = sv.pivot_table(
            index="id_student", columns="activity_type",
            values="sum_click", aggfunc="sum", fill_value=0
        )
        pivot.columns = [f"vle_{c}" for c in pivot.columns]
        pivot.reset_index(inplace=True)
        base = base.merge(pivot, on="id_student", how="left")

    return base


def engineer_registration_features(dfs: dict) -> pd.DataFrame:
    """Derive registration-timing features per student."""
    sr = dfs["studentRegistration"].copy()
    if sr.empty:
        return pd.DataFrame()

    sr["date_registration"]   = pd.to_numeric(sr["date_registration"],   errors="coerce")
    sr["date_unregistration"] = pd.to_numeric(sr["date_unregistration"], errors="coerce")

    sr["registered_early"]    = (sr["date_registration"] < 0).astype(int)
    sr["withdrew_early"]      = sr["date_unregistration"].notna().astype(int)
    sr["days_enrolled"]       = sr["date_unregistration"].fillna(0) - sr["date_registration"]

    agg = sr.groupby("id_student").agg(
        registered_early = ("registered_early", "max"),
        withdrew_early   = ("withdrew_early",   "max"),
        days_enrolled    = ("days_enrolled",    "mean"),
        num_registrations= ("code_module",      "count"),
    ).reset_index()
    return agg


def build_master_dataset(dfs: dict) -> pd.DataFrame:
    """Merge all feature groups with studentInfo as the base."""
    info = dfs["studentInfo"].copy()
    if info.empty:
        raise ValueError("studentInfo.csv is empty or missing.")

    # Target
    info["is_at_risk"] = info["final_result"].map(RISK_MAP).fillna(0).astype(int)

    # Assessment features
    af = engineer_assessment_features(dfs)
    if not af.empty:
        info = info.merge(af, on="id_student", how="left")

    # VLE features
    vf = engineer_vle_features(dfs)
    if not vf.empty:
        info = info.merge(vf, on="id_student", how="left")

    # Registration features
    rf = engineer_registration_features(dfs)
    if not rf.empty:
        info = info.merge(rf, on="id_student", how="left")

    # Course length
    c = dfs["courses"].copy()
    if not c.empty:
        info = info.merge(c, on=["code_module", "code_presentation"], how="left")

    return info


CATEGORICAL_COLS = ["gender", "region", "highest_education",
                    "imd_band", "age_band", "disability", "code_module",
                    "code_presentation"]

FEATURE_COLS = [
    # Demographic
    "gender", "age_band", "highest_education", "imd_band",
    "disability", "num_of_prev_attempts", "studied_credits",
    # Academic / assessment
    "avg_score", "min_score", "max_score", "std_score",
    "num_assessments", "num_late_submissions", "avg_submission_timing",
    "num_banked",
    # VLE / behavioural
    "total_clicks", "avg_clicks_per_day", "num_active_days",
    "num_resources_accessed",
    # Registration
    "registered_early", "withdrew_early", "days_enrolled",
    "num_registrations",
    # Course
    "module_presentation_length",
]


def preprocess(df: pd.DataFrame, encoders: dict = None,
               scaler: StandardScaler = None, fit: bool = True):
    """
    Encode categoricals, impute, scale numeric columns.
    Returns (X, y, encoders, scaler, feature_names)
    """
    df = df.copy()

    # Keep only columns that exist
    cat_cols  = [c for c in CATEGORICAL_COLS if c in df.columns]
    feat_cols = [c for c in FEATURE_COLS      if c in df.columns]

    # Encode categoricals
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in feat_cols:
            continue
        df[col] = df[col].astype(str).fillna("Unknown")
        if fit:
            le = LabelEncoder()
            # Add "Unknown" to known classes so transform never fails
            le.fit(list(df[col].unique()) + ["Unknown"])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col] = 0
                continue
            # Handle unseen labels
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "Unknown")
        df[col] = le.transform(df[col])

    # Subset to existing feature cols
    X_df = df[feat_cols].copy()

    # Impute numeric
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    imp = SimpleImputer(strategy="median")
    if fit:
        X_df[num_cols] = imp.fit_transform(X_df[num_cols])
    else:
        X_df[num_cols] = imp.fit_transform(X_df[num_cols])  # always fit for simplicity

    # Scale
    if fit:
        scaler = StandardScaler()
        X_df[num_cols] = scaler.fit_transform(X_df[num_cols])
    else:
        if scaler is not None:
            # Only scale columns that were fitted
            cols_to_scale = [c for c in num_cols if c in X_df.columns]
            try:
                X_df[cols_to_scale] = scaler.transform(X_df[cols_to_scale])
            except Exception:
                pass

    y = df["is_at_risk"].values if "is_at_risk" in df.columns else None
    return X_df.values, y, encoders, scaler, feat_cols


def get_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a human-readable table of students for the dashboard."""
    cols = ["id_student", "code_module", "gender", "age_band", "highest_education",
            "imd_band", "num_of_prev_attempts", "studied_credits", "disability",
            "final_result", "is_at_risk",
            "avg_score", "total_clicks", "num_active_days",
            "registered_early", "withdrew_early"]
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()
