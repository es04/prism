"""
PRISM – Supabase Data Sync
Read/write helpers that push uploaded CSVs, the engineered student table,
and model predictions to Supabase, and pull them back for history views.

Every public function here is defensive: if Supabase isn't configured, or
a call fails for any reason (network, bad credentials, RLS, etc.), the
function returns a (False, message) / empty result instead of raising —
PRISM's local pipeline must keep working even when the cloud sync doesn't.
"""

import time
import pandas as pd
import numpy as np

from utils.supabase_client import get_supabase_client

UPLOAD_BUCKET = "prism-uploads"
CHUNK_SIZE = 1000

STUDENT_COLUMNS = [
    "id_student", "code_module", "gender", "age_band", "highest_education",
    "imd_band", "num_of_prev_attempts", "studied_credits", "disability",
    "final_result", "is_at_risk", "avg_score", "total_clicks",
    "num_active_days", "registered_early", "withdrew_early",
]

PREDICTION_COLUMNS = [
    "id_student", "code_module", "code_presentation", "gender", "age_band",
    "final_result", "xgb_prob", "lr_prob", "risk_label", "is_at_risk",
    "avg_score", "total_clicks", "num_active_days", "studied_credits",
    "num_of_prev_attempts", "num_late_submissions", "withdrew_early",
]


# ── Internal helpers ───────────────────────────────────────────────────────
def _sanitize_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame into JSON-safe records: no numpy scalar types,
    no NaN/NaT (Postgres/JSON wants `null`, not float('nan'))."""
    clean = df.astype(object).where(pd.notnull(df), None)
    records = clean.to_dict(orient="records")

    safe = []
    for rec in records:
        safe_rec = {}
        for k, v in rec.items():
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            safe_rec[k] = v
        safe.append(safe_rec)
    return safe


def _chunked(records: list, size: int = CHUNK_SIZE):
    for i in range(0, len(records), size):
        yield records[i:i + size]


def _prepare(df: pd.DataFrame, columns: list, run_id: str) -> list:
    keep = [c for c in columns if c in df.columns]
    sub = df[keep].copy()
    sub["run_id"] = run_id
    return _sanitize_records(sub)


# ── Uploaded CSVs ────────────────────────────────────────────────────────
def log_upload(dataset_name: str, file_bytes: bytes, file_name: str,
               row_count: int, col_count: int, schema_valid: bool):
    """
    Upload a raw CSV to Supabase Storage and record its metadata in the
    `dataset_uploads` table. Returns (success: bool, message: str).
    """
    client = get_supabase_client()
    if client is None:
        return False, "Supabase is not configured."

    storage_path = f"{dataset_name}/{int(time.time())}_{file_name}"

    try:
        client.storage.from_(UPLOAD_BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
    except Exception as e:
        return False, f"Storage upload failed: {e}"

    try:
        client.table("dataset_uploads").insert({
            "dataset_name": dataset_name,
            "file_name":    file_name,
            "storage_path": storage_path,
            "row_count":    int(row_count),
            "col_count":    int(col_count),
            "schema_valid": bool(schema_valid),
        }).execute()
    except Exception as e:
        return False, f"File uploaded, but failed to log metadata: {e}"

    return True, "Saved to Supabase."


def fetch_upload_history(limit: int = 50) -> pd.DataFrame:
    """Return the most recent dataset uploads logged in Supabase."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("dataset_uploads")
            .select("dataset_name, file_name, row_count, col_count, schema_valid, uploaded_at")
            .order("uploaded_at", desc=True)
            .limit(limit)
            .execute()
        )
        return pd.DataFrame(resp.data)
    except Exception:
        return pd.DataFrame()


def download_uploaded_csv(storage_path: str):
    """Fetch raw CSV bytes for a previously uploaded file from Storage."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        return client.storage.from_(UPLOAD_BUCKET).download(storage_path)
    except Exception:
        return None


# ── Students & predictions ──────────────────────────────────────────────
def save_run_to_supabase(display_df: pd.DataFrame, pred_df: pd.DataFrame, run_id: str):
    """
    Push the current pipeline run's student table and predictions table to
    Supabase, tagged with `run_id`. Returns (success: bool, message: str).
    """
    client = get_supabase_client()
    if client is None:
        return False, "Supabase is not configured."

    n_students, n_preds = 0, 0

    try:
        student_records = _prepare(display_df, STUDENT_COLUMNS, run_id)
        for chunk in _chunked(student_records):
            client.table("students").insert(chunk).execute()
        n_students = len(student_records)
    except Exception as e:
        return False, f"Failed to sync students table: {e}"

    try:
        pred_records = _prepare(pred_df, PREDICTION_COLUMNS, run_id)
        for chunk in _chunked(pred_records):
            client.table("predictions").insert(chunk).execute()
        n_preds = len(pred_records)
    except Exception as e:
        return False, f"Synced {n_students} students, but predictions failed: {e}"

    return True, f"Synced {n_students:,} students and {n_preds:,} predictions to Supabase."


def fetch_latest_run_predictions(limit: int = 5000) -> pd.DataFrame:
    """Fetch predictions belonging to the most recently synced run."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    try:
        latest = (
            client.table("predictions")
            .select("run_id")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if not latest.data:
            return pd.DataFrame()
        run_id = latest.data[0]["run_id"]

        resp = (
            client.table("predictions")
            .select("*")
            .eq("run_id", run_id)
            .limit(limit)
            .execute()
        )
        return pd.DataFrame(resp.data)
    except Exception:
        return pd.DataFrame()


def list_synced_runs(limit: int = 20) -> pd.DataFrame:
    """Return a summary of distinct pipeline runs stored in Supabase."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    try:
        resp = (
            client.table("predictions")
            .select("run_id, created_at")
            .order("created_at", desc=True)
            .limit(limit * 50)  # over-fetch then dedupe client-side
            .execute()
        )
        df = pd.DataFrame(resp.data)
        if df.empty:
            return df
        summary = (
            df.groupby("run_id")
            .agg(students=("run_id", "count"), synced_at=("created_at", "max"))
            .reset_index()
            .sort_values("synced_at", ascending=False)
            .head(limit)
        )
        return summary
    except Exception:
        return pd.DataFrame()


def clear_all_synced_data():
    """Delete every row from `students` and `predictions` (housekeeping)."""
    client = get_supabase_client()
    if client is None:
        return False, "Supabase is not configured."
    try:
        client.table("students").delete().gte("id", 0).execute()
        client.table("predictions").delete().gte("id", 0).execute()
        return True, "Cleared all synced students and predictions from Supabase."
    except Exception as e:
        return False, f"Failed to clear data: {e}"