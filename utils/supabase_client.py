import os
import requests
import streamlit as st
from supabase import create_client, Client

# ── Schema DDL (executed once via the Management API) ─────────────────────────
_SCHEMA_SQL = """
-- pgcrypto for gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- dataset_uploads: log of every CSV uploaded through the Upload page
CREATE TABLE IF NOT EXISTS dataset_uploads (
    id             BIGSERIAL PRIMARY KEY,
    dataset_name   TEXT        NOT NULL,
    file_name      TEXT        NOT NULL,
    storage_path   TEXT        NOT NULL,
    row_count      INTEGER,
    col_count      INTEGER,
    schema_valid   BOOLEAN     DEFAULT FALSE,
    uploaded_at    TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_uploads_uploaded_at ON dataset_uploads (uploaded_at DESC);

-- students: engineered feature snapshot per run
CREATE TABLE IF NOT EXISTS students (
    id                    BIGSERIAL PRIMARY KEY,
    run_id                UUID        NOT NULL,
    id_student            BIGINT,
    code_module           TEXT,
    gender                TEXT,
    age_band              TEXT,
    highest_education     TEXT,
    imd_band              TEXT,
    num_of_prev_attempts  INTEGER,
    studied_credits       INTEGER,
    disability            TEXT,
    final_result          TEXT,
    is_at_risk            BOOLEAN,
    avg_score             NUMERIC(7,3),
    total_clicks          BIGINT,
    num_active_days       INTEGER,
    registered_early      BOOLEAN,
    withdrew_early        BOOLEAN,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_students_run_id     ON students (run_id);
CREATE INDEX IF NOT EXISTS idx_students_id_student ON students (id_student);
CREATE INDEX IF NOT EXISTS idx_students_created_at ON students (created_at DESC);

-- predictions: risk scores and labels per student per run
CREATE TABLE IF NOT EXISTS predictions (
    id                    BIGSERIAL PRIMARY KEY,
    run_id                UUID        NOT NULL,
    id_student            BIGINT,
    code_module           TEXT,
    code_presentation     TEXT,
    gender                TEXT,
    age_band              TEXT,
    final_result          TEXT,
    xgb_prob              NUMERIC(6,2),
    lr_prob               NUMERIC(6,2),
    risk_label            TEXT,
    is_at_risk            BOOLEAN,
    avg_score             NUMERIC(7,3),
    total_clicks          BIGINT,
    num_active_days       INTEGER,
    studied_credits       INTEGER,
    num_of_prev_attempts  INTEGER,
    num_late_submissions  INTEGER,
    withdrew_early        BOOLEAN,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_predictions_run_id    ON predictions (run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created   ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk      ON predictions (risk_label);

-- RLS: enable but allow all (service-role key bypasses RLS anyway)
ALTER TABLE dataset_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE students        ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions     ENABLE ROW LEVEL SECURITY;

DO $do$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='dataset_uploads' AND policyname='prism_allow_all') THEN
    CREATE POLICY prism_allow_all ON dataset_uploads FOR ALL USING (TRUE) WITH CHECK (TRUE);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='students' AND policyname='prism_allow_all') THEN
    CREATE POLICY prism_allow_all ON students FOR ALL USING (TRUE) WITH CHECK (TRUE);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='predictions' AND policyname='prism_allow_all') THEN
    CREATE POLICY prism_allow_all ON predictions FOR ALL USING (TRUE) WITH CHECK (TRUE);
  END IF;
END
$do$;
"""

_BUCKET_NAME = "prism-uploads"


# Credential helpers
def _read_credentials():
    """Read Supabase credentials from st.secrets then env vars."""
    url = key = pat = None
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        pat = st.secrets["supabase"].get("pat")
    except Exception:
        pass

    url = url or os.environ.get("SUPABASE_URL")
    key = key or os.environ.get("SUPABASE_KEY")
    pat = pat or os.environ.get("SUPABASE_PAT")
    return url, key, pat


def _project_ref(url: str) -> str:
    """Extract the project reference ID from a Supabase project URL."""
    # https://<ref>.supabase.co  →  <ref>
    return url.rstrip("/").split("https://")[1].split(".supabase.co")[0]


# Supabase client (cached)
@st.cache_resource(show_spinner=False)
def get_supabase_client() -> "Client | None":
    """
    Returns a cached Supabase client, or None if credentials are missing.
    Never raises — callers treat None as 'not configured'.
    """
    url, key, _ = _read_credentials()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"[PRISM] Supabase client error: {e}")
        return None


def is_connected() -> bool:
    return get_supabase_client() is not None


# Auto-provisioner
def _run_sql_via_management_api(url: str, pat: str, sql: str) -> tuple[bool, str]:
    """
    Execute raw SQL against the project's database using the
    Supabase Management API (requires a Personal Access Token).
    Returns (success, message).
    """
    ref = _project_ref(url)
    endpoint = f"https://api.supabase.com/v1/projects/{ref}/database/query"
    try:
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {pat}",
                "Content-Type": "application/json",
            },
            json={"query": sql},
            timeout=30,
        )
        if resp.status_code in (200, 201):
            return True, "Schema created successfully."
        else:
            return (
                False,
                f"Management API returned {resp.status_code}: {resp.text[:300]}",
            )
    except requests.exceptions.Timeout:
        return False, "Management API request timed out."
    except Exception as e:
        return False, f"Management API error: {e}"


def _ensure_storage_bucket(client: "Client") -> tuple[bool, str]:
    """Create the prism-uploads storage bucket if it doesn't exist."""
    try:
        buckets = [b.name for b in client.storage.list_buckets()]
        if _BUCKET_NAME not in buckets:
            client.storage.create_bucket(_BUCKET_NAME, options={"public": False})
        return True, f"Bucket '{_BUCKET_NAME}' ready."
    except Exception as e:
        return False, f"Storage bucket setup failed: {e}"


def _tables_exist(client: "Client") -> bool:
    """
    Quick check: try to select 0 rows from `predictions`.
    Returns True if the table exists, False if PostgREST returns an error.
    """
    try:
        client.table("predictions").select("id").limit(0).execute()
        return True
    except Exception:
        return False


def ensure_schema() -> dict:
    """
    Auto-provision the Supabase schema on first run.

    Returns a status dict:
        { "ran": bool, "ok": bool, "message": str, "needs_pat": bool }

    - If tables already exist → skips silently (ran=False).
    - If PAT is configured   → creates tables + bucket automatically (ran=True).
    - If PAT is missing      → returns needs_pat=True so the UI can prompt.
    """
    # Already done this session?
    cached = st.session_state.get("_supabase_schema_status")
    if cached is not None:
        return cached

    url, key, pat = _read_credentials()

    # No credentials at all → Supabase not configured, that's fine
    if not url or not key:
        result = {
            "ran": False,
            "ok": False,
            "message": "Supabase not configured.",
            "needs_pat": False,
        }
        st.session_state["_supabase_schema_status"] = result
        return result

    client = get_supabase_client()
    if client is None:
        result = {
            "ran": False,
            "ok": False,
            "message": "Could not create Supabase client.",
            "needs_pat": False,
        }
        st.session_state["_supabase_schema_status"] = result
        return result

    # Tables already exist → nothing to do
    if _tables_exist(client):
        result = {
            "ran": False,
            "ok": True,
            "message": "Schema already in place.",
            "needs_pat": False,
        }
        st.session_state["_supabase_schema_status"] = result
        return result

    # Tables missing — need PAT to create them
    if not pat:
        result = {
            "ran": False,
            "ok": False,
            "needs_pat": True,
            "message": (
                "Supabase tables are not set up yet. "
                "Add SUPABASE_PAT to your .env file so PRISM can create them automatically. "
                "Get a token at: https://supabase.com/dashboard/account/tokens"
            ),
        }
        st.session_state["_supabase_schema_status"] = result
        return result

    # Run DDL via Management API
    ok, msg = _run_sql_via_management_api(url, pat, _SCHEMA_SQL)
    if ok:
        # Also create the storage bucket
        _ensure_storage_bucket(client)

    result = {"ran": True, "ok": ok, "message": msg, "needs_pat": False}
    st.session_state["_supabase_schema_status"] = result
    return result
