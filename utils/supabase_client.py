import os
import streamlit as st
from supabase import create_client, Client


def _read_credentials():
    """Look for Supabase credentials in st.secrets, then env vars."""
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        if url and key:
            return url, key
    except Exception:
        pass

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    return url, key


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    """
    Returns a cached Supabase client, or None if credentials are missing
    or invalid. Never raises — callers should treat None as "not configured"
    and degrade gracefully (PRISM must keep working fully offline).
    """
    url, key = _read_credentials()
    if not url or not key:
        return None

    try:
        client: Client = create_client(url, key)
        return client
    except Exception as e:
        print(f"[PRISM] Failed to create Supabase client: {e}")
        return None


def is_connected() -> bool:
    """Quick boolean check used by pages to decide whether to show cloud UI."""
    return get_supabase_client() is not None