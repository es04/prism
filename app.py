from dotenv import load_dotenv

load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="PRISM | Student Risk Monitor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject theme CSS (reads dark_mode from session_state)
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import theme

# Initialise dark mode preference
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

theme.inject_theme_css()

# ── Auto-provision Supabase schema ────────────────────────────────────────────
import utils.supabase_client as sb

if "supabase_setup_done" not in st.session_state:
    try:
        _schema_status = sb.ensure_schema()
    except Exception as e:
        _schema_status = {"success": False, "message": str(e)}
    st.session_state["supabase_setup_done"] = True
    st.session_state["supabase_schema_status"] = _schema_status
else:
    _schema_status = st.session_state.get("supabase_schema_status", {})

_sb_ok = sb.is_connected

# ── Navigation ────────────────────────────────────────────────────────────────
NAV_SECTIONS = {
    "Overview": [
        ("🏠  Dashboard", "Dashboard Overview"),
    ],
    "Data Pipeline": [
        ("📤  Upload Data", "Upload & Configure Data"),
        ("📋  Dataset Overview", "Dataset Overview"),
        ("⚙️  Preprocessing", "Preprocessing Pipeline"),
    ],
    "Model": [
        ("🤖  Train Model", "Model Training & Tuning"),
        ("🔍  Predictions", "Student Risk Predictions"),
        ("📊  Risk Report", "Prediction Summary"),
    ],
    "Insights": [
        ("💡  Explainability", "Explainability Insights"),
        ("📈  Feature Importance", "Feature Importance"),
    ],
    "Info": [
        ("ℹ️  About PRISM", "About PRISM"),
    ],
}

all_nav = [
    (label, key)
    for section_items in NAV_SECTIONS.values()
    for label, key in section_items
]
nav_labels = [p[0] for p in all_nav]
nav_keys = [p[1] for p in all_nav]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown(
        """
<div style="text-align:center;padding:1.4rem 0 0.2rem;">
    <div style="display:inline-flex;align-items:center;justify-content:center;
                width:52px;height:52px;border-radius:14px;
                background:rgba(37,99,235,0.18);border:1px solid rgba(37,99,235,0.3);
                margin-bottom:0.6rem;">
        <svg width="28" height="28" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
            <line x1="2"  y1="17" x2="13" y2="17" stroke="#E2E8F0" stroke-width="1.7"/>
            <polygon points="13,8 13,26 24,17" fill="none" stroke="#E2E8F0" stroke-width="1.7"/>
            <line x1="24" y1="17" x2="32" y2="6"  stroke="#8B5CF6" stroke-width="1.7"/>
            <line x1="24" y1="17" x2="33" y2="12" stroke="#3B82F6" stroke-width="1.7"/>
            <line x1="24" y1="17" x2="33" y2="17" stroke="#14B8A6" stroke-width="1.7"/>
            <line x1="24" y1="17" x2="33" y2="22" stroke="#F59E0B" stroke-width="1.7"/>
            <line x1="24" y1="17" x2="32" y2="28" stroke="#EF4444" stroke-width="1.7"/>
        </svg>
    </div>
    <div style="font-size:1.45rem;font-weight:800;letter-spacing:5px;
                font-family:'Inter',sans-serif;color:#fff;">PRISM</div>
    <div style="font-size:0.61rem;color:#64748B;margin-top:2px;
                letter-spacing:0.06em;font-family:'JetBrains Mono',monospace;line-height:1.5;">
        Student Risk Monitor
    </div>
</div>
<div class="accent-bar"></div>
""",
        unsafe_allow_html=True,
    )

    # Dark mode toggle
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            '<div style="padding:0.35rem 0 0 0.3rem;font-size:0.75rem;color:#64748B;">🌙 Dark mode</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        dark = st.toggle(
            "",
            value=st.session_state["dark_mode"],
            key="dark_toggle",
            label_visibility="collapsed",
        )
        if dark != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = dark
            st.rerun()

    st.markdown("<div class='accent-bar'></div>", unsafe_allow_html=True)

    # Navigation (grouped by section headers)
    all_flat_labels = nav_labels  # flat list for radio
    page = st.radio("Navigation", nav_labels, label_visibility="collapsed")

    st.markdown("<div class='accent-bar'></div>", unsafe_allow_html=True)

    # Status pills
    if st.session_state.get("prism_ready"):
        st.markdown(
            '<div style="text-align:center;padding:0.15rem 0;"><span class="pill pill-green">✓ &nbsp;Model ready</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:center;padding:0.15rem 0;"><span class="pill pill-amber">◌ &nbsp;No data loaded</span></div>',
            unsafe_allow_html=True,
        )

    if _sb_ok():
        if _schema_status.get("ok"):
            if _schema_status.get("ran"):
                st.markdown(
                    '<div style="text-align:center;padding:0.1rem 0 0.3rem;"><span class="pill pill-green">☁ &nbsp;Cloud set up ✓</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="text-align:center;padding:0.1rem 0 0.3rem;"><span class="pill pill-blue">☁ &nbsp;Cloud sync active</span></div>',
                    unsafe_allow_html=True,
                )
        elif _schema_status.get("needs_pat"):
            st.markdown(
                '<div style="text-align:center;padding:0.1rem 0 0.3rem;"><span class="pill pill-amber">☁ &nbsp;Cloud: needs setup</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center;padding:0.1rem 0 0.3rem;"><span class="pill pill-red">☁ &nbsp;Cloud setup failed</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div style="text-align:center;padding:0.1rem 0 0.3rem;"><span class="pill pill-amber">☁ &nbsp;Cloud sync off</span></div>',
            unsafe_allow_html=True,
        )

    # Onboarding hint
    if not st.session_state.get("prism_ready"):
        st.markdown(
            """
<div style="margin:0.7rem 0.5rem 0;padding:0.7rem 0.85rem;
            background:rgba(37,99,235,0.08);border-radius:10px;
            border:1px solid rgba(37,99,235,0.18);font-size:0.74rem;
            color:#93C5FD;line-height:1.6;">
    <strong style="color:#60A5FA;">👋 Getting started</strong><br>
    Upload your OULAD files, then follow the pipeline steps top-to-bottom.
</div>
""",
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
<div class="sidebar-footer">
    Taylor's University<br>
    <strong>MAC Capstone 2025/26</strong><br>
    Supervisor: Assoc. Prof. Dr. Shakeel Ahmed
</div>
""",
        unsafe_allow_html=True,
    )

# ── Supabase setup banners ────────────────────────────────────────────────────
if _schema_status.get("ran") and _schema_status.get("ok"):
    st.toast("☁️ Supabase database set up automatically!", icon="✅")
elif _schema_status.get("needs_pat"):
    st.markdown(
        """
<div class="setup-banner">
    <strong>☁️ One-time cloud setup needed</strong><br>
    Add your Supabase Personal Access Token to <code>.env</code>:<br><br>
    <code>SUPABASE_PAT=your_token_here</code><br><br>
    Get it at <strong>supabase.com → Account → Access Tokens</strong>.
    Restart the app after adding it. Cloud sync is optional — the app works fully without it.
</div>
""",
        unsafe_allow_html=True,
    )
elif _schema_status.get("ran") and not _schema_status.get("ok"):
    st.warning(
        f"⚠️ Auto Supabase setup failed: {_schema_status.get('message','')}  \nApp still works fully.",
        icon="⚠️",
    )

# ── Page routing ──────────────────────────────────────────────────────────────
selected_key = nav_keys[nav_labels.index(page)]

if selected_key == "Dashboard Overview":
    from pages import dashboard

    dashboard.render()
elif selected_key == "Upload & Configure Data":
    from pages import upload_data

    upload_data.render()
elif selected_key == "Dataset Overview":
    from pages import dataset_overview

    dataset_overview.render()
elif selected_key == "Preprocessing Pipeline":
    from pages import preprocessing_page

    preprocessing_page.render()
elif selected_key == "Model Training & Tuning":
    from pages import model_training

    model_training.render()
elif selected_key == "Student Risk Predictions":
    from pages import predictions

    predictions.render()
elif selected_key == "Prediction Summary":
    from pages import prediction_summary

    prediction_summary.render()
elif selected_key == "Explainability Insights":
    from pages import explainability

    explainability.render()
elif selected_key == "Feature Importance":
    from pages import feature_importance

    feature_importance.render()
elif selected_key == "About PRISM":
    from pages import about

    about.render()
