from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="PRISM | Predictive Risk Identification for Student Monitoring",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    /* Ink (sidebar + dark surfaces) */
    --ink:          #0B0E16;
    --ink-2:        #141A29;
    --ink-line:     rgba(255,255,255,0.07);
    --text-on-ink:      #E7E9F0;
    --text-on-ink-dim:  #7C8298;

    /* Spectrum — the dispersion scale. Doubles as the risk-tier scale:
       teal (calm/low) -> amber (caution/medium) -> coral (alert/high) */
    --spectrum-violet: #7C6FF0;
    --spectrum-blue:   #4F8EF7;
    --spectrum-teal:   #14B8A6;
    --spectrum-amber:  #F5A623;
    --spectrum-coral:  #F0506B;
    --spectrum-gradient: linear-gradient(90deg, var(--spectrum-violet), var(--spectrum-blue), var(--spectrum-teal), var(--spectrum-amber), var(--spectrum-coral));

    /* Paper (light canvas, used by cards/badges/boxes across pages) */
    --ink-on-paper:    #1B2030;
    --muted-on-paper:  #64748B;
    --hairline:        #E2E8F0;
}

@media (prefers-reduced-motion: no-preference) {
    @keyframes prism-shimmer {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }
    .spectrum-divider, .page-header::before { animation: prism-shimmer 6s ease-in-out infinite; }
}

body, .stApp { font-family: 'Inter', sans-serif; }

/* ── Sidebar base ── */
[data-testid="stSidebar"] {
    background: radial-gradient(140% 100% at 0% 0%, var(--ink-2) 0%, var(--ink) 55%) !important;
    border-right: 1px solid var(--ink-line) !important;
}
[data-testid="stSidebar"] * { color: var(--text-on-ink) !important; }

/* ── Spectrum divider — the signature element ── */
.spectrum-divider {
    height: 2px;
    margin: 0.85rem 1.1rem;
    border-radius: 2px;
    background: var(--spectrum-gradient);
    background-size: 200% 100%;
    opacity: 0.85;
}

/* ── Section eyebrows (CSS-only, no DOM change needed) ── */
[data-testid="stSidebar"] .sidebar-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--text-on-ink-dim) !important;
    padding: 0.5rem 1.2rem 0.3rem !important;
}

/* ── Radio group (nav) ── */
[data-testid="stSidebar"] .stRadio > div { gap: 1px !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }

[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    padding: 0.5rem 1rem 0.5rem 1.1rem !important;
    margin: 1px 8px !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    border-left: 3px solid transparent !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }

/* Eyebrow labels — encode the real pipeline grouping (Overview / Data / Model / Insights / Info) */
[data-testid="stSidebar"] .stRadio label::before { content: none; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(1)::before  { content: "Overview"; margin-top: 4px; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(2)::before  { content: "Data";      margin-top: 14px; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(4)::before  { content: "Model";     margin-top: 14px; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(6)::before  { content: "Insights";  margin-top: 14px; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(10)::before { content: "Info";      margin-top: 14px; }
[data-testid="stSidebar"] .stRadio label:nth-of-type(1)::before,
[data-testid="stSidebar"] .stRadio label:nth-of-type(2)::before,
[data-testid="stSidebar"] .stRadio label:nth-of-type(4)::before,
[data-testid="stSidebar"] .stRadio label:nth-of-type(6)::before,
[data-testid="stSidebar"] .stRadio label:nth-of-type(10)::before {
    display: block;
    flex-basis: 100%;
    order: -1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-on-ink-dim);
    padding: 0 0.1rem 0.4rem;
}

/* Hover */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(79,142,247,0.07) !important;
    border-left: 3px solid rgba(79,142,247,0.4) !important;
}

/* Active item (modern browsers via :has) */
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(79,142,247,0.12) !important;
    border-left: 3px solid var(--spectrum-blue) !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #fff !important;
    font-weight: 600 !important;
}

/* Visible keyboard focus */
[data-testid="stSidebar"] .stRadio label:focus-within {
    outline: 2px solid var(--spectrum-blue) !important;
    outline-offset: 2px !important;
}

/* ── Status pills ── */
.status-ready, .status-waiting, .status-info {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 999px; padding: 3px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.02em;
}
.status-ready {
    background: rgba(20,184,166,0.12);
    border: 1px solid rgba(20,184,166,0.3);
    color: #5eead4 !important;
}
.status-waiting {
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.3);
    color: #fcd34d !important;
}
.status-info {
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: #93c5fd !important;
}

/* ── Metric cards (light canvas) ── */
[data-testid="metric-container"] {
    background: #fff; border-radius: 12px; padding: 1rem;
    box-shadow: 0 2px 10px rgba(15,23,42,0.06);
    border-left: 4px solid var(--spectrum-blue);
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Page header banner ── */
.page-header {
    position: relative;
    background: linear-gradient(135deg, var(--ink) 0%, var(--ink-2) 100%);
    color: #fff; padding: 1.3rem 1.6rem 1.2rem; border-radius: 12px;
    margin-bottom: 1.5rem; overflow: hidden;
}
.page-header::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--spectrum-gradient); background-size: 100% 200%;
}
.page-header h2 {
    margin: 0; font-size: 1.45rem;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
}
.page-header p { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.75; }

/* ── Risk badges — directly reuse the spectrum's risk-tier mapping ── */
.badge-high   { background:#fef1f3; color:#b91c3c; border:1px solid rgba(240,80,107,0.35);  border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-medium { background:#fef8ec; color:#92400e; border:1px solid rgba(245,166,35,0.35);  border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }
.badge-low    { background:#effbf8; color:#0f766e; border:1px solid rgba(20,184,166,0.35);  border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600; }

/* ── Info / warn / success boxes ── */
.info-box    { background:#eff6ff; border-left:4px solid var(--spectrum-blue);  padding:.75rem 1rem; border-radius:6px; margin:.5rem 0; }
.warn-box    { background:#fef8ec; border-left:4px solid var(--spectrum-amber); padding:.75rem 1rem; border-radius:6px; margin:.5rem 0; }
.success-box { background:#effbf8; border-left:4px solid var(--spectrum-teal);  padding:.75rem 1rem; border-radius:6px; margin:.5rem 0; }

.tech-box {
    background:#F8FAFC; border:1px solid var(--hairline); border-radius:8px;
    padding:1rem; font-family:'IBM Plex Mono', monospace; font-size:.8rem;
}

hr.section {
    border: none; height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, var(--hairline) 15%, var(--hairline) 85%, transparent);
}

/* ── Sidebar footer card ── */
.sidebar-footer {
    text-align:center; margin-top:1.2rem; padding:0.85rem 0.8rem;
    background:rgba(255,255,255,0.03); border-radius:10px;
    border:1px solid var(--ink-line);
}
.sidebar-footer .org   { font-size:0.63rem; color:var(--text-on-ink-dim); line-height:1.7; font-family:'IBM Plex Mono', monospace; }
.sidebar-footer .proj  { color:#9aa3b8; }
.sidebar-footer .super { color:#4b5567; font-size:0.58rem; }

/* Hide Streamlit's auto-generated pages nav */
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    # ── Logo / Branding ────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="text-align:center; padding:1.6rem 0 0.4rem;">
        <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:56px; height:56px; border-radius:14px;
            background:var(--ink-2);
            border:1px solid var(--ink-line);
            margin-bottom:0.7rem;">
            <svg width="32" height="32" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
                <line x1="2" y1="17" x2="13" y2="17" stroke="#E7E9F0" stroke-width="1.6"/>
                <polygon points="13,8 13,26 24,17" fill="none" stroke="#E7E9F0" stroke-width="1.6"/>
                <line x1="24" y1="17" x2="32" y2="6"  stroke="#7C6FF0" stroke-width="1.6"/>
                <line x1="24" y1="17" x2="33" y2="12" stroke="#4F8EF7" stroke-width="1.6"/>
                <line x1="24" y1="17" x2="33" y2="17" stroke="#14B8A6" stroke-width="1.6"/>
                <line x1="24" y1="17" x2="33" y2="22" stroke="#F5A623" stroke-width="1.6"/>
                <line x1="24" y1="17" x2="32" y2="28" stroke="#F0506B" stroke-width="1.6"/>
            </svg>
        </div>
        <div style="
            font-size:1.4rem; font-weight:700; letter-spacing:3px;
            font-family:'Space Grotesk', sans-serif; color:#fff;">
            PRISM
        </div>
        <div style="font-size:0.62rem; color:#7C8298; margin-top:3px; letter-spacing:0.04em; font-family:'IBM Plex Mono', monospace;">
            Predictive Risk Identification<br>for Student Monitoring
        </div>
    </div>
    <div class="spectrum-divider"></div>
    """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Upload & Configure Data",
            "Dataset Overview",
            "Preprocessing Pipeline",
            "Model Training & Tuning",
            "Student Risk Predictions",
            "Prediction Summary",
            "Explainability Insights",
            "Feature Importance",
            "About PRISM",
        ],
        label_visibility="collapsed",
    )

    # ── Status ─────────────────────────────────────────────────────────────
    st.markdown("<div class='spectrum-divider'></div>", unsafe_allow_html=True)

    if st.session_state.get("prism_ready"):
        st.markdown(
            """
        <div style="text-align:center; padding:0.3rem 0;">
            <span class="status-ready">● &nbsp;Models ready</span>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div style="text-align:center; padding:0.3rem 0;">
            <span class="status-waiting">◌ &nbsp;Awaiting data…</span>
        </div>""",
            unsafe_allow_html=True,
        )

    from utils.supabase_client import is_connected as _supabase_connected
    if _supabase_connected():
        st.markdown(
            """
        <div style="text-align:center; padding:0.15rem 0;">
            <span class="status-info">☁ &nbsp;Supabase connected</span>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div style="text-align:center; padding:0.15rem 0;">
            <span class="status-waiting">☁ &nbsp;Supabase not configured</span>
        </div>""",
            unsafe_allow_html=True,
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown(
        """
    <div class="sidebar-footer">
        <div class="org">
            Taylor's University<br>
            <span class="proj">MAC Capstone Project</span><br>
            <span class="super">Supervisor: Assoc. Prof. Dr. Shakeel Ahmed</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

if "Dashboard Overview" in page:
    from pages import dashboard

    dashboard.render()
elif "Upload & Configure Data" in page:
    from pages import upload_data

    upload_data.render()
elif "Dataset Overview" in page:
    from pages import dataset_overview

    dataset_overview.render()
elif "Preprocessing Pipeline" in page:
    from pages import preprocessing_page

    preprocessing_page.render()
elif "Model Training & Tuning" in page:
    from pages import model_training

    model_training.render()
elif "Student Risk Predictions" in page:
    from pages import predictions

    predictions.render()
elif "Prediction Summary" in page:
    from pages import prediction_summary

    prediction_summary.render()
elif "Explainability Insights" in page:
    from pages import explainability

    explainability.render()
elif "Feature Importance" in page:
    from pages import feature_importance

    feature_importance.render()
elif "About PRISM" in page:
    from pages import about

    about.render()