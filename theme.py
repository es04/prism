"""
PRISM v2 – Shared Design System
Centralised CSS, colour tokens, and component helpers.
"""

import streamlit as st

# ── Colour tokens (light + dark) ──────────────────────────────────────────────
LIGHT = {
    "bg":          "#F0F2F7",
    "surface":     "#FFFFFF",
    "surface2":    "#F8FAFC",
    "border":      "#E2E8F0",
    "text":        "#0F172A",
    "text2":       "#475569",
    "text3":       "#94A3B8",

    "blue":        "#2563EB",
    "blue_light":  "#DBEAFE",
    "teal":        "#0D9488",
    "teal_light":  "#CCFBF1",
    "amber":       "#D97706",
    "amber_light": "#FEF3C7",
    "red":         "#DC2626",
    "red_light":   "#FEE2E2",
    "violet":      "#7C3AED",
    "violet_light":"#EDE9FE",

    "sidebar_bg":  "#0F172A",
    "sidebar_text":"#E2E8F0",
    "sidebar_muted":"#64748B",
}

DARK = {
    "bg":          "#0B0F1A",
    "surface":     "#111827",
    "surface2":    "#1A2234",
    "border":      "#1E2D4A",
    "text":        "#F1F5F9",
    "text2":       "#94A3B8",
    "text3":       "#64748B",

    "blue":        "#3B82F6",
    "blue_light":  "rgba(59,130,246,0.15)",
    "teal":        "#14B8A6",
    "teal_light":  "rgba(20,184,166,0.15)",
    "amber":       "#F59E0B",
    "amber_light": "rgba(245,158,11,0.15)",
    "red":         "#EF4444",
    "red_light":   "rgba(239,68,68,0.15)",
    "violet":      "#8B5CF6",
    "violet_light":"rgba(139,92,246,0.15)",

    "sidebar_bg":  "#070B14",
    "sidebar_text":"#E2E8F0",
    "sidebar_muted":"#475569",
}


def is_dark() -> bool:
    return st.session_state.get("dark_mode", False)


def tok(key: str) -> str:
    """Return current theme colour token."""
    return (DARK if is_dark() else LIGHT)[key]


def inject_theme_css():
    dark = is_dark()
    t = DARK if dark else LIGHT

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {{
    --bg:           {t['bg']};
    --surface:      {t['surface']};
    --surface2:     {t['surface2']};
    --border:       {t['border']};
    --text:         {t['text']};
    --text2:        {t['text2']};
    --text3:        {t['text3']};
    --blue:         {t['blue']};
    --blue-light:   {t['blue_light']};
    --teal:         {t['teal']};
    --teal-light:   {t['teal_light']};
    --amber:        {t['amber']};
    --amber-light:  {t['amber_light']};
    --red:          {t['red']};
    --red-light:    {t['red_light']};
    --violet:       {t['violet']};
    --violet-light: {t['violet_light']};
    --sidebar-bg:   {t['sidebar_bg']};
    --sidebar-text: {t['sidebar_text']};
    --sidebar-muted:{t['sidebar_muted']};
    --radius-sm:    8px;
    --radius-md:    12px;
    --radius-lg:    16px;
    --shadow:       {'0 1px 3px rgba(0,0,0,0.08),0 2px 8px rgba(0,0,0,0.06)' if not dark else '0 1px 3px rgba(0,0,0,0.4),0 2px 8px rgba(0,0,0,0.3)'};
    --shadow-md:    {'0 4px 16px rgba(0,0,0,0.10)' if not dark else '0 4px 16px rgba(0,0,0,0.5)'};
    --font:         'Inter', -apple-system, sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --accent-bar:   linear-gradient(90deg, #7C3AED, #2563EB, #0D9488, #D97706, #DC2626);
}}

/* ── Base ── */
html, body, .stApp {{
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--sidebar-text) !important; }}

[data-testid="collapsedControl"] {{
    background: var(--sidebar-bg) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
    color: var(--sidebar-text) !important;
}}

/* Hide default Streamlit nav */
[data-testid="stSidebarNav"] {{ display: none !important; }}

/* ── Sidebar nav radio ── */
[data-testid="stSidebar"] .stRadio > label {{ display: none !important; }}
[data-testid="stSidebar"] .stRadio > div   {{ gap: 1px !important; }}
[data-testid="stSidebar"] .stRadio label {{
    display: flex !important;
    align-items: center !important;
    padding: 0.55rem 0.9rem 0.55rem 1rem !important;
    margin: 0 6px !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.12s !important;
    border-left: 3px solid transparent !important;
    background: transparent !important;
}}
[data-testid="stSidebar"] .stRadio label > div:first-child {{ display: none !important; }}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(59,130,246,0.10) !important;
    border-left-color: rgba(59,130,246,0.4) !important;
}}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: rgba(37,99,235,0.18) !important;
    border-left-color: var(--blue) !important;
}}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {{
    color: #fff !important;
    font-weight: 700 !important;
}}

/* ── Main area padding ── */
[data-testid="stMain"] > div:first-child {{ padding-top: 0.75rem !important; }}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: var(--surface2) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    padding: 3px !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: var(--surface) !important;
    border-radius: calc(var(--radius-sm) - 2px) !important;
    color: var(--blue) !important;
    box-shadow: var(--shadow) !important;
}}

/* ── Buttons ── */
.stButton > button {{
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    background: var(--blue-light) !important;
}}
.stButton > button[kind="primary"] {{
    background: var(--blue) !important;
    color: #fff !important;
    border-color: var(--blue) !important;
}}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stMultiSelect [data-baseweb="select"] {{
    font-family: var(--font) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.88rem !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    background: var(--surface) !important;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    font-family: var(--font) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    border-radius: var(--radius-sm) !important;
    background: var(--teal) !important;
    color: #fff !important;
    border: none !important;
}}
.stDownloadButton > button:hover {{
    opacity: 0.9 !important;
}}

/* ── Spinner / progress ── */
.stSpinner > div > div {{ border-top-color: var(--blue) !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--surface2); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 99px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text3); }}

/* ── Accent bar ── */
.accent-bar {{
    height: 3px;
    background: var(--accent-bar);
    border-radius: 3px;
    margin: 0.5rem 0.8rem;
    opacity: 0.75;
}}

/* ── Page header ── */
.page-header {{
    background: {'linear-gradient(135deg,#0F172A 0%,#1E2D4A 100%)' if not dark else 'linear-gradient(135deg,#070B14 0%,#0F172A 100%)'};
    border-radius: var(--radius-lg);
    padding: 1.3rem 1.6rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
}}
.page-header::before {{
    content:"";
    position:absolute;
    left:0; top:0; bottom:0;
    width:4px;
    background: var(--accent-bar);
}}
.page-header h2  {{ margin:0; font-size:1.35rem; font-weight:800; letter-spacing:-0.02em; }}
.page-header p   {{ margin:0.25rem 0 0; font-size:0.83rem; opacity:0.6; }}

/* ── Cards ── */
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem;
    box-shadow: var(--shadow);
}}
.card-accent-blue   {{ border-top: 3px solid var(--blue); }}
.card-accent-teal   {{ border-top: 3px solid var(--teal); }}
.card-accent-amber  {{ border-top: 3px solid var(--amber); }}
.card-accent-red    {{ border-top: 3px solid var(--red); }}
.card-accent-violet {{ border-top: 3px solid var(--violet); }}

/* ── KPI cards ── */
.kpi-label {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text2);
    margin-bottom: 5px;
}}
.kpi-value {{
    font-size: 1.7rem;
    font-weight: 800;
    font-family: var(--font-mono);
    color: var(--text);
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 0.72rem;
    color: var(--text3);
    margin-top: 4px;
}}

/* ── Risk badges ── */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 700;
    font-family: var(--font);
    white-space: nowrap;
}}
.badge-high   {{ background: var(--red-light);    color: {'#B91C1C' if not dark else '#FCA5A5'}; border: 1px solid {'rgba(220,38,38,0.2)' if not dark else 'rgba(239,68,68,0.3)'}; }}
.badge-medium {{ background: var(--amber-light);  color: {'#92400E' if not dark else '#FDE68A'}; border: 1px solid {'rgba(217,119,6,0.2)'  if not dark else 'rgba(245,158,11,0.3)'}; }}
.badge-low    {{ background: var(--teal-light);   color: {'#065F46' if not dark else '#6EE7B7'}; border: 1px solid {'rgba(13,148,136,0.2)'  if not dark else 'rgba(20,184,166,0.3)'}; }}

/* ── Pills (sidebar status) ── */
.pill {{
    display: inline-flex; align-items: center; gap: 5px;
    border-radius: 999px; padding: 3px 10px;
    font-size: 0.71rem; font-weight: 600; font-family: var(--font-mono);
}}
.pill-green  {{ background: rgba(13,148,136,0.15); border: 1px solid rgba(13,148,136,0.3); color: #34D399 !important; }}
.pill-amber  {{ background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: #FCD34D !important; }}
.pill-blue   {{ background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.3); color: #93C5FD !important; }}
.pill-red    {{ background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.3);  color: #FCA5A5 !important; }}

/* ── Info boxes ── */
.info-box    {{ background:var(--blue-light);   border-left:4px solid var(--blue);  padding:.75rem 1rem; border-radius:0 var(--radius-sm) var(--radius-sm) 0; margin:.5rem 0; font-size:0.87rem; color:var(--text); }}
.warn-box    {{ background:var(--amber-light);  border-left:4px solid var(--amber); padding:.75rem 1rem; border-radius:0 var(--radius-sm) var(--radius-sm) 0; margin:.5rem 0; font-size:0.87rem; color:var(--text); }}
.success-box {{ background:var(--teal-light);   border-left:4px solid var(--teal);  padding:.75rem 1rem; border-radius:0 var(--radius-sm) var(--radius-sm) 0; margin:.5rem 0; font-size:0.87rem; color:var(--text); }}
.error-box   {{ background:var(--red-light);    border-left:4px solid var(--red);   padding:.75rem 1rem; border-radius:0 var(--radius-sm) var(--radius-sm) 0; margin:.5rem 0; font-size:0.87rem; color:var(--text); }}

/* ── Section divider ── */
hr.section {{
    border:none; height:1px; margin:1.25rem 0;
    background: linear-gradient(90deg, transparent, var(--border) 10%, var(--border) 90%, transparent);
}}

/* ── Sidebar footer ── */
.sidebar-footer {{
    text-align:center; padding:0.8rem 0.7rem;
    background:rgba(255,255,255,0.03); border-radius:var(--radius-md);
    border:1px solid rgba(255,255,255,0.05); font-size:0.64rem;
    color:var(--sidebar-muted) !important; font-family:var(--font-mono); line-height:1.8;
}}
.sidebar-footer strong {{ color:#9AA3B8 !important; }}

/* ── Setup banner ── */
.setup-banner {{
    background: linear-gradient(135deg,#1E2D4A,#0F172A);
    border:1px solid rgba(59,130,246,0.2); border-left:4px solid #3B82F6;
    border-radius:var(--radius-md); padding:1rem 1.2rem; margin-bottom:1rem;
    color:#E8ECF5; font-size:0.87rem; line-height:1.6;
}}
.setup-banner code {{
    background:rgba(255,255,255,0.08); border-radius:4px; padding:1px 6px;
    font-family:var(--font-mono); font-size:0.81rem; color:#93C5FD;
}}

/* ── Table ── */
.prism-table {{
    width:100%; border-collapse:collapse;
    font-size:0.85rem; font-family:var(--font); color:var(--text);
    white-space:nowrap;
}}
.prism-table thead tr {{
    background:var(--surface2);
    border-bottom:2px solid var(--border);
}}
.prism-table thead th {{
    padding:10px 14px; text-align:left;
    font-weight:700; color:var(--text2);
    font-size:0.71rem; text-transform:uppercase; letter-spacing:0.06em;
}}
.prism-table tbody tr {{
    border-bottom:1px solid var(--border);
    transition:background 0.1s;
    cursor:pointer;
}}
.prism-table tbody tr:hover {{ background:var(--surface2); }}
.prism-table tbody td {{ padding:10px 14px; color:var(--text); }}
.prism-table .mono {{ font-family:var(--font-mono); font-size:0.82rem; }}

/* ── Chart card wrapper ── */
.chart-card {{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius-md);
    padding:1rem 1rem 0;
    box-shadow:var(--shadow);
    margin-bottom:0.5rem;
}}
.chart-card-title {{
    font-size:0.93rem; font-weight:700;
    color:var(--text); margin-bottom:2px;
}}
.chart-card-sub {{
    font-size:0.74rem; color:var(--text3); margin-bottom:0.3rem;
}}

/* ── Dark mode toggle in sidebar ── */
.dark-toggle-wrap {{
    display:flex; align-items:center; justify-content:center;
    gap:8px; padding:0.4rem 0;
    font-size:0.78rem; color:var(--sidebar-muted);
    font-family:var(--font);
}}

/* ── Student profile modal ── */
.profile-header {{
    background: linear-gradient(135deg, #1E2D4A, #0F172A);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1rem;
}}
.profile-stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    text-align: center;
}}
.profile-stat-label {{ font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:var(--text3); }}
.profile-stat-value {{ font-size:1.2rem; font-weight:800; font-family:var(--font-mono); color:var(--text); margin-top:3px; }}

/* ── Export buttons area ── */
.export-bar {{
    display: flex; gap: 8px; align-items: center;
    flex-wrap: wrap; margin: 0.5rem 0;
}}

@media (max-width: 768px) {{
    .page-header {{ padding: 1rem 1.1rem; }}
    .page-header h2 {{ font-size: 1.1rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ── Component helpers ──────────────────────────────────────────────────────────

def page_header(icon: str, title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
<div class="page-header">
    <h2>{icon}&nbsp; {title}</h2>
    {sub}
</div>
""", unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str = "blue") -> None:
    st.markdown(f"""
<div class="card card-accent-{accent}" style="height:100%;">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    {'<div class="kpi-sub">'+sub+'</div>' if sub else ''}
</div>
""", unsafe_allow_html=True)


def chart_card_header(title: str, sub: str = ""):
    sub_html = f'<div class="chart-card-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-md);
            padding:1rem 1rem 0;box-shadow:var(--shadow);margin-bottom:4px;">
    <div class="chart-card-title">{title}</div>
    {sub_html}
</div>
""", unsafe_allow_html=True)


def badge_html(label: str) -> str:
    cls = {
        "High Risk":   "badge-high",
        "Medium Risk": "badge-medium",
        "Low Risk":    "badge-low",
    }.get(label, "badge-low")
    dot = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}.get(label, "")
    return f'<span class="badge {cls}">{dot} {label}</span>'


def info_box(text: str, kind: str = "info"):
    cls = {"info": "info-box", "warn": "warn-box", "success": "success-box", "error": "error-box"}.get(kind, "info-box")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def section_div():
    st.markdown("<hr class='section'>", unsafe_allow_html=True)


# Plotly theme helpers
def plotly_layout(dark_mode: bool) -> dict:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": DARK["surface2"] if dark_mode else "#F8FAFC",
        "font": dict(family="Inter, sans-serif", color=DARK["text"] if dark_mode else LIGHT["text"], size=12),
        "margin": dict(t=24, b=28, l=16, r=16),
    }


RISK_COLORS = {
    "High Risk":   "#DC2626",
    "Medium Risk": "#D97706",
    "Low Risk":    "#0D9488",
}
