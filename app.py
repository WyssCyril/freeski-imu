import streamlit as st
import socket
from utils import brand

st.set_page_config(
    page_title="Impactmessung Park & Pipe",
    page_icon="IMU",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{brand.base_css()}</style>", unsafe_allow_html=True)

import os as _os_check
_is_local = _os_check.path.exists("/Users/cyrilwyss/AAMasterarbeit")

if False and not _is_local:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.markdown("""
        <style>
        .login-box {
            max-width: 380px; margin: 80px auto; padding: 40px;
            background: white; border-radius: 16px;
            box-shadow: 0 4px 24px rgba(226,0,26,0.12);
            border-top: 5px solid #E2001A;
        }
        </style>
        <div class="login-box">
          <img src="__SS_LOGO__" style="width:180px;margin-bottom:24px;display:block;">
          <p style="color:#555;margin-bottom:8px;">Impactmessung Park & Pipe</p>
        </div>
        """.replace("__SS_LOGO__", brand.logo_uri()), unsafe_allow_html=True)
        pw = st.text_input("Passwort", type="password")
        if st.button("Einloggen", type="primary"):
            if pw == "afterbang":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort")
        st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

:root {
    --ss-red: #E2001A;
    --ss-red-dark: #C00418;
    --ss-ink: #3E3E3E;
    --ss-muted: #6E6E6E;
    --ss-bg-head: #F4F4F4;
    --ss-bg-nav: #F7F7F7;
    --ss-line: #E3E3E3;
}

/* Hide Streamlit's own top bar so the custom header sits at the very top */
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 0 !important; }

/* Full-bleed helper: stretch an element over the whole viewport width */
.swiss-header,
.st-key-active_tab {
    width: 100vw !important;
    max-width: 100vw !important;
    position: relative;
    left: 50%;
    margin-left: -50vw !important;
}

/* Header band */
.swiss-header {
    font-family: 'Cera Pro', 'Helvetica Neue', Arial, sans-serif;
    background:
        radial-gradient(ellipse 520px 260px at 84% 55%, rgba(226,0,26,0.16), rgba(226,0,26,0) 70%),
        var(--ss-bg-head);
    border-bottom: 1px solid var(--ss-line);
}
.swiss-header-inner {
    max-width: 1600px;
    margin: 0 auto;
    padding: 44px 5rem 30px 5rem;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
}
.swiss-header h1 {
    font-family: 'Cera Pro', 'Helvetica Neue', Arial, sans-serif;
    color: var(--ss-ink);
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
    padding: 0;
}
.swiss-header h1 .dot { color: var(--ss-red); }
.swiss-header .subtitle {
    color: var(--ss-muted);
    font-size: 1.05rem;
    margin-top: 10px;
}
.swiss-header img { height: 40px; margin-top: 2px; }

/* Main navigation: uppercase text tabs with red underline */
.st-key-active_tab {
    font-family: 'Cera Pro', 'Helvetica Neue', Arial, sans-serif;
    background: var(--ss-bg-nav);
    border-bottom: 1px solid var(--ss-line);
    margin-top: -1rem;
    margin-bottom: 1.5rem;
}
.st-key-active_tab div[data-testid="stRadio"] {
    max-width: 1600px;
    margin: 0 auto;
    padding: 0 5rem;
}
.st-key-active_tab div[role="radiogroup"] { gap: 32px !important; flex-wrap: wrap; }
.st-key-active_tab label[data-testid="stRadioOption"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    padding: 20px 0 16px 0 !important;
    margin: 0 !important;
    color: var(--ss-muted) !important;
}
.st-key-active_tab label[data-testid="stRadioOption"] p {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.st-key-active_tab label[data-testid="stRadioOption"]:hover {
    background: transparent !important;
    color: var(--ss-ink) !important;
}
.st-key-active_tab label[data-testid="stRadioOption"]:has(input:checked) {
    color: var(--ss-ink) !important;
    border-bottom-color: var(--ss-red) !important;
}

@media (max-width: 760px) {
    .swiss-header-inner { padding: 48px 1rem 20px 1rem; }
    .swiss-header h1 { font-size: 1.5rem; }
    .swiss-header img { height: 28px; }
    .st-key-active_tab div[data-testid="stRadio"] { padding: 0 1rem; }
    .st-key-active_tab div[role="radiogroup"] { gap: 20px !important; flex-wrap: nowrap; overflow-x: auto; }
    .st-key-active_tab label[data-testid="stRadioOption"] { padding: 14px 0 10px 0 !important; flex-shrink: 0; }
    .st-key-active_tab label[data-testid="stRadioOption"] p { white-space: nowrap; font-size: 0.8rem !important; }
}

/* Secondary selectors inside pages (radios as pills) */
div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"] {
    padding: 8px 20px; border-radius: 999px; background: white;
    border: 1px solid var(--ss-line); color: var(--ss-ink); font-size: 15px; font-weight: 600;
    cursor: pointer;
}
div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {
    border-color: #BDBDBD;
}
div[data-testid="stRadio"] label[data-testid="stRadioOption"]:has(input:checked) {
    background: #3E3E3E; border-color: #3E3E3E; color: white;
}
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:not(:has(p)) { display: none !important; }
div[data-testid="stRadio"] label[data-testid="stRadioOption"] p { color: inherit !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none !important; }
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important; font-weight: 600 !important;
    padding: 8px 20px !important; border-radius: 999px !important;
    background: white !important; border: 1px solid var(--ss-line) !important;
    color: var(--ss-ink) !important;
}
.stTabs [aria-selected="true"] {
    background: #3E3E3E !important; color: white !important; border-color: #3E3E3E !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* Buttons: pill shape, primary in Swiss-Ski red */
.stButton > button, .stDownloadButton > button { border-radius: 999px !important; font-weight: 600; }
.stButton > button[kind="primary"] {
    background: var(--ss-red) !important;
    border-color: var(--ss-red) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--ss-red-dark) !important;
    border-color: var(--ss-red-dark) !important;
}

/* Metrics: red accent line */
[data-testid="stMetric"] {
    border-left: 3px solid var(--ss-red);
    padding-left: 10px;
}
</style>

<div class="swiss-header">
  <div class="swiss-header-inner">
    <div>
      <h1>Impactmessung Park &amp; Pipe<span class="dot">.</span></h1>
      <div class="subtitle">Sprung- und Landekraftanalyse Swiss Freeski · Saison 2026/27</div>
    </div>
    <img src="__SS_LOGO__" alt="Swiss-Ski">
  </div>
</div>
""".replace("__SS_LOGO__", brand.logo_uri()), unsafe_allow_html=True)

# ── Bereichswahl ──────────────────────────────────────────────────────────────
if "bereich" not in st.session_state:
    st.session_state["bereich"] = "analyse"

def _safe_show(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as e:
        st.error(f"Fehler: {e}")
        st.exception(e)

bereich = st.session_state["bereich"]

if bereich == "validierung":
    # ── Validierungsbereich ────────────────────────────────────────────
    col_back, col_title = st.columns([1, 8])
    if col_back.button("Zuruck zur Analyse"):
        st.session_state["bereich"] = "analyse"
        st.rerun()
    col_title.subheader("Validierung — Sensor vs. Kraftmessplatte")
    from pages import validation_page
    _safe_show(validation_page.show)

else:
    # ── Sprunganalyse ──────────────────────────────────────────────────
    # Nur der gewählte Bereich wird gerechnet (st.tabs würde bei jedem Klick alle Tabs neu rendern)
    TABS = ["Daten laden", "Sprunganalyse", "Ergebnisse", "GPS & Sprünge", "GPS-Rohdaten"]
    if "_goto_tab" in st.session_state:
        st.session_state["active_tab"] = st.session_state.pop("_goto_tab")
    tab = st.radio("Bereich", TABS, horizontal=True, key="active_tab",
                   label_visibility="collapsed")

    if tab == "Daten laden":
        from pages import import_page
        _safe_show(import_page.show)

    elif tab == "Sprunganalyse":
        from pages import analyse_page
        _safe_show(analyse_page.show)
        st.divider()
        from pages import stats_page
        _safe_show(stats_page.show)

    elif tab == "Ergebnisse":
        from pages import results_page
        _safe_show(results_page.show)

    elif tab == "GPS & Sprünge":
        from pages import map_page
        _safe_show(map_page.show)

    elif tab == "GPS-Rohdaten":
        from pages import gnss_page
        _safe_show(gnss_page.show)

    # ── Validierung Button unten ───────────────────────────────────────
    st.divider()
    st.markdown("**Validierung**")
    st.caption("Rohdaten-Vergleich IMU vs. Kraftmessplatte zur Qualitätssicherung.")
    if st.button("Validierung öffnen"):
        st.session_state["bereich"] = "validierung"
        st.rerun()
