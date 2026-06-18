import streamlit as st

st.set_page_config(
    page_title="Impactmessung Park & Pipe",
    page_icon="⛷️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import socket
_is_local = socket.gethostname() != "streamlit"

if not _is_local:
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        st.markdown("### Impactmessung Park & Pipe")
        pw = st.text_input("Passwort", type="password")
        if st.button("Einloggen"):
            if pw == "afterbang":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort")
        st.stop()

st.markdown("### Impactmessung Park & Pipe")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px !important;
    font-weight: 500 !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    background: white !important;
    border: 1.5px solid rgba(0,0,0,0.2) !important;
    color: rgba(0,0,0,0.5) !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f0f2f6 !important;
    color: rgba(0,0,0,0.8) !important;
}
.stTabs [aria-selected="true"] {
    background: #185FA5 !important;
    color: #E6F1FB !important;
    border-color: #185FA5 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

if _is_local:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Daten laden", "📊 Sprunganalyse", "🗺️ GPS & Sprünge", "📡 GPS-Rohdaten", "🔬 Validierung"])
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Daten laden", "📊 Sprunganalyse", "🗺️ GPS & Sprünge", "📡 GPS-Rohdaten"])
    tab5 = None

with tab1:
    from pages import import_page
    import_page.show()

with tab2:
    from pages import analyse_page
    analyse_page.show()
    st.divider()
    from pages import stats_page
    stats_page.show()

with tab3:
    from pages import map_page
    map_page.show()

with tab4:
    from pages import gnss_page
    gnss_page.show()

if tab5 is not None:
    with tab5:
        from pages import validation_page
        validation_page.show()
