import streamlit as st

st.set_page_config(
    page_title="Impactmessung Park & Pipe",
    page_icon="⛷️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Passwort-Login
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

tab1, tab2, tab3, tab4 = st.tabs(["📁 Daten laden", "📊 Sprunganalyse", "🗺️ GPS & Speed", "🔬 Validierung"])

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
    from pages import gnss_page
    gnss_page.show()

with tab4:
    from pages import validation_page
    validation_page.show()
