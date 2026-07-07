"""
Ergebnisse: Gespeicherte Sprungauswertungen pro Athlet/Sensor, Excel-Export.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _collect_results(sessions_loaded: dict) -> pd.DataFrame:
    """Sammelt alle gespeicherten Sprungresultate aus dem Session-State."""
    jump_results = st.session_state.get("jump_results", {})
    rows = []
    for run_key, entry in jump_results.items():
        jumps = entry.get("jumps")
        m = entry.get("meta")
        if jumps is None or (hasattr(jumps, "empty") and jumps.empty):
            continue
        try:
            date_fmt = pd.to_datetime(str(m.date), format="%Y%m%d").strftime("%d.%m.%Y") if m else ""
        except Exception:
            date_fmt = m.date if m else ""
        for _, jrow in jumps.iterrows():
            rows.append({
                "Athlet": m.athlete_code if m else run_key,
                "Datum": date_fmt,
                "Ort": m.location if m else "",
                "Position": m.position_label if m else "",
                "Sprung": jrow.get("jump_id", ""),
                "Flugzeit (s)": round(float(jrow.get("flight_time_s", 0)), 3),
                "Peak (g)": round(float(jrow.get("peak_res_g", 0)), 2),
                "Peak roh (g)": round(float(jrow.get("peak_res_g_raw", jrow.get("peak_res_g", 0))), 2),
                "TTP (s)": round(float(jrow.get("time_to_peak_s", 0)), 4),
                "RFD (g/s)": round(float(jrow.get("rfd_g_per_s", 0)), 2),
                "Impuls (g·s)": round(float(jrow.get("impulse_g_s", 0)), 4),
                "16g geclippt": bool(jrow.get("clipped_16g", False)),
                "Landungsart": jrow.get("landing_type", ""),
                "Kommentar": jrow.get("comment", ""),
            })
    return pd.DataFrame(rows)


def show():
    st.header("Ergebnisse")

    sessions_loaded = st.session_state.get("loaded_sessions", {})
    jump_results = st.session_state.get("jump_results", {})

    if not jump_results:
        st.info("Noch keine Auswertungen vorhanden. Zuerst Sprunganalyse durchführen.")
        return

    df = _collect_results(sessions_loaded)
    if df.empty:
        st.info("Noch keine Sprünge ausgewertet.")
        return

    st.caption(f"{len(df)} Sprünge aus {df['Athlet'].nunique()} Athleten, {df['Ort'].nunique()} Orten")

    # ── Filter ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    all_athletes = sorted(df["Athlet"].unique())
    sel_athletes = col1.multiselect("Athlet", all_athletes, default=all_athletes)
    all_orte = sorted(df["Ort"].unique())
    sel_orte = col2.multiselect("Ort", all_orte, default=all_orte)
    all_pos = sorted(df["Position"].unique())
    sel_pos = col3.multiselect("Sensorposition", all_pos, default=all_pos)

    mask = (
        df["Athlet"].isin(sel_athletes) &
        df["Ort"].isin(sel_orte) &
        df["Position"].isin(sel_pos)
    )
    df_filtered = df[mask].reset_index(drop=True)

    st.markdown(f"**{len(df_filtered)} Sprünge gefiltert**")

    # ── Kennzahlen Übersicht ──────────────────────────────────────────────
    if not df_filtered.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ø Peak (g)", f"{df_filtered['Peak (g)'].mean():.2f}")
        c2.metric("Max. Peak (g)", f"{df_filtered['Peak (g)'].max():.2f}")
        c3.metric("Ø Flugzeit (s)", f"{df_filtered['Flugzeit (s)'].mean():.3f}")
        c4.metric("Geclippt (16g)", int(df_filtered["16g geclippt"].sum()))

    # ── Tabelle pro Athlet ────────────────────────────────────────────────
    st.divider()
    group_by = st.radio("Gruppieren nach", ["Athlet", "Ort", "Position", "Datum"], horizontal=True)
    group_col = {"Athlet": "Athlet", "Ort": "Ort", "Position": "Position", "Datum": "Datum"}[group_by]

    display_cols = ["Athlet", "Datum", "Ort", "Position", "Run", "Sprung",
                    "Flugzeit (s)", "Peak (g)", "Peak roh (g)", "TTP (s)", "RFD (g/s)",
                    "Impuls (g·s)", "16g geclippt", "Landungsart", "Kommentar"]

    for grp_val, grp_df in df_filtered.groupby(group_col, sort=True):
        with st.expander(f"**{grp_val}** — {len(grp_df)} Sprünge", expanded=False):
            st.dataframe(grp_df[display_cols].reset_index(drop=True), use_container_width=True)

            # Kennzahlen pro Gruppe
            c1, c2, c3 = st.columns(3)
            c1.metric("Ø Peak (g)", f"{grp_df['Peak (g)'].mean():.2f}")
            c2.metric("Ø Flugzeit (s)", f"{grp_df['Flugzeit (s)'].mean():.3f}")
            c3.metric("Max. Peak (g)", f"{grp_df['Peak (g)'].max():.2f}")

    # ── Excel-Export ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Excel-Export")

    if st.button("Excel erstellen", type="primary"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # Gesamtübersicht
            df_filtered[display_cols].to_excel(writer, sheet_name="Alle Sprünge", index=False)

            # Pro Athlet ein Sheet
            for athlet, grp in df_filtered.groupby("Athlet"):
                sheet_name = f"Athlet {athlet}"[:31]
                grp[display_cols].to_excel(writer, sheet_name=sheet_name, index=False)

            # Zusammenfassung
            summary = df_filtered.groupby(["Athlet", "Ort", "Position"]).agg(
                Anzahl_Sprünge=("Sprung", "count"),
                Peak_mean=("Peak (g)", "mean"),
                Peak_max=("Peak (g)", "max"),
                Flugzeit_mean=("Flugzeit (s)", "mean"),
                Flugzeit_max=("Flugzeit (s)", "max"),
                TTP_mean=("TTP (s)", "mean"),
                RFD_mean=("RFD (g/s)", "mean"),
                Geclippt=("16g geclippt", "sum"),
            ).round(3).reset_index()
            summary.to_excel(writer, sheet_name="Zusammenfassung", index=False)

        buf.seek(0)
        st.download_button(
            "Excel herunterladen",
            data=buf,
            file_name="impactmessung_ergebnisse.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
