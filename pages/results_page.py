"""
Ergebnisse: Gespeicherte Sprungauswertungen pro Athlet/Sensor, Excel-Export + Import.
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Spaltenmapping Excel → interne Spalten (für Statistik-Seite)
_EXCEL_TO_INTERNAL = {
    "Peak (g)":      "peak_res_g",
    "Peak roh (g)":  "peak_res_g_raw",
    "Flugzeit (s)":  "flight_time_s",
    "TTP (s)":       "time_to_peak_s",
    "RFD (g/s)":     "rfd_g_per_s",
    "Impuls (g·s)":  "impulse_net_g_s",
    "16g geclippt":  "clipped_16g",
    "Landungsart":   "landing_type",
    "Sprung":        "jump_id",
    "Kommentar":     "comment",
}


class _SimpleMeta:
    """Minimales Meta-Objekt wie admos_parser.SensorMeta, aber aus Excel gebaut."""
    def __init__(self, athlete_code, date, location, position_label):
        self.athlete_code   = athlete_code
        self.date           = date
        self.location       = location
        self.position_label = position_label
        self.sensor_id      = ""


def _import_excel(file) -> int:
    """
    Liest Batch-Excel (Sheet 'Einzelsprünge') und fügt Ergebnisse in
    st.session_state['jump_results'] ein. Gibt Anzahl importierter Sprünge zurück.
    """
    try:
        df = pd.read_excel(file, sheet_name="Einzelsprünge")
    except Exception as e:
        st.error(f"Fehler beim Lesen: {e}")
        return 0

    if df.empty:
        return 0

    # Interne Spaltennamen hinzufügen
    df_int = df.rename(columns=_EXCEL_TO_INTERNAL)

    # Pro Athlet + Position eine synthetische jump_results-Gruppe anlegen
    if "jump_results" not in st.session_state:
        st.session_state["jump_results"] = {}

    count = 0
    group_cols = [c for c in ["Athlet", "Datum", "Ort", "Position"] if c in df_int.columns]
    for group_vals, grp in df_int.groupby(group_cols, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        vals = dict(zip(group_cols, group_vals))

        athlet   = vals.get("Athlet", "?")
        datum    = vals.get("Datum", "")
        ort      = vals.get("Ort", "")
        position = vals.get("Position", "")

        # Datum in YYYYMMDD-Format zurück für SensorMeta
        try:
            date_int = pd.to_datetime(datum, dayfirst=True).strftime("%Y%m%d")
        except Exception:
            date_int = str(datum)

        meta  = _SimpleMeta(str(athlet), date_int, ort, position)
        key   = f"import_{athlet}_{date_int}_{ort}_{position}"

        jumps_df = grp.reset_index(drop=True)

        st.session_state["jump_results"][key] = {
            "jumps":    jumps_df,
            "meta":     meta,
            "run_note": "",
        }
        count += len(grp)

    return count


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
        run_note = entry.get("run_note", "")
        for _, jrow in jumps.iterrows():
            rows.append({
                "Athlet": m.athlete_code if m else run_key,
                "Datum": date_fmt,
                "Ort": m.location if m else "",
                "Position": m.position_label if m else "",
                "Sprung": jrow.get("jump_id", ""),
                "Tricks / Notiz": run_note,
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

    # ── Batch-Import ──────────────────────────────────────────────────────
    with st.expander("📥 Frühere Batches importieren", expanded=False):
        st.caption(
            "Lade hier Excel-Dateien hoch, die du in einem früheren Batch exportiert hast. "
            "Die Daten werden mit den aktuellen Auswertungen zusammengeführt."
        )
        uploaded_batches = st.file_uploader(
            "Excel-Dateien hochladen",
            type=["xlsx"],
            accept_multiple_files=True,
            key="batch_import_upload",
            label_visibility="collapsed",
        )
        if uploaded_batches:
            import_key = "batch_imported_" + "_".join(f.name for f in uploaded_batches)
            if import_key not in st.session_state:
                total = 0
                for f in uploaded_batches:
                    n = _import_excel(f)
                    total += n
                st.session_state[import_key] = total
                st.rerun()
            else:
                st.success(f"{st.session_state[import_key]} Sprünge aus {len(uploaded_batches)} Datei(en) importiert.")

        imported_keys = [k for k in st.session_state.get("jump_results", {}) if k.startswith("import_")]
        if imported_keys:
            n_imp = sum(
                len(st.session_state["jump_results"][k]["jumps"])
                for k in imported_keys
                if st.session_state["jump_results"][k].get("jumps") is not None
            )
            st.info(f"Aktuell {n_imp} importierte Sprünge aus {len(imported_keys)} Batch-Gruppen geladen.")
            if st.button("Importierte Daten löschen", key="clear_imports"):
                for k in imported_keys:
                    del st.session_state["jump_results"][k]
                # Auch Import-Keys löschen
                for k in list(st.session_state.keys()):
                    if k.startswith("batch_imported_"):
                        del st.session_state[k]
                st.rerun()

    st.divider()

    sessions_loaded = st.session_state.get("loaded_sessions", {})
    jump_results = st.session_state.get("jump_results", {})

    if not jump_results:
        st.info("Noch keine Auswertungen vorhanden. Zuerst Sprunganalyse durchführen oder Batch importieren.")
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

    all_cols = ["Athlet", "Datum", "Ort", "Position", "Sprung", "Tricks / Notiz",
                "Flugzeit (s)", "Peak (g)", "Peak roh (g)", "TTP (s)", "RFD (g/s)",
                "Impuls (g·s)", "16g geclippt", "Landungsart", "Kommentar"]
    display_cols = [c for c in all_cols if c in df_filtered.columns]

    for grp_val, grp_df in df_filtered.groupby(group_col, sort=True):
        with st.expander(f"**{grp_val}** — {len(grp_df)} Sprünge", expanded=False):
            st.dataframe(grp_df[display_cols].reset_index(drop=True), use_container_width=True)

            # Kennzahlen pro Gruppe
            c1, c2, c3 = st.columns(3)
            c1.metric("Ø Peak (g)", f"{grp_df['Peak (g)'].mean():.2f}")
            c2.metric("Ø Flugzeit (s)", f"{grp_df['Flugzeit (s)'].mean():.3f}")
            c3.metric("Max. Peak (g)", f"{grp_df['Peak (g)'].max():.2f}")

    # ── Durchschnittlicher Impact ─────────────────────────────────────────
    st.divider()
    st.subheader("Durchschnittlicher Impact")
    if not df_filtered.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ø Peak (g) — alle Sprünge", f"{df_filtered['Peak (g)'].mean():.2f}")
        c2.metric("Median Peak (g)", f"{df_filtered['Peak (g)'].median():.2f}")
        c3.metric("Max. Peak (g)", f"{df_filtered['Peak (g)'].max():.2f}")
        c4.metric("Anzahl Sprünge", len(df_filtered))

        # Pro Position
        st.markdown("**Ø Peak (g) pro Sensorposition:**")
        pos_summary = df_filtered.groupby("Position")["Peak (g)"].agg(["mean", "median", "max", "count"]).round(2)
        pos_summary.columns = ["Ø Peak (g)", "Median (g)", "Max. (g)", "Anzahl"]
        st.dataframe(pos_summary, use_container_width=True)

    # ── Excel-Export ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Excel-Export")

    agg_dict = dict(
        Anzahl_Sprünge=("Sprung", "count"),
        Peak_g_mean=("Peak (g)", "mean"),
        Peak_g_median=("Peak (g)", "median"),
        Peak_g_max=("Peak (g)", "max"),
        Peak_g_min=("Peak (g)", "min"),
        Flugzeit_mean=("Flugzeit (s)", "mean"),
        Flugzeit_max=("Flugzeit (s)", "max"),
        TTP_mean=("TTP (s)", "mean"),
        RFD_mean=("RFD (g/s)", "mean"),
        Impuls_mean=("Impuls (g·s)", "mean"),
        Geclippt_16g=("16g geclippt", "sum"),
    )

    jump_cols = ["Athlet", "Datum", "Ort", "Position", "Sprung",
                 "Flugzeit (s)", "Peak (g)", "Peak roh (g)",
                 "TTP (s)", "RFD (g/s)", "Impuls (g·s)",
                 "16g geclippt", "Landungsart", "Kommentar"]

    def _make_excel(data: pd.DataFrame, label: str) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            data[jump_cols].to_excel(writer, sheet_name="Einzelsprünge", index=False)
            sess = data.groupby(["Athlet", "Datum", "Ort", "Position"], sort=True).agg(**agg_dict).round(3).reset_index()
            sess.to_excel(writer, sheet_name="Sessions", index=False)
            gesamt = data.groupby(["Athlet", "Position"], sort=True).agg(**agg_dict).round(3).reset_index()
            gesamt.to_excel(writer, sheet_name="Gesamt", index=False)
            # Durchschnitt über alle Sprünge
            overall = pd.DataFrame([{
                "Ort": label,
                "Anzahl_Sprünge": len(data),
                "Peak_g_mean": round(data["Peak (g)"].mean(), 3),
                "Peak_g_median": round(data["Peak (g)"].median(), 3),
                "Peak_g_max": round(data["Peak (g)"].max(), 3),
                "Flugzeit_mean": round(data["Flugzeit (s)"].mean(), 3),
                "TTP_mean": round(data["TTP (s)"].mean(), 3),
                "RFD_mean": round(data["RFD (g/s)"].mean(), 3),
            }])
            overall.to_excel(writer, sheet_name="Durchschnitt", index=False)
        buf.seek(0)
        return buf.getvalue()

    orte = sorted(df_filtered["Ort"].dropna().unique())
    cols = st.columns(len(orte) + 1)

    for i, ort in enumerate(orte):
        df_ort = df_filtered[df_filtered["Ort"] == ort]
        cols[i].download_button(
            f"Excel {ort}",
            data=_make_excel(df_ort, ort),
            file_name=f"impactmessung_{ort}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{ort}",
        )

    cols[-1].download_button(
        "Excel Alle",
        data=_make_excel(df_filtered, "Alle"),
        file_name="impactmessung_alle.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_alle",
    )
