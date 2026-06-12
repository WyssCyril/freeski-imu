"""
Daten laden: Drag & Drop oder aus festem Ordner.
"""
import io
import datetime
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.admos_parser import (parse_filename, load_imu_raw, load_gnss_raw,
                                find_csv_pairs, classify_sensor_file,
                                sensor_file_base, DATA_FOLDER, SensorMeta)


def show():
    st.header("Daten laden")

    # ── Aus festem Ordner ──────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    if col_btn.button("Aus Ordner laden", type="primary"):
        pairs = find_csv_pairs(DATA_FOLDER)
        if not pairs:
            st.warning(f"Keine CSV-Dateien gefunden in: {DATA_FOLDER}")
        else:
            loaded = {}
            progress = st.progress(0, text="Lade Dateien …")
            status = st.empty()
            for i, p in enumerate(pairs):
                key = p["meta"].filename.replace("_imuData.csv", "").replace("_imu.csv", "").replace("_IMU.csv", "")
                pct = int((i + 1) / len(pairs) * 100)
                progress.progress(pct, text=f"Lade {i+1}/{len(pairs)}: {p['meta'].athlete_code} | {p['meta'].position_label}")
                try:
                    imu_df = load_imu_raw(p["imu_path"])
                    gnss_df = load_gnss_raw(p["gnss_path"]) if p["gnss_path"] else None
                    loaded[key] = {
                        "imu": imu_df, "gnss": gnss_df,
                        "imu_path": p["imu_path"], "gnss_path": p["gnss_path"],
                        "meta": p["meta"],
                    }
                except Exception as e:
                    st.warning(f"Fehler bei {p['meta'].filename}: {e}")
            progress.empty()
            existing = st.session_state.get("loaded_sessions", {})
            existing.update(loaded)
            st.session_state["loaded_sessions"] = existing
            status.success(f"{len(loaded)} Sensor-Dateien geladen.")


    # ── Manueller Upload mit Metadaten ─────────────────────────────────────
    st.divider()
    st.subheader("Dateien hochladen")
    st.caption("CSV-Dateien hochladen und Messungs-Infos eingeben.")

    uploaded = st.file_uploader(
        "CSV-Dateien", type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        staged: dict[str, dict] = st.session_state.get("_upload_staged", {})
        for f in uploaded:
            kind = classify_sensor_file(f.name)
            if kind is None:
                st.warning(f"Nicht erkannt (erwartet *_IMU.csv / *_GNSS.csv etc.): **{f.name}**")
                continue
            base = sensor_file_base(f.name)
            key = f"{kind}_bytes"
            if base not in staged or key not in staged[base]:
                f.seek(0)
                staged.setdefault(base, {})[key] = f.read()
        st.session_state["_upload_staged"] = staged

    staged = st.session_state.get("_upload_staged", {})

    if staged:
        st.markdown(f"**{len(staged)} Datei(en) bereit** — Infos aus Dateiname erkannt, bitte prüfen:")

        pos_options = ["Bauch", "Fuss rechts", "Fuss links"]
        pos_label_map = {"Bauch": "Bauch", "Fuss_re": "Fuss rechts", "Fuss_li": "Fuss links",
                         "FussRe": "Fuss rechts", "FussLi": "Fuss links",
                         "fuss_re": "Fuss rechts", "fuss_li": "Fuss links"}

        for base in list(staged.keys()):
            # Metadaten aus Dateiname lesen
            try:
                meta = parse_filename(base + "_imuData.csv")
                try:
                    default_date = datetime.date(int(meta.date[:4]), int(meta.date[4:6]), int(meta.date[6:8]))
                except Exception:
                    default_date = datetime.date.today()
                default_ort = meta.location or ""
                default_athlet = meta.athlete_code or ""
                default_pos = pos_label_map.get(meta.position, "Bauch")
            except Exception:
                default_date = datetime.date.today()
                default_ort = ""
                default_athlet = ""
                default_pos = "Bauch"

            with st.expander(f"📄 {base}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                datum = c1.date_input("Datum", value=default_date, key=f"datum_{base}")
                ort = c2.text_input("Ort", value=default_ort, placeholder="z.B. Laax", key=f"ort_{base}")
                athlet = c3.text_input("Athlet", value=default_athlet, placeholder="z.B. 01", key=f"athlet_{base}")
                pos_idx = pos_options.index(default_pos) if default_pos in pos_options else 0
                position = c4.selectbox("Position", pos_options, index=pos_idx, key=f"pos_{base}")

        if st.button("Hochgeladene Dateien laden", type="primary"):
            loaded = {}
            errors = []
            for base, files in staged.items():
                try:
                    datum = st.session_state.get(f"datum_{base}", datetime.date.today())
                    ort = st.session_state.get(f"ort_{base}", "Unbekannt") or "Unbekannt"
                    athlet = st.session_state.get(f"athlet_{base}", "00") or "00"
                    position = st.session_state.get(f"pos_{base}", "Bauch")

                    pos_map = {"Bauch": "Bauch", "Fuss rechts": "Fuss_re", "Fuss links": "Fuss_li"}
                    pos_key = pos_map.get(position, position)

                    date_str = datum.strftime("%Y%m%d")
                    new_key = f"{date_str}_{ort}_{athlet}_{pos_key}"

                    meta = SensorMeta(
                        filename=f"{new_key}_imuData.csv",
                        date=date_str,
                        location=ort,
                        sensor_id=athlet,
                        athlete_code=athlet,
                        position=pos_key,
                        position_label=position,
                    )

                    imu_df = gnss_df = None
                    if "imu_bytes" in files:
                        imu_df = pd.read_csv(io.BytesIO(files["imu_bytes"]))
                    if "gnss_bytes" in files:
                        gnss_df = pd.read_csv(io.BytesIO(files["gnss_bytes"]))

                    if imu_df is not None or gnss_df is not None:
                        loaded[new_key] = {"imu": imu_df, "gnss": gnss_df,
                                           "imu_path": None, "gnss_path": None, "meta": meta}
                except Exception as e:
                    errors.append(f"{base}: {e}")

            if loaded:
                existing = st.session_state.get("loaded_sessions", {})
                existing.update(loaded)
                st.session_state["loaded_sessions"] = existing
                st.session_state["_upload_staged"] = {}
                for err in errors:
                    st.warning(f"Fehler: {err}")
                st.rerun()
            else:
                for err in errors:
                    st.warning(f"Fehler: {err}")
                if not errors:
                    st.warning("Keine Dateien geladen — prüfe die Dateinamen.")

    # ── Übersicht geladener Daten ──────────────────────────────────────────
    sessions = st.session_state.get("loaded_sessions", {})
    if not sessions:
        st.info("Noch keine Daten geladen.")
        return

    st.divider()
    st.subheader(f"Geladene Dateien ({len(sessions)})")

    group_by = st.radio("Gruppieren nach", ["Athlet", "Testtag", "Ort"], horizontal=True)

    rows = []
    for key, s in sessions.items():
        m = s["meta"]
        rows.append({
            "key": key,
            "Athlet": m.athlete_code,
            "Datum": m.date,
            "Ort": m.location,
            "Sensor-ID": m.sensor_id,
            "Position": m.position_label,
            "GNSS": "✓" if s["gnss"] is not None else "—",
        })
    df_overview = pd.DataFrame(rows)

    group_col = {"Athlet": "Athlet", "Testtag": "Datum", "Ort": "Ort"}[group_by]

    if group_by == "Athlet":
        st.caption("Dateien mit gleicher Athleten-Nummer werden zusammengefasst.")

    for grp_val, grp_df in df_overview.groupby(group_col, sort=True):
        n = len(grp_df)
        with st.expander(f"{group_col}: **{grp_val}** — {n} Datei{'en' if n != 1 else ''}", expanded=False):

            header = st.columns([2, 2, 2, 1, 1, 1])
            for h, lbl in zip(header, ["Datum", "Ort", "Position", "GNSS", "Sensor", "Löschen"]):
                h.markdown(f"**{lbl}**")

            for _, row in grp_df.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 1, 1, 1])
                c1.write(row["Datum"])
                c2.write(row["Ort"])
                c3.write(row["Position"])
                c4.write(row["GNSS"])
                c5.write(row["Sensor-ID"])
                if c6.button("🗑️", key=f"del_{row['key']}", help="Datei entfernen"):
                    sessions.pop(row["key"], None)
                    st.session_state["loaded_sessions"] = sessions
                    st.rerun()

            st.divider()
            if st.button(f"Alle '{grp_val}' entfernen", key=f"rm_{grp_val}", type="secondary"):
                for k in grp_df["key"].values:
                    sessions.pop(k, None)
                st.session_state["loaded_sessions"] = sessions
                st.rerun()
