"""
Daten laden: Drag & Drop → sofort laden. Kein separater Lade-Button.
"""
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.admos_parser import (parse_filename, load_imu_raw, load_gnss_raw,
                                find_csv_pairs, classify_sensor_file,
                                sensor_file_base, DATA_FOLDER, SensorMeta)

LIFT_ALT_MIN_GAIN    = 50.0   # m Höhengewinn
LIFT_SPEED_MAX       = 15.0   # km/h Ø-Geschwindigkeit
LIFT_MIN_DURATION_S  = 180.0  # s Mindestdauer (Pipe-Runs < 60s → kein False-Positive)


def _strip_lifts_from_imu(imu_df: pd.DataFrame, gnss_df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Liftfahrten aus IMU-Daten anhand GNSS-Zeitstempel.
    Erkennt Liftphasen (Höhengewinn >30m + Ø-Speed <25 km/h) und löscht
    die entsprechenden IMU-Zeilen anhand des Zeitstempels.
    """
    imu_t_col  = "imuTimestamp [us]"
    gnss_t_col = "timestamp [us]"
    alt_col    = "altitude [m]"
    if imu_t_col not in imu_df.columns or gnss_t_col not in gnss_df.columns:
        return imu_df
    if alt_col not in gnss_df.columns:
        return imu_df

    gnss = gnss_df.copy().reset_index(drop=True)
    sN = gnss["speedN [m/s]"] if "speedN [m/s]" in gnss.columns else pd.Series(np.zeros(len(gnss)))
    sE = gnss["speedE [m/s]"] if "speedE [m/s]" in gnss.columns else pd.Series(np.zeros(len(gnss)))
    sD = gnss["speedD [m/s]"] if "speedD [m/s]" in gnss.columns else pd.Series(np.zeros(len(gnss)))
    gnss["speed_kmh"] = np.sqrt(sN**2 + sE**2 + sD**2) * 3.6

    alt  = gnss[alt_col].rolling(5, center=True, min_periods=1).mean().values
    spd  = gnss["speed_kmh"].values
    ts   = gnss[gnss_t_col].values
    n    = len(alt)

    lift_intervals = []
    i = 0
    while i < n - 1:
        if alt[i] < alt[i + 1]:
            start_i = i
            start_alt = alt[i]
            j = i + 1
            while j < n - 1 and alt[j + 1] >= alt[j] - 2.0:
                j += 1
            gain = alt[j] - start_alt
            if gain > LIFT_ALT_MIN_GAIN:
                avg_spd = float(np.mean(spd[start_i:j + 1]))
                duration_s = (ts[j] - ts[start_i]) / 1e6
                if avg_spd < LIFT_SPEED_MAX and duration_s >= LIFT_MIN_DURATION_S:
                    lift_intervals.append((ts[start_i], ts[j]))
            i = j + 1
        else:
            i += 1

    if not lift_intervals:
        return imu_df

    imu_ts = imu_df[imu_t_col].values
    keep   = np.ones(len(imu_ts), dtype=bool)
    for t_start, t_end in lift_intervals:
        keep &= ~((imu_ts >= t_start) & (imu_ts <= t_end))

    n_removed = int((~keep).sum())
    n_total   = len(imu_df)
    if n_removed > 0:
        st.caption(f"Liftfahrten entfernt: {n_removed}/{n_total} IMU-Zeilen ({n_removed/n_total*100:.0f}%)")

    return imu_df[keep].reset_index(drop=True)


POS_LABEL_MAP = {
    "Bauch": "Bauch", "Fuss_re": "Fuss rechts", "Fuss_li": "Fuss links",
    "FussRe": "Fuss rechts", "FussLi": "Fuss links",
    "fuss_re": "Fuss rechts", "fuss_li": "Fuss links",
}
POS_OPTIONS = ["Bauch", "Fuss rechts", "Fuss links"]
POS_KEY_MAP = {"Bauch": "Bauch", "Fuss rechts": "Fuss_re", "Fuss links": "Fuss_li"}


def _meta_from_base(base: str) -> dict:
    """Liest Metadaten aus Dateiname. Gibt Defaults zurück wenn nicht erkennbar."""
    try:
        meta = parse_filename(base + "_imuData.csv")
        try:
            date = datetime.date(int(meta.date[:4]), int(meta.date[4:6]), int(meta.date[6:8]))
        except Exception:
            date = datetime.date.today()
        return {
            "date": date,
            "ort": meta.location or "",
            "athlet": meta.athlete_code or "",
            "pos": POS_LABEL_MAP.get(meta.position, "Bauch"),
        }
    except Exception:
        return {"date": datetime.date.today(), "ort": "", "athlet": "", "pos": "Bauch"}


def _load_staged_entry(base: str, files: dict) -> dict | None:
    """Lädt eine staged Datei in eine Session. Gibt None bei Fehler zurück."""
    datum    = st.session_state.get(f"datum_{base}",  datetime.date.today())
    ort      = st.session_state.get(f"ort_{base}",    "") or "Unbekannt"
    athlet   = st.session_state.get(f"athlet_{base}", "") or "00"
    position = st.session_state.get(f"pos_{base}",    "Bauch")
    pos_key  = POS_KEY_MAP.get(position, position)
    date_str = datum.strftime("%Y%m%d")
    new_key  = f"{date_str}_{ort}_{athlet}_{pos_key}"

    meta = SensorMeta(
        filename=f"{new_key}_imuData.csv",
        date=date_str, location=ort, sensor_id=athlet,
        athlete_code=athlet, position=pos_key, position_label=position,
    )
    imu_df = gnss_df = None
    if "imu_bytes" in files:
        imu_df = pd.read_csv(io.BytesIO(files["imu_bytes"]))
    if "gnss_bytes" in files:
        gnss_df = pd.read_csv(io.BytesIO(files["gnss_bytes"]))
    if imu_df is None and gnss_df is None:
        return None

    return {"key": new_key, "imu": imu_df, "gnss": gnss_df,
            "imu_path": None, "gnss_path": None, "meta": meta}


def show():
    st.header("Daten laden")

    # ── Aus festem Ordner (nur lokal) ────────────────────────────────────
    import os as _oslocal
    _is_local = bool(DATA_FOLDER) and _oslocal.path.exists(DATA_FOLDER)
    if _is_local and st.button("Aus Ordner laden", type="primary"):
        pairs = find_csv_pairs(DATA_FOLDER)
        if not pairs:
            st.warning(f"Keine CSV-Dateien gefunden in: {DATA_FOLDER}")
        else:
            loaded = {}
            prog = st.progress(0)
            for i, p in enumerate(pairs):
                key = p["meta"].filename.replace("_imuData.csv","").replace("_imu.csv","").replace("_IMU.csv","")
                prog.progress(int((i+1)/len(pairs)*100),
                              text=f"Lade {i+1}/{len(pairs)}: {p['meta'].athlete_code} | {p['meta'].position_label}")
                try:
                    imu_df  = load_imu_raw(p["imu_path"])
                    gnss_df = load_gnss_raw(p["gnss_path"]) if p["gnss_path"] else None
                    loaded[key] = {"imu": imu_df, "gnss": gnss_df,
                                   "imu_path": p["imu_path"], "gnss_path": p["gnss_path"],
                                   "meta": p["meta"]}
                except Exception as e:
                    st.warning(f"Fehler bei {p['meta'].filename}: {e}")
            prog.empty()
            existing = st.session_state.get("loaded_sessions", {})
            existing.update(loaded)
            st.session_state["loaded_sessions"] = existing
            st.success(f"{len(loaded)} Sensor-Dateien geladen.")

    # ── Protokoll laden (jumps.xlsx) ─────────────────────────────────────
    st.subheader("Protokoll laden")
    st.caption("jumps.xlsx mit Datum, Athlet ID, Run-Nummern, Tricks und Landungsarten.")
    proto_uploaded = st.file_uploader(
        "jumps.xlsx hochladen", type=["xlsx"],
        key="proto_upload",
        label_visibility="collapsed",
    )
    if proto_uploaded:
        try:
            df_proto = pd.read_excel(proto_uploaded)
            st.session_state["protocol_df"] = df_proto
            st.success(f"Protokoll geladen: {len(df_proto)} Einträge, "
                       f"{df_proto['Athlet ID'].nunique()} Athleten")
        except Exception as e:
            st.error(f"Fehler beim Laden: {e}")

    if st.session_state.get("protocol_df") is not None:
        df_p = st.session_state["protocol_df"]
        ids = sorted([x for x in df_p["Athlet ID"].dropna().unique()])
        st.caption(f"Protokoll aktiv: {len(df_p)} Zeilen | Athlet IDs: {ids}")
        if st.button("Protokoll entfernen", key="rm_proto"):
            del st.session_state["protocol_df"]
            st.rerun()

    st.divider()

    # ── Upload: Dateien wählen → sofort laden ─────────────────────────────
    st.subheader("Dateien hochladen")
    st.caption("IMU- und GNSS-Dateien wählen — werden sofort geladen. Mehrere Dateien gleichzeitig möglich.")

    uploaded = st.file_uploader(
        "CSV", type=["csv"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        staged: dict = st.session_state.get("_upload_staged", {})
        newly_added = []
        for f in uploaded:
            kind = classify_sensor_file(f.name)
            if kind is None:
                st.warning(f"Nicht erkannt (*_IMU.csv / *_GNSS.csv erwartet): **{f.name}**")
                continue
            base = sensor_file_base(f.name)
            bkey = f"{kind}_bytes"
            is_new = base not in staged or bkey not in staged[base]
            if is_new:
                f.seek(0)
                staged.setdefault(base, {})[bkey] = f.read()
                newly_added.append(base)
            # Metadaten immer aus Dateiname setzen — bei jedem Render aktuell halten
            m = _meta_from_base(base)
            st.session_state[f"datum_{base}"]  = m["date"]
            st.session_state[f"ort_{base}"]    = m["ort"]
            st.session_state[f"athlet_{base}"] = m["athlet"]
            st.session_state[f"pos_{base}"]    = m["pos"]
        st.session_state["_upload_staged"] = staged

        # Alle staged Einträge anzeigen — kompakt, editierbar
        staged = st.session_state.get("_upload_staged", {})
        if staged:
            st.markdown(f"**{len(staged)} Sensor(en) erkannt** — Metadaten prüfen, dann laden:")

            for base in list(staged.keys()):
                has_imu  = "imu_bytes"  in staged[base]
                has_gnss = "gnss_bytes" in staged[base]
                badges   = ("IMU ✓" if has_imu else "IMU —") + "  |  " + ("GNSS ✓" if has_gnss else "GNSS —")

                with st.expander(f"📄 {base}   —   {badges}", expanded=True):
                    if has_imu:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.date_input("Datum",   key=f"datum_{base}")
                        c2.text_input("Ort",     key=f"ort_{base}",   placeholder="z.B. Laax")
                        c3.text_input("Athlet",  key=f"athlet_{base}", placeholder="z.B. 01")
                        pos_idx = POS_OPTIONS.index(st.session_state.get(f"pos_{base}", "Bauch")) \
                                  if st.session_state.get(f"pos_{base}", "Bauch") in POS_OPTIONS else 0
                        c4.selectbox("Position", POS_OPTIONS, index=pos_idx, key=f"pos_{base}")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.date_input("Datum",  key=f"datum_{base}")
                        c2.text_input("Ort",    key=f"ort_{base}",    placeholder="z.B. Laax")
                        c3.text_input("Athlet", key=f"athlet_{base}", placeholder="z.B. 01")
                        st.session_state[f"pos_{base}"] = "—"

            if st.button("✅ Alle laden", type="primary"):
                loaded = {}
                errors = []
                for base, files in staged.items():
                    try:
                        entry = _load_staged_entry(base, files)
                        if entry:
                            loaded[entry.pop("key")] = entry
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

    rows = []
    for key, s in sessions.items():
        m = s["meta"]
        rows.append({
            "key": key, "Athlet": m.athlete_code, "Datum": m.date,
            "Ort": m.location, "Sensor-ID": m.sensor_id,
            "Position": m.position_label,
            "GNSS": "✓" if s["gnss"] is not None else "—",
            "Athlet_Ort": f"{m.athlete_code} | {m.location}",
        })
    df_overview = pd.DataFrame(rows)

    # Athleten-Filter — Athlet + Ort kombiniert (Athlet 01 Schilthorn ≠ Athlet 01 Corvatsch)
    # Sortierung nach Datum, dann Athlet
    athlet_datum = (
        df_overview.groupby("Athlet_Ort")["Datum"].min().reset_index()
        .sort_values(["Datum", "Athlet_Ort"], ascending=[False, True])
    )
    def _fmt_athlet(row):
        try:
            d = pd.to_datetime(str(row["Datum"]), format="%Y%m%d").strftime("%d.%m.%Y")
        except Exception:
            d = str(row["Datum"])
        parts = row["Athlet_Ort"].split(" | ")
        ort = parts[1] if len(parts) > 1 else ""
        athlet = parts[0]
        return f"{athlet} | {d} | {ort}"
    athlet_datum["label"] = athlet_datum.apply(_fmt_athlet, axis=1)
    label_to_key = dict(zip(athlet_datum["label"], athlet_datum["Athlet_Ort"]))
    key_to_label = dict(zip(athlet_datum["Athlet_Ort"], athlet_datum["label"]))
    all_athletes_labels = athlet_datum["label"].tolist()
    all_athletes = athlet_datum["Athlet_Ort"].tolist()
    filter_key = "import_athlete_filter"
    stored = st.session_state.get(filter_key, [])
    if not stored or not all(a in all_athletes_labels for a in stored):
        st.session_state[filter_key] = all_athletes_labels
    sel_labels = st.multiselect(
        "Athleten auswählen", all_athletes_labels, key=filter_key,
    )
    if not sel_labels:
        sel_labels = all_athletes_labels
    sel_athletes = [label_to_key[l] for l in sel_labels]
    df_overview = df_overview[df_overview["Athlet_Ort"].isin(sel_athletes)]

    n_total = len(sessions)
    n_shown = len(df_overview)
    st.subheader(f"Geladene Dateien ({n_shown} von {n_total})")

    group_by  = st.radio("Gruppieren nach", ["Athlet", "Testtag", "Ort", "Position", "Position & Ort"], horizontal=True)
    # Athlet+Ort kombiniert damit 01 Schilthorn ≠ 01 Corvatsch
    group_col = {
        "Athlet":          "Athlet_Ort",
        "Testtag":         "Datum",
        "Ort":             "Ort",
        "Position":        "Position",
        "Position & Ort":  "Pos_Ort",
    }[group_by]
    df_overview["Pos_Ort"] = df_overview["Position"] + " | " + df_overview["Ort"]

    for grp_val, grp_df in df_overview.groupby(group_col, sort=True):
        n = len(grp_df)
        with st.expander(f"{group_col}: **{grp_val}** — {n} Datei{'en' if n != 1 else ''}", expanded=False):
            header = st.columns([2, 2, 2, 1, 1, 1])
            for h, lbl in zip(header, ["Datum", "Ort", "Position", "GNSS", "Sensor", "Löschen"]):
                h.markdown(f"**{lbl}**")
            for _, row in grp_df.iterrows():
                c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 1, 1, 1])
                c1.write(row["Datum"]); c2.write(row["Ort"]); c3.write(row["Position"])
                c4.write(row["GNSS"]); c5.write(row["Sensor-ID"])
                if c6.button("🗑️", key=f"del_{row['key']}", help="Entfernen"):
                    sessions.pop(row["key"], None)
                    st.session_state["loaded_sessions"] = sessions
                    st.rerun()
            st.divider()
            if st.button(f"Alle '{grp_val}' entfernen", key=f"rm_{grp_val}", type="secondary"):
                for k in grp_df["key"].values:
                    sessions.pop(k, None)
                st.session_state["loaded_sessions"] = sessions
                st.rerun()
