"""
GNSS-Analyse: GPS-Track auf Karte, Speed-Verlauf, nach Ort gruppiert.
Jeder Athlet hat eine eigene Farbe, Linie wird dunkler bei höherer Geschwindigkeit.
"""
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.admos_parser import classify_sensor_file, sensor_file_base, parse_filename, load_gnss_raw

# Athleten-Farben (Hue-Basis, wird nach Helligkeit variiert)
ATHLETE_COLORS_BASE = [
    (31,  119, 180),   # Blau
    (255, 127,  14),   # Orange
    (44,  160,  44),   # Grün
    (214,  39,  40),   # Rot
    (148, 103, 189),   # Lila
    (140,  86,  75),   # Braun
    (227, 119, 194),   # Pink
    (188, 189,  34),   # Gelbgrün
    ( 23, 190, 207),   # Türkis
    (127, 127, 127),   # Grau
]

# GPS-Begrenzung Schweiz + Nachbarländer
LAT_MIN, LAT_MAX = 45.5, 48.5
LON_MIN, LON_MAX = 5.5, 11.0
ALT_MAX = 5000


def _speed_color(r: int, g: int, b: int, speed_norm: float) -> str:
    """
    Gibt eine rgba-Farbe zurück:
    - Basisfarbe bei speed_norm=0 (hell/transparent)
    - Dunkel + opak bei speed_norm=1
    """
    factor = 0.25 + 0.75 * speed_norm   # 0.25 (hell) → 1.0 (voll)
    alpha  = 0.3  + 0.7  * speed_norm   # 0.3 (transparent) → 1.0 (opak)
    r2 = int(r * factor)
    g2 = int(g * factor)
    b2 = int(b * factor)
    return f"rgba({r2},{g2},{b2},{alpha:.2f})"


def _load_gnss(sess: dict) -> pd.DataFrame | None:
    """Lädt und bereinigt GNSS-Daten aus einer Session."""
    gnss = sess.get("gnss")
    if gnss is None:
        path = sess.get("gnss_path")
        if path and os.path.exists(path):
            gnss = pd.read_csv(path)
        else:
            return None

    df = gnss.copy()
    required = ["latitude [deg]", "longitude [deg]", "speedN [m/s]", "speedE [m/s]", "speedD [m/s]"]
    if not all(c in df.columns for c in required):
        return None

    df["speed_kmh"] = np.sqrt(
        df["speedN [m/s]"]**2 + df["speedE [m/s]"]**2 + df["speedD [m/s]"]**2
    ) * 3.6

    # GPS-Fehler entfernen (Athlet nie schneller als 90 km/h)
    df = df[df["speed_kmh"] <= 90]

    # Ungültige GPS-Punkte entfernen
    df = df[
        (df["latitude [deg]"]  > LAT_MIN) & (df["latitude [deg]"]  < LAT_MAX) &
        (df["longitude [deg]"] > LON_MIN) & (df["longitude [deg]"] < LON_MAX)
    ]
    if "altitude [m]" in df.columns:
        df = df[(df["altitude [m]"] > 0) & (df["altitude [m]"] < ALT_MAX)]

    # Ausreisser entfernen: Punkte > 20km vom Median-Zentrum des Tracks
    df = df.reset_index(drop=True)
    lat_med = df["latitude [deg]"].median()
    lon_med = df["longitude [deg]"].median()
    # Haversine-Distanz in km (vereinfacht)
    dlat = (df["latitude [deg]"] - lat_med) * 111.0
    dlon = (df["longitude [deg]"] - lon_med) * 111.0 * np.cos(np.radians(lat_med))
    dist_km = np.sqrt(dlat**2 + dlon**2)
    df = df[dist_km <= 20].reset_index(drop=True)

    # Koordinaten glätten (Medianfilter, Fenster 5)
    df["latitude [deg]"]  = df["latitude [deg]"].rolling(5, center=True, min_periods=1).median()
    df["longitude [deg]"] = df["longitude [deg]"].rolling(5, center=True, min_periods=1).median()
    if "altitude [m]" in df.columns:
        df["altitude [m]"] = df["altitude [m]"].rolling(5, center=True, min_periods=1).median()

    return df.reset_index(drop=True) if len(df) > 10 else None


def _downsample(df: pd.DataFrame, max_pts: int = 2000) -> pd.DataFrame:
    """Dezimiert für schnelles Rendering."""
    if len(df) <= max_pts:
        return df
    step = len(df) // max_pts
    return df.iloc[::step].reset_index(drop=True)


def _load_gnss_upload_section():
    """Upload-Bereich für GNSS-Dateien direkt im GPS-Tab."""
    gnss_extra = st.session_state.get("gnss_only_sessions", {})
    n_loaded = len(gnss_extra)
    label = f"GNSS-Dateien laden ({n_loaded} geladen)" if n_loaded else "GNSS-Dateien laden"

    with st.expander(label, expanded=(n_loaded == 0)):
        st.caption(
            "GNSS-CSV-Dateien hochladen (*_GNSS.csv, *_gnss.csv, *_gnssData.csv). "
            "Mehrere Dateien gleichzeitig möglich."
        )

        uploaded = st.file_uploader(
            "GNSS-CSV", type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="gnss_uploader",
        )

        # Bytes sofort beim Upload lesen und in session_state zwischenspeichern.
        # f.read() muss VOR dem Button-Klick passieren, da der Cursor sonst am Ende steht.
        if uploaded:
            staged = st.session_state.get("_gnss_upload_staged", {})
            new_found = 0
            unrecognized = []
            for f in uploaded:
                kind = classify_sensor_file(f.name)
                if kind != "gnss":
                    unrecognized.append(f.name)
                    continue
                base = sensor_file_base(f.name)
                if base not in staged:
                    f.seek(0)
                    staged[base] = f.read()
                    new_found += 1
            st.session_state["_gnss_upload_staged"] = staged
            if unrecognized:
                st.warning(
                    f"Nicht als GNSS erkannt (erwartet *_GNSS.csv / *_gnss.csv / *_gnssData.csv): "
                    + ", ".join(f"**{n}**" for n in unrecognized)
                )
            if staged:
                st.caption(f"{len(staged)} GNSS-Datei(en) bereit: {', '.join(staged.keys())}")

        staged = st.session_state.get("_gnss_upload_staged", {})
        if staged:
            if st.button("Laden", type="primary", key="btn_load_gnss"):
                errors = []
                loaded_count = 0
                progress = st.progress(0)
                items = list(staged.items())
                for i, (base, raw_bytes) in enumerate(items):
                    progress.progress(int((i + 1) / len(items) * 100))
                    try:
                        meta = parse_filename(base + "_gnss.csv")
                        gnss_df = pd.read_csv(io.BytesIO(raw_bytes))
                        gnss_extra[base] = {
                            "gnss": gnss_df,
                            "imu": None,
                            "imu_path": None,
                            "gnss_path": None,
                            "meta": meta,
                        }
                        loaded_count += 1
                    except Exception as e:
                        errors.append(f"{base}: {e}")
                progress.empty()
                st.session_state["gnss_only_sessions"] = gnss_extra
                st.session_state["_gnss_upload_staged"] = {}
                if loaded_count:
                    st.success(f"{loaded_count} GNSS-Datei(en) geladen.")
                for err in errors:
                    st.warning(f"Fehler: {err}")
                st.rerun()

        # Geladene GNSS-only-Dateien anzeigen
        if gnss_extra:
            st.markdown("**Geladene GNSS-Dateien:**")
            for key, sess in list(gnss_extra.items()):
                m = sess.get("meta")
                label_txt = f"{m.athlete_code} | {m.date} | {m.location}" if m else key
                col_lbl, col_del = st.columns([6, 1])
                col_lbl.write(label_txt)
                if col_del.button("🗑️", key=f"del_gnss_{key}"):
                    gnss_extra.pop(key, None)
                    st.session_state["gnss_only_sessions"] = gnss_extra
                    st.rerun()


def show():
    st.header("GNSS — GPS-Track & Geschwindigkeit")

    # Kombiniere IMU-geladene Sessions (mit GNSS) + direkt geladene GNSS-only Sessions
    imu_sessions  = st.session_state.get("loaded_sessions", {})
    gnss_extra    = st.session_state.get("gnss_only_sessions", {})

    gnss_sessions = {
        k: v for k, v in imu_sessions.items()
        if v.get("gnss") is not None or v.get("gnss_path")
    }
    gnss_sessions.update(gnss_extra)

    if not gnss_sessions:
        st.info("Noch keine GNSS-Daten geladen. Dateien oben hochladen oder im Tab 'Daten laden' laden.")
        return

    # ── Ort wählen ────────────────────────────────────────────────────────
    orte = sorted(set(s["meta"].location for s in gnss_sessions.values() if s.get("meta")))
    sel_ort = st.selectbox("Ort", ["Alle"] + orte, key="gnss_sel_ort")

    filtered = {k: v for k, v in gnss_sessions.items()
                if sel_ort == "Alle" or (v.get("meta") and v["meta"].location == sel_ort)}

    if not filtered:
        st.info("Keine Daten für diesen Ort.")
        return

    # Athleten-Auswahl
    all_athletes = sorted(set(v["meta"].athlete_code for v in filtered.values() if v.get("meta")))
    sel_athletes = st.multiselect("Athleten anzeigen", all_athletes, default=all_athletes)

    # Athleten → Farbe zuweisen
    unique_athletes = sorted(set(v["meta"].athlete_code for v in filtered.values() if v.get("meta")))
    ath_color = {ath: ATHLETE_COLORS_BASE[i % len(ATHLETE_COLORS_BASE)]
                 for i, ath in enumerate(unique_athletes)}

    # ── Karte aufbauen ────────────────────────────────────────────────────
    st.subheader("GPS-Track")
    st.caption("Linienfarbe: pro Athlet unterschiedlich | Dunkler = schneller")

    map_traces = []
    speed_chart_data = []

    # Datenpunkte je Track als farbige Segmente via Scattermapbox
    for key, sess in filtered.items():
        meta = sess.get("meta")
        if not meta or meta.athlete_code not in sel_athletes:
            continue

        df_gnss = _load_gnss(sess)
        if df_gnss is None or len(df_gnss) < 10:
            continue

        df_d = _downsample(df_gnss, max_pts=3000)
        rgb = ath_color[meta.athlete_code]
        speed_max = df_d["speed_kmh"].max() if df_d["speed_kmh"].max() > 0 else 1.0
        speed_norm = (df_d["speed_kmh"] / speed_max).clip(0, 1)

        # Farbliste für jeden Punkt
        colors = [_speed_color(*rgb, float(s)) for s in speed_norm]

        # Scattermapbox-Trace (Linie + Punkte)
        map_traces.append(go.Scattermapbox(
            lat=df_d["latitude [deg]"].tolist(),
            lon=df_d["longitude [deg]"].tolist(),
            mode="lines+markers",
            line=dict(
                width=2,
                color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
            ),
            marker=dict(
                size=4,
                color=[f"rgb({int(rgb[0]*f)},{int(rgb[1]*f)},{int(rgb[2]*f)})"
                       for f in (0.25 + 0.75 * speed_norm).tolist()],
                opacity=0.85,
            ),
            text=[f"{meta.athlete_code} | {v:.1f} km/h" for v in df_d["speed_kmh"]],
            hovertemplate="%{text}<extra></extra>",
            name=f"{meta.athlete_code} | {meta.date}",
            showlegend=True,
        ))

        # Für Speed-Chart
        if "time [POSIXms]" in df_d.columns:
            t = (df_d["time [POSIXms]"] - df_d["time [POSIXms]"].iloc[0]) / 1000
        else:
            t = pd.Series(range(len(df_d)))
        speed_chart_data.append({
            "key": key,
            "meta": meta,
            "t": t,
            "speed": df_d["speed_kmh"],
            "rgb": rgb,
        })

    if not map_traces:
        st.info("Keine gültigen GPS-Daten für die Auswahl.")
        return

    # Mittelpunkt berechnen
    all_lats = [lat for t in map_traces for lat in t.lat]
    all_lons = [lon for t in map_traces for lon in t.lon]
    center_lat = float(np.median(all_lats))
    center_lon = float(np.median(all_lons))

    fig_map = go.Figure(map_traces)
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white"),
            x=0.01, y=0.99,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})

    # ── Geschwindigkeitsverlauf ───────────────────────────────────────────
    st.subheader("Geschwindigkeitsverlauf")
    fig_speed = go.Figure()
    for item in speed_chart_data:
        rgb = item["rgb"]
        fig_speed.add_trace(go.Scatter(
            x=item["t"],
            y=item["speed"],
            mode="lines",
            line=dict(color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", width=1.2),
            name=f"{item['meta'].athlete_code} | {item['meta'].date}",
            hovertemplate="%{y:.1f} km/h<extra>" + item['meta'].athlete_code + "</extra>",
        ))
    fig_speed.update_layout(
        xaxis_title="Zeit (s)",
        yaxis_title="Geschwindigkeit (km/h)",
        template="plotly_white",
        height=300,
        legend=dict(orientation="h", y=-0.3),
    )
    st.plotly_chart(fig_speed, use_container_width=True, config={"scrollZoom": True})

    # ── Höhenprofil ───────────────────────────────────────────────────────
    has_alt = any(
        sess.get("gnss") is not None and "altitude [m]" in sess["gnss"].columns
        for sess in filtered.values()
    )
    if has_alt:
        st.subheader("Höhenprofil")
        fig_alt = go.Figure()
        for key, sess in filtered.items():
            meta = sess.get("meta")
            if not meta or meta.athlete_code not in sel_athletes:
                continue
            df_gnss = _load_gnss(sess)
            if df_gnss is None or "altitude [m]" not in df_gnss.columns:
                continue
            df_d = _downsample(df_gnss, 2000)
            if "time [POSIXms]" in df_d.columns:
                t = (df_d["time [POSIXms]"] - df_d["time [POSIXms]"].iloc[0]) / 1000
            else:
                t = pd.Series(range(len(df_d)))
            rgb = ath_color[meta.athlete_code]
            fig_alt.add_trace(go.Scatter(
                x=t, y=df_d["altitude [m]"],
                mode="lines",
                line=dict(color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})", width=1.2),
                fill="tozeroy", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)",
                name=f"{meta.athlete_code} | {meta.date}",
            ))
        fig_alt.update_layout(
            xaxis_title="Zeit (s)", yaxis_title="Höhe (m ü. M.)",
            template="plotly_white", height=260,
            legend=dict(orientation="h", y=-0.35),
        )
        st.plotly_chart(fig_alt, use_container_width=True, config={"scrollZoom": True})

    # ── Statistik-Tabelle ─────────────────────────────────────────────────
    st.subheader("Statistik pro Athlet")
    stat_rows = []
    for key, sess in filtered.items():
        meta = sess.get("meta")
        if not meta or meta.athlete_code not in sel_athletes:
            continue
        df_gnss = _load_gnss(sess)
        if df_gnss is None:
            continue
        row = {
            "Athlet": meta.athlete_code,
            "Datum": meta.date,
            "Ort": meta.location,
            "Position": meta.position_label,
            "Max Speed (km/h)": round(df_gnss["speed_kmh"].max(), 1),
            "Ø Speed (km/h)": round(df_gnss["speed_kmh"].mean(), 1),
            "GPS-Punkte": len(df_gnss),
        }
        if "altitude [m]" in df_gnss.columns:
            row["Höhendiff. (m)"] = round(df_gnss["altitude [m]"].max() - df_gnss["altitude [m]"].min(), 0)
        stat_rows.append(row)

    if stat_rows:
        st.dataframe(
            pd.DataFrame(stat_rows).sort_values(["Athlet", "Datum"]),
            use_container_width=True, hide_index=True,
        )
