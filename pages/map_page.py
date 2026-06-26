"""
Kombinierter GPS + Sprung-Tab:
- Liftfahrten (>80m Aufstieg) werden ausgeblendet
- GPS-Track auf Karte mit Sprungmarkierungen
- Höhenprofil mit Absprung/Landung pro Sprung
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.admos_parser import load_gnss_raw

LAT_MIN, LAT_MAX = 45.5, 48.5
LON_MIN, LON_MAX = 5.5, 11.0
ALT_MAX = 5000
LIFT_ALT_MIN_GAIN  = 30.0   # Mindest-Höhengewinn für Lift-Kandidat (m)
LIFT_SPEED_MAX     = 25.0   # Max. Durchschnittsgeschwindigkeit auf Lift (km/h)

ATHLETE_COLORS = [
    (31,  119, 180),
    (255, 127,  14),
    (44,  160,  44),
    (214,  39,  40),
    (148, 103, 189),
]


def _load_gnss(sess: dict) -> pd.DataFrame | None:
    gnss = sess.get("gnss")
    if gnss is None:
        path = sess.get("gnss_path")
        if path and os.path.exists(path):
            gnss = pd.read_csv(path)
        else:
            return None
    df = gnss.copy()
    required = ["latitude [deg]", "longitude [deg]"]
    if not all(c in df.columns for c in required):
        return None
    df = df[
        (df["latitude [deg]"] > LAT_MIN) & (df["latitude [deg]"] < LAT_MAX) &
        (df["longitude [deg]"] > LON_MIN) & (df["longitude [deg]"] < LON_MAX)
    ]
    if "altitude [m]" in df.columns:
        df = df[(df["altitude [m]"] > 0) & (df["altitude [m]"] < ALT_MAX)]
    if "speedN [m/s]" in df.columns and "speedE [m/s]" in df.columns:
        df["speed_kmh"] = np.sqrt(
            df["speedN [m/s]"]**2 + df["speedE [m/s]"]**2 +
            df.get("speedD [m/s]", pd.Series(np.zeros(len(df))))**2
        ) * 3.6
        df = df[df["speed_kmh"] <= 90]  # GPS-Fehler entfernen
    return df.reset_index(drop=True) if len(df) > 10 else None


def _remove_lifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erkennt Liftfahrten automatisch aus Höhe + Geschwindigkeit:
    - Aufstiegs-Segment wird gefunden (Höhe steigt kontinuierlich)
    - Gilt als Lift wenn: Höhengewinn > LIFT_ALT_MIN_GAIN UND Ø-Geschwindigkeit < LIFT_SPEED_MAX
    - Ohne Geschwindigkeitsdaten: Fallback auf reinen Höhenanstieg > 80m
    """
    if "altitude [m]" not in df.columns:
        return df

    # Höhe glätten um GPS-Rauschen zu reduzieren
    alt = pd.Series(df["altitude [m]"].values).rolling(5, center=True, min_periods=1).mean().values
    has_speed = "speed_kmh" in df.columns
    spd = df["speed_kmh"].values if has_speed else None
    n = len(alt)
    keep = np.ones(n, dtype=bool)

    i = 0
    while i < n - 1:
        if alt[i] < alt[i + 1]:
            start_i = i
            start_alt = alt[i]
            j = i + 1
            # Aufstieg verfolgen (kleine Rückgänge <2m erlauben für GPS-Rauschen)
            while j < n - 1:
                if alt[j + 1] >= alt[j] - 2.0:
                    j += 1
                else:
                    break
            end_alt = alt[j]
            gain = end_alt - start_alt

            is_lift = False
            if has_speed and gain > LIFT_ALT_MIN_GAIN:
                avg_spd = float(np.mean(spd[start_i:j + 1]))
                is_lift = avg_spd < LIFT_SPEED_MAX
            elif not has_speed and gain > 80.0:
                is_lift = True

            if is_lift:
                keep[start_i:j + 1] = False
            i = j + 1
        else:
            i += 1

    return df[keep].reset_index(drop=True)


def _time_axis(df: pd.DataFrame) -> pd.Series:
    if "time [POSIXms]" in df.columns:
        return (df["time [POSIXms]"] - df["time [POSIXms]"].iloc[0]) / 1000
    return pd.Series(np.arange(len(df)) / 1.0)


def _downsample(df: pd.DataFrame, max_pts: int = 3000) -> pd.DataFrame:
    if len(df) <= max_pts:
        return df
    step = len(df) // max_pts
    return df.iloc[::step].reset_index(drop=True)


def _get_jump_gnss_positions(df_gnss: pd.DataFrame, df_imu: pd.DataFrame,
                              jumps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Versucht Sprung-Zeitstempel (IMU) auf GPS-Koordinaten abzubilden.
    Funktioniert wenn beide Datensätze POSIX-Zeitstempel haben.
    """
    if jumps_df is None or jumps_df.empty:
        return pd.DataFrame()
    if "time [POSIXms]" not in df_gnss.columns:
        return pd.DataFrame()
    if "imuTimestamp [us]" not in df_imu.columns:
        return pd.DataFrame()

    imu_t = df_imu["imuTimestamp [us]"].values
    gps_t = df_gnss["time [POSIXms]"].values * 1000  # ms → µs

    rows = []
    for _, jump in jumps_df.iterrows():
        to_idx  = int(jump["takeoff_idx"])
        la_idx  = int(jump["landing_idx"])
        pk_idx  = int(jump["peak_res_idx"])

        if to_idx >= len(imu_t) or la_idx >= len(imu_t):
            continue

        to_us  = imu_t[to_idx]
        la_us  = imu_t[la_idx]
        pk_us  = imu_t[pk_idx]

        # Nächsten GPS-Punkt zum Peak-Zeitpunkt finden
        gps_pk_idx = int(np.argmin(np.abs(gps_t - pk_us)))
        lat = float(df_gnss["latitude [deg]"].iloc[gps_pk_idx])
        lon = float(df_gnss["longitude [deg]"].iloc[gps_pk_idx])

        if not (LAT_MIN < lat < LAT_MAX and LON_MIN < lon < LON_MAX):
            continue

        rows.append({
            "jump_id":    jump["jump_id"],
            "peak_res_g": float(jump["peak_res_g"]),
            "flight_s":   float(jump["flight_time_s"]),
            "lat":        lat,
            "lon":        lon,
        })
    return pd.DataFrame(rows)


def show():
    st.header("GPS-Track & Sprunganalyse")

    sessions_loaded = st.session_state.get("loaded_sessions", {})
    jump_results    = st.session_state.get("jump_results", {})

    gnss_sessions = {k: v for k, v in sessions_loaded.items()
                     if v.get("gnss") is not None or v.get("gnss_path")}

    if not gnss_sessions:
        st.info("Keine GNSS-Daten geladen. Zuerst Daten mit GPS-Datei im Tab 'Daten laden' importieren.")
        return

    # ── Auswahl ──────────────────────────────────────────────────────────
    orte = sorted(set(s["meta"].location for s in gnss_sessions.values() if s.get("meta")))
    sel_ort = st.selectbox("Ort", ["Alle"] + orte, key="map_sel_ort")

    filtered = {k: v for k, v in gnss_sessions.items()
                if sel_ort == "Alle" or (v.get("meta") and v["meta"].location == sel_ort)}

    all_athletes = sorted(set(v["meta"].athlete_code for v in filtered.values() if v.get("meta")))
    sel_athletes = st.multiselect("Athleten", all_athletes, default=all_athletes)

    ath_color = {a: ATHLETE_COLORS[i % len(ATHLETE_COLORS)]
                 for i, a in enumerate(sorted(set(v["meta"].athlete_code
                                                   for v in filtered.values() if v.get("meta"))))}

    # ── Daten vorbereiten ─────────────────────────────────────────────────
    track_data = []
    for key, sess in filtered.items():
        meta = sess.get("meta")
        if not meta or meta.athlete_code not in sel_athletes:
            continue
        df_gnss = _load_gnss(sess)
        if df_gnss is None:
            continue
        df_gnss = _remove_lifts(df_gnss)
        if len(df_gnss) < 10:
            continue

        # Sprungdaten zu diesem Sensor suchen
        jumps_df = pd.DataFrame()
        df_imu   = None
        for jkey, jr in jump_results.items():
            if key in jkey:
                jumps_df = jr.get("jumps", pd.DataFrame())
                break

        # IMU-DataFrame für Zeitstempel-Mapping
        imu_raw = sess.get("imu")
        if imu_raw is not None:
            df_imu = imu_raw

        jump_positions = pd.DataFrame()
        if df_imu is not None and not jumps_df.empty:
            jump_positions = _get_jump_gnss_positions(df_gnss, df_imu, jumps_df)

        track_data.append({
            "key":      key,
            "meta":     meta,
            "gnss":     df_gnss,
            "jumps_df": jumps_df,
            "jump_pos": jump_positions,
            "rgb":      ath_color[meta.athlete_code],
        })

    if not track_data:
        st.info("Keine gültigen GPS-Daten für die Auswahl.")
        return

    # ── Karte ─────────────────────────────────────────────────────────────
    st.subheader("GPS-Track")
    st.caption("Liftfahrten automatisch erkannt und ausgeblendet (langsamer Aufstieg). Sprünge als rote Punkte markiert (Grösse = Peak-g).")
    map_traces = []

    for td in track_data:
        df_d = _downsample(td["gnss"], 3000)
        rgb  = td["rgb"]
        meta = td["meta"]

        # Speed-Farbe
        if "speed_kmh" in df_d.columns:
            spd_max  = max(df_d["speed_kmh"].max(), 1.0)
            spd_norm = (df_d["speed_kmh"] / spd_max).clip(0, 1)
            marker_colors = [
                f"rgb({int(rgb[0]*(0.3+0.7*s))},{int(rgb[1]*(0.3+0.7*s))},{int(rgb[2]*(0.3+0.7*s))})"
                for s in spd_norm
            ]
            hover = [f"{meta.athlete_code} | {v:.1f} km/h" for v in df_d["speed_kmh"]]
        else:
            marker_colors = [f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"] * len(df_d)
            hover = [meta.athlete_code] * len(df_d)

        map_traces.append(go.Scattermapbox(
            lat=df_d["latitude [deg]"].tolist(),
            lon=df_d["longitude [deg]"].tolist(),
            mode="lines+markers",
            line=dict(width=2, color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"),
            marker=dict(size=4, color=marker_colors, opacity=0.85),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name=f"{meta.athlete_code} | {meta.date}",
        ))

        # Sprungpositionen einzeichnen
        jp = td["jump_pos"]
        if not jp.empty:
            sizes = (jp["peak_res_g"] / jp["peak_res_g"].max() * 20 + 8).clip(8, 28).tolist()
            map_traces.append(go.Scattermapbox(
                lat=jp["lat"].tolist(),
                lon=jp["lon"].tolist(),
                mode="markers+text",
                marker=dict(size=sizes, color="red", opacity=0.85),
                text=jp["jump_id"].tolist(),
                textposition="top right",
                hovertemplate="<b>%{text}</b><br>" +
                    jp.apply(lambda r: f"{r['peak_res_g']:.1f}g | {r['flight_s']:.2f}s", axis=1).tolist()[0]
                    if len(jp) == 1 else
                    "<b>%{text}</b><extra></extra>",
                name=f"Sprünge {meta.athlete_code}",
                showlegend=True,
            ))

    all_lats = [lat for t in map_traces for lat in (t.lat or [])]
    all_lons = [lon for t in map_traces for lon in (t.lon or [])]
    fig_map = go.Figure(map_traces)
    fig_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.median(all_lats)), lon=float(np.median(all_lons))),
            zoom=13,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550,
        legend=dict(bgcolor="rgba(255,255,255,0.8)", x=0.01, y=0.99),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})

    # ── Höhenprofil + Sprünge ──────────────────────────────────────────────
    has_alt = any("altitude [m]" in td["gnss"].columns for td in track_data)
    if has_alt:
        st.subheader("Höhenprofil mit Sprüngen")
        fig_alt = go.Figure()

        for td in track_data:
            df_g = td["gnss"]
            if "altitude [m]" not in df_g.columns:
                continue
            df_d  = _downsample(df_g, 3000)
            t     = _time_axis(df_d)
            rgb   = td["rgb"]
            meta  = td["meta"]

            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            fig_alt.add_trace(go.Scatter(
                x=t, y=df_d["altitude [m]"].tolist(),
                mode="lines",
                line=dict(color=f"rgb({r},{g},{b})", width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba({r},{g},{b},0.08)",
                name=f"{meta.athlete_code} | {meta.date}",
                hovertemplate="%{y:.0f} m<extra>" + str(meta.athlete_code) + "</extra>",
            ))

            pass  # Sprung-Zeitstempel-Mapping im Höhenprofil folgt später

        fig_alt.update_layout(
            xaxis_title="Zeit (s)", yaxis_title="Höhe (m ü. M.)",
            template="plotly_white", height=300,
            legend=dict(orientation="h", y=-0.35),
        )
        st.plotly_chart(fig_alt, use_container_width=True, config={"scrollZoom": True})

    # ── Sprungübersicht pro Athlet ─────────────────────────────────────────
    jump_rows = []
    for td in track_data:
        jdf = td["jumps_df"]
        if jdf.empty:
            continue
        meta = td["meta"]
        for _, row in jdf.iterrows():
            rot = row.get("rotations")
            rot_dir = row.get("rotation_dir", "")
            jump_rows.append({
                "Athlet":       meta.athlete_code,
                "Sprung":       row["jump_id"],
                "Flugzeit (s)": f"{row['flight_time_s']:.3f}",
                "Peak (g)":     f"{'⚠️ ' if row.get('clipped_16g') else ''}{row['peak_res_g']:.2f}",
                "Rotation":     f"{rot}x {rot_dir}" if rot is not None else "—",
                "Landungsart":  row.get("landing_type", ""),
            })

    if jump_rows:
        st.subheader("Sprungübersicht")
        st.dataframe(pd.DataFrame(jump_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Noch keine Sprunganalyse verfügbar — zuerst Tab 'Sprunganalyse' öffnen.")
