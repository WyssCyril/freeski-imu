"""
Gespeicherte GPS-Daten ansehen (ohne Hochladen): eine Karte pro Messtag,
GPS-Tracks der gewählten Athleten, Sprungpositionen, Höhenprofil und Geschwindigkeit.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils import gsheets

_COLORS = ["#E3001B", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f", "#bcbd22", "#e377c2"]


@st.cache_data(ttl=120, show_spinner="Lade gespeicherte GPS-Daten…")
def _load():
    res = gsheets.load()
    try:
        cur = gsheets.load_curves()
    except Exception:
        cur = pd.DataFrame()
    try:
        gnss = gsheets.load_gnss()
    except Exception:
        gnss = pd.DataFrame()
    return res, cur, gnss


def clear_cache():
    _load.clear()


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().upper() not in ("FALSE", "0", "NEIN")
    return True if v is None or v != v else bool(v)


def show(mode: str = "map"):
    st.info("Keine GPS-Dateien hochgeladen – hier siehst du die **gespeicherten GPS-Daten** aus dem Google Sheet. "
            "Jeder Messtag hat seine eigene Karte.")
    if not gsheets.enabled():
        st.warning("Google Sheet nicht verbunden.")
        return
    try:
        res, cur, gnss = _load()
    except Exception as e:
        st.error(f"Google Sheet nicht erreichbar: {e}")
        return
    if gnss.empty:
        st.info("Noch keine GPS-Tracks gespeichert. Sie werden beim nächsten Speichern einer Analyse mit GPS-Datei ergänzt.")
        return

    gnss = gnss.copy()
    gnss["_tag"] = gnss["Datum"].astype(str) + " · " + gnss["Ort"].astype(str)
    tage = sorted(gnss["_tag"].unique(), key=lambda t: pd.to_datetime(t.split(" · ")[0], format="%d.%m.%Y", errors="coerce"))
    c1, c2 = st.columns([1, 2])
    tag = c1.selectbox("Messtag (Karte)", tage, index=len(tage) - 1, key=f"sg_tag_{mode}")
    d, o = tag.split(" · ", 1)
    g_day = gnss[gnss["_tag"] == tag]
    aths = sorted(g_day["Athlet"].unique())
    sel = c2.multiselect("Athleten", aths, default=aths, key=f"sg_ath_{mode}_{tag}")
    g_day = g_day[g_day["Athlet"].isin(sel)]
    if g_day.empty:
        st.info("Mindestens einen Athleten wählen.")
        return
    for c in ["Lat", "Lon", "Höhe (m)", "Speed (km/h)", "Zeit (POSIX ms)"]:
        g_day[c] = pd.to_numeric(g_day[c], errors="coerce")
    g_day["_t"] = pd.to_datetime(g_day["Zeit (POSIX ms)"], unit="ms", utc=True).dt.tz_convert("Europe/Zurich")
    color = {a: _COLORS[i % len(_COLORS)] for i, a in enumerate(aths)}

    # Sprungpositionen (aus Blatt «Kurven») mit Kennwerten (Blatt «Sprünge»)
    jumps = pd.DataFrame()
    if not cur.empty and "Lat Landung" in cur.columns:
        jc = cur[(cur["Datum"].astype(str) == d) & (cur["Ort"].astype(str) == o) & (cur["Athlet"].isin(sel))].copy()
        if not res.empty:
            keys = ["Athlet", "Datum", "Ort", "Position", "Run", "Sprung"]
            r = res.copy()
            for k in keys:
                r[k] = r[k].astype(str); jc[k] = jc[k].astype(str)
            cols = [c for c in ["Peak roh (g)", "Flugzeit (s)", "Landeimpuls 0.25 s (g·s)", "Landungsart", "In Auswertung"] if c in r.columns]
            jc = jc.merge(r[keys + cols], on=keys, how="left")
        for c in ["Lat Landung", "Lon Landung", "Lat Absprung", "Lon Absprung", "Höhe Landung (m)", "Zeit Landung (POSIX ms)", "Peak roh (g)", "Flugzeit (s)"]:
            if c in jc.columns:
                jc[c] = pd.to_numeric(jc[c], errors="coerce")
        jc["_ok"] = jc["In Auswertung"].map(_as_bool) if "In Auswertung" in jc.columns else True
        jumps = jc.dropna(subset=["Lat Landung", "Lon Landung"])
    only_ok = st.checkbox("Nur Sprünge in der Auswertung", value=True, key=f"sg_ok_{mode}")
    jp = jumps[jumps["_ok"]] if (only_ok and not jumps.empty) else jumps

    m1, m2, m3 = st.columns(3)
    m1.metric("Athleten", len(sel)); m2.metric("GPS-Punkte", f"{len(g_day):,}".replace(",", "'"))
    m3.metric("Sprünge auf der Karte", len(jp))

    # ── Karte ────────────────────────────────────────────────────────────
    fig = go.Figure()
    for a in sel:
        ga = g_day[g_day["Athlet"] == a].sort_values("Zeit (POSIX ms)")
        fig.add_trace(go.Scattermap(lat=ga["Lat"], lon=ga["Lon"], mode="lines", line=dict(width=2, color=color[a]),
                                    name=f"A{a}", hovertext=[f"A{a} · {t:%H:%M:%S} · {v:.0f} km/h" for t, v in zip(ga["_t"], ga["Speed (km/h)"].fillna(0))],
                                    hoverinfo="text"))
    if not jp.empty and mode == "map":
        size = (pd.to_numeric(jp.get("Peak roh (g)"), errors="coerce").fillna(8) / 25 * 18 + 7).clip(7, 25)
        fig.add_trace(go.Scattermap(
            lat=jp["Lat Landung"], lon=jp["Lon Landung"], mode="markers", name="Landungen",
            marker=dict(size=size, color=["#000000" if ok else "#BDBDBD" for ok in jp["_ok"]], opacity=0.8),
            customdata=np.stack([jp["Athlet"], jp["Run"], jp["Sprung"], pd.to_numeric(jp.get("Peak roh (g)"), errors="coerce").round(1),
                                 pd.to_numeric(jp.get("Flugzeit (s)"), errors="coerce").round(2), jp.get("Landungsart", "").fillna("")], axis=1),
            hovertemplate="A%{customdata[0]} · %{customdata[1]} · %{customdata[2]}<br>Peak %{customdata[3]} g · Flug %{customdata[4]} s · %{customdata[5]}<extra></extra>"))
    fig.update_layout(map=dict(style="open-street-map", zoom=14,
                               center=dict(lat=float(g_day["Lat"].median()), lon=float(g_day["Lon"].median()))),
                      height=560, margin=dict(l=0, r=0, t=0, b=0), legend=dict(bgcolor="rgba(255,255,255,0.8)", x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True}, key=f"sg_map_{mode}")
    if mode == "map":
        st.caption("Linien = GPS-Tracks (1 Punkt alle 2 s). Punkte = Landungen (Grösse = Peak); grau = nicht in der Auswertung.")

    # ── Höhenprofil + Geschwindigkeit ────────────────────────────────────
    fh = go.Figure(); fs = go.Figure()
    for a in sel:
        ga = g_day[g_day["Athlet"] == a].sort_values("Zeit (POSIX ms)")
        fh.add_trace(go.Scatter(x=ga["_t"], y=ga["Höhe (m)"], mode="lines", line=dict(color=color[a], width=1.3), name=f"A{a}"))
        fs.add_trace(go.Scatter(x=ga["_t"], y=ga["Speed (km/h)"], mode="lines", line=dict(color=color[a], width=1), name=f"A{a}"))
    if not jp.empty and "Zeit Landung (POSIX ms)" in jp.columns and mode == "map":
        tj = pd.to_datetime(jp["Zeit Landung (POSIX ms)"], unit="ms", utc=True).dt.tz_convert("Europe/Zurich")
        fh.add_trace(go.Scatter(x=tj, y=jp["Höhe Landung (m)"], mode="markers", name="Landungen",
                                marker=dict(size=7, color="#000000", symbol="triangle-down")))
    fh.update_layout(template="plotly_white", height=320, yaxis_title="Höhe (m ü. M.)", margin=dict(l=10, r=10, t=30, b=10), title="Höhenprofil")
    fs.update_layout(template="plotly_white", height=280, yaxis_title="Geschwindigkeit (km/h)", margin=dict(l=10, r=10, t=30, b=10), title="Geschwindigkeit")
    st.plotly_chart(fh, use_container_width=True, key=f"sg_alt_{mode}")
    st.plotly_chart(fs, use_container_width=True, key=f"sg_spd_{mode}")
