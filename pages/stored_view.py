"""
Gespeicherte Sprünge ansehen: Ergebnisse + Kurvenausschnitte aus dem Google Sheet,
ohne dass Rohdaten hochgeladen werden müssen (nur Ansicht).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils import gsheets

_ART_COLORS = {"vorwärts": "#2ca02c", "switch": "#ff7f0e", "crash": "#d62728"}


@st.cache_data(ttl=120, show_spinner="Lade gespeicherte Sprünge…")
def _load():
    res = gsheets.load()
    try:
        cur = gsheets.load_curves()
    except Exception:
        cur = pd.DataFrame()
    return res, cur


def clear_cache():
    _load.clear()


def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().upper() not in ("FALSE", "0", "NEIN")
    return True if v is None or v != v else bool(v)


def _curve_fig(cur: pd.Series, title: str, color: str, excluded: bool) -> go.Figure:
    vals = np.array([float(x) for x in str(cur["accRes (g)"]).split(",") if x != ""])
    fs = float(cur.get("Abtastrate (Hz)", 200) or 200)
    t = float(cur["Start (s rel. Landung)"]) + np.arange(len(vals)) / fs
    to, pk = float(cur["Absprung (s rel. Landung)"]), float(cur["Peak (s rel. Landung)"])
    fig = go.Figure(go.Scatter(x=t, y=vals, mode="lines", line=dict(color="#1f77b4", width=1.2), showlegend=False))
    fig.add_vrect(x0=to, x1=0, fillcolor="lightblue", opacity=0.18, line_width=0)
    fig.add_vrect(x0=0, x1=0.25, fillcolor="gray", opacity=0.10, line_width=0)
    i_pk = int(np.argmin(np.abs(t - pk)))
    fig.add_trace(go.Scatter(x=[t[i_pk]], y=[vals[i_pk]], mode="markers", showlegend=False,
                             marker=dict(size=10, color=color, symbol="triangle-down")))
    fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1)
    fig.add_hline(y=16, line_dash="dot", line_color="#E3001B", line_width=1)
    fig.update_layout(title=dict(text=title, font=dict(size=12, color="#999" if excluded else None)),
                      height=260, margin=dict(l=10, r=10, t=36, b=10), template="plotly_white",
                      xaxis_title="Zeit rel. Landung (s)", yaxis_title="accRes (g)")
    return fig


def _sites(cur_day: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Absprungstellen eines Messtags: Absprünge aller Athleten werden nach GPS-Position gruppiert
    (Dichte der Absprünge entlang der Falllinie) und von oben nach unten nummeriert. Rückgabe: {(Athlet, Run, Sprung): Label}, Übersicht.
    """
    need = {"Lat Absprung", "Lon Absprung"}
    if cur_day.empty or not need <= set(cur_day.columns):
        return {}, pd.DataFrame()
    d = cur_day.copy()
    for c in ["Lat Absprung", "Lon Absprung", "Höhe Landung (m)"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Lat Absprung", "Lon Absprung"])
    if len(d) < 3:
        return {}, pd.DataFrame()
    lat0, lon0 = d["Lat Absprung"].median(), d["Lon Absprung"].median()
    x = ((d["Lon Absprung"] - lon0) * 111320 * np.cos(np.radians(lat0))).values
    y = ((d["Lat Absprung"] - lat0) * 110540).values
    # Position entlang der Falllinie (Hauptrichtung der Absprünge), dann Gruppen an Lücken > gap_m trennen
    X = np.c_[x, y]; X = X - X.mean(axis=0)
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    pos = X @ u
    # Dichte entlang der Falllinie (Gauss-Kern 8 m); Stellen = Dichtemaxima, Grenzen = Minima dazwischen
    grid = np.arange(pos.min() - 30, pos.max() + 30, 1.0)
    dens = np.exp(-0.5 * ((grid[:, None] - pos[None, :]) / 8.0) ** 2).sum(axis=1)
    pk = [i for i in range(1, len(grid) - 1)
          if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1] and dens[i] >= 0.15 * dens.max()]
    bounds = [grid[pk[j] + int(np.argmin(dens[pk[j]:pk[j + 1]]))] for j in range(len(pk) - 1)]
    lab = np.searchsorted(bounds, pos)
    d["_cl"] = lab
    min_n = max(3, int(0.03 * len(d)))
    alt = d.groupby("_cl")["Höhe Landung (m)"].median() if "Höhe Landung (m)" in d.columns else pd.Series(0, index=d["_cl"].unique())
    big = [k for k in alt.sort_values(ascending=False).index if (d["_cl"] == k).sum() >= min_n]
    names = {k: f"Stelle {i + 1}" for i, k in enumerate(big)}
    d["Absprungstelle"] = d["_cl"].map(names).fillna("einzeln")
    mapping = {(str(r.Athlet), str(r.Run), str(r.Sprung)): r.Absprungstelle for r in d.itertuples()}
    summ = d.groupby("Absprungstelle").agg(Sprünge=("Sprung", "count"),
                                           Höhe_m=("Höhe Landung (m)", "median") if "Höhe Landung (m)" in d.columns else ("Sprung", "count"))
    return mapping, summ.reset_index()


def show():
    st.info("Keine Rohdaten hochgeladen – hier siehst du die **gespeicherten Sprünge** aus dem Google Sheet (nur Ansicht). "
            "Neue Messungen lädst du im Tab «Daten laden» hoch; sie werden automatisch zu den bestehenden Daten ergänzt.")
    if not gsheets.enabled():
        st.warning("Google Sheet nicht verbunden.")
        return
    try:
        res, cur = _load()
    except Exception as e:
        st.error(f"Google Sheet nicht erreichbar: {e}")
        return
    if res.empty:
        st.info("Noch keine gespeicherten Sprünge.")
        return

    res = res.copy()
    res["Landungsart"] = res.get("Landungsart", "").fillna("").astype(str).str.strip()
    res["_ok"] = res["In Auswertung"].map(_as_bool) if "In Auswertung" in res.columns else True

    c1, c2, c3 = st.columns(3)
    ath = c1.selectbox("Athlet", sorted(res["Athlet"].unique()), key="sv_ath")
    r_a = res[res["Athlet"] == ath]
    tage = sorted((r_a["Datum"].astype(str) + " · " + r_a["Ort"].astype(str)).unique())
    tag = c2.selectbox("Messtag", tage, key="sv_tag")
    d, o = tag.split(" · ", 1)
    r_t = r_a[(r_a["Datum"].astype(str) == d) & (r_a["Ort"].astype(str) == o)]
    if r_t["Position"].nunique() > 1:
        pos = c3.selectbox("Sensorposition", sorted(r_t["Position"].unique()), key="sv_pos")
        r_t = r_t[r_t["Position"] == pos]
    only_ok = st.checkbox("Nur Sprünge in der Auswertung", value=False, key="sv_ok")
    if only_ok:
        r_t = r_t[r_t["_ok"]]

    site_map, site_summ = {}, pd.DataFrame()
    if not cur.empty:
        cur_day = cur[(cur["Datum"].astype(str) == d) & (cur["Ort"].astype(str) == o)]
        site_map, site_summ = _sites(cur_day)
        cur = cur_day[cur_day["Athlet"] == ath]
    if site_map:
        r_t = r_t.copy()
        r_t["Absprungstelle"] = [site_map.get((str(a), str(r), str(j)), "–") for a, r, j in zip(r_t["Athlet"], r_t["Run"], r_t["Sprung"])]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sprünge", len(r_t)); m2.metric("In Auswertung", int(r_t["_ok"].sum()))
    m3.metric("Runs", r_t["Run"].nunique())
    li = pd.to_numeric(r_t.get("Landeimpuls 0.25 s (g·s)"), errors="coerce")
    m4.metric("Ø Landeimpuls (g·s)", f"{li[r_t['_ok']].mean():.2f}" if li.notna().any() else "–")
    if cur.empty:
        st.caption("Für diesen Messtag sind noch keine Kurven gespeichert – es werden nur die Kennwerte angezeigt.")
    if not site_summ.empty:
        with st.expander("Absprungstellen an diesem Messtag (GPS)", expanded=False):
            res_day = res[(res["Datum"].astype(str) == d) & (res["Ort"].astype(str) == o)].copy()
            res_day["Absprungstelle"] = [site_map.get((str(a), str(r), str(j)), "–") for a, r, j in zip(res_day["Athlet"], res_day["Run"], res_day["Sprung"])]
            for c in ["Flugzeit (s)", "Peak roh (g)"]:
                res_day[c] = pd.to_numeric(res_day[c], errors="coerce")
            agg = res_day.groupby("Absprungstelle").agg(**{"Flugzeit Median (s)": ("Flugzeit (s)", "median"), "Peak roh Median (g)": ("Peak roh (g)", "median"),
                                                           "davon in Auswertung": ("_ok", "sum")}).reset_index()
            tab = site_summ.merge(agg, on="Absprungstelle", how="left").rename(columns={"Höhe_m": "Höhe (m ü. M.)"})
            st.dataframe(tab.round(2), use_container_width=True, hide_index=True)
            st.caption("Alle Absprünge des Messtags (alle Athleten) werden nach GPS-Position gruppiert und von oben (Stelle 1) nach unten nummeriert. "
                       "Stellen mit sehr kurzer Flugzeit sind meist Wellen statt Kicker. GPS-Ungenauigkeiten einzelner Sensoren können zu Fehlzuordnungen führen.")

    show_cols = [c for c in ["Sprung", "Absprungstelle", "Flugzeit (s)", "Peak roh (g)", "Peak (g)", "Landeimpuls 0.25 s (g·s)", "TTP (s)",
                             "16g geclippt", "Landungsart", "Tricks / Notiz", "Kommentar", "In Auswertung", "Ausschlussgrund"]
                 if c in r_t.columns]
    for run, g in r_t.groupby("Run", sort=True):
        g = g.sort_values("Sprung")
        n_out = int((~g["_ok"]).sum())
        trick = str(g["Tricks / Notiz"].iloc[0]) if "Tricks / Notiz" in g.columns else ""
        label = f"Run {run} — {len(g)} Sprünge" + (f" | {n_out} nicht in Auswertung" if n_out else "") + (f" | {trick}" if trick.strip() else "")
        with st.expander(label, expanded=False):
            st.dataframe(g[show_cols], use_container_width=True, hide_index=True)
            if cur.empty:
                continue
            cr = cur[cur["Run"].astype(str) == str(run)].set_index("Sprung")
            cols = st.columns(3)
            for i, (_, j) in enumerate(g.iterrows()):
                if j["Sprung"] not in cr.index:
                    continue
                art = j.get("Landungsart", "")
                site = j.get("Absprungstelle", "")
                title = (f"{'✕ ' if not j['_ok'] else ''}{j['Sprung']}  {float(j['Peak roh (g)']):.1f} g  "
                         f"{float(j['Flugzeit (s)']):.2f} s  {art or ''}" + (f"  · {site}" if site and site != "–" else ""))
                fig = _curve_fig(cr.loc[j["Sprung"]], title, _ART_COLORS.get(art, "#d62728"), not j["_ok"])
                cols[i % 3].plotly_chart(fig, use_container_width=True, key=f"sv_{run}_{j['Sprung']}")
