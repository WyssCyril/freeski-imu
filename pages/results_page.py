"""
Ergebnisse: Gespeicherte Sprungauswertungen pro Athlet/Sensor, Excel-Export + Import.
"""
import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import gsheets

# Spaltenmapping Excel → interne Spalten (für Statistik-Seite)
_EXCEL_TO_INTERNAL = {
    "Peak (g)":      "peak_res_g",
    "Peak roh (g)":  "peak_res_g_raw",
    "Flugzeit (s)":  "flight_time_s",
    "TTP (s)":       "time_to_peak_s",
    "RFD (g/s)":     "rfd_g_per_s",
    "Impuls (g·s)":  "impulse_net_g_s",
    "Landeimpuls 0.25 s (g·s)": "landing_impulse_025_g_s",
    "16g geclippt":  "clipped_16g",
    "Landungsart":   "landing_type",
    "Sprung":        "jump_id",
    "In Auswertung": "plausible",
    "Ausschlussgrund": "exclusion_reason",
    "Kommentar":     "comment",
}


FLIGHT_MIN_S = 1.5   # Kicker-Sprünge hier typ. 1.8–2.2 s; kürzer → Rail/Roller?
PEAK_MIN_G   = 7.0   # Kicker-Landungen typ. 8–14 g


def jump_hint(peak_g: float, flight_s: float, j_num: int, run_matched: bool) -> str:
    """Automatischer Prüfhinweis pro Sprung (leer = unauffällig)."""
    hints = []
    if j_num > 3:
        hints.append(f"{j_num}. Sprung im Run (Protokoll hat 3)")
    if flight_s < FLIGHT_MIN_S or peak_g < PEAK_MIN_G:
        hints.append("kurz/klein → Rail oder Roller?")
    if not run_matched:
        hints.append("kein Protokoll-Match")
    return "; ".join(hints)


class _SimpleMeta:
    """Minimales Meta-Objekt wie admos_parser.SensorMeta, aber aus Excel gebaut."""
    def __init__(self, athlete_code, date, location, position_label):
        self.athlete_code   = athlete_code
        self.date           = date
        self.location       = location
        self.position_label = position_label
        self.sensor_id      = ""


def _import_excel(file) -> int:
    """Liest Batch-Excel (Sheet 'Einzelsprünge') und importiert die Sprünge."""
    try:
        df = pd.read_excel(file, sheet_name="Einzelsprünge")
    except Exception as e:
        st.error(f"Fehler beim Lesen: {e}")
        return 0
    return _import_df(df)


def _import_df(df: pd.DataFrame) -> int:
    """
    Fügt Sprünge (Excel-Spaltennamen) in st.session_state['jump_results'] ein.
    Gibt Anzahl importierter Sprünge zurück.
    """
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


def _as_bool(v) -> bool:
    """Leere Zellen (ältere Sheet-Zeilen ohne Spalte) gelten als True."""
    if isinstance(v, str):
        return v.strip().upper() not in ("FALSE", "0", "NEIN")
    try:
        return bool(v) if v == v else True
    except Exception:
        return True


def _num(v, ndigits: int) -> float:
    """Zahl gerundet; fehlende/leere Werte (z.B. alte Sheet-Zeilen) → NaN."""
    v = pd.to_numeric(v, errors="coerce")
    return float("nan") if pd.isna(v) else round(float(v), ndigits)


def _collect_results(sessions_loaded: dict) -> pd.DataFrame:
    """Sammelt alle gespeicherten Sprungresultate aus dem Session-State."""
    jump_results = st.session_state.get("jump_results", {})
    rows = []

    def _group(m):
        return (gsheets.norm_athlet(m.athlete_code), str(m.date), m.location, m.position_label)

    # Frisch analysierte Athleten (Tag+Sensor) ersetzen die aus dem Sheet geladenen Zeilen komplett
    fresh_groups = {
        _group(e["meta"]) for k, e in jump_results.items()
        if not k.startswith("import_") and e.get("meta") is not None
    }
    for run_key, entry in jump_results.items():
        jumps = entry.get("jumps")
        m = entry.get("meta")
        if jumps is None or (hasattr(jumps, "empty") and jumps.empty):
            continue
        if run_key.startswith("import_") and m is not None and _group(m) in fresh_groups:
            continue
        try:
            date_fmt = pd.to_datetime(str(m.date), format="%Y%m%d").strftime("%d.%m.%Y") if m else ""
        except Exception:
            date_fmt = m.date if m else ""
        run_note = entry.get("run_note", "")
        run_lbl  = entry.get("run", "")
        run_matched = bool(str(run_note).strip()) or (
            "landing_type" in jumps.columns
            and jumps["landing_type"].astype(str).str.strip().ne("").any()
        )
        for pos, (_, jrow) in enumerate(jumps.iterrows(), 1):
            jid_m = re.search(r"(\d+)$", str(jrow.get("jump_id", "")))
            j_num = int(jid_m.group(1)) if jid_m else pos
            peak_v   = float(jrow.get("peak_res_g", 0))
            flight_v = float(jrow.get("flight_time_s", 0))
            hint = jrow.get("Hinweis") if str(jrow.get("Hinweis", "")).strip() else \
                jump_hint(peak_v, flight_v, j_num, run_matched)
            rows.append({
                "Hinweis": hint,
                "Athlet": gsheets.norm_athlet(m.athlete_code) if m else run_key,
                "Datum": date_fmt,
                "Ort": m.location if m else "",
                "Position": m.position_label if m else "",
                "Run": run_lbl or jrow.get("Run", ""),
                "Sprung": jrow.get("jump_id", ""),
                "Tricks / Notiz": run_note,
                "Flugzeit (s)": round(float(jrow.get("flight_time_s", 0)), 3),
                "Peak (g)": round(float(jrow.get("peak_res_g", 0)), 2),
                "Peak roh (g)": round(float(jrow.get("peak_res_g_raw", jrow.get("peak_res_g", 0))), 2),
                "TTP (s)": round(float(jrow.get("time_to_peak_s", 0)), 4),
                "RFD (g/s)": round(float(jrow.get("rfd_g_per_s", 0)), 2),
                "Impuls (g·s)": round(float(jrow.get("impulse_g_s", 0)), 4),
                "Landeimpuls 0.25 s (g·s)": _num(jrow.get("landing_impulse_025_g_s"), 4),
                "16g geclippt": bool(jrow.get("clipped_16g", False)),
                "Landungsart": jrow.get("landing_type", ""),
                "Kommentar": jrow.get("comment", ""),
                "In Auswertung": _as_bool(jrow.get("plausible", True)),
                "Ausschlussgrund": jrow.get("exclusion_reason", "") or "",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Frische Auswertungen (später im Dict) gewinnen gegenüber aus dem Sheet geladenen
    return df.drop_duplicates(gsheets.KEY_COLS, keep="last").reset_index(drop=True)


# ── Automatisches Speichern + Warnung bei ungespeicherten Korrekturen ─────────
_FP_COLS = ["Athlet", "Datum", "Ort", "Position", "Run", "Sprung", "Landungsart", "Tricks / Notiz", "Kommentar",
            "Peak (g)", "Flugzeit (s)", "In Auswertung"]


def _fresh_groups() -> set:
    """Athlet/Tag/Ort/Position, die in dieser Sitzung frisch analysiert wurden (nicht aus dem Sheet geladen)."""
    jr = st.session_state.get("jump_results", {})
    return {(gsheets.norm_athlet(e["meta"].athlete_code), str(e["meta"].date), e["meta"].location, e["meta"].position_label)
            for k, e in jr.items() if not k.startswith("import_") and e.get("meta") is not None
            and e.get("jumps") is not None and not e["jumps"].empty}


def _fingerprint(df: pd.DataFrame) -> str:
    import hashlib
    cols = [c for c in _FP_COLS if c in df.columns]
    if df.empty or not cols:
        return ""
    d = df[cols].astype(str).sort_values(cols).to_csv(index=False)
    return hashlib.md5(d.encode()).hexdigest()


def _sheet_groups() -> set:
    """Athlet/Tag/Ort/Position, die bereits im Sheet stehen (Datum im Format YYYYMMDD wie in den Metadaten)."""
    df = gsheets.load()
    if df.empty:
        return set()
    out = set()
    for r in df[["Athlet", "Datum", "Ort", "Position"]].astype(str).itertuples(index=False):
        d = r.Datum.strip()
        if len(d) == 10 and d[2] == "." and d[5] == ".":
            d = d[6:] + d[3:5] + d[:2]
        out.add((gsheets.norm_athlet(r.Athlet), d, r.Ort, r.Position))
    return out


CURVE_PRE_S, CURVE_POST_S = 0.4, 0.8   # Ausschnitt: 0.4 s vor Absprung bis 0.8 s nach Landung


def build_curve_rows() -> pd.DataFrame:
    """Kurvenausschnitte (accRes, 200 Hz) aller frisch analysierten Sprünge für das Blatt «Kurven»."""
    rows = []
    for k, e in st.session_state.get("jump_results", {}).items():
        jumps, m, df_imu = e.get("jumps"), e.get("meta"), e.get("df_imu")
        if k.startswith("import_") or m is None or df_imu is None or jumps is None or jumps.empty:
            continue
        col = "accRes [g]" if "accRes [g]" in df_imu.columns else None
        if col is None:
            continue
        acc = df_imu[col].values
        try:
            date_fmt = pd.to_datetime(str(m.date), format="%Y%m%d").strftime("%d.%m.%Y")
        except Exception:
            date_fmt = str(m.date)
        for _, j in jumps.iterrows():
            to, la, pk = int(j["takeoff_idx"]), int(j["landing_idx"]), int(j.get("peak_res_idx", j["landing_idx"]))
            i0 = max(0, to - int(CURVE_PRE_S * 200)); i1 = min(len(acc), la + int(CURVE_POST_S * 200) + 1)
            rows.append({"Athlet": gsheets.norm_athlet(m.athlete_code), "Datum": date_fmt, "Ort": m.location,
                         "Position": m.position_label, "Run": e.get("run", ""), "Sprung": j.get("jump_id", ""),
                         "Start (s rel. Landung)": round((i0 - la) / 200, 3), "Absprung (s rel. Landung)": round((to - la) / 200, 3),
                         "Peak (s rel. Landung)": round((pk - la) / 200, 3), "Abtastrate (Hz)": 200,
                         "accRes (g)": ",".join(f"{v:.2f}" for v in acc[i0:i1])})
    return pd.DataFrame(rows)


def save_results(sessions_loaded: dict) -> tuple[int, int]:
    """Speichert alle aktuellen Ergebnisse ins Sheet und merkt sich den gespeicherten Stand."""
    df = _collect_results(sessions_loaded)
    n_new, n_upd = gsheets.save(df)
    try:
        gsheets.save_curves(build_curve_rows())
    except Exception as e:
        st.warning(f"Kurven konnten nicht gespeichert werden: {e}")
    try:
        from pages import stored_view
        stored_view.clear_cache()
    except Exception:
        pass
    st.session_state["_saved_fp"] = _fingerprint(df)
    st.session_state.setdefault("_autosaved_groups", set()).update(_fresh_groups())
    return n_new, n_upd


def autosave_and_warn(sessions_loaded: dict, where: str) -> None:
    """
    Neu analysierte Athleten werden einmal automatisch gespeichert.
    Ändern sich danach Korrekturen (Landungsart, Kommentar, Trick …), erscheint eine Warnung mit Speichern-Knopf.
    """
    if not gsheets.enabled():
        return
    fresh = _fresh_groups()
    if not fresh:
        return
    done = st.session_state.setdefault("_autosaved_groups", set())
    skipped = st.session_state.setdefault("_autosave_skipped", set())
    new = fresh - done - skipped
    if new:
        try:
            existing = _sheet_groups()
        except Exception as e:
            st.error(f"Google Sheet nicht erreichbar, nicht gespeichert: {e}")
            return
        already = {g for g in new if g in existing}
        if already:
            # Nicht automatisch überschreiben: gespeicherte Landungsarten/Kommentare könnten verloren gehen
            skipped.update(already)
        if new - already:
            try:
                with st.spinner("Speichere automatisch im Google Sheet…"):
                    save_results(sessions_loaded)
                st.toast("Automatisch im Google Sheet gespeichert", icon="✅")
            except Exception as e:
                st.error(f"Automatisches Speichern fehlgeschlagen: {e}")
            return
    if fresh & skipped - done:
        c1, c2 = st.columns([3, 1])
        c1.warning("⚠ Dieser Athlet ist **bereits im Google Sheet gespeichert** und wurde deshalb nicht automatisch "
                   "überschrieben. Speichern ersetzt die gespeicherte Version (inkl. Landungsarten und Kommentare) durch diese Analyse.")
        if c2.button("Gespeicherte Version ersetzen", key=f"save_over_{where}"):
            try:
                with st.spinner("Speichere…"):
                    save_results(sessions_loaded)
                skipped.difference_update(fresh)
                st.rerun()
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")
        return
    if _fingerprint(_collect_results(sessions_loaded)) != st.session_state.get("_saved_fp"):
        c1, c2 = st.columns([3, 1])
        c1.warning("⚠ Du hast Korrekturen vorgenommen, die noch **nicht im Google Sheet gespeichert** sind. "
                   "Beim Laden eines neuen Athleten oder Neuladen der Seite gehen sie verloren.")
        if c2.button("Korrekturen jetzt speichern", type="primary", key=f"save_corr_{where}"):
            try:
                with st.spinner("Speichere…"):
                    n_new, n_upd = save_results(sessions_loaded)
                st.toast(f"Gespeichert: {n_new} neue, {n_upd} aktualisierte Sprünge", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")


def has_unsaved_corrections(sessions_loaded: dict) -> bool:
    fresh = _fresh_groups()
    if not fresh or not gsheets.enabled():
        return False
    if not fresh <= st.session_state.get("_autosaved_groups", set()):
        return bool(fresh & st.session_state.get("_autosave_skipped", set()))
    return _fingerprint(_collect_results(sessions_loaded)) != st.session_state.get("_saved_fp")


def _sheet_sync():
    """Statuszeile + einmaliges Laden aus dem Google Sheet beim Start."""
    if not gsheets.enabled():
        return
    if not st.session_state.get("_gs_loaded"):
        try:
            with st.spinner("Lade gespeicherte Auswertungen…"):
                df = gsheets.load()
            n = _import_df(df) if not df.empty else 0
            st.session_state["_gs_loaded"] = True
            st.session_state["_gs_n"] = n
        except Exception as e:
            st.session_state["_gs_loaded"] = True
            st.session_state["_gs_n"] = -1
            st.warning(f"Google Sheet nicht erreichbar: {e}")
    n = st.session_state.get("_gs_n", 0)
    txt = f"{n} Sprünge aus der Datenablage geladen" if n >= 0 else "Datenablage nicht erreichbar"
    st.caption(f"Datenablage: [Google Sheet]({gsheets.sheet_url()}) · {txt}")


# ── Vergleich mit Boxplots ───────────────────────────────────────────────────
_CMP_PARAMS = {
    "Landeimpuls 0.25 s (g·s)": "Netto-Landeimpuls 0–0.25 s (g·s)",
    "Peak roh (g)": "Peak roh (g)",
    "Peak (g)": "Peak korrigiert (g)",
    "Flugzeit (s)": "Flugzeit (s)",
    "TTP (s)": "Time to Peak (s)",
    "RFD (g/s)": "RFD (g/s)",
}
_CMP_GROUPS = ["Landungsart", "Athlet", "Athlet × Landungsart", "Ort", "Datum"]
_ART_COLORS = {"vorwärts": "#4D4D4D", "switch": "#E3001B", "crash": "#F4A6AE", "ohne Angabe": "#BDBDBD"}


def _compare_section(df_filtered: pd.DataFrame, df_eval: pd.DataFrame) -> None:
    """Boxplot-Vergleich eines Kennwerts zwischen frei wählbaren Gruppen."""
    import plotly.graph_objects as go
    from scipy import stats

    c1, c2, c3 = st.columns(3)
    params = [p for p in _CMP_PARAMS if p in df_filtered.columns]
    param = c1.selectbox("Kennwert", params, format_func=lambda p: _CMP_PARAMS[p], key="cmp_param")
    group_by = c2.selectbox("Vergleichen nach", _CMP_GROUPS, key="cmp_group")
    use_all = c3.checkbox("Auch ausgeschlossene Sprünge", value=False, key="cmp_all",
                          help="Standard: nur Sprünge «In Auswertung» (Flugzeit ≥ 1.4 s, max. 3 pro Run).")
    data = (df_filtered if use_all else df_eval).copy()
    data[param] = pd.to_numeric(data[param], errors="coerce")
    data = data.dropna(subset=[param])
    if data.empty:
        st.info("Keine Daten für diesen Kennwert mit den aktuellen Filtern.")
        return
    if group_by == "Athlet × Landungsart":
        data["_grp"] = data["Athlet"].astype(str) + " · " + data["Landungsart"].astype(str)
    else:
        data["_grp"] = data[group_by].astype(str)

    groups_all = sorted(data["_grp"].unique())
    default = groups_all
    if group_by == "Landungsart":
        default = [g for g in ["vorwärts", "switch"] if g in groups_all] or groups_all
    c4, c5 = st.columns([3, 1])
    sel = c4.multiselect("Gruppen vergleichen", groups_all, default=default, key=f"cmp_sel_{group_by}")
    show_pts = c5.checkbox("Einzelpunkte", value=True, key="cmp_pts")
    data = data[data["_grp"].isin(sel)]
    if data.empty:
        st.info("Mindestens eine Gruppe auswählen.")
        return

    fig = go.Figure()
    for g in sel:
        vals = data.loc[data["_grp"] == g, param]
        if vals.empty:
            continue
        art = g.split(" · ")[-1] if group_by in ("Landungsart", "Athlet × Landungsart") else None
        color = _ART_COLORS.get(art, "#E3001B")
        fig.add_trace(go.Box(y=vals, name=f"{g} (n={len(vals)})", marker_color=color, line_color=color,
                             fillcolor="rgba(0,0,0,0)", boxpoints="all" if show_pts else False,
                             jitter=0.35, pointpos=0, marker=dict(size=5, opacity=0.6), boxmean=False))
    if param == "Peak roh (g)":
        fig.add_hline(y=16, line_dash="dot", line_color="gray",
                      annotation_text="Messgrenze 16 g", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=430, showlegend=False,
                      yaxis_title=_CMP_PARAMS[param], margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Kennwerte pro Gruppe
    summ = data.groupby("_grp")[param].agg(n="count", Median="median",
                                           Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75),
                                           Mittelwert="mean", SD="std").reindex([g for g in sel if g in set(data["_grp"])])
    summ.index.name = group_by
    st.dataframe(summ.round(3), use_container_width=True)

    # Statistischer Test (Orientierung)
    samples = [data.loc[data["_grp"] == g, param].values for g in summ.index if (data["_grp"] == g).sum() >= 2]
    if len(samples) == 2:
        a, b = samples
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        t = stats.ttest_ind(a, b, equal_var=False)
        sd_p = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (b.mean() - a.mean()) / sd_p if sd_p > 0 else np.nan
        st.markdown(f"**Test {summ.index[0]} vs. {summ.index[1]}:** Mann-Whitney-U p = {u.pvalue:.3f} · "
                    f"Welch-t-Test p = {t.pvalue:.3f} · Cohen's d = {d:.2f}")
    elif len(samples) >= 3:
        k = stats.kruskal(*samples)
        st.markdown(f"**Kruskal-Wallis-Test über {len(samples)} Gruppen:** p = {k.pvalue:.3f}")
    st.caption("Die Tests behandeln alle Sprünge als unabhängig und dienen nur zur Orientierung. "
               "Für die Masterarbeit gilt das gemischte Modell mit Athlet als Zufallseffekt (Berechnungsprotokoll Kap. 8).")


def show():
    st.header("Ergebnisse")
    _sheet_sync()

    # ── Batch-Import ──────────────────────────────────────────────────────
    with st.expander("Frühere Batches importieren", expanded=False):
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

    n_excl = int((~df["In Auswertung"]).sum()) if "In Auswertung" in df.columns else 0
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Sprünge total", len(df), help=f"davon {n_excl} nicht in Auswertung" if n_excl else None)
    c_b.metric("Athleten", df["Athlet"].nunique())
    c_c.metric("Orte", df["Ort"].nunique())

    if gsheets.enabled():
        autosave_and_warn(sessions_loaded, "results")
        if st.button("In Google Sheet speichern", key="gs_save",
                     help="Neu analysierte Athleten werden automatisch gespeichert. Hier kannst du jederzeit manuell speichern."):
            try:
                with st.spinner("Speichere…"):
                    n_new, n_upd = save_results(sessions_loaded)
                st.success(f"Gespeichert: {n_new} neue, {n_upd} aktualisierte Sprünge.")
            except Exception as e:
                st.error(f"Speichern fehlgeschlagen: {e}")

    # ── Filter ────────────────────────────────────────────────────────────
    df["Landungsart"] = df["Landungsart"].fillna("").astype(str).str.strip().replace("", "ohne Angabe")
    col1, col2, col3 = st.columns(3)
    all_athletes = sorted(df["Athlet"].unique())
    sel_athletes = col1.multiselect("Athlet", all_athletes, default=all_athletes, key="res_f_ath")
    all_orte = sorted(df["Ort"].unique())
    sel_orte = col2.multiselect("Ort", all_orte, default=all_orte, key="res_f_ort")
    art_order = ["vorwärts", "switch", "crash", "ohne Angabe"]
    all_arten = [a for a in art_order if a in set(df["Landungsart"])] + \
                sorted(set(df["Landungsart"]) - set(art_order))
    sel_arten = col3.multiselect("Landungsart", all_arten, default=all_arten, key="res_f_art")

    with st.expander("Weitere Filter", expanded=False):
        f1, f2, f3 = st.columns(3)
        all_pos = sorted(df["Position"].unique())
        sel_pos = f1.multiselect("Sensorposition", all_pos, default=all_pos, key="res_f_pos")
        only_eval = f2.checkbox("Nur Sprünge in der Auswertung anzeigen", key="res_only_eval")
        only_hint = f3.checkbox("Nur auffällige Sprünge anzeigen", key="only_hint")

    mask = (
        df["Athlet"].isin(sel_athletes) &
        df["Ort"].isin(sel_orte) &
        df["Landungsart"].isin(sel_arten) &
        df["Position"].isin(sel_pos)
    )
    df_filtered = df[mask].reset_index(drop=True)
    if only_hint:
        df_filtered = df_filtered[df_filtered["Hinweis"].astype(str).str.strip() != ""].reset_index(drop=True)
    if "In Auswertung" in df_filtered.columns:
        if only_eval:
            df_filtered = df_filtered[df_filtered["In Auswertung"]].reset_index(drop=True)
        df_eval = df_filtered[df_filtered["In Auswertung"]].reset_index(drop=True)
    else:
        df_eval = df_filtered
    n_out = len(df_filtered) - len(df_eval)

    # ── Kennzahlen (nur Sprünge in der Auswertung) ────────────────────────
    st.markdown(f"**{len(df_filtered)} Sprünge gefiltert**"
                + (f" · davon {n_out} nicht in der Auswertung (Flugzeit < 1.4 s oder mehr als 3 Sprünge im Run)" if n_out else ""))
    if not df_eval.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("In Auswertung", len(df_eval))
        c2.metric("Ø Peak (g)", f"{df_eval['Peak (g)'].mean():.2f}")
        li = pd.to_numeric(df_eval.get("Landeimpuls 0.25 s (g·s)"), errors="coerce")
        c3.metric("Ø Landeimpuls (g·s)", f"{li.mean():.2f}" if li.notna().any() else "–")
        c4.metric("Ø Flugzeit (s)", f"{df_eval['Flugzeit (s)'].mean():.2f}")
        c5.metric("Geclippt (16 g)", f"{df_eval['16g geclippt'].astype(bool).mean():.0%}")

    # ── Alle Sprünge ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Alle geladenen Sprünge")
    all_cols = ["In Auswertung", "Athlet", "Datum", "Ort", "Run", "Sprung", "Landungsart", "Tricks / Notiz",
                "Flugzeit (s)", "Peak (g)", "Peak roh (g)", "Landeimpuls 0.25 s (g·s)", "TTP (s)", "RFD (g/s)",
                "Impuls (g·s)", "16g geclippt", "Ausschlussgrund", "Hinweis", "Position", "Kommentar"]
    display_cols = [c for c in all_cols if c in df_filtered.columns]
    sort_cols = [c for c in ["Athlet", "Datum", "Run", "Sprung"] if c in df_filtered.columns]
    table = df_filtered.sort_values(sort_cols)[display_cols].reset_index(drop=True) if sort_cols else df_filtered[display_cols]
    st.dataframe(
        table, use_container_width=True, hide_index=True, height=min(600, 38 + 35 * max(len(table), 1)),
        column_config={
            "In Auswertung": st.column_config.CheckboxColumn("In Auswertung", help="Nur diese Sprünge zählen in Kennzahlen, Statistik und Export."),
            "16g geclippt": st.column_config.CheckboxColumn("Geclippt"),
            "Flugzeit (s)": st.column_config.NumberColumn(format="%.2f"),
            "Peak (g)": st.column_config.NumberColumn("Peak korr. (g)", format="%.1f"),
            "Peak roh (g)": st.column_config.NumberColumn(format="%.1f"),
            "Landeimpuls 0.25 s (g·s)": st.column_config.NumberColumn("Landeimpuls (g·s)", format="%.2f"),
            "TTP (s)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    # ── Übersicht Athlet × Landungsart ────────────────────────────────────
    if not df_eval.empty:
        st.markdown("**Übersicht pro Athlet und Landungsart** (nur Sprünge in der Auswertung)")
        ev = df_eval.assign(
            _li=pd.to_numeric(df_eval.get("Landeimpuls 0.25 s (g·s)"), errors="coerce"),
            _clip=df_eval["16g geclippt"].astype(bool))
        summ = ev.groupby(["Athlet", "Landungsart"], sort=True).agg(
            Sprünge=("Sprung", "count"), Peak_korr=("Peak (g)", "mean"), Peak_roh_Median=("Peak roh (g)", "median"),
            Landeimpuls=("_li", "mean"), Flugzeit=("Flugzeit (s)", "mean"), Geclippt=("_clip", "mean")).reset_index()
        summ["Geclippt"] = (summ["Geclippt"] * 100).round(0)
        summ = summ.rename(columns={"Peak_korr": "Ø Peak korr. (g)", "Peak_roh_Median": "Median Peak roh (g)",
                                    "Landeimpuls": "Ø Landeimpuls (g·s)", "Flugzeit": "Ø Flugzeit (s)", "Geclippt": "Geclippt (%)"})
        st.dataframe(summ.round(2), use_container_width=True, hide_index=True)

    # ── Vergleich mit Boxplots ────────────────────────────────────────────
    st.divider()
    st.subheader("Vergleich mit Boxplots")
    _compare_section(df_filtered, df_eval)

    # ── Durchschnittlicher Impact ─────────────────────────────────────────
    st.divider()
    st.subheader("Durchschnittlicher Impact")
    if not df_eval.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ø Peak (g) — alle Sprünge", f"{df_eval['Peak (g)'].mean():.2f}")
        c2.metric("Median Peak (g)", f"{df_eval['Peak (g)'].median():.2f}")
        c3.metric("Max. Peak (g)", f"{df_eval['Peak (g)'].max():.2f}")
        c4.metric("Anzahl Sprünge", len(df_eval))

        # Pro Position
        st.markdown("**Ø Peak (g) pro Sensorposition:**")
        pos_summary = df_eval.groupby("Position")["Peak (g)"].agg(["mean", "median", "max", "count"]).round(2)
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
        Landeimpuls_025_mean=("Landeimpuls 0.25 s (g·s)", "mean"),
        Geclippt_16g=("16g geclippt", "sum"),
    )

    jump_cols = ["Hinweis", "Athlet", "Datum", "Ort", "Position", "Run", "Sprung",
                 "Flugzeit (s)", "Peak (g)", "Peak roh (g)",
                 "TTP (s)", "RFD (g/s)", "Impuls (g·s)", "Landeimpuls 0.25 s (g·s)",
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

    orte = sorted(df_eval["Ort"].dropna().unique())
    cols = st.columns(len(orte) + 1)

    for i, ort in enumerate(orte):
        df_ort = df_eval[df_eval["Ort"] == ort]
        cols[i].download_button(
            f"Excel {ort}",
            data=_make_excel(df_ort, ort),
            file_name=f"impactmessung_{ort}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{ort}",
        )

    cols[-1].download_button(
        "Excel Alle",
        data=_make_excel(df_eval, "Alle"),
        file_name="impactmessung_alle.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_alle",
    )
