"""
Sprunganalyse: Pipeline Messtag → Session → Run → Sprünge
"""
import json
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.admos_parser import load_imu_raw
from utils.jump_detector import (
    detect_jumps, compute_landing_params, session_summary,
    preprocess_imu, detect_vertical_axis,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))
import sensor_lib


def _get_raw_df(sess: dict) -> pd.DataFrame | None:
    path = sess.get("imu_path")
    if path and os.path.exists(path):
        return load_imu_raw(path)
    df = sess.get("imu")
    if df is not None:
        col_remap = {
            "acc_x_g": "accX [g]", "acc_y_g": "accY [g]", "acc_z_g": "accZ [g]",
            "gyr_x_dps": "gyrX [dps]", "gyr_y_dps": "gyrY [dps]", "gyr_z_dps": "gyrZ [dps]",
            "timestamp_us": "imuTimestamp [us]",
        }
        return df.rename(columns={k: v for k, v in col_remap.items() if k in df.columns})
    return None


def _detect_jumps_for_segment(df_imu: pd.DataFrame, axis_vert: str, params: dict,
                              position_label: str = "") -> pd.DataFrame:
    if len(df_imu) < 100:
        return pd.DataFrame()
    try:
        jumps = sensor_lib.detect_jumps_snow(
            df_imu,
            axis_vert=axis_vert,
            th_core_accVert=params["th_vert"],
            th_core_accRes=params["th_res"],
            th_crossing=params["th_cross"],
            min_duration_s=params["min_dur"],
        )
    except Exception as e:
        return pd.DataFrame()
    if jumps.empty:
        return pd.DataFrame()
    return compute_landing_params(df_imu, jumps, position_label=position_label)


def _run_pipeline(raw_df: pd.DataFrame, gnss_df: pd.DataFrame | None, params: dict,
                  position_label: str = "") -> dict:
    df = preprocess_imu(raw_df)
    axis_vert = detect_vertical_axis(raw_df)

    sessions_dict, _ = sensor_lib.detect_sessions_imu(df)
    result = {"sessions": {}, "axis_vert": axis_vert}

    for s_id, df_session in sessions_dict.items():
        session_result = {"df": df_session, "runs": {}}
        runs_df = pd.DataFrame()

        gnss_s = None
        if gnss_df is not None:
            try:
                gnss_clean = sensor_lib.add_resultant_speed(gnss_df)
                gnss_sessions, _ = sensor_lib.detect_sessions_gnss(gnss_clean)
                gnss_s = gnss_sessions.get(s_id, gnss_clean)
                runs_df = sensor_lib.detect_runs(
                    gnss_s,
                    v_start=params["v_start"],
                    v_hold=params["v_hold"],
                    alt_rise_end=params["alt_rise_end"],
                    alt_drop_min=params["alt_drop_min"],
                    run_duration_min=params["run_duration_min"],
                )
                if runs_df is None:
                    runs_df = pd.DataFrame()
            except Exception:
                runs_df = pd.DataFrame()

        session_result["gnss"] = gnss_s  # für synchronisierten Plot

        if not runs_df.empty:
            t_col = "imuTimestamp [us]"
            for _, run_row in runs_df.iterrows():
                r_id = run_row.get("run_id", "Run")
                if t_col in df_session.columns and "start_time_us" in run_row:
                    run_imu = df_session.loc[
                        (df_session[t_col] >= run_row["start_time_us"]) &
                        (df_session[t_col] <= run_row["end_time_us"])
                    ].reset_index(drop=True)
                else:
                    run_imu = df_session
                jumps_df = _detect_jumps_for_segment(run_imu, axis_vert, params, position_label)
                session_result["runs"][str(r_id)] = {
                    "df_imu": run_imu,
                    "run_meta": run_row.to_dict(),
                    "jumps": jumps_df,
                }
        else:
            jumps_df = _detect_jumps_for_segment(df_session, axis_vert, params, position_label)
            session_result["runs"]["Alle"] = {
                "df_imu": df_session,
                "run_meta": {},
                "jumps": jumps_df,
            }

        result["sessions"][str(s_id)] = session_result

    return result


def _plot_run(df_imu: pd.DataFrame, jumps_df: pd.DataFrame | None,
              axis_vert: str, title: str = "") -> go.Figure:
    t_col = "imuTimestamp [us]"
    if t_col in df_imu.columns:
        t = (df_imu[t_col].values - df_imu[t_col].values[0]) / 1e6
    else:
        t = np.arange(len(df_imu)) / 200.0

    acc_res_col = "accRes [g]" if "accRes [g]" in df_imu.columns else "acc_norm_g"
    acc_res = df_imu.get(acc_res_col, pd.Series(np.ones(len(df_imu))))
    acc_vert = df_imu.get(axis_vert, pd.Series(np.zeros(len(df_imu))))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        subplot_titles=["accRes [g]", f"Vertikale Achse ({axis_vert})"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=t, y=acc_res, line=dict(color="#1f77b4", width=0.8),
                             name="accRes [g]"), row=1, col=1)
    fig.add_hline(y=16, line_dash="dot", line_color="red",
                  annotation_text="16g Limit", row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=acc_vert, line=dict(color="#2ca02c", width=0.8),
                             name=axis_vert), row=2, col=1)

    if jumps_df is not None and not jumps_df.empty:
        for _, row in jumps_df.iterrows():
            lt = row.get("landing_type", "")
            color = "#2ca02c" if lt == "vorwärts" else "#ff7f0e" if lt == "switch" else "#d62728"

            def safe_t(idx_val):
                idx = min(int(idx_val), len(t) - 1)
                return float(t[idx])

            fig.add_vrect(x0=safe_t(row["takeoff_idx"]), x1=safe_t(row["landing_idx"]),
                          fillcolor="lightblue", opacity=0.12, line_width=0, row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[safe_t(row["peak_res_idx"])], y=[row["peak_res_g"]],
                mode="markers+text",
                marker=dict(size=9, color=color, symbol="triangle-down"),
                text=[f"{row['jump_id']}<br>{row['peak_res_g']:.1f}g"],
                textposition="top center", showlegend=False,
                hovertemplate=f"{row['jump_id']}: {row['peak_res_g']:.2f}g, {row['flight_time_s']:.2f}s<extra></extra>"),
                row=1, col=1)

    fig.update_layout(title=title, height=700, template="plotly_white", xaxis2_title="Zeit (s)")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=2, col=1)
    return fig


def _format_run_time(run_meta: dict) -> str:
    """Konvertiert start_time_us aus run_meta in einen lesbaren Zeitstring."""
    ts = run_meta.get("start_time_us")
    if ts is None:
        return ""
    try:
        ts = float(ts)
        # POSIX-Millisekunden → Sekunden (Wert > 1e12 = ms, sonst µs)
        if ts > 1e15:
            ts_s = ts / 1e6   # Mikrosekunden
        elif ts > 1e12:
            ts_s = ts / 1e3   # Millisekunden
        else:
            ts_s = ts         # bereits Sekunden
        dt = pd.Timestamp(ts_s, unit="s", tz="UTC").tz_convert("Europe/Zurich")
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ""


CARD_W = 300   # px pro Sprung-Karte im Scroller
CARD_H = 320   # px Höhe der Karte


def _jump_scroller(df_imu: pd.DataFrame, jumps_df: pd.DataFrame,
                   axis_vert: str, label_map: dict) -> None:
    """Rendert einen horizontal scrollbaren Streifen, eine Karte pro Sprung."""
    t_col = "imuTimestamp [us]"
    if t_col in df_imu.columns:
        t_all = (df_imu[t_col].values - df_imu[t_col].values[0]) / 1e6
    else:
        t_all = np.arange(len(df_imu)) / 200.0

    acc_res_col = "accRes [g]" if "accRes [g]" in df_imu.columns else "acc_norm_g"
    acc_arr = df_imu.get(acc_res_col, pd.Series(np.ones(len(df_imu)))).values

    pre_s, post_s = 0.4, 0.8   # Sekunden vor Absprung / nach Landung

    fig_jsons = []
    labels_out = []
    for _, row in jumps_df.iterrows():
        jid   = row["jump_id"]
        lt    = label_map.get(jid, row.get("landing_type", ""))
        peak  = float(row["peak_res_g"])
        flight = float(row["flight_time_s"])
        clipped = bool(row.get("clipped_16g", False))

        t0_idx = max(0,            int(row["takeoff_idx"]) - int(pre_s * 200))
        t1_idx = min(len(t_all)-1, int(row["landing_idx"]) + int(post_s * 200))

        t_seg   = t_all[t0_idx:t1_idx]
        acc_seg = acc_arr[t0_idx:t1_idx]

        peak_t = float(t_all[min(int(row["peak_res_idx"]), len(t_all)-1)])
        to_t   = float(t_all[min(int(row["takeoff_idx"]),  len(t_all)-1)])
        la_t   = float(t_all[min(int(row["landing_idx"]),  len(t_all)-1)])

        color = "#2ca02c" if lt == "vorwärts" else "#ff7f0e" if lt == "switch" else "#d62728"
        rot = row.get("rotations")
        rot_dir = row.get("rotation_dir", "")
        rot_str = f"  {rot}x {rot_dir}" if rot is not None else ""
        title_txt = f"{jid}  {peak:.1f}g  {flight:.2f}s{rot_str}{'  ⚠️' if clipped else ''}"
        lt_label  = lt if lt else "—"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_seg.tolist(), y=acc_seg.tolist(),
            mode="lines", line=dict(color="#1f77b4", width=1.2), showlegend=False,
        ))
        # Flugphase markieren
        fig.add_vrect(x0=to_t, x1=la_t, fillcolor="lightblue", opacity=0.18, line_width=0)
        # Peak
        fig.add_trace(go.Scatter(
            x=[peak_t], y=[peak],
            mode="markers", marker=dict(size=10, color=color, symbol="triangle-down"),
            showlegend=False,
        ))
        # 1g Flugschwelle
        fig.add_hline(y=1, line_dash="dot", line_color="gray", line_width=1,
                      annotation_text="1g (Flug)", annotation_position="right",
                      annotation_font=dict(size=8, color="gray"))
        # 16g Threshold
        fig.add_hline(y=16, line_dash="dot", line_color="red", line_width=1,
                      annotation_text="16g", annotation_position="right")
        # Absprung
        fig.add_vline(x=to_t, line_dash="dash", line_color="green", line_width=1.5,
                      annotation_text="Absprung", annotation_position="top left",
                      annotation_font=dict(size=8, color="green"))
        # Landung
        fig.add_vline(x=la_t, line_dash="dash", line_color="orange", line_width=1.5,
                      annotation_text="Landung", annotation_position="top right",
                      annotation_font=dict(size=8, color="orange"))
        fig.update_layout(
            title=dict(text=title_txt, font=dict(size=11), x=0.5),
            annotations=[dict(
                text=f"<b>{lt_label}</b>",
                x=0.5, y=1.0, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color=color),
                xanchor="center", yanchor="bottom",
            )],
            margin=dict(l=30, r=10, t=46, b=30),
            xaxis=dict(title="s", tickfont=dict(size=9)),
            yaxis=dict(title="g", tickfont=dict(size=9)),
            template="plotly_white",
            width=CARD_W, height=CARD_H,
        )
        fig_jsons.append(pio.to_json(fig))
        labels_out.append(lt_label)

    if not fig_jsons:
        return

    divs   = "".join(f'<div id="jc{i}" style="min-width:{CARD_W}px;height:{CARD_H}px;cursor:pointer;" onclick="openModal({i})"></div>'
                     for i in range(len(fig_jsons)))
    plots  = "".join(f'Plotly.newPlot("jc{i}",JSON.parse(figs[{i}]).data,JSON.parse(figs[{i}]).layout,{{displayModeBar:false}});'
                     for i in range(len(fig_jsons)))
    figs_json = json.dumps(fig_jsons)

    html = f"""
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<div style="display:flex;overflow-x:auto;gap:10px;padding:6px 2px;
            scrollbar-width:thin;-webkit-overflow-scrolling:touch;">
  {divs}
</div>

<!-- Modal -->
<div id="modal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,0.7);z-index:9999;align-items:center;justify-content:center;">
  <div style="background:white;border-radius:8px;padding:10px;width:90%;max-width:900px;position:relative;">
    <button onclick="closeModal()" style="position:absolute;top:8px;right:12px;font-size:20px;
            border:none;background:none;cursor:pointer;">✕</button>
    <div id="modal-plot" style="width:100%;height:500px;"></div>
  </div>
</div>

<script>
const figs={figs_json};
{plots}

function openModal(i) {{
  const modal = document.getElementById('modal');
  modal.style.display = 'flex';
  const data = JSON.parse(figs[i]);
  data.layout.width = null;
  data.layout.height = 480;
  data.layout.title.font = {{size: 14}};
  Plotly.newPlot('modal-plot', data.data, data.layout, {{displayModeBar: true}});
}}

function closeModal() {{
  document.getElementById('modal').style.display = 'none';
}}

document.getElementById('modal').addEventListener('click', function(e) {{
  if (e.target === this) closeModal();
}});
</script>
"""
    components.html(html, height=CARD_H + 40, scrolling=False)


def _render_run(cache_key: str, sess_id: str, run_id: str,
                result: dict, key: str, meta, axis_vert: str) -> None:
    """Rendert einen einzelnen Run: Overview-Plot, Jump-Scroller, Label-Tabelle."""
    run_data  = result["sessions"][sess_id]["runs"][run_id]
    df_imu    = run_data["df_imu"]
    raw_jumps = run_data["jumps"]
    jumps_df  = raw_jumps.copy() if raw_jumps is not None and not raw_jumps.empty else pd.DataFrame()

    label_key = f"labels_{key}_{sess_id}_{run_id}"
    if label_key not in st.session_state:
        st.session_state[label_key] = {}

    # Gespeicherte Labels in df übernehmen
    if not jumps_df.empty:
        for _, row in jumps_df.iterrows():
            jid = row["jump_id"]
            jumps_df.loc[jumps_df["jump_id"] == jid, "landing_type"] = \
                st.session_state[label_key].get(jid, row.get("landing_type", ""))

    # Run-Metriken
    run_meta = run_data.get("run_meta", {})
    summ = session_summary(jumps_df)
    col_m = st.columns(5)
    if run_meta:
        col_m[0].metric("Dauer", f"{run_meta.get('duration_s', '—')} s")
        col_m[1].metric("Höhendiff.", f"{run_meta.get('alt_drop_m', '—')} m")
    if summ:
        col_m[2].metric("Sprünge",      summ["Anzahl Sprünge"])
        col_m[3].metric("Max. Peak-g",  f"{summ['Max. Peak-g']:.2f} g")
        col_m[4].metric("Ø Peak-g",     f"{summ['Ø Peak-g']:.2f} g")

    # Overview-Plot
    title = (f"{meta.athlete_code if meta else key} | "
             f"{meta.position_label if meta else ''} | "
             f"Session {sess_id} | Run {run_id}")
    fig = _plot_run(df_imu, jumps_df if not jumps_df.empty else None, axis_vert, title=title)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    if jumps_df.empty:
        st.info("Keine Sprünge erkannt.")
        return

    # ── Horizontaler Jump-Scroller ────────────────────────────────────────
    st.markdown("**Sprünge — seitlich scrollen:**")
    _jump_scroller(df_imu, jumps_df, axis_vert, st.session_state[label_key])

    # ── Landungsart-Tabelle ───────────────────────────────────────────────
    comment_key = f"comments_{key}_{sess_id}_{run_id}"
    if comment_key not in st.session_state:
        st.session_state[comment_key] = {}

    # Bias-Korrektur Hinweis
    pos_lbl = meta.position_label if meta else ""
    from utils.jump_detector import _BIAS_CORRECTION
    if pos_lbl in _BIAS_CORRECTION:
        c = _BIAS_CORRECTION[pos_lbl]
        st.info(f"Bias-Korrektur aktiv ({pos_lbl}): Peak-g = {c['slope']:.3f} × IMU + {c['intercept']:.3f}  "
                f"(Validierung Kraftmessplatte, Magglingen 2026)")
    elif pos_lbl:
        st.warning(f"Keine Bias-Korrektur für {pos_lbl} — Korrelation mit Kraftmessplatte nicht signifikant (p>0.05)")

    st.markdown("**Landungsart zuweisen:**")
    hdr = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1, 2, 2, 3])
    for h, lbl in zip(hdr, ["Sprung", "Flugzeit (s)", "Peak (g)", "TTP (s)", "RFD (g/s)", "16g", "Rotation (est.)", "Landungsart", "Kommentar"]):
        h.write(f"**{lbl}**")

    for _, row in jumps_df.iterrows():
        jid = row["jump_id"]
        c0, c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1, 2, 2, 3])
        c0.write(jid)
        c1.write(f"{row['flight_time_s']:.3f}")
        c2.write(f"{'⚠️ ' if row.get('clipped_16g') else ''}{row['peak_res_g']:.2f}")
        c3.write(f"{row.get('time_to_peak_s', '—')}")
        c4.write(f"{row.get('rfd_g_per_s', '—')}")
        c5.write("⚠️" if row.get("clipped_16g") else "✓")
        rot = row.get("rotations")
        rot_dir = row.get("rotation_dir", "")
        c6.write(f"{rot}x {rot_dir}" if rot is not None else "—")
        saved_comment = st.session_state[comment_key].get(jid, "")
        new_comment = c8.text_input("", value=saved_comment, placeholder="Kommentar...",
                                     key=f"cmt_{key}_{sess_id}_{run_id}_{jid}",
                                     label_visibility="collapsed")
        if new_comment != saved_comment:
            st.session_state[comment_key][jid] = new_comment
        saved = st.session_state[label_key].get(jid, "")
        new_label = c7.selectbox(
            "", ["", "vorwärts", "switch"],
            index=["", "vorwärts", "switch"].index(saved) if saved in ["", "vorwärts", "switch"] else 0,
            key=f"lb_{key}_{sess_id}_{run_id}_{jid}",
            label_visibility="collapsed",
        )
        if new_label != saved:
            st.session_state[label_key][jid] = new_label
            result["sessions"][sess_id]["runs"][run_id]["jumps"].loc[
                result["sessions"][sess_id]["runs"][run_id]["jumps"]["jump_id"] == jid,
                "landing_type"
            ] = new_label

    # Export
    csv = jumps_df.to_csv(index=False).encode()
    st.download_button("CSV exportieren", csv,
                       file_name=f"{key}_s{sess_id}_r{run_id}_jumps.csv",
                       mime="text/csv", key=f"csv_{key}_{sess_id}_{run_id}")

    if "jump_results" not in st.session_state:
        st.session_state["jump_results"] = {}
    st.session_state["jump_results"][f"{key}_{sess_id}_{run_id}"] = {
        "jumps": jumps_df, "meta": meta,
    }


# Farbe pro Sensorposition
POSITION_COLORS = {
    "Körperschwerpunkt (Bauch)": "#1f77b4",   # blau
    "Fuss rechts":               "#2ca02c",   # grün
    "Fuss links":                "#ff7f0e",   # orange
}
_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#17becf"]


def _pos_color(position_label: str, idx: int = 0) -> str:
    return POSITION_COLORS.get(position_label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _plot_overlay(sensor_traces: list[dict]) -> go.Figure:
    """
    Overlay-Plot mehrerer Sensoren auf gemeinsamer Zeitachse.
    sensor_traces: [{"df": df_imu, "label": str, "color": str}, ...]
    Zeitachse: absolute POSIX-Sekunden, normiert auf frühsten Zeitpunkt.
    """
    t_col = "imuTimestamp [us]"
    acc_res_col_opts = ["accRes [g]", "acc_norm_g"]

    # Gemeinsamer Nullpunkt = frühster Timestamp über alle Sensoren
    t0_global = None
    for tr in sensor_traces:
        df = tr["df"]
        if t_col in df.columns and len(df) > 0:
            t0 = df[t_col].iloc[0]
            if t0_global is None or t0 < t0_global:
                t0_global = t0

    fig = go.Figure()
    for tr in sensor_traces:
        df    = tr["df"]
        label = tr["label"]
        color = tr["color"]
        acc_col = next((c for c in acc_res_col_opts if c in df.columns), None)
        if acc_col is None:
            continue
        if t_col in df.columns and t0_global is not None:
            t = (df[t_col].values - t0_global) / 1e6
        else:
            t = np.arange(len(df)) / 200.0

        fig.add_trace(go.Scatter(
            x=t, y=df[acc_col].values,
            mode="lines", line=dict(color=color, width=0.9),
            name=label,
        ))

    fig.add_hline(y=16, line_dash="dot", line_color="red",
                  annotation_text="16g", annotation_position="top right")
    fig.update_layout(
        height=380, template="plotly_white",
        xaxis_title="Zeit (s)", yaxis_title="accRes [g]",
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=30),
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04))
    return fig


def show():
    st.header("Sprunganalyse")

    sessions_loaded = st.session_state.get("loaded_sessions", {})
    if not sessions_loaded:
        st.warning("Zuerst Daten laden (Tab 'Daten laden').")
        return

    # ── Athleten-Auswahl ─────────────────────────────────────────────────
    # Alle verfügbaren Athleten ermitteln
    athlete_groups: dict[str, list[str]] = {}
    group_sort: dict[str, str] = {}
    for k, s in sessions_loaded.items():
        m = s.get("meta")
        if m:
            try:
                date_fmt = pd.to_datetime(str(m.date), format="%Y%m%d").strftime("%d.%m.%Y")
            except Exception:
                date_fmt = m.date
            gkey = f"{m.athlete_code}  |  {date_fmt}  |  {m.location}"
            group_sort[gkey] = m.date  # für Sortierung nach Datum
        else:
            gkey = k
            group_sort[gkey] = ""
        athlete_groups.setdefault(gkey, []).append(k)

    group_names = sorted(athlete_groups.keys(), key=lambda g: group_sort.get(g, ""), reverse=True)
    sel_groups = st.multiselect("Athleten auswählen", group_names, default=group_names[:1])
    if not sel_groups:
        st.info("Mindestens einen Athleten auswählen.")
        return

    # Alle Sensor-Keys der gewählten Athleten sammeln
    group_keys = [k for g in sel_groups for k in athlete_groups[g]]

    # ── Positions-Filter ─────────────────────────────────────────────────
    all_positions = sorted({
        sessions_loaded[k].get("meta").position_label
        for k in group_keys
        if sessions_loaded[k].get("meta")
    })
    if len(all_positions) > 1:
        sel_positions = st.multiselect("Sensorposition", all_positions, default=all_positions)
        if not sel_positions:
            sel_positions = all_positions
        group_keys = [k for k in group_keys
                      if sessions_loaded[k].get("meta") and
                      sessions_loaded[k]["meta"].position_label in sel_positions]

    # ── Sensor-Auswahl ───────────────────────────────────────────────────
    def _pos_label(k):
        m = sessions_loaded[k].get("meta")
        if m:
            return f"{m.athlete_code} | {m.position_label}"
        return k

    sel_keys = st.multiselect(
        "Sensoren wählen",
        options=group_keys,
        default=group_keys,
        format_func=_pos_label,
    )
    if not sel_keys:
        st.info("Mindestens einen Sensor auswählen.")
        return

    with st.expander("Erkennungs-Parameter anpassen"):
        c1, c2, c3, c4 = st.columns(4)
        th_vert  = c1.slider("G-Schwelle vertikal (g)",     0.3, 0.9, 0.60, 0.05)
        th_res   = c2.slider("G-Schwelle resultierend (g)", 0.3, 0.9, 0.70, 0.05)
        th_cross = c3.slider("G-Schwelle Crossing (g)",     0.5, 2.0, 1.0,  0.1)
        min_dur  = c4.slider("Minimale Flugzeit (s)",        0.5, 3.0, 1.2,  0.1)

    # Fixe Parameter
    v_start          = 7.0
    v_hold           = 1.5
    alt_rise_end     = 20.0
    alt_drop_min     = 30.0
    run_duration_min = 15.0

    params = dict(th_vert=th_vert, th_res=th_res, th_cross=th_cross, min_dur=min_dur,
                  v_start=v_start, v_hold=v_hold, alt_rise_end=alt_rise_end,
                  alt_drop_min=alt_drop_min, run_duration_min=run_duration_min)

    # ── Pipeline für jeden gewählten Sensor ──────────────────────────────
    results: dict[str, dict] = {}
    for key in sel_keys:
        sess = sessions_loaded[key]
        pos_lbl = sess.get("meta").position_label if sess.get("meta") else ""
        cache_key = f"pipeline_v5_{key}_{'_'.join(str(v) for v in params.values())}"
        if cache_key not in st.session_state:
            raw_df = _get_raw_df(sess)
            if raw_df is None:
                st.error(f"IMU-Daten konnten nicht geladen werden: {key}")
                continue
            with st.spinner(f"Pipeline: {_pos_label(key)} …"):
                try:
                    st.session_state[cache_key] = _run_pipeline(raw_df, sess.get("gnss"), params,
                                                                 position_label=pos_lbl)
                except Exception as e:
                    st.error(f"Pipeline-Fehler ({key}): {e}")
                    continue
        results[key] = st.session_state[cache_key]

    if not results:
        return

    # ── Overlay-Plot ──────────────────────────────────────────────────────
    if len(sel_keys) > 1:
        st.subheader("Overlay — alle gewählten Sensoren")
        traces = []
        for i, key in enumerate(sel_keys):
            meta   = sessions_loaded[key].get("meta")
            color  = _pos_color(meta.position_label if meta else "", i)
            result = results[key]
            # Ersten Session-DataFrame verwenden
            first_sess = next(iter(result["sessions"].values()), {})
            df_raw = first_sess.get("df")
            if df_raw is not None:
                # Downsample auf max. 5000 Punkte für den Overlay-Plot
                step = max(1, len(df_raw) // 5000)
                traces.append({"df": df_raw.iloc[::step].reset_index(drop=True),
                               "label": _pos_label(key), "color": color})
        if traces:
            st.plotly_chart(_plot_overlay(traces), use_container_width=True,
                            config={"scrollZoom": True})

    # ── Pro-Sensor-Blöcke ─────────────────────────────────────────────────
    for i, key in enumerate(sel_keys):
        meta      = sessions_loaded[key].get("meta")
        result    = results[key]
        axis_vert = result["axis_vert"]
        color     = _pos_color(meta.position_label if meta else "", i)
        sessions_dict = result["sessions"]

        pos_label = _pos_label(key)
        # Farbiger Trennbalken per Position
        st.markdown(
            f'<div style="border-left:5px solid {color};padding-left:10px;margin:18px 0 6px 0;">'
            f'<b>{pos_label}</b>'
            + (f' — Sensor {meta.sensor_id} | {meta.location}' if meta else '')
            + '</div>',
            unsafe_allow_html=True,
        )

        if not sessions_dict:
            st.warning("Keine Sessions erkannt.")
            continue

        session_ids = list(sessions_dict.keys())
        # Session-Auswahl nur wenn mehrere vorhanden
        if len(session_ids) > 1:
            sel_session = st.selectbox(
                f"Session ({len(session_ids)} erkannt)",
                session_ids,
                format_func=lambda s: f"Session {s}",
                key=f"sess_{key}",
            )
        else:
            sel_session = session_ids[0]

        run_ids = list(sessions_dict[sel_session]["runs"].keys())
        if not run_ids:
            st.warning("Keine Runs erkannt.")
            continue

        cache_key = f"pipeline_v5_{key}_{'_'.join(str(v) for v in params.values())}"

        # ── Session-Overview mit Run-Markierungen (IMU + GNSS sync) ─────
        session_df = sessions_dict[sel_session].get("df")
        gnss_sess  = sessions_dict[sel_session].get("gnss")
        runs_with_meta = [
            (rid, sessions_dict[sel_session]["runs"][rid].get("run_meta", {}))
            for rid in run_ids
            if sessions_dict[sel_session]["runs"][rid].get("run_meta")
        ]
        if session_df is not None:
            imu_t_col    = "imuTimestamp [us]"
            gnss_t_col   = "timestamp [us]"
            acc_res_col  = "accRes [g]" if "accRes [g]" in session_df.columns else "acc_norm_g"

            if imu_t_col in session_df.columns and acc_res_col in session_df.columns:
                # Gemeinsamer Nullpunkt: frühester Timestamp über beide Sensoren
                t0 = session_df[imu_t_col].values[0]
                if gnss_sess is not None and gnss_t_col in gnss_sess.columns:
                    t0 = min(t0, gnss_sess[gnss_t_col].values[0])

                t_imu = (session_df[imu_t_col].values - t0) / 1e6

                has_gnss = (gnss_sess is not None and gnss_t_col in gnss_sess.columns
                            and "speedRes [m/s]" in gnss_sess.columns)
                n_rows = 3 if (has_gnss and "altitude [m]" in gnss_sess.columns) else (2 if has_gnss else 1)
                row_h  = [0.5, 0.3, 0.2][:n_rows]

                subplot_titles = ["accRes [g]"]
                if has_gnss:
                    subplot_titles.append("Geschwindigkeit (m/s)")
                if has_gnss and "altitude [m]" in gnss_sess.columns:
                    subplot_titles.append("Höhe (m)")

                from plotly.subplots import make_subplots as _make_subplots
                fig_ov = _make_subplots(
                    rows=n_rows, cols=1, shared_xaxes=True,
                    row_heights=row_h, subplot_titles=subplot_titles,
                    vertical_spacing=0.06,
                )

                fig_ov.add_trace(go.Scatter(
                    x=t_imu, y=session_df[acc_res_col].values,
                    line=dict(color="#555", width=0.7), name="accRes [g]", showlegend=False,
                ), row=1, col=1)
                fig_ov.add_hline(y=16, line_dash="dot", line_color="red",
                                 annotation_text="16g", row=1, col=1)

                if has_gnss:
                    t_gnss = (gnss_sess[gnss_t_col].values - t0) / 1e6
                    fig_ov.add_trace(go.Scatter(
                        x=t_gnss, y=gnss_sess["speedRes [m/s]"].values,
                        line=dict(color="#1f77b4", width=1.0), name="Speed", showlegend=False,
                    ), row=2, col=1)
                    if "altitude [m]" in gnss_sess.columns:
                        fig_ov.add_trace(go.Scatter(
                            x=t_gnss, y=gnss_sess["altitude [m]"].values,
                            line=dict(color="#2ca02c", width=1.0), name="Höhe", showlegend=False,
                            fill="tozeroy", fillcolor="rgba(44,160,44,0.08)",
                        ), row=3, col=1)

                # Run-Markierungen in alle Subplots
                run_colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd",
                              "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
                for r_idx, (rid, run_meta) in enumerate(runs_with_meta):
                    rcolor   = run_colors[r_idx % len(run_colors)]
                    start_us = run_meta.get("start_time_us")
                    end_us   = run_meta.get("end_time_us")
                    if start_us is None or end_us is None:
                        continue
                    t_start = (float(start_us) - t0) / 1e6
                    t_end   = (float(end_us)   - t0) / 1e6
                    for row_n in range(1, n_rows + 1):
                        fig_ov.add_vrect(
                            x0=t_start, x1=t_end,
                            fillcolor=rcolor, opacity=0.10, line_width=0,
                            row=row_n, col=1,
                        )
                    fig_ov.add_vline(
                        x=t_start, line_dash="dash", line_color=rcolor, line_width=1.5,
                        annotation_text=f"▶ {rid}", annotation_position="top left",
                        annotation_font=dict(size=10, color=rcolor),
                        row=1, col=1,
                    )

                fig_ov.update_layout(
                    title=f"Session {sel_session} — {len(runs_with_meta)} Run(s)"
                          + (" | IMU + GNSS synchronisiert" if has_gnss else ""),
                    height=200 + 150 * n_rows, template="plotly_white",
                    margin=dict(t=50, b=30),
                )
                fig_ov.update_xaxes(
                    title_text="Zeit (s)", row=n_rows, col=1,
                )
                fig_ov.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=n_rows, col=1)
                st.plotly_chart(fig_ov, use_container_width=True, config={"scrollZoom": True})

        st.markdown(f"{len(run_ids)} Run(s) in Session {sel_session}:")

        for run_id in run_ids:
            run_data   = sessions_dict[sel_session]["runs"][run_id]
            run_meta   = run_data.get("run_meta", {})
            raw_jumps  = run_data.get("jumps")
            n_jumps    = len(raw_jumps) if raw_jumps is not None and not raw_jumps.empty else 0
            dur        = run_meta.get("duration_s", "—")
            alt        = run_meta.get("alt_drop_m", "—")
            start_time = _format_run_time(run_meta)
            time_str   = f"  🕐 {start_time}" if start_time else ""
            exp_label  = (f"Run {run_id}{time_str}  —  {n_jumps} Jump{'s' if n_jumps != 1 else ''}  "
                          + (f"| {dur} s  | Δ{alt} m" if run_meta else ""))

            with st.expander(exp_label, expanded=(len(run_ids) == 1)):
                _render_run(cache_key, sel_session, run_id, result, key, meta, axis_vert)
