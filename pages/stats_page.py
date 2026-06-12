"""
Sprungauswertung: Alle Sprünge + optionaler Vorwärts vs. Switch Vergleich
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.statistics import compare_groups

PARAMS = {
    "peak_res_g":      "Peak-g (g)",
    "flight_time_s":   "Flugzeit (s)",
    "time_to_peak_s":  "Time to Peak (s)",
    "rfd_g_per_s":     "RFD (g/s)",
    "impulse_net_g_s": "Impuls netto (g·s)",
}


def _boxplot(data_dict: dict, param: str, title: str) -> go.Figure:
    """Boxplot mit einem Trace pro Gruppe."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig = go.Figure()
    for i, (name, vals) in enumerate(data_dict.items()):
        if len(vals) == 0:
            continue
        fig.add_trace(go.Box(
            y=vals, name=name,
            marker_color=colors[i % len(colors)],
            boxpoints="all", jitter=0.3, pointpos=-1.5,
        ))
    fig.update_layout(title=title, yaxis_title=PARAMS.get(param, param),
                      template="plotly_white", height=380)
    return fig


def show():
    results = st.session_state.get("jump_results", {})
    if not results:
        return

    all_jumps = pd.concat(
        [r["jumps"] for r in results.values()
         if r.get("jumps") is not None and not r["jumps"].empty],
        ignore_index=True,
    )
    if all_jumps.empty:
        return

    st.subheader("Sprungauswertung")

    # ── Alle Sprünge zusammen ──────────────────────────────────────────────
    avail_params = [p for p in PARAMS if p in all_jumps.columns]
    n_total = len(all_jumps)
    n_clipped = int(all_jumps.get("clipped_16g", pd.Series([False]*n_total)).sum())

    st.caption(f"Gesamt: **{n_total}** Sprünge | Geclippt (16g): **{n_clipped}**")

    # Zusammenfassungstabelle
    summary_rows = []
    for param in avail_params:
        vals = all_jumps[param].dropna().values
        if len(vals) == 0:
            continue
        summary_rows.append({
            "Parameter": PARAMS[param],
            "n": len(vals),
            "Mittelwert": f"{np.mean(vals):.3f}",
            "SD": f"{np.std(vals, ddof=1):.3f}",
            "Min": f"{np.min(vals):.3f}",
            "Median": f"{np.median(vals):.3f}",
            "Max": f"{np.max(vals):.3f}",
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Boxplot alle Sprünge
    sel_param_all = st.selectbox(
        "Boxplot Parameter",
        avail_params,
        format_func=lambda k: PARAMS[k],
        key="stats_param_all",
    )
    fig_all = _boxplot({"Alle Sprünge": all_jumps[sel_param_all].dropna().values},
                       sel_param_all, PARAMS[sel_param_all])
    st.plotly_chart(fig_all, use_container_width=True)

    # ── Vorwärts vs. Switch (nur wenn Labels vorhanden) ───────────────────
    labeled = all_jumps[all_jumps["landing_type"].isin(["vorwärts", "switch"])]
    if labeled.empty:
        st.info("Noch keine Sprünge als 'vorwärts' oder 'switch' labeliert — "
                "Labels in der Sprunganalyse vergeben für den Gruppenvergleich.")
        return

    st.divider()
    st.subheader("Vorwärts vs. Switch")

    fwd = labeled[labeled["landing_type"] == "vorwärts"]
    swt = labeled[labeled["landing_type"] == "switch"]
    st.caption(f"Vorwärts: {len(fwd)} | Switch: {len(swt)}")

    if len(fwd) < 2 or len(swt) < 2:
        st.warning("Zu wenig gelabelte Sprünge für Gruppenvergleich (mind. 2 pro Gruppe).")
        return

    stat_rows = []
    for param in avail_params:
        a = fwd[param].dropna().values
        b = swt[param].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        res = compare_groups(a, b, "vorwärts", "switch")
        stat_rows.append({
            "Parameter": PARAMS[param],
            "Ø vorwärts": f"{res['mean_vorwärts']:.3f}",
            "SD vorwärts": f"{res['std_vorwärts']:.3f}",
            "Ø switch": f"{res['mean_switch']:.3f}",
            "SD switch": f"{res['std_switch']:.3f}",
            "Test": res["test"],
            "p-Wert": f"{res['p_value']:.4f}",
            "Cohen's d": f"{res['cohens_d']:.3f}",
            "Signifikant": "✓" if res["p_value"] < 0.05 else "",
        })
    if stat_rows:
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    sel_param_grp = st.selectbox(
        "Boxplot Parameter",
        avail_params,
        format_func=lambda k: PARAMS[k],
        key="stats_param_grp",
    )
    fig_grp = _boxplot(
        {
            "vorwärts": fwd[sel_param_grp].dropna().values,
            "switch": swt[sel_param_grp].dropna().values,
        },
        sel_param_grp,
        f"{PARAMS[sel_param_grp]}: Vorwärts vs. Switch",
    )
    st.plotly_chart(fig_grp, use_container_width=True)
