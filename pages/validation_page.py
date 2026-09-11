import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr

VAL_DIR = (
    "/Users/cyrilwyss/AAMasterarbeit/zz_Data Claude/Datenfiles Validierung/"
    "Validierungsmessungen/Datasheets Validierungsmessungen neu (Nils & Cyril)"
)
_VAL_DIR_EXISTS = os.path.exists(VAL_DIR)

SENSORS = {
    "Bauch (b1)":       "peak_res_g_b1",
    "Bauch (b2)":       "peak_res_g_b2",
    "Fuss links (li1)": "peak_res_g_li1",
    "Fuss links (li2)": "peak_res_g_li2",
    "Fuss rechts (re1)":"peak_res_g_re1",
    "Fuss rechts (re2)":"peak_res_g_re2",
}

_BIAS_CORRECTION_REFERENCE = {
    "Bauch": {"slope": 0.4323, "intercept": 3.9842},
}


@st.cache_data
def _load_from_paths() -> pd.DataFrame:
    path_c = os.path.join(VAL_DIR, "Validierung Cyril.xlsx")
    path_n = os.path.join(VAL_DIR, "Validierung Nils.xlsx")
    df_c = pd.read_excel(path_c, sheet_name="Validierung")
    if "athlete_id" not in df_c.columns:
        df_c["athlete_id"] = "Cyril"
    df_n = pd.read_excel(path_n, sheet_name="Validierung")
    if "athlete_id" not in df_n.columns:
        df_n["athlete_id"] = "Nils"
    if "peak_res_g_KMP" not in df_n.columns or df_n["peak_res_g_KMP"].isna().all():
        if "peak_landing_F" in df_n.columns and "body_mass" in df_n.columns:
            df_n["peak_res_g_KMP"] = df_n["peak_landing_F"] / (df_n["body_mass"] * 9.81)
    df = pd.concat([df_c, df_n], ignore_index=True)
    df = df[df["exercise"].notna()].reset_index(drop=True)
    return df


def _load_from_upload(files) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name="Validierung")
            name = f.name.lower()
            if "nils" in name:
                if "athlete_id" not in df.columns:
                    df["athlete_id"] = "Nils"
                if "peak_res_g_KMP" not in df.columns or df["peak_res_g_KMP"].isna().all():
                    if "peak_landing_F" in df.columns and "body_mass" in df.columns:
                        df["peak_res_g_KMP"] = df["peak_landing_F"] / (df["body_mass"] * 9.81)
            elif "cyril" in name:
                if "athlete_id" not in df.columns:
                    df["athlete_id"] = "Cyril"
            dfs.append(df)
        except Exception as e:
            st.warning(f"Fehler beim Lesen von {f.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    if "exercise" in combined.columns:
        combined = combined[combined["exercise"].notna()].reset_index(drop=True)
    return combined


def _scatter_with_regression(x, y, colors, title):
    r_val, p_val = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.75),
        name="Messpunkte",
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#555555", width=2, dash="dash"),
        name="Regressionsgerade",
    ))
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
    fig.update_layout(
        title=f"{title}   r = {r_val:.3f}, p = {p_str}",
        xaxis_title="Peak KMP (g)",
        yaxis_title="Peak IMU (g)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig, r_val, p_val, slope, intercept


def _bland_altman_chart(x, y):
    mean_vals = (x + y) / 2.0
    diff_vals = y - x
    bias = float(np.mean(diff_vals))
    sd = float(np.std(diff_vals, ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    cv_pct = sd / float(np.mean(x)) * 100 if np.mean(x) != 0 else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_vals, y=diff_vals,
        mode="markers",
        marker=dict(size=7, opacity=0.75),
        name="Differenz",
    ))
    fig.add_hline(y=bias, line_color="#1a73e8", line_width=2,
                  annotation_text=f"Bias: {bias:.3f} g", annotation_position="top right")
    fig.add_hline(y=loa_upper, line_color="#e84040", line_width=1.5, line_dash="dash",
                  annotation_text=f"+1.96 SD: {loa_upper:.3f} g", annotation_position="top right")
    fig.add_hline(y=loa_lower, line_color="#e84040", line_width=1.5, line_dash="dash",
                  annotation_text=f"-1.96 SD: {loa_lower:.3f} g", annotation_position="bottom right")
    fig.update_layout(
        title="Bland-Altman (IMU - KMP)",
        xaxis_title="Mittelwert KMP und IMU (g)",
        yaxis_title="Differenz IMU - KMP (g)",
        margin=dict(t=50, b=40),
    )
    return fig, bias, loa_upper, loa_lower, cv_pct


def _color_by_athlete(df_sub):
    palette = {"Cyril": "#1a73e8", "Nils": "#e84040"}
    if "athlete_id" in df_sub.columns:
        return [palette.get(a, "#888888") for a in df_sub["athlete_id"]]
    return ["#888888"] * len(df_sub)


def _render_sensor_tab(df_filtered, sensor_name, imu_col):
    ref_col = "peak_res_g_KMP"
    sub = df_filtered[[ref_col, imu_col] +
                       ([c for c in ["athlete_id", "exercise"] if c in df_filtered.columns])
                      ].dropna(subset=[ref_col, imu_col]).reset_index(drop=True)

    if len(sub) < 4:
        st.info(f"Zu wenig Datenpunkte (n = {len(sub)}) fuer {sensor_name}.")
        return None

    x = sub[ref_col].values.astype(float)
    y = sub[imu_col].values.astype(float)
    colors = _color_by_athlete(sub)

    with st.expander("Rohdaten", expanded=False):
        raw_df = pd.DataFrame({
            "Athlet": sub["athlete_id"] if "athlete_id" in sub.columns else ["—"] * len(sub),
            "Uebung": sub["exercise"] if "exercise" in sub.columns else ["—"] * len(sub),
            "Trial": sub.index + 1,
            "Peak KMP (g)": np.round(x, 4),
            "Peak IMU (g)": np.round(y, 4),
            "Differenz (g)": np.round(y - x, 4),
        })
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    st.subheader("Streudiagramm")
    fig_scatter, r_val, p_val, slope, intercept = _scatter_with_regression(
        x, y, colors, f"{sensor_name}: IMU vs. KMP"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Bland-Altman")
    fig_ba, bias, loa_upper, loa_lower, cv_pct = _bland_altman_chart(x, y)
    st.plotly_chart(fig_ba, use_container_width=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bias (g)", f"{bias:.3f}")
    m2.metric("+LoA (g)", f"{loa_upper:.3f}")
    m3.metric("-LoA (g)", f"{loa_lower:.3f}")
    m4.metric("CV%", f"{cv_pct:.1f}")

    st.subheader("Bias-Korrektur (lineare Regression)")
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
    st.code(
        f"Korrekturformel: peak_g_korrigiert = {slope:.4f} x peak_g_IMU + {intercept:.4f}",
        language="python",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r", f"{r_val:.4f}")
    c2.metric("p", p_str)
    c3.metric("slope", f"{slope:.4f}")
    c4.metric("intercept", f"{intercept:.4f}")

    if "bias_corrections" not in st.session_state:
        st.session_state["bias_corrections"] = {}

    if st.button(f"Diese Korrektur uebernehmen ({sensor_name})", key=f"btn_corr_{imu_col}"):
        st.session_state["bias_corrections"][sensor_name] = {
            "slope": float(slope),
            "intercept": float(intercept),
        }
        st.success(
            f"Korrektur fuer {sensor_name} gespeichert: "
            f"slope = {slope:.4f}, intercept = {intercept:.4f}"
        )

    with st.expander("Aktuell in jump_detector.py hinterlegte Korrekturen (Referenz)", expanded=False):
        ref_rows = []
        for k, v in _BIAS_CORRECTION_REFERENCE.items():
            ref_rows.append({"Sensor": k, "slope": v["slope"], "intercept": v["intercept"]})
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

    return {
        "Sensor": sensor_name,
        "n": len(sub),
        "Bias (g)": round(bias, 4),
        "+LoA": round(loa_upper, 4),
        "-LoA": round(loa_lower, 4),
        "CV%": round(cv_pct, 2),
        "r": round(r_val, 4),
        "p": round(p_val, 4),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
    }


def show():
    st.header("Validierung")
    st.caption("Kraftmessplatte (Referenz) vs. IMU-Sensoren")

    if _VAL_DIR_EXISTS:
        try:
            df_all = _load_from_paths()
        except Exception as e:
            st.error(f"Fehler beim Laden der lokalen Daten: {e}")
            return
    else:
        st.info(
            "Lokale Validierungsdaten nicht gefunden. "
            "Bitte 'Validierung Cyril.xlsx' und/oder 'Validierung Nils.xlsx' hochladen "
            "(Sheet-Name: 'Validierung')."
        )
        uploaded = st.file_uploader(
            "Validierungs-Excel hochladen",
            type=["xlsx"],
            accept_multiple_files=True,
            key="val_upload",
        )
        if not uploaded:
            return
        df_all = _load_from_upload(uploaded)
        if df_all.empty:
            st.warning("Keine auswertbaren Daten in den hochgeladenen Dateien.")
            return

    col_a, col_b = st.columns(2)
    athletes_avail = sorted(df_all["athlete_id"].dropna().unique().tolist()) if "athlete_id" in df_all.columns else []
    exercises_avail = sorted(df_all["exercise"].dropna().unique().tolist()) if "exercise" in df_all.columns else []

    sel_athletes = col_a.multiselect(
        "Athlet", options=["Alle"] + athletes_avail, default=["Alle"]
    )
    sel_exercises = col_b.multiselect(
        "Uebung", options=["Alle"] + exercises_avail, default=["Alle"]
    )

    df = df_all.copy()
    if "Alle" not in sel_athletes and sel_athletes:
        df = df[df["athlete_id"].isin(sel_athletes)]
    if "Alle" not in sel_exercises and sel_exercises:
        df = df[df["exercise"].isin(sel_exercises)]

    tab_labels = list(SENSORS.keys()) + ["Ueberblick"]
    tabs = st.tabs(tab_labels)

    overview_rows = []
    for tab, (sensor_name, imu_col) in zip(tabs[:-1], SENSORS.items()):
        with tab:
            row = _render_sensor_tab(df, sensor_name, imu_col)
            if row is not None:
                overview_rows.append(row)

    with tabs[-1]:
        st.subheader("Alle Sensoren im Ueberblick")
        if overview_rows:
            order = list(SENSORS.keys())
            ov_df = pd.DataFrame(overview_rows)
            ov_df["_sort"] = ov_df["Sensor"].apply(
                lambda s: order.index(s) if s in order else 999
            )
            ov_df = ov_df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
            st.dataframe(ov_df, use_container_width=True, hide_index=True)
        else:
            st.info("Noch keine Daten fuer den Ueberblick verfuegbar.")
