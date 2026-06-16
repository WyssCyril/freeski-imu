"""
Sprunganalyse — verwendet die originale sensor_lib Pipeline.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import sensor_lib

SAMPLE_RATE = 200.0


def detect_vertical_axis(df: pd.DataFrame) -> str:
    """Bestimmt die vertikale Achse (accX oder accY) anhand des Medians."""
    med_x = abs(df['accX [g]'].median())
    med_y = abs(df['accY [g]'].median())
    return 'accX [g]' if med_x > med_y else 'accY [g]'


def preprocess_imu(df: pd.DataFrame) -> pd.DataFrame:
    """Dreht Sensor falls nötig und fügt accRes hinzu. Erwartet originale AdMos-Spaltennamen."""
    df = df.copy()
    axis_vert = detect_vertical_axis(df)
    if df[axis_vert].median() < 0:
        df = sensor_lib.rotate_sensor_data(df, 'z')
    df = sensor_lib.add_resultant_acc(df)
    return df


def detect_jumps(df_raw: pd.DataFrame,
                 th_core_accVert: float = 0.60,
                 th_core_accRes: float = 0.70,
                 th_crossing: float = 1.0,
                 min_duration_s: float = 1.2,
                 window_size: int = 10) -> pd.DataFrame:
    """
    Erkennt Sprünge mit sensor_lib.detect_jumps_snow.
    Erwartet DataFrame mit originalen AdMos-Spaltennamen oder preprocessed df (mit accRes [g]).
    """
    df = df_raw.copy()

    if "accRes [g]" not in df.columns:
        df = preprocess_imu(df)

    axis_vert = detect_vertical_axis(df)

    return sensor_lib.detect_jumps_snow(
        df,
        axis_vert=axis_vert,
        th_core_accVert=th_core_accVert,
        th_core_accRes=th_core_accRes,
        th_crossing=th_crossing,
        min_duration_s=min_duration_s,
        window_size=window_size,
    )


def compute_rotation(df_raw: pd.DataFrame, takeoff_idx: int, landing_idx: int) -> dict:
    """
    Schätzt Rotationen während der Flugphase aus Gyroskopdaten.
    Methode: Integration der resultierenden Winkelgeschwindigkeit (dps → Grad total).
    Richtung: Vorzeichen der dominanten Achse (gyrZ für Bauch-Sensor ≈ Yaw).
    """
    gyr_cols = [c for c in ["gyrX [dps]", "gyrY [dps]", "gyrZ [dps]"] if c in df_raw.columns]
    if not gyr_cols or landing_idx <= takeoff_idx:
        return {"rotations": None, "direction": None, "total_deg": None}

    i0 = max(0, takeoff_idx)
    i1 = min(len(df_raw) - 1, landing_idx)
    seg = df_raw.iloc[i0:i1 + 1]

    dt = 1.0 / SAMPLE_RATE

    # Gesamtrotation = Integral des Betrags (alle Achsen)
    gyr_res = np.sqrt(sum(seg[c].values ** 2 for c in gyr_cols))
    total_deg = float(np.sum(gyr_res) * dt)
    rotations = round(total_deg / 360, 1)

    # Richtung: dominante Achse mit höchstem integriertem Absolutwert
    # Für Bauch-Sensor ≈ gyrZ ist Yaw (Körperdrehung links/rechts)
    # Positive Werte = links, negative = rechts (AdMos-Konvention)
    dominant_col = max(gyr_cols, key=lambda c: abs(float(np.sum(seg[c].values) * dt)))
    net_angle = float(np.sum(seg[dominant_col].values) * dt)
    direction = "links" if net_angle > 0 else "rechts"

    return {
        "rotations": rotations,
        "direction": direction,
        "total_deg": round(total_deg, 1),
    }


def compute_landing_params(df_raw: pd.DataFrame, jumps_df: pd.DataFrame) -> pd.DataFrame:
    """Fügt Time to Peak, RFD und Impuls zu jedem Sprung hinzu."""
    df = df_raw.copy()
    if "accRes [g]" not in df.columns:
        df = preprocess_imu(df)

    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    rows = []

    for _, row in jumps_df.iterrows():
        landing_idx = int(row["landing_idx"])
        peak_idx = int(row["peak_res_idx"])

        delta_t = (peak_idx - landing_idx) / SAMPLE_RATE
        acc_at_land = float(df["accRes [g]"].iloc[landing_idx])
        peak_g = float(row["peak_res_g"])

        rfd = (peak_g - acc_at_land) / delta_t if delta_t > 0 else 0.0
        impulse = float(trapz(df["accRes [g]"].iloc[landing_idx:peak_idx + 1], dx=1 / SAMPLE_RATE))
        impulse_net = float(trapz(df["accRes [g]"].iloc[landing_idx:peak_idx + 1] - 1.0, dx=1 / SAMPLE_RATE))
        clipped = bool((df["accRes [g]"].iloc[landing_idx:peak_idx + 1] >= 15.5).any())

        rot = compute_rotation(df, int(row["takeoff_idx"]), landing_idx)

        rows.append({
            **row.to_dict(),
            "time_to_peak_s": round(delta_t, 4),
            "rfd_g_per_s": round(rfd, 2),
            "impulse_g_s": round(impulse, 4),
            "impulse_net_g_s": round(impulse_net, 4),
            "clipped_16g": clipped,
            "landing_type": row.get("landing_type", ""),
            "rotations": rot["rotations"],
            "rotation_dir": rot["direction"],
            "rotation_deg": rot["total_deg"],
        })

    return pd.DataFrame(rows)


def session_summary(jumps_df: pd.DataFrame) -> dict:
    if jumps_df is None or jumps_df.empty:
        return {}
    peaks = jumps_df["peak_res_g"].values
    flights = jumps_df["flight_time_s"].values
    clipped = int(jumps_df.get("clipped_16g", pd.Series([False] * len(jumps_df))).sum())
    return {
        "Anzahl Sprünge": len(jumps_df),
        "Max. Flugzeit (s)": round(float(np.max(flights)), 3),
        "Ø Flugzeit (s)": round(float(np.mean(flights)), 3),
        "Max. Peak-g": round(float(np.max(peaks)), 2),
        "Ø Peak-g": round(float(np.mean(peaks)), 2),
        "Geclippt (16g)": clipped,
    }
