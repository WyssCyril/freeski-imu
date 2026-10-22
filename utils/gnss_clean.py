"""
Ausreisser-Filter für GPS-Tracks.

Entfernt Punkte, die physikalisch nicht möglich sind:
1. weiter als MAX_DIST_M vom Median-Ort des Tracks (z. B. alte Startposition des Sensors),
2. gemeldete Geschwindigkeit > MAX_SPEED_KMH,
3. Positionssprung zum letzten gültigen Punkt mit > MAX_SPEED_KMH,
4. Höhe weicht mehr als MAX_ALT_RANGE_M vom Median des Tracks oder mehr als MAX_ALT_DEV_M vom gleitenden Median ab.
"""
import numpy as np
import pandas as pd

MAX_SPEED_KMH = 100.0
MAX_DIST_M = 5000.0
MAX_ALT_DEV_M = 30.0
MAX_ALT_RANGE_M = 800.0   # Höhe max. so weit vom Median des Tracks entfernt


def clean_mask(lat, lon, t_ms, alt=None, speed_kmh=None,
               max_speed_kmh: float = MAX_SPEED_KMH) -> np.ndarray:
    """True = Punkt behalten. Erwartet zeitlich sortierte Arrays (t_ms in Millisekunden)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); t = np.asarray(t_ms, float)
    n = len(lat)
    keep = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(t) & (np.abs(lat) > 1) & (np.abs(lon) > 1)
    if keep.sum() < 3:
        return keep
    lat0, lon0 = np.median(lat[keep]), np.median(lon[keep])
    kx = 111320 * np.cos(np.radians(lat0)); ky = 110540
    x, y = (lon - lon0) * kx, (lat - lat0) * ky
    keep &= np.hypot(x, y) <= MAX_DIST_M
    if speed_kmh is not None:
        sp = np.asarray(speed_kmh, float)
        keep &= ~(np.isfinite(sp) & (sp > max_speed_kmh))
    # Positionssprünge: Vergleich mit dem letzten behaltenen Punkt
    last = None
    for i in range(n):
        if not keep[i]:
            continue
        if last is not None:
            dt = (t[i] - t[last]) / 1000.0
            if dt <= 0 or np.hypot(x[i] - x[last], y[i] - y[last]) / dt * 3.6 > max_speed_kmh:
                keep[i] = False
                continue
        last = i
    if alt is not None:
        a = pd.Series(np.asarray(alt, float))
        if keep.any():
            keep &= ~((a - np.nanmedian(a[keep])).abs() > MAX_ALT_RANGE_M).fillna(True).values
        med = a.where(keep).rolling(31, center=True, min_periods=5).median()
        keep &= ~((a - med).abs() > MAX_ALT_DEV_M).fillna(False).values
    return keep


def clean_df(df: pd.DataFrame, lat="Lat", lon="Lon", t="Zeit (POSIX ms)", alt="Höhe (m)",
             speed="Speed (km/h)", group=None) -> pd.DataFrame:
    """Gefiltertes DataFrame (pro Gruppe, z. B. Athlet, separat gefiltert)."""
    if df.empty:
        return df
    parts = []
    for _, d in (df.groupby(group, sort=False) if group else [(None, df)]):
        d = d.sort_values(t)
        m = clean_mask(d[lat], d[lon], d[t], d[alt] if alt in d.columns else None,
                       d[speed] if speed in d.columns else None)
        parts.append(d[m])
    return pd.concat(parts) if parts else df.iloc[0:0]
