"""
Ausreisser-Filter für GPS-Tracks.

Entfernt Punkte, die physikalisch nicht möglich sind:
1. weiter als MAX_DIST_M vom Median-Ort des Tracks (z. B. alte Startposition des Sensors),
2. gemeldete Geschwindigkeit > MAX_SPEED_KMH,
3. Positionssprung zum letzten gültigen Punkt mit > MAX_SPEED_KMH,
4. Höhe weicht mehr als MAX_ALT_RANGE_M vom Median des Tracks oder mehr als MAX_ALT_DEV_M vom gleitenden Median ab,
5. Höhe passt nicht zum Ort: Der Athlet fährt dieselben Pisten mehrfach, deshalb wird pro Rasterzelle
   (ALT_CELL_M) die Median-Höhe aller Durchgänge gebildet; Abweichungen > ALT_CELL_DEV_M gelten als Höhenfehler,
6. vertikale Geschwindigkeit zum letzten gültigen Punkt > MAX_VSPEED_MS,
7. kein gültiger GNSS-Fix: Position/Höhe über mehrere Punkte eingefroren (identische Werte) oder
   gemeldete Geschwindigkeit exakt 0, obwohl sich die Position deutlich verschiebt,
8. Höhe ausserhalb des 0.5–99.5-%-Bands des Tracks ± ALT_BAND_MARGIN_M.
Zusätzlich: break_gaps() fügt bei Lücken > MAX_GAP_S eine Unterbrechung ein, damit Plots keine Geraden über Lücken ziehen.
"""
import numpy as np
import pandas as pd

MAX_SPEED_KMH = 100.0
MAX_DIST_M = 5000.0
MAX_ALT_DEV_M = 30.0
MAX_ALT_RANGE_M = 800.0   # Höhe max. so weit vom Median des Tracks entfernt
ALT_CELL_M = 25.0         # Rastergrösse für den Höhen-Ort-Abgleich
ALT_CELL_DEV_M = 40.0     # max. Abweichung von der Median-Höhe derselben Rasterzelle
ALT_CELL_MIN_N = 5        # nur Zellen mit genügend Punkten prüfen
MAX_VSPEED_MS = 12.0      # max. vertikale Geschwindigkeit (Lift ~5 m/s, Abfahrt selten > 10 m/s)
MAX_GAP_S = 30.0
FROZEN_MIN_N = 3          # so viele identische Punkte hintereinander = eingefroren
ZERO_SPEED_MOVE_KMH = 5.0
ALT_BAND_MARGIN_M = 60.0  # Toleranz um das 0.5–99.5-%-Höhenband # Speed exakt 0, aber Positionsänderung schneller als dies = ungültig          # Lücke, ab der Linien in Plots unterbrochen werden


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
    # 7) eingefrorene Werte (kein Fix): identische Position und Höhe über >= FROZEN_MIN_N Punkte
    same = np.r_[False, (np.diff(lat) == 0) & (np.diff(lon) == 0)]
    if alt is not None:
        av0 = np.asarray(alt, float)
        same &= np.r_[False, np.diff(av0) == 0]
    run_id = np.cumsum(~same)
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).values
    keep &= ~(run_len >= FROZEN_MIN_N)
    if speed_kmh is not None:
        sp0 = np.asarray(speed_kmh, float)
        dt0 = np.r_[np.nan, np.diff(t) / 1000.0]
        v_impl = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))] / np.where(dt0 > 0, dt0, np.nan) * 3.6
        keep &= ~((sp0 == 0) & (v_impl > ZERO_SPEED_MOVE_KMH))
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
        av = a.values
        # 5) Höhe passend zum Ort (Median-Höhe pro Rasterzelle über alle Durchgänge)
        if keep.sum() >= 3:
            cx = np.floor(x / ALT_CELL_M); cy = np.floor(y / ALT_CELL_M)
            cells = pd.DataFrame({"cx": cx, "cy": cy, "a": np.where(keep, av, np.nan)})
            g = cells.groupby(["cx", "cy"])["a"]
            cmed = g.transform("median").values; cn = g.transform("count").values
            keep &= ~((cn >= ALT_CELL_MIN_N) & np.isfinite(cmed) & (np.abs(av - cmed) > ALT_CELL_DEV_M))
        # 6) vertikale Sprünge zum letzten gültigen Punkt
        last = None
        for i in range(n):
            if not keep[i] or not np.isfinite(av[i]):
                continue
            if last is not None:
                dt = (t[i] - t[last]) / 1000.0
                if dt > 0 and abs(av[i] - av[last]) / dt > MAX_VSPEED_MS:
                    keep[i] = False
                    continue
            last = i
        # 8) einzelne verbleibende Höhenausreisser ausserhalb des üblichen Höhenbands des Tracks
        if keep.sum() >= 50:
            lo, hi = np.nanpercentile(av[keep], [0.5, 99.5])
            keep &= (av >= lo - ALT_BAND_MARGIN_M) & (av <= hi + ALT_BAND_MARGIN_M)
    return keep


def break_gaps(df: pd.DataFrame, t="Zeit (POSIX ms)", group=None, max_gap_s: float = MAX_GAP_S) -> pd.DataFrame:
    """Fügt nach jeder Lücke > max_gap_s eine leere Zeile ein (NaN), damit Linien dort unterbrochen werden."""
    if df.empty:
        return df
    parts = []
    for _, d in (df.groupby(group, sort=False) if group else [(None, df)]):
        d = d.sort_values(t)
        gap = (d[t].diff() / 1000.0 > max_gap_s).values
        if gap.any():
            blank = d[gap].copy()
            for c in blank.columns:
                if c != group and pd.api.types.is_numeric_dtype(blank[c]):
                    blank[c] = np.nan
            blank[t] = d[t].values[gap] - 1   # direkt vor dem Punkt nach der Lücke
            d = pd.concat([d, blank]).sort_values(t, kind="stable")
        parts.append(d)
    return pd.concat(parts)


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
