"""
Datenprüfung beim Hochladen: findet Probleme in IMU- und GNSS-Dateien, bevor die Daten
ausgewertet und gespeichert werden. Jede Meldung hat eine Stufe:
  "fehler"  → Daten werden NICHT automatisch gespeichert (Entscheid durch Nutzer:in)
  "warnung" → wird angezeigt, Speichern bleibt automatisch
"""
import numpy as np
import pandas as pd

IMU_TS = "imuTimestamp [us]"
GNSS_COLS = ["time [POSIXms]", "timestamp [us]", "latitude [deg]", "longitude [deg]",
             "speedN [m/s]", "speedE [m/s]", "speedD [m/s]", "altitude [m]"]


def check_files(imu: pd.DataFrame | None, gnss: pd.DataFrame | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    # ── IMU ──────────────────────────────────────────────────────────────
    if imu is None or imu.empty:
        out.append(("fehler", "Keine IMU-Daten in der Datei."))
    else:
        missing = [c for c in [IMU_TS, "accX [g]", "accY [g]", "accZ [g]"] if c not in imu.columns]
        if missing:
            out.append(("fehler", f"IMU-Datei: Spalten fehlen ({', '.join(missing)}). Wurde die ASI-Datei korrekt konvertiert?"))
        else:
            n = len(imu)
            if n < 200 * 60:
                out.append(("fehler", f"IMU-Datei sehr kurz ({n / 200 / 60:.1f} min) – vermutlich leer oder abgebrochen."))
            t = imu[IMU_TS].values.astype(float)
            dt = np.diff(t)
            n_back = int((dt <= 0).sum())
            if n_back > max(5, 0.0005 * n):
                out.append(("fehler", f"IMU-Zeitstempel springen {n_back}-mal zurück – Datei vermutlich beschädigt "
                                      "(Sessions und Sprünge werden falsch erkannt)."))
            med = float(np.median(dt[dt > 0])) if (dt > 0).any() else np.nan
            if np.isfinite(med) and abs(med - 5000) > 500:
                out.append(("warnung", f"IMU-Abtastrate {1e6 / med:.0f} Hz statt 200 Hz – Flugzeit und Impuls wären falsch."))
            acc = imu[["accX [g]", "accY [g]", "accZ [g]"]].abs()
            if (acc >= 15.99).mean().max() > 0.02:
                out.append(("warnung", "Mehr als 2 % der IMU-Werte am Messlimit (16 g) – Sensor ungewöhnlich oft übersteuert."))
    # ── GNSS ─────────────────────────────────────────────────────────────
    if gnss is None or gnss.empty:
        out.append(("warnung", "Keine GNSS-Datei – Abfahrten können nicht erkannt werden, Sprünge werden über die ganze Aufzeichnung gesucht."))
        return out
    missing = [c for c in GNSS_COLS if c not in gnss.columns]
    if missing:
        out.append(("fehler", f"GNSS-Datei hat ein abweichendes Format (es fehlen: {', '.join(missing)})."))
        return out
    valid = (gnss["time [POSIXms]"] > 1.58e12) & (gnss["latitude [deg]"].abs() > 1) & (gnss["longitude [deg]"].abs() > 1)
    if valid.sum() < 10 * 60 * 10:
        out.append(("fehler", f"GNSS: nur {valid.sum() / 10 / 60:.1f} min mit gültiger Position – kein brauchbarer GPS-Empfang."))
    elif valid.mean() < 0.5:
        out.append(("warnung", f"GNSS: nur {valid.mean():.0%} der Punkte mit gültiger Position."))
    if valid.sum() > 10:
        from utils.gnss_clean import clean_mask
        g = gnss[valid].sort_values("time [POSIXms]")
        spd = np.sqrt(g["speedN [m/s]"] ** 2 + g["speedE [m/s]"] ** 2 + g["speedD [m/s]"] ** 2) * 3.6
        keep = clean_mask(g["latitude [deg]"], g["longitude [deg]"], g["time [POSIXms]"], g["altitude [m]"], spd)
        share = 1 - keep.mean()
        if share > 0.2:
            out.append(("warnung", f"GNSS: {share:.0%} der Punkte sind Ausreisser (> 100 km/h oder Positionssprünge)."))
    # ── IMU und GNSS passen zusammen? ────────────────────────────────────
    if imu is not None and not imu.empty and IMU_TS in imu.columns:
        ti = imu[IMU_TS].values.astype(float)
        tg = gnss["timestamp [us]"].values.astype(float)
        lo, hi = max(np.nanmin(ti), np.nanmin(tg)), min(np.nanmax(ti), np.nanmax(tg))
        overlap = max(0.0, hi - lo) / max(1.0, np.nanmax(ti) - np.nanmin(ti))
        if overlap < 0.5:
            out.append(("fehler", f"IMU und GNSS überlappen zeitlich nur zu {overlap:.0%} – passen die beiden Dateien zusammen?"))
    return out


def check_analysis(result: dict) -> list[tuple[str, str]]:
    """Prüfung nach der Sprungerkennung."""
    out = []
    runs = [r for s in result.get("sessions", {}).values() for k, r in s.get("runs", {}).items() if k != "Alle"]
    n_j = sum(len(r["jumps"]) for s in result.get("sessions", {}).values() for r in s.get("runs", {}).values()
              if r.get("jumps") is not None)
    if not runs:
        out.append(("fehler", "Keine Abfahrten im GPS erkannt – Sprünge können keinem Run zugeordnet werden."))
    if n_j == 0:
        out.append(("fehler", "Keine Sprünge erkannt."))
    if len(result.get("sessions", {})) > 10:
        out.append(("fehler", f"{len(result['sessions'])} Sessions erkannt – meist ein Zeichen für beschädigte Zeitstempel."))
    return out
