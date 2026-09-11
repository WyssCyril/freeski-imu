"""
Lokales Preprocessing: IMU auf GNSS-erkannte Run-Segmente zuschneiden.
Originaldateien werden NIE verändert. Cut-Dateien in neuem Ordner gespeichert.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from utils.admos_parser import load_imu_raw, load_gnss_raw, find_csv_pairs, DATA_FOLDER
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
import sensor_lib

# ── Konfiguration ─────────────────────────────────────────────────────────────
INPUT_FOLDER  = DATA_FOLDER          # Originaldateien
OUTPUT_FOLDER = os.path.join(os.path.dirname(DATA_FOLDER), "Cut Datafiles")  # Unterordner: <Ort>/<Datum>
BUFFER_S      = 60.0                 # Sekunden Puffer vor/nach jedem Run

# GNSS Run-Erkennungsparameter (identisch zur App)
GNSS_PARAMS = dict(v_start=7.0, v_hold=1.5, alt_rise_end=20.0,
                   alt_drop_min=30.0, run_duration_min=15.0)

# ── Ordner anlegen ────────────────────────────────────────────────────────────
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
print(f"\n{'='*60}")
print(f"Input:  {INPUT_FOLDER}")
print(f"Output: {OUTPUT_FOLDER}")
print(f"Puffer: ±{BUFFER_S}s pro Run")
print(f"{'='*60}\n")

# ── Alle CSV-Paare finden ─────────────────────────────────────────────────────
pairs = find_csv_pairs(INPUT_FOLDER)
if not pairs:
    print("Keine CSV-Paare gefunden.")
    sys.exit(1)

print(f"{len(pairs)} Sensor-Dateien gefunden:\n")

total_saved = 0
total_skipped = 0

for p in pairs:
    meta     = p["meta"]
    imu_path = p["imu_path"]
    gnss_path = p["gnss_path"]
    label    = f"{meta.athlete_code} | {meta.position_label} | {meta.date}"

    print(f"─── {label}")
    print(f"    IMU:  {Path(imu_path).name}")

    # IMU laden
    try:
        imu_df = load_imu_raw(imu_path)
    except Exception as e:
        print(f"    FEHLER beim Laden IMU: {e}\n")
        total_skipped += 1
        continue

    imu_rows_orig = len(imu_df)

    # GNSS laden und Run-Erkennung
    run_intervals = []
    if gnss_path:
        try:
            gnss_df = load_gnss_raw(gnss_path)
            gnss_clean = sensor_lib.add_resultant_speed(gnss_df)
            runs_df = sensor_lib.detect_runs(gnss_clean, **GNSS_PARAMS)

            if runs_df is not None and not runs_df.empty:
                for _, row in runs_df.iterrows():
                    s = float(row["start_time_us"])
                    e = float(row["end_time_us"])
                    run_intervals.append((s, e))
                print(f"    GNSS: {len(run_intervals)} Runs erkannt")
            else:
                print(f"    GNSS: Keine Runs erkannt → volle IMU-Datei behalten")
        except Exception as e:
            print(f"    GNSS-Fehler: {e} → volle IMU-Datei behalten")
    else:
        print(f"    Kein GNSS → volle IMU-Datei behalten")

    # IMU zuschneiden
    if run_intervals:
        t_col   = "imuTimestamp [us]"
        buf_us  = BUFFER_S * 1_000_000
        keep    = pd.Series(False, index=imu_df.index)
        for s_us, e_us in run_intervals:
            keep |= (imu_df[t_col] >= s_us - buf_us) & (imu_df[t_col] <= e_us + buf_us)
        imu_cut = imu_df[keep].reset_index(drop=True)
        pct     = len(imu_cut) / imu_rows_orig * 100
        print(f"    IMU:  {imu_rows_orig:>8,} → {len(imu_cut):>8,} Zeilen  ({pct:.1f}% behalten)")
    else:
        imu_cut = imu_df
        print(f"    IMU:  {imu_rows_orig:>8,} Zeilen (vollständig, kein Cut)")

    # Speichern
    base_name  = Path(imu_path).stem.replace("_imuData", "").replace("_imu", "").replace("_IMU", "")
    out_name   = f"{base_name}_imuData_cut.csv"
    session_dir = os.path.join(OUTPUT_FOLDER, meta.location, meta.date)
    Path(session_dir).mkdir(parents=True, exist_ok=True)
    out_path   = os.path.join(session_dir, out_name)
    imu_cut.to_csv(out_path, index=False)
    size_mb    = os.path.getsize(out_path) / 1_000_000
    if gnss_path:
        shutil.copy2(gnss_path, os.path.join(session_dir, Path(gnss_path).name))
    print(f"    → {Path(session_dir).name}/{out_name}  ({size_mb:.1f} MB)\n")
    total_saved += 1

print(f"{'='*60}")
print(f"Fertig: {total_saved} Dateien gespeichert, {total_skipped} übersprungen")
print(f"Cut-Dateien in: {OUTPUT_FOLDER}")
print(f"\nNächster Schritt: Cut-IMU + Original-GNSS in die App hochladen.")
print(f"{'='*60}\n")
