"""
AdMos CSV Parser — für AdMos Sensoren von Archinisis GmbH
Liest imuData.csv und gnssData.csv, normalisiert Spaltennamen,
extrahiert Metadaten aus dem Dateinamen.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

SAMPLE_RATE_IMU = 200   # Hz
SAMPLE_RATE_GNSS = 10   # Hz
CLIP_LIMIT_G = 16.0     # Sensorbegrenzung AdMos

import os as _os
DATA_FOLDER = "/Users/cyrilwyss/AAMasterarbeit/zz_Data Claude" if _os.path.exists("/Users/cyrilwyss/AAMasterarbeit/zz_Data Claude") else ""

# Ordner-Namen (lowercase) die beim On-Snow-Scan übersprungen werden
EXCLUDE_DIRS = {"datenfiles validierung", "nicht verwenden", "nicht verwenden!", "runs", "__pycache__"}


@dataclass
class SensorMeta:
    filename: str
    date: str
    location: str
    sensor_id: str
    athlete_code: str
    position: str          # "Bauch", "Fuss_re", "Fuss_li"
    position_label: str    # "Körperschwerpunkt (Bauch)", etc.


def parse_filename(filepath: str) -> SensorMeta:
    """Extrahiert Metadaten aus AdMos-Dateiname.

    Schema: YYYYMMDD_Location_IDxxx_xX_Position_imuData.csv
    Beispiel: 20260402_Laax_ID327_5A_Bauch_imuData.csv
    """
    name = Path(filepath).stem
    name = re.sub(r"_(imuData|gnssData|imu|gnss)$", "", name, flags=re.IGNORECASE)

    parts = name.split("_")
    date = parts[0] if len(parts) > 0 else ""
    location = parts[1] if len(parts) > 1 else ""

    pos_map = {
        "Bauch": "Körperschwerpunkt (Bauch)",
        "Fuss_re": "Fuss rechts",
        "Fuss_li": "Fuss links",
        "Bau1ch": "Körperschwerpunkt (Bauch)",  # Tippfehler in Dateinamen
    }

    # Format erkennen: 5+ Teile = alt (SensorID_AthleteCode_Position),
    #                  4 Teile  = neu (AthleteID_Position)
    if len(parts) >= 5:
        # Altes Format: YYYYMMDD_Ort_SensorID_AthleteCode_Position
        sensor_id = parts[2]
        athlete_code = parts[3]
        position_raw = "_".join(parts[4:])
    else:
        # Neues Format: YYYYMMDD_Ort_AthleteID_Position
        sensor_id = parts[2] if len(parts) > 2 else ""
        athlete_code = parts[2] if len(parts) > 2 else ""   # AthleteID als Code
        position_raw = "_".join(parts[3:]) if len(parts) > 3 else ""

    # Normalisierung: "04.2" → "04" (Ersatzsensor selber Athlet)
    import re as _re
    athlete_code = _re.sub(r'\.\d+$', '', athlete_code)

    position_label = pos_map.get(position_raw, position_raw or "Unbekannt")

    return SensorMeta(
        filename=Path(filepath).name,
        date=date,
        location=location,
        sensor_id=sensor_id,
        athlete_code=athlete_code,
        position=position_raw,
        position_label=position_label,
    )


def load_imu_raw(filepath) -> pd.DataFrame:
    """Lädt imuData.csv mit originalen AdMos-Spaltennamen (für sensor_lib)."""
    return pd.read_csv(filepath)


def load_gnss_raw(filepath) -> pd.DataFrame:
    """Lädt gnssData.csv mit originalen Spaltennamen."""
    return pd.read_csv(filepath)


def classify_sensor_file(filename: str) -> str | None:
    """
    Gibt 'imu' oder 'gnss' zurück wenn der Dateiname erkannt wird, sonst None.
    Erkennt (case-insensitiv): _IMU.csv, _imu.csv, _imuData.csv,
                                _GNSS.csv, _gnss.csv, _gnssData.csv
    Ignoriert: Checkpoint-Dateien, Summary-Dateien, etc.
    """
    name = Path(filename).name
    if "-checkpoint" in name.lower():
        return None
    stem = Path(filename).stem  # ohne .csv
    last = stem.rsplit("_", 1)[-1].lower()
    if last in ("imu", "imudata"):
        return "imu"
    if last in ("gnss", "gnssdata"):
        return "gnss"
    return None


def sensor_file_base(filepath: str) -> str:
    """Gibt den Basis-Schlüssel zurück (alles ohne Typ-Suffix und Extension)."""
    stem = Path(filepath).stem
    last = stem.rsplit("_", 1)[-1].lower()
    if last in ("imu", "imudata", "gnss", "gnssdata"):
        return stem.rsplit("_", 1)[0]
    return stem


def find_csv_pairs(folder: str) -> list[dict]:
    """
    Findet alle IMU/GNSS Paare rekursiv.
    Unterstützt alle Varianten: _IMU/_imu/_imuData / _GNSS/_gnss/_gnssData
    Ignoriert Checkpoint-, Summary- und unbekannte Dateien.
    """
    folder = Path(folder)
    by_base: dict[str, dict] = {}

    def _iter_csv(root: Path):
        """Rekursiver Generator — überspringt ausgeschlossene Unterordner."""
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                if entry.name.lower() not in EXCLUDE_DIRS:
                    yield from _iter_csv(entry)
            elif entry.suffix.lower() == ".csv":
                yield entry

    for csv_f in _iter_csv(folder):
        kind = classify_sensor_file(str(csv_f))
        if kind is None:
            continue
        base_key = str(csv_f.parent / sensor_file_base(str(csv_f)))
        entry = by_base.setdefault(base_key, {"imu_path": None, "gnss_path": None})
        if kind == "imu" and entry["imu_path"] is None:
            entry["imu_path"] = str(csv_f)
        elif kind == "gnss" and entry["gnss_path"] is None:
            entry["gnss_path"] = str(csv_f)

    pairs = []
    for base_key, entry in by_base.items():
        if entry["imu_path"] is None:
            continue  # IMU ist Pflicht
        meta = parse_filename(entry["imu_path"])
        pairs.append({
            "imu_path": entry["imu_path"],
            "gnss_path": entry["gnss_path"],
            "meta": meta,
            "label": f"{meta.date} | {meta.location} | {meta.athlete_code} | {meta.position_label}",
        })
    return sorted(pairs, key=lambda p: p["label"])
