"""
Persistente Datenablage: Sprungauswertungen in einem Google Sheet.
Zugriff über Service Account (st.secrets['gcp_service_account'], st.secrets['sheet_id']).
"""
import datetime
import streamlit as st
import pandas as pd

WORKSHEET = "Sprünge"
KEY_COLS = ["Athlet", "Datum", "Ort", "Position", "Run", "Sprung"]
_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def norm_athlet(v) -> str:
    """'2', '2.0', 2, '02' → '02'; nicht-numerische Codes bleiben unverändert."""
    s = str(v).strip()
    try:
        f = float(s)
        if f.is_integer():
            return f"{int(f):02d}"
    except ValueError:
        pass
    return s


def enabled() -> bool:
    try:
        return "gcp_service_account" in st.secrets and "sheet_id" in st.secrets
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _worksheet():
    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=_SCOPES
    )
    sh = gspread.authorize(creds).open_by_key(st.secrets["sheet_id"])
    try:
        return sh.worksheet(WORKSHEET)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(WORKSHEET, rows=2000, cols=20)


def load() -> pd.DataFrame:
    """Alle gespeicherten Sprünge als DataFrame (leer, wenn das Sheet leer ist)."""
    df = pd.DataFrame(_worksheet().get_all_records())
    # RAW-Schreiben legt Booleans als Text "TRUE"/"FALSE" ab
    df = df.replace({"TRUE": True, "FALSE": False})
    if "Athlet" in df.columns:
        df["Athlet"] = df["Athlet"].map(norm_athlet)
    return df


GROUP_COLS = ["Athlet", "Datum", "Ort", "Position"]


def _content_key(df: pd.DataFrame) -> pd.Series:
    """Inhaltlicher Schlüssel: gleicher Run + gleicher Peak + gleiche Flugzeit = gleicher Sprung."""
    return (df["Athlet"].astype(str) + "|" + df["Datum"].astype(str) + "|" + df["Ort"].astype(str)
            + "|" + df["Position"].astype(str) + "|" + df["Run"].astype(str)
            + "|" + pd.to_numeric(df["Peak (g)"], errors="coerce").round(2).astype(str)
            + "|" + pd.to_numeric(df["Flugzeit (s)"], errors="coerce").round(3).astype(str))


def save(df: pd.DataFrame) -> tuple[int, int]:
    """
    Speichern ohne Doppelte: Für jeden enthaltenen Athleten (Athlet/Datum/Ort/Position)
    werden ALLE bisherigen Zeilen im Sheet durch die aktuellen ersetzt.
    Gibt (neu, ersetzt) zurück.
    """
    df = df.copy()
    df["Gespeichert am"] = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    df["Athlet"] = df["Athlet"].map(norm_athlet)
    for c in KEY_COLS:
        df[c] = df[c].astype(str)
    df = df.drop_duplicates(KEY_COLS, keep="last")
    df = df[~_content_key(df).duplicated(keep="last")]

    existing = load()
    if existing.empty:
        merged, n_new, n_upd = df, len(df), 0
    else:
        for c in KEY_COLS:
            existing[c] = existing[c].astype(str)
        groups = set(map(tuple, df[GROUP_COLS].drop_duplicates().values))
        in_groups = existing[GROUP_COLS].apply(tuple, axis=1).isin(groups)
        n_upd = int(in_groups.sum())
        n_new = max(len(df) - n_upd, 0)
        merged = pd.concat([existing[~in_groups], df], ignore_index=True)
        merged = merged.drop_duplicates(KEY_COLS, keep="last")
        merged = merged[~_content_key(merged).duplicated(keep="last")]

    cols = list(df.columns) + [c for c in merged.columns if c not in df.columns]
    merged = merged.reindex(columns=cols).fillna("")
    values = [cols] + merged.astype(object).values.tolist()

    ws = _worksheet()
    ws.clear()
    ws.update(values, value_input_option="RAW")
    return n_new, n_upd


def sheet_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{st.secrets['sheet_id']}"
