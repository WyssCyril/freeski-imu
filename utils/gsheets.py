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
    return df.replace({"TRUE": True, "FALSE": False})


def save(df: pd.DataFrame) -> tuple[int, int]:
    """
    Upsert: bestehende Zeilen mit gleichem Schlüssel (Athlet/Datum/Ort/Position/Sprung)
    werden ersetzt, neue angehängt. Gibt (neu, aktualisiert) zurück.
    """
    df = df.copy()
    df["Gespeichert am"] = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    for c in KEY_COLS:
        df[c] = df[c].astype(str)

    existing = load()
    if existing.empty:
        merged, n_new, n_upd = df, len(df), 0
    else:
        for c in KEY_COLS:
            existing[c] = existing[c].astype(str)
        new_keys = set(map(tuple, df[KEY_COLS].values))
        old_keys = set(map(tuple, existing[KEY_COLS].values))
        n_upd = len(new_keys & old_keys)
        n_new = len(new_keys - old_keys)
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(KEY_COLS, keep="last")

    cols = list(df.columns) + [c for c in merged.columns if c not in df.columns]
    merged = merged.reindex(columns=cols).fillna("")
    values = [cols] + merged.astype(object).values.tolist()

    ws = _worksheet()
    ws.clear()
    ws.update(values, value_input_option="RAW")
    return n_new, n_upd


def sheet_url() -> str:
    return f"https://docs.google.com/spreadsheets/d/{st.secrets['sheet_id']}"
