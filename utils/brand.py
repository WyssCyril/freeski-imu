"""
Swiss-Ski corporate design for the dashboards (Swiss-Ski STUDIOS brand portal, primary colour world).

Only look and feel: colours, logo, font. No calculation depends on this module.
The same file is used in freeski-imu, snowboard-halfpipe-imu and corvatsch-imu.

Rule from the brand portal: never Snow White on Ice Blue – on Ice Blue only Alps Grey.
"""
import base64
import functools
import os

import plotly.io as pio

# ── Primary colour world ─────────────────────────────────────────────────────
RED = "#E2001A"         # Swiss-Ski Red      (PANTONE 485)
MIDDLE_RED = "#C00418"  # Swiss-Ski Middle Red (PANTONE 7621)
DARK_RED = "#8B0E13"    # Swiss-Ski Dark Red (PANTONE 7624)
ICE_BLUE = "#39C3D4"    # Swiss-Ski Ice Blue (PANTONE 3125) – highlights, never with white text
ALPS_GREY = "#3E3E3E"   # Swiss-Ski Alps Grey (PANTONE Black 7) – text
SNOW_WHITE = "#FFFFFF"  # Swiss-Ski Snow White

# Tints / shades of the primary colours for charts and surfaces
RED_TINT = "#F28C96"
RED_LIGHT = "#FDEDEF"
RED_BORDER = "#F5B3BB"
ICE_DARK = "#1E8A97"
ICE_LIGHT = "#9FE1E9"
GREY_MID = "#6E6E6E"
GREY_LIGHT = "#9E9E9E"
GREY_LINE = "#E3E3E3"
GREY_BG = "#F4F4F4"
GREY_BG_2 = "#F7F7F7"

# Categorical order for charts (athletes, sessions, series)
SEQ = [RED, ALPS_GREY, ICE_BLUE, DARK_RED, GREY_LIGHT, RED_TINT, ICE_DARK, MIDDLE_RED, "#BDBDBD", ICE_LIGHT,
       GREY_MID, RED_BORDER]
# Continuous scales
SCALE_RED = [[0.0, RED_LIGHT], [0.5, RED], [1.0, DARK_RED]]
SCALE_SPEED = [[0.0, ICE_DARK], [0.35, ICE_BLUE], [0.65, RED], [1.0, DARK_RED]]  # slow = blue, fast = red

FONT_FAMILY = "'Cera Pro', 'Helvetica Neue', Arial, sans-serif"

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(_HERE, "..", "static")


@functools.lru_cache(maxsize=None)
def logo_uri(variant: str = "red") -> str:
    """Swiss-Ski logo as data URI. variant: red (default), white (on red/dark), grey."""
    with open(os.path.join(_STATIC, f"swiss-ski-logo-{variant}.svg"), "rb") as fh:
        return "data:image/svg+xml;base64," + base64.b64encode(fh.read()).decode()


@functools.lru_cache(maxsize=None)
def font_css() -> str:
    """@font-face rules for Cera Pro (woff2, Latin subset) as embedded data URIs."""
    rules = []
    for name, weight in [("Regular", 400), ("Medium", 500), ("Bold", 700), ("Black", 900)]:
        path = os.path.join(_STATIC, "fonts", f"CeraPro-{name}.woff2")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        rules.append("@font-face{font-family:'Cera Pro';font-style:normal;font-display:swap;"
                     f"font-weight:{weight};src:url(data:font/woff2;base64,{b64}) format('woff2');}}")
    return "\n".join(rules)


def base_css() -> str:
    """Font faces plus the Cera Pro font for the whole Streamlit page."""
    return (font_css() + "\n"
            f"html, body, [class*='st-'], .stMarkdown, .stMetric, button, input, textarea, select, "
            f"h1, h2, h3, h4, h5, h6, p, label {{ font-family: {FONT_FAMILY} !important; }}\n"
            # keep Streamlit's icon font for icons
            "[data-testid='stIconMaterial'], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }\n")


def apply_plotly_theme():
    """Brand colours and font for every Plotly chart (also when a figure sets template='plotly_white')."""
    for name in ("plotly_white", "plotly"):
        t = pio.templates[name]
        t.layout.colorway = SEQ
        t.layout.font.family = FONT_FAMILY
        t.layout.font.color = ALPS_GREY
        t.layout.title.font.color = ALPS_GREY
        t.layout.colorscale.sequential = SCALE_RED
    pio.templates.default = "plotly_white"


apply_plotly_theme()
