"""
Statistik-Funktionen: Bland-Altman, Typical Error, Gruppenvergleich.
"""
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go


def bland_altman(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Bland-Altman Analyse.
    x = Referenz (Kraftmessplatte), y = IMU-Sensor
    """
    diff = y - x
    mean = (x + y) / 2.0

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    n = len(diff)
    se = sd / np.sqrt(n)

    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # 95% CI für bias
    ci_bias = stats.t.ppf(0.975, df=n - 1) * se
    # 95% CI für LoA
    se_loa = np.sqrt(3 * sd**2 / n)
    ci_loa = stats.t.ppf(0.975, df=n - 1) * se_loa

    # Proportional bias (trend)
    slope, intercept, r, p, _ = stats.linregress(mean, diff)

    # Outlier: |diff - bias| > 2*sd
    outlier_mask = np.abs(diff - bias) > 2 * sd

    # Typical Error, CV
    te = sd / np.sqrt(2)
    grand_mean = float(np.mean(mean))
    cv = (te / grand_mean * 100) if grand_mean != 0 else np.nan

    return {
        "bias": round(bias, 4),
        "sd": round(sd, 4),
        "loa_upper": round(loa_upper, 4),
        "loa_lower": round(loa_lower, 4),
        "ci_bias": round(ci_bias, 4),
        "ci_loa": round(ci_loa, 4),
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r": round(r, 4),
        "p_trend": round(p, 6),
        "outlier_mask": outlier_mask,
        "typical_error": round(te, 4),
        "cv_pct": round(cv, 2),
        "n": n,
        "diff": diff,
        "mean": mean,
    }


def plot_bland_altman(ba: dict, title: str = "",
                      warn_pct: float = 15.0, excl_pct: float = 25.0) -> go.Figure:
    mean = ba["mean"]
    diff = ba["diff"]
    bias = ba["bias"]
    loa_upper = ba["loa_upper"]
    loa_lower = ba["loa_lower"]
    ci_bias = ba["ci_bias"]
    ci_loa = ba["ci_loa"]
    slope = ba["slope"]
    intercept = ba["intercept"]
    p = ba["p_trend"]
    outlier_mask = ba["outlier_mask"]

    grand_mean = float(np.mean(mean))
    warn_abs = grand_mean * warn_pct / 100
    excl_abs = grand_mean * excl_pct / 100

    fig = go.Figure()

    # Scatter: normale Punkte
    fig.add_trace(go.Scatter(
        x=mean[~outlier_mask], y=diff[~outlier_mask],
        mode="markers", marker=dict(color="#1f77b4", size=7, opacity=0.7),
        name="Messpunkte",
    ))
    # Outlier
    if outlier_mask.any():
        fig.add_trace(go.Scatter(
            x=mean[outlier_mask], y=diff[outlier_mask],
            mode="markers", marker=dict(color="#d62728", size=9, symbol="x"),
            name="Ausreisser",
        ))

    x_range = [float(np.min(mean)) * 0.95, float(np.max(mean)) * 1.05]

    # Bias + CI
    fig.add_hline(y=bias, line_color="#333", line_dash="solid",
                  annotation_text=f"Bias = {bias:.3f}", annotation_position="right")
    fig.add_hrect(y0=bias - ci_bias, y1=bias + ci_bias,
                  fillcolor="grey", opacity=0.12, line_width=0)

    # LoA
    for loa, sign in [(loa_upper, "+"), (loa_lower, "−")]:
        fig.add_hline(y=loa, line_color="#ff7f0e", line_dash="dash",
                      annotation_text=f"{sign}1.96 SD = {loa:.3f}")
        fig.add_hrect(y0=loa - ci_loa, y1=loa + ci_loa,
                      fillcolor="#ff7f0e", opacity=0.10, line_width=0)

    # Warn/Ausschluss-Schwellen
    for y_abs, color, label in [
        (warn_abs, "#f0a500", f"Warnschwelle {warn_pct}%"),
        (excl_abs, "#d62728", f"Ausschluss {excl_pct}%"),
    ]:
        for sign in [1, -1]:
            fig.add_hline(y=sign * y_abs, line_color=color, line_dash="dot",
                          annotation_text=label if sign == 1 else "")

    # Trendlinie
    trend_sig = p < 0.05
    x_arr = np.array(x_range)
    y_trend = slope * x_arr + intercept
    p_trend_str = "< 0.05" if trend_sig else f"{p:.2f}"
    fig.add_trace(go.Scatter(
        x=x_arr, y=y_trend,
        mode="lines",
        line=dict(color="#9467bd", dash="dot", width=1.5),
        name=f"Trend (p={p_trend_str})",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mittelwert (Referenz + IMU) / 2",
        yaxis_title="Differenz (IMU − Referenz)",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_correlation(x: np.ndarray, y: np.ndarray,
                     x_label: str = "Kraftmessplatte", y_label: str = "IMU",
                     title: str = "") -> go.Figure:
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_range = np.array([float(np.min(x)), float(np.max(x))])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color="#1f77b4", size=7, opacity=0.7),
        name="Messpunkte",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=slope * x_range + intercept,
        mode="lines", line=dict(color="#ff7f0e"), name=f"Regression (r={r:.3f})",
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=x_range,
        mode="lines", line=dict(color="grey", dash="dot"), name="Identitätslinie",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=400,
    )
    return fig


def compare_groups(a: np.ndarray, b: np.ndarray,
                   name_a: str = "A", name_b: str = "B") -> dict:
    """Shapiro-Wilk → t-Test oder Wilcoxon, plus Cohen's d."""
    _, p_norm_a = stats.shapiro(a)
    _, p_norm_b = stats.shapiro(b)
    normal = p_norm_a > 0.05 and p_norm_b > 0.05

    if normal:
        stat, p_val = stats.ttest_ind(a, b)
        test_name = "t-Test"
    else:
        stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Wilcoxon"

    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan

    return {
        "test": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 6),
        "cohens_d": round(float(cohens_d), 4),
        "normal": normal,
        f"mean_{name_a}": round(float(np.mean(a)), 4),
        f"mean_{name_b}": round(float(np.mean(b)), 4),
        f"std_{name_a}": round(float(np.std(a, ddof=1)), 4),
        f"std_{name_b}": round(float(np.std(b, ddof=1)), 4),
    }
