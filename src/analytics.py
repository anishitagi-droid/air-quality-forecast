from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import AQI_CATEGORIES


# ── Pollutant metadata ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PollutantInfo:
    code: str
    full_name: str
    sources: str
    health_effects: str
    sensitive_groups: str
    unit: str
    icon: str


POLLUTANT_INFO: dict[str, PollutantInfo] = {
    "PM2.5": PollutantInfo(
        code="PM2.5",
        full_name="Fine Particulate Matter (PM2.5)",
        sources="Vehicle exhaust, wildfires, industrial emissions, wood burning",
        health_effects=(
            "Penetrates deep into lung tissue and enters the bloodstream. "
            "Short-term: coughing, reduced lung function, aggravated asthma. "
            "Long-term: heart disease, stroke, premature death."
        ),
        sensitive_groups="Children, elderly, people with heart or lung disease, pregnant women",
        unit="µg/m³",
        icon="🌫️",
    ),
    "Ozone (O₃)": PollutantInfo(
        code="O3",
        full_name="Ground-Level Ozone (O₃)",
        sources="Forms when NOₓ and VOCs react in sunlight — cars, power plants, industry",
        health_effects=(
            "Irritates respiratory system, reduces lung capacity, inflames airways. "
            "Can trigger asthma attacks. Repeated exposure may permanently scar lung tissue."
        ),
        sensitive_groups="Children, active outdoor workers, people with asthma or COPD, elderly",
        unit="ppb",
        icon="☀️",
    ),
    "Nitrogen Dioxide": PollutantInfo(
        code="NO2",
        full_name="Nitrogen Dioxide (NO₂)",
        sources="Burning fossil fuels — vehicles, power plants, off-road equipment",
        health_effects=(
            "Irritates airways and aggravates respiratory diseases, particularly asthma. "
            "Long-term exposure linked to asthma development in children."
        ),
        sensitive_groups="Children, elderly, people with asthma",
        unit="ppb",
        icon="🏭",
    ),
    "Carbon Monoxide": PollutantInfo(
        code="CO",
        full_name="Carbon Monoxide (CO)",
        sources="Incomplete combustion — vehicles, gas appliances, wildfires",
        health_effects=(
            "Reduces oxygen delivery to organs. High levels cause headache, dizziness, "
            "confusion, and can be fatal. Especially dangerous indoors."
        ),
        sensitive_groups="People with heart disease, fetuses, infants, elderly",
        unit="ppm",
        icon="🔥",
    ),
    "PM10": PollutantInfo(
        code="PM10",
        full_name="Coarse Particulate Matter (PM10)",
        sources="Dust, pollen, mold, construction sites, agriculture",
        health_effects=(
            "Irritates eyes, nose, throat. Aggravates asthma and respiratory conditions. "
            "Less penetrating than PM2.5 but still a concern for sensitive groups."
        ),
        sensitive_groups="People with asthma, allergies, or respiratory conditions",
        unit="µg/m³",
        icon="💨",
    ),
    "Sulfur Dioxide": PollutantInfo(
        code="SO2",
        full_name="Sulfur Dioxide (SO₂)",
        sources="Burning sulfur-containing fuels — coal, oil, volcanic eruptions",
        health_effects=(
            "Irritates respiratory tract; can trigger bronchospasm in people with asthma. "
            "Contributes to acid rain and fine particulate formation."
        ),
        sensitive_groups="People with asthma, children, elderly",
        unit="ppb",
        icon="⚗️",
    ),
}

_DEFAULT_INFO = PollutantInfo(
    code="UNKNOWN", full_name="Unknown Pollutant", sources="Various",
    health_effects="Consult local air quality authority for details.",
    sensitive_groups="Sensitive individuals", unit="AQI units", icon="🌡️",
)


def get_pollutant_info(name: str) -> PollutantInfo:
    return POLLUTANT_INFO.get(name, _DEFAULT_INFO)


# ── Health advisories ──────────────────────────────────────────────────────────

@dataclass
class HealthAdvisory:
    aqi: int
    category: str
    color: str
    general: str
    sensitive: str
    outdoor_activity: str
    icon: str


_ADVISORIES = [
    (50,  "Good",               "#00E400", "✅",
     "Air quality is satisfactory. Enjoy normal outdoor activities.",
     "No precautions needed.",
     "Great day for outdoor exercise."),
    (100, "Moderate",           "#FFFF00", "🟡",
     "Air quality is acceptable. Some pollutants may affect very sensitive individuals.",
     "Consider reducing prolonged outdoor exertion if you experience symptoms.",
     "Fine for most people. Unusually sensitive individuals should limit prolonged outdoor exertion."),
    (150, "Unhealthy for Some", "#FF7E00", "🟠",
     "Members of sensitive groups may experience health effects.",
     "Reduce prolonged or heavy outdoor exertion. Watch for symptoms.",
     "Sensitive groups should limit prolonged outdoor exertion."),
    (200, "Unhealthy",          "#FF0000", "🔴",
     "Everyone may begin to experience health effects.",
     "Avoid prolonged outdoor exertion. Move activities indoors or reschedule.",
     "Everyone should limit prolonged outdoor exertion."),
    (300, "Very Unhealthy",     "#8F3F97", "🟣",
     "Health alert: everyone may experience serious health effects.",
     "Avoid all outdoor exertion. Stay indoors with windows closed.",
     "Avoid all outdoor physical activity."),
    (500, "Hazardous",          "#7E0023", "⛔",
     "Health emergency: everyone is likely to be affected.",
     "Stay indoors. Use air purifiers. Seek medical attention if symptoms occur.",
     "Do NOT go outside. Emergency conditions."),
]


def get_health_advisory(aqi: int) -> HealthAdvisory:
    for ceiling, cat, color, icon, general, sensitive, outdoor in _ADVISORIES:
        if aqi <= ceiling:
            return HealthAdvisory(aqi, cat, color, general, sensitive, outdoor, icon)
    _, cat, color, icon, general, sensitive, outdoor = _ADVISORIES[-1]
    return HealthAdvisory(aqi, cat, color, general, sensitive, outdoor, icon)


# ── Trend analysis ─────────────────────────────────────────────────────────────

@dataclass
class TrendSummary:
    direction: str
    direction_icon: str
    pct_change_7d: float
    rolling_7d_avg: float
    streak_days: int
    streak_label: str
    best_day: str
    worst_day: str
    volatility: float


def compute_trend(daily: pd.Series, threshold: int = 100) -> TrendSummary:
    clean = daily.dropna()
    if clean.empty:
        return TrendSummary("stable", "→", 0.0, 0.0, 0, "No data", "—", "—", 0.0)

    rolling = clean.rolling(7, min_periods=1).mean()
    r7_now  = float(rolling.iloc[-1])
    r7_prev = float(rolling.iloc[-8]) if len(rolling) >= 8 else float(rolling.iloc[0])

    pct_change = ((r7_now - r7_prev) / max(r7_prev, 1)) * 100

    if pct_change < -5:
        direction, icon = "improving", "↓"
    elif pct_change > 5:
        direction, icon = "worsening", "↑"
    else:
        direction, icon = "stable", "→"

    recent = clean.tail(30).values[::-1]
    streak = 0
    above  = recent[0] > threshold if len(recent) else False
    for v in recent:
        if (v > threshold) == above:
            streak += 1
        else:
            break

    qualifier   = "above" if above else "below"
    streak_label = (
        f"{streak} consecutive days {qualifier} AQI {threshold}"
        if streak > 1
        else "No notable streak"
    )

    import platform as _plt
    _dfmt = "%b %#d" if _plt.system() == "Windows" else "%b %-d"
    best_day  = clean.idxmin().strftime(_dfmt)
    worst_day = clean.idxmax().strftime(_dfmt)
    volatility = float(clean.tail(14).std()) if len(clean) >= 2 else 0.0

    return TrendSummary(
        direction       = direction,
        direction_icon  = icon,
        pct_change_7d   = round(pct_change, 1),
        rolling_7d_avg  = round(r7_now, 1),
        streak_days     = streak,
        streak_label    = streak_label,
        best_day        = best_day,
        worst_day       = worst_day,
        volatility      = round(volatility, 1),
    )


# ── Primary pollutant ──────────────────────────────────────────────────────────

@dataclass
class PrimaryPollutantSummary:
    name: str
    aqi: int
    category: str
    color: str
    info: PollutantInfo
    recent_avg: float
    trend_pct: float
    trend_icon: str


def identify_primary_pollutant(
    df_current: pd.DataFrame,
    df_hist: pd.DataFrame,
) -> Optional[PrimaryPollutantSummary]:
    if df_current.empty:
        return None

    row   = df_current.loc[df_current["aqi"].idxmax()]
    name  = str(row["pollutant"])
    aqi   = int(row["aqi"])
    cat   = str(row["category"])
    color = str(row["color"])
    info  = get_pollutant_info(name)

    recent_avg, trend_pct, trend_icon = 0.0, 0.0, "→"
    if not df_hist.empty and "pollutant" in df_hist.columns:
        p_hist = df_hist[df_hist["pollutant"] == name].sort_values("date")
        if not p_hist.empty:
            w7      = p_hist.tail(7)["aqi"]
            w14     = p_hist.tail(14)["aqi"]
            recent_avg = round(float(w7.mean()), 1)
            prev7_avg  = float(w14.head(7).mean()) if len(w14) >= 7 else recent_avg
            pct        = ((recent_avg - prev7_avg) / max(prev7_avg, 1)) * 100
            trend_pct  = round(pct, 1)
            trend_icon = "↑" if pct > 5 else ("↓" if pct < -5 else "→")

    return PrimaryPollutantSummary(
        name=name, aqi=aqi, category=cat, color=color, info=info,
        recent_avg=recent_avg, trend_pct=trend_pct, trend_icon=trend_icon,
    )


# ── Per-pollutant stats ────────────────────────────────────────────────────────

def pollutant_stats_table(df_hist: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty or "pollutant" not in df_hist.columns:
        return pd.DataFrame()
    rows = []
    for pollutant, grp in df_hist.groupby("pollutant"):
        sg = grp.sort_values("date")
        r7 = sg.tail(7)["aqi"]
        p7 = sg.tail(14).head(7)["aqi"]
        pct = ((r7.mean() - p7.mean()) / max(p7.mean(), 1)) * 100 if len(p7) else 0.0
        arrow = "↑" if pct > 5 else ("↓" if pct < -5 else "→")
        rows.append({
            "Pollutant":      pollutant,
            "Mean AQI":       round(grp["aqi"].mean(), 1),
            "Peak AQI":       int(grp["aqi"].max()),
            "Best AQI":       int(grp["aqi"].min()),
            "Days > 100":     int((grp["aqi"] > 100).sum()),
            "7-Day Trend":    f"{arrow} {abs(pct):.1f}%",
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Peak AQI", ascending=False)
        .reset_index(drop=True)
    )
