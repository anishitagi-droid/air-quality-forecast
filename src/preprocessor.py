from __future__ import annotations

import pandas as pd

from config import AQI_CATEGORIES, POLLUTANT_LABELS


def aqi_category(aqi: int) -> tuple[str, str]:
    for ceiling, label, color in AQI_CATEGORIES:
        if aqi <= ceiling:
            return label, color
    return "Hazardous", "#7E0023"


def normalize_pollutant(raw: str) -> str:
    return POLLUTANT_LABELS.get(raw, raw)


def parse_historical(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for rec in records:
        aqi_val = rec.get("AQI", -1)
        if aqi_val < 0:
            continue
        label, color = aqi_category(int(aqi_val))
        raw_date = rec.get("_date") or rec.get("DateObserved", "")
        rows.append({
            "date":           pd.Timestamp(raw_date),
            "pollutant":      normalize_pollutant(rec.get("ParameterName", "Unknown")),
            "aqi":            int(aqi_val),
            "category":       label,
            "color":          color,
            "reporting_area": rec.get("ReportingArea", ""),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def daily_max_aqi(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    series = df.groupby("date")["aqi"].max().astype(float).rename("aqi")
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_idx)


def parse_current(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for rec in records:
        aqi_val = rec.get("AQI", -1)
        if aqi_val < 0:
            continue
        label, color = aqi_category(int(aqi_val))
        rows.append({
            "pollutant": normalize_pollutant(rec.get("ParameterName", "Unknown")),
            "aqi":       int(aqi_val),
            "category":  label,
            "color":     color,
        })
    return pd.DataFrame(rows)
