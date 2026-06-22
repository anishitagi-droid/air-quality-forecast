from __future__ import annotations

from datetime import date, timedelta

import numpy as np

_RNG = np.random.default_rng(42)

# (api_code, base_aqi, noise, spike_prob, spike_magnitude)
_PROFILES = [
    ("PM2.5", 58.0, 16.0, 0.06, 60.0),
    ("O3",    48.0, 14.0, 0.04, 45.0),
    ("NO2",   38.0, 10.0, 0.03, 30.0),
    ("CO",    20.0,  6.0, 0.02, 20.0),
]


def _simulate(days: int, base: float, noise: float,
              spike_prob: float, spike_mag: float) -> np.ndarray:
    values = np.empty(days)
    val = base
    for i in range(days):
        val = 0.82 * val + 0.18 * base + _RNG.normal(0, noise)
        if _RNG.random() < spike_prob:
            val += _RNG.uniform(spike_mag * 0.5, spike_mag)
        values[i] = np.clip(val, 8, 250)
    return values.round(1)


def generate_demo_historical(zip_code: str = "00000", days: int = 30) -> list[dict]:
    today = date.today()
    records = []
    for api_code, base, noise, sp, sm in _PROFILES:
        series = _simulate(days, base, noise, sp, sm)
        for delta, aqi_val in enumerate(series, start=1):
            obs_date = today - timedelta(days=delta)
            records.append({
                "_date":         obs_date.isoformat(),
                "ParameterName": api_code,
                "AQI":           int(aqi_val),
                "ReportingArea": f"Demo Area ({zip_code})",
            })
    return records


def generate_demo_current(zip_code: str = "00000") -> list[dict]:
    aqis = {
        "PM2.5": int(_RNG.integers(45, 110)),
        "O3":    int(_RNG.integers(30,  95)),
        "NO2":   int(_RNG.integers(20,  70)),
        "CO":    int(_RNG.integers(10,  45)),
    }
    return [
        {"ParameterName": code, "AQI": aqi, "ReportingArea": f"Demo Area ({zip_code})"}
        for code, aqi in aqis.items()
    ]
