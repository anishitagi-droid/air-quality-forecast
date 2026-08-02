from __future__ import annotations

from datetime import date, timedelta

import numpy as np

# (api_code, base_aqi, noise, spike_prob, spike_magnitude)
_PROFILES = [
    ("PM2.5", 58.0, 16.0, 0.06, 60.0),
    ("O3",    48.0, 14.0, 0.04, 45.0),
    ("NO2",   38.0, 10.0, 0.03, 30.0),
    ("CO",    20.0,  6.0, 0.02, 20.0),
]


def _simulate(rng: np.random.Generator, days: int, base: float, noise: float,
              spike_prob: float, spike_mag: float) -> np.ndarray:
    values = np.empty(days)
    val = base
    for i in range(days):
        val = 0.82 * val + 0.18 * base + rng.normal(0, noise)
        if rng.random() < spike_prob:
            val += rng.uniform(spike_mag * 0.5, spike_mag)
        values[i] = np.clip(val, 8, 250)
    return values.round(1)


def generate_demo_historical(zip_code: str = "00000", days: int = 30, seed: int = 42) -> list[dict]:
    # A fresh generator per call, not a shared module-level one: Streamlit
    # re-runs this whole script on every widget interaction but only imports
    # each module once per server process, so a shared, mutating RNG instance
    # kept advancing across "reruns" within the same process -- the seed
    # clearly intended reproducible demo data, but two calls in the same
    # process produced different AQI values every time. Verified concretely.
    rng = np.random.default_rng(seed)
    today = date.today()
    records = []
    for api_code, base, noise, sp, sm in _PROFILES:
        series = _simulate(rng, days, base, noise, sp, sm)
        for delta, aqi_val in enumerate(series, start=1):
            obs_date = today - timedelta(days=delta)
            records.append({
                "_date":         obs_date.isoformat(),
                "ParameterName": api_code,
                "AQI":           int(aqi_val),
                "ReportingArea": f"Demo Area ({zip_code})",
            })
    return records


def generate_demo_current(zip_code: str = "00000", seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    aqis = {
        "PM2.5": int(rng.integers(45, 110)),
        "O3":    int(rng.integers(30,  95)),
        "NO2":   int(rng.integers(20,  70)),
        "CO":    int(rng.integers(10,  45)),
    }
    return [
        {"ParameterName": code, "AQI": aqi, "ReportingArea": f"Demo Area ({zip_code})"}
        for code, aqi in aqis.items()
    ]
