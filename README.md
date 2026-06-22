# 🌬️ Air Quality Forecast

A Streamlit dashboard that fetches real-time AQI data from the **EPA AirNow API**, runs a time-series forecast, surfaces health advisories, and maps air quality conditions to sector-level market impact signals.

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

- **Live AQI data** from EPA AirNow (or demo mode — no key required)
- **Primary pollutant panel** — dominant pollutant with 7-day trend and prior-week delta
- **Health advisory** — EPA-based guidance for general public, sensitive groups, outdoor activity
- **Trend analysis** — direction, streak detection, volatility, best/worst day
- **7-day forecast** — ARIMA → ETS → mean fallback chain with 95% confidence intervals
- **Market impact analysis** — 9 sectors, 30+ tickers, backed by peer-reviewed research

---

## Quickstart

```bash
git clone https://github.com/your-username/air-quality-forecast.git
cd air-quality-forecast

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**.

Get a free AirNow API key at **https://docs.airnowapi.org/** — takes ~2 minutes.  
Leave the key blank to explore the UI with synthetic demo data.

---

## Project Structure

```
air-quality-forecast/
├── app.py               ← Streamlit UI and orchestration
├── config.py            ← Constants: AQI categories, API endpoints
├── requirements.txt
├── .gitignore
├── data/
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── data_fetcher.py  ← EPA AirNow API client
    ├── preprocessor.py  ← Raw JSON → tidy DataFrames
    ├── forecaster.py    ← ARIMA / ETS / mean fallback chain
    ├── demo_data.py     ← Synthetic data for demo mode
    ├── analytics.py     ← Trend analysis, health advisories, primary pollutant
    └── stock_impact.py  ← Sector/ticker market impact analysis
```

---

## Forecasting Model

| Available data | Model |
|---|---|
| ≥ 14 days | ARIMA(2,1,2) via `statsmodels` |
| 7–13 days | Simple Exponential Smoothing |
| < 7 days  | Trailing 7-day mean (baseline) |

Predictions are clipped to [0, 500]. Bounds are 95% prediction intervals.

---

## Market Impact Research

Sector-level relationships are derived from:

- Graff Zivin & Neidell (2012, *AER*) — pollution and worker productivity
- Knittel et al. (2020) — wildfire smoke and financial markets
- Hong et al. (2019, *Journal of Finance*) — pollution and analyst accuracy
- USDA / NBER working papers on crop yield and hospital revenue

**This is educational analysis, not investment advice.**

---

## AQI Categories

| Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

Source: [EPA AQI Basics](https://www.airnow.gov/aqi/aqi-basics/)

---

## License

MIT
