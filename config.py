from __future__ import annotations

AIRNOW_API_URL     = "https://www.airnowapi.org/aq"
HISTORY_DAYS       = 30
FORECAST_DAYS      = 7
DEFAULT_RADIUS_MILES = 25
PAGE_TITLE         = "Air Quality Forecast"
PAGE_ICON          = "🌬️"
LAYOUT             = "wide"

AQI_CATEGORIES: list[tuple[int, str, str]] = [
    (50,  "Good",               "#00E400"),
    (100, "Moderate",           "#FFFF00"),
    (150, "Unhealthy for Some", "#FF7E00"),
    (200, "Unhealthy",          "#FF0000"),
    (300, "Very Unhealthy",     "#8F3F97"),
    (500, "Hazardous",          "#7E0023"),
]

POLLUTANT_LABELS: dict[str, str] = {
    "O3":   "Ozone (O₃)",
    "PM2.5":"PM2.5",
    "PM10": "PM10",
    "CO":   "Carbon Monoxide",
    "NO2":  "Nitrogen Dioxide",
    "SO2":  "Sulfur Dioxide",
}
