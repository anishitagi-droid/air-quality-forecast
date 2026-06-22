from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import requests

from config import AIRNOW_API_URL, DEFAULT_RADIUS_MILES

log = logging.getLogger(__name__)


class AirNowError(Exception):
    """Raised when the AirNow API returns an unexpected response."""


class APIKeyMissingError(AirNowError):
    """Raised when no API key is provided."""


def _get(url: str, params: dict) -> list[dict]:
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise AirNowError("Request timed out. Check your connection.")
    except requests.exceptions.HTTPError as exc:
        raise AirNowError(f"API returned HTTP {exc.response.status_code}.") from exc
    except requests.exceptions.RequestException as exc:
        raise AirNowError(f"Network error: {exc}") from exc

    data = r.json()
    if not isinstance(data, list):
        raise AirNowError(f"Unexpected API response format: {type(data)}")
    return data


def fetch_current(zip_code: str, api_key: str) -> list[dict]:
    if not api_key:
        raise APIKeyMissingError("An AirNow API key is required.")
    return _get(
        f"{AIRNOW_API_URL}/observation/zipCode/current/",
        {"format": "application/json", "zipCode": zip_code,
         "distance": DEFAULT_RADIUS_MILES, "API_KEY": api_key},
    )


def fetch_historical_day(zip_code: str, api_key: str, obs_date: date) -> list[dict]:
    if not api_key:
        raise APIKeyMissingError("An AirNow API key is required.")
    date_str = obs_date.strftime("%Y-%m-%dT00-0000")
    records = _get(
        f"{AIRNOW_API_URL}/observation/zipCode/historical/",
        {"format": "application/json", "zipCode": zip_code,
         "date": date_str, "distance": DEFAULT_RADIUS_MILES, "API_KEY": api_key},
    )
    for rec in records:
        rec["_date"] = obs_date.isoformat()
    return records


def fetch_historical_range(zip_code: str, api_key: str, days: int = 30) -> list[dict]:
    records: list[dict] = []
    today = date.today()
    for delta in range(1, days + 1):
        target = today - timedelta(days=delta)
        try:
            records.extend(fetch_historical_day(zip_code, api_key, target))
        except AirNowError as exc:
            log.warning("Skipping %s: %s", target, exc)
    return records


def fetch_forecast(
    zip_code: str, api_key: str, forecast_date: Optional[date] = None
) -> list[dict]:
    if not api_key:
        raise APIKeyMissingError("An AirNow API key is required.")
    target = forecast_date or (date.today() + timedelta(days=1))
    return _get(
        f"{AIRNOW_API_URL}/forecast/zipCode/",
        {"format": "application/json", "zipCode": zip_code,
         "date": target.strftime("%Y-%m-%d"),
         "distance": DEFAULT_RADIUS_MILES, "API_KEY": api_key},
    )
