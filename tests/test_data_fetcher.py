import sys
from pathlib import Path
from datetime import date
from unittest.mock import patch, MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_fetcher import (
    fetch_current, fetch_historical_day, fetch_historical_range, fetch_forecast,
    AirNowError, APIKeyMissingError,
)


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_fetch_current_requires_an_api_key():
    with pytest.raises(APIKeyMissingError):
        fetch_current("10001", "")


def test_fetch_historical_day_requires_an_api_key():
    with pytest.raises(APIKeyMissingError):
        fetch_historical_day("10001", "", date.today())


def test_fetch_forecast_requires_an_api_key():
    with pytest.raises(APIKeyMissingError):
        fetch_forecast("10001", "")


def test_get_raises_airnow_error_on_timeout():
    with patch("requests.get", side_effect=requests.exceptions.Timeout()):
        with pytest.raises(AirNowError, match="timed out"):
            fetch_current("10001", "fake-key")


def test_get_raises_airnow_error_on_http_error():
    with patch("requests.get", return_value=_mock_response([], status_code=500)):
        with pytest.raises(AirNowError, match="HTTP 500"):
            fetch_current("10001", "fake-key")


def test_get_raises_airnow_error_on_unexpected_response_shape():
    with patch("requests.get", return_value=_mock_response({"not": "a list"})):
        with pytest.raises(AirNowError, match="Unexpected API response format"):
            fetch_current("10001", "fake-key")


def test_fetch_historical_range_aggregates_all_days_and_stamps_each_with_its_date():
    """Verifies the parallelized version (ThreadPoolExecutor) still produces the
    same end result as the original sequential loop: every day's records present,
    each correctly stamped with its own _date."""
    call_count = 0

    def fake_get(url, params, timeout):
        nonlocal call_count
        call_count += 1
        return _mock_response([{"ParameterName": "O3", "AQI": 50}])

    with patch("requests.get", side_effect=fake_get):
        records = fetch_historical_range("10001", "fake-key", days=10)

    assert call_count == 10
    assert len(records) == 10
    dates = {r["_date"] for r in records}
    assert len(dates) == 10  # every day distinct, none dropped or duplicated


def test_fetch_historical_range_skips_a_failing_day_without_losing_the_rest():
    """Regression test for the parallelization change: a single day's request
    failing must not abort the whole range (matching the original sequential
    behavior's per-day try/except), and the other 9 days' data must still come
    back even though they complete concurrently in a non-deterministic order."""
    from datetime import timedelta

    failing_date_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%dT00-0000")

    def fake_get(url, params, timeout):
        if params["date"] == failing_date_str:
            raise requests.exceptions.Timeout()
        return _mock_response([{"ParameterName": "O3", "AQI": 50}])

    with patch("requests.get", side_effect=fake_get):
        records = fetch_historical_range("10001", "fake-key", days=10)

    assert len(records) == 9  # 10 requested, 1 failed and was skipped
