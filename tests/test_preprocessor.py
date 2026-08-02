import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessor import aqi_category, normalize_pollutant, parse_historical, parse_current, daily_max_aqi


def test_aqi_category_boundaries_match_epa_breakpoints():
    assert aqi_category(0) == ("Good", "#00E400")
    assert aqi_category(50) == ("Good", "#00E400")
    assert aqi_category(51) == ("Moderate", "#FFFF00")
    assert aqi_category(100) == ("Moderate", "#FFFF00")
    assert aqi_category(101) == ("Unhealthy for Some", "#FF7E00")
    assert aqi_category(500) == ("Hazardous", "#7E0023")


def test_aqi_category_above_500_still_returns_hazardous():
    assert aqi_category(600) == ("Hazardous", "#7E0023")


def test_normalize_pollutant_maps_api_codes_to_labels():
    assert normalize_pollutant("O3") == "Ozone (O₃)"
    assert normalize_pollutant("PM2.5") == "PM2.5"


def test_normalize_pollutant_passes_through_unknown_codes_unchanged():
    assert normalize_pollutant("SOMETHING_NEW") == "SOMETHING_NEW"


def test_parse_historical_empty_input_returns_empty_dataframe_not_a_crash():
    df = parse_historical([])
    assert df.empty


def test_parse_historical_drops_negative_aqi_values():
    records = [
        {"_date": "2024-01-01", "ParameterName": "O3", "AQI": 45, "ReportingArea": "Test"},
        {"_date": "2024-01-01", "ParameterName": "PM2.5", "AQI": -999, "ReportingArea": "Test"},
    ]
    df = parse_historical(records)
    assert (df["aqi"] >= 0).all()
    assert len(df) == 1


def test_daily_max_aqi_takes_the_worst_pollutant_per_day():
    records = [
        {"_date": "2024-01-01", "ParameterName": "O3", "AQI": 40, "ReportingArea": "Test"},
        {"_date": "2024-01-01", "ParameterName": "PM2.5", "AQI": 85, "ReportingArea": "Test"},
    ]
    df = parse_historical(records)
    daily = daily_max_aqi(df)
    assert daily.iloc[0] == 85


def test_parse_current_normalizes_pollutant_names():
    records = [{"ParameterName": "O3", "AQI": 60, "ReportingArea": "Test"}]
    df = parse_current(records)
    assert df.iloc[0]["pollutant"] == "Ozone (O₃)"
