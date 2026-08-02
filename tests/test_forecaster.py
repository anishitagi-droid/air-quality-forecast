import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forecaster import generate_forecast


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = 60 + rng.normal(0, 8, n)
    return pd.Series(values, index=dates)


def test_selects_arima_for_14_or_more_days():
    forecast = generate_forecast(_series(20), steps=5)
    assert forecast.method == "ARIMA(2,1,2)"


def test_selects_exponential_smoothing_for_7_to_13_days():
    forecast = generate_forecast(_series(10), steps=5)
    assert forecast.method == "Exponential Smoothing"


def test_selects_mean_baseline_for_fewer_than_7_days():
    forecast = generate_forecast(_series(4), steps=5)
    assert forecast.method == "7-Day Mean (baseline)"


def test_forecast_output_length_matches_requested_steps():
    for n, steps in [(20, 7), (10, 5), (4, 3)]:
        forecast = generate_forecast(_series(n), steps=steps)
        assert len(forecast.point) == steps
        assert len(forecast.lower) == steps
        assert len(forecast.upper) == steps
        assert len(forecast.dates) == steps


def test_forecast_values_are_clipped_to_valid_aqi_range():
    for n, steps in [(20, 7), (10, 5), (4, 3)]:
        forecast = generate_forecast(_series(n), steps=steps)
        assert (forecast.lower >= 0).all()
        assert (forecast.upper <= 500).all()


def test_confidence_interval_widens_with_forecast_horizon_for_exponential_smoothing():
    """Regression test for a real bug: the ETS forecast's confidence interval used a
    single flat sigma (the model's in-sample residual std) applied identically to
    every forecast step, so a 7-day-out prediction had the exact same interval width
    as tomorrow's -- statistically wrong, since multi-step forecasts should get less
    certain with distance. Now widens via sqrt(horizon), the standard approximation."""
    forecast = generate_forecast(_series(10), steps=7)
    assert forecast.method == "Exponential Smoothing"
    widths = forecast.upper - forecast.lower
    assert all(widths[i] <= widths[i + 1] + 1e-9 for i in range(len(widths) - 1))
    assert widths[-1] > widths[0]  # must actually grow, not just tie


def test_confidence_interval_widens_with_forecast_horizon_for_mean_baseline():
    forecast = generate_forecast(_series(4), steps=7)
    assert forecast.method == "7-Day Mean (baseline)"
    widths = forecast.upper - forecast.lower
    assert all(widths[i] <= widths[i + 1] + 1e-9 for i in range(len(widths) - 1))
    assert widths[-1] > widths[0]


def test_mean_baseline_with_a_single_data_point_does_not_crash():
    """A single-point series can't compute a real std -- must use the documented
    fallback sigma rather than raising or producing NaN."""
    forecast = generate_forecast(_series(1), steps=3)
    assert not np.isnan(forecast.point).any()
    assert not np.isnan(forecast.lower).any()
