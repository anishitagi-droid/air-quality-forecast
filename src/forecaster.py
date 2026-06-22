from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_OBS_ARIMA = 14
MIN_OBS_ETS   = 7


@dataclass
class ForecastResult:
    dates:  pd.DatetimeIndex
    point:  np.ndarray
    lower:  np.ndarray
    upper:  np.ndarray
    method: str


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 500).round(1)


def _arima(series: pd.Series, steps: int) -> ForecastResult:
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(series.dropna(), order=(2, 1, 2)).fit()
        pred   = fitted.get_forecast(steps=steps)
    ci    = pred.conf_int(alpha=0.05)
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return ForecastResult(
        dates=dates,
        point=_clip(pred.predicted_mean.values),
        lower=_clip(ci.iloc[:, 0].values),
        upper=_clip(ci.iloc[:, 1].values),
        method="ARIMA(2,1,2)",
    )


def _ets(series: pd.Series, steps: int) -> ForecastResult:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SimpleExpSmoothing(
            series.dropna(), initialization_method="estimated"
        ).fit()
    point = fitted.forecast(steps)
    sigma = fitted.resid.std()
    z     = 1.96
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    return ForecastResult(
        dates=dates,
        point=_clip(point.values),
        lower=_clip(point.values - z * sigma),
        upper=_clip(point.values + z * sigma),
        method="Exponential Smoothing",
    )


def _mean_baseline(series: pd.Series, steps: int) -> ForecastResult:
    tail  = series.dropna().tail(7)
    mu    = float(tail.mean())
    sigma = float(tail.std()) if len(tail) > 1 else 5.0
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    pt    = np.full(steps, mu)
    return ForecastResult(
        dates=dates,
        point=_clip(pt),
        lower=_clip(pt - 1.96 * sigma),
        upper=_clip(pt + 1.96 * sigma),
        method="7-Day Mean (baseline)",
    )


def generate_forecast(series: pd.Series, steps: int = 7) -> ForecastResult:
    n = series.dropna().shape[0]
    if n >= MIN_OBS_ARIMA:
        try:
            return _arima(series, steps)
        except Exception as exc:
            log.warning("ARIMA failed (%s); trying ETS.", exc)
    if n >= MIN_OBS_ETS:
        try:
            return _ets(series, steps)
        except Exception as exc:
            log.warning("ETS failed (%s); using mean.", exc)
    return _mean_baseline(series, steps)
