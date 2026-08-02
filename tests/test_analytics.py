import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics import get_health_advisory, compute_trend
from config import AQI_CATEGORIES


def test_health_advisory_uses_the_same_breakpoints_as_config():
    """Regression/consistency test for a real DRY violation: analytics.py used to
    define its own separate copy of the (ceiling, category, color) breakpoint
    table instead of reusing config.AQI_CATEGORIES, with the two tables verified
    byte-for-byte identical at the time -- a latent risk that they'd silently
    drift apart on a future edit to just one of them. Checks every category
    boundary explicitly, not just a couple of spot values."""
    for ceiling, expected_category, expected_color in AQI_CATEGORIES:
        advisory = get_health_advisory(ceiling)
        assert advisory.category == expected_category
        assert advisory.color == expected_color


def test_health_advisory_has_text_for_every_category():
    for aqi in [25, 75, 125, 175, 250, 400]:
        advisory = get_health_advisory(aqi)
        assert advisory.general
        assert advisory.sensitive
        assert advisory.outdoor_activity
        assert advisory.icon


def test_compute_trend_direction_reflects_actual_change():
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    improving = pd.Series([100 - i * 3 for i in range(14)], index=dates)
    trend = compute_trend(improving)
    assert trend.direction == "improving"
