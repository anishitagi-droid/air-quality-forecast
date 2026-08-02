import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.demo_data import generate_demo_historical, generate_demo_current


def test_demo_historical_is_reproducible_across_repeated_calls_in_the_same_process():
    """Regression test for a real bug: a shared module-level RNG instance kept
    advancing its state across calls, so two calls to generate_demo_historical()
    in the same process produced DIFFERENT results despite the fixed seed=42 --
    verified concretely before fixing. This matters because Streamlit re-runs the
    whole script on every widget interaction but only imports each module once
    per server process, so demo data was silently changing on every interaction."""
    first = generate_demo_historical("10001", days=15)
    second = generate_demo_historical("10001", days=15)
    assert first == second


def test_demo_current_is_reproducible_across_repeated_calls():
    first = generate_demo_current("10001")
    second = generate_demo_current("10001")
    assert first == second


def test_different_seeds_produce_different_data():
    a = generate_demo_historical("10001", days=10, seed=1)
    b = generate_demo_historical("10001", days=10, seed=2)
    assert a != b


def test_demo_historical_covers_four_pollutants_across_the_requested_days():
    records = generate_demo_historical("10001", days=10)
    assert len(records) == 4 * 10
    pollutants = {r["ParameterName"] for r in records}
    assert pollutants == {"PM2.5", "O3", "NO2", "CO"}


def test_demo_data_stays_within_valid_aqi_bounds():
    records = generate_demo_historical("10001", days=30)
    assert all(0 <= r["AQI"] <= 500 for r in records)
