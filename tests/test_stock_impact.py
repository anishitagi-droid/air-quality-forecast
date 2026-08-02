import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stock_impact import get_all_impacts, get_impact_score


def test_every_sector_has_a_sensitivity_score():
    """Regression test for a real bug: app.py used to hardcode a bare positional
    list of 9 sensitivity numbers matched to get_all_impacts() purely by index,
    with no named link to which sector each value described -- one add/remove/
    reorder of a sector here would have silently misaligned the chart with the
    wrong sectors. sensitivity_score is now a real field on SectorImpact itself."""
    for sector in get_all_impacts():
        assert isinstance(sector.sensitivity_score, int)
        assert -10 <= sector.sensitivity_score <= 10


def test_sensitivity_score_sign_matches_expected_impact_direction():
    for sector in get_all_impacts():
        if sector.expected_impact == "positive":
            assert sector.sensitivity_score > 0, sector.sector
        elif sector.expected_impact == "negative":
            assert sector.sensitivity_score < 0, sector.sector


def test_get_impact_score_counts_sectors_crossing_the_current_aqi_threshold():
    score_low_aqi = get_impact_score(10)
    score_high_aqi = get_impact_score(400)
    assert score_high_aqi["total_active"] >= score_low_aqi["total_active"]


def test_get_all_impacts_returns_nine_sectors_with_unique_names():
    sectors = get_all_impacts()
    names = [s.sector for s in sectors]
    assert len(names) == len(set(names))
    assert len(sectors) == 9
