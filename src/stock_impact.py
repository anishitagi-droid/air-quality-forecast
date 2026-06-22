from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Impact = Literal["positive", "negative", "neutral", "mixed"]


@dataclass(frozen=True)
class TickerImpact:
    ticker: str
    name: str
    expected_impact: Impact
    rationale: str
    sensitivity: str   # "high" | "medium" | "low"


@dataclass(frozen=True)
class SectorImpact:
    sector: str
    icon: str
    expected_impact: Impact
    short_rationale: str
    detail: str
    tickers: list[TickerImpact]
    aqi_threshold: int
    confidence: str   # "strong" | "moderate" | "weak"


SECTOR_IMPACTS: list[SectorImpact] = [
    SectorImpact(
        sector="Healthcare & Pharma",
        icon="🏥",
        expected_impact="positive",
        short_rationale="Higher hospitalizations drive revenue for respiratory drug makers and hospital networks.",
        detail=(
            "Poor air quality days directly increase ER visits and admissions for asthma, COPD, "
            "and cardiovascular events. Drug manufacturers of bronchodilators and corticosteroids "
            "see prescription upticks. NBER (2022) found a statistically significant link between "
            "PM2.5 events and hospital net revenue."
        ),
        tickers=[
            TickerImpact("AZN",  "AstraZeneca",    "positive", "Breztri/Fasenra asthma biologic demand", "high"),
            TickerImpact("GSK",  "GSK",             "positive", "Trelegy Ellipta COPD demand",             "high"),
            TickerImpact("HCA",  "HCA Healthcare",  "positive", "ER visit volume and admissions rise",     "medium"),
            TickerImpact("THC",  "Tenet Healthcare","positive", "Hospital admission revenue uplift",        "medium"),
        ],
        aqi_threshold=100,
        confidence="strong",
    ),
    SectorImpact(
        sector="Air Filtration",
        icon="💨",
        expected_impact="positive",
        short_rationale="Consumer and industrial air purifier demand spikes during high-AQI events.",
        detail=(
            "Persistent poor air quality drives double-digit sales growth for residential purifiers. "
            "Industrial filtration companies benefit from regulatory tightening after sustained "
            "pollution events. The 2020–2023 wildfire seasons produced multi-quarter revenue uplifts."
        ),
        tickers=[
            TickerImpact("CECO", "CECO Environmental", "positive", "Industrial air quality systems", "high"),
            TickerImpact("CLFD", "Clearfield",         "positive", "Environmental infrastructure",   "medium"),
            TickerImpact("PESI", "Perma-Pipe",         "positive", "Environmental remediation",      "medium"),
        ],
        aqi_threshold=100,
        confidence="strong",
    ),
    SectorImpact(
        sector="Clean Energy & EVs",
        icon="⚡",
        expected_impact="positive",
        short_rationale="Air quality crises accelerate policy momentum and consumer interest in clean alternatives.",
        detail=(
            "Sustained poor air events correlate with EV purchase intent increases and solar "
            "installation inquiries. California wildfire seasons drove record rooftop solar adoption. "
            "Investors rotate toward ESG-aligned names during high-visibility pollution events."
        ),
        tickers=[
            TickerImpact("TSLA", "Tesla",         "positive", "EV demand sentiment boost",            "medium"),
            TickerImpact("ENPH", "Enphase Energy", "positive", "Rooftop solar install interest uptick","medium"),
            TickerImpact("NEE",  "NextEra Energy", "positive", "Regulatory tailwinds",                 "low"),
        ],
        aqi_threshold=100,
        confidence="weak",
    ),
    SectorImpact(
        sector="Outdoor Recreation",
        icon="🏔️",
        expected_impact="negative",
        short_rationale="Foot traffic, park visits, and outdoor spending fall sharply on high-AQI days.",
        detail=(
            "Stanford research shows high-pollution days reduce outdoor activity 7–12% even among "
            "healthy adults. Theme parks, sporting goods, and outdoor hospitality see immediate "
            "foot traffic suppression. Nike has noted ozone events affect same-store running gear sales."
        ),
        tickers=[
            TickerImpact("NKE",  "Nike",              "negative", "Outdoor retail traffic drops",      "high"),
            TickerImpact("COLM", "Columbia Sportswear","negative", "Outdoor recreation demand falls",  "high"),
            TickerImpact("SIX",  "Six Flags",          "negative", "Theme park attendance drops",      "high"),
            TickerImpact("BGFV", "Big 5 Sporting",     "negative", "Foot traffic and outdoor sales",   "medium"),
        ],
        aqi_threshold=100,
        confidence="strong",
    ),
    SectorImpact(
        sector="Airlines & Travel",
        icon="✈️",
        expected_impact="negative",
        short_rationale="Wildfire smoke disrupts visibility minimums; sustained events suppress leisure bookings.",
        detail=(
            "FAA visibility minimums are violated by smoke events, causing ground stops. Surveys "
            "show leisure booking intent drops 15–25% in regions with AQI > 150 over 3+ days. "
            "Corporate travel managers also defer discretionary travel during high-alert periods."
        ),
        tickers=[
            TickerImpact("AAL",  "American Airlines","negative", "Ground stops and demand drag",         "high"),
            TickerImpact("DAL",  "Delta Air Lines",  "negative", "Disruption and booking softness",      "high"),
            TickerImpact("UAL",  "United Airlines",  "negative", "Smoke-related holds and delays",       "high"),
            TickerImpact("BKNG", "Booking Holdings", "negative", "Leisure travel intent drops",          "medium"),
        ],
        aqi_threshold=150,
        confidence="moderate",
    ),
    SectorImpact(
        sector="Agriculture & Food",
        icon="🌾",
        expected_impact="negative",
        short_rationale="Ozone is a primary crop yield suppressor; PM2.5 reduces photosynthetic efficiency.",
        detail=(
            "USDA estimates ozone at current US levels reduces soybean yields 5–15% and wheat 5–10% "
            "annually. Wildfires add crop damage through ash deposition and reduced solar radiation, "
            "directly impacting commodity processors and equipment demand."
        ),
        tickers=[
            TickerImpact("ADM", "Archer-Daniels-Midland","negative","Crop yield suppression from ozone","high"),
            TickerImpact("BG",  "Bunge Global",           "negative","Agricultural commodity volume",    "medium"),
            TickerImpact("DE",  "Deere & Company",        "negative","Lower crop revenue cuts equipment","low"),
        ],
        aqi_threshold=100,
        confidence="moderate",
    ),
    SectorImpact(
        sector="Insurance",
        icon="🛡️",
        expected_impact="mixed",
        short_rationale="Claims rise but so do premiums — net effect varies heavily by geographic exposure.",
        detail=(
            "Property-casualty insurers face direct wildfire losses. Health insurers absorb higher "
            "respiratory claims. Reinsurers that have already re-priced climate risk may benefit "
            "from rising premium income. Large dispersion between names based on geographic exposure."
        ),
        tickers=[
            TickerImpact("ALL", "Allstate",      "negative", "California wildfire losses",           "high"),
            TickerImpact("UNH", "UnitedHealth",  "negative", "Respiratory claims volume increase",   "medium"),
            TickerImpact("CB",  "Chubb",         "mixed",    "High exposure vs premium pricing power","medium"),
        ],
        aqi_threshold=150,
        confidence="moderate",
    ),
    SectorImpact(
        sector="Real Estate",
        icon="🏢",
        expected_impact="negative",
        short_rationale="Chronic poor air quality measurably depresses property values and housing demand.",
        detail=(
            "Harvard research found a 1 µg/m³ increase in long-run PM2.5 is associated with a "
            "0.5–1.5% reduction in home values. Migration patterns from high-AQI regions show "
            "sustained impact on housing market dynamics and insurance availability."
        ),
        tickers=[
            TickerImpact("RDFN", "Redfin",             "negative","Migration shifts from high-AQI areas","medium"),
            TickerImpact("EQR",  "Equity Residential",  "negative","Urban apartment demand suppressed",   "low"),
            TickerImpact("OPEN", "Opendoor",            "negative","Home buying intent drops",             "medium"),
        ],
        aqi_threshold=150,
        confidence="moderate",
    ),
    SectorImpact(
        sector="Worker Productivity",
        icon="🧠",
        expected_impact="negative",
        short_rationale="Air quality measurably reduces cognitive performance, suppressing output-driven revenue.",
        detail=(
            "Graff Zivin & Neidell (2012, AER) found a 10 µg/m³ PM2.5 increase reduces worker "
            "productivity 1.5% same-day. A 2019 Journal of Finance paper found pollution impairs "
            "analyst forecast accuracy. Amazon benefits from increased home delivery demand."
        ),
        tickers=[
            TickerImpact("AMZN", "Amazon",  "positive", "Home delivery demand rises as people stay in","low"),
            TickerImpact("AAPL", "Apple",   "negative", "Retail traffic and campus productivity fall",  "low"),
            TickerImpact("GOOG", "Alphabet","neutral",  "Some ad demand impact; mostly indoor work",    "low"),
        ],
        aqi_threshold=100,
        confidence="weak",
    ),
]


def get_active_impacts(aqi: int) -> list[SectorImpact]:
    return [s for s in SECTOR_IMPACTS if aqi >= s.aqi_threshold]


def get_all_impacts() -> list[SectorImpact]:
    return SECTOR_IMPACTS


def get_impact_score(aqi: int) -> dict[str, int]:
    active = get_active_impacts(aqi)
    counts: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    for s in active:
        counts[s.expected_impact] += 1
    counts["total_active"] = len(active)
    return counts
