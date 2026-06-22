"""
app.py  ·  Air Quality Forecast
================================
Run:  streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import config
from src.data_fetcher import fetch_historical_range, fetch_current, AirNowError
from src.preprocessor  import parse_historical, parse_current, daily_max_aqi, aqi_category
from src.forecaster    import generate_forecast
from src.demo_data     import generate_demo_historical, generate_demo_current
from src.analytics     import (
    compute_trend,
    identify_primary_pollutant,
    get_health_advisory,
    pollutant_stats_table,
)
from src.stock_impact  import get_all_impacts, get_impact_score

logging.basicConfig(level=logging.WARNING)

# ── Page ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Layout ── */
.block-container { padding: 1.75rem 2rem 3rem; max-width: 1400px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background : var(--background-color, #f9fafb);
    border     : 1px solid rgba(0,0,0,0.07);
    border-radius: 10px;
    padding    : 1rem 1.1rem;
}
[data-testid="metric-container"] label {
    font-size       : 0.68rem !important;
    font-weight     : 600 !important;
    letter-spacing  : 0.09em;
    text-transform  : uppercase;
    color           : #6b7280 !important;
}
[data-testid="stMetricValue"] {
    font-size   : 1.85rem !important;
    font-weight : 600 !important;
    color       : #111827 !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Section header ── */
.s-hdr {
    font-size      : 0.65rem;
    font-weight    : 700;
    letter-spacing : 0.13em;
    text-transform : uppercase;
    color          : #9ca3af;
    border-bottom  : 1px solid #e5e7eb;
    padding-bottom : 5px;
    margin         : 0.25rem 0 1rem;
}

/* ── Primary-pollutant hero card ── */
.hero {
    border-radius : 12px;
    padding       : 1.6rem 1.8rem;
    position      : relative;
    overflow      : hidden;
}
.hero::after {
    content       : "";
    position      : absolute;
    inset         : 0;
    background    : linear-gradient(135deg, rgba(255,255,255,0.14), transparent 60%);
    border-radius : 12px;
    pointer-events: none;
}
.hero-tag  { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
             text-transform: uppercase; opacity: 0.65; margin-bottom: 7px; }
.hero-name { font-size: 1.15rem; font-weight: 600; line-height: 1.3; margin-bottom: 3px; }
.hero-aqi  { font-size: 4rem; font-weight: 700; line-height: 1;
             letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.hero-cat  { font-size: 1rem; font-weight: 500; opacity: 0.82; margin-top: 2px; }
.hero-trd  { font-size: 0.8rem; opacity: 0.65; margin-top: 10px;
             font-variant-numeric: tabular-nums; }

/* ── Advisory card ── */
.adv-card  {
    background    : #ffffff;
    border        : 1px solid #e5e7eb;
    border-radius : 12px;
    padding       : 1.2rem 1.4rem;
    height        : 100%;
}
.adv-title { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em;
             text-transform: uppercase; color: #6b7280; margin-bottom: 10px; }
.adv-row   { font-size: 0.875rem; color: #374151; line-height: 1.6; margin-bottom: 8px; }
.adv-row strong { color: #111827; font-weight: 600; }

/* ── Stat blocks ── */
.stat-blk  {
    background    : #f9fafb;
    border        : 1px solid #e5e7eb;
    border-radius : 10px;
    padding       : 0.85rem 1rem;
    height        : 100%;
}
.stat-lbl  { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.09em;
             text-transform: uppercase; color: #9ca3af; margin-bottom: 5px; }
.stat-val  { font-size: 0.88rem; color: #111827; line-height: 1.55; }

/* ── Trend pills ── */
.pill       { display: inline-flex; align-items: center; gap: 3px;
              padding: 3px 10px; border-radius: 999px;
              font-size: 0.75rem; font-weight: 600; }
.pill-red   { background: #fee2e2; color: #991b1b; }
.pill-green { background: #dcfce7; color: #15803d; }
.pill-gray  { background: #f3f4f6; color: #6b7280; }

/* ── Pollutant current cards ── */
.pc-wrap {
    border-radius : 10px;
    padding       : 1rem;
    text-align    : center;
}
.pc-name { font-size: 0.72rem; font-weight: 600; margin-bottom: 4px; }
.pc-aqi  { font-size: 2.25rem; font-weight: 700; line-height: 1;
           font-variant-numeric: tabular-nums; }
.pc-cat  { font-size: 0.7rem; opacity: 0.82; margin-top: 3px; }

/* ── Sector cards ── */
.sector-card {
    background    : #ffffff;
    border        : 1px solid #e5e7eb;
    border-radius : 10px;
    padding       : 1rem 1.1rem;
    margin-bottom : 10px;
}
.sector-head   { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px; }
.sector-icon   { font-size: 1.4rem; line-height: 1; flex-shrink: 0; margin-top: 1px; }
.sector-name   { font-size: 0.9rem; font-weight: 600; color: #111827; }
.sector-meta   { font-size: 0.7rem; color: #9ca3af; margin-top: 1px; }
.sector-detail { font-size: 0.82rem; color: #374151; line-height: 1.6; margin-bottom: 9px; }
.ticker-row    { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 6px; }
.t-pill        { display: inline-flex; align-items: center; gap: 4px;
                 padding: 3px 9px; border-radius: 6px;
                 background: #f3f4f6; border: 1px solid #e5e7eb;
                 font-size: 0.75rem; color: #111827; }
.t-pill-name   { color: #6b7280; font-size: 0.7rem; }
.conf-line     { font-size: 0.7rem; color: #9ca3af; margin-top: 5px; }

/* ── Disclaimer ── */
.disclaimer {
    background    : #fffbeb;
    border-left   : 3px solid #f59e0b;
    border-radius : 0 6px 6px 0;
    padding       : 0.6rem 0.9rem;
    font-size     : 0.78rem;
    color         : #92400e;
    margin-bottom : 1rem;
}

/* ── Tab tweaks ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"]      { font-size: 0.82rem; font-weight: 500;
                                    padding: 5px 14px; border-radius: 6px; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌬️ Air Quality Forecast")
    st.caption("Powered by EPA AirNow · Free API")
    st.divider()

    st.markdown("**ZIP Code**")
    zip_code = st.text_input("ZIP", value="10001", max_chars=5,
                              label_visibility="collapsed")

    st.markdown("**AirNow API Key**")
    api_key = st.text_input(
        "Key", type="password",
        placeholder="Leave blank for demo mode",
        label_visibility="collapsed",
        help="Free at https://docs.airnowapi.org/",
    )

    st.divider()

    st.markdown("**History window**")
    history_days = st.slider("History", 7, 60, config.HISTORY_DAYS,
                              label_visibility="collapsed", format="%d days")

    st.markdown("**Forecast horizon**")
    forecast_days = st.slider("Forecast", 3, 14, config.FORECAST_DAYS,
                               label_visibility="collapsed", format="%d days")

    st.divider()
    run = st.button("Fetch & Forecast", use_container_width=True, type="primary")

    if not api_key:
        st.info("No API key — running in **demo mode**.", icon="ℹ️")

    st.markdown(
        "<div style='margin-top:1.5rem; font-size:0.72rem; color:#9ca3af; line-height:2;'>"
        "<a href='https://docs.airnowapi.org/' style='color:#6b7280;'>Get a free API key</a><br>"
        "<a href='https://www.airnow.gov/aqi/aqi-basics/' style='color:#6b7280;'>AQI basics (EPA)</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _txt(aqi: int) -> str:
    """White text for dark AQI backgrounds, dark text for light ones."""
    return "#ffffff" if aqi >= 150 else "#111827"


def _fmt_date(d) -> str:
    """Cross-platform day formatting (no zero-padding). Linux: %-d, Windows: %#d."""
    import platform
    fmt = "%b %#d" if platform.system() == "Windows" else "%b %-d"
    return d.strftime(fmt)


def _fmt_date_long(d) -> str:
    """E.g. 'Monday, Jun 9'. Cross-platform — no zero-padded day."""
    import platform
    fmt = "%A, %b %#d" if platform.system() == "Windows" else "%A, %b %-d"
    return d.strftime(fmt)


def _pill(label: str, kind: str = "gray") -> str:
    cls = {"red": "pill-red", "green": "pill-green", "gray": "pill-gray"}.get(kind, "pill-gray")
    return f'<span class="pill {cls}">{label}</span>'


def _impact_pill(impact: str) -> str:
    mapping = {
        "positive": ("green", "↑ Positive"),
        "negative": ("red",   "↓ Negative"),
        "mixed":    ("gray",  "~ Mixed"),
        "neutral":  ("gray",  "→ Neutral"),
    }
    kind, label = mapping.get(impact, ("gray", impact.title()))
    return _pill(label, kind)


def _conf_color(conf: str) -> str:
    return {"strong": "#15803d", "moderate": "#b45309", "weak": "#6b7280"}.get(conf, "#6b7280")


@st.cache_data(show_spinner=False)
def load_data(zip_code: str, api_key: str, history_days: int):
    if api_key:
        raw_hist    = fetch_historical_range(zip_code, api_key, days=history_days)
        raw_current = fetch_current(zip_code, api_key)
    else:
        raw_hist    = generate_demo_historical(zip_code, days=history_days)
        raw_current = generate_demo_current(zip_code)
    return parse_historical(raw_hist), parse_current(raw_current)


def _aqi_zone_shapes() -> list[dict]:
    zones = [
        (0,   50,  "rgba(0,228,0,0.055)"),
        (50,  100, "rgba(255,230,0,0.065)"),
        (100, 150, "rgba(255,126,0,0.065)"),
        (150, 200, "rgba(255,0,0,0.06)"),
        (200, 300, "rgba(143,63,151,0.06)"),
    ]
    return [
        dict(type="rect", xref="paper", yref="y",
             x0=0, x1=1, y0=lo, y1=hi,
             fillcolor=fill, line_width=0, layer="below")
        for lo, hi, fill in zones
    ]


def _base_layout(**kwargs) -> dict:
    """
    Build a base Plotly layout dict.
    kwargs always win — callers can override margin, legend, or any other key
    without triggering a 'multiple values for keyword argument' TypeError.
    """
    defaults: dict = {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "margin":        dict(l=0, r=0, t=10, b=0),
        "hovermode":     "x unified",
        "legend":        dict(orientation="h", yanchor="bottom", y=1.02,
                              xanchor="right", x=1, font=dict(size=11)),
        "font":          dict(family="inherit", size=11),
    }
    defaults.update(kwargs)   # kwargs win; no collision possible
    return defaults


# ── Gate ───────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='font-size:1.6rem; font-weight:700; margin-bottom:3px;'>"
    "Air Quality Forecast</h2>",
    unsafe_allow_html=True,
)
st.caption(
    f"ZIP **{zip_code}** · {history_days}-day history · "
    f"{forecast_days}-day forecast · Source: EPA AirNow"
)

if not run:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Enter a ZIP code in the sidebar and click **Fetch & Forecast**.", icon="👈")
    st.stop()


# ── Data loading ───────────────────────────────────────────────────────────────

with st.spinner("Fetching from EPA AirNow…"):
    try:
        df_hist, df_current = load_data(zip_code, api_key, history_days)
    except AirNowError as exc:
        st.error(f"API error: {exc}")
        st.stop()

if df_hist.empty:
    st.warning(
        "No data returned for this ZIP code. "
        "Try a different ZIP or increase the history window."
    )
    st.stop()


# ── Derived ────────────────────────────────────────────────────────────────────

daily        = daily_max_aqi(df_hist)
clean_daily  = daily.dropna()

if clean_daily.empty:
    st.error("Historical data parsed but all AQI values were invalid.")
    st.stop()

forecast     = generate_forecast(daily, steps=forecast_days)
trend        = compute_trend(daily)
primary      = identify_primary_pollutant(df_current, df_hist)
stats_tbl    = pollutant_stats_table(df_hist)
all_sectors  = get_all_impacts()
# impact_score counts ALL sectors (regardless of AQI threshold) so the
# "Sectors Analyzed / Benefiting / Pressured / Mixed" cards always show
# the full breakdown. The threshold concept is surfaced in the chart below.
_all_impact_counts: dict = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
for _s in all_sectors:
    _all_impact_counts[_s.expected_impact] += 1
impact_score = {**_all_impact_counts, "total_active": get_impact_score(int(clean_daily.iloc[-1]))["total_active"]}

latest_aqi   = int(clean_daily.iloc[-1])
prev_aqi     = int(clean_daily.iloc[-2]) if len(clean_daily) >= 2 else latest_aqi
aqi_delta    = latest_aqi - prev_aqi


# ── ① KPI row ──────────────────────────────────────────────────────────────────

st.markdown("<div class='s-hdr'>Overview</div>", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current AQI",      latest_aqi,
          delta=f"{aqi_delta:+d} vs yesterday", delta_color="inverse")
c2.metric("7-Day Rolling Avg", trend.rolling_7d_avg)
c3.metric("Forecast Peak",     int(forecast.point.max()),
          help=f"Highest single-day AQI in the {forecast_days}-day forecast")
c4.metric("Forecast Avg",      int(forecast.point.mean()),
          help=f"Mean AQI across {forecast_days}-day forecast")
c5.metric("Volatility (σ)",   trend.volatility,
          help="Std deviation of 14-day AQI — higher = less predictable")

st.markdown("<br>", unsafe_allow_html=True)


# ── ② Primary pollutant + Health Advisory ──────────────────────────────────────

st.markdown("<div class='s-hdr'>Primary Pollutant</div>", unsafe_allow_html=True)

hero_col, adv_col = st.columns([1, 2], gap="large")

with hero_col:
    if primary:
        tc   = _txt(primary.aqi)
        sign = "+" if primary.trend_pct >= 0 else ""
        st.markdown(
            f"""
            <div class="hero" style="background:{primary.color}; color:{tc};">
                <div class="hero-tag">⭐ Primary Pollutant · Highest AQI Now</div>
                <div class="hero-name">{primary.info.full_name}</div>
                <div class="hero-aqi">{primary.aqi}</div>
                <div class="hero-cat">{primary.category}</div>
                <div class="hero-trd">
                    {primary.trend_icon} 7-day avg {primary.recent_avg}
                    &nbsp;·&nbsp; {sign}{primary.trend_pct}% vs prior week
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Current pollutant data not available.")

with adv_col:
    adv = get_health_advisory(latest_aqi)
    st.markdown(
        f"""
        <div class="adv-card">
            <div class="adv-title">
                {adv.icon} Health Advisory · AQI {latest_aqi} — {adv.category}
            </div>
            <div class="adv-row">
                <strong>General public →</strong>&nbsp; {adv.general}
            </div>
            <div class="adv-row">
                <strong>Sensitive groups →</strong>&nbsp; {adv.sensitive}
            </div>
            <div class="adv-row">
                <strong>Outdoor activity →</strong>&nbsp; {adv.outdoor_activity}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── ③ Trend row ────────────────────────────────────────────────────────────────

st.markdown("<div class='s-hdr'>Recent Trends</div>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)

with t1:
    pill_kind = (
        "red"   if trend.direction == "worsening" else
        "green" if trend.direction == "improving" else
        "gray"
    )
    sign = "+" if trend.pct_change_7d >= 0 else ""
    st.markdown(
        f"""
        <div class="stat-blk">
            <div class="stat-lbl">7-Day Trend</div>
            <div class="stat-val">
                {_pill(f"{trend.direction_icon} {trend.direction.title()}", pill_kind)}
                <div style="font-size:0.75rem; color:#6b7280; margin-top:5px;">
                    {sign}{trend.pct_change_7d}% vs prior 7 days
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t2:
    st.markdown(
        f"""
        <div class="stat-blk">
            <div class="stat-lbl">Current Streak</div>
            <div class="stat-val">{trend.streak_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t3:
    st.markdown(
        f"""
        <div class="stat-blk">
            <div class="stat-lbl">Best Day in Window</div>
            <div class="stat-val" style="font-size:1.15rem; font-weight:600;">
                {trend.best_day}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with t4:
    st.markdown(
        f"""
        <div class="stat-blk">
            <div class="stat-lbl">Worst Day in Window</div>
            <div class="stat-val" style="font-size:1.15rem; font-weight:600;">
                {trend.worst_day}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ── ④ Tabs ─────────────────────────────────────────────────────────────────────

tab_frc, tab_hist, tab_poll, tab_mkt, tab_about = st.tabs([
    "📈 Forecast",
    "📆 Historical",
    "🔬 Pollutants",
    "📊 Market Impact",
    "ℹ️ About",
])


# ─── Forecast tab ──────────────────────────────────────────────────────────────

with tab_frc:
    fig = go.Figure()

    rolling_avg = clean_daily.rolling(7, min_periods=3).mean()

    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.values,
        mode="lines", name="Historical AQI",
        line=dict(color="#2563eb", width=2.5),
        connectgaps=False,
        hovertemplate="<b>%{x|%b %-d}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=rolling_avg.index, y=rolling_avg.values,
        mode="lines", name="7-Day Avg",
        line=dict(color="#7c3aed", width=1.5, dash="dash"),
        hovertemplate="<b>%{x|%b %-d}</b><br>7-day avg: %{y:.1f}<extra></extra>",
    ))

    # Confidence band — upper then lower reversed to fill between
    fig.add_trace(go.Scatter(
        x=list(forecast.dates) + list(forecast.dates[::-1]),
        y=list(forecast.upper) + list(forecast.lower[::-1]),
        fill="toself",
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=forecast.dates, y=forecast.point,
        mode="lines+markers",
        name=f"Forecast ({forecast.method})",
        line=dict(color="#dc2626", width=2, dash="dot"),
        marker=dict(size=7, color="#dc2626",
                    line=dict(width=2, color="white")),
        hovertemplate="<b>%{x|%b %-d}</b><br>Forecast: %{y:.0f}<extra></extra>",
    ))

    # AQI category annotations on right axis
    for lo, hi, label, _ in [
        (0,   50,  "Good",               None),
        (50,  100, "Moderate",           None),
        (100, 150, "Unhealthy for Some", None),
        (150, 200, "Unhealthy",          None),
        (200, 300, "Very Unhealthy",     None),
    ]:
        fig.add_annotation(
            x=1.01, xref="paper",
            y=(lo + hi) / 2, yref="y",
            text=label, showarrow=False,
            font=dict(size=9, color="#9ca3af"),
            xanchor="left",
        )

    fig.update_layout(
        **_base_layout(
            shapes=_aqi_zone_shapes(),
            yaxis=dict(
                title="AQI",
                range=[0, max(220, int(forecast.upper.max()) + 30)],
                gridcolor="#f3f4f6",
            ),
            margin=dict(l=0, r=100, t=10, b=0),
            height=400,
        )
    )
    fig.update_xaxes(showgrid=False, tickformat="%b %-d")

    st.plotly_chart(fig, width="stretch")

    # Forecast table
    st.markdown("<div class='s-hdr'>Day-by-Day Forecast</div>",
                unsafe_allow_html=True)
    rows = []
    for d, pt, lo, hi in zip(
        forecast.dates, forecast.point, forecast.lower, forecast.upper
    ):
        cat, _ = aqi_category(int(pt))
        rows.append({
            "Date":       _fmt_date_long(d),
            "AQI":        int(pt),
            "Low (95%)":  int(lo),
            "High (95%)": int(hi),
            "Category":   cat,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Model: **{forecast.method}** · 95% prediction intervals")


# ─── Historical tab ────────────────────────────────────────────────────────────

with tab_hist:
    bar_data = clean_daily.reset_index()
    bar_data.columns = ["date", "aqi"]

    def _aqi_color(v: float) -> str:
        for ceil, _, color in config.AQI_CATEGORIES:
            if v <= ceil:
                return color
        return "#7E0023"

    bar_colors = [_aqi_color(v) for v in bar_data["aqi"]]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=bar_data["date"], y=bar_data["aqi"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        name="Daily Max AQI",
        hovertemplate="<b>%{x|%b %-d, %Y}</b><br>Max AQI: %{y}<extra></extra>",
    ))
    rolling7 = bar_data.set_index("date")["aqi"].rolling(7, min_periods=2).mean()
    fig2.add_trace(go.Scatter(
        x=rolling7.index, y=rolling7.values,
        mode="lines", name="7-Day Avg",
        line=dict(color="#2563eb", width=2),
        hovertemplate="<b>%{x|%b %-d}</b><br>7-day avg: %{y:.1f}<extra></extra>",
    ))

    fig2.update_layout(
        **_base_layout(
            yaxis=dict(title="AQI", gridcolor="#f3f4f6"),
            height=320,
        )
    )
    fig2.update_xaxes(showgrid=False, tickformat="%b %-d")
    st.plotly_chart(fig2, width="stretch")

    col_tbl, col_pie = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown("<div class='s-hdr'>Pollutant Statistics</div>",
                    unsafe_allow_html=True)
        if not stats_tbl.empty:
            st.dataframe(stats_tbl, hide_index=True)

    with col_pie:
        st.markdown("<div class='s-hdr'>Category Distribution</div>",
                    unsafe_allow_html=True)
        cat_counts = df_hist["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_color_map = {label: color for _, label, color in config.AQI_CATEGORIES}
        cat_counts["color"] = cat_counts["Category"].map(cat_color_map).fillna("#d1d5db")

        fig3 = go.Figure(go.Pie(
            labels=cat_counts["Category"],
            values=cat_counts["Count"],
            marker_colors=cat_counts["color"].tolist(),
            hole=0.58,
            textinfo="percent",
            textfont=dict(size=11),
        ))
        fig3.update_layout(
            **_base_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(font=dict(size=11), orientation="v"),
                height=240,
            )
        )
        st.plotly_chart(fig3, width="stretch")


# ─── Pollutants tab ────────────────────────────────────────────────────────────

with tab_poll:
    # Current conditions cards
    st.markdown("<div class='s-hdr'>Current Conditions</div>",
                unsafe_allow_html=True)

    if df_current.empty:
        st.info("No current conditions data available for this location.")
    else:
        n = len(df_current)
        cols = st.columns(min(n, 4))
        for i, (_, row) in enumerate(df_current.iterrows()):
            tc = _txt(int(row["aqi"]))
            is_primary = (
                primary is not None and row["pollutant"] == primary.name
            )
            border = "2.5px solid rgba(0,0,0,0.22)" if is_primary else "none"
            badge  = "⭐ PRIMARY · " if is_primary else ""
            with cols[i % 4]:
                st.markdown(
                    f"""
                    <div class="pc-wrap"
                         style="background:{row['color']}; color:{tc}; border:{border};">
                        <div class="pc-name">{badge}{row['pollutant']}</div>
                        <div class="pc-aqi">{row['aqi']}</div>
                        <div class="pc-cat">{row['category']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Primary pollutant deep-dive
    if primary:
        st.markdown(
            f"<div class='s-hdr'>About {primary.name} — Today's Primary Pollutant</div>",
            unsafe_allow_html=True,
        )
        info = primary.info
        di1, di2, di3 = st.columns(3, gap="medium")
        for col, lbl, val in [
            (di1, "Common Sources",  info.sources),
            (di2, "Health Effects",  info.health_effects),
            (di3, "Most at Risk",    info.sensitive_groups),
        ]:
            with col:
                st.markdown(
                    f"""
                    <div class="stat-blk">
                        <div class="stat-lbl">{lbl}</div>
                        <div class="stat-val">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # Historical per-pollutant lines
    st.markdown("<div class='s-hdr'>Historical AQI by Pollutant</div>",
                unsafe_allow_html=True)

    if "pollutant" in df_hist.columns and df_hist["pollutant"].nunique() >= 1:
        palette = ["#2563eb", "#dc2626", "#d97706", "#059669",
                   "#7c3aed", "#0891b2"]
        fig4 = go.Figure()
        for idx, (pollutant, grp) in enumerate(df_hist.groupby("pollutant")):
            grp_s   = grp.sort_values("date")
            is_prim = primary is not None and pollutant == primary.name
            fig4.add_trace(go.Scatter(
                x=grp_s["date"], y=grp_s["aqi"],
                mode="lines+markers",
                name=pollutant,
                line=dict(
                    color=palette[idx % len(palette)],
                    width=2.5 if is_prim else 1.5,
                ),
                marker=dict(size=3),
                hovertemplate=(
                    f"<b>{pollutant}</b><br>"
                    f"%{{x|%b %-d}}: AQI %{{y}}<extra></extra>"
                ),
            ))
        fig4.update_layout(
            **_base_layout(
                yaxis=dict(title="AQI", gridcolor="#f3f4f6"),
                height=320,
            )
        )
        fig4.update_xaxes(showgrid=False, tickformat="%b %-d")
        st.plotly_chart(fig4, width="stretch")

    # Box plot
    st.markdown("<div class='s-hdr'>Distribution by Pollutant</div>",
                unsafe_allow_html=True)
    if not df_hist.empty and "pollutant" in df_hist.columns:
        palette = ["#2563eb", "#dc2626", "#d97706", "#059669"]
        fig5 = go.Figure()
        for idx, (pollutant, grp) in enumerate(df_hist.groupby("pollutant")):
            fig5.add_trace(go.Box(
                y=grp["aqi"], name=pollutant,
                marker_color=palette[idx % len(palette)],
                boxpoints="outliers", jitter=0.3,
                hovertemplate=(
                    f"<b>{pollutant}</b><br>AQI: %{{y}}<extra></extra>"
                ),
            ))
        fig5.update_layout(
            **_base_layout(
                yaxis=dict(title="AQI", gridcolor="#f3f4f6"),
                showlegend=False,
                height=280,
            )
        )
        st.plotly_chart(fig5, width="stretch")


# ─── Market Impact tab ─────────────────────────────────────────────────────────

with tab_mkt:
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <strong>Educational analysis only — not investment advice.</strong> "
        "Sector relationships are derived from peer-reviewed research. "
        "Past correlations do not predict future returns. "
        "Consult a qualified financial advisor before making investment decisions."
        "</div>",
        unsafe_allow_html=True,
    )

    # Impact summary KPIs
    st.markdown("<div class='s-hdr'>Sector Impact Summary — AQI "
                f"{latest_aqi}</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sectors Analyzed",  len(all_sectors))
    m2.metric("Likely Benefiting", impact_score["positive"],
              help="Sectors where high AQI historically correlates with positive revenue")
    m3.metric("Likely Pressured",  impact_score["negative"],
              help="Sectors where high AQI historically correlates with negative revenue")
    m4.metric("Active at AQI " + str(latest_aqi), impact_score["total_active"],
              help=f"Sectors whose AQI threshold ({latest_aqi}) has been crossed")

    st.markdown("<br>", unsafe_allow_html=True)

    # Sensitivity bar chart
    st.markdown("<div class='s-hdr'>Sector Sensitivity Score</div>",
                unsafe_allow_html=True)

    sector_names   = [s.sector for s in all_sectors]
    # Scores: rough directional + magnitude (positive up, negative down)
    sensitivity    = [8, 7, 4, -8, -7, -6, -3, -4, -5]
    bar_colors_mkt = ["#16a34a" if v > 0 else "#dc2626" for v in sensitivity]

    fig6 = go.Figure(go.Bar(
        x=sensitivity,
        y=sector_names,
        orientation="h",
        marker=dict(color=bar_colors_mkt, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Score: %{x:+d}<extra></extra>",
    ))
    fig6.update_layout(
        **_base_layout(
            xaxis=dict(
                title="Sensitivity Score (positive = benefits, negative = hurt)",
                gridcolor="#f3f4f6",
                zeroline=True,
                zerolinecolor="#d1d5db",
                zerolinewidth=1,
            ),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(size=11),
            ),
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
    )
    st.plotly_chart(fig6, width="stretch")

    # AQI threshold chart
    st.markdown("<div class='s-hdr'>Cumulative Sectors Affected by AQI Level</div>",
                unsafe_allow_html=True)

    aqi_levels   = list(range(0, 210, 25))
    sector_counts = [get_impact_score(a)["total_active"] for a in aqi_levels]

    fig7 = go.Figure(go.Scatter(
        x=aqi_levels, y=sector_counts,
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2.5),
        marker=dict(size=8, color="#7c3aed",
                    line=dict(width=2, color="white")),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        hovertemplate="AQI %{x} → %{y} sectors affected<extra></extra>",
    ))
    # Vertical line at current AQI
    fig7.add_vline(
        x=latest_aqi,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Current: {latest_aqi}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#dc2626"),
    )
    fig7.update_layout(
        **_base_layout(
            xaxis=dict(title="AQI Level", gridcolor="#f3f4f6",
                       range=[0, 200]),
            yaxis=dict(title="Sectors affected",
                       gridcolor="#f3f4f6",
                       range=[0, len(all_sectors) + 1]),
            height=260,
        )
    )
    st.plotly_chart(fig7, width="stretch")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter buttons
    st.markdown("<div class='s-hdr'>Sector Details</div>", unsafe_allow_html=True)
    filter_opts = ["All", "Positive", "Negative", "Mixed"]
    chosen = st.radio(
        "Filter by impact:",
        filter_opts,
        horizontal=True,
        label_visibility="collapsed",
    )
    chosen_key = chosen.lower()

    for sector in all_sectors:
        if chosen_key != "all" and sector.expected_impact != chosen_key:
            continue

        conf_color   = _conf_color(sector.confidence)
        impact_badge = _impact_pill(sector.expected_impact)
        ticker_html  = ""
        for t in sector.tickers:
            t_dir = {"positive": "↑", "negative": "↓",
                     "mixed": "~", "neutral": "→"}.get(t.expected_impact, "→")
            t_col = {"positive": "#15803d", "negative": "#991b1b",
                     "mixed": "#92400e", "neutral": "#6b7280"}.get(
                t.expected_impact, "#6b7280"
            )
            ticker_html += (
                f'<span class="t-pill">'
                f'<strong>{t.ticker}</strong>'
                f'<span class="t-pill-name">{t.name}</span>'
                f'<span style="color:{t_col}; font-weight:600;">{t_dir}</span>'
                f'</span>'
            )

        st.markdown(
            f"""
            <div class="sector-card">
                <div class="sector-head">
                    <div class="sector-icon">{sector.icon}</div>
                    <div>
                        <div class="sector-name">
                            {sector.sector}
                            &nbsp;{impact_badge}
                            &nbsp;<span style="font-size:0.7rem; color:#9ca3af;
                                               font-weight:400;">
                                AQI threshold {sector.aqi_threshold}+
                            </span>
                        </div>
                        <div class="sector-meta">{sector.short_rationale}</div>
                    </div>
                </div>
                <div class="sector-detail">{sector.detail}</div>
                <div class="ticker-row">{ticker_html}</div>
                <div class="conf-line">
                    Research confidence:&nbsp;
                    <span style="color:{conf_color}; font-weight:600;">
                        {sector.confidence.title()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─── About tab ─────────────────────────────────────────────────────────────────

with tab_about:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### Data Source")
        st.markdown(
            "All air quality data comes from the **U.S. EPA AirNow** program — "
            "a partnership between EPA, NOAA, NPS, and state/local agencies.\n\n"
            "- API docs: [docs.airnowapi.org](https://docs.airnowapi.org/)\n"
            "- AQI basics: [airnow.gov/aqi/aqi-basics](https://www.airnow.gov/aqi/aqi-basics/)\n"
            "- Free key: [airnowapi.org/account/request](https://docs.airnowapi.org/account/request/)"
        )
        st.markdown("#### Forecasting Model")
        st.markdown(
            "Model selected automatically based on data availability:\n\n"
            "| Available days | Model |\n"
            "|---|---|\n"
            "| ≥ 14 | ARIMA(2,1,2) via `statsmodels` |\n"
            "| 7–13 | Simple Exponential Smoothing |\n"
            "| < 7  | Trailing 7-day mean (baseline) |\n\n"
            "Predictions are clipped to [0, 500]. Bounds are 95% prediction intervals."
        )
        st.markdown("#### Market Impact Research")
        st.markdown(
            "Sector analysis is based on:\n"
            "- Graff Zivin & Neidell (2012, *AER*): Pollution → worker productivity\n"
            "- Knittel et al. (2020): Wildfire smoke → financial markets\n"
            "- Hong et al. (2019, *Journal of Finance*): Pollution → analyst accuracy\n"
            "- USDA / NBER working papers on crop yield and hospital revenue\n"
        )

    with col_b:
        st.markdown("#### AQI Categories")
        for ceiling, label, color in config.AQI_CATEGORIES:
            tc = "#fff" if color in ("#FF0000", "#8F3F97", "#7E0023") else "#111"
            st.markdown(
                f'<div style="background:{color}; color:{tc}; padding:6px 14px; '
                f'border-radius:6px; margin-bottom:5px; font-size:0.85rem; '
                f'font-weight:500;">0–{ceiling} &nbsp;·&nbsp; {label}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("#### Disclaimer")
        st.markdown(
            "<span style='font-size:0.8rem; color:#6b7280;'>"
            "For informational purposes only. Always verify at "
            "<a href='https://www.airnow.gov'>airnow.gov</a> "
            "and follow local health authority guidance during air quality emergencies. "
            "Market impact analysis is educational — not investment advice."
            "</span>",
            unsafe_allow_html=True,
        )
