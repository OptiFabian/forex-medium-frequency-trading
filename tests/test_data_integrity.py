"""On-disk data integrity checks for the stored EURUSD + USDCHF bars.

Each test encodes a single first-principles assumption about what the
data must look like. Failures here mean the data on disk is corrupted,
incomplete, or violates an invariant of the fetch/storage pipeline.

The tests are parametrized over (pair, side) and skip cleanly when a
side has no data on disk (so they don't blow up on a fresh checkout).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.quality import analyze
from fxalgo.data.storage import BAR_COLUMNS, BarsStorage
from fxalgo.settings import PROJECT_ROOT

# Market data is not distributed with this repository (see README). Skip the
# whole module on a checkout that has not fetched it yet.
if not (PROJECT_ROOT / "data" / "raw").is_dir():
    pytest.skip("no market data on disk (data/raw missing); see README", allow_module_level=True)

# Pairs whose data we expect to be complete enough to validate.
PAIRS_TO_CHECK = ("EURUSD", "USDCHF")
SIDES = ("bid", "ask")
PAIR_SIDE_PARAMS = [(p, s) for p in PAIRS_TO_CHECK for s in SIDES]
PAIR_SIDE_IDS = [f"{p}-{s}" for p, s in PAIR_SIDE_PARAMS]

# ------ Derived expectations from first principles ------
#
# Continuous 24/7 minutes per week:    7 days * 24 hours * 60 min   = 10080
# Subtract FX weekend close:           48h * 60 min                 = -2880
# Subtract daily reset (Mon-Thu):      4 * ~16 min                  = -64
# Net expected bars per "normal" week: ~7136
MIN_PER_DAY = 24 * 60
MIN_PER_WEEK = 7 * MIN_PER_DAY               # 10,080
WEEKEND_CLOSE_MIN = 48 * 60                   # 2,880
DAILY_RESETS_MIN = 4 * 16                     # 64 (Mon-Thu, ~16 min each)
BARS_PER_WEEK = MIN_PER_WEEK - WEEKEND_CLOSE_MIN - DAILY_RESETS_MIN  # 7,136

# Tolerances: the data source can have holiday closures, occasional missed
# minutes, and in 2005-2010 the daily reset was wider (~2:26h).
# These tolerances should accommodate normal variance but flag real
# issues like a chunk-sized hole.
ROW_COUNT_TOLERANCE_FRAC = 0.05    # +/- 5% of weeks * BARS_PER_WEEK
WEEKEND_COUNT_TOLERANCE_FRAC = 0.10  # +/- 10% of total weekends in range
ACTIVE_GAP_BUDGET_PER_YEAR = 25    # max plausible holiday + reset-artifact gaps
MAX_TOLERABLE_ACTIVE_GAP = pd.Timedelta(days=5)  # holiday weekends top out ~4d


# ------ Fixtures: load each (pair, side) once and reuse ------


@pytest.fixture(scope="module")
def storage_root() -> Path:
    return PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def loaded() -> dict[tuple[str, str], pd.DataFrame]:
    """Read every (pair, side) we plan to check, once."""
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    root = PROJECT_ROOT / "data" / "raw"
    for pair in PAIRS_TO_CHECK:
        for side in SIDES:
            storage = BarsStorage(root, pair, side)
            cache[(pair, side)] = storage.read_all()
    return cache


def _df_or_skip(loaded: dict, pair: str, side: str) -> pd.DataFrame:
    df = loaded[(pair, side)]
    if df.empty:
        pytest.skip(f"{pair} {side} has no data on disk")
    return df


# ============================================================
# 1. Schema + presence
# ============================================================


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_data_present(loaded, pair, side) -> None:
    """Each checked (pair, side) must have non-zero rows on disk."""
    df = loaded[(pair, side)]
    assert len(df) > 0, f"{pair} {side} is empty"


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_columns_match_schema(loaded, pair, side) -> None:
    """DataFrame columns and order must match the canonical schema."""
    df = _df_or_skip(loaded, pair, side)
    assert tuple(df.columns) == BAR_COLUMNS, (
        f"{pair} {side}: columns {tuple(df.columns)} != {BAR_COLUMNS}"
    )


# ============================================================
# 2. Timestamp invariants
# ============================================================


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_timestamps_unique(loaded, pair, side) -> None:
    """Storage dedups on timestamp - there must be zero duplicates."""
    df = _df_or_skip(loaded, pair, side)
    dupes = df["timestamp"].duplicated().sum()
    assert dupes == 0, f"{pair} {side}: {dupes} duplicate timestamps"


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_timestamps_sorted(loaded, pair, side) -> None:
    """Storage sorts on append - the read-back must be monotonically increasing."""
    df = _df_or_skip(loaded, pair, side)
    assert df["timestamp"].is_monotonic_increasing, (
        f"{pair} {side}: timestamps are not sorted ascending"
    )


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_timestamps_on_minute_grid(loaded, pair, side) -> None:
    """Every timestamp must land exactly on a minute (seconds=0, microseconds=0)."""
    df = _df_or_skip(loaded, pair, side)
    bad_seconds = (df["timestamp"].dt.second != 0).sum()
    bad_micros = (df["timestamp"].dt.microsecond != 0).sum()
    assert bad_seconds == 0, f"{pair} {side}: {bad_seconds} bars off the minute grid (seconds!=0)"
    assert bad_micros == 0, f"{pair} {side}: {bad_micros} bars off the minute grid (microseconds!=0)"


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_timestamps_are_utc(loaded, pair, side) -> None:
    """All timestamps must be tz-aware UTC (any precision)."""
    df = _df_or_skip(loaded, pair, side)
    dtype = df["timestamp"].dtype
    assert isinstance(dtype, pd.DatetimeTZDtype), (
        f"{pair} {side}: timestamp dtype is {dtype}, not a DatetimeTZDtype"
    )
    assert str(dtype.tz) == "UTC", f"{pair} {side}: timestamp tz is {dtype.tz}, not UTC"


# ============================================================
# 3. OHLCV value invariants
# ============================================================


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_ohlcv_finite_and_positive(loaded, pair, side) -> None:
    """OHLC must be finite and positive. Volume must be finite and either
    >= 0 or exactly -1 (the "volume unavailable" sentinel: spot FX has no
    centralized volume).
    """
    df = _df_or_skip(loaded, pair, side)
    for col in ("open", "high", "low", "close"):
        non_finite = (~np.isfinite(df[col])).sum()
        non_positive = (df[col] <= 0).sum()
        assert non_finite == 0, f"{pair} {side}: {non_finite} non-finite values in {col}"
        assert non_positive == 0, f"{pair} {side}: {non_positive} non-positive values in {col}"

    vol = df["volume"]
    non_finite_vol = (~np.isfinite(vol)).sum()
    assert non_finite_vol == 0, f"{pair} {side}: {non_finite_vol} non-finite volume rows"
    bad_volume = ((vol < 0) & (vol != -1.0)).sum()
    assert bad_volume == 0, (
        f"{pair} {side}: {bad_volume} volume values are negative but not the -1 sentinel"
    )


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_ohlc_relations(loaded, pair, side) -> None:
    """`high` must be the max and `low` must be the min of open/high/low/close per bar."""
    df = _df_or_skip(loaded, pair, side)
    high_violations = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    ).sum()
    low_violations = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    ).sum()
    assert high_violations == 0, f"{pair} {side}: {high_violations} bars where high is not the max"
    assert low_violations == 0, f"{pair} {side}: {low_violations} bars where low is not the min"


# ============================================================
# 4. Aggregate row count vs first-principles expectation
# ============================================================


def _weeks_covered(df: pd.DataFrame) -> float:
    span = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    return span.total_seconds() / (7 * 24 * 3600)


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_row_count_matches_expected(loaded, pair, side) -> None:
    """Actual rows must be within +/- 5% of (weeks_covered * BARS_PER_WEEK).

    Underage indicates holidays + the wider 2005-2010 daily reset; deficit
    above the tolerance means real data is missing.
    """
    df = _df_or_skip(loaded, pair, side)
    weeks = _weeks_covered(df)
    expected = weeks * BARS_PER_WEEK
    actual = len(df)
    ratio = actual / expected
    lower = 1.0 - ROW_COUNT_TOLERANCE_FRAC
    upper = 1.0 + ROW_COUNT_TOLERANCE_FRAC
    msg = (
        f"{pair} {side}: row count out of tolerance.\n"
        f"  weeks covered:  {weeks:.2f}\n"
        f"  expected rows:  {expected:,.0f}\n"
        f"  actual rows:    {actual:,}\n"
        f"  actual/expected: {ratio:.4f} (tolerance {lower:.2f}..{upper:.2f})"
    )
    assert lower <= ratio <= upper, msg


# ============================================================
# 5. Gap counts vs calendar
# ============================================================


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_weekend_count_matches_calendar(loaded, pair, side) -> None:
    """Number of weekend gaps must be within +/- 10% of the calendar weekends in range."""
    df = _df_or_skip(loaded, pair, side)
    summary = analyze(df)
    weeks = _weeks_covered(df)
    expected_weekends = math.floor(weeks)
    actual = summary.weekend_gaps
    lower = expected_weekends * (1 - WEEKEND_COUNT_TOLERANCE_FRAC)
    upper = expected_weekends * (1 + WEEKEND_COUNT_TOLERANCE_FRAC)
    assert lower <= actual <= upper, (
        f"{pair} {side}: {actual} weekend gaps, expected ~{expected_weekends} "
        f"({lower:.0f}..{upper:.0f})"
    )


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_daily_reset_count_proportional_to_weekends(loaded, pair, side) -> None:
    """Daily-reset count should be roughly 4x the weekend count (Mon-Thu per week)."""
    df = _df_or_skip(loaded, pair, side)
    summary = analyze(df)
    if summary.weekend_gaps == 0:
        pytest.skip("not enough range to evaluate ratio")
    ratio = summary.daily_reset_gaps / summary.weekend_gaps
    # 3 in case some Mon/Thu resets fall into the wider 2005-2010 window.
    # 5 in case some weekends straddle holidays (slightly extra resets nearby).
    assert 3.0 <= ratio <= 5.0, (
        f"{pair} {side}: daily_reset/weekend = {ratio:.2f}; "
        f"expected ~4x (resets={summary.daily_reset_gaps}, weekends={summary.weekend_gaps})"
    )


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_active_gaps_bounded(loaded, pair, side) -> None:
    """Active-gap count should stay under a budget proportional to years covered."""
    df = _df_or_skip(loaded, pair, side)
    summary = analyze(df)
    years = _weeks_covered(df) / 52.18
    budget = math.ceil(years * ACTIVE_GAP_BUDGET_PER_YEAR)
    assert summary.active_gaps <= budget, (
        f"{pair} {side}: {summary.active_gaps} active gaps over {years:.1f} years "
        f"(budget {budget}). Largest: {summary.largest_active_gap}. "
        f"Samples: {summary.sample_active_gaps[:3]}"
    )


@pytest.mark.parametrize(("pair", "side"), PAIR_SIDE_PARAMS, ids=PAIR_SIDE_IDS)
def test_no_extreme_active_gap(loaded, pair, side) -> None:
    """No single active gap may exceed MAX_TOLERABLE_ACTIVE_GAP (5 days)."""
    df = _df_or_skip(loaded, pair, side)
    summary = analyze(df)
    largest = summary.largest_active_gap
    if largest is None:
        return
    assert largest <= MAX_TOLERABLE_ACTIVE_GAP, (
        f"{pair} {side}: largest active gap is {largest}, exceeds {MAX_TOLERABLE_ACTIVE_GAP}. "
        f"Samples: {summary.sample_active_gaps[:3]}"
    )


# ============================================================
# 6. Inter-side sanity (bid vs ask)
# ============================================================


@pytest.mark.parametrize("pair", PAIRS_TO_CHECK)
def test_bid_ask_row_counts_close(loaded, pair) -> None:
    """For a given pair, BID and ASK row counts should differ by < 0.1%."""
    bid = loaded[(pair, "bid")]
    ask = loaded[(pair, "ask")]
    if bid.empty or ask.empty:
        pytest.skip(f"{pair}: one side empty")
    larger = max(len(bid), len(ask))
    diff = abs(len(bid) - len(ask))
    frac = diff / larger
    assert frac < 0.001, (
        f"{pair}: bid={len(bid):,} ask={len(ask):,} differ by {frac:.4%}"
    )


@pytest.mark.parametrize("pair", PAIRS_TO_CHECK)
def test_bid_ask_ranges_close(loaded, pair) -> None:
    """For a given pair, BID and ASK start/end timestamps should be within 1 day."""
    bid = loaded[(pair, "bid")]
    ask = loaded[(pair, "ask")]
    if bid.empty or ask.empty:
        pytest.skip(f"{pair}: one side empty")
    start_diff = abs(bid["timestamp"].iloc[0] - ask["timestamp"].iloc[0])
    end_diff = abs(bid["timestamp"].iloc[-1] - ask["timestamp"].iloc[-1])
    one_day = pd.Timedelta(days=1)
    assert start_diff < one_day, f"{pair}: bid/ask start timestamps differ by {start_diff}"
    assert end_diff < one_day, f"{pair}: bid/ask end timestamps differ by {end_diff}"
