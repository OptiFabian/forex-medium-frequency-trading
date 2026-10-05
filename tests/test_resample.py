"""Tests for gap-safe 1-min -> coarser-timeframe resampling (fxalgo.data.resample)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.quality import NY_TZ
from fxalgo.data.resample import (
    bars_per_window,
    resample_merged,
    resample_sessions,
    session_close_utc,
    session_keys,
    session_label_utc,
    side_bars,
)


def _merged(minute_offsets, base=1.10, spread=0.0002):
    """Build a 1-min merged bid/ask frame at the given minute offsets from 2024-01-01.

    Each minute m gets mid = base + m*1e-5; bid = mid - spread/2, ask = mid + spread/2;
    OHLC all equal to that side's price (flat 1-min bars) so aggregation is checkable.
    """
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 00:00", tz="UTC") + pd.Timedelta(minutes=int(m)) for m in minute_offsets]
    )
    mid = base + np.arange(len(minute_offsets)) * 1e-5
    half = spread / 2.0
    data = {}
    for col in ("open", "high", "low", "close"):
        data[f"{col}_bid"] = mid - half
        data[f"{col}_ask"] = mid + half
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


def test_bars_per_window():
    assert bars_per_window("15min") == 15
    assert bars_per_window("5min") == 5
    assert bars_per_window("1h") == 60


def test_two_clean_windows_ohlc():
    """30 consecutive minutes -> two 15-min bars at :00 and :15 with correct OHLC."""
    df = _merged(range(30))
    out = resample_merged(df, "15min")
    assert len(out) == 2
    assert list(out.index) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:15", tz="UTC"),
    ]
    # Window :00 aggregates minutes 0..14 ONLY (not minute 15).
    assert out["open_bid"].iloc[0] == df["open_bid"].iloc[0]      # first
    assert out["close_bid"].iloc[0] == df["close_bid"].iloc[14]   # last in window
    assert out["high_ask"].iloc[0] == df["high_ask"].iloc[:15].max()
    assert out["low_bid"].iloc[0] == df["low_bid"].iloc[:15].min()
    assert (out["n_minutes"] == 15).all()


def test_gap_does_not_span_windows():
    """Minutes [0..14] and [30..44] present, window :15 entirely empty.
    Must yield bars at :00 and :30 only -- never a bar spanning the gap."""
    df = _merged(list(range(15)) + list(range(30, 45)))
    out = resample_merged(df, "15min")
    assert list(out.index) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:30", tz="UTC"),
    ]
    assert pd.Timestamp("2024-01-01 00:15", tz="UTC") not in out.index
    assert (out["n_minutes"] == 15).all()


def test_partial_bar_is_kept_and_counted():
    """A window with only 5 minutes is kept, flagged by n_minutes."""
    df = _merged(range(5))  # only 00:00..00:04
    out = resample_merged(df, "15min")
    assert len(out) == 1
    assert out["n_minutes"].iloc[0] == 5
    # close is the last present minute.
    assert out["close_bid"].iloc[0] == df["close_bid"].iloc[4]


def test_all_bars_on_clean_boundaries():
    df = _merged(range(200))
    out = resample_merged(df, "15min")
    mins = out.index.minute
    secs = out.index.second
    assert set(np.unique(mins)).issubset({0, 15, 30, 45})
    assert (secs == 0).all()


def test_mid_and_spread_recomputed_at_bar_close():
    df = _merged(range(15), spread=0.0002)
    out = resample_merged(df, "15min")
    row = out.iloc[0]
    assert row["mid_close"] == pytest.approx((row["close_bid"] + row["close_ask"]) / 2.0)
    assert row["spread_close"] == pytest.approx(row["close_ask"] - row["close_bid"])
    assert row["spread_close"] == pytest.approx(0.0002)


def test_side_bars_schema():
    df = _merged(range(30))
    out = resample_merged(df, "15min")
    bid = side_bars(out, "bid")
    assert list(bid.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert (bid["volume"] == -1.0).all()
    assert len(bid) == 2
    ask = side_bars(out, "ask")
    # ask close > bid close (positive spread).
    assert (ask["close"].to_numpy() > bid["close"].to_numpy()).all()


def test_requires_side_columns():
    df = pd.DataFrame({"mid_close": [1.0, 2.0]},
                      index=pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC"))
    with pytest.raises(KeyError, match="side-OHLC"):
        resample_merged(df, "15min")


# --------------------------------------------------------------------------
# Session bars: daily (17:00 NY close) and weekly (Sun 17:00 -> Fri 17:00 NY)
# --------------------------------------------------------------------------


def _merged_ny(ny_times, mids=None, spread=0.0002):
    """Build a 1-min merged frame from NEW YORK local wall times.

    `ny_times` are strings interpreted in America/New_York (so the tests read
    in the same units as the 17:00-NY session convention); the frame's index
    is the corresponding UTC instants, as the loader produces.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz=NY_TZ) for t in ny_times]).tz_convert("UTC")
    mid = (
        np.asarray(mids, dtype=float)
        if mids is not None
        else 1.10 + np.arange(len(ny_times)) * 1e-5
    )
    half = spread / 2.0
    data = {}
    for col in ("open", "high", "low", "close"):
        data[f"{col}_bid"] = mid - half
        data[f"{col}_ask"] = mid + half
    df = pd.DataFrame(data, index=idx)
    df.index.name = "timestamp"
    return df


def test_daily_boundary_is_1700_new_york():
    """16:59 NY closes one session; 17:00 NY opens the next."""
    df = _merged_ny(["2024-07-01 16:59", "2024-07-01 17:00"])
    out = resample_sessions(df, "daily")
    assert len(out) == 2
    # 16:59 belongs to the session that opened Sunday 17:00 NY (= 21:00 UTC).
    assert out.index[0] == pd.Timestamp("2024-06-30 21:00", tz="UTC")
    # 17:00 opens Tuesday's session, labeled at Monday 17:00 NY.
    assert out.index[1] == pd.Timestamp("2024-07-01 21:00", tz="UTC")


def test_daily_label_is_dst_correct():
    """17:00 NY is 21:00 UTC in summer (EDT) and 22:00 UTC in winter (EST)."""
    summer = resample_sessions(_merged_ny(["2024-07-01 17:00", "2024-07-01 18:00"]), "daily")
    winter = resample_sessions(_merged_ny(["2024-01-08 17:00", "2024-01-08 18:00"]), "daily")
    assert summer.index[0] == pd.Timestamp("2024-07-01 21:00", tz="UTC")
    assert winter.index[0] == pd.Timestamp("2024-01-08 22:00", tz="UTC")
    # Both are 17:00 in NY local terms -- that is the invariant.
    for out in (summer, winter):
        assert out.index.tz_convert(NY_TZ).hour[0] == 17


def test_daily_ohlc_and_spread_aggregation():
    """One session's OHLC = first / max / min / last of its minutes."""
    mids = [1.10, 1.15, 1.05, 1.12]
    df = _merged_ny(
        ["2024-07-01 17:00", "2024-07-01 20:00", "2024-07-02 03:00", "2024-07-02 16:59"],
        mids=mids,
    )
    out = resample_sessions(df, "daily")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_minutes"] == 4
    assert row["mid_open"] == pytest.approx(1.10)
    assert row["mid_high"] == pytest.approx(1.15)
    assert row["mid_low"] == pytest.approx(1.05)
    assert row["mid_close"] == pytest.approx(1.12)
    assert row["open_ask"] > row["open_bid"]
    assert row["spread_close"] == pytest.approx(0.0002)


def test_weekend_produces_no_bar_and_never_merges():
    """Friday-before-close and Sunday-after-open are separate daily bars; no
    Saturday bar is fabricated."""
    df = _merged_ny(["2024-07-05 16:59", "2024-07-07 17:00", "2024-07-08 09:00"])
    out = resample_sessions(df, "daily")
    assert len(out) == 2
    labels_ny = out.index.tz_convert(NY_TZ)
    # Friday's minute closes the session that opened Thursday 17:00.
    assert labels_ny[0].strftime("%a %H:%M") == "Thu 17:00"
    # Sunday 17:00 opens Monday's session; Monday 09:00 joins it.
    assert labels_ny[1].strftime("%a %H:%M") == "Sun 17:00"
    assert out["n_minutes"].tolist() == [1, 2]


def test_holiday_session_is_absent_not_empty():
    """A day with no minutes at all yields no bar (no fabrication)."""
    df = _merged_ny(["2024-07-01 18:00", "2024-07-03 18:00"])  # Jul 2 session missing
    out = resample_sessions(df, "daily")
    assert len(out) == 2
    gap_days = (out.index[1] - out.index[0]).days
    assert gap_days == 2  # the skipped session leaves a hole, not a bar


def test_weekly_spans_sunday_open_to_friday_close():
    """Sunday 17:00 NY through Friday 16:59 NY is ONE weekly bar; the next
    Sunday opens a new one."""
    df = _merged_ny(
        [
            "2024-06-30 17:00",  # Sunday open  -> week 1
            "2024-07-03 12:00",  # midweek      -> week 1
            "2024-07-05 16:59",  # Friday close -> week 1
            "2024-07-07 17:00",  # Sunday open  -> week 2
        ],
        mids=[1.10, 1.20, 1.05, 1.30],
    )
    out = resample_sessions(df, "weekly")
    assert len(out) == 2
    assert out["n_minutes"].tolist() == [3, 1]
    first = out.iloc[0]
    assert first["mid_open"] == pytest.approx(1.10)
    assert first["mid_high"] == pytest.approx(1.20)
    assert first["mid_low"] == pytest.approx(1.05)
    assert first["mid_close"] == pytest.approx(1.05)  # last minute of the week
    # Both bars are labeled at a Sunday 17:00 NY open.
    labels_ny = out.index.tz_convert(NY_TZ)
    assert [t.strftime("%a %H:%M") for t in labels_ny] == ["Sun 17:00", "Sun 17:00"]
    assert (out.index[1] - out.index[0]) == pd.Timedelta(days=7)


def test_weekly_bar_is_the_union_of_its_daily_bars():
    """Daily and weekly aggregation of the same minutes must agree on the
    week's open/high/low/close."""
    times = [
        "2024-06-30 17:00", "2024-07-01 10:00", "2024-07-02 10:00",
        "2024-07-03 10:00", "2024-07-04 10:00", "2024-07-05 16:00",
    ]
    mids = [1.10, 1.13, 1.09, 1.17, 1.04, 1.11]
    df = _merged_ny(times, mids=mids)
    daily = resample_sessions(df, "daily")
    weekly = resample_sessions(df, "weekly")
    assert len(weekly) == 1
    assert weekly["n_minutes"].iloc[0] == daily["n_minutes"].sum() == len(times)
    assert weekly["mid_open"].iloc[0] == pytest.approx(daily["mid_open"].iloc[0])
    assert weekly["mid_close"].iloc[0] == pytest.approx(daily["mid_close"].iloc[-1])
    assert weekly["mid_high"].iloc[0] == pytest.approx(daily["mid_high"].max())
    assert weekly["mid_low"].iloc[0] == pytest.approx(daily["mid_low"].min())


def test_partial_session_is_kept_and_counted():
    """A session with only a few minutes (window edge / holiday half-day) is
    kept, flagged by n_minutes."""
    df = _merged_ny(["2024-07-02 09:00", "2024-07-02 09:01"])
    out = resample_sessions(df, "daily")
    assert len(out) == 1
    assert out["n_minutes"].iloc[0] == 2


def test_session_keys_and_labels_round_trip():
    idx = pd.DatetimeIndex(
        [pd.Timestamp(t, tz=NY_TZ) for t in ["2024-07-01 17:00", "2024-07-02 16:59"]]
    ).tz_convert("UTC")
    keys = session_keys(idx, "daily")
    assert keys.tolist() == [pd.Timestamp("2024-07-02"), pd.Timestamp("2024-07-02")]
    labels = session_label_utc(pd.DatetimeIndex(keys.unique()))
    assert labels.tolist() == [pd.Timestamp("2024-07-01 21:00", tz="UTC")]


def test_session_resampler_rejects_bad_input():
    df = _merged_ny(["2024-07-01 18:00"])
    with pytest.raises(ValueError, match="period must be one of"):
        resample_sessions(df, "monthly")
    naive = df.copy()
    naive.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        resample_sessions(naive, "daily")
    with pytest.raises(KeyError, match="side-OHLC"):
        resample_sessions(df.drop(columns=["open_bid"]), "daily")


# --------------------------------------------------------------------------
# Session-anchored INTRADAY bars (2h / 4h nested inside the 17:00-NY session)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("period,per_session,first_boundaries", [
    ("4h", 6, ["17:00", "21:00", "01:00", "05:00", "09:00", "13:00"]),
    ("2h", 12, ["17:00", "19:00", "21:00", "23:00", "01:00", "03:00"]),
])
def test_intraday_grid_is_anchored_to_the_1700_ny_open(period, per_session, first_boundaries):
    """A full session yields exactly 24/N bars, and the grid starts at 17:00 NY."""
    # One minute every 10 minutes across a full Tue-17:00 -> Wed-17:00 session.
    start = pd.Timestamp("2024-07-02 17:00", tz=NY_TZ)
    times = [(start + pd.Timedelta(minutes=10 * i)).strftime("%Y-%m-%d %H:%M")
             for i in range(6 * 24)]
    out = resample_sessions(_merged_ny(times), period)
    assert len(out) == per_session
    labels_ny = out.index.tz_convert(NY_TZ)
    assert [t.strftime("%H:%M") for t in labels_ny[: len(first_boundaries)]] == first_boundaries
    # Every bar carries the same number of source minutes on a full session.
    assert out["n_minutes"].nunique() == 1


def test_intraday_bars_nest_exactly_inside_the_daily_session():
    """The 4h sub-bars of one session must reconstruct that session's daily bar."""
    start = pd.Timestamp("2024-07-02 17:00", tz=NY_TZ)
    times = [(start + pd.Timedelta(minutes=10 * i)).strftime("%Y-%m-%d %H:%M")
             for i in range(6 * 24)]
    mids = 1.10 + np.sin(np.arange(6 * 24) / 7.0) * 0.01
    df = _merged_ny(times, mids=mids)
    sub = resample_sessions(df, "4h")
    day = resample_sessions(df, "daily")
    assert len(day) == 1
    assert sub["n_minutes"].sum() == day["n_minutes"].iloc[0]
    assert sub["mid_open"].iloc[0] == pytest.approx(day["mid_open"].iloc[0])
    assert sub["mid_close"].iloc[-1] == pytest.approx(day["mid_close"].iloc[0])
    assert sub["mid_high"].max() == pytest.approx(day["mid_high"].iloc[0])
    assert sub["mid_low"].min() == pytest.approx(day["mid_low"].iloc[0])
    # The daily bar's own label is the first sub-bar's label.
    assert sub.index[0] == day.index[0]


def test_intraday_bar_never_spans_the_weekend_gap():
    """Friday's last minutes and Sunday's first must land in different bars."""
    df = _merged_ny(
        ["2024-07-05 16:50", "2024-07-05 16:59",   # Friday, last 4h bar (13:00-17:00)
         "2024-07-07 17:00", "2024-07-07 17:30"],  # Sunday open, a new session
        mids=[1.10, 1.11, 1.30, 1.31],
    )
    out = resample_sessions(df, "4h")
    assert len(out) == 2
    labels_ny = out.index.tz_convert(NY_TZ)
    assert [t.strftime("%a %H:%M") for t in labels_ny] == ["Fri 13:00", "Sun 17:00"]
    assert out["n_minutes"].tolist() == [2, 2]
    # No bar mixes the two sides of the weekend.
    assert out["mid_high"].iloc[0] == pytest.approx(1.11)
    assert out["mid_low"].iloc[1] == pytest.approx(1.30)


def test_intraday_holiday_hole_leaves_a_missing_bar_not_a_merged_one():
    """A 4h window with no minutes simply does not appear."""
    df = _merged_ny(
        ["2024-07-02 17:30",   # bar 17:00-21:00
         "2024-07-03 05:30"],  # bar 05:00-09:00; 21:00, 01:00 windows empty
    )
    out = resample_sessions(df, "4h")
    assert len(out) == 2
    labels_ny = out.index.tz_convert(NY_TZ)
    assert [t.strftime("%H:%M") for t in labels_ny] == ["17:00", "05:00"]


@pytest.mark.parametrize("period", ["2h", "4h"])
def test_intraday_boundaries_are_dst_correct(period):
    """Across both US transitions the grid stays pinned to the NY session
    clock: every label sits on the 17:00-anchored wall-clock grid, even though
    the UTC offset changes."""
    hours = {"2h": 2, "4h": 4}[period]
    for day in ("2024-03-08", "2024-03-12", "2024-11-01", "2024-11-05"):
        start = pd.Timestamp(f"{day} 17:00", tz=NY_TZ)
        times = [(start + pd.Timedelta(minutes=30 * i)).strftime("%Y-%m-%d %H:%M")
                 for i in range(2 * 24)]
        out = resample_sessions(_merged_ny(times), period)
        labels_ny = out.index.tz_convert(NY_TZ)
        # Hours since the 17:00 anchor must be exact multiples of the step.
        offsets = [((t.hour - 17) % 24) for t in labels_ny]
        assert all(o % hours == 0 for o in offsets), f"{day}: {offsets}"
        assert len(out) == 24 // hours


def test_intraday_session_close_is_wall_clock_exact():
    labels = pd.DatetimeIndex(
        [pd.Timestamp(t, tz=NY_TZ) for t in ["2024-07-02 17:00", "2024-07-02 21:00"]]
    ).tz_convert("UTC")
    closes = session_close_utc(labels, "4h")
    assert list(closes.tz_convert(NY_TZ).strftime("%H:%M")) == ["21:00", "01:00"]


@pytest.mark.parametrize("period", ["2h", "4h"])
def test_intraday_no_lookahead_later_data_cannot_change_earlier_bars(period):
    """Corruption test: rewriting every 1-minute bar after time T must leave
    every output bar that CLOSES at or before T bit-identical."""
    start = pd.Timestamp("2024-07-02 17:00", tz=NY_TZ)
    times = [(start + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M")
             for i in range(12 * 24)]
    rng = np.random.default_rng(5)
    mids = 1.10 + np.cumsum(rng.normal(0, 0.0004, len(times)))
    df = _merged_ny(times, mids=mids)
    base = resample_sessions(df, period)
    closes = session_close_utc(base.index, period)

    for frac in (0.25, 0.5, 0.8):
        t = df.index[int(len(df) * frac)]
        corrupted = df.copy()
        after = corrupted.index > t
        for col in corrupted.columns:
            corrupted.loc[after, col] = 9.99
        got = resample_sessions(corrupted, period)
        settled = closes <= t
        pd.testing.assert_frame_equal(base[settled], got.loc[base.index[settled]])


def test_intraday_period_is_validated():
    df = _merged_ny(["2024-07-02 18:00"])
    with pytest.raises(ValueError, match="period must be one of"):
        resample_sessions(df, "3h")


def test_session_bars_are_storage_ready():
    df = _merged_ny(["2024-07-01 18:00", "2024-07-02 18:00"])
    out = resample_sessions(df, "daily")
    bid = side_bars(out, "bid")
    assert list(bid.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(bid) == 2
    assert (bid["volume"] == -1.0).all()
