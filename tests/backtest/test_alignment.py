"""Tests for cross-timeframe alignment (`strategies.multi_timeframe`).

The load-bearing test in this file is the cross-timeframe causality one: a
higher-TF bar that has not CLOSED yet must be invisible on the fine timeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.quality import NY_TZ
from fxalgo.data.resample import session_close_utc
from fxalgo.strategies.multi_timeframe import align_completed_signal


def _htf(values, start="2024-01-01 09:00", freq="1h"):
    idx = pd.date_range(start=start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(np.asarray(values, dtype=np.int8), index=idx, name="htf_bias")


def _minutes(times):
    return pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t in times])


# ============================================================
# 1. Cross-timeframe causality: only CLOSED higher-TF bars are visible.
# ============================================================


def test_bias_comes_from_the_last_closed_higher_tf_bar():
    """The worked example: at 10:37 the 1-hour bias is the 09:00-10:00 bar's."""
    sig = _htf([1, -1, 1])  # bars starting 09:00, 10:00, 11:00
    target = _minutes(
        ["2024-01-01 09:30", "2024-01-01 09:59", "2024-01-01 10:00",
         "2024-01-01 10:37", "2024-01-01 11:00", "2024-01-01 11:05"]
    )
    bias = align_completed_signal(sig, target, "1h")
    #  09:30 / 09:59 -> the 09:00 bar is still IN PROGRESS -> no bias yet
    #  10:00         -> the 09:00 bar has just closed        -> +1
    #  10:37         -> still the 09:00 bar (10:00 bar open)  -> +1
    #  11:00 / 11:05 -> the 10:00 bar has closed              -> -1
    np.testing.assert_array_equal(bias.to_numpy(), [0, 0, 1, 1, -1, -1])


def test_in_progress_bar_is_invisible_no_lookahead_under_corruption():
    """Rewriting higher-TF bars that have not closed by time T cannot change
    the aligned bias at any minute <= T."""
    sig = _htf([1, -1, 1, -1, 1])  # 09:00 .. 13:00
    target = pd.date_range("2024-01-01 09:00", "2024-01-01 14:00", freq="1min", tz="UTC")
    base = align_completed_signal(sig, target, "1h")

    for cutoff in ("2024-01-01 10:37", "2024-01-01 11:00", "2024-01-01 12:59"):
        t = pd.Timestamp(cutoff, tz="UTC")
        corrupted = sig.copy()
        # Wreck every bar that is NOT fully closed by t.
        not_closed = corrupted.index + pd.Timedelta("1h") > t
        corrupted[not_closed] = -corrupted[not_closed]
        after = align_completed_signal(corrupted, target, "1h")
        np.testing.assert_array_equal(
            base[base.index <= t].to_numpy(),
            after[after.index <= t].to_numpy(),
            err_msg=f"bias before {cutoff} changed when a not-yet-closed bar was altered",
        )


def test_alignment_works_for_15min_and_30min():
    sig = _htf([1, -1], start="2024-01-01 09:00", freq="15min")
    target = _minutes(["2024-01-01 09:14", "2024-01-01 09:15", "2024-01-01 09:30"])
    np.testing.assert_array_equal(
        align_completed_signal(sig, target, "15min").to_numpy(), [0, 1, -1]
    )
    sig30 = _htf([1, -1], start="2024-01-01 09:00", freq="30min")
    target30 = _minutes(["2024-01-01 09:29", "2024-01-01 09:30", "2024-01-01 10:00"])
    np.testing.assert_array_equal(
        align_completed_signal(sig30, target30, "30min").to_numpy(), [0, 1, -1]
    )


def test_alignment_carries_last_completed_bar_across_a_gap():
    """A missing higher-TF bar (empty window) holds the previous bias, it does
    not reach forward to the next one."""
    sig = pd.Series(
        np.array([1, -1], dtype=np.int8),
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01 09:00", tz="UTC"),
             pd.Timestamp("2024-01-01 12:00", tz="UTC")]  # 10:00 and 11:00 missing
        ),
    )
    target = _minutes(
        ["2024-01-01 10:30", "2024-01-01 11:30", "2024-01-01 12:59", "2024-01-01 13:00"]
    )
    np.testing.assert_array_equal(
        align_completed_signal(sig, target, "1h").to_numpy(), [1, 1, 1, -1]
    )


# ---- session (daily / weekly) bars: not fixed-width ----


def _sessions(values, starts):
    idx = pd.DatetimeIndex([pd.Timestamp(s, tz=NY_TZ) for s in starts]).tz_convert("UTC")
    return pd.Series(np.asarray(values, dtype=np.int8), index=idx, name="htf_bias")


def test_daily_bias_comes_from_the_last_completed_17_00_ny_session():
    """A daily bar labeled Mon 17:00 NY closes at Tue 17:00 NY; it must be
    invisible to every 15-min bar before that."""
    sig = _sessions([1, -1], ["2024-07-01 17:00", "2024-07-02 17:00"])  # Mon, Tue opens
    target = pd.DatetimeIndex(
        [pd.Timestamp(t, tz=NY_TZ) for t in
         ["2024-07-02 09:00",   # inside the Mon session -> still in progress
          "2024-07-02 16:45",   # last bar before the close
          "2024-07-02 17:00",   # Mon session has just closed
          "2024-07-03 10:00",   # still Mon's bias (Tue session in progress)
          "2024-07-03 17:00"]]  # Tue session closed
    ).tz_convert("UTC")
    bias = align_completed_signal(sig, target, "daily")
    np.testing.assert_array_equal(bias.to_numpy(), [0, 0, 1, 1, -1])


def test_weekly_bias_closes_on_friday_not_seven_days_later():
    """A weekly bar labeled Sunday 17:00 NY ends at FRIDAY 17:00 NY (five wall
    days). Using a naive 7-day width would hide it for two extra days."""
    sig = _sessions([1, -1], ["2024-06-30 17:00", "2024-07-07 17:00"])  # two Sunday opens
    target = pd.DatetimeIndex(
        [pd.Timestamp(t, tz=NY_TZ) for t in
         ["2024-07-05 16:45",   # Friday, week still in progress
          "2024-07-05 17:00",   # Friday close -> week 1 complete
          "2024-07-07 17:00",   # Sunday open of week 2 -> week 1's bias
          "2024-07-10 12:00"]]  # midweek 2 -> still week 1's bias
    ).tz_convert("UTC")
    bias = align_completed_signal(sig, target, "weekly")
    np.testing.assert_array_equal(bias.to_numpy(), [0, 1, 1, 1])


def test_session_close_is_dst_correct():
    """A session that spans a DST switch is 23 or 25 hours long in absolute
    time, and still closes at 17:00 NY -- not at a fixed UTC offset."""
    labels = pd.DatetimeIndex(
        [pd.Timestamp(t, tz=NY_TZ) for t in
         ["2024-03-09 17:00",   # session spanning the spring-forward (Mar 10)
          "2024-11-02 17:00",   # session spanning the fall-back (Nov 3)
          "2024-07-01 17:00"]]  # ordinary session
    ).tz_convert("UTC")
    closes = session_close_utc(labels, "daily")
    assert list(closes.tz_convert(NY_TZ).hour) == [17, 17, 17]
    assert (closes - labels).tolist() == [
        pd.Timedelta(hours=23), pd.Timedelta(hours=25), pd.Timedelta(hours=24)
    ]


@pytest.mark.parametrize("period,starts", [
    ("daily", ["2024-07-01 17:00", "2024-07-02 17:00", "2024-07-03 17:00",
               "2024-07-05 17:00"]),
    ("weekly", ["2024-06-30 17:00", "2024-07-07 17:00", "2024-07-14 17:00"]),
])
def test_session_alignment_no_lookahead_under_corruption(period, starts):
    """Rewriting session bars that have not closed by T cannot change the bias
    at any bar <= T."""
    sig = _sessions([1, -1, 1, -1][: len(starts)], starts)
    closes = session_close_utc(sig.index, period)
    target = pd.date_range(sig.index[0], closes[-1], freq="15min", tz="UTC")
    base = align_completed_signal(sig, target, period)

    for t in target[:: max(len(target) // 7, 1)]:
        corrupted = sig.copy()
        not_closed = closes > t
        corrupted[not_closed] = -corrupted[not_closed]
        after = align_completed_signal(corrupted, target, period)
        np.testing.assert_array_equal(
            base[base.index <= t].to_numpy(),
            after[after.index <= t].to_numpy(),
            err_msg=f"{period} bias at or before {t} changed when an open session was altered",
        )


def test_alignment_rejects_unsorted_or_wrong_types():
    sig = _htf([1, -1])
    with pytest.raises(ValueError, match="sorted"):
        align_completed_signal(sig.iloc[::-1], _minutes(["2024-01-01 11:00"]), "1h")
    with pytest.raises(TypeError):
        align_completed_signal(pd.Series([1, 0]), _minutes(["2024-01-01 11:00"]), "1h")
