"""Tests for `fxalgo.data.quality` - gap classification.

Reminders:
  - America/New_York 17:00 = 22:00 UTC during winter (EST, UTC-5).
  - America/New_York 17:00 = 21:00 UTC during summer (EDT, UTC-4).
  - US DST 2024 starts Sun 2024-03-10, ends Sun 2024-11-03.
"""

from __future__ import annotations

import pandas as pd

from fxalgo.data.quality import analyze


def _bars_from_index(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [1.0] * len(idx),
            "high": [1.0] * len(idx),
            "low": [1.0] * len(idx),
            "close": [1.0] * len(idx),
            "volume": [0.0] * len(idx),
        }
    )


def test_empty_summary() -> None:
    s = analyze(_bars_from_index(pd.DatetimeIndex([], tz="UTC")))
    assert s.row_count == 0
    assert s.earliest is None
    assert s.weekend_gaps == 0
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 0


def test_continuous_series_has_no_gaps() -> None:
    # Monday morning, 60 contiguous 1-min bars (well before any rollover)
    idx = pd.date_range("2024-03-04 09:00", periods=60, freq="1min", tz="UTC")
    s = analyze(_bars_from_index(idx))
    assert s.row_count == 60
    assert s.weekend_gaps == 0
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 0


# -------- Weekend --------


def test_weekend_gap_winter_est() -> None:
    # Jan 2024 = EST: NY 17:00 = 22:00 UTC.
    # Pre: Fri 2024-01-05 21:59 UTC (= 16:59 NY)
    # Post: Sun 2024-01-07 22:00 UTC (= 17:00 NY)
    pre = pd.date_range("2024-01-05 21:00", "2024-01-05 21:59", freq="1min", tz="UTC")
    post = pd.date_range("2024-01-07 22:00", "2024-01-07 22:59", freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.weekend_gaps == 1
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 0


def test_weekend_gap_summer_edt() -> None:
    # July 2024 = EDT: NY 17:00 = 21:00 UTC.
    # Pre: Fri 2024-07-05 20:59 UTC (= 16:59 NY)
    # Post: Sun 2024-07-07 21:15 UTC (= 17:15 NY) - reopen delayed by reset
    pre = pd.date_range("2024-07-05 20:00", "2024-07-05 20:59", freq="1min", tz="UTC")
    post = pd.date_range("2024-07-07 21:15", "2024-07-07 22:00", freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.weekend_gaps == 1
    assert s.active_gaps == 0


# -------- Daily reset --------


def test_daily_reset_winter_est() -> None:
    # Tue 2024-03-05 (EST, pre-DST): NY 17:00 = 22:00 UTC.
    # Pre: 21:50..21:54 UTC (= 16:50..16:54 NY)
    # Post: 22:15..22:20 UTC (= 17:15..17:20 NY)
    # Missing 21:55..22:14 UTC = NY 16:55..17:14 (the 15-min reset window).
    pre = pd.date_range("2024-03-05 21:50", "2024-03-05 21:54", freq="1min", tz="UTC")
    post = pd.date_range("2024-03-05 22:15", "2024-03-05 22:20", freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.daily_reset_gaps == 1
    assert s.weekend_gaps == 0
    assert s.active_gaps == 0


def test_daily_reset_summer_edt() -> None:
    # Wed 2024-03-13 (EDT, post-DST): NY 17:00 = 21:00 UTC.
    # Pre: 20:50..20:54 UTC (= 16:50..16:54 NY)
    # Post: 21:15..21:20 UTC (= 17:15..17:20 NY)
    pre = pd.date_range("2024-03-13 20:50", "2024-03-13 20:54", freq="1min", tz="UTC")
    post = pd.date_range("2024-03-13 21:15", "2024-03-13 21:20", freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.daily_reset_gaps == 1
    assert s.weekend_gaps == 0
    assert s.active_gaps == 0


def test_friday_reset_is_absorbed_into_weekend_not_daily() -> None:
    # Fri 2024-03-15 (EDT): NY 17:00 = 21:00 UTC.
    # Pre: Fri 20:59 UTC (= 16:59 NY)
    # Post: Sun 2024-03-17 21:14 UTC (= 17:14 NY, delayed reopen)
    pre = pd.date_range("2024-03-15 20:00", "2024-03-15 20:59", freq="1min", tz="UTC")
    post = pd.date_range("2024-03-17 21:14", "2024-03-17 22:00", freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.weekend_gaps == 1
    assert s.daily_reset_gaps == 0  # the Fri 17:00 reset must NOT be double-counted
    assert s.active_gaps == 0


# -------- Active --------


def test_active_trading_gap() -> None:
    # Monday morning gap (well before NY 17:00):
    # pre 09:00..09:29 UTC (= ~04-05 NY), post 10:00..10:29 UTC.
    pre = pd.date_range("2024-03-04 09:00", periods=30, freq="1min", tz="UTC")
    post = pd.date_range("2024-03-04 10:00", periods=30, freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.weekend_gaps == 0
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 1
    assert s.largest_active_gap == pd.Timedelta(minutes=31)


def test_long_intraweek_gap_is_active_not_weekend() -> None:
    # A 3-day gap mid-week shouldn't trigger the weekend rule even though
    # duration > 24h, because the NY-anchored bounds won't fit.
    pre = pd.date_range("2024-03-04 09:00", periods=10, freq="1min", tz="UTC")
    post = pd.date_range("2024-03-07 09:00", periods=10, freq="1min", tz="UTC")
    s = analyze(_bars_from_index(pre.append(post)))
    assert s.weekend_gaps == 0
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 1


# -------- Mixed --------


def test_mixed_gaps_no_reset() -> None:
    """Mix of intraday gaps + weekend, without a Tuesday-reset block.

    Confirms that long non-weekend gaps stay 'active' and don't get
    mis-classified just because they overlap a reset window.
    """
    morning_a = pd.date_range("2024-03-05 09:00", periods=10, freq="1min", tz="UTC")
    morning_b = pd.date_range("2024-03-05 10:00", periods=10, freq="1min", tz="UTC")
    post_reset_zone = pd.date_range("2024-03-05 22:15", periods=6, freq="1min", tz="UTC")
    pre_weekend = pd.date_range("2024-03-08 21:50", periods=5, freq="1min", tz="UTC")
    post_weekend = pd.date_range("2024-03-10 22:14", periods=10, freq="1min", tz="UTC")

    idx = (
        morning_a.append(morning_b).append(post_reset_zone)
        .append(pre_weekend).append(post_weekend)
    )
    s = analyze(_bars_from_index(idx))

    # Three intra-week gaps, none fit the daily-reset window:
    #   09:09 -> 10:00 (51 min mid-morning)
    #   10:09 -> 22:15 (~12h, crosses the Tue reset but far too large to BE the reset)
    #   22:20 Tue -> 21:50 Fri (~3 days)
    # And one weekend: 21:54 Fri -> 22:14 Sun.
    assert s.weekend_gaps == 1
    assert s.daily_reset_gaps == 0
    assert s.active_gaps == 3


def test_mixed_with_real_daily_reset() -> None:
    """A more representative mix: morning gap, daily reset, weekend close."""
    # Tue 2024-03-05 (EST):
    morning_a = pd.date_range("2024-03-05 12:00", periods=10, freq="1min", tz="UTC")
    morning_b = pd.date_range("2024-03-05 13:00", periods=10, freq="1min", tz="UTC")  # active gap
    pre_reset = pd.date_range("2024-03-05 21:50", periods=5, freq="1min", tz="UTC")
    post_reset = pd.date_range("2024-03-05 22:15", periods=10, freq="1min", tz="UTC")  # daily_reset

    # ... continue to Friday 21:54 UTC (16:54 NY)
    pre_weekend = pd.date_range("2024-03-08 21:50", periods=5, freq="1min", tz="UTC")
    post_weekend = pd.date_range("2024-03-10 22:14", periods=10, freq="1min", tz="UTC")  # weekend

    idx = (
        morning_a.append(morning_b).append(pre_reset).append(post_reset)
        .append(pre_weekend).append(post_weekend)
    )
    s = analyze(_bars_from_index(idx))

    # 12:09 -> 13:00 = active
    # 13:09 -> 21:50 = active (a multi-hour mid-day gap)
    # 21:54 -> 22:15 (Tue) = daily_reset
    # 22:24 (Tue) -> 21:50 (Fri) = active
    # 21:54 (Fri) -> 22:14 (Sun) = weekend
    assert s.weekend_gaps == 1
    assert s.daily_reset_gaps == 1
    assert s.active_gaps == 3
