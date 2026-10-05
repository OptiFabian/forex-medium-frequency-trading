"""Tests for triple-barrier labelling.

The load-bearing ones: the corruption test (no label may know its own future),
the ATR close-time lag, and monotonicity of touch time in k.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from fxalgo.labels.triple_barrier import (
    HORIZON,
    KS,
    _k_tag,
    build_labels,
    calendar_columns,
    label_columns,
    next_ny_close,
    scan_reference,
)

ATR = 0.0010  # flat ATR in price units


def _bars(mids, highs=None, lows=None, start="2024-07-02 00:00", freq="1min", index=None):
    mids = np.asarray(mids, dtype="float64")
    n = len(mids)
    if index is None:
        index = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "mid_open": mids,
            "mid_high": mids if highs is None else np.asarray(highs, dtype="float64"),
            "mid_low": mids if lows is None else np.asarray(lows, dtype="float64"),
            "mid_close": mids,
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


def _flat_atr(index, value=ATR):
    """A 15-min ATR series covering `index`, constant."""
    fifteen = pd.date_range(index[0] - pd.Timedelta("2h"), index[-1], freq="15min", tz="UTC")
    return pd.Series(value, index=fifteen, name="atr_14")


def _random_bars(n=1500, seed=0, base=1.10):
    rng = np.random.default_rng(seed)
    mid = base * np.exp(np.cumsum(rng.normal(0, 3e-5, n)))
    span = np.abs(rng.normal(0, 2e-5, n))
    return _bars(mid, highs=mid + span, lows=mid - span)


# ============================================================
# 1. Synthetic paths -- one barrier, the other, neither, both.
# ============================================================


def _one_event(highs, lows):
    """Build a tiny dataset whose event 0 sees exactly the given path."""
    n = HORIZON + 2
    mids = np.full(n, 1.10)
    hi = np.full(n, 1.10)
    lo = np.full(n, 1.10)
    hi[1 : 1 + len(highs)] = highs
    lo[1 : 1 + len(lows)] = lows
    bars = _bars(mids, highs=hi, lows=lo)
    return build_labels(bars, _flat_atr(bars.index))


def test_upper_touch_only():
    out = _one_event([1.10, 1.10, 1.10 + 2.1 * ATR], [1.10, 1.10, 1.10])
    assert out["label_2.0"].iloc[0] == 1
    assert out["bars_to_touch_2.0"].iloc[0] == 3
    assert out["touch_price_2.0"].iloc[0] == pytest.approx(1.10 + 2.0 * ATR)
    assert not out["ambiguous_2.0"].iloc[0]


def test_lower_touch_only():
    out = _one_event([1.10, 1.10, 1.10], [1.10, 1.10 - 2.5 * ATR, 1.10])
    assert out["label_2.0"].iloc[0] == -1
    assert out["bars_to_touch_2.0"].iloc[0] == 2
    assert out["touch_price_2.0"].iloc[0] == pytest.approx(1.10 - 2.0 * ATR)


def test_neither_touched_is_timeout():
    out = _one_event([1.10], [1.10])
    assert out["label_2.0"].iloc[0] == 0
    assert np.isnan(out["bars_to_touch_2.0"].iloc[0])
    assert np.isnan(out["touch_price_2.0"].iloc[0])


def test_same_bar_both_touched_is_ambiguous_and_labelled_adverse():
    out = _one_event([1.10, 1.10 + 3 * ATR], [1.10, 1.10 - 3 * ATR])
    assert out["ambiguous_2.0"].iloc[0]
    assert out["label_2.0"].iloc[0] == -1, "ambiguity must resolve to the ADVERSE outcome"
    assert out["bars_to_touch_2.0"].iloc[0] == 2


def test_touch_on_the_anchor_bar_is_one_bar():
    out = _one_event([1.10 + 2.5 * ATR], [1.10])
    assert out["label_2.0"].iloc[0] == 1
    assert out["bars_to_touch_2.0"].iloc[0] == 1


def test_anchor_is_open_of_t_plus_1_not_close_of_t():
    mids = np.arange(HORIZON + 3, dtype="float64") * 1e-4 + 1.10
    bars = _bars(mids)
    out = build_labels(bars, _flat_atr(bars.index))
    assert out["anchor_price"].iloc[0] == pytest.approx(bars["mid_open"].iloc[1])
    assert out["anchor_price"].iloc[0] != pytest.approx(bars["mid_close"].iloc[0])


# ============================================================
# 2. ATR lag: only a CLOSED 15-min bar may be used.
# ============================================================


def test_atr_comes_from_the_last_closed_15min_bar():
    """A 15-min bar covering t is still in progress and must be invisible."""
    idx = pd.date_range("2024-07-02 09:00", periods=HORIZON + 40, freq="1min", tz="UTC")
    bars = _bars(np.full(len(idx), 1.10), index=idx)
    atr = pd.Series(
        [0.001, 0.002, 0.003, 0.004],
        index=pd.DatetimeIndex(
            [pd.Timestamp(f"2024-07-02 0{h}:{m:02d}", tz="UTC")
             for h, m in ((8, 30), (8, 45), (9, 0), (9, 15))]
        ),
        name="atr_14",
    )
    out = build_labels(bars, atr)
    # t = 09:00 -> the 08:45 bar closed exactly at 09:00; the 09:00 bar has NOT.
    assert out["atr_at_entry"].loc[pd.Timestamp("2024-07-02 09:00", tz="UTC")] == 0.002
    # t = 09:07 falls MID-bar: still the 08:45 bar's ATR.
    assert out["atr_at_entry"].loc[pd.Timestamp("2024-07-02 09:07", tz="UTC")] == 0.002
    # t = 09:14 -- one minute before the 09:00 bar closes.
    assert out["atr_at_entry"].loc[pd.Timestamp("2024-07-02 09:14", tz="UTC")] == 0.002
    # t = 09:15 -- the 09:00 bar has now closed.
    assert out["atr_at_entry"].loc[pd.Timestamp("2024-07-02 09:15", tz="UTC")] == 0.003


def test_atr_before_any_closed_bar_is_nan_and_blocks_labelling():
    idx = pd.date_range("2024-07-02 09:00", periods=HORIZON + 5, freq="1min", tz="UTC")
    bars = _bars(np.full(len(idx), 1.10), index=idx)
    atr = pd.Series([0.001], index=pd.DatetimeIndex([pd.Timestamp("2024-07-02 10:00", tz="UTC")]))
    out = build_labels(bars, atr)
    assert np.isnan(out["atr_at_entry"].iloc[0])
    assert np.isnan(out["label_2.0"].iloc[0])


# ============================================================
# 3. Corruption test -- non-negotiable.
# ============================================================


def test_no_lookahead_corrupting_the_future_leaves_settled_labels_identical():
    """Overwrite everything after T; every event whose full 240-bar window
    closes at or before T must be bit-identical across EVERY column."""
    bars = _random_bars(n=1600, seed=11)
    atr = _flat_atr(bars.index)
    base = build_labels(bars, atr)

    for cut in (600, 900, 1200):
        corrupted = bars.copy()
        for col in ("mid_open", "mid_high", "mid_low", "mid_close"):
            corrupted.iloc[cut + 1 :, corrupted.columns.get_loc(col)] = 9.99
        got = build_labels(corrupted, atr)
        # event i's window closes at bar i + HORIZON
        settled = np.arange(len(bars)) + HORIZON <= cut
        for col in label_columns():
            a = base[col].to_numpy()[settled]
            b = got[col].to_numpy()[settled]
            if a.dtype.kind in "OU" or str(base[col].dtype) == "category":
                assert list(a) == list(b), f"{col} changed"
            else:
                np.testing.assert_array_equal(
                    a, b, err_msg=f"{col} leaked future information at T={cut}")


def test_corruption_of_the_atr_source_after_t_does_not_change_earlier_atr():
    bars = _random_bars(n=900, seed=12)
    atr = _flat_atr(bars.index)
    base = build_labels(bars, atr)
    cut_ts = bars.index[400]
    corrupted_atr = atr.copy()
    corrupted_atr[corrupted_atr.index > cut_ts] = 99.0
    got = build_labels(bars, corrupted_atr)
    mask = bars.index <= cut_ts
    np.testing.assert_array_equal(
        base["atr_at_entry"].to_numpy()[mask], got["atr_at_entry"].to_numpy()[mask])


# ============================================================
# 4. Monotonicity in k -- catches most barrier-comparison bugs.
# ============================================================


def test_touch_time_is_non_decreasing_in_k():
    """Wider barriers can only be reached later, never earlier. And if a wide
    barrier is touched, every narrower one must have been touched too."""
    bars = _random_bars(n=4000, seed=13)
    out = build_labels(bars, _flat_atr(bars.index, value=2e-4))
    tags = [_k_tag(k) for k in KS]
    for a, b in pairwise(tags):
        ta, tb = out[f"bars_to_touch_{a}"], out[f"bars_to_touch_{b}"]
        both = ta.notna() & tb.notna()
        assert both.sum() > 100
        assert (ta[both] <= tb[both]).all(), f"touch time went backwards from k={a} to k={b}"
        # a touch at the wider barrier implies a touch at the narrower one
        assert ta[tb.notna()].notna().all(), f"k={b} touched where k={a} did not"


def test_timeout_rate_is_non_decreasing_in_k():
    bars = _random_bars(n=4000, seed=14)
    out = build_labels(bars, _flat_atr(bars.index, value=2e-4))
    rates = [(out[f"label_{_k_tag(k)}"] == 0).mean() for k in KS]
    assert all(x <= y + 1e-12 for x, y in pairwise(rates))


# ============================================================
# 5. Fast path == reference implementation.
# ============================================================


def test_fast_scan_matches_the_slow_reference():
    bars = _random_bars(n=2500, seed=15)
    atr_series = _flat_atr(bars.index, value=1.5e-4)
    out = build_labels(bars, atr_series)

    anchor = out["anchor_price"].to_numpy()
    atr = out["atr_at_entry"].to_numpy()
    rng = np.random.default_rng(0)
    events = rng.choice(len(bars) - HORIZON, size=400, replace=False)
    ref = scan_reference(
        bars["mid_high"].to_numpy(), bars["mid_low"].to_numpy(),
        anchor, atr, KS, HORIZON, events=events)
    for k in KS:
        tag = _k_tag(k)
        np.testing.assert_array_equal(out[f"label_{tag}"].to_numpy()[events], ref[k]["label"])
        np.testing.assert_array_equal(out[f"bars_to_touch_{tag}"].to_numpy()[events], ref[k]["bars"])
        np.testing.assert_array_equal(
            out[f"ambiguous_{tag}"].to_numpy()[events], ref[k]["ambiguous"])


# ============================================================
# 6. Gap / calendar columns -- clock only, no lookahead.
# ============================================================


def test_next_ny_close_is_dst_correct_and_strictly_forward():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-07-02 20:59", tz="UTC"),   # 16:59 NY (EDT) -> 21:00 UTC
        pd.Timestamp("2024-07-02 21:00", tz="UTC"),   # exactly 17:00 NY -> next day
        pd.Timestamp("2024-01-08 21:59", tz="UTC"),   # 16:59 NY (EST) -> 22:00 UTC
    ])
    closes = next_ny_close(idx)
    assert list(closes.tz_convert("America/New_York").strftime("%H:%M")) == ["17:00"] * 3
    assert (closes > idx).all()
    assert closes[0] == pd.Timestamp("2024-07-02 21:00", tz="UTC")
    assert closes[1] == pd.Timestamp("2024-07-03 21:00", tz="UTC")


def test_calendar_columns_flag_the_upcoming_gap():
    idx = pd.DatetimeIndex([
        pd.Timestamp("2024-07-02 20:00", tz="UTC"),   # 16:00 NY Tue -> 60 min to close
        pd.Timestamp("2024-07-02 12:00", tz="UTC"),   # 08:00 NY Tue -> 540 min, no gap
        pd.Timestamp("2024-07-05 20:00", tz="UTC"),   # 16:00 NY FRIDAY -> weekend
    ])
    cal = calendar_columns(idx)
    assert cal["minutes_to_close"].tolist() == [60, 540, 60]
    assert cal["spans_gap"].tolist() == [True, False, True]
    assert list(cal["gap_type"]) == ["daily_reset", "none", "weekend"]
    assert cal["is_friday"].tolist() == [False, False, True]
    assert cal["bars_to_gap"].tolist() == cal["minutes_to_close"].tolist()


def test_calendar_columns_depend_only_on_the_clock():
    """Same timestamps + totally different prices => identical calendar columns."""
    bars = _random_bars(n=800, seed=16)
    other = _random_bars(n=800, seed=17)
    other.index = bars.index
    a = build_labels(bars, _flat_atr(bars.index))
    b = build_labels(other, _flat_atr(bars.index))
    for col in ("spans_gap", "bars_to_gap", "is_friday", "minutes_to_close"):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy(), err_msg=col)
    assert list(a["gap_type"]) == list(b["gap_type"])


# ============================================================
# 7. Structure.
# ============================================================


def test_tail_events_are_nan_not_truncated():
    bars = _random_bars(n=600, seed=18)
    out = build_labels(bars, _flat_atr(bars.index))
    assert len(out) == len(bars), "rows must not be silently dropped"
    tail = out.iloc[-HORIZON:]
    for k in KS:
        assert tail[f"label_{_k_tag(k)}"].isna().all()
    assert out.iloc[: len(bars) - HORIZON][f"label_{_k_tag(2.0)}"].notna().all()


def test_requires_mid_columns_and_sorted_index():
    bars = _random_bars(n=300, seed=19)
    with pytest.raises(KeyError):
        build_labels(bars.drop(columns=["mid_high"]), _flat_atr(bars.index))
    with pytest.raises(ValueError, match="sorted"):
        build_labels(bars.iloc[::-1], _flat_atr(bars.index))
