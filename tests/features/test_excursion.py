"""Tests for the excursion-history family.

The load-bearing ones:

1. COUNTER UNITS (`test_counter_counts_higher_tf_bars_not_15min_rows`). The
   specific failure mode this family risks: computing a run length on the
   ALIGNED 15-min series instead of on the higher timeframe's own bars, which
   returns 16x too much at 4h and ~470x at weekly. Price is held above the band
   for exactly 3 closed bars and the counter must read 3 -- and the test also
   pins what the wrong implementation WOULD have produced, so it fails loudly
   rather than silently drifting.
2. CAUSALITY. Same standard as the htf build: the corruption test runs the full
   chain from 1-minute bars on synthetic data carrying real FX weekend geometry,
   with a companion test asserting the corruption actually bites.
3. ALIGNMENT, including the weekly case where the Friday 17:00 NY close has no
   15-min bar and the value first becomes visible at Sunday's reopen.
4. SCALE INVARIANCE at 0.01x / 137.5x / 1000x. Every one of the 38 columns must
   pass -- they all derive from `bb_position` (already scale-free) or from a
   rank, so a failure here is a bug, not an exception.
5. WARMUP is NaN, never partial, and `bars_since_band_cross` is NaN before the
   first crossing rather than 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.resample import (
    resample_merged,
    resample_sessions,
    session_close_utc,
    session_keys,
)
from fxalgo.features import excursion, htf
from fxalgo.features.build import apply_pipeline
from fxalgo.strategies.multi_timeframe import align_completed_series

CLOCK_TFS = ("30min", "1h")
SESSION_TFS = ("2h", "4h", "daily", "weekly")
NEW_COLS = excursion.feature_columns()
PRICE_COLS = [
    "open_bid", "high_bid", "low_bid", "close_bid",
    "open_ask", "high_ask", "low_ask", "close_ask",
]


# --------------------------------------------------------------- fixtures


def _minute_index(start: str, end: str) -> pd.DatetimeIndex:
    """1-minute UTC index with the FX weekend cut out (see test_htf)."""
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    keys = session_keys(idx, "daily")
    return idx[~np.isin(keys.weekday, (5, 6))]


def _synthetic_minutes(start="2024-01-01", end="2025-01-01", seed=11) -> pd.DataFrame:
    idx = _minute_index(start, end)
    rng = np.random.default_rng(seed)
    n = len(idx)
    mid_close = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 8e-5, n)))
    mid_open = np.concatenate([[mid_close[0]], mid_close[:-1]])
    wiggle = np.abs(rng.normal(0.0, 6e-5, n))
    mid_high = np.maximum(mid_open, mid_close) + wiggle
    mid_low = np.minimum(mid_open, mid_close) - wiggle
    spread = 0.00008 + np.abs(rng.normal(0.0, 0.00004, n))

    half = spread / 2.0
    df = pd.DataFrame(index=idx)
    for name, mid in (("open", mid_open), ("high", mid_high),
                      ("low", mid_low), ("close", mid_close)):
        df[f"{name}_bid"] = mid - half
        df[f"{name}_ask"] = mid + half
    df.index.name = "timestamp"
    return df


def _frames(minutes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = apply_pipeline(resample_merged(minutes, "15min"))
    frames = {tf: apply_pipeline(resample_merged(minutes, tf)) for tf in CLOCK_TFS}
    frames.update({tf: apply_pipeline(resample_sessions(minutes, tf)) for tf in SESSION_TFS})
    return base, frames


def _build(minutes: pd.DataFrame, **kw) -> pd.DataFrame:
    base, frames = _frames(minutes)
    # A synthetic year holds ~52 weeks and ~25k 15-min bars, so the shipped
    # windows (52 weeks / 23,710 bars) would never become valid here. Both are
    # parameters precisely so the test can shorten them; the causality being
    # tested does not depend on their length.
    kw.setdefault("spread_pctile_weeks", 8)
    kw.setdefault("spread_pctile_rolling_window", 500)
    return excursion.add_features(base, frames, **kw)


@pytest.fixture(scope="module")
def minutes() -> pd.DataFrame:
    return _synthetic_minutes()


@pytest.fixture(scope="module")
def built(minutes) -> pd.DataFrame:
    return _build(minutes)


# ------------------------------------------------- 0. shape and definitions


def test_adds_exactly_the_documented_columns(minutes, built):
    base, _ = _frames(minutes)
    added = [c for c in built.columns if c not in base.columns]
    assert added == NEW_COLS
    # 12 run + 12 occupancy + 6 recency + 6 depth + 2 spread percentiles
    assert len(NEW_COLS) == 38


def test_occupancy_columns_carry_the_window_in_their_name():
    assert "frac_above_band_4h_20" in NEW_COLS
    assert "bars_above_band_4h" in NEW_COLS


def test_at_most_one_run_counter_is_nonzero(built):
    for tf in htf.HIGHER_TIMEFRAMES:
        a = built[f"bars_above_band_{tf}"]
        b = built[f"bars_below_band_{tf}"]
        both = (a > 0) & (b > 0)
        assert not both.any(), f"{tf}: above and below both nonzero"


def test_run_counter_is_capped(built):
    for tf in htf.HIGHER_TIMEFRAMES:
        for side in ("above", "below"):
            s = built[f"bars_{side}_band_{tf}"].dropna()
            assert s.max() <= excursion.RUN_CAP


# ---------------------------------- 1. THE COUNTER-UNIT TEST (the whole point)


def _flat_htf_positions(minutes: pd.DataFrame, tf: str, pattern: list[float]) -> pd.Series:
    """A bb_position series on `tf`'s REAL bar labels, taking `pattern` values.

    Uses the actual resampled index so the close instants, and therefore the
    projection geometry, are the real ones.
    """
    frame = (resample_sessions(minutes, tf) if tf in SESSION_TFS
             else resample_merged(minutes, tf))
    idx = frame.index[: len(pattern)]
    return pd.Series(pattern[: len(idx)], index=idx, dtype="float64")


@pytest.mark.parametrize(
    "tf,rows_per_bar",
    [("4h", 16), ("daily", 96), ("weekly", 480)],
)
def test_counter_counts_higher_tf_bars_not_15min_rows(minutes, tf, rows_per_bar):
    """Above the band for exactly 3 CLOSED bars -> the counter reads 3.

    A run computed on the aligned 15-min series instead would read
    3 x rows_per_bar. The test asserts the correct value AND that it differs
    from the wrong one, so the wrong implementation cannot pass.
    """
    pattern = [0.5] * 6 + [1.2, 1.4, 1.1] + [0.5] * 15
    pos = _flat_htf_positions(minutes, tf, pattern)
    block = excursion.excursion_block(pos)

    # On the higher timeframe's own bars the run is 1, 2, 3 across the excursion.
    assert block["bars_above_band"].iloc[6:9].tolist() == [1.0, 2.0, 3.0]

    target = resample_merged(minutes, "15min").index
    aligned = align_completed_series(block["bars_above_band"], target, tf)

    # The third outside bar closes here; from that instant until the next bar
    # closes, the projected value must be exactly 3.
    third_close = session_close_utc(pos.index[8:9], tf)[0] if tf in SESSION_TFS \
        else pos.index[8] + pd.Timedelta(tf)
    visible = aligned.loc[aligned.index >= third_close]
    assert len(visible) >= 2
    assert visible.iloc[0] == 3.0

    held = int((aligned == 3.0).sum())
    assert held > 1, "expected the value to be held across intervening 15-min rows"
    wrong = 3 * rows_per_bar
    assert visible.iloc[0] != wrong, (
        f"{tf}: counter returned the 15-min row count ({wrong}), not the bar count"
    )
    assert aligned.max() == 3.0


def test_counter_holds_constant_between_higher_tf_closes(minutes, built):
    """The documented step-function behaviour, asserted rather than assumed."""
    base, frames = _frames(minutes)
    closes = session_close_utc(frames["4h"].index, "4h")
    col = built["bars_above_band_4h"]
    # Group the 15-min rows by which higher-TF bar was last closed; every group
    # must be constant.
    which = np.searchsorted(closes.to_numpy(), col.index.to_numpy(), side="right")
    grouped = pd.Series(col.to_numpy(), index=which).groupby(level=0).nunique()
    assert (grouped <= 1).all(), "4h counter changed between 4h bar closes"
    assert (pd.Series(col.to_numpy(), index=which).groupby(level=0).size() > 1).any()


# ------------------------------------------- 2. hand-checked state machines


def _pos_series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


def test_run_resets_on_a_single_bar_back_inside():
    pos = _pos_series([1.2] * 6 + [0.5] + [1.2] * 6)
    b = excursion.excursion_block(pos, fraction_window=3)
    assert b["bars_above_band"].tolist() == [1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]


def test_occupancy_sees_what_the_run_counter_forgets():
    """The 6/1/6 path: run counter says 1, occupancy says 12 of 13."""
    pos = _pos_series([1.2] * 6 + [0.5] + [1.2] * 6)
    b = excursion.excursion_block(pos, fraction_window=13)
    assert b["bars_above_band"].iloc[7] == 1.0
    assert b["frac_above_band"].iloc[-1] == pytest.approx(12 / 13)


def test_bars_since_cross_is_one_at_the_crossing_and_nan_before_any():
    pos = _pos_series([0.5, 0.5, 1.2, 1.3, 0.5, 0.4, 0.5, -0.1, 0.5])
    b = excursion.excursion_block(pos, fraction_window=2)
    s = b["bars_since_band_cross"]
    assert s.iloc[:2].isna().all(), "no crossing has happened yet"
    assert s.iloc[2] == 1.0          # inside -> outside
    assert s.iloc[3] == 2.0
    assert s.iloc[6] == 5.0          # keeps counting while inside
    assert s.iloc[7] == 1.0          # next crossing, other side, resets to 1


def test_series_starting_outside_the_band_has_no_crossing_yet():
    pos = _pos_series([1.2, 1.3, 1.4, 0.5])
    b = excursion.excursion_block(pos, fraction_window=2)
    assert b["bars_since_band_cross"].iloc[:3].isna().all()
    assert b["bars_above_band"].iloc[:3].tolist() == [1, 2, 3]


def test_depth_tracks_the_current_excursion_only():
    pos = _pos_series([0.5, 1.2, 1.9, 1.1, 0.5, 1.05])
    b = excursion.excursion_block(pos, fraction_window=2)
    d = b["max_stretch_current_excursion"]
    assert d.iloc[0] == 0.0
    assert d.iloc[1] == pytest.approx(0.7)
    assert d.iloc[2] == pytest.approx(1.4)
    assert d.iloc[3] == pytest.approx(1.4), "max is held for the whole excursion"
    assert d.iloc[4] == 0.0, "back inside resets"
    assert d.iloc[5] == pytest.approx(0.55), "a new excursion starts fresh"


def test_depth_resets_when_the_excursion_flips_side():
    pos = _pos_series([1.9, -0.2])
    b = excursion.excursion_block(pos, fraction_window=2)
    assert b["max_stretch_current_excursion"].iloc[1] == pytest.approx(0.7)


def test_nan_resets_every_state_and_blanks_the_row():
    pos = _pos_series([1.2, 1.3, np.nan, 1.2, 1.3])
    b = excursion.excursion_block(pos, fraction_window=2)
    assert b.iloc[2].isna().all()
    assert b["bars_above_band"].tolist()[3:] == [1.0, 2.0]


def test_cap_folds_the_tail_without_touching_the_body():
    pos = _pos_series([1.2] * 10)
    b = excursion.excursion_block(pos, run_cap=6, fraction_window=2)
    assert b["bars_above_band"].tolist() == [1, 2, 3, 4, 5, 6, 6, 6, 6, 6]


# --------------------------------------------------------- 3. spread rank


def test_spread_percentile_is_a_trailing_same_slot_rank():
    idx = pd.date_range("2024-01-01", periods=8 * 672, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    s = pd.Series(rng.random(len(idx)) * 1e-4, index=idx)
    out = excursion.spread_pctile_trailing(s, weeks=4)
    slot = excursion.minute_of_week(idx)
    # Pick a slot and verify one value by hand against its own history.
    target = slot[slot == slot.iloc[0]].index
    hist = s.loc[target]
    k = 5  # 6th same-slot observation; a 4-wide window covers positions 2..5
    window = hist.iloc[k - 3 : k + 1]
    assert len(window) == 4
    expected = window.rank(pct=True).iloc[-1]
    assert out.loc[target[k]] == pytest.approx(expected)
    assert out.loc[target[:3]].isna().all()
    assert out.dropna().between(0.0, 1.0).all()


def test_spread_percentile_uses_new_york_slots_not_utc():
    """The relabelling must move with the NY offset, not sit on a UTC grid."""
    winter = pd.Timestamp("2024-01-10 14:00", tz="UTC")   # 09:00 NY (EST)
    summer = pd.Timestamp("2024-07-10 13:00", tz="UTC")   # 09:00 NY (EDT)
    idx = pd.DatetimeIndex([winter, summer])
    slots = excursion.minute_of_week(idx)
    assert slots.iloc[0] == slots.iloc[1]
    utc_slots = idx.dayofweek * 1440 + idx.hour * 60 + idx.minute
    assert utc_slots[0] != utc_slots[1]


# ---------------------------------------------------------- 4. CAUSALITY


def _corrupt_minutes_after(minutes: pd.DataFrame, cut: pd.Timestamp, value=9.99) -> pd.DataFrame:
    out = minutes.copy()
    mask = out.index >= cut
    assert mask.any() and not mask.all()
    for col in PRICE_COLS:
        out.loc[mask, col] = value + (0.5 if col.endswith("ask") else 0.0)
    return out


@pytest.mark.parametrize("frac", [0.55, 0.8])
def test_no_lookahead_in_every_new_column(minutes, built, frac):
    index = built.index
    cut = index[int(len(index) * frac)]
    corrupted = _build(_corrupt_minutes_after(minutes, cut))

    head_a = built.loc[built.index < cut, NEW_COLS]
    head_b = corrupted.loc[corrupted.index < cut, NEW_COLS]
    assert len(head_a) and head_a.index.equals(head_b.index)

    nan_moved = (head_a.isna() != head_b.isna()).any()
    assert not nan_moved.any(), (
        f"LOOKAHEAD: NaN pattern changed for {nan_moved[nan_moved].index.tolist()}"
    )
    for col in NEW_COLS:
        a, b = head_a[col].to_numpy(), head_b[col].to_numpy()
        mask = ~pd.isna(a)
        assert np.array_equal(a[mask], b[mask]), (
            f"LOOKAHEAD in '{col}': values before {cut} changed when later "
            f"1-minute bars were corrupted"
        )


def test_corruption_actually_bites(minutes, built):
    """An all-NaN build must not be able to pass the lookahead test."""
    index = built.index
    cut = index[int(len(index) * 0.55)]
    corrupted = _build(_corrupt_minutes_after(minutes, cut))
    for col in ("bars_above_band_daily", "spread_pctile_trailing",
                "spread_pctile_trailing_rolling"):
        a = built.loc[built.index > cut, col].dropna()
        b = corrupted.loc[corrupted.index > cut, col]
        assert len(a) > 100, col
        assert not np.array_equal(a.to_numpy(), b.reindex(a.index).to_numpy()), col


# ---------------------------------------------------------- 5. ALIGNMENT


def _closes(frames: dict[str, pd.DataFrame], tf: str) -> pd.DatetimeIndex:
    if tf in SESSION_TFS:
        return session_close_utc(frames[tf].index, tf)
    return frames[tf].index + pd.Timedelta(tf)


@pytest.mark.parametrize("tf", htf.HIGHER_TIMEFRAMES)
def test_every_column_reflects_only_closed_bars(minutes, built, tf):
    """Exhaustive: every timestamp, every block column, the last CLOSED bar."""
    base, frames = _frames(minutes)
    block = excursion.excursion_block(htf.bb_position(frames[tf]))
    closes = _closes(frames, tf)
    pos = np.searchsorted(closes.to_numpy(), built.index.to_numpy(), side="right") - 1
    valid = pos >= 0

    for stem in excursion._BLOCK_STEMS:
        src = block[stem].to_numpy()
        want = np.full(len(built), np.nan)
        want[valid] = src[pos[valid]]
        got = built[excursion.column_name(stem, tf)].to_numpy()
        assert np.array_equal(got, want, equal_nan=True), f"{tf}/{stem}"

    covering = np.searchsorted(frames[tf].index.to_numpy(), built.index.to_numpy(),
                               side="right") - 1
    assert (covering != pos).mean() > 0.9


def test_weekly_value_first_appears_at_sundays_reopen(minutes, built):
    """The Friday 17:00 NY close has no 15-min bar; Sunday's reopen sees it."""
    base, frames = _frames(minutes)
    block = excursion.excursion_block(htf.bb_position(frames["weekly"]))
    closes = session_close_utc(frames["weekly"].index, "weekly")
    col = built["bars_since_band_cross_weekly"]
    idx = col.index

    checked = 0
    for i, close_at in enumerate(closes[:-1]):
        if np.isnan(block["bars_since_band_cross"].iloc[i]):
            continue
        assert close_at not in idx, "a weekly close should fall inside the weekend gap"
        p = int(idx.searchsorted(close_at, side="left"))
        if p == 0 or p >= len(idx):
            continue
        # The bar before the close is Friday's last 15-min bar; the bar at or
        # after it is Sunday's reopen.
        assert idx[p - 1] < close_at < idx[p]
        assert (idx[p] - idx[p - 1]) > pd.Timedelta(hours=24), "expected the weekend gap"
        assert col.iloc[p] == block["bars_since_band_cross"].iloc[i]
        if i > 0 and not np.isnan(block["bars_since_band_cross"].iloc[i - 1]):
            assert col.iloc[p - 1] == block["bars_since_band_cross"].iloc[i - 1]
        checked += 1
    assert checked > 10


# --------------------------------------------------- 6. SCALE INVARIANCE (F7)


def _scaled(minutes: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = minutes.copy()
    out[PRICE_COLS] = out[PRICE_COLS] * factor
    return out


@pytest.mark.parametrize("factor", [0.01, 137.5, 1000.0])
def test_every_new_column_is_scale_invariant(minutes, built, factor):
    """No exceptions here: every column is a band position, a count, or a rank."""
    scaled = _build(_scaled(minutes, factor))
    for col in NEW_COLS:
        a = built[col].to_numpy()
        b = scaled[col].to_numpy()
        assert not (pd.isna(a) != pd.isna(b)).any(), f"{col} NaN pattern moved at {factor}x"
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-8, equal_nan=True,
            err_msg=f"{col} is not scale-invariant at {factor}x",
        )


# ------------------------------------------------------------- 7. WARMUP


def test_warmup_is_nan_not_partial(built):
    for tf in htf.HIGHER_TIMEFRAMES:
        for stem in excursion._BLOCK_STEMS:
            col = built[excursion.column_name(stem, tf)]
            first = col.first_valid_index()
            assert first is not None, f"{tf}/{stem} never becomes valid"
            assert col.loc[:first].iloc[:-1].isna().all(), f"{tf}/{stem} partial value"
        # Occupancy needs a full window of DEFINED bars, so it cannot become
        # valid before the run counter does.
        assert (built[excursion.column_name("frac_above_band", tf)].first_valid_index()
                >= built[f"bars_above_band_{tf}"].first_valid_index())


@pytest.mark.parametrize("col", ["spread_pctile_trailing", "spread_pctile_trailing_rolling"])
def test_spread_percentile_warmup_is_nan(built, col):
    s = built[col]
    first = s.first_valid_index()
    assert first is not None
    assert s.loc[:first].iloc[:-1].isna().all()


def test_rolling_percentile_is_a_plain_trailing_rank():
    """The contaminated twin: a plain window, ranked at its last value."""
    rng = np.random.default_rng(9)
    idx = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    s = pd.Series(rng.random(300) * 1e-4, index=idx)
    out = excursion.spread_pctile_trailing_rolling(s, window=50)
    assert out.iloc[:49].isna().all()
    expected = s.iloc[100 - 49 : 101].rank(pct=True).iloc[-1]
    assert out.iloc[100] == pytest.approx(expected)
    assert out.dropna().between(0.0, 1.0).all()


def test_the_two_spread_percentiles_are_different_columns(built):
    """They must not collapse to the same thing -- the pair is the point."""
    a = built["spread_pctile_trailing"]
    b = built["spread_pctile_trailing_rolling"]
    both = a.notna() & b.notna()
    assert both.sum() > 1000
    assert not np.allclose(a[both], b[both])
    assert 0.0 < abs(float(a.corr(b))) < 0.99
