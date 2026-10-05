"""Excursion history on the higher-timeframe band positions, plus a causal
trailing spread percentile.

WHY
---
The 140-column frame carries `bb_position` at seven timeframes, but each one is
a SNAPSHOT. `bb_position_daily = 1.05` says price is outside the band NOW; it
says nothing about whether that started this bar or three days ago, how deep the
excursion went, or whether price has been hovering at the band and ducking in
and out. This family carries that history.

Four descriptions of the same excursion, none of which substitutes for another:

  bars_above_band / bars_below_band   DURATION of the current unbroken run.
  frac_above_band / frac_below_band   OCCUPANCY of the last 20 closed bars.
      The run counter resets to zero on a single bar back inside, so a path of
      6 outside / 1 inside / 6 outside reads as 1 on the run counter while
      being 13 mostly-outside bars. The window is 20 to match the Bollinger
      lookback: it asks how much of the window the bands are computed FROM was
      spent outside them.
  bars_since_band_cross               RECENCY -- a fresh crossing (1) versus a
      stale one (40), regardless of where price sits now.
  max_stretch_current_excursion       DEPTH. A shallow six-bar excursion and a
      violent six-bar excursion are identical on the run counter.

CAUSALITY -- and the unit trap this family exists to avoid
-----------------------------------------------------------
Every counter is measured in the HIGHER TIMEFRAME'S OWN CLOSED BARS and only
then projected onto the 15-min index with `align_completed_series`. Computing
the same counter on an already-aligned 15-min series would count 15-MINUTE ROWS
instead of higher-TF bars and silently return values 16x too large at 4h and
~470x too large at weekly. `test_counter_counts_higher_tf_bars_not_15min_rows`
constructs exactly that case and pins the correct answer.

`align_completed_series` is used throughout -- never `align_completed_signal`,
which casts to int8 and would truncate the fractions to 0.

MECHANICAL CONSEQUENCE, by design: between higher-TF bar closes every column
here is CONSTANT across all intervening 15-min rows. `bars_above_band_4h` holds
one value for sixteen consecutive 15-min bars; the weekly columns hold one value
for a whole trading week. That is correct -- the current higher-TF bar's state
is unknowable until it closes -- but it makes these columns STEP FUNCTIONS, not
smooth series, and their effective sample size is the number of higher-TF bars,
not the number of rows.

WARMUP
------
NaN, never partial. A bar with an undefined `bb_position` resets all run state
and emits NaN across the family. `bars_since_band_cross` is NaN until the first
genuine inside -> outside transition has been observed -- not 0, which would be
indistinguishable from "crossed just now", and not a sentinel.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fxalgo.features.htf import HIGHER_TIMEFRAMES, bb_position
from fxalgo.strategies.multi_timeframe import align_completed_series

# The occupancy window matches the Bollinger lookback (20 periods).
FRACTION_WINDOW = 20

# Run-length cap. NOT a round number chosen a priori: it comes from the observed
# distribution. Pooled over all 12 run columns on EURUSD's five-year window, in
# units of INDEPENDENT label windows (N_eff = 13,619 at k=2.0, ~9.04 15-min rows
# each), the values are carried by roughly:
#   1: 4,796w   2: 2,206w   3: 1,079w   4: 516w   5: 226w
#   6: 68w      7: 23w      8: 8w       9: 2w     10: 0w
# Every value at or below 6 is carried by >=68 independent windows; everything
# above it is 0-23 windows per level, which is noise a tree would happily split
# on. The longest run observed at ANY timeframe is 10, so a cap of 10 would not
# be a cap at all.
RUN_CAP = 6

# Trailing weeks of the SAME minute-of-week used for the spread percentile.
# See `spread_pctile_trailing` for why the baseline is minute-of-week rather
# than a plain rolling window, and why 52.
SPREAD_PCTILE_WEEKS = 52

# Plain rolling window for the second spread percentile, matching
# `features.moments.PERCENTILE_WINDOW_BARS` (250 sessions x 94.84 bars). Both
# variants are shipped deliberately: the rolling one carries ~20% time-of-day
# variance that the minute-of-week one removes, and keeping both makes that
# contamination measurable in feature importance rather than only in a one-off
# diagnostic. See `spread_pctile_trailing_rolling`.
ROLLING_PCTILE_WINDOW = 23_710

_BLOCK_STEMS: tuple[str, ...] = (
    "bars_above_band",
    "bars_below_band",
    "frac_above_band",
    "frac_below_band",
    "bars_since_band_cross",
    "max_stretch_current_excursion",
)
SPREAD_PCTILE_COL = "spread_pctile_trailing"
SPREAD_PCTILE_ROLLING_COL = "spread_pctile_trailing_rolling"


def column_name(stem: str, tf: str) -> str:
    """Final column name for a block stem at timeframe `tf`.

    The occupancy columns carry the window length after the timeframe
    (`frac_above_band_4h_20`); everything else is plain `<stem>_<tf>`.
    """
    if stem.startswith("frac_"):
        return f"{stem}_{tf}_{FRACTION_WINDOW}"
    return f"{stem}_{tf}"


def feature_columns() -> list[str]:
    """The exact ordered list of columns `add_features` appends."""
    cols: list[str] = []
    for group in (
        ("bars_above_band", "bars_below_band"),
        ("frac_above_band", "frac_below_band"),
        ("bars_since_band_cross",),
        ("max_stretch_current_excursion",),
    ):
        for tf in HIGHER_TIMEFRAMES:
            cols.extend(column_name(stem, tf) for stem in group)
    cols.extend([SPREAD_PCTILE_COL, SPREAD_PCTILE_ROLLING_COL])
    return cols


# ------------------------------------------------------------------ the block


def excursion_block(
    pos: pd.Series,
    *,
    run_cap: int = RUN_CAP,
    fraction_window: int = FRACTION_WINDOW,
) -> pd.DataFrame:
    """The six excursion series for ONE timeframe, on that timeframe's own bars.

    `pos` is that timeframe's `bb_position` (unclipped): > 1.0 is above the
    upper band, < 0.0 below the lower one, matching the frame's convention.

    A single forward pass carries all the path state. It is a plain loop rather
    than a vectorised trick because four interacting state machines (two run
    counters, a recency counter and an excursion-depth accumulator) are far
    easier to verify by reading than to reconstruct from cumsum arithmetic, and
    the longest series here is 62k rows.
    """
    if run_cap < 1:
        raise ValueError(f"run_cap must be >= 1, got {run_cap}")
    v = pos.to_numpy(dtype="float64")
    n = len(v)
    above = np.full(n, np.nan)
    below = np.full(n, np.nan)
    since = np.full(n, np.nan)
    depth = np.full(n, np.nan)

    run_above = 0
    run_below = 0
    excursion_side = 0        # side of the current unbroken excursion
    excursion_max = 0.0
    armed = False             # has an inside -> outside crossing been seen yet?
    counter = 0
    prev_state: int | None = None

    for i in range(n):
        x = v[i]
        if np.isnan(x):
            # Undefined bar: every column is NaN and ALL path state resets, so
            # a run can never be stitched across a hole. In practice this is
            # only the leading Bollinger warmup.
            run_above = run_below = 0
            excursion_side = 0
            excursion_max = 0.0
            armed = False
            counter = 0
            prev_state = None
            continue

        state = 1 if x > 1.0 else (-1 if x < 0.0 else 0)
        stretch = abs(x - 0.5)

        run_above = run_above + 1 if state == 1 else 0
        run_below = run_below + 1 if state == -1 else 0
        above[i] = min(run_above, run_cap)
        below[i] = min(run_below, run_cap)

        if state == 0:
            excursion_side = 0
            excursion_max = 0.0
            # Zero, not NaN: "no excursion in progress" is a real, meaningful
            # depth, and there is no ambiguity -- an excursion always has
            # |pos - 0.5| > 0.5, so 0.0 can only mean "inside". Using NaN would
            # collide with the warmup NaN and destroy that distinction.
            depth[i] = 0.0
        else:
            if state != excursion_side:
                excursion_side = state
                excursion_max = stretch
            else:
                excursion_max = max(excursion_max, stretch)
            depth[i] = excursion_max

        if prev_state == 0 and state != 0:
            armed = True
            counter = 1
        elif armed:
            counter += 1
        if armed:
            since[i] = float(counter)

        prev_state = state

    # NaN, not False, where the bar is undefined -- so a window containing one
    # falls below min_periods instead of counting it as "not outside".
    outside_above = pd.Series(np.where(np.isnan(v), np.nan, (v > 1.0).astype(float)),
                              index=pos.index)
    outside_below = pd.Series(np.where(np.isnan(v), np.nan, (v < 0.0).astype(float)),
                              index=pos.index)

    def occupancy(flag: pd.Series) -> pd.Series:
        # min_periods == window, so a window holding any undefined bar is NaN
        # rather than an average over a partial window.
        return flag.rolling(window=fraction_window, min_periods=fraction_window).mean()

    return pd.DataFrame(
        {
            "bars_above_band": above,
            "bars_below_band": below,
            "frac_above_band": occupancy(outside_above),
            "frac_below_band": occupancy(outside_below),
            "bars_since_band_cross": since,
            "max_stretch_current_excursion": depth,
        },
        index=pos.index,
    )


# ------------------------------------------------------- trailing spread rank


def minute_of_week(index: pd.DatetimeIndex, tz: str = "America/New_York") -> pd.Series:
    """Minute-of-week slot in NEW YORK wall time.

    NY, not UTC, and for the same reason every session boundary in this project
    is NY-anchored: the FX day rolls at 17:00 NY, so a UTC slot is "the London
    open" for eight months of the year and something else for four. Measured on
    EURUSD's five-year window, NY slots explain 42.5% of raw spread variance
    against UTC slots' 32.9%, and the NY-anchored rank leaves 0.05% residual
    time-of-day structure against the UTC-anchored version's 2.4%.

    The 15-minute grid is unaffected by the relabelling: US offsets are whole
    hours, so bars stay on :00/:15/:30/:45 in NY wall time too.
    """
    local = index.tz_convert(tz)
    return pd.Series(
        local.dayofweek * 1440 + local.hour * 60 + local.minute, index=index
    )


def spread_pctile_trailing(
    spread: pd.Series, *, weeks: int = SPREAD_PCTILE_WEEKS, tz: str = "America/New_York"
) -> pd.Series:
    """Rank of the current spread among the last `weeks` SAME-minute-of-week values.

    In [0, 1], strictly causal: `Rolling.rank` ranks the LAST value of each
    window, so the window ends at bar t. This is the same primitive
    `features.moments` uses for its percentile columns -- a full-sample rank
    would be the very lookahead this column exists to remove.

    WHY MINUTE-OF-WEEK RATHER THAN A PLAIN ROLLING WINDOW. The spread is
    strongly diurnal (the 17:00 NY rollover is the widest minute of the day by
    25-46x). A plain rolling window contains every time-of-day in proportion, so
    the rollover minute ranks near 1.0 essentially every day and the column
    mostly restates the clock: measured on EURUSD, time-of-day explains 17.5%
    of a 23,710-bar rolling rank's variance versus 0.05% of this one. Note that
    the usual check -- correlation against `hour_sin` / `hour_cos` / the session
    flags -- does NOT catch this: the worst single correlation for the rolling
    version is only 0.28, because the diurnal spread pattern is a spike rather
    than a sinusoid. The variance share is the honest measure.

    WHY 52 WEEKS. It is one full year of same-slot history, so the rank is not
    itself a seasonal artifact, and it gives 1.9% rank resolution. It costs
    almost nothing: its first valid timestamp is 2022-05-26, only six days later
    than `ret_skew_96_pctile` at 2022-05-19, which already binds the frame's
    fully-defined region. A 26-week window buys back those six days at half the
    resolution (the two correlate 0.955).
    """
    if weeks < 2:
        raise ValueError(f"weeks must be >= 2, got {weeks}")
    if not isinstance(spread.index, pd.DatetimeIndex):
        raise TypeError("spread_pctile_trailing requires a DatetimeIndex")
    if spread.index.tz is None:
        raise ValueError("spread_pctile_trailing requires a tz-aware index")

    out = pd.Series(np.nan, index=spread.index, dtype="float64")
    for _slot, group in spread.groupby(minute_of_week(spread.index, tz)):
        out.loc[group.index] = group.rolling(window=weeks, min_periods=weeks).rank(
            pct=True
        )
    return out


def spread_pctile_trailing_rolling(
    spread: pd.Series, *, window: int = ROLLING_PCTILE_WINDOW
) -> pd.Series:
    """Rank of the current spread within a plain trailing window of `window` bars.

    Shipped ALONGSIDE the minute-of-week version, not instead of it, and both
    are causal (`Rolling.rank` ranks the window's LAST value). The pair exists
    so the difference between them is measurable downstream.

    This one is the CONTAMINATED variant, deliberately kept: a plain window
    contains every time-of-day in proportion, so the 17:00 NY rollover -- the
    widest minute of the day by 25-46x -- ranks near 1.0 essentially every
    session. Measured on EURUSD, minute-of-day explains 19.97% of this column's
    variance against 0.05% of `spread_pctile_trailing`'s. If a model later
    prefers this column to its clean twin, that preference IS the measurement of
    how much of the apparent signal was the clock; a one-off diagnostic could
    never show that.

    The window matches `features.moments.PERCENTILE_WINDOW_BARS` so its warmup
    coincides with the frame's existing percentile columns.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    return spread.rolling(window=window, min_periods=window).rank(pct=True)


# ---------------------------------------------------------------- entry point


def add_features(
    base: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
    *,
    run_cap: int = RUN_CAP,
    fraction_window: int = FRACTION_WINDOW,
    spread_pctile_weeks: int = SPREAD_PCTILE_WEEKS,
    spread_pctile_rolling_window: int = ROLLING_PCTILE_WINDOW,
) -> pd.DataFrame:
    """Append the 36 excursion columns and the two trailing spread percentiles.

    `htf_frames` maps timeframe name -> that timeframe's feature frame, covering
    every entry of `HIGHER_TIMEFRAMES`. Each block is computed on the higher
    timeframe's own bars and projected with `align_completed_series`.
    """
    missing = [tf for tf in HIGHER_TIMEFRAMES if tf not in htf_frames]
    if missing:
        raise KeyError(f"htf_frames is missing timeframes {missing}")
    if "spread_close" not in base.columns:
        raise KeyError("base frame must carry 'spread_close'")

    out = base.copy()
    index = out.index

    blocks = {
        tf: excursion_block(
            bb_position(htf_frames[tf]),
            run_cap=run_cap,
            fraction_window=fraction_window,
        )
        for tf in HIGHER_TIMEFRAMES
    }
    for stem in _BLOCK_STEMS:
        for tf in HIGHER_TIMEFRAMES:
            out[column_name(stem, tf)] = align_completed_series(
                blocks[tf][stem], index, tf
            )

    out[SPREAD_PCTILE_COL] = spread_pctile_trailing(
        out["spread_close"], weeks=spread_pctile_weeks
    )
    out[SPREAD_PCTILE_ROLLING_COL] = spread_pctile_trailing_rolling(
        out["spread_close"], window=spread_pctile_rolling_window
    )
    return out[list(base.columns) + feature_columns()]
