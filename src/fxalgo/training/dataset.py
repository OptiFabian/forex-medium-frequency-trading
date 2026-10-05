"""Join minute-level triple-barrier labels to the 15-minute feature frame.

BUILD ONLY. Nothing here splits, fits, scores or selects anything.

THE JOIN
--------
One row per 1-minute event. Event `t` receives the feature values of the last
15-minute bar CLOSED at or before `t`. A bar merely COVERING `t` is still in
progress and would leak. Every column goes through
`strategies.multi_timeframe.align_completed_series` -- the lag rule is never
reimplemented here.

WHY MINUTE EVENTS AT ALL
------------------------
Without step 2 below, the 15 events inside one 15-minute block would carry
IDENTICAL feature vectors and differ only in their label -- 15 copies of one
observation, which is worse than useless to a model. The live-price
recomputation is what makes a minute event distinct from its neighbours.

The rule: a feature of the form "WHERE IS PRICE relative to level X" is
recomputed from the live minute price against the FROZEN level from the last
closed 15-minute bar. A feature of the form "WHAT HAS THE MARKET BEEN DOING" is
carried forward unchanged.

The live price is `anchor_price` from the labels file -- mid open(t+1), the same
price the backtest engine fills at and the same price the label's barriers are
measured from. Taking it from the labels file rather than re-deriving it means
the feature and the label cannot drift apart.

Levels are frozen, prices are live. `bb_upper_daily` at 09:37 is the value from
the last daily bar that closed; the price compared against it is 09:37's. That
asymmetry is the whole design: the level is a decision made at the last close,
the price is now.

Each level is taken from the last bar closed on ITS OWN timeframe, not routed
through the 15-minute frame first, so a 30-minute bar closing at 09:30 is
visible at 09:31 rather than 09:45. See `frozen_levels` for the mixed-vintage
trade-off that buys.

WHAT IS NOT RECOMPUTED, AND WHY
-------------------------------
- Oscillators, volatility, moments, wick shape, spread, time, ADX: these are
  summaries of a window of history, not statements about the current price.
  Recomputing them would require rebuilding the whole indicator at 1-minute
  resolution, which is a different feature set, not this one.
- The excursion family counts CLOSED higher-timeframe bars. It is history by
  construction and must not move inside a block.
- **The gates (`gate_cell_d`, `gate_stack`, `regime_daily`,
  `regime_daily_active`) stay FROZEN at their 15-minute value.** Recomputing
  them at the minute price would make them diverge from the strategy versions
  that reproduced 984 cell-D and 897 stack entries exactly -- that verified
  reproduction is the reason those columns are trusted, and it would be traded
  away for nothing. The live-price information they would have gained is
  already available separately through the recomputed `bb_pct_b`.

COLUMN CONTRACT
---------------
The file carries only what is a feature or is needed to trade and diagnose:
the feature columns minus every price-unit one (decision F7), the anchor price,
the k=2.0 label family, the return at the time barrier, the calendar columns,
and three bookkeeping columns. The other four barrier widths and `atr_at_entry`
stay in `data/labels/eurusd_triple_barrier.parquet` and rejoin on timestamp.

SAMPLING
--------
Three events per block, at least `MIN_SPACING_MINUTES` apart, with offsets drawn
randomly PER BLOCK from a seeded generator. Fixed offsets (always minute 2, 5,
8) would inherit whatever systematic property those positions carry -- minute 0
sits on the bar boundary, minute 14 is a full bar of drift away from it.

Sampling is a modelling convenience: `sampled_flag` marks the selection and the
full joined set is written alongside, so it is reversible without a rebuild.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fxalgo.features.donchian import DONCHIAN_WINDOWS
from fxalgo.features.htf import HIGHER_TIMEFRAMES, SPREAD_TIMEFRAMES, STRETCH_TIMEFRAMES
from fxalgo.features.trend import EMA_PERIODS
from fxalgo.strategies.multi_timeframe import align_completed_series

BASE_FREQ = "15min"
ANCHOR_COL = "anchor_price"
BLOCK_KEY_COL = "atr_at_entry"

# Columns that carry PRICE UNITS and are excluded from the training file
# (decision F7 -- a pooled six-pair model cannot use them). They are still read
# during the build: the Bollinger bands, EMAs and Donchian channels below are
# exactly the frozen LEVELS the live-price recomputation measures against.
PRICE_UNIT_COLUMNS: tuple[str, ...] = (
    # raw loader / resampler output (13)
    "open_bid", "high_bid", "low_bid", "close_bid",
    "open_ask", "high_ask", "low_ask", "close_ask",
    "mid_open", "mid_high", "mid_low", "mid_close", "spread_close",
    # trend levels and price-difference oscillators (8)
    "ema_9", "ema_21", "ema_50", "ema_99", "ema_200",
    "macd", "macd_signal", "macd_hist",
    # volatility levels (4)
    "atr_14", "bb_middle", "bb_upper", "bb_lower",
    # Donchian channels (12)
    "don_high_60", "don_low_60", "don_high_180", "don_low_180",
    "don_high_240", "don_low_240", "don_high_360", "don_low_360",
    "don_high_720", "don_low_720", "don_high_1440", "don_low_1440",
    # candle geometry in price units (4)
    "body", "upper_wick", "lower_wick", "total_range",
    # spread levels (2)
    "spread_mean_15", "spread_mean_60",
)

# The label-side columns the training file keeps. Everything else -- the four
# other barrier widths and their touch columns, and `atr_at_entry` -- stays in
# data/labels/eurusd_triple_barrier.parquet and can be rejoined on timestamp.
KEPT_LABEL_COLUMNS: tuple[str, ...] = (
    ANCHOR_COL,
    "label_2.0", "bars_to_touch_2.0", "touch_price_2.0", "ambiguous_2.0",
    "ret_at_time_barrier_bps",
    "spans_gap", "gap_type", "bars_to_gap", "is_friday", "minutes_to_close",
)

AUX_COLUMNS: tuple[str, ...] = ("block_id", "event_offset", "sampled_flag")

SAMPLE_SEED = 20260814
EVENTS_PER_BLOCK = 3
MIN_SPACING_MINUTES = 3

# The 15-min band position lives under its pipeline name. `features.htf` makes
# the same identification; a second column holding identical values would be
# dead weight.
BB_POSITION_15M = "bb_pct_b"


def recomputed_columns() -> list[str]:
    """Every column rebuilt at the live minute price, in output order."""
    cols = [BB_POSITION_15M]
    cols += [f"bb_position_{tf}" for tf in HIGHER_TIMEFRAMES]
    cols += [f"dist_ema_{n}" for n in EMA_PERIODS]
    cols += [f"don_pos_{w}" for w in DONCHIAN_WINDOWS]
    cols += [f"bb_pos_spread_15m_{tf}" for tf in SPREAD_TIMEFRAMES]
    cols += ["tf_stretch_count", "tf_stretch_max"]
    return cols


def kept_feature_columns(features: pd.DataFrame) -> list[str]:
    """Feature columns that reach the training file, in frame order."""
    drop = set(PRICE_UNIT_COLUMNS)
    return [c for c in features.columns if c not in drop]


# ------------------------------------------------------------------- the join


def _align_column(col: pd.Series, target: pd.DatetimeIndex, freq: str) -> pd.Series:
    """Project one feature column onto the event index, dtype preserved.

    `align_completed_series` is dtype-agnostic but introduces NaN before the
    first closed bar, which would silently upcast a bool column to object. Bool
    and integer columns are therefore carried into pandas' NULLABLE dtypes,
    which can hold the gap honestly instead of inventing a value for it.
    """
    kind = col.dtype.kind
    if kind in "bi":
        aligned = align_completed_series(col.astype("float64"), target, freq)
        values = aligned.to_numpy()
        missing = np.isnan(values)
        if kind == "b":
            out = pd.array(values != 0, dtype="boolean")
        else:
            out = pd.array(np.nan_to_num(values).astype("int64"), dtype="Int64")
        out[missing] = pd.NA
        return pd.Series(out, index=target, name=col.name)
    return align_completed_series(col, target, freq)


def join_features(
    events: pd.DatetimeIndex, features: pd.DataFrame, *, freq: str = BASE_FREQ
) -> pd.DataFrame:
    """Every feature column at its last-CLOSED-bar value for each event."""
    if not features.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by timestamp")
    return pd.DataFrame(
        {name: _align_column(features[name], events, freq) for name in features.columns},
        index=events,
    )


def frozen_levels(
    events: pd.DatetimeIndex,
    features: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """The price LEVELS a live price is measured against, each from the last bar
    that CLOSED at or before the event -- ON ITS OWN TIMEFRAME.

    The 15-minute levels (Bollinger bands, EMAs, Donchian channels) are on the
    frame already and project at `freq="15min"`.

    The higher-timeframe BANDS are not on the frame at all -- only the band
    POSITIONS are -- so they come from each timeframe's own feature frame and
    project DIRECTLY onto the event index at that timeframe's own freq. A
    30-minute bar closing at 09:30 is therefore visible at 09:31.

    MIXED VINTAGE, ACCEPTED DELIBERATELY. Routing those bands through the
    15-minute frame first would delay them to 09:45, which is up to 15 minutes
    of information thrown away for tidiness. The cost of taking the direct path
    is that `bb_position_30min` can describe the 09:00-09:30 bar while
    `rsi_14_30min` beside it -- carried forward from the 15-minute frame -- still
    describes 08:30-09:00. Two columns about "the 30-minute bar" can disagree
    about which bar that is, for up to 15 minutes. That is a real inconsistency
    and it is accepted: fresher information is worth more than columns agreeing
    on a vintage. A model is free to learn from either.

    What is NOT negotiable is causality: a bar contributes only once it has
    CLOSED at or before the event timestamp. `align_completed_series` is the
    only thing that decides that, here as everywhere else.
    """
    out: dict[str, pd.Series] = {}
    for name in ("bb_upper", "bb_lower"):
        out[name] = align_completed_series(features[name], events, BASE_FREQ)
    for n in EMA_PERIODS:
        out[f"ema_{n}"] = align_completed_series(features[f"ema_{n}"], events, BASE_FREQ)
    for w in DONCHIAN_WINDOWS:
        for side in ("high", "low"):
            key = f"don_{side}_{w}"
            out[key] = align_completed_series(features[key], events, BASE_FREQ)
    for tf in HIGHER_TIMEFRAMES:
        frame = htf_frames[tf]
        for name in ("bb_upper", "bb_lower"):
            out[f"{name}_{tf}"] = align_completed_series(frame[name], events, tf)
    return pd.DataFrame(out, index=events)


# ------------------------------------------------------ live-price recompute


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Guarded division: a zero denominator yields NaN, never inf."""
    return numerator / denominator.where(denominator != 0, np.nan)


def recompute_live_price(
    joined: pd.DataFrame, anchor: pd.Series, levels: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the price-relative columns at the live minute price.

    Returns a copy of `joined` with the recomputed columns overwritten in
    place. Levels are never touched.
    """
    out = joined.copy()
    for col in recomputed_columns():
        if col not in out.columns:
            raise KeyError(f"joined frame is missing '{col}'")

    # Band positions -- UNBOUNDED and unclipped at every timeframe. A value
    # above 1.0 says price has crossed the upper band and by how much; that
    # magnitude is the signal, and clipping would delete it.
    out[BB_POSITION_15M] = _ratio(
        anchor - levels["bb_lower"], levels["bb_upper"] - levels["bb_lower"]
    )
    for tf in HIGHER_TIMEFRAMES:
        out[f"bb_position_{tf}"] = _ratio(
            anchor - levels[f"bb_lower_{tf}"],
            levels[f"bb_upper_{tf}"] - levels[f"bb_lower_{tf}"],
        )

    # Fractional distance to each EMA. This matches `features.trend`'s
    # definition exactly -- see the unit note in scripts/build_training_events.py.
    for n in EMA_PERIODS:
        ema = levels[f"ema_{n}"]
        out[f"dist_ema_{n}"] = _ratio(anchor - ema, ema)

    for w in DONCHIAN_WINDOWS:
        high = levels[f"don_high_{w}"]
        low = levels[f"don_low_{w}"]
        out[f"don_pos_{w}"] = _ratio(anchor - low, high - low)

    # Cross-timeframe columns are rebuilt FROM the recomputed positions, not
    # carried forward -- carrying them would leave them inconsistent with the
    # very columns they are defined in terms of.
    for tf in SPREAD_TIMEFRAMES:
        out[f"bb_pos_spread_15m_{tf}"] = out[BB_POSITION_15M] - out[f"bb_position_{tf}"]

    stretch = pd.DataFrame(
        {
            tf: (out[BB_POSITION_15M] if tf == "15m" else out[f"bb_position_{tf}"])
            .sub(0.5)
            .abs()
            for tf in STRETCH_TIMEFRAMES
        },
        index=out.index,
    )
    complete = stretch.notna().all(axis=1)
    out["tf_stretch_count"] = (stretch > 0.5).sum(axis=1).where(complete).astype("float64")
    out["tf_stretch_max"] = stretch.max(axis=1).where(complete)
    return out


# ---------------------------------------------------------------- block index


def block_id(block_key: pd.Series) -> np.ndarray:
    """Run id over consecutive events sharing one `atr_at_entry`.

    That run IS the set of minute events whose barrier width came from the same
    closed 15-minute bar, and it coincides exactly with the 15-minute grid.
    """
    changed = block_key.ne(block_key.shift())
    changed.iloc[0] = True
    return np.asarray(changed.cumsum().to_numpy(), dtype="int64")


def sample_offsets(
    rng: np.random.Generator,
    n: int,
    *,
    per_block: int = EVENTS_PER_BLOCK,
    min_gap: int = MIN_SPACING_MINUTES,
) -> np.ndarray:
    """`per_block` positions out of `n`, pairwise at least `min_gap` apart.

    Drawn UNIFORMLY over all valid combinations, not by rejection sampling:
    choosing m positions from n with a minimum gap g is a bijection onto
    choosing m positions freely from n - (m-1)(g-1), then spreading them. So
    every legal set is equally likely and no draw is ever rejected.

    Short blocks (the irregular-gap cases) take as many events as the spacing
    allows: m = min(per_block, 1 + (n-1)//g).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    m = min(per_block, 1 + (n - 1) // min_gap)
    span = n - (m - 1) * (min_gap - 1)
    picks = np.sort(rng.choice(span, size=m, replace=False))
    return picks + np.arange(m) * (min_gap - 1)


def sample_blocks(
    blocks: np.ndarray,
    *,
    seed: int = SAMPLE_SEED,
    per_block: int = EVENTS_PER_BLOCK,
    min_gap: int = MIN_SPACING_MINUTES,
) -> np.ndarray:
    """Boolean mask selecting `per_block` spaced events from each block."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(blocks), dtype=bool)
    starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
    stops = np.r_[starts[1:], len(blocks)]
    for start, stop in zip(starts, stops, strict=True):
        offsets = sample_offsets(rng, stop - start, per_block=per_block, min_gap=min_gap)
        mask[start + offsets] = True
    return mask


def event_offsets(blocks: np.ndarray) -> np.ndarray:
    """Position of each event inside its own block (0-based)."""
    starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
    stops = np.r_[starts[1:], len(blocks)]
    out = np.empty(len(blocks), dtype="int64")
    for start, stop in zip(starts, stops, strict=True):
        out[start:stop] = np.arange(stop - start)
    return out


# ---------------------------------------------------------------- entry point


def assemble(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
    *,
    sampled_mask: np.ndarray | None = None,
    blocks: np.ndarray | None = None,
    drop_price_units: bool = True,
    keep_label_columns: tuple[str, ...] = KEPT_LABEL_COLUMNS,
) -> pd.DataFrame:
    """Join, recompute and annotate one contiguous slice of events.

    `sampled_mask` / `blocks` are computed once over the WHOLE label index and
    passed in per slice, so a chunked build produces exactly the same rows as a
    single-shot one.

    Price-unit feature columns are dropped from the OUTPUT but still read on the
    way through: the bands, EMAs and channels among them are the frozen levels
    the recomputation measures against. They are excluded from the join itself,
    not filtered afterwards, so nothing is aligned that will not be kept.
    """
    overlap = set(labels.columns) & set(features.columns)
    if overlap:
        raise ValueError(f"label and feature columns collide: {sorted(overlap)}")

    events = labels.index
    anchor = labels[ANCHOR_COL]
    keep = kept_feature_columns(features) if drop_price_units else list(features.columns)
    joined = join_features(events, features[keep])
    levels = frozen_levels(events, features, htf_frames)
    out = recompute_live_price(joined, anchor, levels)

    if blocks is None:
        blocks = block_id(labels[BLOCK_KEY_COL])
    out.insert(0, "block_id", blocks)
    out.insert(1, "event_offset", event_offsets(blocks))
    out.insert(2, "sampled_flag",
               np.zeros(len(out), dtype=bool) if sampled_mask is None else sampled_mask)
    missing = [c for c in keep_label_columns if c not in labels.columns]
    if missing:
        raise KeyError(f"labels frame is missing {missing}")
    for name in keep_label_columns:
        out[name] = labels[name]
    return out
