"""Higher-timeframe features projected onto the 15-minute index.

Everything here answers one question: **what did the coarser timeframes look
like at the moment this 15-minute bar closed?** Three families:

1. PER-TIMEFRAME BLOCKS (`bb_position`, `bb_width_atr`, `rsi_14`, `atr_bps`)
   computed on 30min / 1h / 2h / 4h / daily / weekly bars and re-stamped onto
   the 15-min timeline.
2. CROSS-TIMEFRAME columns -- the 15-min-vs-higher band-position spreads and
   the "how many timeframes are stretched at once" summaries. The project's
   best result (cell D) came from COMBINING timescales, so the relationship is
   encoded directly rather than left for a tree to reconstruct.
3. WORKING-CONFIG columns (decision F4) and the scale-free siblings of the
   price-unit columns the feature inventory flagged (decision F7).

CAUSALITY
---------
A higher-TF bar labeled `L` covers `[L, L + span)` and is only CLOSED at
`L + span`. Its value must be invisible on the 15-min timeline before that
instant. This module never implements that lag itself: every projection goes
through `strategies.multi_timeframe.align_completed_series`, the one primitive
that owns the rule (`align_completed_signal` is used ONLY for the {-1,0,+1}
regime column -- it casts to int8 and would truncate a float feature).

The derived quantity is computed on the higher timeframe FIRST and the
resulting series is then aligned, so each 15-min bar sees a value that was
fully determined by bars closed at or before its own timestamp.

Session anchoring is the resampler's business: `daily` / `weekly` bars sit on
the 17:00 America/New_York grid and their close instants come from
`resample.session_close_utc`; 2h / 4h are session-anchored sub-bars whose close
is exactly `label + 2h / 4h` (verified against `session_close_utc` on every bar
of the five-year window); 30min / 1h are fixed CLOCK bars (`resample_merged`)
that close at `label + freq`. All four of those cases are what
`align_completed_series` already does with a plain `freq` string.

WARMUP
------
NaN, never a partial window -- the module-wide policy. A higher-TF column is
NaN on every 15-min bar before that timeframe's first COMPLETED bar with a
defined value, so `bb_position_weekly` needs 20 completed weekly bars (~5
months) before it exists. The cross-timeframe summaries (`tf_stretch_count`,
`tf_stretch_max`) require all seven inputs and are therefore NaN until the
slowest one is defined.

The two gate columns are BOOLEAN, not float: they mirror an entry FSM whose
warmup branch forces "flat", so a bar with any NaN input is `False` (the entry
is not permitted), not NaN.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fxalgo.backtest.costs import DEFAULT_COMMISSION, CommissionModel
from fxalgo.features.donchian import DONCHIAN_WINDOWS
from fxalgo.strategies.bollinger_reversion import BollingerReversion
from fxalgo.strategies.multi_timeframe import (
    align_completed_series,
    align_completed_signal,
)

# The six higher timeframes, coarsest last. Each name is BOTH the key into the
# supplied frame mapping AND the `freq` handed to the causality primitive.
HIGHER_TIMEFRAMES: tuple[str, ...] = ("30min", "1h", "2h", "4h", "daily", "weekly")
# The band-position spread is defined against these five only (no 30min).
SPREAD_TIMEFRAMES: tuple[str, ...] = ("1h", "2h", "4h", "daily", "weekly")
# The stretch summaries span the base timeframe plus all six higher ones.
STRETCH_TIMEFRAMES: tuple[str, ...] = ("15m", *HIGHER_TIMEFRAMES)
PER_TF_STEMS: tuple[str, ...] = ("bb_position", "bb_width_atr", "rsi_14", "atr_bps")

REGIME_TF = "daily"
REGIME_COL = "regime_daily"
REGIME_ACTIVE_COL = "regime_daily_active"

# The 15-min band position already exists on the frame under its pipeline name:
# volatility.add_features writes bb_pct_b = (mid_close - bb_lower) / band_width,
# which IS the bb_position definition (unbounded, zero-width guarded).
BB_POSITION_15M = "bb_pct_b"

# Working-config constants, read from the strategy code and the scripts that
# established the two configs. See the module-level notes in
# scripts/build_htf_features.py for provenance.
RSI_LOWER = 30.0
RSI_UPPER = 70.0
NOTIONAL = 100_000.0
COST_MULTIPLE = 2.0

BPS = 1e4


def feature_columns() -> list[str]:
    """The exact ordered list of columns `add_features` appends."""
    cols = [f"{stem}_{tf}" for tf in HIGHER_TIMEFRAMES for stem in PER_TF_STEMS]
    cols += [f"bb_pos_spread_15m_{tf}" for tf in SPREAD_TIMEFRAMES]
    cols += ["tf_stretch_count", "tf_stretch_max"]
    cols += [REGIME_COL, REGIME_ACTIVE_COL]
    cols += ["gate_cell_d", "gate_stack"]
    cols += [f"don_pos_{w}" for w in DONCHIAN_WINDOWS]
    cols += ["atr_bps_15m"]
    return cols


# ---------------------------------------------------------------- primitives


def _guard(denominator: pd.Series) -> pd.Series:
    """Zero -> NaN, so a degenerate denominator yields NaN rather than inf."""
    return denominator.where(denominator != 0, np.nan)


def bb_position(frame: pd.DataFrame) -> pd.Series:
    """(price - bb_lower) / (bb_upper - bb_lower). UNBOUNDED and unclipped.

    > 1 means price is above the upper band, < 0 below the lower band, and the
    magnitude beyond the band is the signal -- clipping would destroy exactly
    the information the column exists to carry. Identical in definition to the
    pipeline's `bb_pct_b`; a test asserts they agree bit-for-bit.
    """
    width = frame["bb_upper"] - frame["bb_lower"]
    return (frame["mid_close"] - frame["bb_lower"]) / _guard(width)


def atr_bps(frame: pd.DataFrame) -> pd.Series:
    """ATR expressed in basis points of price -- scale-free (decision F7)."""
    price = frame["mid_close"]
    return frame["atr_14"] / price.where(price > 0, np.nan) * BPS


def htf_block(frame: pd.DataFrame) -> pd.DataFrame:
    """The four per-timeframe features, on that timeframe's OWN index.

    `frame` is a timeframe's feature frame as written by
    `features.build.apply_pipeline`, so its Bollinger bands are the project
    convention (20-period, 2.0 std, on mid_close) computed on that timeframe's
    own bars, and `atr_14` is Wilder's ATR on the same bars.
    """
    required = ("mid_close", "bb_upper", "bb_lower", "atr_14", "rsi_14")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"htf_block requires columns {missing}")
    atr = frame["atr_14"]
    return pd.DataFrame(
        {
            "bb_position": bb_position(frame),
            "bb_width_atr": (frame["bb_upper"] - frame["bb_lower"]) / _guard(atr),
            "rsi_14": frame["rsi_14"],
            "atr_bps": atr_bps(frame),
        },
        index=frame.index,
    )


def align_block(
    block: pd.DataFrame, target_index: pd.DatetimeIndex, tf: str
) -> pd.DataFrame:
    """Project a per-timeframe block onto `target_index`, suffixing each column.

    Every column goes through `align_completed_series` (the float-safe
    projection); nothing is filled, so pre-first-completed-bar rows stay NaN.
    """
    return pd.DataFrame(
        {
            f"{col}_{tf}": align_completed_series(block[col], target_index, tf)
            for col in block.columns
        },
        index=target_index,
    )


# ------------------------------------------------------------- cost gate math


def commission_per_leg(
    price: np.ndarray, notional: float, commission: CommissionModel | None
) -> np.ndarray:
    """Vectorized twin of `CommissionModel.commission_for`.

    Kept in lockstep with the scalar model by
    `test_commission_matches_scalar_model`, which compares the two elementwise.
    """
    if commission is None:
        return np.zeros_like(price, dtype="float64")
    trade_value = price * notional
    raw = np.maximum(commission.min_per_order, commission.rate * trade_value)
    return np.minimum(commission.cap_pct * trade_value, raw)


def passes_cost_filter_vec(
    price: np.ndarray,
    middle: np.ndarray,
    spread: np.ndarray,
    *,
    notional: float,
    commission: CommissionModel | None,
    cost_multiple: float,
) -> np.ndarray:
    """Vectorized twin of `strategies.cost_filter.passes_cost_filter`.

    NOT scale-free, and deliberately so: the round-trip cost contains the
    commission model's absolute per-order minimum (2.00), which does not move when prices are
    rescaled. See the F7 note in `scripts/build_htf_features.py`.
    """
    profit_potential = np.abs(middle - price) * notional
    round_trip = spread * notional + 2.0 * commission_per_leg(price, notional, commission)
    return np.asarray(profit_potential >= cost_multiple * round_trip, dtype=bool)


# ------------------------------------------------------------------- the gates


def _touch_side(frame: pd.DataFrame) -> np.ndarray:
    """+1 at a lower-band touch, -1 at an upper-band touch, 0 otherwise.

    Mirrors the entry branches of `BollingerRsiConfluence.generate_signals`:
    the lower-band branch is tested first, and any NaN comparison is False, so
    a warmup bar is 0 (no candidate) exactly as the FSM forces flat.
    """
    price = frame["mid_close"].to_numpy(dtype="float64")
    upper = frame["bb_upper"].to_numpy(dtype="float64")
    lower = frame["bb_lower"].to_numpy(dtype="float64")
    side = np.zeros(len(frame), dtype="int8")
    with np.errstate(invalid="ignore"):
        side[price <= lower] = 1
        side[(side == 0) & (price >= upper)] = -1
    return side


def _rsi_confluence(frame: pd.DataFrame, side: np.ndarray,
                    rsi_lower: float, rsi_upper: float) -> np.ndarray:
    """RSI confirms the touch: long needs rsi <= 30, short needs rsi >= 70."""
    rsi = frame["rsi_14"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        short_ok = np.where(side == -1, rsi >= rsi_upper, False)
        confirmed = np.where(side == 1, rsi <= rsi_lower, short_ok)
    return np.asarray(confirmed, dtype=bool)


def gate_stack_column(
    frame: pd.DataFrame,
    *,
    spread_gate_max: float,
    rsi_lower: float = RSI_LOWER,
    rsi_upper: float = RSI_UPPER,
    notional: float = NOTIONAL,
    commission: CommissionModel | None = DEFAULT_COMMISSION,
    cost_multiple: float = COST_MULTIPLE,
) -> np.ndarray:
    """The `gate_stack` entry condition, as a function of the spread ceiling.

    Extracted so the ceiling can be VARIED without reimplementing the gate. The
    ceiling is the one part of this config that is derived from data rather than
    chosen a priori, so a walk-forward has to re-derive it per training fold --
    and the only safe way to do that is to call the same function with a
    different number, never to write the condition out a second time.
    """
    side = _touch_side(frame)
    rsi_ok = _rsi_confluence(frame, side, rsi_lower, rsi_upper)
    price = frame["mid_close"].to_numpy(dtype="float64")
    middle = frame["bb_middle"].to_numpy(dtype="float64")
    spread = frame["spread_close"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        cost_ok = passes_cost_filter_vec(
            price, middle, spread,
            notional=notional, commission=commission, cost_multiple=cost_multiple,
        )
        spread_ok = spread < spread_gate_max
    # A NaN anywhere in the FSM's inputs is a warmup bar: it forces flat, so no
    # entry is permitted. NaN propagates to False through the comparisons above
    # EXCEPT via `cost_ok`, where NaN >= NaN is already False -- assert the
    # intent explicitly rather than relying on it.
    defined = ~(np.isnan(price) | np.isnan(middle) | np.isnan(spread))
    return np.asarray((side != 0) & rsi_ok & cost_ok & spread_ok & defined, dtype=bool)


# ---------------------------------------------------------------- entry point


def add_features(
    base: pd.DataFrame,
    htf_frames: Mapping[str, pd.DataFrame],
    *,
    spread_gate_max: float,
    rsi_lower: float = RSI_LOWER,
    rsi_upper: float = RSI_UPPER,
    notional: float = NOTIONAL,
    commission: CommissionModel | None = DEFAULT_COMMISSION,
    cost_multiple: float = COST_MULTIPLE,
) -> pd.DataFrame:
    """Append every higher-timeframe / cross-timeframe / F4 / F7 column.

    Parameters
    ----------
    base:
        The 15-minute feature frame to extend (the 98-column moments frame).
    htf_frames:
        Mapping of timeframe name -> that timeframe's feature frame, covering
        every entry of `HIGHER_TIMEFRAMES`.
    spread_gate_max:
        The absolute `spread_close` ceiling of the `gate_stack` config. A
        CONSTANT of the config, supplied by the caller (the establishing script
        set it to the full-sample median 15-min spread); passing it in keeps
        this function a pure per-bar computation.
    """
    missing_tf = [tf for tf in HIGHER_TIMEFRAMES if tf not in htf_frames]
    if missing_tf:
        raise KeyError(f"htf_frames is missing timeframes {missing_tf}")
    if BB_POSITION_15M not in base.columns:
        raise KeyError(f"base frame must carry '{BB_POSITION_15M}'")

    out = base.copy()
    index = out.index

    # 1. Per-timeframe blocks, computed on their own bars then projected.
    for tf in HIGHER_TIMEFRAMES:
        block = align_block(htf_block(htf_frames[tf]), index, tf)
        for col in block.columns:
            out[col] = block[col]

    # 2. Cross-timeframe relationships.
    pos_15m = bb_position(out)
    for tf in SPREAD_TIMEFRAMES:
        out[f"bb_pos_spread_15m_{tf}"] = pos_15m - out[f"bb_position_{tf}"]

    stretch = pd.DataFrame(
        {
            tf: (pos_15m if tf == "15m" else out[f"bb_position_{tf}"]).sub(0.5).abs()
            for tf in STRETCH_TIMEFRAMES
        },
        index=index,
    )
    # All seven must be defined: a partial count would understate the stretch
    # purely because a slow timeframe has not warmed up yet.
    complete = stretch.notna().all(axis=1)
    out["tf_stretch_count"] = (stretch > 0.5).sum(axis=1).where(complete).astype("float64")
    out["tf_stretch_max"] = stretch.max(axis=1).where(complete)

    # 3. Decision F4 -- the daily regime, persisted at last.
    daily_shadow = BollingerReversion().generate_signals(htf_frames[REGIME_TF])
    out[REGIME_COL] = align_completed_signal(daily_shadow, index, REGIME_TF)
    out[REGIME_ACTIVE_COL] = out[REGIME_COL].to_numpy() != 0

    # 4. Decision F4 -- the two working configs as per-bar entry conditions.
    side = _touch_side(out)
    rsi_ok = _rsi_confluence(out, side, rsi_lower, rsi_upper)
    out["gate_cell_d"] = (side != 0) & rsi_ok & out[REGIME_ACTIVE_COL].to_numpy()

    out["gate_stack"] = gate_stack_column(
        out,
        spread_gate_max=spread_gate_max,
        rsi_lower=rsi_lower,
        rsi_upper=rsi_upper,
        notional=notional,
        commission=commission,
        cost_multiple=cost_multiple,
    )

    # 5. Decision F7 -- scale-free siblings. The price-unit originals stay.
    for w in DONCHIAN_WINDOWS:
        high = out[f"don_high_{w}"]
        low = out[f"don_low_{w}"]
        out[f"don_pos_{w}"] = (out["mid_close"] - low) / _guard(high - low)
    out["atr_bps_15m"] = atr_bps(out)

    return out
