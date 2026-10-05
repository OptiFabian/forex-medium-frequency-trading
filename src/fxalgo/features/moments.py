"""Distributional moments of log returns: skew, kurtosis and their robust twins.

Everything else in the feature frame that measures dispersion is SECOND moment
(ATR, ret_std, Bollinger width). This family adds the third and fourth moments
plus quantile-based estimators of the same shape, so a model can distinguish
"volatile" from "skewed" or "fat-tailed".

Input series
------------
All features are computed on the LOG RETURN of ``mid_close``::

    r_t = log(mid_close_t) - log(mid_close_{t-1})

NOT on price levels: the skewness of raw closes is largely a restatement of
trend direction and would be collinear with the existing EMA/MACD columns.
Log returns are also what makes the family scale-free (see below).

Gap masking (``mask_gap_returns``, default True)
------------------------------------------------
A return that spans a market gap is not a market move -- it is the price
difference across a weekend, a holiday, or the daily 17:00 NY reset. A single
weekend gap inside a 96-bar window dominates the fourth moment completely.

A transition is gap-spanning when the index step exceeds the series' modal bar
interval. That is exactly the set of discontinuities `fxalgo.data.quality`
classifies (as ``weekend`` / ``daily_reset`` / ``active``); ALL kinds are
excluded here, because none of them is a tradeable price move. On the 15-min
frames this masks ~1.05% of returns (~1,297 per pair over the 5-year window:
~1,030 daily-reset holes and ~243 weekends).

Masked returns are set to NaN and simply DROPPED from each rolling window --
never filled, never interpolated. A window that contains a weekend therefore
computes on 95 observations instead of 96. Set ``mask_gap_returns=False`` to
measure the effect of the masking itself.

WINDOW LENGTHS ARE IN BARS, NOT CLOCK TIME. "96 bars" is approximately 24
hours only *within a continuous session*; a 96-bar window that straddles a
weekend spans considerably more wall-clock time (roughly 3 calendar days).
The bar count is the honest unit here -- calendar-anchored windows would have
a varying number of observations.

Estimators (stated so the numbers are reproducible)
---------------------------------------------------
- ``ret_skew_*``  : bias-corrected Fisher-Pearson standardized moment
  coefficient G1 (pandas ``rolling.skew``; equals ``scipy.stats.skew(bias=False)``).
  Symmetric distribution -> 0.
- ``ret_kurt_*``  : bias-corrected EXCESS kurtosis G2 (pandas ``rolling.kurt``;
  equals ``scipy.stats.kurtosis(fisher=True, bias=False)``). **Normal -> 0.**
- ``ret_bowley_skew_*`` : Bowley (quartile) skewness,
  ``(Q3 + Q1 - 2*Q2) / (Q3 - Q1)``, bounded in [-1, 1]. Symmetric -> 0.
- ``ret_moors_kurt_*``  : Moors octile kurtosis,
  ``((E7 - E5) + (E3 - E1)) / (E6 - E2)`` where ``Ei`` is the i/8 quantile.
  **Normal -> 1.2331** (NOT 0); this is a raw, not excess, measure.

Both quantile estimators divide by the IQR (``E6 - E2``). In a flat window --
one where at least half the returns are identical, which happens in very thin
liquidity -- that denominator collapses. Windows with an IQR at or below
``IQR_EPS`` yield **NaN, never inf**; the diagnostics report how often.

The moment estimators need the same protection for a different reason: on a
window of *constant* returns pandas returns a finite ``kurt`` of -3.0 (the
excess offset with a zero numerator) rather than NaN, which would read as a
strong thin-tail signal where the statistic is simply undefined. Windows with
zero return variance therefore yield NaN for skew and kurtosis too.

Divergence features
-------------------
Both divergences are the difference of TRAILING Z-SCORES, each computed over
the same 250-session window used by the percentile features::

    z(x)_t = (x_t - mean(x over W ending at t)) / std(x over W ending at t)
    skew_robust_divergence_96 = z(ret_skew_96) - z(ret_bowley_skew_96)
    kurt_robust_divergence_96 = z(ret_kurt_96) - z(ret_moors_kurt_96)

A RAW difference does not work for either pair, for the same reason. The two
estimators in each pair have wildly different dispersions -- moment skew has
SD ~1.29 against Bowley's ~0.14, and excess kurtosis is 0-centred and unbounded
above while Moors sits near 1.2331 -- so a raw subtraction is dominated by the
moment term and reproduces it almost exactly. The skew divergence was first
built as a raw difference and measured **r = +0.994 against ret_skew_96**: a
copy, not a divergence. Standardizing both terms first is what makes the
column measure disagreement rather than magnitude.

This is causal and unitless, and puts both estimators on a common
regime-relative scale. The cost is that BOTH divergence columns inherit the
long (~250-session) warmup -- the diagnostics report first-valid timestamps.

There is deliberately no 24-bar divergence: the fast window exists to react
quickly, and a 250-session standardization on top of it would defeat that.

Semi-deviation
--------------
Lower/upper partial second moments about zero, in BASIS POINTS::

    ret_semidev_down = 1e4 * sqrt( mean( min(r, 0)^2 ) )
    ret_semidev_up   = 1e4 * sqrt( mean( max(r, 0)^2 ) )

The mean is over ALL valid observations in the window (not only the negative
ones), which is the standard lower-partial-moment convention with target 0.
``ret_semidev_ratio = down / up`` is unitless; a zero upside deviation yields
NaN rather than inf.

Percentile ranks
----------------
``ret_skew_96_pctile`` / ``ret_kurt_96_pctile`` give the rank of the current
value within a TRAILING window of the same feature, in [0, 1]. These are the
columns expected to travel across eras and pairs, since a raw kurtosis of 4
means different things in calm and stressed regimes.

The window is 250 trading sessions. At 15-min bars that is **23,710 bars**,
measured rather than assumed: all six pairs hold 123,106-123,110 bars across
1,298 sessions over the 5-year window = **94.84 bars per session**, not the 96
a naive 24h/15min calculation gives (the daily reset costs roughly one bar and
sessions are not uniformly populated). See ``PERCENTILE_WINDOW_BARS``.

Causality and warmup
--------------------
Every window ENDS at bar t, so a value at t is available for a decision at
t+1, matching the rest of the feature frame. Warmup rows emit NaN rather than
partial-window values: a feature is defined only once bar t has a full
``window`` of preceding bars AND at least half of them carry unmasked returns
(``MIN_VALID_FRACTION``, a sanity floor that never binds on clean 15-min data,
where the worst 96-bar window still holds 95 valid returns).

Scale-free by construction (decision F7): log returns are invariant to
multiplying the price series by a constant, and every output is either a
unitless shape statistic or a bps-scaled deviation. No column carries price
units, which is what allows one pooled model across USDJPY (~150) and EURUSD
(~1.08).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COL = "mid_close"

# Window lengths in BARS. Fixed by prior decision -- not tuned here.
PRIMARY_WINDOW = 96
FAST_WINDOW = 24
WINDOWS: tuple[int, ...] = (PRIMARY_WINDOW, FAST_WINDOW)

# Trailing window for the regime-relative percentile ranks and for the
# kurtosis-divergence standardization. 250 sessions x 94.84 bars/session,
# measured across all six pairs' 15-min frames (see module docstring).
PERCENTILE_TRADING_SESSIONS = 250
BARS_PER_SESSION_15MIN = 94.84
PERCENTILE_WINDOW_BARS = int(round(PERCENTILE_TRADING_SESSIONS * BARS_PER_SESSION_15MIN))  # 23_710

# Moors octile kurtosis of a normal distribution (reference value, for reading
# the raw statistic -- it is NOT subtracted).
MOORS_NORMAL = 1.2331

# Quantile-estimator denominators at or below this are treated as degenerate
# and yield NaN instead of a huge or infinite value. On log-return scale 1e-12
# is ~1e-8 bps, i.e. unmeasurably small.
IQR_EPS = 1e-12

# A window must carry at least this fraction of unmasked returns to emit a
# value. A sanity floor, not a tuned parameter: on clean 15-min data the worst
# 96-bar window still holds 95 of 96 returns.
MIN_VALID_FRACTION = 0.5

OCTILES: tuple[float, ...] = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


def gap_return_mask(index: pd.DatetimeIndex, bar_interval: pd.Timedelta | None = None) -> pd.Series:
    """Boolean Series: True where the return INTO that bar spans a market gap.

    A transition is gap-spanning when the index step exceeds the modal bar
    interval -- the same discontinuities `fxalgo.data.quality` classifies as
    weekend / daily_reset / active. The first bar is True (no prior bar, so no
    return). Pass `bar_interval` to override the inferred modal step.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("gap_return_mask requires a DatetimeIndex")
    steps = index.to_series().diff()
    if bar_interval is None:
        modal = steps.mode()
        if modal.empty:
            raise ValueError("cannot infer a bar interval from fewer than 2 timestamps")
        bar_interval = modal.iloc[0]
    mask = steps > bar_interval
    mask.iloc[0] = True  # the first bar has no return at all
    return mask


def _rolling_octiles(r: pd.Series, window: int, min_periods: int) -> dict[float, pd.Series]:
    roll = r.rolling(window=window, min_periods=min_periods)
    return {q: roll.quantile(q) for q in OCTILES}


def _guarded_ratio(num: pd.Series, denom: pd.Series, eps: float = IQR_EPS) -> pd.Series:
    """num / denom, returning NaN (never +/-inf) where denom is degenerate."""
    safe = denom.where(denom.abs() > eps)
    return num / safe


def _trailing_z(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window=window, min_periods=window)
    sd = roll.std(ddof=1)
    return (s - roll.mean()) / sd.where(sd > 0)


def add_features(
    df: pd.DataFrame,
    *,
    windows: tuple[int, ...] = WINDOWS,
    mask_gap_returns: bool = True,
    percentile_window: int = PERCENTILE_WINDOW_BARS,
    price_col: str = PRICE_COL,
) -> pd.DataFrame:
    """Append the distributional-moment columns. Pure: does not touch disk.

    Columns added (18): ret_skew_{96,24}, ret_kurt_{96,24},
    ret_bowley_skew_{96,24}, ret_moors_kurt_{96,24},
    ret_semidev_{down,up,ratio}_{96,24}, skew_robust_divergence_96,
    kurt_robust_divergence_96, ret_skew_96_pctile, ret_kurt_96_pctile.
    """
    if price_col not in df.columns:
        raise KeyError(f"moments.add_features requires column {price_col!r}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("moments.add_features requires a DatetimeIndex")

    out = df.copy()
    price = out[price_col].astype("float64")
    r = np.log(price).diff()

    if mask_gap_returns:
        r = r.where(~gap_return_mask(out.index).to_numpy())

    n = len(out)
    position = np.arange(n)

    for w in windows:
        min_periods = max(4, int(np.ceil(MIN_VALID_FRACTION * w)))
        # A value needs a FULL w-bar window of history behind it; anything
        # earlier is warmup and stays NaN.
        warm = position < w

        roll = r.rolling(window=w, min_periods=min_periods)
        # Guard degenerate (zero-variance) windows: pandas returns a finite
        # -3.0 for kurt on constant input, which would masquerade as a real
        # thin-tail reading. Undefined -> NaN.
        defined = roll.std(ddof=1) > 0
        skew = roll.skew().where(defined)
        kurt = roll.kurt().where(defined)

        oct_ = _rolling_octiles(r, w, min_periods)
        e1, e2, e3 = oct_[0.125], oct_[0.25], oct_[0.375]
        e4, e5, e6, e7 = oct_[0.5], oct_[0.625], oct_[0.75], oct_[0.875]
        iqr = e6 - e2
        bowley = _guarded_ratio(e6 + e2 - 2.0 * e4, iqr)
        moors = _guarded_ratio((e7 - e5) + (e3 - e1), iqr)

        down = r.clip(upper=0.0)
        up = r.clip(lower=0.0)
        semidev_down = 1e4 * np.sqrt(
            (down**2).rolling(window=w, min_periods=min_periods).mean()
        )
        semidev_up = 1e4 * np.sqrt(
            (up**2).rolling(window=w, min_periods=min_periods).mean()
        )
        semidev_ratio = _guarded_ratio(semidev_down, semidev_up, eps=0.0)

        for name, series in (
            (f"ret_skew_{w}", skew),
            (f"ret_kurt_{w}", kurt),
            (f"ret_bowley_skew_{w}", bowley),
            (f"ret_moors_kurt_{w}", moors),
            (f"ret_semidev_down_{w}", semidev_down),
            (f"ret_semidev_up_{w}", semidev_up),
            (f"ret_semidev_ratio_{w}", semidev_ratio),
        ):
            out[name] = series.mask(warm)

    # --- divergence between moment and robust estimators (primary window) ---
    # BOTH terms are z-scored first: a raw difference is swamped by the moment
    # term's much larger dispersion and just reproduces it (see docstring).
    pw = PRIMARY_WINDOW
    out[f"skew_robust_divergence_{pw}"] = (
        _trailing_z(out[f"ret_skew_{pw}"], percentile_window)
        - _trailing_z(out[f"ret_bowley_skew_{pw}"], percentile_window)
    )
    out[f"kurt_robust_divergence_{pw}"] = (
        _trailing_z(out[f"ret_kurt_{pw}"], percentile_window)
        - _trailing_z(out[f"ret_moors_kurt_{pw}"], percentile_window)
    )

    # --- regime-relative percentile ranks over a TRAILING window ---
    # Rolling.rank ranks the LAST value of each window, so the window ends at
    # bar t. A full-sample rank here would be a lookahead bug.
    for base in (f"ret_skew_{pw}", f"ret_kurt_{pw}"):
        out[f"{base}_pctile"] = (
            out[base]
            .rolling(window=percentile_window, min_periods=percentile_window)
            .rank(pct=True)
        )

    return out


def feature_columns(windows: tuple[int, ...] = WINDOWS) -> list[str]:
    """The exact list of columns `add_features` appends, in order."""
    cols: list[str] = []
    for w in windows:
        cols += [
            f"ret_skew_{w}", f"ret_kurt_{w}", f"ret_bowley_skew_{w}",
            f"ret_moors_kurt_{w}", f"ret_semidev_down_{w}",
            f"ret_semidev_up_{w}", f"ret_semidev_ratio_{w}",
        ]
    cols += [
        f"skew_robust_divergence_{PRIMARY_WINDOW}",
        f"kurt_robust_divergence_{PRIMARY_WINDOW}",
        f"ret_skew_{PRIMARY_WINDOW}_pctile",
        f"ret_kurt_{PRIMARY_WINDOW}_pctile",
    ]
    return cols
