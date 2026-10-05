"""Directional-movement features: ADX, +DI, -DI, and the DI spread.

Wilder's Average Directional Index and the two Directional Indicators,
computed on mid_high / mid_low / mid_close (consistent with ATR and
Bollinger). Standard Wilder method:

1. True Range (reused from `volatility._true_range`) and Directional
   Movement per bar:
     up   = high_t - high_{t-1}
     down = low_{t-1} - low_t
     +DM  = up   if up > down and up > 0   else 0
     -DM  = down if down > up and down > 0  else 0
2. Wilder-smooth TR, +DM, -DM with alpha = 1/period (the same SMA-seeded
   RMA the RSI uses, via `momentum._wilder_rma`). The smoothed TR is the
   ATR; the /period factors cancel in the DI ratio, so the RMA (mean)
   form gives Wilder's DI values exactly.
3. +DI = 100 * smoothed(+DM) / ATR ; -DI = 100 * smoothed(-DM) / ATR.
4. DX  = 100 * |+DI - -DI| / (+DI + -DI).
5. ADX = Wilder-smooth of DX (a SECOND smoothing).

Warmup / causality
------------------
Both smoothings are causal (`_wilder_rma` seeds from a trailing SMA then
runs forward). The DI series is defined from index `period` (the first
smoothing needs `period` DM/TR values); ADX is defined from index
`2*period - 1` (the second smoothing needs `period` DX values). Earlier
rows are NaN per the feature warmup policy. A lookahead test enforces
that no value at row t depends on any row > t.

Interpretation
-------------
ADX measures trend STRENGTH regardless of direction (0-100; conventional
bands: <20 no trend / ranging, 20-25 weak, 25-40 trending, >40 strong).
+DI/-DI carry the direction ADX omits. `di_spread = +DI - -DI` is a
signed directional-strength feature (>0 up-pressure, <0 down-pressure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.features.momentum import _wilder_rma
from fxalgo.features.volatility import _true_range

ADX_PERIOD = 14


def _directional_movement(high: pd.Series, low: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(+DM, -DM) per bar. Index 0 is 0 (no previous bar)."""
    up = high.diff().to_numpy(dtype=np.float64).copy()
    down = (-low.diff()).to_numpy(dtype=np.float64).copy()  # low_{t-1} - low_t
    up[0] = 0.0
    down[0] = 0.0
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    return plus_dm, minus_dm


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder ADX, +DI, -DI over `period`.

    Returns (adx, plus_di, minus_di) aligned to `close.index`. Causal;
    DI NaN for the first `period` rows, ADX NaN for the first `2*period-1`.
    """
    if period < 1:
        raise ValueError(f"ADX period must be >= 1, got {period}")

    idx = close.index
    tr = _true_range(high, low, close).to_numpy(dtype=np.float64)
    plus_dm, minus_dm = _directional_movement(high, low)

    atr = _wilder_rma(tr, period, start=1)
    plus_dm_s = _wilder_rma(plus_dm, period, start=1)
    minus_dm_s = _wilder_rma(minus_dm, period, start=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_s / atr
        minus_di = 100.0 * minus_dm_s / atr
        di_sum = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / di_sum

    # DX is defined from index `period` (where the DI ratio is defined). A
    # perfectly flat bar gives +DI = -DI = 0 -> 0/0 DX; treat that as 0
    # (no directional information) so the second smoothing stays finite.
    n = len(close)
    defined = np.arange(n) >= period
    dx = np.where(defined & ~np.isfinite(dx), 0.0, dx)

    adx_vals = _wilder_rma(dx, period, start=period)

    return (
        pd.Series(adx_vals, index=idx, name=f"adx_{period}"),
        pd.Series(plus_di, index=idx, name=f"plus_di_{period}"),
        pd.Series(minus_di, index=idx, name=f"minus_di_{period}"),
    )


def add_features(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Append `adx_{p}`, `plus_di_{p}`, `minus_di_{p}`, `di_spread_{p}`.

    Computed on mid_high / mid_low / mid_close. `di_spread` = +DI - -DI
    (signed directional strength, the information ADX alone discards).
    """
    out = df.copy()
    adx_s, plus_di, minus_di = adx(
        out["mid_high"], out["mid_low"], out["mid_close"], period=period
    )
    out[f"adx_{period}"] = adx_s
    out[f"plus_di_{period}"] = plus_di
    out[f"minus_di_{period}"] = minus_di
    out[f"di_spread_{period}"] = plus_di - minus_di
    return out
