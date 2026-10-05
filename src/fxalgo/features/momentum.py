"""Momentum features: Wilder's RSI.

Causal / trailing only, computed on `mid_close` (consistent with the
other indicators). Two periods are produced by default: the classic
14-period RSI and a short 2-period RSI (used by Connors-style mean
reversion) -- we may want either when designing an RSI + Bollinger
combination later.

Wilder's RSI uses the textbook SMA seed: the average of the first
`period` gains/losses seeds the running average, after which the
recursion smooths with alpha = 1/period. The first `period` rows (where
the seed is not yet defined) are left as NaN, per the feature warmup
policy. This is the same Wilder smoothing family the ATR uses, but seeded
the standard RSI way so a hand computation reproduces the values exactly.

Edge cases (post-warmup):
  - a window of only gains  -> avg_loss == 0 -> RSI = 100
  - a window of only losses -> avg_gain == 0 -> RSI = 0
  - a perfectly flat window  -> avg_gain == avg_loss == 0 -> RSI = 50 (neutral)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RSI_PERIODS: tuple[int, ...] = (14, 2)


def _wilder_rma(values: np.ndarray, period: int, start: int) -> np.ndarray:
    """SMA-seeded Wilder running moving average (a.k.a. RMA / SMMA).

    `values` holds gains or losses; `start` is the first index carrying a
    valid value (1 for a diff-derived series, since index 0 is undefined).
    The seed -- the mean of values[start : start+period] -- lands at index
    start+period-1, then the recursion
        rma_t = rma_{t-1} + (values_t - rma_{t-1}) / period
    runs forward. Positions before the seed are NaN.

    Wilder's recursion with alpha = 1/period is exactly an ewm(adjust=False)
    initialized at the SMA seed, so we seed the tail and let ewm vectorize it.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    seed_idx = start + period - 1
    if seed_idx >= n:
        return out
    seed = float(np.mean(values[start : start + period]))
    tail = values[seed_idx:].astype(np.float64).copy()
    tail[0] = seed
    rma_tail = pd.Series(tail).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    out[seed_idx:] = rma_tail
    return out


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI of `close` over `period`.

    Causal: RSI at row t depends only on rows <= t. NaN for the first
    `period` rows (the seed needs `period` price changes).
    """
    if period < 1:
        raise ValueError(f"RSI period must be >= 1, got {period}")
    delta = close.diff().to_numpy(dtype=np.float64)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    # delta[0] is NaN; np.where(NaN > 0, ...) is False -> 0 already, but be explicit.
    gain[0] = 0.0
    loss[0] = 0.0

    avg_gain = _wilder_rma(gain, period, start=1)
    avg_loss = _wilder_rma(loss, period, start=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_vals = 100.0 - 100.0 / (1.0 + rs)

    defined = ~np.isnan(avg_gain)  # avg_loss has the same NaN warmup pattern
    # Only gains in window -> RSI 100.
    rsi_vals = np.where(defined & (avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi_vals)
    # Flat window (no gains, no losses) -> neutral 50.
    rsi_vals = np.where(defined & (avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi_vals)
    return pd.Series(rsi_vals, index=close.index, name=f"rsi_{period}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append RSI columns (`rsi_{N}` for N in RSI_PERIODS) computed on mid_close."""
    out = df.copy()
    close = out["mid_close"]
    for p in RSI_PERIODS:
        out[f"rsi_{p}"] = rsi(close, p)
    return out
