"""Volatility features: ATR, rolling return std, Bollinger Bands.

Operates on mid_high/mid_low/mid_close. All causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_PERIOD = 14
RET_STD_WINDOWS: tuple[int, ...] = (15, 60)
BB_PERIOD = 20
BB_NUM_STD = 2.0


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(high-low, |high - prev_close|, |low - prev_close|)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append volatility columns.

    Columns added:
      - atr_{ATR_PERIOD}: Wilder's average true range on mid_high/low/close
      - ret_std_{w} for w in RET_STD_WINDOWS: rolling std of 1-min log returns
      - bb_middle, bb_upper, bb_lower, bb_pct_b (Bollinger on mid_close)
    """
    out = df.copy()
    high = out["mid_high"]
    low = out["mid_low"]
    close = out["mid_close"]

    tr = _true_range(high, low, close)
    # Wilder's smoothing == EMA with alpha = 1/N.
    out[f"atr_{ATR_PERIOD}"] = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    log_ret_1 = np.log(close) - np.log(close.shift(1))
    for w in RET_STD_WINDOWS:
        out[f"ret_std_{w}"] = log_ret_1.rolling(window=w, min_periods=w).std()

    middle = close.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean()
    std = close.rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std()
    upper = middle + BB_NUM_STD * std
    lower = middle - BB_NUM_STD * std
    band_width = upper - lower
    # Guard against zero-width band (constant close over window).
    pct_b = (close - lower) / band_width.where(band_width != 0, np.nan)
    out["bb_middle"] = middle
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_pct_b"] = pct_b
    return out
