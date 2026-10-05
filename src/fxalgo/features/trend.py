"""Trend features: EMAs, EMA distance, MACD, log returns.

All operate on `mid_close` and use only trailing data (causal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EMA_PERIODS: tuple[int, ...] = (9, 21, 50, 99, 200)
LOG_RET_LOOKBACKS: tuple[int, ...] = (1, 5, 15, 60)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard recursive EMA (adjust=False): EMA_t = alpha*x_t + (1-alpha)*EMA_{t-1}."""
    return series.ewm(span=span, adjust=False).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append trend columns to `df` and return it.

    Columns added:
      - ema_{N} for N in EMA_PERIODS
      - dist_ema_{N} = (mid_close - ema_{N}) / ema_{N}  (fractional distance)
      - macd, macd_signal, macd_hist  (12/26/9 of mid_close)
      - log_ret_{k} = ln(mid_close_t / mid_close_{t-k}) for k in LOG_RET_LOOKBACKS
    """
    out = df.copy()
    close = out["mid_close"]

    for n in EMA_PERIODS:
        ema = _ema(close, n)
        out[f"ema_{n}"] = ema
        out[f"dist_ema_{n}"] = (close - ema) / ema

    ema_fast = _ema(close, MACD_FAST)
    ema_slow = _ema(close, MACD_SLOW)
    macd = ema_fast - ema_slow
    signal = _ema(macd, MACD_SIGNAL)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    log_close = np.log(close)
    for k in LOG_RET_LOOKBACKS:
        out[f"log_ret_{k}"] = log_close - log_close.shift(k)

    return out
