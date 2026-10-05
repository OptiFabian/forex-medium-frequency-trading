"""Candle-wick / body-shape features on the 15-min (or any) mid OHLC.

Uses MIDPOINT OHLC (mid_open/high/low/close), not bid/ask, so the shape
reflects true price action without spread artifacts. All values come from a
single bar's own OHLC, known at that bar's close -> strictly causal (a bar's
wick features never depend on any later bar). Verified by the pipeline-wide
lookahead test plus a dedicated one.

Per bar:
  body           = |mid_close - mid_open|
  upper_wick     = mid_high - max(mid_open, mid_close)
  lower_wick     = min(mid_open, mid_close) - mid_low
  total_range    = mid_high - mid_low
  upper_wick_frac = upper_wick / total_range
  lower_wick_frac = lower_wick / total_range
  body_frac       = body / total_range
  wick_imbalance  = (upper_wick - lower_wick) / total_range
      Signed: > 0 = top rejection (long upper wick, price pushed up then
      sold back down); < 0 = bottom rejection.

By construction upper_wick_frac + lower_wick_frac + body_frac == 1 on any
bar with total_range > 0.

Flat bars (total_range == 0, i.e. mid_open==high==low==close): the four
fractions are UNDEFINED and set to NaN (a flat bar has no shape). The raw
body / wick / range columns are all 0 there. NaN fractions are treated as
warmup/skip by downstream consumers, per the feature NaN policy. On 15-min
mid data flat bars are essentially nonexistent, but the guard is explicit.

Volatility-normalized wick SIZE (a fraction of a small candle is not a big
wick -- these measure "big compared to what is normal now"):
  upper_wick_atr / lower_wick_atr : wick in units of ATR(14). Added only when
      the `atr_14` column is present (the pipeline computes it before wick).
  upper_wick_rel / lower_wick_rel : wick divided by the recent average TOTAL
      wick (mean over the prior WICK_REL_WINDOW=20 bars, current bar excluded
      via shift(1) so a bar is never in its own denominator). NaN for the
      first ~21 bars (rolling + shift warmup) and where the recent average is 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATR_COL = "atr_14"
WICK_REL_WINDOW = 20


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the eight wick/body columns computed from mid OHLC."""
    out = df.copy()
    o = out["mid_open"]
    h = out["mid_high"]
    l = out["mid_low"]
    c = out["mid_close"]

    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    total_range = h - l

    # Numerical guard: tiny negatives from float noise clip to 0.
    upper_wick = upper_wick.clip(lower=0.0)
    lower_wick = lower_wick.clip(lower=0.0)

    safe_range = total_range.where(total_range > 0, np.nan)

    out["body"] = body
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick
    out["total_range"] = total_range
    out["upper_wick_frac"] = upper_wick / safe_range
    out["lower_wick_frac"] = lower_wick / safe_range
    out["body_frac"] = body / safe_range
    out["wick_imbalance"] = (upper_wick - lower_wick) / safe_range

    # Volatility-normalized wick SIZE (fixes the tiny-candle problem: a large
    # fraction of a small candle is not a big wick). Two references:
    #   *_atr : wick in units of ATR(14)  -- "big vs recent volatility".
    #   *_rel : wick vs the recent average total wick (prior WICK_REL_WINDOW
    #           bars, current bar excluded) -- "big vs what's normal now".
    if ATR_COL in out.columns:
        atr = out[ATR_COL]
        safe_atr = atr.where(atr > 0, np.nan)
        out["upper_wick_atr"] = upper_wick / safe_atr
        out["lower_wick_atr"] = lower_wick / safe_atr

    total_wick = upper_wick + lower_wick
    avg_wick = (
        total_wick.rolling(WICK_REL_WINDOW, min_periods=WICK_REL_WINDOW).mean().shift(1)
    )
    safe_avg = avg_wick.where(avg_wick > 0, np.nan)
    out["upper_wick_rel"] = upper_wick / safe_avg
    out["lower_wick_rel"] = lower_wick / safe_avg
    return out
