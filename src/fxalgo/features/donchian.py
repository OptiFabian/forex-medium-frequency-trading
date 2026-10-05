"""Donchian channel features: rolling N-bar high/low, current bar EXCLUDED.

The channel at bar t is the extreme of the PREVIOUS `w` bars:

    don_high_w[t] = max(mid_high[t-w .. t-1])
    don_low_w[t]  = min(mid_low[t-w .. t-1])

Excluding the current bar is deliberate: with the current bar included,
price could never close ABOVE the channel high (the bar containing the
new extreme would extend the channel to meet it), so a close-vs-channel
breakout rule would be degenerate. Excluding bar t makes "close[t] >
don_high_w[t]" a true breakout: the close exceeded every high of the
prior `w` bars.

Implemented as rolling(w).max()/min() then shift(1) -- causal by
construction (value at t depends on bars <= t-1). Warmup: the first `w`
rows are NaN (`min_periods=w` plus the shift), per the feature NaN policy.

Windows (60, 180, 240, 360, 720, 1440 bars) are Turtle-style
breakout/exit lengths; here they feed the XGBoost feature set as channel
positions (`don_pos_*`, see `features.htf` / `training.dataset`).
"""

from __future__ import annotations

import pandas as pd

DONCHIAN_WINDOWS: tuple[int, ...] = (60, 180, 240, 360, 720, 1440)


def donchian(
    high: pd.Series, low: pd.Series, window: int
) -> tuple[pd.Series, pd.Series]:
    """(channel_high, channel_low) over the previous `window` bars, current bar excluded."""
    if window < 1:
        raise ValueError(f"Donchian window must be >= 1, got {window}")
    ch_high = high.rolling(window=window, min_periods=window).max().shift(1)
    ch_low = low.rolling(window=window, min_periods=window).min().shift(1)
    return ch_high, ch_low


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append `don_high_{w}` / `don_low_{w}` for w in DONCHIAN_WINDOWS.

    Computed on mid_high / mid_low (consistent with ATR and Bollinger:
    channels describe the mid-price path; execution costs are the
    engine's concern).
    """
    out = df.copy()
    high = out["mid_high"]
    low = out["mid_low"]
    for w in DONCHIAN_WINDOWS:
        ch_high, ch_low = donchian(high, low, w)
        out[f"don_high_{w}"] = ch_high
        out[f"don_low_{w}"] = ch_low
    return out
