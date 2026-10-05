"""Spread features: rolling stats on the bid/ask spread.

Operates on the `spread_close` column from the loader (already clipped at 0).
All causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPREAD_WINDOWS: tuple[int, ...] = (15, 60)
SPREAD_ZSCORE_WINDOW = 60


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append spread features.

    Columns added:
      - spread_mean_{w} for w in SPREAD_WINDOWS: rolling mean of spread_close
      - spread_ratio_{w} for w in SPREAD_WINDOWS: spread / rolling mean
      - spread_zscore_{SPREAD_ZSCORE_WINDOW}: (spread - mean) / rolling std
    """
    out = df.copy()
    spread = out["spread_close"]

    for w in SPREAD_WINDOWS:
        mean = spread.rolling(window=w, min_periods=w).mean()
        out[f"spread_mean_{w}"] = mean
        out[f"spread_ratio_{w}"] = spread / mean.where(mean != 0, np.nan)

    w = SPREAD_ZSCORE_WINDOW
    mean = spread.rolling(window=w, min_periods=w).mean()
    std = spread.rolling(window=w, min_periods=w).std()
    out[f"spread_zscore_{w}"] = (spread - mean) / std.where(std != 0, np.nan)
    return out
