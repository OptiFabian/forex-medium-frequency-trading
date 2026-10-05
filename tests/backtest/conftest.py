"""Shared fixtures for backtest engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_engine_frame(
    mids,
    *,
    spread: float = 0.0002,
    start: str = "2024-01-01 00:00",
    freq: str = "1min",
    highs=None,
    lows=None,
) -> pd.DataFrame:
    """Build the MINIMAL frame the engine needs.

    `mids` is mid_open == mid_close per bar (flat OHLC by default).
    bid = mid - spread/2, ask = mid + spread/2.

    Optional `highs` / `lows` override mid_high / mid_low for intrabar-exit
    tests. Default: mid_high == mid_low == mid_close (no intrabar range).
    """
    mids = np.asarray(mids, dtype="float64")
    n = len(mids)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    half = spread / 2.0
    if highs is None:
        highs = mids
    else:
        highs = np.asarray(highs, dtype="float64")
        assert len(highs) == n
    if lows is None:
        lows = mids
    else:
        lows = np.asarray(lows, dtype="float64")
        assert len(lows) == n
    return pd.DataFrame(
        {
            "open_bid": mids - half,
            "open_ask": mids + half,
            "mid_open": mids,
            "mid_close": mids,
            "mid_high": highs,
            "mid_low": lows,
        },
        index=idx,
    )


def make_signals(values, index) -> pd.Series:
    """Wrap a list of ints into a Series aligned to `index`."""
    return pd.Series(np.asarray(values, dtype=np.int8), index=index, name="signal")
