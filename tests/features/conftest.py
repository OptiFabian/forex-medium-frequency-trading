"""Shared fixtures for feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(
    closes,
    *,
    spreads=None,
    start: str = "2024-01-01 00:00",
    freq: str = "1min",
    bid_offset: float = 0.0001,
) -> pd.DataFrame:
    """Build a feature-ready frame with mid_close = closes.

    OHLC are all set to the same value (flat bars) so high/low/open == close
    and mid_high == mid_low == mid_close. This keeps unit tests focused on a
    single varying series without OHLC noise.

    `spreads`: optional list/array same length as `closes`; if None, spread is 0.
    """
    closes = np.asarray(closes, dtype="float64")
    n = len(closes)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")

    if spreads is None:
        spread = np.zeros(n, dtype="float64")
    else:
        spread = np.asarray(spreads, dtype="float64")
        assert len(spread) == n

    half_spread = spread / 2.0
    close_bid = closes - half_spread
    close_ask = closes + half_spread

    df = pd.DataFrame(
        {
            "open_bid": close_bid,
            "high_bid": close_bid,
            "low_bid": close_bid,
            "close_bid": close_bid,
            "open_ask": close_ask,
            "high_ask": close_ask,
            "low_ask": close_ask,
            "close_ask": close_ask,
            "mid_open": closes,
            "mid_high": closes,
            "mid_low": closes,
            "mid_close": closes,
            "spread_close": spread.clip(min=0.0),
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def constant_frame():
    """120 rows where mid_close = 1.10 (no movement)."""
    return make_frame([1.10] * 120)


@pytest.fixture
def ramp_frame():
    """120 rows where mid_close ramps linearly from 1.10 to 1.12."""
    return make_frame(np.linspace(1.10, 1.12, 120))


@pytest.fixture
def random_walk_frame():
    """500 rows of synthetic random-walk prices (seeded)."""
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(0.0, 0.0001, size=500)
    closes = 1.10 + np.cumsum(steps)
    return make_frame(closes)
