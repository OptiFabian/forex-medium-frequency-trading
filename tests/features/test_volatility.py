"""Tests for fxalgo.features.volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import volatility
from tests.features.conftest import make_frame


def test_atr_of_flat_series_is_zero(constant_frame):
    out = volatility.add_features(constant_frame)
    # In make_frame, high == low == close, so TR is always 0, ATR is 0.
    atr = out[f"atr_{volatility.ATR_PERIOD}"]
    np.testing.assert_allclose(atr.to_numpy(), 0.0, atol=1e-12)


def test_atr_known_value_with_explicit_high_low():
    """Hand-verify TR on a tiny frame where high - low = 1.0 throughout."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "mid_open": np.ones(n) * 100.0,
            "mid_high": np.ones(n) * 100.5,
            "mid_low": np.ones(n) * 99.5,
            "mid_close": np.ones(n) * 100.0,
            "open_bid": np.ones(n) * 100.0,
            "high_bid": np.ones(n) * 100.5,
            "low_bid": np.ones(n) * 99.5,
            "close_bid": np.ones(n) * 100.0,
            "open_ask": np.ones(n) * 100.0,
            "high_ask": np.ones(n) * 100.5,
            "low_ask": np.ones(n) * 99.5,
            "close_ask": np.ones(n) * 100.0,
            "spread_close": np.zeros(n),
        },
        index=idx,
    )
    out = volatility.add_features(df)
    # TR = max(high-low, |high-prev_close|, |low-prev_close|) = max(1, 0.5, 0.5) = 1
    # (row 0: prev_close NaN, TR = high - low = 1)
    # Wilder's ATR with alpha=1/14 of a constant 1.0 series is 1.0 from row 0.
    atr = out[f"atr_{volatility.ATR_PERIOD}"]
    assert atr.iloc[0] == pytest.approx(1.0)
    assert atr.iloc[-1] == pytest.approx(1.0)


def test_bollinger_pct_b_at_known_positions(ramp_frame):
    out = volatility.add_features(ramp_frame)
    # By construction (linear ramp), close is monotonic. After warmup,
    # most recent close should be near the upper band (> 0.5), and 20 rows
    # before that should also be > 0.5 (because std stays nonzero and
    # close is at the top of the trailing window).
    pct_b = out["bb_pct_b"].iloc[-1]
    assert pct_b > 0.5 and pct_b <= 1.0


def test_bollinger_bands_constant_series_are_flat(constant_frame):
    out = volatility.add_features(constant_frame)
    # Once enough rows: middle = close, std = 0, pct_b undefined (NaN by guard).
    middle = out["bb_middle"]
    upper = out["bb_upper"]
    lower = out["bb_lower"]
    # All defined values of middle should equal the constant.
    np.testing.assert_allclose(middle.dropna().to_numpy(), 1.10)
    # Upper and lower coincide with middle when std == 0.
    np.testing.assert_allclose(
        upper.dropna().to_numpy(),
        middle.dropna().to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lower.dropna().to_numpy(),
        middle.dropna().to_numpy(),
        atol=1e-12,
    )
    # %B with zero band width is intentionally NaN (no information).
    assert np.isnan(out["bb_pct_b"].iloc[-1])


def test_ret_std_warmup_rows_are_nan(ramp_frame):
    out = volatility.add_features(ramp_frame)
    for w in volatility.RET_STD_WINDOWS:
        col = f"ret_std_{w}"
        # First w rows of the rolling stat aren't defined (need w return obs,
        # which requires w+1 closes; row 0's return is NaN so add 1).
        assert out[col].iloc[: w].isna().all()
