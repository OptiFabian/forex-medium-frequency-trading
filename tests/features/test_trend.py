"""Tests for fxalgo.features.trend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import trend
from tests.features.conftest import make_frame


def test_ema_of_constant_is_constant(constant_frame):
    out = trend.add_features(constant_frame)
    for n in trend.EMA_PERIODS:
        col = f"ema_{n}"
        # All values equal the constant (1.10) with no NaN.
        np.testing.assert_allclose(out[col].to_numpy(), 1.10)
        # Distance is zero everywhere.
        np.testing.assert_allclose(out[f"dist_ema_{n}"].to_numpy(), 0.0)


def test_ema_hand_computed_span_3():
    """Recursive EMA: alpha=2/(N+1)=0.5; EMA_0 = x_0; EMA_t = 0.5*x_t + 0.5*EMA_{t-1}."""
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    df = make_frame(closes)
    ema = df["mid_close"].ewm(span=3, adjust=False).mean()
    expected = [10.0, 10.5, 11.25, 12.125, 13.0625]
    np.testing.assert_allclose(ema.to_numpy(), expected)


def test_log_return_known_values():
    closes = [100.0, 105.0, 110.0, 100.0]
    df = make_frame(closes)
    out = trend.add_features(df)
    # log_ret_1
    assert np.isnan(out["log_ret_1"].iloc[0])
    assert out["log_ret_1"].iloc[1] == pytest.approx(np.log(105.0 / 100.0))
    assert out["log_ret_1"].iloc[2] == pytest.approx(np.log(110.0 / 105.0))
    assert out["log_ret_1"].iloc[3] == pytest.approx(np.log(100.0 / 110.0))


def test_macd_of_constant_is_zero(constant_frame):
    out = trend.add_features(constant_frame)
    # MACD of a constant series is identically zero.
    np.testing.assert_allclose(out["macd"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["macd_signal"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["macd_hist"].to_numpy(), 0.0, atol=1e-12)


def test_macd_sign_on_uptrend(ramp_frame):
    """On a monotonic uptrend, MACD should be positive once warmed up."""
    out = trend.add_features(ramp_frame)
    # Skip warmup window: by row 60 the fast EMA should be above slow.
    assert (out["macd"].iloc[60:] > 0).all()


def test_first_log_return_row_is_nan():
    df = make_frame([1.0, 1.1, 1.2, 1.3, 1.4])
    out = trend.add_features(df)
    for k in trend.LOG_RET_LOOKBACKS:
        # First k rows are NaN (no row at t-k yet).
        assert out[f"log_ret_{k}"].iloc[:k].isna().all()
        if len(df) > k:
            assert not np.isnan(out[f"log_ret_{k}"].iloc[k])
