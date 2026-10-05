"""Tests for fxalgo.features.spread_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import spread_features
from tests.features.conftest import make_frame


def test_rolling_mean_of_constant_spread():
    n = 100
    closes = np.full(n, 1.10)
    spreads = np.full(n, 0.0001)
    df = make_frame(closes, spreads=spreads)
    out = spread_features.add_features(df)
    for w in spread_features.SPREAD_WINDOWS:
        col = f"spread_mean_{w}"
        # First w-1 rows are NaN, the rest equal the constant.
        assert out[col].iloc[: w - 1].isna().all()
        np.testing.assert_allclose(out[col].iloc[w - 1 :].to_numpy(), 0.0001)


def test_spread_ratio_equals_one_when_spread_is_constant():
    n = 100
    closes = np.full(n, 1.10)
    spreads = np.full(n, 0.0001)
    df = make_frame(closes, spreads=spreads)
    out = spread_features.add_features(df)
    for w in spread_features.SPREAD_WINDOWS:
        col = f"spread_ratio_{w}"
        # Once the window is full, ratio is 1.0.
        np.testing.assert_allclose(out[col].iloc[w - 1 :].to_numpy(), 1.0)


def test_spread_zscore_known_step():
    # 70 rows: first 60 at 0.0001, then 10 at 0.0005.
    spreads = [0.0001] * 60 + [0.0005] * 10
    closes = [1.10] * len(spreads)
    df = make_frame(closes, spreads=spreads)
    out = spread_features.add_features(df)
    # At row 60: window includes 59 values of 0.0001 and 1 of 0.0005. The
    # mean is just above 0.0001 and the std is small; the z-score should be
    # large and positive.
    z60 = out["spread_zscore_60"].iloc[60]
    assert z60 > 5.0
