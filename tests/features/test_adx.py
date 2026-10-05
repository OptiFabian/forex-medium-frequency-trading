"""Tests for ADX / +DI / -DI / DI-spread (Wilder directional movement).

The centerpiece is a fully hand-computed period=2 series: every
intermediate (TR, +DM/-DM, the two Wilder smoothings, DI, DX, and the
second-smoothed ADX) is worked out in the test so the asserted values are
independently verifiable, not pinned to whatever the code happens to emit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features.adx import ADX_PERIOD, add_features, adx
from tests.features.conftest import make_frame


def _ohlc(highs, lows, closes):
    n = len(highs)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return (
        pd.Series(np.asarray(highs, float), index=idx),
        pd.Series(np.asarray(lows, float), index=idx),
        pd.Series(np.asarray(closes, float), index=idx),
    )


# ============================================================
# 1. Fully hand-computed period=2 series.
# ============================================================


def test_hand_verified_adx_period_2():
    """5-bar series worked out by hand (period=2).

    bar  H    L    C
     0   10    8    9
     1   12    9   11
     2   11    7    8
     3   13   10   12
     4   14   11   13

    TR:   t1 max(3,3,0)=3  t2 max(4,0,4)=4  t3 max(3,5,2)=5  t4 max(3,2,1)=3
    +DM:  t1 2  t2 0  t3 2  t4 1
    -DM:  t1 0  t2 2  t3 0  t4 0
    RMA(2) TR:   seed@2 (3+4)/2=3.5 ; @3 4.25 ; @4 3.625
    RMA(2) +DM:  seed@2 (2+0)/2=1.0 ; @3 1.5  ; @4 1.25
    RMA(2) -DM:  seed@2 (0+2)/2=1.0 ; @3 0.5  ; @4 0.25
    +DI@4 = 100*1.25/3.625 = 34.4828   -DI@4 = 100*0.25/3.625 = 6.8966
    DX:   @2 0 ; @3 100*|35.294-11.765|/47.059 = 50 ; @4 100*|34.483-6.897|/41.379 = 66.6667
    ADX = RMA(2) of DX, seed@3 (0+50)/2 = 25.0 ; @4 25+(66.6667-25)/2 = 45.8333
    """
    high, low, close = _ohlc(
        [10, 12, 11, 13, 14], [8, 9, 7, 10, 11], [9, 11, 8, 12, 13]
    )
    a, plus_di, minus_di = adx(high, low, close, period=2)

    # DI defined from index 2, ADX from index 3 (=2*period-1).
    assert plus_di.iloc[:2].isna().all()
    assert a.iloc[:3].isna().all()

    assert plus_di.iloc[4] == pytest.approx(34.48276, rel=1e-4)
    assert minus_di.iloc[4] == pytest.approx(6.89655, rel=1e-4)
    assert a.iloc[3] == pytest.approx(25.0, rel=1e-6)
    assert a.iloc[4] == pytest.approx(45.83333, rel=1e-4)


# ============================================================
# 2. Pure trends: exact DI / ADX (degenerate but hand-checkable).
# ============================================================


def test_pure_uptrend_gives_plus_di_50_minus_di_0_adx_100():
    """H=10+t, L=8+t, C=9+t: every bar +DM=1, -DM=0, TR=2.
    -> smoothed(+DM)=1, ATR=2 -> +DI=50, -DI=0, DX=100, ADX=100 exactly."""
    n = 40
    t = np.arange(n, dtype=float)
    high, low, close = _ohlc(10 + t, 8 + t, 9 + t)
    a, plus_di, minus_di = adx(high, low, close, period=5)
    assert plus_di.iloc[-1] == pytest.approx(50.0, rel=1e-9)
    assert minus_di.iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert a.iloc[-1] == pytest.approx(100.0, rel=1e-9)


def test_pure_downtrend_gives_minus_di_50_plus_di_0_adx_100():
    n = 40
    t = np.arange(n, dtype=float)
    high, low, close = _ohlc(20 - t, 18 - t, 19 - t)
    a, plus_di, minus_di = adx(high, low, close, period=5)
    assert minus_di.iloc[-1] == pytest.approx(50.0, rel=1e-9)
    assert plus_di.iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert a.iloc[-1] == pytest.approx(100.0, rel=1e-9)


# ============================================================
# 3. Warmup, bounds, DI spread.
# ============================================================


@pytest.mark.parametrize("period", [2, 5, 14])
def test_warmup_nan_counts(period):
    rng = np.random.default_rng(3)
    n = 4 * period + 10
    closes = 1.10 + np.cumsum(rng.normal(0, 0.0003, n))
    df = make_frame(closes)
    out = add_features(df, period=period)
    di = out[f"plus_di_{period}"]
    a = out[f"adx_{period}"]
    assert di.iloc[:period].isna().all()
    assert di.iloc[period:].notna().all()
    assert a.iloc[: 2 * period - 1].isna().all()
    assert a.iloc[2 * period - 1 :].notna().all()


def test_bounds_and_di_spread():
    rng = np.random.default_rng(9)
    closes = 1.10 + np.cumsum(rng.normal(0, 0.0004, 600))
    highs = closes + rng.uniform(0, 0.0003, 600)
    lows = closes - rng.uniform(0, 0.0003, 600)
    idx = pd.date_range("2024-01-01", periods=600, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"mid_high": highs, "mid_low": lows, "mid_close": closes}, index=idx
    )
    out = add_features(df)
    p = ADX_PERIOD
    for col in (f"adx_{p}", f"plus_di_{p}", f"minus_di_{p}"):
        v = out[col].dropna().to_numpy()
        assert (v >= -1e-9).all() and (v <= 100.0 + 1e-9).all(), col
    spread = out[f"di_spread_{p}"]
    expected = out[f"plus_di_{p}"] - out[f"minus_di_{p}"]
    pd.testing.assert_series_equal(spread, expected, check_names=False)


def test_invalid_period_rejected():
    s = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC"))
    with pytest.raises(ValueError, match="period"):
        adx(s, s, s, period=0)


# ============================================================
# 4. No lookahead.
# ============================================================


def test_no_lookahead_in_adx():
    rng = np.random.default_rng(21)
    closes = 1.10 + np.cumsum(rng.normal(0, 0.0004, 400))
    df = make_frame(closes)
    split = 200

    full = add_features(df)
    corrupted = df.copy()
    corrupted.iloc[split + 1 :, :] = 9.99
    corr = add_features(corrupted)

    new_cols = [c for c in full.columns if c.startswith(("adx_", "plus_di_", "minus_di_", "di_spread_"))]
    assert new_cols
    for col in new_cols:
        a = full[col].iloc[: split + 1].to_numpy()
        b = corr[col].iloc[: split + 1].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=col)
        mask = ~np.isnan(a)
        assert np.array_equal(a[mask], b[mask]), f"lookahead in {col}"
