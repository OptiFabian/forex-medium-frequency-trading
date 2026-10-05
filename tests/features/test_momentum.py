"""Tests for fxalgo.features.momentum (Wilder's RSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import momentum
from tests.features.conftest import make_frame


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1min", tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=idx, name="mid_close")


# ============================================================
# 1. Hand-verified RSI on a small known series (period=2).
# ============================================================


def test_rsi_hand_verified_period_2():
    """closes = [10, 11, 10, 11, 10], period=2.

    deltas:        _, +1, -1, +1, -1
    gain:          _,  1,  0,  1,  0     loss: _, 0, 1, 0, 1
    seed @ idx2:   avg_gain = mean(1,0)=0.5   avg_loss = mean(0,1)=0.5  -> RSI 50
    idx3:          avg_gain=(0.5+1)/2=0.75     avg_loss=0.25            -> RS 3   -> RSI 75
    idx4:          avg_gain=0.375              avg_loss=0.625           -> RS 0.6 -> RSI 37.5
    First `period` rows are NaN (warmup).
    """
    r = momentum.rsi(_series([10, 11, 10, 11, 10]), period=2)
    assert np.isnan(r.iloc[0])
    assert np.isnan(r.iloc[1])
    assert r.iloc[2] == pytest.approx(50.0)
    assert r.iloc[3] == pytest.approx(75.0)
    assert r.iloc[4] == pytest.approx(37.5)


# ============================================================
# 2. Degenerate windows: all gains -> 100, all losses -> 0, flat -> 50.
# ============================================================


def test_rsi_all_gains_is_100():
    r = momentum.rsi(_series(np.arange(1.0, 40.0)), period=14)
    # Every change is a gain -> avg_loss == 0 -> RSI 100 for all defined rows.
    defined = r.dropna()
    np.testing.assert_allclose(defined.to_numpy(), 100.0)


def test_rsi_all_losses_is_0():
    r = momentum.rsi(_series(np.arange(40.0, 1.0, -1.0)), period=14)
    defined = r.dropna()
    np.testing.assert_allclose(defined.to_numpy(), 0.0)


def test_rsi_flat_series_is_50():
    r = momentum.rsi(_series([1.10] * 40), period=14)
    defined = r.dropna()
    np.testing.assert_allclose(defined.to_numpy(), 50.0)


# ============================================================
# 3. Warmup: first `period` rows are NaN, then defined.
# ============================================================


@pytest.mark.parametrize("period", [2, 14])
def test_rsi_warmup_is_nan(period):
    r = momentum.rsi(_series(1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, 200))), period)
    assert r.iloc[:period].isna().all()
    assert not np.isnan(r.iloc[period])


# ============================================================
# 4. RSI is bounded in [0, 100].
# ============================================================


def test_rsi_bounded_0_100():
    rng = np.random.default_rng(3)
    closes = 1.10 + np.cumsum(rng.normal(0, 2e-4, 1000))
    for period in (2, 14):
        r = momentum.rsi(_series(closes), period).dropna().to_numpy()
        assert (r >= 0.0).all() and (r <= 100.0).all()


# ============================================================
# 5. add_features attaches the configured columns on mid_close.
# ============================================================


def test_add_features_adds_rsi_columns():
    df = make_frame(1.10 + np.cumsum(np.random.default_rng(1).normal(0, 1e-4, 100)))
    out = momentum.add_features(df)
    for p in momentum.RSI_PERIODS:
        assert f"rsi_{p}" in out.columns
    # Computed on mid_close: a standalone call must match the column.
    expected = momentum.rsi(df["mid_close"], 14).to_numpy()
    np.testing.assert_array_equal(
        np.nan_to_num(out["rsi_14"].to_numpy(), nan=-1.0),
        np.nan_to_num(expected, nan=-1.0),
    )


# ============================================================
# 6. No lookahead: RSI at row t unchanged when future rows are altered.
# ============================================================


def test_rsi_no_lookahead():
    rng = np.random.default_rng(7)
    closes = 1.10 + np.cumsum(rng.normal(0, 1e-4, 600))
    df = make_frame(closes)
    out_orig = momentum.add_features(df)

    for T in (50, 200, 400):
        df_corr = df.copy()
        df_corr.iloc[T + 1 :, df_corr.columns.get_loc("mid_close")] = 9.99
        out_corr = momentum.add_features(df_corr)
        for p in momentum.RSI_PERIODS:
            a = out_orig[f"rsi_{p}"].iloc[: T + 1].to_numpy()
            b = out_corr[f"rsi_{p}"].iloc[: T + 1].to_numpy()
            mask = ~np.isnan(a)
            assert (np.isnan(a) == np.isnan(b)).all(), f"rsi_{p} NaN pattern changed at T={T}"
            assert np.array_equal(a[mask], b[mask]), f"lookahead in rsi_{p} at T={T}"
