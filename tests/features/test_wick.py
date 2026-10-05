"""Tests for candle-wick / body-shape features (hand-verified candle shapes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features.wick import add_features


def _frame(rows, atr=None):
    """rows: list of (o, h, l, c) mid OHLC tuples. Optional atr_14 column."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    arr = np.asarray(rows, dtype="float64")
    data = {"mid_open": arr[:, 0], "mid_high": arr[:, 1], "mid_low": arr[:, 2], "mid_close": arr[:, 3]}
    if atr is not None:
        data["atr_14"] = np.asarray(atr, dtype="float64")
    return pd.DataFrame(data, index=idx)


# ============================================================
# 1. Hand-verified canonical candle shapes.
# ============================================================


def test_shooting_star():
    """Small body at the bottom, long UPPER wick, ~no lower wick -> top rejection.
    o=10.0 h=13.0 l=10.0 c=10.2:
      body=0.2  upper=13-10.2=2.8  lower=10-10=0  range=3.0
      upper_frac=0.9333 lower_frac=0 body_frac=0.0667 imbalance=+0.9333"""
    out = add_features(_frame([(10.0, 13.0, 10.0, 10.2)]))
    r = out.iloc[0]
    assert r["body"] == pytest.approx(0.2)
    assert r["upper_wick"] == pytest.approx(2.8)
    assert r["lower_wick"] == pytest.approx(0.0)
    assert r["total_range"] == pytest.approx(3.0)
    assert r["upper_wick_frac"] == pytest.approx(2.8 / 3.0)
    assert r["lower_wick_frac"] == pytest.approx(0.0)
    assert r["body_frac"] == pytest.approx(0.2 / 3.0)
    assert r["wick_imbalance"] == pytest.approx(2.8 / 3.0)


def test_hammer():
    """Small body at the top, long LOWER wick -> bottom rejection.
    o=10.0 h=10.2 l=8.0 c=10.2:
      body=0.2 upper=10.2-10.2=0 lower=10-8=2.0 range=2.2
      lower_frac=0.9091 upper_frac=0 imbalance=-0.9091"""
    out = add_features(_frame([(10.0, 10.2, 8.0, 10.2)]))
    r = out.iloc[0]
    assert r["upper_wick"] == pytest.approx(0.0)
    assert r["lower_wick"] == pytest.approx(2.0)
    assert r["total_range"] == pytest.approx(2.2)
    assert r["lower_wick_frac"] == pytest.approx(2.0 / 2.2)
    assert r["upper_wick_frac"] == pytest.approx(0.0)
    assert r["wick_imbalance"] == pytest.approx(-2.0 / 2.2)


def test_doji():
    """Tiny body, wicks both sides -> near-zero imbalance.
    o=10.00 h=10.50 l=9.50 c=10.02:
      body=0.02 upper=10.5-10.02=0.48 lower=10.0-9.5=0.5 range=1.0
      upper_frac=0.48 lower_frac=0.5 body_frac=0.02 imbalance=-0.02"""
    out = add_features(_frame([(10.00, 10.50, 9.50, 10.02)]))
    r = out.iloc[0]
    assert r["body"] == pytest.approx(0.02)
    assert r["upper_wick"] == pytest.approx(0.48)
    assert r["lower_wick"] == pytest.approx(0.5)
    assert r["upper_wick_frac"] == pytest.approx(0.48)
    assert r["lower_wick_frac"] == pytest.approx(0.5)
    assert r["body_frac"] == pytest.approx(0.02)
    assert r["wick_imbalance"] == pytest.approx(-0.02)


def test_fractions_sum_to_one():
    out = add_features(_frame([
        (10.0, 13.0, 10.0, 10.2),
        (10.0, 10.2, 8.0, 10.2),
        (10.0, 10.5, 9.5, 10.02),
        (9.0, 11.0, 8.5, 10.5),
    ]))
    s = out["upper_wick_frac"] + out["lower_wick_frac"] + out["body_frac"]
    np.testing.assert_allclose(s.to_numpy(), 1.0, atol=1e-12)


# ============================================================
# 2. Flat bar -> NaN fractions (documented guard).
# ============================================================


def test_flat_bar_fracs_nan():
    out = add_features(_frame([(10.0, 10.0, 10.0, 10.0)]))
    r = out.iloc[0]
    assert r["total_range"] == 0.0
    assert r["body"] == 0.0
    for col in ("upper_wick_frac", "lower_wick_frac", "body_frac", "wick_imbalance"):
        assert np.isnan(r[col]), col


# ============================================================
# 3. Down-candle wick assignment (body uses open/close, not high/low).
# ============================================================


def test_down_candle_wicks():
    """Bearish bar o=11 c=10 with wicks both ends.
    h=11.5 l=9.5: upper=11.5-max(11,10)=0.5 lower=min(11,10)-9.5=0.5 body=1 range=2.
    imbalance 0."""
    out = add_features(_frame([(11.0, 11.5, 9.5, 10.0)]))
    r = out.iloc[0]
    assert r["upper_wick"] == pytest.approx(0.5)
    assert r["lower_wick"] == pytest.approx(0.5)
    assert r["body"] == pytest.approx(1.0)
    assert r["wick_imbalance"] == pytest.approx(0.0)


# ============================================================
# 3b. Volatility-normalized wick SIZE (atr + rolling-rel).
# ============================================================


def test_wick_atr_normalization():
    """upper_wick_atr = upper_wick / atr_14. Shooting star upper_wick=2.8, atr=1.4 -> 2.0."""
    out = add_features(_frame([(10.0, 13.0, 10.0, 10.2)], atr=[1.4]))
    r = out.iloc[0]
    assert r["upper_wick_atr"] == pytest.approx(2.8 / 1.4)
    assert r["lower_wick_atr"] == pytest.approx(0.0)


def test_wick_atr_columns_absent_without_atr():
    """No atr_14 column -> the *_atr columns are simply not produced (rel still is)."""
    out = add_features(_frame([(10.0, 13.0, 10.0, 10.2)]))
    assert "upper_wick_atr" not in out.columns
    assert "upper_wick_rel" in out.columns


def test_wick_rel_uses_prior_window_excluding_current():
    """rel = wick / mean(total_wick over prior 20 bars). 20 bars total_wick=1,
    then bar 20 has upper_wick=2 -> upper_wick_rel = 2 / 1 = 2 (current excluded)."""
    rows = []
    # 20 bars each with total_wick = 1.0 (upper 0.5, lower 0.5, body 0, range 1).
    for _ in range(20):
        rows.append((10.0, 10.5, 9.5, 10.0))  # upper=0.5, lower=0.5
    # bar 20: big upper wick 2.0 (o=c=10 so body 0; high 12 -> upper 2; low 10 -> lower 0).
    rows.append((10.0, 12.0, 10.0, 10.0))
    out = add_features(_frame(rows))
    # first 20 rel values are NaN (rolling+shift warmup).
    assert out["upper_wick_rel"].iloc[:20].isna().all()
    # bar 20: prior-20 mean total_wick = 1.0 -> rel = 2.0 / 1.0.
    assert out["upper_wick_rel"].iloc[20] == pytest.approx(2.0)
    assert out["lower_wick_rel"].iloc[20] == pytest.approx(0.0)


def test_wick_rel_causal_denominator_excludes_self():
    """A giant wick must NOT deflate its own denominator (current bar excluded)."""
    rows = [(10.0, 10.5, 9.5, 10.0)] * 20 + [(10.0, 20.0, 10.0, 10.0)]
    out = add_features(_frame(rows))
    # denominator = 1.0 (prior bars only), so rel = 10 / 1 = 10, not diluted by the 10 itself.
    assert out["upper_wick_rel"].iloc[20] == pytest.approx(10.0)


# ============================================================
# 4. Causality: values at t unaffected by later bars.
# ============================================================


def test_no_lookahead_in_wick():
    rng = np.random.default_rng(4)
    n = 200
    o = 1.10 + np.cumsum(rng.normal(0, 0.0003, n))
    h = o + np.abs(rng.normal(0, 0.0004, n))
    l = o - np.abs(rng.normal(0, 0.0004, n))
    c = o + rng.normal(0, 0.0002, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"mid_open": o, "mid_high": np.maximum(h, np.maximum(o, c)),
                       "mid_low": np.minimum(l, np.minimum(o, c)), "mid_close": c}, index=idx)
    split = 100
    full = add_features(df)
    corrupted = df.copy()
    corrupted.iloc[split + 1 :, :] = 9.99
    corr = add_features(corrupted)
    new_cols = [x for x in full.columns if x not in df.columns]
    assert new_cols
    for col in new_cols:
        a = full[col].iloc[: split + 1].to_numpy()
        b = corr[col].iloc[: split + 1].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=col)
        mask = ~np.isnan(a)
        assert np.array_equal(a[mask], b[mask]), f"lookahead in {col}"
