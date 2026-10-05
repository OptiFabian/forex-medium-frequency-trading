"""Tests for Donchian channel features (current-bar exclusion is the point)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features.donchian import DONCHIAN_WINDOWS, add_features, donchian
from tests.features.conftest import make_frame


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1min", tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=idx)


# ============================================================
# 1. Hand-verified values + current-bar exclusion.
# ============================================================


def test_hand_verified_channel_excludes_current_bar():
    """don_high_3[t] = max(high[t-3..t-1]); the current bar must NOT be in it."""
    highs = _series([1.0, 2.0, 3.0, 9.0, 4.0, 5.0])
    lows = _series([1.0, 2.0, 3.0, 9.0, 4.0, 5.0])
    ch_high, ch_low = donchian(highs, lows, window=3)

    # Rows 0-2: fewer than 3 PRIOR bars -> NaN.
    assert ch_high.iloc[:3].isna().all()
    # Row 3: max(1,2,3) = 3 -- crucially NOT 9 (the current bar's spike).
    assert ch_high.iloc[3] == 3.0
    # Row 4: max(2,3,9) = 9 -- the spike enters the channel one bar later.
    assert ch_high.iloc[4] == 9.0
    # Row 5: max(3,9,4) = 9.
    assert ch_high.iloc[5] == 9.0
    # Low channel mirrors: row 3 min(1,2,3)=1, row 4 min(2,3,9)=2, row 5 min(3,9,4)=3.
    assert ch_low.iloc[3] == 1.0
    assert ch_low.iloc[4] == 2.0
    assert ch_low.iloc[5] == 3.0


def test_new_extreme_never_visible_same_bar():
    """A bar that sets a new high can still CLOSE above its own channel --
    the degeneracy the exclusion is designed to avoid."""
    highs = _series([1.0, 1.0, 1.0, 1.0, 5.0])
    lows = highs.copy()
    ch_high, _ = donchian(highs, lows, window=4)
    # At the spike bar the channel is still the old extreme.
    assert ch_high.iloc[4] == 1.0
    assert highs.iloc[4] > ch_high.iloc[4]  # a close here IS a breakout


# ============================================================
# 2. Warmup NaN policy.
# ============================================================


@pytest.mark.parametrize("window", [1, 3, 10])
def test_warmup_is_exactly_window_rows(window):
    n = 30
    highs = _series(np.linspace(1.0, 2.0, n))
    ch_high, ch_low = donchian(highs, highs, window=window)
    assert ch_high.iloc[:window].isna().all()
    assert ch_high.iloc[window:].notna().all()
    assert ch_low.iloc[:window].isna().all()
    assert ch_low.iloc[window:].notna().all()


def test_invalid_window_rejected():
    s = _series([1.0, 2.0])
    with pytest.raises(ValueError, match="window"):
        donchian(s, s, window=0)


# ============================================================
# 3. add_features wiring.
# ============================================================


def test_add_features_adds_all_window_columns():
    df = make_frame(np.linspace(1.10, 1.12, 2000))
    out = add_features(df)
    for w in DONCHIAN_WINDOWS:
        assert f"don_high_{w}" in out.columns
        assert f"don_low_{w}" in out.columns
        # Monotone ramp with flat OHLC: channel high at t is close[t-1].
        expected = df["mid_close"].shift(1).iloc[w + 5]
        assert out[f"don_high_{w}"].iloc[w + 5] == expected


def test_channel_bounds_ordering():
    """don_low <= don_high wherever both are defined."""
    rng = np.random.default_rng(seed=11)
    closes = 1.10 + np.cumsum(rng.normal(0, 0.0001, size=500))
    df = make_frame(closes)
    out = add_features(df)
    for w in (60, 180):
        hi = out[f"don_high_{w}"]
        lo = out[f"don_low_{w}"]
        mask = hi.notna() & lo.notna()
        assert (lo[mask] <= hi[mask]).all()


# ============================================================
# 4. No lookahead (dedicated; the pipeline-wide test also covers this).
# ============================================================


def test_no_lookahead_in_donchian():
    rng = np.random.default_rng(seed=5)
    closes = 1.10 + np.cumsum(rng.normal(0, 0.0001, size=300))
    df = make_frame(closes)
    split = 150

    feats_full = add_features(df)
    corrupted = df.copy()
    corrupted.iloc[split + 1 :, :] = 9.99
    feats_corrupted = add_features(corrupted)

    new_cols = [c for c in feats_full.columns if c.startswith("don_")]
    assert new_cols
    for col in new_cols:
        a = feats_full[col].iloc[: split + 1].to_numpy()
        b = feats_corrupted[col].iloc[: split + 1].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        mask = ~np.isnan(a)
        assert np.array_equal(a[mask], b[mask]), f"lookahead in {col}"
