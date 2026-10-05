"""Tests for fxalgo.features.bollinger_utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import volatility
from fxalgo.features.bollinger_utils import rescale_bollinger
from fxalgo.strategies.bollinger_reversion import BollingerReversion
from tests.features.conftest import make_frame


def _frame_with_bands(closes):
    """Build a feature-style frame with proper bb_* columns from synthetic closes."""
    df = make_frame(closes)
    return volatility.add_features(df)


def test_rescale_at_k2_matches_originals():
    """Rescaling to k=2 must reproduce the original bb_upper/bb_lower exactly."""
    rng = np.random.default_rng(seed=42)
    closes = 1.10 + np.cumsum(rng.normal(0.0, 0.0001, size=300))
    df = _frame_with_bands(closes)

    rescale_bollinger(df, k=2.0, upper_name="bb_upper_k2", lower_name="bb_lower_k2")

    # Drop the warmup NaN region and compare.
    valid = df.dropna(subset=["bb_upper", "bb_lower", "bb_upper_k2", "bb_lower_k2"])
    np.testing.assert_allclose(
        valid["bb_upper_k2"].to_numpy(),
        valid["bb_upper"].to_numpy(),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        valid["bb_lower_k2"].to_numpy(),
        valid["bb_lower"].to_numpy(),
        atol=1e-14,
    )


def test_rescale_negative_or_zero_k_rejected():
    df = _frame_with_bands(np.full(100, 1.10))
    with pytest.raises(ValueError, match="k must be positive"):
        rescale_bollinger(df, k=0, upper_name="a", lower_name="b")
    with pytest.raises(ValueError, match="k must be positive"):
        rescale_bollinger(df, k=-1.0, upper_name="a", lower_name="b")


def test_widening_k_reduces_or_holds_equal_entry_signals():
    """For a fixed series, widening k (more extreme entry threshold) must
    produce <= as many ENTRIES (0 -> non-zero transitions) as a narrower k."""
    rng = np.random.default_rng(seed=11)
    n = 5000
    closes = 1.10 + np.cumsum(rng.normal(0.0, 0.00005, size=n))
    df = _frame_with_bands(closes)

    entries_by_k: list[tuple[float, int]] = []
    for k in (2.0, 2.5, 3.0, 4.0):
        df_k = df.copy()
        rescale_bollinger(
            df_k, k=k, upper_name=f"bb_upper_k{k}", lower_name=f"bb_lower_k{k}"
        )
        strategy = BollingerReversion(
            upper_col=f"bb_upper_k{k}",
            lower_col=f"bb_lower_k{k}",
            middle_col="bb_middle",
        )
        s = strategy.generate_signals(df_k)
        # Entry = transition from 0 to non-zero.
        prev = pd.Series(np.concatenate([[0], s.to_numpy()[:-1]]), index=s.index)
        entries = int(((prev == 0) & (s != 0)).sum())
        entries_by_k.append((k, entries))

    # Monotone non-increasing in k.
    counts = [c for _, c in entries_by_k]
    for i in range(len(counts) - 1):
        assert counts[i] >= counts[i + 1], (
            f"entries not monotone non-increasing across k: {entries_by_k}"
        )
    # And we should actually have SOME entries at k=2 on this random walk
    # so the test isn't vacuous.
    assert counts[0] > 0


def test_rescale_with_custom_source_columns():
    """upper_col / middle_col args let us point at differently-named source columns."""
    closes = np.full(50, 1.10)
    df = _frame_with_bands(closes)
    df = df.rename(columns={"bb_upper": "custom_u", "bb_middle": "custom_m"})
    rescale_bollinger(
        df,
        k=2.5,
        upper_name="out_u",
        lower_name="out_l",
        upper_col="custom_u",
        middle_col="custom_m",
    )
    assert "out_u" in df.columns
    assert "out_l" in df.columns
