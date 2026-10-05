"""Tests for the Bollinger close-at-mean three-state strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.strategies.bollinger_reversion import BollingerReversion


def _frame(mid, upper, middle, lower):
    n = len(mid)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "mid_close": mid,
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
        },
        index=idx,
    )


def _const_bands(mid, upper=1.10, middle=1.05, lower=1.00):
    n = len(mid)
    return _frame(mid, [upper] * n, [middle] * n, [lower] * n)


# ============================================================
# 1. Canonical state-machine walk.
# ============================================================


def test_state_machine_walk_through_all_transitions():
    """Lower touch -> long; cross middle -> flat; upper touch -> short; cross middle -> flat.
    Then assert it STAYS FLAT in the middle without a fresh touch.
    """
    #            0     1     2     3     4     5     6     7     8     9
    #          start lower -> middle  ?   upper -> middle  ?    ?
    mids = [1.05, 1.00, 1.02, 1.05, 1.08, 1.10, 1.07, 1.05, 1.03, 1.05]
    expected = [0, 1, 1, 0, 0, -1, -1, 0, 0, 0]
    df = _const_bands(mids)
    s = BollingerReversion().generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), expected)


def test_must_be_flat_to_enter_blocks_re_entry_without_fresh_touch():
    """After exiting long at the middle, the strategy must NOT re-enter long
    just because price dips toward (but not to) the lower band again."""
    mids = [
        1.05,  # flat
        1.00,  # touch lower -> long
        1.03,  # hold long (below middle, above lower)
        1.05,  # cross middle -> flat
        1.03,  # back below middle, but NOT at lower; must stay flat
        1.02,  # still not at lower; flat
        1.01,  # still not at lower; flat
    ]
    expected = [0, 1, 1, 0, 0, 0, 0]
    df = _const_bands(mids)
    s = BollingerReversion().generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), expected)


# ============================================================
# 2. Overshoot: jump from below mean past upper band in one bar.
# ============================================================


def test_overshoot_exits_to_flat_not_short_in_same_bar():
    """Long position; next bar's mid jumps past the upper band. The state
    machine must transition to FLAT (long-exit fires) and NOT directly to SHORT.
    """
    #          flat lower flatten? 1.12=above upper
    mids = [1.05, 1.00, 1.12]
    expected = [0, 1, 0]  # bar 2: was long, now flat (NOT -1)
    df = _const_bands(mids)
    s = BollingerReversion().generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), expected)


def test_overshoot_then_next_bar_upper_touch_triggers_short():
    """After the overshoot exit-to-flat, a SUBSEQUENT bar at the upper band
    can trigger short -- demonstrating the 'fresh touch' requirement."""
    mids = [1.05, 1.00, 1.12, 1.11]  # bar 3: still at/above upper, flat -> short
    expected = [0, 1, 0, -1]
    df = _const_bands(mids)
    s = BollingerReversion().generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), expected)


# ============================================================
# 3. Warmup: NaN bands keep state flat.
# ============================================================


def test_nan_bands_keep_state_flat():
    """Rows where any band is NaN must yield a flat signal regardless of price."""
    mids = [1.00, 1.10, 1.00, 1.10]
    df = _frame(
        mid=mids,
        upper=[np.nan, np.nan, 1.10, 1.10],
        middle=[np.nan, np.nan, 1.05, 1.05],
        lower=[np.nan, np.nan, 1.00, 1.00],
    )
    s = BollingerReversion().generate_signals(df)
    # First two rows are NaN-band warmup -> 0. Bar 2: lower touch -> long.
    # Bar 3: above middle -> flat.
    np.testing.assert_array_equal(s.to_numpy(), [0, 0, 1, 0])


# ============================================================
# 4. No lookahead: state at row T unchanged if rows > T are modified.
# ============================================================


def test_no_lookahead_in_stateful_signal():
    rng = np.random.default_rng(seed=7)
    n = 500
    base = 1.10 + np.cumsum(rng.normal(0.0, 0.0002, size=n))
    bb_mid = pd.Series(base).rolling(20, min_periods=20).mean().to_numpy()
    bb_std = pd.Series(base).rolling(20, min_periods=20).std().to_numpy()
    bb_up = bb_mid + 2 * bb_std
    bb_lo = bb_mid - 2 * bb_std
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df_orig = pd.DataFrame(
        {"mid_close": base, "bb_upper": bb_up, "bb_middle": bb_mid, "bb_lower": bb_lo},
        index=idx,
    )

    sig_orig = BollingerReversion().generate_signals(df_orig)

    for T in (100, 250, 400):
        df_corr = df_orig.copy()
        # Wreck future rows with extreme values.
        df_corr.iloc[T + 1 :, df_corr.columns.get_loc("mid_close")] = 99.0
        df_corr.iloc[T + 1 :, df_corr.columns.get_loc("bb_upper")] = 99.0
        df_corr.iloc[T + 1 :, df_corr.columns.get_loc("bb_middle")] = 99.0
        df_corr.iloc[T + 1 :, df_corr.columns.get_loc("bb_lower")] = 99.0
        sig_corr = BollingerReversion().generate_signals(df_corr)
        np.testing.assert_array_equal(
            sig_orig.iloc[: T + 1].to_numpy(),
            sig_corr.iloc[: T + 1].to_numpy(),
            err_msg=f"lookahead detected: signals at rows [0..{T}] changed",
        )


# ============================================================
# 5. Missing column error.
# ============================================================


def test_missing_column_raises():
    df = _const_bands([1.05, 1.06]).drop(columns=["bb_middle"])
    with pytest.raises(KeyError, match="bb_middle"):
        BollingerReversion().generate_signals(df)


def test_signal_dtype_and_alignment():
    df = _const_bands([1.05, 1.00, 1.05])
    s = BollingerReversion().generate_signals(df)
    assert s.dtype == np.int8
    assert s.index.equals(df.index)
