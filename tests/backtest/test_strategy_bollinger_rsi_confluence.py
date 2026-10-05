"""Tests for the Bollinger + RSI confluence strategy and its toggles.

Confluence requires BOTH a band touch AND RSI confirmation to enter. The
RSI, daily-regime and cost gates are independent toggles; with all off the
class must reproduce BollingerReversion exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.backtest.costs import DEFAULT_COMMISSION
from fxalgo.strategies.bollinger_reversion import BollingerReversion
from fxalgo.strategies.bollinger_rsi_confluence import BollingerRsiConfluence

N = 100_000.0
SPREAD = 0.0001


def _frame(mid, upper, middle, lower, rsi=None, spread=SPREAD):
    n = len(mid)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    data = {
        "mid_close": mid,
        "bb_upper": upper,
        "bb_middle": middle,
        "bb_lower": lower,
        "spread_close": [spread] * n,
    }
    if rsi is not None:
        data["rsi_14"] = rsi
    return pd.DataFrame(data, index=idx)


def _const(mid, rsi, upper=1.10, middle=1.05, lower=1.00):
    n = len(mid)
    return _frame(mid, [upper] * n, [middle] * n, [lower] * n, rsi=rsi)


# ============================================================
# 1. Confluence requires BOTH band touch AND RSI confirmation.
# ============================================================


def test_long_needs_band_and_rsi():
    """Lower-band touch with RSI > 30 does NOT enter; once RSI <= 30 it does
    (RSI may confirm a bar or two into the touch -- not locked out)."""
    mids = [1.05, 1.00, 1.00, 1.02, 1.05]
    rsi = [50, 35, 25, 40, 50]
    s = BollingerRsiConfluence(use_rsi=True, use_filter=False).generate_signals(_const(mids, rsi))
    np.testing.assert_array_equal(s.to_numpy(), [0, 0, 1, 1, 0])


def test_short_needs_band_and_rsi():
    mids = [1.05, 1.10, 1.10, 1.08, 1.05]
    rsi = [50, 65, 75, 60, 50]
    s = BollingerRsiConfluence(use_rsi=True, use_filter=False).generate_signals(_const(mids, rsi))
    np.testing.assert_array_equal(s.to_numpy(), [0, 0, -1, -1, 0])


def test_rsi_off_enters_on_band_touch_alone():
    """Same series; with RSI off the band touch alone triggers entry at bar 1."""
    mids = [1.05, 1.00, 1.00, 1.02, 1.05]
    rsi = [50, 35, 25, 40, 50]
    s = BollingerRsiConfluence(use_rsi=False, use_filter=False).generate_signals(_const(mids, rsi))
    np.testing.assert_array_equal(s.to_numpy(), [0, 1, 1, 1, 0])


# ============================================================
# 2. Both gates together: RSI confirms but cost filter decides.
# ============================================================


def test_both_gates_filter_skips_low_vol_even_with_rsi():
    """RSI confirms (25 <= 30) but band-to-mean ($20) < 2x cost ($28) -> skip."""
    df = _frame(
        mid=[1.0002, 1.0000],
        upper=[1.0004, 1.0004],
        middle=[1.0002, 1.0002],
        lower=[1.0000, 1.0000],
        rsi=[50, 25],
    )
    strat = BollingerRsiConfluence(
        use_rsi=True, use_filter=True, cost_multiple=2.0, notional=N, commission=DEFAULT_COMMISSION
    )
    s = strat.generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), [0, 0])
    assert strat.n_entries_skipped == 1
    assert strat.n_entries_taken == 0


def test_both_gates_enter_when_rsi_and_filter_pass():
    """RSI confirms AND band-to-mean ($50) >= 2x cost ($28) -> enter."""
    df = _frame(
        mid=[1.0005, 1.0000],
        upper=[1.0010, 1.0010],
        middle=[1.0005, 1.0005],
        lower=[1.0000, 1.0000],
        rsi=[50, 25],
    )
    strat = BollingerRsiConfluence(
        use_rsi=True, use_filter=True, cost_multiple=2.0, notional=N, commission=DEFAULT_COMMISSION
    )
    s = strat.generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), [0, 1])
    assert strat.n_entries_taken == 1


def test_both_gates_rsi_block_is_not_a_filter_skip():
    """High-vol (filter would pass) but RSI not confirming -> 'wait', NOT a
    filter skip (so n_entries_skipped stays 0)."""
    df = _frame(
        mid=[1.0005, 1.0000],
        upper=[1.0010, 1.0010],
        middle=[1.0005, 1.0005],
        lower=[1.0000, 1.0000],
        rsi=[50, 40],  # 40 > 30 -> RSI does not confirm
    )
    strat = BollingerRsiConfluence(
        use_rsi=True, use_filter=True, cost_multiple=2.0, notional=N, commission=DEFAULT_COMMISSION
    )
    s = strat.generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), [0, 0])
    assert strat.n_entries_skipped == 0
    assert strat.n_entries_taken == 0


# ============================================================
# 2b. Daily-regime entry gate.
# ============================================================


def _frame_with_col(mid, rsi, extra_col, extra_vals, upper=1.10, middle=1.05, lower=1.00):
    n = len(mid)
    df = _frame(mid, [upper] * n, [middle] * n, [lower] * n, rsi=rsi)
    df[extra_col] = extra_vals
    return df


def test_regime_gate_blocks_when_higher_tf_is_flat():
    """A band touch while the regime column is 0 (higher timeframe sitting at
    its own mean) is skipped."""
    df = _frame_with_col(
        mid=[1.05, 1.00], rsi=[50, 25], extra_col="regime", extra_vals=[0, 0]
    )
    strat = BollingerRsiConfluence(
        use_rsi=False, use_filter=False, use_regime=True, regime_col="regime"
    )
    s = strat.generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), [0, 0])
    assert strat.n_entries_taken == 0
    assert strat.n_entries_skipped == 1


@pytest.mark.parametrize("regime_value", [1, -1])
def test_regime_gate_admits_either_direction(regime_value):
    """The gate deliberately ignores the SIGN: a LONG 15-min entry is admitted
    by a SHORT higher-TF regime just as readily as by a long one."""
    df = _frame_with_col(
        mid=[1.05, 1.00], rsi=[50, 25],
        extra_col="regime", extra_vals=[regime_value, regime_value],
    )
    strat = BollingerRsiConfluence(
        use_rsi=False, use_filter=False, use_regime=True, regime_col="regime"
    )
    s = strat.generate_signals(df)
    np.testing.assert_array_equal(s.to_numpy(), [0, 1])
    assert strat.n_entries_taken == 1


def test_regime_gate_off_by_default_and_ignores_column():
    df = _const(mid=[1.05, 1.00], rsi=[50, 25])  # no regime column
    strat = BollingerRsiConfluence(use_rsi=True, use_filter=False)
    s = strat.generate_signals(df)  # must not raise
    np.testing.assert_array_equal(s.to_numpy(), [0, 1])


def test_regime_gate_missing_column_raises():
    df = _const(mid=[1.05, 1.00], rsi=[50, 25])
    with pytest.raises(KeyError, match="regime"):
        BollingerRsiConfluence(use_regime=True).generate_signals(df)


def test_regime_gate_enforced_at_decision_bar():
    """Every entry the regime-gated strategy makes has a non-zero regime flag
    at the DECISION bar."""
    df = _random_bands(seed=31, n=800)
    rng = np.random.default_rng(31)
    df["regime"] = rng.choice([-1, 0, 1], size=len(df))
    strat = BollingerRsiConfluence(
        use_rsi=False, use_filter=False, use_regime=True, regime_col="regime"
    )
    sig = strat.generate_signals(df).to_numpy()
    reg = df["regime"].to_numpy()
    entries = np.where((sig != 0) & (np.r_[0, sig[:-1]] == 0))[0]
    assert len(entries) > 0
    for t in entries:
        assert reg[t] != 0, f"entry at bar {t} had a flat regime flag"


def test_regime_gate_all_directional_reproduces_ungated():
    """A regime column that is never flat must leave the strategy unchanged."""
    df = _random_bands(seed=33, n=600)
    df["regime"] = 1
    plain = BollingerRsiConfluence(use_rsi=False, use_filter=False)
    gated = BollingerRsiConfluence(
        use_rsi=False, use_filter=False, use_regime=True, regime_col="regime"
    )
    np.testing.assert_array_equal(
        plain.generate_signals(df).to_numpy(), gated.generate_signals(df).to_numpy()
    )


# ============================================================
# 3. Toggle independence: reproduce the reference strategies exactly.
# ============================================================


def _random_bands(seed, n=600):
    rng = np.random.default_rng(seed)
    base = 1.10 + np.cumsum(rng.normal(0.0, 0.0002, size=n))
    s = pd.Series(base)
    mid = s.rolling(20, min_periods=20).mean().to_numpy()
    std = s.rolling(20, min_periods=20).std().to_numpy()
    rsi = 50.0 + 50.0 * np.tanh(pd.Series(base).diff().fillna(0).rolling(14).mean().to_numpy() * 5e3)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "mid_close": base,
            "bb_upper": mid + 2 * std,
            "bb_middle": mid,
            "bb_lower": mid - 2 * std,
            "rsi_14": rsi,
            "spread_close": [SPREAD] * n,
        },
        index=idx,
    )


def test_both_off_reproduces_plain_bollinger():
    df = _random_bands(seed=1)
    a = BollingerRsiConfluence(use_rsi=False, use_filter=False).generate_signals(df)
    b = BollingerReversion().generate_signals(df)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_rsi_only_is_subset_of_plain_entries():
    """RSI confluence can only REMOVE entries vs plain Bollinger (a stricter
    gate), so its taken count must not exceed the plain strategy's."""
    df = _random_bands(seed=3)
    plain = BollingerRsiConfluence(use_rsi=False, use_filter=False)
    rsi_only = BollingerRsiConfluence(use_rsi=True, use_filter=False)
    plain.generate_signals(df)
    rsi_only.generate_signals(df)
    assert rsi_only.n_entries_taken <= plain.n_entries_taken


# ============================================================
# 4. Warmup + missing columns.
# ============================================================


def test_nan_rsi_keeps_flat():
    df = _const([1.00, 1.00, 1.00], rsi=[np.nan, np.nan, 25])
    s = BollingerRsiConfluence(use_rsi=True, use_filter=False).generate_signals(df)
    # First two bars: RSI NaN -> warmup flat. Bar 2: touch + rsi 25 -> long.
    np.testing.assert_array_equal(s.to_numpy(), [0, 0, 1])


def test_missing_rsi_column_raises():
    df = _frame([1.05, 1.00], [1.10, 1.10], [1.05, 1.05], [1.00, 1.00])  # no rsi_14
    with pytest.raises(KeyError, match="rsi_14"):
        BollingerRsiConfluence(use_rsi=True).generate_signals(df)


# ============================================================
# 5. No lookahead (both gates on).
# ============================================================


def test_no_lookahead_confluence():
    df_orig = _random_bands(seed=9, n=500)
    strat = lambda: BollingerRsiConfluence(  # noqa: E731
        use_rsi=True, use_filter=True, notional=N, commission=DEFAULT_COMMISSION
    )
    sig_orig = strat().generate_signals(df_orig)
    for T in (100, 250, 400):
        df_corr = df_orig.copy()
        for col in ("mid_close", "bb_upper", "bb_middle", "bb_lower", "rsi_14", "spread_close"):
            df_corr.iloc[T + 1 :, df_corr.columns.get_loc(col)] = 99.0
        sig_corr = strat().generate_signals(df_corr)
        np.testing.assert_array_equal(
            sig_orig.iloc[: T + 1].to_numpy(),
            sig_corr.iloc[: T + 1].to_numpy(),
            err_msg=f"lookahead at T={T}",
        )
