"""Trust tests for intrabar TP/SL handling.

These exercise the new `level_policy` path. The engine's pessimistic
same-bar rule and exact fill arithmetic are the load-bearing pieces.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fxalgo.backtest.engine import simulate
from fxalgo.backtest.level_exits import (
    AtrBarrier,
    NoLevels,
)
from tests.backtest.conftest import make_engine_frame, make_signals


# A test-only policy that just returns fixed levels. Lets the tests state
# the exact TP and SL without going through ATR or band arithmetic.
class _FixedLevels:
    def __init__(self, tp: float | None, sl: float | None) -> None:
        self.tp = tp
        self.sl = sl

    def levels_at_entry(self, *, direction, entry_price, bar):
        return self.tp, self.sl


# ============================================================
# 1. Pessimistic same-bar rule: SL wins.
# ============================================================


def test_same_bar_tp_and_sl_resolves_as_stop():
    """Long position. The bar after entry has a range spanning both TP and SL.
    Engine must fill the STOP (the unfavorable outcome) and record exit_reason='intrabar_sl'.
    """
    # Bar 0: flat. Bar 1: enter long at open (mid=1.10, ask=1.1001).
    # Bar 2: range is [1.0980, 1.1100] -- spans both SL=1.0990 and TP=1.1050.
    # By the pessimistic rule, SL fires.
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1100]
    lows = [1.10, 1.10, 1.0980]
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=1.1050, sl=1.0990)

    result = simulate(
        features, signals, notional=1.0, commission=None, level_policy=policy
    )

    assert result.n_intrabar_sl == 1
    assert result.n_intrabar_tp == 0
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "intrabar_sl"
    # Fill = SL level - half_spread (we sold at the bid).
    expected_fill = 1.0990 - spread / 2.0
    assert trade["exit_price"] == pytest.approx(expected_fill)


# ============================================================
# 2. TP-only bar fills at TP.
# ============================================================


def test_tp_only_bar_fills_at_tp():
    """Bar's range hits TP but not SL: long exits at TP - half_spread."""
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1100]
    lows = [1.10, 1.10, 1.0995]  # > SL=1.0990
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=1.1050, sl=1.0990)

    result = simulate(
        features, signals, notional=1.0, commission=None, level_policy=policy
    )

    assert result.n_intrabar_tp == 1
    assert result.n_intrabar_sl == 0
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "intrabar_tp"
    expected_fill = 1.1050 - spread / 2.0
    assert trade["exit_price"] == pytest.approx(expected_fill)


# ============================================================
# 3. SL-only bar fills at SL.
# ============================================================


def test_sl_only_bar_fills_at_sl():
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1040]  # < TP=1.1050
    lows = [1.10, 1.10, 1.0980]
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=1.1050, sl=1.0990)

    result = simulate(
        features, signals, notional=1.0, commission=None, level_policy=policy
    )

    assert result.n_intrabar_sl == 1
    assert result.n_intrabar_tp == 0
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "intrabar_sl"
    expected_fill = 1.0990 - spread / 2.0
    assert trade["exit_price"] == pytest.approx(expected_fill)


# ============================================================
# 4. Short-side mirrors of TP and SL detection.
# ============================================================


def test_short_tp_only_triggers():
    """For a short, TP is BELOW entry and triggers when mid_low <= TP."""
    # Bar 1: enter short. Bar 2: low touches TP, high doesn't touch SL.
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1040]  # < SL=1.1080 -> no SL
    lows = [1.10, 1.10, 1.0945]  # <= TP=1.0950 -> TP triggers
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([-1, -1, -1], features.index)
    policy = _FixedLevels(tp=1.0950, sl=1.1080)

    result = simulate(
        features, signals, notional=1.0, commission=None, level_policy=policy
    )

    assert result.n_intrabar_tp == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    # Short exits buy at ask: fill = TP + half_spread.
    expected_fill = 1.0950 + spread / 2.0
    assert trade["exit_price"] == pytest.approx(expected_fill)


# ============================================================
# 5. Hand-calculated P&L on a complete TP trade.
# ============================================================


def test_hand_calculated_pnl_on_full_tp_trade():
    """Long entered at bar 1 (ask 1.1001), exited via TP at bar 2 (TP=1.1050).
    Exit fill = 1.1050 - half_spread = 1.1049.
    Gross PnL per unit = 1.1049 - 1.1001 = 0.0048.
    """
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1100]
    lows = [1.10, 1.10, 1.0998]
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=1.1050, sl=None)
    result = simulate(features, signals, notional=1.0, commission=None, level_policy=policy)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(1.1001)
    assert trade["exit_price"] == pytest.approx(1.1049)
    assert trade["pnl"] == pytest.approx(0.0048)
    # Spread cost: half-spread at entry (0.0001) + half-spread at exit (0.0001).
    assert result.cum_spread_cost == pytest.approx(0.0002)
    # Per-trade cost attribution: spread_cost is both legs; commission is 0 here.
    assert trade["spread_cost"] == pytest.approx(0.0002)
    assert trade["commission"] == pytest.approx(0.0)
    # pnl is net of spread; adding spread_cost back recovers the mid-to-mid move.
    assert trade["pnl"] + trade["spread_cost"] == pytest.approx(0.005)  # 1.105 - 1.10
    # Final equity = realized P&L (we exited intrabar at bar 2; cur_pos=0 after).
    assert result.equity.iloc[-1] == pytest.approx(0.0048)


def test_per_trade_costs_reconcile_with_cumulative_totals():
    """Across a multi-trade run with commission, the per-trade spread_cost and
    commission columns must sum to the BacktestResult cumulative totals."""
    n = 12
    mids = [1.10] * n
    highs = list(mids)
    lows = list(mids)
    highs[2] = 1.1100  # intrabar TP at bar 2
    highs[8] = 1.1100  # intrabar TP at bar 8 (after a fresh entry)
    spread = 0.0002
    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    # Enter, intrabar-exit, go flat, re-enter, intrabar-exit again.
    signals = make_signals([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], features.index)
    policy = _FixedLevels(tp=1.1050, sl=None)
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION, level_policy=policy
    )
    assert len(result.trades) == 2
    assert result.trades["spread_cost"].sum() == pytest.approx(result.cum_spread_cost)
    assert result.trades["commission"].sum() == pytest.approx(result.cum_commission)


# ============================================================
# 6. Hand-calculated P&L on a complete SL trade.
# ============================================================


def test_hand_calculated_pnl_on_full_sl_trade():
    """Long entered at bar 1 (ask 1.1001). SL=1.0990. Bar 2 low touches SL.
    Fill = 1.0990 - half_spread = 1.0989.
    PnL per unit = 1.0989 - 1.1001 = -0.0012.
    """
    mids = [1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.1005]  # nowhere near TP
    lows = [1.10, 1.10, 1.0985]
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=None, sl=1.0990)
    result = simulate(features, signals, notional=1.0, commission=None, level_policy=policy)

    trade = result.trades.iloc[0]
    assert trade["exit_price"] == pytest.approx(1.0989)
    assert trade["pnl"] == pytest.approx(-0.0012)
    assert result.cum_spread_cost == pytest.approx(0.0002)
    assert result.equity.iloc[-1] == pytest.approx(-0.0012)


# ============================================================
# 7. Suppression: after intrabar exit, re-entry waits for signal change.
# ============================================================


def test_no_reentry_until_signal_changes_after_intrabar_exit():
    """Strategy keeps emitting +1 for many bars. Intrabar TP fires at bar 2.
    Without suppression, the engine would re-open at bar 3 (target = signals[2] = +1).
    With suppression, position must stay flat until signal differs from +1.
    """
    n = 8
    mids = [1.10] * n
    highs = list(mids)
    lows = list(mids)
    # Make bar 2 hit TP at 1.1050.
    highs[2] = 1.1100
    # Then bar 6 the signal flips to 0.
    spread = 0.0002

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    # Signals: +1 for bars 0..5, then 0 onwards. target_position = signals shifted.
    signals = make_signals([1, 1, 1, 1, 1, 1, 0, 0], features.index)
    policy = _FixedLevels(tp=1.1050, sl=None)
    result = simulate(features, signals, notional=1.0, commission=None, level_policy=policy)

    # Only ONE trade should be recorded (the intrabar TP at bar 2).
    assert len(result.trades) == 1
    # Position must be 0 for bars 2..end (intrabar exit at 2, then suppressed).
    # Specifically, bars 3, 4, 5 should be flat despite signal saying +1.
    assert result.position.iloc[3] == 0
    assert result.position.iloc[4] == 0
    assert result.position.iloc[5] == 0


# ============================================================
# 8. No-lookahead: exit decision uses only the current bar's OHLC.
# ============================================================


def test_intrabar_exit_uses_only_current_bar():
    """Modify rows AFTER an intrabar exit and verify the exit's price/PnL
    don't change. The exit decision must depend only on bar t's data.
    """
    spread = 0.0002
    n = 6
    mids = [1.10] * n
    highs = list(mids)
    lows = list(mids)
    highs[2] = 1.1100  # TP triggers at bar 2

    features = make_engine_frame(mids, spread=spread, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1, 1, 0, 0], features.index)
    policy = _FixedLevels(tp=1.1050, sl=None)

    result_orig = simulate(features, signals, notional=1.0, commission=None, level_policy=policy)

    # Now corrupt bars 3..end with crazy values.
    features2 = features.copy()
    for col in ("open_bid", "open_ask", "mid_open", "mid_close", "mid_high", "mid_low"):
        features2.iloc[3:, features2.columns.get_loc(col)] = 99.0

    result_corrupt = simulate(features2, signals, notional=1.0, commission=None, level_policy=policy)

    # The trade closed at bar 2 should be IDENTICAL.
    t_orig = result_orig.trades.iloc[0]
    t_corr = result_corrupt.trades.iloc[0]
    assert t_orig["exit_price"] == pytest.approx(t_corr["exit_price"])
    assert t_orig["pnl"] == pytest.approx(t_corr["pnl"])


# ============================================================
# 9. Policies: NoLevels means no intrabar exits ever fire.
# ============================================================


def test_no_levels_policy_means_no_intrabar_exits():
    """Even with a NoLevels policy attached, no intrabar exit should fire."""
    mids = [1.10] * 5
    highs = [1.10, 1.10, 1.1100, 1.10, 1.10]
    lows = list(mids)
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1, 0, 0], features.index)
    result = simulate(features, signals, notional=1.0, commission=None, level_policy=NoLevels())
    assert result.n_intrabar_tp == 0
    assert result.n_intrabar_sl == 0
    # Exit is signal-driven at bar 4 (target[4] = signals[3] = 0).
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "signal"


# ============================================================
# 10. AtrBarrier: symmetric levels from the decision bar's ATR.
# ============================================================


def test_atr_barrier_levels_long_and_short():
    bar = pd.Series({"atr_decision": 0.001, "atr_14": 0.009})
    tp, sl = AtrBarrier(k=2.0).levels_at_entry(direction=1, entry_price=1.10, bar=bar)
    assert tp == pytest.approx(1.102)
    assert sl == pytest.approx(1.098)
    tp, sl = AtrBarrier(k=2.0).levels_at_entry(direction=-1, entry_price=1.10, bar=bar)
    assert tp == pytest.approx(1.098)
    assert sl == pytest.approx(1.102)


def test_atr_barrier_uses_the_decision_bar_column_not_atr_14():
    bar = pd.Series({"atr_decision": 0.001, "atr_14": 0.009})
    tp, _ = AtrBarrier(k=1.0).levels_at_entry(direction=1, entry_price=1.0, bar=bar)
    assert tp == pytest.approx(1.001)


def test_atr_barrier_returns_none_without_a_valid_atr():
    for atr in (float("nan"), 0.0, -1.0):
        bar = pd.Series({"atr_decision": atr})
        assert AtrBarrier().levels_at_entry(direction=1, entry_price=1.0, bar=bar) == (None, None)
    bar = pd.Series({"atr_decision": 0.001})
    assert AtrBarrier().levels_at_entry(direction=0, entry_price=1.0, bar=bar) == (None, None)


def test_atr_barrier_end_to_end_in_the_engine():
    """A long whose bar range reaches entry + 2 ATR exits intrabar at the TP."""
    mids = [1.1000, 1.1000, 1.1000, 1.1000, 1.1000]
    highs = [1.1000, 1.1000, 1.1000, 1.1030, 1.1000]
    features = make_engine_frame(mids, highs=highs, spread=0.0)
    features["atr_decision"] = 0.001
    signals = make_signals([1, 1, 1, 1, 1], features.index)
    result = simulate(features, signals, notional=1.0, commission=None, level_policy=AtrBarrier())
    t = result.trades.iloc[0]
    assert t["exit_reason"] == "intrabar_tp"
    assert t["exit_price"] == pytest.approx(1.1020)
