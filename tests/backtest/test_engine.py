"""Trust tests for the backtest engine.

These tests are deliberately arithmetic-heavy. Each one hand-calculates
the expected P&L and asserts the engine matches to the last pip. If the
engine ever drifts from these, something fundamental is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.backtest.engine import simulate
from tests.backtest.conftest import make_engine_frame, make_signals


# ============================================================
# 1. Round trip on a perfectly flat market loses exactly the spread.
# ============================================================


def test_flat_market_round_trip_loses_exactly_the_spread():
    """Mid stays at 1.1000 forever. 2 pip spread.
    bid = 1.0999, ask = 1.1001.

    Plan: 5 bars; signal at t=1 -> +1 (long), signal at t=3 -> 0 (close).
    Execution is shifted by +1 bar:
      - target[2] = +1 -> at open of bar 2: BUY at ASK = 1.1001
      - target[4] = 0  -> at open of bar 4: SELL at BID = 1.0999
    Round-trip P&L = sell - buy = 1.0999 - 1.1001 = -0.0002 == -spread.
    Spread cost (sum of half-spreads): 0.0001 (entry) + 0.0001 (exit) = 0.0002.
    """
    spread = 0.0002
    features = make_engine_frame([1.1000] * 5, spread=spread)
    signals = make_signals([0, 1, 1, 0, 0], features.index)

    result = simulate(features, signals, warmup=0)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == 1
    assert trade["entry_price"] == pytest.approx(1.1001)
    assert trade["exit_price"] == pytest.approx(1.0999)
    assert trade["pnl"] == pytest.approx(-spread)
    assert result.cum_spread_cost == pytest.approx(spread)
    assert result.equity.iloc[-1] == pytest.approx(-spread)


# ============================================================
# 2. Long -> short flip pays the spread on both legs.
# ============================================================


def test_long_to_short_flip_pays_spread_on_both_legs():
    """5 bars, mid flat at 1.1000, spread 0.0002.

    Plan: t=1 signal +1 (long), t=3 signal -1 (flip to short), t=4 signal 0.
    Executions (shift by +1):
      - target[2] = +1: BUY at ASK 1.1001
      - target[4] = -1: SELL closes long at BID 1.0999, then SELL opens short at BID 1.0999
      - (we never see the short close, it's still open at end)

    Realized P&L (closed long only) = 1.0999 - 1.1001 = -0.0002.
    Unrealized at end (short opened at 1.0999, mid_close 1.1000):
      P&L = entry - mid = 1.0999 - 1.1000 = -0.0001.
    Total equity at end = -0.0002 + (-0.0001) = -0.0003.
    Spread cost:
      Long entry: 0.0001 (paid ask - mid)
      Long exit:  0.0001 (received bid; mid - bid)
      Short entry: 0.0001 (sold at bid; mid - bid)
      Total spread cost = 0.0003.
    """
    spread = 0.0002
    features = make_engine_frame([1.1000] * 5, spread=spread)
    signals = make_signals([0, 1, 1, -1, 0], features.index)

    result = simulate(features, signals, warmup=0)

    # One closed trade (the long).
    assert len(result.trades) == 1
    closed = result.trades.iloc[0]
    assert closed["direction"] == 1
    assert closed["pnl"] == pytest.approx(-0.0002)

    # Equity reflects realized (-0.0002) + unrealized short (-0.0001).
    assert result.equity.iloc[-1] == pytest.approx(-0.0003)

    # Spread cost = 3 half-spreads = 1.5 * spread = 0.0003.
    assert result.cum_spread_cost == pytest.approx(0.0003)

    # Position at end is short.
    assert result.position.iloc[-1] == -1


# ============================================================
# 3. Execution uses bar t+1, not bar t.
# ============================================================


def test_execution_is_at_t_plus_1_not_t():
    """Construct a case where t vs t+1 fills give clearly different P&L.

    7 bars, spread 0. Mids:
      bar 0: 1.10
      bar 1: 1.10
      bar 2: 1.10       <- signal triggers here (+1)
      bar 3: 1.30       <- if t-execution, we entered at the 1.10 open; if t+1, we enter here at 1.30
      bar 4: 1.10
      bar 5: 1.10
      bar 6: 1.10       <- signal=0 here, so close at t+1 (bar 7? no, last)
    For simplicity: 7 bars, signal +1 only at bar 2, 0 elsewhere.

    Under t+1 execution:
      target[3] = +1 -> BUY at open of bar 3 at mid (spread=0) = 1.30
      target[r] = 0 for r >= 4 (signals are 0 from bar 3 onwards)
      target[4] = signals[3] = 0 -> SELL at open of bar 4 = 1.10
    Round-trip P&L (t+1 execution) = 1.10 - 1.30 = -0.20.

    Under t-execution (the WRONG behaviour we're guarding against),
    the entry would be at bar 2's open = 1.10, exit at bar 3's open = 1.30,
    P&L = +0.20. The sign flip is the smoking gun.
    """
    mids = [1.10, 1.10, 1.10, 1.30, 1.10, 1.10, 1.10]
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([0, 0, 1, 0, 0, 0, 0], features.index)

    result = simulate(features, signals, warmup=0)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(1.30), (
        "Entry must be at bar 3's open (1.30), not bar 2's open (1.10). "
        "If you see 1.10 here, the engine is executing at bar t — lookahead bug."
    )
    assert trade["exit_price"] == pytest.approx(1.10)
    assert trade["pnl"] == pytest.approx(-0.20)


# ============================================================
# 4. Hand-arithmetic on a slightly more elaborate sequence.
# ============================================================


def test_hand_arithmetic_long_then_short_complete():
    """Six bars with a known long + short + close. Hand-verify every fill.

    Mids: 1.10, 1.10, 1.12, 1.12, 1.08, 1.08
    Spread: 0.0002 (so bid = mid - 0.0001, ask = mid + 0.0001).

    Signals: [0, +1, +1, -1, -1, 0]
    Targets (shift +1, first bar zeroed): [0, 0, +1, +1, -1, -1]

    Bar 2: open of bar 2: BUY long at ASK = 1.12 + 0.0001 = 1.1201
    Bar 4: open of bar 4: target flips +1 -> -1. SELL long at BID = 1.08 - 0.0001 = 1.0799.
                          Then SELL open short at BID = 1.0799.
    No further changes; the short stays open through bar 5.

    Closed long P&L = 1.0799 - 1.1201 = -0.0402
    Unrealized short at end: entry 1.0799, mid_close[5] = 1.08
                              P&L = entry - mid = 1.0799 - 1.08 = -0.0001
    Total equity at end = -0.0402 + (-0.0001) = -0.0403

    Spread cost:
      long entry (bar 2): ask - mid = 0.0001
      long exit  (bar 4): mid - bid = 0.0001
      short open (bar 4): mid - bid = 0.0001
      Total = 0.0003
    """
    mids = [1.10, 1.10, 1.12, 1.12, 1.08, 1.08]
    features = make_engine_frame(mids, spread=0.0002)
    signals = make_signals([0, 1, 1, -1, -1, 0], features.index)

    result = simulate(features, signals, warmup=0)

    assert len(result.trades) == 1
    closed = result.trades.iloc[0]
    assert closed["direction"] == 1
    assert closed["entry_price"] == pytest.approx(1.1201)
    assert closed["exit_price"] == pytest.approx(1.0799)
    assert closed["pnl"] == pytest.approx(-0.0402)
    assert closed["bars_held"] == 2

    assert result.equity.iloc[-1] == pytest.approx(-0.0403)
    assert result.cum_spread_cost == pytest.approx(0.0003)
    assert result.position.iloc[-1] == -1


# ============================================================
# 5. Warmup bars suppress trading.
# ============================================================


def test_warmup_forces_flat_and_no_trades():
    """With warmup=3, the first 3 bars must be flat regardless of signals."""
    mids = [1.10, 1.20, 1.30, 1.40, 1.50, 1.40, 1.30]
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([1, -1, 1, -1, 1, -1, 1], features.index)

    result = simulate(features, signals, warmup=3)

    # Positions during the warmup region must be 0.
    np.testing.assert_array_equal(result.position.iloc[:3].to_numpy(), 0)


# ============================================================
# 6. Constant signal with no flips and one entry generates one open trade
#    that is unrealized to the end.
# ============================================================


def test_single_open_position_never_closed():
    """Signal = +1 from bar 1 onward, never flips. Long stays open."""
    mids = [1.10, 1.12, 1.14, 1.16]
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([1, 1, 1, 1], features.index)

    result = simulate(features, signals, warmup=0)

    # No CLOSED trades.
    assert len(result.trades) == 0

    # target[1] = signals[0] = +1 -> BUY at open of bar 1 = 1.12
    # Equity at end: entry 1.12, mid_close 1.16, PnL = +0.04.
    assert result.equity.iloc[-1] == pytest.approx(0.04)
    assert result.position.iloc[-1] == 1


# ============================================================
# 7. Notional scaling: same scenario at N=1 vs N=100,000 -> P&L scales linearly.
# ============================================================


def test_notional_scales_pnl_linearly():
    """Same flat-market round-trip at notional=1 and notional=100_000.
    P&L should be exactly 100_000x. No commission in either run.
    """
    spread = 0.0002
    features = make_engine_frame([1.1000] * 5, spread=spread)
    signals = make_signals([0, 1, 1, 0, 0], features.index)

    small = simulate(features, signals, notional=1.0, commission=None)
    big = simulate(features, signals, notional=100_000.0, commission=None)

    assert small.equity.iloc[-1] == pytest.approx(-spread)
    assert big.equity.iloc[-1] == pytest.approx(-spread * 100_000)
    assert big.cum_spread_cost == pytest.approx(small.cum_spread_cost * 100_000)
    assert big.cum_commission == 0.0
    assert small.cum_commission == 0.0


# ============================================================
# 8. Commission charged per order leg; flip pays TWO commissions.
# ============================================================


def test_commission_charged_per_order_leg():
    """One round trip = 2 commissions (open + close)."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    features = make_engine_frame([1.1000] * 5, spread=0.0002)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    # 2 order legs (open + close).
    assert result.n_order_legs == 2
    # Each fill is ~$110,000 trade value -> 0.00002 * 110,000 = $2.20.
    # The exact fill prices are 1.1001 (ask) and 1.0999 (bid).
    expected = (
        DEFAULT_COMMISSION.commission_for(1.1001, 100_000)
        + DEFAULT_COMMISSION.commission_for(1.0999, 100_000)
    )
    assert result.cum_commission == pytest.approx(expected)


def test_long_to_short_flip_pays_commission_on_both_legs():
    """0 -> +1 -> -1 -> 0 generates 4 order legs total."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    # 7 bars. signals: [0, 1, 1, -1, -1, 0, 0]
    # target (shift +1, first=0): [0, 0, 1, 1, -1, -1, 0]
    # Position changes at bars 2 (open long), 4 (flip = close+open = 2 legs),
    # and 6 (close short). 1 + 2 + 1 = 4 order legs.
    features = make_engine_frame([1.1000] * 7, spread=0.0002)
    signals = make_signals([0, 1, 1, -1, -1, 0, 0], features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    assert result.n_position_changes == 3
    assert result.n_order_legs == 4
    # Commission should be charged 4 times (one per leg).
    one_leg_at_bid = DEFAULT_COMMISSION.commission_for(1.0999, 100_000)
    one_leg_at_ask = DEFAULT_COMMISSION.commission_for(1.1001, 100_000)
    # Bar 2 (open long): ask. Bar 4 (close long + open short): bid + bid. Bar 6 (close short): ask.
    expected = one_leg_at_ask + one_leg_at_bid + one_leg_at_bid + one_leg_at_ask
    assert result.cum_commission == pytest.approx(expected, rel=1e-9)


# ============================================================
# 9. Accounting reconciliation: total_pnl == gross_price_pnl - spread - commission.
# ============================================================


def test_accounting_reconciles_in_flat_market():
    """In a perfectly flat market, gross_price_pnl = 0, so total_pnl == -(spread + commission)."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    features = make_engine_frame([1.1000] * 5, spread=0.0002)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    expected = -(result.cum_spread_cost + result.cum_commission)
    assert result.equity.iloc[-1] == pytest.approx(expected)


def test_accounting_reconciles_when_market_moves():
    """gross_price_pnl computed from synthetic prices must match total + costs."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    # Bars 0-1: 1.10. Bar 2 onwards: 1.12. Long held bar 2..3, then closed.
    mids = [1.10, 1.10, 1.12, 1.12, 1.12]
    notional = 100_000.0
    features = make_engine_frame(mids, spread=0.0002)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = simulate(features, signals, notional=notional, commission=DEFAULT_COMMISSION)

    # gross price P&L = (mid_exit - mid_entry) * notional * direction
    # Entry at bar 2 (mid 1.12). Exit at bar 4 (mid 1.12). gross = 0 (mids equal).
    # So total should be -(spread + commission).
    expected = -(result.cum_spread_cost + result.cum_commission)
    assert result.equity.iloc[-1] == pytest.approx(expected)


def test_simulate_rejects_invalid_notional():
    features = make_engine_frame([1.10] * 3, spread=0.0)
    signals = make_signals([0, 0, 0], features.index)
    with pytest.raises(ValueError, match="notional"):
        simulate(features, signals, notional=0)
    with pytest.raises(ValueError, match="notional"):
        simulate(features, signals, notional=-1)


# ============================================================
# 10. Three-state engine support: no cost while flat, no cost while holding.
#     The first strategy to exercise flat mid-stream (Bollinger) needs the
#     engine to charge costs ONLY when position actually changes.
# ============================================================


def test_no_cost_or_pnl_when_always_flat():
    """All signals 0 -> no fills, no spread, no commission, zero equity."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    features = make_engine_frame([1.10] * 10, spread=0.0002)
    signals = make_signals([0] * 10, features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    assert result.n_order_legs == 0
    assert result.n_position_changes == 0
    assert result.cum_spread_cost == 0.0
    assert result.cum_commission == 0.0
    assert (result.equity == 0.0).all()


def test_no_extra_cost_while_holding_position():
    """Once entered, holding the position must not accrue per-bar charges."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    # Stay long for many bars; only one entry fill.
    features = make_engine_frame([1.10] * 100, spread=0.0002)
    signals = make_signals([0] + [1] * 99, features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    # 1 fill (the entry at bar 1).
    assert result.n_order_legs == 1
    expected_comm = DEFAULT_COMMISSION.commission_for(1.1001, 100_000.0)
    assert result.cum_commission == pytest.approx(expected_comm, rel=1e-9)
    # Spread cost = one half-spread on entry only.
    assert result.cum_spread_cost == pytest.approx(0.0001 * 100_000.0)


def test_long_flat_short_flat_charges_exactly_four_legs():
    """Three-state pattern (0 -> +1 -> 0 -> -1 -> 0) charges 4 legs, no more."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    features = make_engine_frame([1.10] * 9, spread=0.0)
    # signals shifted +1 by engine. Want targets:
    #   bar 1: 0 -> open long
    #   bar 3: long -> flat (close)
    #   bar 5: flat -> open short
    #   bar 7: short -> flat (close)
    # So raw signals at t such that target[t+1] is desired:
    #   t=0 -> target[1] = 1 (open) -> signals[0]=1
    #   t=1 -> target[2] = 1
    #   t=2 -> target[3] = 0 (exit) -> signals[2]=0
    #   t=4 -> target[5] = -1
    #   t=5 -> target[6] = -1
    #   t=6 -> target[7] = 0 (exit)
    signals = make_signals([1, 1, 0, 0, -1, -1, 0, 0, 0], features.index)
    result = simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    assert result.n_order_legs == 4
    assert result.n_position_changes == 4
