"""Trust tests for the engine's time-based exit (max_hold_bars).

A position held for max_hold_bars without a signal/TP/SL exit is
force-closed at the open of the next bar, recording exit_reason='time_exit'.
"""

from __future__ import annotations

import pytest

from fxalgo.backtest.costs import DEFAULT_COMMISSION
from fxalgo.backtest.engine import simulate
from tests.backtest.conftest import make_engine_frame, make_signals

N = 100_000.0
SPREAD = 0.0002


class _FixedLevels:
    """Far-away TP/SL that never fire -- isolates the time-exit path."""

    def __init__(self, tp, sl):
        self.tp = tp
        self.sl = sl

    def levels_at_entry(self, *, direction, entry_price, bar):
        return self.tp, self.sl


# ============================================================
# 1. Long that never hits TP/SL exits at exactly the bar limit.
# ============================================================


def test_time_exit_long_fires_at_bar_limit():
    """Entry at bar 1; max_hold_bars=3 -> time-exit at bar 4 (bars_held=3),
    selling at that bar's open_bid, paying commission. TP/SL never reached.
    """
    n = 6
    mids = [1.10] * n  # flat OHLC -> no intrabar range, TP/SL impossible
    features = make_engine_frame(mids, spread=SPREAD)
    signals = make_signals([1] * n, features.index)
    policy = _FixedLevels(tp=1.20, sl=1.00)  # both far away

    result = simulate(
        features,
        signals,
        notional=N,
        commission=DEFAULT_COMMISSION,
        level_policy=policy,
        max_hold_bars=3,
    )

    assert result.n_time_exit == 1
    assert result.n_intrabar_tp == 0
    assert result.n_intrabar_sl == 0
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_exit"
    assert trade["bars_held"] == 3
    assert trade["direction"] == 1

    open_bid = 1.10 - SPREAD / 2.0  # long exits at the bid
    open_ask = 1.10 + SPREAD / 2.0  # entry filled at the ask
    assert trade["exit_price"] == pytest.approx(open_bid)
    assert trade["entry_price"] == pytest.approx(open_ask)

    expected_comm = DEFAULT_COMMISSION.commission_for(open_ask, N) + DEFAULT_COMMISSION.commission_for(
        open_bid, N
    )
    assert result.cum_commission == pytest.approx(expected_comm)
    assert result.n_order_legs == 2


# ============================================================
# 2. Short side mirrors the fill side.
# ============================================================


def test_time_exit_short_buys_at_ask():
    n = 6
    mids = [1.10] * n
    features = make_engine_frame(mids, spread=SPREAD)
    signals = make_signals([-1] * n, features.index)

    result = simulate(features, signals, notional=N, commission=None, max_hold_bars=3)

    assert result.n_time_exit == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "time_exit"
    assert trade["direction"] == -1
    assert trade["bars_held"] == 3
    open_ask = 1.10 + SPREAD / 2.0  # short exits by buying at the ask
    assert trade["exit_price"] == pytest.approx(open_ask)


# ============================================================
# 3. No time-exit when the limit is not configured.
# ============================================================


def test_no_time_exit_without_max_hold():
    n = 6
    mids = [1.10] * n
    features = make_engine_frame(mids, spread=SPREAD)
    signals = make_signals([1] * n, features.index)
    result = simulate(features, signals, notional=N, commission=None)
    # Position stays open to the end; never closed -> no trades, no time-exit.
    assert result.n_time_exit == 0
    assert len(result.trades) == 0


# ============================================================
# 4. After a time-exit, re-entry is suppressed until the signal changes.
# ============================================================


def test_time_exit_suppresses_reentry_until_signal_changes():
    """Strategy keeps emitting +1. After the time-exit the engine must not
    immediately re-open while the signal value is unchanged."""
    n = 10
    mids = [1.10] * n
    features = make_engine_frame(mids, spread=SPREAD)
    signals = make_signals([1] * n, features.index)
    result = simulate(features, signals, notional=N, commission=None, max_hold_bars=3)
    # Exactly one trade (entry bar 1 -> time-exit bar 4); then flat & suppressed.
    assert result.n_time_exit == 1
    assert len(result.trades) == 1
    assert (result.position.iloc[5:] == 0).all()


# ============================================================
# 5. An earlier TP pre-empts the time-exit (whichever triggers first).
# ============================================================


def test_intrabar_tp_before_time_limit_wins():
    """A TP that fires before the bar limit closes the trade as intrabar_tp,
    not time_exit."""
    n = 6
    mids = [1.10] * n
    highs = list(mids)
    highs[2] = 1.1100  # TP at bar 2, before max_hold_bars=4 would force-close
    features = make_engine_frame(mids, spread=SPREAD, highs=highs)
    signals = make_signals([1] * n, features.index)
    policy = _FixedLevels(tp=1.1050, sl=None)
    result = simulate(
        features,
        signals,
        notional=N,
        commission=None,
        level_policy=policy,
        max_hold_bars=4,
    )
    assert result.n_intrabar_tp == 1
    assert result.n_time_exit == 0
    assert result.trades.iloc[0]["exit_reason"] == "intrabar_tp"


# ============================================================
# 6. Bad max_hold_bars rejected.
# ============================================================


def test_bad_max_hold_bars_raises():
    features = make_engine_frame([1.10, 1.10, 1.10], spread=SPREAD)
    signals = make_signals([1, 1, 1], features.index)
    with pytest.raises(ValueError, match="max_hold_bars"):
        simulate(features, signals, max_hold_bars=0)
