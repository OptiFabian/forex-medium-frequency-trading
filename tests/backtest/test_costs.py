"""Tests for fxalgo.backtest.costs."""

from __future__ import annotations

import pytest

from fxalgo.backtest.costs import DEFAULT_COMMISSION, CommissionModel


def test_default_commission_100k_lot_at_1_10():
    """Default commission model on a 100k EUR notional at 1.10 EURUSD:
    trade_value = 110,000; 0.20 bps -> $22... wait, 0.20 bps = 0.00002.
    rate_based = 0.00002 * 110_000 = $2.20
    minimum    = $2.00
    cap        = 0.002 * 110_000 = $220 (doesn't bind)
    commission = $2.20.
    """
    c = DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=100_000)
    assert c == pytest.approx(2.20)


def test_minimum_binds_on_small_trades():
    """Tiny notional: rate_based is well below $2; minimum binds."""
    # trade_value = 1.10 * 10_000 = 11_000; rate_based = $0.22; min = $2.00.
    # cap = 0.002 * 11_000 = $22 (no bind).
    # Expected: $2.00.
    c = DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=10_000)
    assert c == pytest.approx(2.00)


def test_rate_binds_on_large_trades():
    """Large notional: rate_based exceeds both minimum and cap thresholds."""
    # trade_value = 1.10 * 10_000_000 = 11_000_000; rate_based = $220.
    # min = $2.00 (no bind). cap = 0.002 * 11M = $22,000 (no bind).
    # Expected: $220.
    c = DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=10_000_000)
    assert c == pytest.approx(220.0)


def test_cap_binds_for_micro_trade_where_min_would_exceed_cap():
    """Degenerate micro trade: minimum would exceed 0.2% of trade value."""
    # trade_value = 1.10 * 100 = $110; rate_based = $0.0022.
    # max(min, rate) = $2.00. cap = 0.002 * 110 = $0.22.
    # commission = min(cap, raw) = $0.22.
    c = DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=100)
    assert c == pytest.approx(0.22)


def test_min_to_rate_crossover_is_at_100k_trade_value():
    """The $2.00 minimum stops binding at trade value = $100,000, NOT $10M.

    rate_based = 0.00002 * trade_value = $2.00  <=>  trade_value = $100,000.
    Below $100k the $2 min binds; above it the 0.20 bps rate binds. At a
    EURUSD price of 1.08 this crossover is ~92,600 EUR of notional, so a
    100k EUR lot already sits (just) on the rate side -- the minimum does
    not bind at any notional >= 100k EUR at realistic EURUSD prices.
    """
    # Exactly at the crossover: min and rate both == $2.00.
    at = DEFAULT_COMMISSION.commission_for(fill_price=1.0, notional=100_000)
    assert at == pytest.approx(2.00)
    # Just below trade value $100k -> min binds ($2.00 > rate).
    below = DEFAULT_COMMISSION.commission_for(fill_price=1.0, notional=90_000)
    assert below == pytest.approx(2.00)  # rate would be $1.80; min wins
    # Just above -> rate binds and exceeds $2.00.
    above = DEFAULT_COMMISSION.commission_for(fill_price=1.0, notional=110_000)
    assert above == pytest.approx(2.20)  # 0.00002 * 110_000
    # A 100k EUR lot at 1.08 is already on the rate side (> $2.00).
    lot = DEFAULT_COMMISSION.commission_for(fill_price=1.08, notional=100_000)
    assert lot == pytest.approx(2.16)
    assert lot > 2.00


def test_custom_commission_model_independent_of_default():
    custom = CommissionModel(rate=0.0001, min_per_order=1.0, cap_pct=0.01)
    # trade_value = 1.0 * 50_000 = 50_000; rate = 0.0001*50_000 = 5.00.
    # max(min=1.0, rate=5.0) = 5.00. cap = 0.01*50_000 = 500. Result: 5.00.
    c = custom.commission_for(fill_price=1.0, notional=50_000)
    assert c == pytest.approx(5.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="notional"):
        DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=0)
    with pytest.raises(ValueError, match="notional"):
        DEFAULT_COMMISSION.commission_for(fill_price=1.10, notional=-100)
    with pytest.raises(ValueError, match="fill_price"):
        DEFAULT_COMMISSION.commission_for(fill_price=0, notional=100_000)
