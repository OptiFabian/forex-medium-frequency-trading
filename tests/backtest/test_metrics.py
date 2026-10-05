"""Tests for fxalgo.backtest.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.backtest import engine, metrics
from tests.backtest.conftest import make_engine_frame, make_signals


def test_metrics_on_winning_trade():
    """Long entered low, exited high. Verifies n_trades / win_rate / PnL plumbing.

    Mids: 1.10, 1.10, 1.10, 1.20, 1.20
    Signals: [0, 1, 1, 0, 0]
    Targets (shift +1): [0, 0, 1, 1, 0]
      - Enter at open of bar 2 (mid=1.10, spread=0 -> ask=1.10).
      - Exit at open of bar 4 (mid=1.20, spread=0 -> bid=1.20).
    PnL = 1.20 - 1.10 = +0.10.
    """
    mids = [1.10, 1.10, 1.10, 1.20, 1.20]
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = engine.simulate(features, signals, warmup=0)

    report = metrics.compute(result, features)
    assert report.n_trades == 1
    assert report.win_rate == 1.0
    assert report.total_pnl == pytest.approx(0.10)
    assert report.cum_spread_cost == 0.0
    assert report.spread_cost_pct_of_abs_pnl == 0.0


def test_max_drawdown_captured():
    """Equity goes up then down: drawdown should equal the drop."""
    mids = [1.10, 1.10, 1.20, 1.20, 1.15, 1.15]
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([0, 1, 1, 1, 1, 0], features.index)
    result = engine.simulate(features, signals, warmup=0)

    report = metrics.compute(result, features)
    # Peak equity at bar 2 = 0.10 (mid_close 1.20 - entry 1.10).
    # Trough later when mid_close = 1.15 -> equity = 0.05.
    # Drawdown = 0.05 - 0.10 = -0.05.
    assert report.max_drawdown == pytest.approx(-0.05)


def test_metrics_handle_no_trades_gracefully():
    mids = [1.10] * 10
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([0] * 10, features.index)
    result = engine.simulate(features, signals, warmup=0)

    report = metrics.compute(result, features)
    assert report.n_trades == 0
    assert report.win_rate == 0.0
    assert report.total_pnl == 0.0
    assert np.isnan(report.spread_cost_pct_of_abs_pnl)


def test_spread_pct_defined_even_for_losing_strategy():
    """Losing trade: spread cost ratio should still be a finite number."""
    mids = [1.20, 1.20, 1.20, 1.10, 1.10]  # uptrend reverses
    features = make_engine_frame(mids, spread=0.0002)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = engine.simulate(features, signals, warmup=0)

    report = metrics.compute(result, features)
    assert report.total_pnl < 0
    # Must not be NaN any more (this is the bug we just fixed).
    assert not np.isnan(report.spread_cost_pct_of_abs_pnl)
    # Sanity: spread cost / |loss| is a positive finite number.
    assert report.spread_cost_pct_of_abs_pnl > 0


def test_format_label_alignment_uniform():
    """Every formatted row should have the colon at the same column."""
    mids = [1.10] * 10
    features = make_engine_frame(mids, spread=0.0)
    signals = make_signals([0] * 10, features.index)
    result = engine.simulate(features, signals, warmup=0)
    report = metrics.compute(result, features)
    formatted = report.format()
    colon_positions = {line.index(":") for line in formatted.splitlines() if ":" in line}
    assert len(colon_positions) == 1, (
        f"colons not aligned; found positions {colon_positions}"
    )


def test_metrics_with_commission_and_notional():
    """Verify commission/total-costs fields are populated and consistent."""
    from fxalgo.backtest.costs import DEFAULT_COMMISSION

    mids = [1.10] * 5
    features = make_engine_frame(mids, spread=0.0002)
    signals = make_signals([0, 1, 1, 0, 0], features.index)
    result = engine.simulate(
        features, signals, notional=100_000.0, commission=DEFAULT_COMMISSION
    )
    report = metrics.compute(result, features)

    assert report.notional == 100_000.0
    # Commission for two ~$110k legs at ~$2.20 each.
    assert report.cum_commission == pytest.approx(4.4, abs=0.01)
    assert report.cum_spread_cost == pytest.approx(0.0002 * 100_000)  # full spread once.
    assert report.total_costs == pytest.approx(
        report.cum_spread_cost + report.cum_commission
    )
    # Both ratios finite and positive (this is a losing scenario).
    assert report.commission_pct_of_abs_pnl > 0
    assert report.total_costs_pct_of_abs_pnl > 0
    # Order leg count.
    assert report.n_order_legs == 2
