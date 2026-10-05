"""Trust tests for Maximum Adverse Excursion (MAE) tracking.

MAE = the deepest the trade went underwater from its ENTRY MID, mid-to-mid,
over the bars it was actually held (entry bar included; a signal/time exit
bar is excluded because the position closed at that bar's open).
"""

from __future__ import annotations

import numpy as np
import pytest

from fxalgo.backtest.engine import simulate
from tests.backtest.conftest import make_engine_frame, make_signals

N = 100_000.0


def test_mae_long_known_worst_point():
    """Long held bars 1-3, signal-exit at bar 4. entry mid = 1.10.
    Lows over held bars: [1.10, 1.08, 1.09] -> worst 1.08 -> MAE = 0.02.
    Bar 4's low (1.00) must be EXCLUDED (exited at its open)."""
    mids = [1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    lows = [1.10, 1.10, 1.08, 1.09, 1.00, 1.10]
    highs = [1.10] * 6
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1, 0, 0, 0], features.index)

    result = simulate(features, signals, notional=N, commission=None, track_mae=True)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["mae_price"] == pytest.approx(0.02)
    assert trade["mae_pnl"] == pytest.approx(0.02 * N)


def test_mae_includes_entry_bar():
    """Worst adverse low is on the ENTRY bar itself -> must be counted."""
    mids = [1.10, 1.10, 1.10, 1.10]
    lows = [1.10, 1.07, 1.09, 1.10]  # entry bar (idx 1) low = 1.07 is the worst
    highs = [1.10] * 4
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([1, 1, 0, 0], features.index)

    result = simulate(features, signals, notional=N, commission=None, track_mae=True)
    trade = result.trades.iloc[0]
    assert trade["mae_price"] == pytest.approx(0.03)  # 1.10 - 1.07


def test_mae_short_known_worst_point():
    """Short held bars 1-3. entry mid = 1.10. Highs [1.10, 1.13, 1.11] ->
    worst 1.13 -> MAE = 0.03. Bar 4 high (1.20) excluded."""
    mids = [1.10, 1.10, 1.10, 1.10, 1.10, 1.10]
    highs = [1.10, 1.10, 1.13, 1.11, 1.20, 1.10]
    lows = [1.10] * 6
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([-1, -1, -1, 0, 0, 0], features.index)

    result = simulate(features, signals, notional=N, commission=None, track_mae=True)
    trade = result.trades.iloc[0]
    assert trade["direction"] == -1
    assert trade["mae_price"] == pytest.approx(0.03)
    assert trade["mae_pnl"] == pytest.approx(0.03 * N)


def test_mae_zero_when_never_underwater():
    """A long whose price only ever rises has MAE = 0 (never went underwater)."""
    mids = [1.10, 1.10, 1.11, 1.12, 1.13]
    lows = [1.10, 1.10, 1.11, 1.12, 1.13]  # never below entry mid 1.10
    highs = mids
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1, 0, 0], features.index)
    result = simulate(features, signals, notional=N, commission=None, track_mae=True)
    assert result.trades.iloc[0]["mae_price"] == pytest.approx(0.0)


def test_mae_nan_when_not_tracked():
    """Without track_mae, the MAE columns are NaN (not measured)."""
    mids = [1.10, 1.10, 1.09, 1.10]
    features = make_engine_frame(mids, spread=0.0002)
    signals = make_signals([1, 1, 0, 0], features.index)
    result = simulate(features, signals, notional=N, commission=None)  # track_mae default False
    assert np.isnan(result.trades.iloc[0]["mae_price"])
    assert np.isnan(result.trades.iloc[0]["mae_pnl"])


def test_mae_intrabar_exit_uses_entry_bar_seed():
    """A trade that opens and intrabar-stops on the SAME bar still has a defined
    MAE from the entry bar's seeded range."""
    from tests.backtest.test_time_exit import _FixedLevels

    mids = [1.10, 1.10, 1.10]
    lows = [1.10, 1.10, 1.094]  # bar 2 entry... but entry is bar 1; see signals
    highs = [1.10] * 3
    # Enter long at bar 1; SL at 1.095 fires intrabar at bar 1? Set low[1]=1.094.
    lows = [1.10, 1.094, 1.10]
    features = make_engine_frame(mids, spread=0.0002, highs=highs, lows=lows)
    signals = make_signals([1, 1, 1], features.index)
    policy = _FixedLevels(tp=None, sl=1.095)
    result = simulate(
        features, signals, notional=N, commission=None, level_policy=policy, track_mae=True
    )
    # Entry at bar 1 (mid 1.10); bar 1 low 1.094 trips SL and seeds MAE.
    assert result.n_intrabar_sl == 1
    trade = result.trades.iloc[0]
    assert trade["mae_price"] == pytest.approx(1.10 - 1.094)
