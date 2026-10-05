"""Backtest engine: simulate a strategy against a feature frame.

Execution timing (no lookahead)
-------------------------------
A signal at row t is computed from data available at the **close** of
bar t. The engine executes that signal at the **open** of bar t+1.

  - target_position[t] = signals[t-1]
  - any position change is filled at open_bid / open_ask of bar t.

Fill prices (realistic bid/ask, never midpoint)
-----------------------------------------------
  Open / hold LONG  : buy at OPEN_ASK
  Close LONG        : sell at OPEN_BID
  Open / hold SHORT : sell at OPEN_BID
  Close SHORT       : buy at OPEN_ASK

A long->short flip is one close + one open: pays spread AND commission
on BOTH legs.

Intrabar level exits (optional, via `level_policy`)
---------------------------------------------------
If a `level_policy` is supplied, the engine asks it for (tp, sl) at the
moment a position opens. On each subsequent bar, the engine checks
whether the bar's mid_high / mid_low touched either level. Whichever
fires first closes the position INTRABAR, at the level itself (adjusted
by the bar's bid-ask half-spread on the exit side).

  Same-bar tie-breaker (PESSIMISTIC): if a single bar's range spans BOTH
  the TP and the SL, the STOP fills. We do not credit the take-profit
  in that case. This is the conservative simplification needed when we
  do not have intrabar tick ordering.

When an intrabar exit fires, the engine "suppresses" the strategy's
current signal value: any subsequent bars with the same target value
yield position=0 (we stay flat). Suppression releases the moment the
target value changes. Without this, a strategy that keeps emitting +1
across many bars (e.g. BollingerReversion holding a long) would re-open
the next bar after an intrabar exit, defeating the purpose.

Time-based exit (optional, via `max_hold_bars`)
-----------------------------------------------
If `max_hold_bars` is supplied, a position that has been held for at
least that many bars without a signal/TP/SL exit is force-closed "at
market" at the OPEN of the next bar. Concretely, at bar t the engine
checks whether the live position has aged out -- `t - entry_idx >=
max_hold_bars` -- and if so closes it at this bar's open (long sells at
open_bid, short buys at open_ask), recording `exit_reason="time_exit"`.

  Ordering within a bar: the time-exit is evaluated at the OPEN, after
  any signal transition but BEFORE the intrabar TP/SL range check. So a
  position entered at bar e gets intrabar TP/SL chances on bars
  e, e+1, ..., e+max_hold_bars-1 (that is `max_hold_bars` bars); if none
  fired, it time-exits at the open of bar e+max_hold_bars with
  bars_held == max_hold_bars. The time-exit is a market-on-open event,
  so at the boundary bar it takes precedence over that same bar's range.

Like an intrabar exit, a time-exit suppresses the strategy's current
signal value until it changes, so a strategy still emitting the held
direction does not immediately re-open.

Position size & costs
---------------------
Configurable `notional` (default 1.0 in `simulate`, 100,000 in `run`).
Commission charged on every order leg via `CommissionModel`. Spread cost
recorded per fill (half-spread on each leg). See `fxalgo.backtest.costs`.

Warmup
------
The first `warmup` bars are forced flat regardless of strategy output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxalgo.backtest.costs import DEFAULT_COMMISSION, CommissionModel
from fxalgo.backtest.level_exits import ExitLevelPolicy
from fxalgo.strategies.base import Strategy

logger = logging.getLogger(__name__)

REQUIRED_COLS_BASE = ("open_bid", "open_ask", "mid_open", "mid_close")
REQUIRED_COLS_INTRABAR = ("mid_high", "mid_low")

DEFAULT_NOTIONAL = 100_000.0


@dataclass
class Trade:
    """One closed round-trip trade.

    `pnl` is the price P&L in quote currency for this trade (scaled by
    notional). It is computed from the bid/ask FILL prices, so it is
    already NET of this trade's spread but GROSS of commission.

    Per-trade cost attribution (both legs of the round trip):
      - `spread_cost`: the entry half-spread + the exit half-spread, in
        quote ccy. `pnl + spread_cost` recovers the mid-to-mid ("price moved
        my way") P&L; `pnl - commission` is the true net P&L of the trade.
      - `commission`: the entry leg + exit leg commission, in quote ccy.

    These sum across trades to `BacktestResult.cum_spread_cost` /
    `cum_commission` (intrabar/time exits included). `exit_reason`
    distinguishes signal-driven, intrabar, and time exits.
    """

    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str = "signal"  # "signal", "intrabar_tp", "intrabar_sl", "time_exit"
    spread_cost: float = 0.0
    commission: float = 0.0
    # Maximum Adverse Excursion: the deepest the trade went underwater from
    # the entry MID, mid-to-mid (before costs), in price and in quote ccy.
    # Only populated when `simulate(..., track_mae=True)`; NaN otherwise.
    # This is an OFFLINE diagnostic measure -- it must never feed a live signal.
    mae_price: float = float("nan")
    mae_pnl: float = float("nan")


@dataclass
class BacktestResult:
    equity: pd.Series
    position: pd.Series
    trades: pd.DataFrame
    cum_spread_cost: float
    cum_commission: float
    notional: float
    warmup_bars: int
    strategy_name: str
    n_bars_after_warmup: int
    n_position_changes: int
    n_order_legs: int
    n_intrabar_tp: int = 0
    n_intrabar_sl: int = 0
    n_time_exit: int = 0
    meta: dict = field(default_factory=dict)


def _validate(features: pd.DataFrame, need_intrabar: bool) -> None:
    cols = list(REQUIRED_COLS_BASE)
    if need_intrabar:
        cols += list(REQUIRED_COLS_INTRABAR)
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise KeyError(f"Backtest requires columns {missing}; not in features")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features.index must be a DatetimeIndex")


def simulate(
    features: pd.DataFrame,
    signals: pd.Series,
    *,
    warmup: int = 0,
    notional: float = 1.0,
    commission: CommissionModel | None = None,
    level_policy: ExitLevelPolicy | None = None,
    max_hold_bars: int | None = None,
    track_mae: bool = False,
    strategy_name: str = "<custom>",
) -> BacktestResult:
    """Low-level simulation. `run` is the strategy-driven convenience wrapper.

    When `level_policy is None`, the engine behaves exactly like the
    signal-only version (no intrabar checks). The feature frame need not
    contain mid_high / mid_low in that case.

    When `max_hold_bars` is set, any position held that many bars without
    an earlier exit is force-closed at the next bar's open (a "time_exit").

    When `track_mae` is True, each trade records its Maximum Adverse
    Excursion (`mae_price`, `mae_pnl`): the deepest underwater the position
    went from its entry MID over the bars it was held (long -> lowest mid_low;
    short -> highest mid_high), mid-to-mid, before costs. Requires mid_high /
    mid_low. This is an offline calibration diagnostic only.
    """
    need_hl = level_policy is not None or track_mae
    _validate(features, need_intrabar=need_hl)
    if len(signals) != len(features):
        raise ValueError(f"signals length {len(signals)} != features length {len(features)}")
    if notional <= 0:
        raise ValueError(f"notional must be positive, got {notional}")
    if max_hold_bars is not None and max_hold_bars <= 0:
        raise ValueError(f"max_hold_bars must be positive, got {max_hold_bars}")

    n = len(features)
    open_bid = features["open_bid"].to_numpy(dtype=np.float64)
    open_ask = features["open_ask"].to_numpy(dtype=np.float64)
    mid_open = features["mid_open"].to_numpy(dtype=np.float64)
    mid_close = features["mid_close"].to_numpy(dtype=np.float64)
    if need_hl:
        mid_high = features["mid_high"].to_numpy(dtype=np.float64)
        mid_low = features["mid_low"].to_numpy(dtype=np.float64)
    else:
        mid_high = mid_low = None  # unused
    timestamps = features.index
    raw_signal = signals.to_numpy().astype(np.int8)

    # target_position[t] = signals[t-1] (executed at open of t). Pre-warmup forced flat.
    target = np.empty(n, dtype=np.int8)
    target[0] = 0
    if n > 1:
        target[1:] = raw_signal[:-1]
    if warmup > 0:
        target[:warmup] = 0

    position_out = np.zeros(n, dtype=np.int8)
    equity_out = np.zeros(n, dtype=np.float64)

    cur_pos = 0
    entry_price = 0.0
    entry_idx = -1
    entry_spread_cost = 0.0  # this trade's entry-leg spread cost (quote ccy)
    entry_commission = 0.0   # this trade's entry-leg commission (quote ccy)
    entry_mid = 0.0          # mid price at entry (for MAE), set on open
    trade_low = np.inf       # lowest mid_low seen while this trade is held
    trade_high = -np.inf     # highest mid_high seen while this trade is held
    active_tp: float | None = None
    active_sl: float | None = None
    realized_pnl = 0.0
    cum_spread_cost = 0.0
    cum_commission = 0.0
    n_position_changes = 0
    n_order_legs = 0
    n_intrabar_tp = 0
    n_intrabar_sl = 0
    n_time_exit = 0
    trades: list[Trade] = []

    # Signal suppression after an intrabar exit: while suppress_active,
    # incoming target values equal to `suppressed_value` are overridden
    # to 0. The moment the target value differs, suppression releases.
    suppress_active = False
    suppressed_value: int = 0

    def charge_commission(fill_price: float) -> float:
        if commission is None:
            return 0.0
        return commission.commission_for(fill_price, notional)

    def mae_for(direction: int) -> tuple[float, float]:
        """(mae_price, mae_pnl) for the trade being closed. NaN if not tracking.
        Uses the worst mid excursion vs entry_mid accumulated over held bars."""
        if not track_mae:
            return float("nan"), float("nan")
        if direction == 1:
            mae_price = max(0.0, entry_mid - trade_low) if np.isfinite(trade_low) else 0.0
        else:
            mae_price = max(0.0, trade_high - entry_mid) if np.isfinite(trade_high) else 0.0
        return mae_price, mae_price * notional

    def close_position_at_open(t_idx: int, exit_reason: str = "signal") -> None:
        """Close cur_pos at this bar's open. `exit_reason` is recorded on the
        trade: "signal" for a strategy-driven exit, "time_exit" for the
        max_hold_bars force-close. Both fill at the bar's open (long sells at
        open_bid, short buys at open_ask).
        """
        nonlocal cur_pos, entry_price, entry_idx, active_tp, active_sl
        nonlocal realized_pnl, cum_spread_cost, cum_commission, n_order_legs
        if cur_pos == 1:
            exit_p = open_bid[t_idx]
            trade_pnl = (exit_p - entry_price) * notional
            realized_pnl += trade_pnl
            exit_spread = (mid_open[t_idx] - exit_p) * notional
            cum_spread_cost += exit_spread
            comm = charge_commission(exit_p)
            cum_commission += comm
            realized_pnl -= comm
            n_order_legs += 1
            mae_price, mae_pnl = mae_for(1)
            trades.append(
                Trade(
                    direction=1,
                    entry_time=timestamps[entry_idx],
                    entry_price=entry_price,
                    exit_time=timestamps[t_idx],
                    exit_price=exit_p,
                    pnl=trade_pnl,
                    bars_held=t_idx - entry_idx,
                    exit_reason=exit_reason,
                    spread_cost=entry_spread_cost + exit_spread,
                    commission=entry_commission + comm,
                    mae_price=mae_price,
                    mae_pnl=mae_pnl,
                )
            )
        elif cur_pos == -1:
            exit_p = open_ask[t_idx]
            trade_pnl = (entry_price - exit_p) * notional
            realized_pnl += trade_pnl
            exit_spread = (exit_p - mid_open[t_idx]) * notional
            cum_spread_cost += exit_spread
            comm = charge_commission(exit_p)
            cum_commission += comm
            realized_pnl -= comm
            n_order_legs += 1
            mae_price, mae_pnl = mae_for(-1)
            trades.append(
                Trade(
                    direction=-1,
                    entry_time=timestamps[entry_idx],
                    entry_price=entry_price,
                    exit_time=timestamps[t_idx],
                    exit_price=exit_p,
                    pnl=trade_pnl,
                    bars_held=t_idx - entry_idx,
                    exit_reason=exit_reason,
                    spread_cost=entry_spread_cost + exit_spread,
                    commission=entry_commission + comm,
                    mae_price=mae_price,
                    mae_pnl=mae_pnl,
                )
            )
        cur_pos = 0
        entry_price = 0.0
        entry_idx = -1
        active_tp = None
        active_sl = None

    def open_position_at_open(t_idx: int, new_pos: int) -> None:
        nonlocal cur_pos, entry_price, entry_idx, active_tp, active_sl
        nonlocal cum_spread_cost, cum_commission, realized_pnl, n_order_legs
        nonlocal entry_spread_cost, entry_commission, entry_mid, trade_low, trade_high
        if track_mae:
            entry_mid = mid_open[t_idx]
            # Seed with the entry bar's range so a same-bar intrabar exit still
            # has a defined adverse excursion.
            trade_low = mid_low[t_idx]
            trade_high = mid_high[t_idx]
        if new_pos == 1:
            entry_price = open_ask[t_idx]
            entry_spread_cost = (entry_price - mid_open[t_idx]) * notional
        elif new_pos == -1:
            entry_price = open_bid[t_idx]
            entry_spread_cost = (mid_open[t_idx] - entry_price) * notional
        else:
            entry_price = 0.0
            entry_spread_cost = 0.0
        cum_spread_cost += entry_spread_cost
        comm = charge_commission(entry_price) if new_pos != 0 else 0.0
        entry_commission = comm
        cum_commission += comm
        realized_pnl -= comm
        if new_pos != 0:
            n_order_legs += 1
        entry_idx = t_idx
        cur_pos = new_pos
        if level_policy is not None and new_pos != 0:
            tp, sl = level_policy.levels_at_entry(
                direction=new_pos,
                entry_price=entry_price,
                bar=features.iloc[t_idx],
            )
            active_tp = tp
            active_sl = sl
        else:
            active_tp = None
            active_sl = None

    for t in range(n):
        raw_target = int(target[t])

        # Release suppression if the target value has changed.
        if suppress_active and raw_target != suppressed_value:
            suppress_active = False

        # Compute the effective target after suppression.
        effective_target = 0 if suppress_active else raw_target

        # --- Signal-based position transition at bar's open ---
        if effective_target != cur_pos:
            if cur_pos != 0:
                close_position_at_open(t)
            if effective_target != 0:
                open_position_at_open(t, effective_target)
            else:
                # Already closed above; just reset bookkeeping.
                cur_pos = 0
                entry_price = 0.0
                entry_idx = -1
                active_tp = None
                active_sl = None
            n_position_changes += 1

        # --- Time-based exit: force-close a position aged out at this open ---
        # Evaluated at the open, after signal transitions but before the
        # intrabar range check. A position opened at bar e time-exits at the
        # open of bar e + max_hold_bars (bars_held == max_hold_bars).
        if (
            max_hold_bars is not None
            and cur_pos != 0
            and (t - entry_idx) >= max_hold_bars
        ):
            close_position_at_open(t, exit_reason="time_exit")
            n_time_exit += 1
            # Suppress re-entry until the strategy's signal value changes.
            suppress_active = True
            suppressed_value = raw_target

        # --- Intrabar level check (only when position is open + policy set) ---
        if (
            cur_pos != 0
            and level_policy is not None
            and (active_tp is not None or active_sl is not None)
        ):
            hi = mid_high[t]
            lo = mid_low[t]
            if cur_pos == 1:
                tp_hit = (active_tp is not None) and (not np.isnan(active_tp)) and (hi >= active_tp)
                sl_hit = (active_sl is not None) and (not np.isnan(active_sl)) and (lo <= active_sl)
            else:  # cur_pos == -1
                tp_hit = (active_tp is not None) and (not np.isnan(active_tp)) and (lo <= active_tp)
                sl_hit = (active_sl is not None) and (not np.isnan(active_sl)) and (hi >= active_sl)

            if tp_hit or sl_hit:
                # Pessimistic same-bar rule: SL wins on a tie.
                if sl_hit:
                    exit_level = float(active_sl)  # type: ignore[arg-type]
                    exit_reason = "intrabar_sl"
                    n_intrabar_sl += 1
                else:
                    exit_level = float(active_tp)  # type: ignore[arg-type]
                    exit_reason = "intrabar_tp"
                    n_intrabar_tp += 1

                half_spread = (open_ask[t] - open_bid[t]) / 2.0
                if cur_pos == 1:
                    fill_price = exit_level - half_spread
                    trade_pnl = (fill_price - entry_price) * notional
                else:
                    fill_price = exit_level + half_spread
                    trade_pnl = (entry_price - fill_price) * notional

                realized_pnl += trade_pnl
                exit_spread = half_spread * notional
                cum_spread_cost += exit_spread
                comm = charge_commission(fill_price)
                cum_commission += comm
                realized_pnl -= comm
                n_order_legs += 1
                mae_price, mae_pnl = mae_for(cur_pos)
                trades.append(
                    Trade(
                        direction=cur_pos,
                        entry_time=timestamps[entry_idx],
                        entry_price=entry_price,
                        exit_time=timestamps[t],
                        exit_price=fill_price,
                        pnl=trade_pnl,
                        bars_held=t - entry_idx,
                        exit_reason=exit_reason,
                        spread_cost=entry_spread_cost + exit_spread,
                        commission=entry_commission + comm,
                        mae_price=mae_price,
                        mae_pnl=mae_pnl,
                    )
                )
                cur_pos = 0
                entry_price = 0.0
                entry_idx = -1
                active_tp = None
                active_sl = None

                # Suppress until the strategy's signal value changes.
                suppress_active = True
                suppressed_value = raw_target

        # MAE: the position SURVIVED bar t (no exit fired), so it was exposed
        # to bar t's full range -- fold it into the running worst excursion.
        # Bars closed at this open (signal/time exit) have cur_pos == 0 here and
        # are correctly excluded; the entry bar was seeded in open_position.
        if track_mae and cur_pos == 1 and mid_low[t] < trade_low:
            trade_low = mid_low[t]
        elif track_mae and cur_pos == -1 and mid_high[t] > trade_high:
            trade_high = mid_high[t]

        # Mark-to-market on mid_close.
        if cur_pos == 1:
            unrealized = (mid_close[t] - entry_price) * notional
        elif cur_pos == -1:
            unrealized = (entry_price - mid_close[t]) * notional
        else:
            unrealized = 0.0

        position_out[t] = cur_pos
        equity_out[t] = realized_pnl + unrealized

    equity_series = pd.Series(equity_out, index=timestamps, name="equity")
    position_series = pd.Series(position_out, index=timestamps, name="position")
    trades_df = (
        pd.DataFrame([t.__dict__ for t in trades])
        if trades
        else pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    )

    n_after_warmup = n - warmup if n > warmup else 0
    return BacktestResult(
        equity=equity_series,
        position=position_series,
        trades=trades_df,
        cum_spread_cost=cum_spread_cost,
        cum_commission=cum_commission,
        notional=notional,
        warmup_bars=warmup,
        strategy_name=strategy_name,
        n_bars_after_warmup=n_after_warmup,
        n_position_changes=n_position_changes,
        n_order_legs=n_order_legs,
        n_intrabar_tp=n_intrabar_tp,
        n_intrabar_sl=n_intrabar_sl,
        n_time_exit=n_time_exit,
    )


def run(
    features: pd.DataFrame,
    strategy: Strategy,
    *,
    warmup: int = 600,
    notional: float = DEFAULT_NOTIONAL,
    commission: CommissionModel | None = DEFAULT_COMMISSION,
    level_policy: ExitLevelPolicy | None = None,
    max_hold_bars: int | None = None,
    track_mae: bool = False,
) -> BacktestResult:
    """High-level entry point. Realistic defaults: 100k notional + the default commission model.

    Pass `commission=None` to disable commission. Pass `level_policy=<policy>`
    to overlay intrabar TP/SL on top of the strategy's signals. Pass
    `max_hold_bars=<n>` to force-close positions held that many bars. Pass
    `track_mae=True` to record per-trade Maximum Adverse Excursion.
    """
    signals = strategy.generate_signals(features)
    return simulate(
        features,
        signals,
        warmup=warmup,
        notional=notional,
        commission=commission,
        level_policy=level_policy,
        max_hold_bars=max_hold_bars,
        track_mae=track_mae,
        strategy_name=strategy.name,
    )
