"""Performance metrics for a `BacktestResult`.

Conventions
-----------
- Risk-free rate: ASSUMED 0. Sharpe is the raw mean/std of resampled
  P&L deltas, annualized.
- Annualization: 252 trading days. Equity is resampled to daily (last
  value per UTC date) and then differenced to get daily P&L.
- "Return" expressed as P&L in quote currency. The strategy trades a
  fixed 1 unit of base currency, so an absolute P&L number is the
  honest answer. We also report a "return % of avg mid_close" for
  scale-free comparison (it's not a true return on capital — that
  would require a notional that we haven't specified).
- Max drawdown: largest peak-to-trough decline in the equity curve.

Spread cost accounting
----------------------
"Total P&L" = realized + unrealized at end of the run (post-warmup).

`spread_cost_pct_of_abs_pnl` = cum_spread_cost / |total_pnl|. We use the
absolute value of P&L so the metric stays meaningful regardless of sign:

- If P&L is positive, this tells you what % of profit was eaten by spread
  (and what the gross-of-friction profit would have been).
- If P&L is negative, this tells you what fraction of the loss can be
  attributed to spread cost.

NaN only when total P&L is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxalgo.backtest.engine import BacktestResult

TRADING_DAYS_PER_YEAR = 252


LABEL_WIDTH = 34
NUMBER_WIDTH = 14


@dataclass
class MetricsReport:
    """Computed metrics from one backtest result."""

    total_pnl: float  # net of spread (implicit in fills) AND commission
    annualized_pnl: float
    notional: float
    avg_price: float
    return_pct_of_notional: float  # total_pnl / (notional * avg_price)
    annualized_return_pct: float
    sharpe: float
    max_drawdown: float
    max_drawdown_pct_of_notional: float
    n_trades: int
    win_rate: float
    avg_trade_pnl: float
    median_trade_pnl: float
    avg_bars_held: float
    cum_spread_cost: float
    spread_cost_pct_of_abs_pnl: float
    cum_commission: float
    commission_pct_of_abs_pnl: float
    total_costs: float
    total_costs_pct_of_abs_pnl: float
    n_order_legs: int
    years_covered: float
    n_bars_after_warmup: int

    def format(self, label: str = "") -> str:
        head = f"{label}\n" if label else ""
        rows = [
            ("notional per trade (base ccy)", _num(self.notional, ",.0f")),
            ("total P&L (quote ccy, net)", _num(self.total_pnl, ",.2f")),
            ("annualized P&L", _num(self.annualized_pnl, ",.2f")),
            ("avg mid price", _num(self.avg_price, ".6f")),
            ("return % of notional value", _pct(self.return_pct_of_notional)),
            ("annualized return %", _pct(self.annualized_return_pct)),
            ("Sharpe (daily, ann., rf=0)", _num(self.sharpe, ".4f")),
            ("max drawdown (quote ccy)", _num(self.max_drawdown, ",.2f")),
            ("max drawdown % of notional", _pct(self.max_drawdown_pct_of_notional)),
            ("number of closed trades", _int(self.n_trades)),
            ("number of order legs", _int(self.n_order_legs)),
            ("win rate", _pct(self.win_rate)),
            ("avg trade P&L (pre-comm.)", _num(self.avg_trade_pnl, ",.2f")),
            ("median trade P&L (pre-comm.)", _num(self.median_trade_pnl, ",.2f")),
            ("avg holding period (bars)", _num(self.avg_bars_held, ",.1f")),
            ("cumulative spread cost", _num(self.cum_spread_cost, ",.2f")),
            ("spread / |total P&L|", _pct(self.spread_cost_pct_of_abs_pnl)),
            ("cumulative commission", _num(self.cum_commission, ",.2f")),
            ("commission / |total P&L|", _pct(self.commission_pct_of_abs_pnl)),
            ("total transaction costs", _num(self.total_costs, ",.2f")),
            ("total costs / |total P&L|", _pct(self.total_costs_pct_of_abs_pnl)),
            ("years covered", _num(self.years_covered, ".2f")),
            ("bars after warmup", _int(self.n_bars_after_warmup)),
        ]
        body = "\n".join(f"  {lbl:<{LABEL_WIDTH}}: {val}" for lbl, val in rows)
        return head + body


def _num(x: float, spec: str) -> str:
    if np.isnan(x):
        return f"{'n/a':>{NUMBER_WIDTH}}"
    return f"{x:>{NUMBER_WIDTH}{spec}}"


def _int(x: int) -> str:
    return f"{x:>{NUMBER_WIDTH},d}"


def _pct(x: float) -> str:
    if np.isnan(x):
        return f"{'n/a':>{NUMBER_WIDTH}}"
    return f"{x:>{NUMBER_WIDTH - 1}.2%}"


def compute(result: BacktestResult, features: pd.DataFrame) -> MetricsReport:
    """Compute metrics from a `BacktestResult` and the source features."""
    # Slice off the warmup region for everything except trade-derived stats.
    if result.warmup_bars > 0:
        equity = result.equity.iloc[result.warmup_bars :]
    else:
        equity = result.equity

    n_bars = len(equity)
    if n_bars < 2:
        raise ValueError("Equity series too short to compute metrics")

    total_pnl = float(equity.iloc[-1] - equity.iloc[0])

    span_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = span_seconds / (365.25 * 24 * 3600)
    annualized_pnl = total_pnl / years if years > 0 else 0.0

    mid_close = features["mid_close"]
    if result.warmup_bars > 0:
        mid_close = mid_close.iloc[result.warmup_bars :]
    avg_price = float(mid_close.mean())
    # "Return % of notional value" = P&L / (notional * avg_price). avg_price is
    # the mean mid over the run, used here only as a reference for the notional's
    # quote-currency value -- not as a base for compounding.
    notional_value = result.notional * avg_price
    return_pct_of_notional = total_pnl / notional_value if notional_value > 0 else float("nan")
    annualized_return_pct = (
        annualized_pnl / notional_value if notional_value > 0 else float("nan")
    )

    # Daily Sharpe: resample equity to daily, take diff, annualize by sqrt(252).
    daily = equity.resample("1D").last().dropna()
    daily_pnl = daily.diff().dropna()
    if len(daily_pnl) > 1 and daily_pnl.std(ddof=1) > 0:
        sharpe = (
            daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = 0.0

    # Max drawdown (in quote ccy).
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = float(drawdown.min())
    max_dd_pct_of_notional = max_dd / notional_value if notional_value > 0 else float("nan")

    trades = result.trades
    # Filter trades that closed inside the warmup region (can happen if signal
    # flips right at the warmup boundary).
    if not trades.empty and result.warmup_bars > 0:
        warmup_cutoff = features.index[result.warmup_bars - 1]
        trades = trades[trades["exit_time"] > warmup_cutoff]

    n_trades = len(trades)
    if n_trades > 0:
        win_rate = float((trades["pnl"] > 0).mean())
        avg_trade = float(trades["pnl"].mean())
        median_trade = float(trades["pnl"].median())
        avg_bars_held = float(trades["bars_held"].mean())
    else:
        win_rate = 0.0
        avg_trade = 0.0
        median_trade = 0.0
        avg_bars_held = 0.0

    abs_pnl = abs(total_pnl)
    if abs_pnl > 0:
        spread_pct_abs = result.cum_spread_cost / abs_pnl
        comm_pct_abs = result.cum_commission / abs_pnl
    else:
        spread_pct_abs = float("nan")
        comm_pct_abs = float("nan")

    total_costs = result.cum_spread_cost + result.cum_commission
    if abs_pnl > 0:
        total_costs_pct_abs = total_costs / abs_pnl
    else:
        total_costs_pct_abs = float("nan")

    return MetricsReport(
        total_pnl=total_pnl,
        annualized_pnl=annualized_pnl,
        notional=result.notional,
        avg_price=avg_price,
        return_pct_of_notional=return_pct_of_notional,
        annualized_return_pct=annualized_return_pct,
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        max_drawdown_pct_of_notional=(
            float(max_dd_pct_of_notional)
            if not np.isnan(max_dd_pct_of_notional)
            else float("nan")
        ),
        n_trades=n_trades,
        win_rate=win_rate,
        avg_trade_pnl=avg_trade,
        median_trade_pnl=median_trade,
        avg_bars_held=avg_bars_held,
        cum_spread_cost=result.cum_spread_cost,
        spread_cost_pct_of_abs_pnl=spread_pct_abs,
        cum_commission=result.cum_commission,
        commission_pct_of_abs_pnl=comm_pct_abs,
        total_costs=total_costs,
        total_costs_pct_of_abs_pnl=total_costs_pct_abs,
        n_order_legs=result.n_order_legs,
        years_covered=years,
        n_bars_after_warmup=result.n_bars_after_warmup,
    )
