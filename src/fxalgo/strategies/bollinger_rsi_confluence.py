"""Bollinger mean reversion with optional RSI, daily-regime and cost filters.

A close-at-mean three-state (LONG / FLAT / SHORT) state machine with three
independent, optional entry gates layered on the Bollinger band touch:

  use_rsi    : require RSI confluence at entry
                 LONG  needs price <= lower band AND rsi <= rsi_lower (30)
                 SHORT needs price >= upper band AND rsi >= rsi_upper (70)
  use_regime : require `features[regime_col] != 0` at the decision bar. The
               column is a REGIME FLAG supplied by the caller: a daily
               Bollinger "shadow" projected onto this timeframe with
               `strategies.multi_timeframe.align_completed_signal`, which uses
               only CLOSED daily bars. Non-zero means the daily price is
               stretched away from its own 20-day mean; the SIGN is
               deliberately ignored -- only "stretched at all" matters.
  use_filter : require the cost-aware filter -- the band-to-mean target must
               clear `cost_multiple` x round-trip cost (`passes_cost_filter`).

With use_rsi and use_regime both on, this is the configuration called
"cell D" in the results.

Exit is always at the mean (bb_middle); there is NO stop-loss. The
"flat in the middle, wait for a fresh band touch" logic is preserved.

A failed regime or cost gate SKIPS the touch episode: the strategy stays flat
until price returns inside the bands. RSI does NOT lock the episode -- it can
legitimately confirm a bar or two into a touch, so it keeps being checked.
Skipping leaves the strategy flat instead of in a position, so it can take
later touches the ungated strategy would have missed while holding: the gated
trade set is not simply a subset of the ungated one.

Causality: every entry/exit decision at bar t uses only (state at t-1, bands
/ price / RSI / regime / spread at t) -- all <= t. A lookahead test enforces
this. The engine fills at the next bar's open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.backtest.costs import DEFAULT_COMMISSION, CommissionModel
from fxalgo.strategies.base import Strategy
from fxalgo.strategies.cost_filter import passes_cost_filter

DEFAULT_UPPER = "bb_upper"
DEFAULT_MIDDLE = "bb_middle"
DEFAULT_LOWER = "bb_lower"
DEFAULT_PRICE = "mid_close"
DEFAULT_SPREAD = "spread_close"


class BollingerRsiConfluence(Strategy):
    """Close-at-mean Bollinger with optional RSI, daily-regime and cost gates."""

    def __init__(
        self,
        *,
        use_rsi: bool = True,
        use_regime: bool = False,
        use_filter: bool = False,
        rsi_period: int = 14,
        rsi_lower: float = 30.0,
        rsi_upper: float = 70.0,
        regime_col: str = "regime_daily",
        cost_multiple: float = 2.0,
        notional: float = 100_000.0,
        commission: CommissionModel | None = DEFAULT_COMMISSION,
        upper_col: str = DEFAULT_UPPER,
        middle_col: str = DEFAULT_MIDDLE,
        lower_col: str = DEFAULT_LOWER,
        price_col: str = DEFAULT_PRICE,
        spread_col: str = DEFAULT_SPREAD,
    ) -> None:
        if cost_multiple <= 0:
            raise ValueError(f"cost_multiple must be positive, got {cost_multiple}")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if not (0.0 <= rsi_lower < rsi_upper <= 100.0):
            raise ValueError(f"need 0 <= rsi_lower < rsi_upper <= 100, got {rsi_lower}/{rsi_upper}")
        self.use_rsi = use_rsi
        self.use_regime = use_regime
        self.use_filter = use_filter
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.regime_col = regime_col
        self.cost_multiple = cost_multiple
        self.notional = notional
        self.commission = commission
        self.upper_col = upper_col
        self.middle_col = middle_col
        self.lower_col = lower_col
        self.price_col = price_col
        self.spread_col = spread_col
        self.rsi_col = f"rsi_{rsi_period}"
        rg = f",regime={regime_col}" if use_regime else ""
        self.name = (
            f"bollinger_confluence(rsi={'on' if use_rsi else 'off'},"
            f"filter={'on' if use_filter else 'off'}{rg})"
        )
        self.n_entries_taken = 0
        self.n_entries_skipped = 0  # regime / cost-filter skips

    def _decide(self, side: int, p: float, mid: float, spread_t: float, rsi_t: float,
                regime_t: float) -> str:
        """Entry decision for a fresh band touch: 'enter', 'skip', or 'wait'.

        'skip' (cost-filter / regime failure) locks the touch episode; 'wait'
        (RSI not confirming) does not, so confluence can confirm later in the
        touch.
        """
        if self.use_rsi:
            rsi_ok = rsi_t <= self.rsi_lower if side == 1 else rsi_t >= self.rsi_upper
            if not rsi_ok:
                return "wait"
        if self.use_filter and not passes_cost_filter(
            p, mid, spread_t,
            notional=self.notional,
            commission=self.commission,
            cost_multiple=self.cost_multiple,
        ):
            return "skip"
        if self.use_regime and regime_t == 0.0:
            # The daily timeframe is sitting at its own mean -- no stretch.
            return "skip"
        return "enter"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        needed = [self.price_col, self.upper_col, self.middle_col, self.lower_col]
        if self.use_rsi:
            needed.append(self.rsi_col)
        if self.use_filter:
            needed.append(self.spread_col)
        if self.use_regime:
            needed.append(self.regime_col)
        missing = [c for c in needed if c not in features.columns]
        if missing:
            raise KeyError(
                f"BollingerRsiConfluence requires columns {missing}; not in feature frame"
            )

        n = len(features)
        zeros = np.zeros(n)
        price = features[self.price_col].to_numpy(dtype=np.float64)
        upper = features[self.upper_col].to_numpy(dtype=np.float64)
        middle = features[self.middle_col].to_numpy(dtype=np.float64)
        lower = features[self.lower_col].to_numpy(dtype=np.float64)
        rsi = features[self.rsi_col].to_numpy(dtype=np.float64) if self.use_rsi else zeros
        spread = features[self.spread_col].to_numpy(dtype=np.float64) if self.use_filter else zeros
        regime = features[self.regime_col].to_numpy(dtype=np.float64) if self.use_regime else zeros

        signals = np.zeros(n, dtype=np.int8)
        state = 0
        pending_side = 0  # skip lock for the current touch episode
        n_taken = 0
        n_skipped = 0

        for t in range(n):
            warmup = (
                np.isnan(upper[t])
                or np.isnan(middle[t])
                or np.isnan(lower[t])
                or np.isnan(price[t])
                or (self.use_rsi and np.isnan(rsi[t]))
                or (self.use_filter and np.isnan(spread[t]))
                or (self.use_regime and np.isnan(regime[t]))
            )
            if warmup:
                state = 0
                pending_side = 0
                signals[t] = 0
                continue

            p = price[t]
            mid = middle[t]

            if state == 0:
                if lower[t] < p < upper[t]:
                    pending_side = 0  # inside the bands -> re-arm
                elif p <= lower[t] and pending_side != 1:
                    action = self._decide(1, p, mid, spread[t], rsi[t], regime[t])
                    if action == "enter":
                        state = 1
                        n_taken += 1
                    elif action == "skip":
                        n_skipped += 1
                        pending_side = 1
                elif p >= upper[t] and pending_side != -1:
                    action = self._decide(-1, p, mid, spread[t], rsi[t], regime[t])
                    if action == "enter":
                        state = -1
                        n_taken += 1
                    elif action == "skip":
                        n_skipped += 1
                        pending_side = -1
            elif (state == 1 and p >= mid) or (state == -1 and p <= mid):
                # reverted to the mean -> flat (close-at-mean exit)
                state = 0
                pending_side = 0

            signals[t] = state

        self.n_entries_taken = n_taken
        self.n_entries_skipped = n_skipped
        return pd.Series(signals, index=features.index, name="signal", dtype=np.int8)
