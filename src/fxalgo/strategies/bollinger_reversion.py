"""Bollinger Band mean-reversion strategy with a close-at-mean exit.

Three-state finite state machine (LONG / FLAT / SHORT) walking the bar
series forward. The "must be flat to enter" rule enforces fresh-touch
re-entry: after closing at the middle, no new position is taken until
price reaches a band again.

State transitions (based on mid_close vs the Bollinger bands at bar t):

  FLAT  + close <= lower  ->  LONG
  LONG  + close >= middle ->  FLAT
  FLAT  + close >= upper  ->  SHORT
  SHORT + close <= middle ->  FLAT
  otherwise                  hold the current state

Causality: the state at bar t depends only on (state at bar t-1, bands at
bar t, close at bar t). All of those are <= t. No lookahead. A lookahead
test enforces this structurally.

Overshoot: only ONE state transition per bar. If price jumps from below
the middle past the upper band in a single bar, the long-exit fires
(close >= middle) and the position becomes FLAT. A subsequent bar's
upper-band touch is required to enter SHORT. This is asserted in
`test_overshoot_exits_to_flat_then_short_on_next_touch`.

Warmup: rows where any Bollinger column is NaN keep the state FLAT.
We start FLAT at the beginning of the series by convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.strategies.base import Strategy

DEFAULT_UPPER = "bb_upper"
DEFAULT_MIDDLE = "bb_middle"
DEFAULT_LOWER = "bb_lower"
DEFAULT_PRICE = "mid_close"


class BollingerReversion(Strategy):
    """Three-state mean reversion: enter at band, exit at middle, wait for next band."""

    def __init__(
        self,
        upper_col: str = DEFAULT_UPPER,
        middle_col: str = DEFAULT_MIDDLE,
        lower_col: str = DEFAULT_LOWER,
        price_col: str = DEFAULT_PRICE,
    ) -> None:
        self.upper_col = upper_col
        self.middle_col = middle_col
        self.lower_col = lower_col
        self.price_col = price_col
        self.name = f"bollinger_reversion({lower_col},{middle_col},{upper_col})"

    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        needed = (self.price_col, self.upper_col, self.middle_col, self.lower_col)
        missing = [c for c in needed if c not in features.columns]
        if missing:
            raise KeyError(
                f"BollingerReversion requires columns {missing}; not in feature frame"
            )

        price = features[self.price_col].to_numpy(dtype=np.float64)
        upper = features[self.upper_col].to_numpy(dtype=np.float64)
        middle = features[self.middle_col].to_numpy(dtype=np.float64)
        lower = features[self.lower_col].to_numpy(dtype=np.float64)

        n = len(features)
        signals = np.zeros(n, dtype=np.int8)
        state = 0  # 0=flat, +1=long, -1=short. Start flat.

        for t in range(n):
            # Any NaN in the bands or price -> warmup row, force flat.
            if (
                np.isnan(upper[t])
                or np.isnan(middle[t])
                or np.isnan(lower[t])
                or np.isnan(price[t])
            ):
                state = 0
                signals[t] = 0
                continue

            p = price[t]
            if state == 0:
                if p <= lower[t]:
                    state = 1
                elif p >= upper[t]:
                    state = -1
                # else: stay flat
            elif state == 1:
                if p >= middle[t]:
                    state = 0
                # else: hold long
            elif state == -1:
                if p <= middle[t]:
                    state = 0
                # else: hold short

            signals[t] = state

        return pd.Series(signals, index=features.index, name="signal", dtype=np.int8)
