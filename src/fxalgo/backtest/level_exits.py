"""Take-profit / stop-loss level policies for intrabar exits.

An `ExitLevelPolicy` returns (take_profit, stop_loss) mid-price levels at
the moment a position is opened. The engine then checks each subsequent
bar's mid_high/mid_low range to see whether either level was touched.

Conventions
-----------
- TP for a LONG position is ABOVE the entry; for a SHORT it is BELOW.
- SL for a LONG is BELOW the entry; for a SHORT is ABOVE.
- A `None` for either field means "no level on this side".
- All levels are in MID-PRICE terms. The engine adjusts the fill by the
  bar's bid-ask half-spread (long exits sell at bid; short exits buy at
  ask) -- consistent with the signal-based fill convention.

Same-bar conflict
-----------------
If a single bar's range touches BOTH the TP and the SL, the STOP wins
(pessimistic assumption). See `engine._check_intrabar_exit` and its test.

Policies provided
-----------------
- `NoLevels`   : no intrabar exits (pure signal-driven exits).
- `AtrBarrier` : symmetric TP and SL at entry +/- k * ATR, where the ATR is
                 the one known at the DECISION bar. With k = 2.0 this is the
                 triple barrier the XGBoost model is trained to predict, so
                 its trades are scored on the outcome it forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class ExitLevelPolicy(Protocol):
    """Computes TP/SL levels at the moment a position opens."""

    def levels_at_entry(
        self,
        *,
        direction: int,
        entry_price: float,
        bar: pd.Series,
    ) -> tuple[float | None, float | None]:
        """Return (tp, sl) in mid-price terms. Either may be None.

        `bar` is the feature row at the entry bar (i.e., `features.iloc[bar_index]`).
        """
        ...


@dataclass(frozen=True)
class NoLevels:
    """No-op policy: never sets intrabar exits."""

    def levels_at_entry(
        self,
        *,
        direction: int,
        entry_price: float,
        bar: pd.Series,
    ) -> tuple[float | None, float | None]:
        return None, None


@dataclass(frozen=True)
class AtrBarrier:
    """Symmetric TP/SL at entry +/- k * ATR of the decision bar.

    The engine passes the ENTRY bar's row, whose ATR already includes the
    entry bar's own close -- information not available when the position
    opens at that bar's open. `atr_col` must therefore hold the PREVIOUS
    bar's ATR (the bar whose close produced the signal); build it with
    `features["atr_decision"] = features["atr_14"].shift(1)`.
    """

    k: float = 2.0
    atr_col: str = "atr_decision"

    def levels_at_entry(
        self,
        *,
        direction: int,
        entry_price: float,
        bar: pd.Series,
    ) -> tuple[float | None, float | None]:
        atr = float(bar[self.atr_col])
        if not np.isfinite(atr) or atr <= 0 or direction not in (1, -1):
            return None, None
        return entry_price + direction * self.k * atr, entry_price - direction * self.k * atr
