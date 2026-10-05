"""Abstract strategy interface.

A strategy is a pure function from a feature frame to a sequence of
position signals: +1 (long), -1 (short), 0 (flat). One signal per row.

Causality invariant
-------------------
The signal at row t must be computable using ONLY rows <= t in the
input frame. Backtests rely on this — see `tests/features/test_no_lookahead.py`
for the framework-wide enforcement.

How the engine consumes signals
-------------------------------
The signal at row t is executed at the OPEN of bar t+1. That delay is
the engine's responsibility, not the strategy's. The strategy simply
emits, for each row t, the position it would want to hold given the
information available at the close of bar t.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, features: pd.DataFrame) -> pd.Series:
        """Return a Series of integers in {-1, 0, +1} aligned to `features.index`.

        Implementations must be causal: signal at row t depends only on rows <= t.
        """
