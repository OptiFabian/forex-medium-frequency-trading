"""Assemble all feature families into one frame and write it to disk.

Warmup / NaN policy
-------------------
Some features need a fixed number of past bars to be defined. We **keep
the warmup rows as NaN** rather than dropping them. Rationale:

- The output frame stays row-aligned with the raw input -- easier to
  join with future targets/labels and easier to debug.
- It is information-free to drop rows; we'd lose the timestamp and the
  raw OHLC of those minutes for no good reason.
- Downstream code (model training, backtest) can `.dropna()` or
  `.iloc[N:]` according to its own needs.

The longest pure-rolling-window warmup is 1440 bars (don_*_1440; the
non-Donchian maximum is 60: ret_std_60, spread_*_60). EMAs are defined
from row 0 but are biased toward the initial value for roughly 3x their
span (so EMA200 is "biased" for the first ~600 rows). Document this in
the consuming model if it matters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fxalgo.features import (
    adx,
    donchian,
    momentum,
    spread_features,
    time_features,
    trend,
    volatility,
    wick,
)
from fxalgo.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)


def apply_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature family in order. Pure: does not touch disk."""
    df = trend.add_features(df)
    df = volatility.add_features(df)
    df = momentum.add_features(df)
    df = adx.add_features(df)
    df = donchian.add_features(df)
    df = wick.add_features(df)
    df = spread_features.add_features(df)
    df = time_features.add_features(df)
    return df


def features_path(pair: str, root: Path | None = None) -> Path:
    """Canonical on-disk location of a pair's feature parquet."""
    base = root or PROJECT_ROOT / "data" / "features"
    return base / f"{pair}.parquet"


def write_features(df: pd.DataFrame, pair: str, root: Path | None = None) -> Path:
    """Persist a feature frame to `data/features/<PAIR>.parquet`."""
    path = features_path(pair, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    return path
