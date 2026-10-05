"""Utilities for transforming the existing Bollinger feature columns.

The feature pipeline writes bands at a fixed 2-std width (see
`fxalgo.features.volatility.add_features`). This module lets a
strategy or sweep script reconstruct bands at ANY width k without
rebuilding the feature frame, by deriving the underlying rolling std
from the existing 2-std bands:

    bb_upper = bb_middle + 2 * std
    bb_lower = bb_middle - 2 * std

So:

    std       = (bb_upper - bb_middle) / 2.0     # exact recovery
    upper(k)  = bb_middle + k * std
    lower(k)  = bb_middle - k * std

At k = 2.0 the reconstructed bands equal the originals (to floating-
point precision). This is exercised by a test.
"""

from __future__ import annotations

import pandas as pd

BASE_K: float = 2.0


def rescale_bollinger(
    df: pd.DataFrame,
    k: float,
    upper_name: str,
    lower_name: str,
    *,
    upper_col: str = "bb_upper",
    middle_col: str = "bb_middle",
) -> pd.DataFrame:
    """Add (or overwrite) two columns containing Bollinger bands at width k.

    Mutates `df` in place and returns it for chaining.

    Derives the rolling std from the existing 2-std `bb_upper - bb_middle`
    gap. `upper_col` and `middle_col` let you point at non-default source
    columns; defaults match the feature pipeline.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    std = (df[upper_col] - df[middle_col]) / BASE_K
    df[upper_name] = df[middle_col] + k * std
    df[lower_name] = df[middle_col] - k * std
    return df
