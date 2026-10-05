"""THE most important test in this file tree.

Verifies the no-lookahead invariant: for any feature, the value at row t
must depend only on rows <= t. We assert this structurally by computing
features twice -- once on the original data, once after replacing all
rows strictly AFTER row T with garbage -- and verifying that every
feature value at every row in [0, T] is identical between the two runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features.build import apply_pipeline
from tests.features.conftest import make_frame


def _corrupt_rows_after(df: pd.DataFrame, t: int, value: float = 9.99) -> pd.DataFrame:
    """Return a copy of df with all OHLC/mid/spread columns after row t set to garbage."""
    out = df.copy()
    cols_to_corrupt = [
        "open_bid", "high_bid", "low_bid", "close_bid",
        "open_ask", "high_ask", "low_ask", "close_ask",
        "mid_open", "mid_high", "mid_low", "mid_close",
        "spread_close",
    ]
    # Use .iloc to avoid the IndexingError that .loc would raise on integer slicing
    # over a DatetimeIndex.
    for col in cols_to_corrupt:
        out.iloc[t + 1 :, out.columns.get_loc(col)] = value
    return out


@pytest.mark.parametrize("split_at", [50, 200, 400])
def test_no_lookahead_in_full_pipeline(random_walk_frame, split_at):
    """All feature values at rows [0..split_at] must be unchanged when later rows are corrupted."""
    original = random_walk_frame
    corrupted = _corrupt_rows_after(original, split_at)

    feats_original = apply_pipeline(original)
    feats_corrupted = apply_pipeline(corrupted)

    # Only check columns added by the pipeline -- raw input columns will obviously
    # differ in the corrupted version (we changed them).
    feature_only_cols = [c for c in feats_original.columns if c not in original.columns]
    assert len(feature_only_cols) > 0

    head_original = feats_original.iloc[: split_at + 1][feature_only_cols]
    head_corrupted = feats_corrupted.iloc[: split_at + 1][feature_only_cols]

    # NaN positions must match identically.
    nan_diff = head_original.isna() != head_corrupted.isna()
    bad_nan_cols = nan_diff.any()
    assert not bad_nan_cols.any(), (
        "LOOKAHEAD: NaN pattern of these features changed when future rows were modified: "
        f"{bad_nan_cols[bad_nan_cols].index.tolist()}"
    )

    # For non-NaN positions, values must be bit-identical.
    for col in feature_only_cols:
        a = head_original[col].to_numpy()
        b = head_corrupted[col].to_numpy()
        mask = ~np.isnan(a)
        if not mask.any():
            continue
        # equal_nan handled by mask; use exact equality so we catch any drift.
        assert np.array_equal(a[mask], b[mask]), (
            f"LOOKAHEAD detected in column '{col}': "
            f"feature values at rows [0..{split_at}] differ when rows [{split_at + 1}:] are modified.\n"
            f"  first differing index: {int(np.argmax(a[mask] != b[mask]))}\n"
            f"  a sample: {a[mask][:5]}\n"
            f"  b sample: {b[mask][:5]}"
        )


def test_no_lookahead_in_session_flags():
    """Time/session features are pure functions of the index. Trivially causal,
    but verify explicitly so future refactors can't introduce a leak."""
    from fxalgo.features import time_features

    df = make_frame(np.linspace(1.0, 2.0, 200))
    feats1 = time_features.add_features(df)
    corrupted = _corrupt_rows_after(df, t=100)
    feats2 = time_features.add_features(corrupted)
    # Time/session columns only depend on the index, so the WHOLE column must
    # match -- not just the head.
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "session_tokyo", "session_london", "session_ny",
                "session_london_ny_overlap"]:
        np.testing.assert_array_equal(feats1[col].to_numpy(), feats2[col].to_numpy())
