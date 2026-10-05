"""Load and merge bid/ask bars for one pair into a single feature-ready frame.

Output frame (returned by `load_pair`):

- index:       tz-aware UTC `DatetimeIndex`, one row per minute
- bid columns: open_bid, high_bid, low_bid, close_bid
- ask columns: open_ask, high_ask, low_ask, close_ask
- mid columns: mid_open, mid_high, mid_low, mid_close = (bid + ask) / 2
- spread:      spread_close = close_ask - close_bid (clipped at 0)

The `volume` column from raw storage is dropped (spot FX has no
centralized volume; it is -1 or meaningless). Negative spreads (rare crossed-quote artifacts) are
clipped to 0 and the count is logged.

Default data window
-------------------
When neither `start` nor `end` is given and `full_history` is False, the
frame is sliced to the project-wide default window: the LAST 5 YEARS,
aligned to a common [start, end] across every pair on disk (see
`fxalgo.data.window`). Pass `full_history=True` or explicit bounds to
override. The full raw history always stays on disk.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from fxalgo.data.storage import BarsStorage
from fxalgo.data.window import DEFAULT_WINDOW_YEARS, common_window
from fxalgo.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)


def load_pair(
    pair: str,
    start: datetime | None = None,
    end: datetime | None = None,
    storage_root: Path | None = None,
    *,
    full_history: bool = False,
) -> pd.DataFrame:
    """Load bid+ask for `pair`, inner-merge on timestamp, add mid and spread.

    Parameters
    ----------
    pair:
        Pair name (e.g. "EURUSD"). Must have data on disk for both sides.
    start, end:
        Optional UTC tz-aware bounds. Both inclusive. Filters after the merge.
        When BOTH are None (and `full_history` is False), the project-wide
        default window applies: the last 5 years, cross-pair aligned
        (see `fxalgo.data.window.common_window`).
    storage_root:
        Override the default `data/raw` location (useful in tests).
    full_history:
        Opt out of the default 5-year window and load the whole history.
        Ignored when explicit `start`/`end` bounds are given.

    Returns
    -------
    Frame indexed by a tz-aware UTC `DatetimeIndex`, sorted ascending.
    """
    root = storage_root or PROJECT_ROOT / "data" / "raw"

    if start is None and end is None and not full_history:
        window = common_window(storage_root=root)
        start, end = window.start, window.end
        logger.info(
            "[%s] applying default data window %s -> %s "
            "(last %d years, cross-pair aligned; use full_history=True to override)",
            pair,
            window.start.isoformat(),
            window.end.isoformat(),
            DEFAULT_WINDOW_YEARS,
        )

    bid = BarsStorage(root, pair, "bid").read_all()
    ask = BarsStorage(root, pair, "ask").read_all()
    if bid.empty or ask.empty:
        raise ValueError(
            f"No data on disk for {pair} (bid={len(bid)}, ask={len(ask)})"
        )

    bid = bid.drop(columns=["volume"])
    ask = ask.drop(columns=["volume"])

    merged = bid.merge(
        ask,
        on="timestamp",
        how="inner",
        suffixes=("_bid", "_ask"),
        validate="one_to_one",
    )

    if start is not None:
        merged = merged[merged["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        merged = merged[merged["timestamp"] <= pd.Timestamp(end)]
    if merged.empty:
        raise ValueError(f"No bars in range for {pair} (start={start}, end={end})")

    for col in ("open", "high", "low", "close"):
        merged[f"mid_{col}"] = (merged[f"{col}_bid"] + merged[f"{col}_ask"]) / 2.0

    raw_spread = merged["close_ask"] - merged["close_bid"]
    n_negative = int((raw_spread < 0).sum())
    if n_negative > 0:
        logger.info(
            "[%s] clipped %d negative spreads to 0 (%.4f%% of %d merged rows)",
            pair,
            n_negative,
            100.0 * n_negative / len(merged),
            len(merged),
        )
    merged["spread_close"] = raw_spread.clip(lower=0.0)

    merged = merged.set_index("timestamp").sort_index()
    return merged
