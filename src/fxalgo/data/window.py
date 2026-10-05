"""Default data window: last N years, aligned to a common range across pairs.

Policy (project-wide default as of 2026-07)
-------------------------------------------
All feature building, backtests, and analyses default to the LAST
``DEFAULT_WINDOW_YEARS`` (5) years of data, on a window that is IDENTICAL
across pairs:

- The window END is the earliest last-bar timestamp across all pairs on
  disk (a min-of-max search: for each pair take its last available merged
  bar, then take the minimum of those). Truncating every pair at that
  timestamp guarantees no pair has bars the others lack at the tail.
- The window START is the same timestamp for every pair: END minus
  ``years``, clamped forward to the latest first-bar across pairs (a
  max-of-min search) so no pair is asked for bars before its history
  begins. All pairs therefore share one evaluation period.

Rationale: over a 21-year history the measured edge decayed by ~40-50%,
with a regime step-down around 2016-17, and per-trade costs fell by roughly
70% over the period. Older data reflects a market that no longer exists.

Older raw history can stay on disk. Overrides:
- ``full_history=True`` restores the whole aligned history (START becomes
  the max-of-min first bar).
- Explicit ``start`` / ``end`` arguments bypass the default window entirely.

A pair's "merged" bounds come from the inner bid/ask join the loader
performs: first = max(first bid, first ask), last = min(last bid,
last ask). Bounds are read from the first/last month files only (cheap).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fxalgo.data.storage import BarsStorage
from fxalgo.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_YEARS = 5


@dataclass(frozen=True)
class Window:
    """A [start, end] evaluation window (both bounds inclusive, UTC)."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Window start {self.start} is after end {self.end}")


def _default_root(storage_root: Path | None) -> Path:
    return Path(storage_root) if storage_root is not None else PROJECT_ROOT / "data" / "raw"


def discover_pairs(storage_root: Path | None = None) -> list[str]:
    """List pairs on disk: subdirectories of the raw root with bid AND ask data."""
    root = _default_root(storage_root)
    if not root.exists():
        return []
    pairs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "bid").is_dir() and (child / "ask").is_dir():
            pairs.append(child.name)
    return pairs


def pair_bounds(pair: str, storage_root: Path | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(first, last) merged-view bounds for one pair.

    The loader inner-joins bid and ask on timestamp, so the merged series
    cannot start before BOTH sides have data nor end after EITHER side
    stops: first = max of the two first bars, last = min of the two last
    bars.
    """
    root = _default_root(storage_root)
    bounds = {}
    for side in ("bid", "ask"):
        earliest, latest = BarsStorage(root, pair, side).timestamp_range()
        if earliest is None:
            raise ValueError(f"No {side} data on disk for {pair} under {root}")
        bounds[side] = (pd.Timestamp(earliest), pd.Timestamp(latest))
    first = max(bounds["bid"][0], bounds["ask"][0])
    last = min(bounds["bid"][1], bounds["ask"][1])
    return first, last


def common_window(
    pairs: list[str] | None = None,
    *,
    storage_root: Path | None = None,
    years: int = DEFAULT_WINDOW_YEARS,
    full_history: bool = False,
) -> Window:
    """Compute the cross-pair aligned evaluation window.

    Parameters
    ----------
    pairs:
        Pairs to align across. Default: every pair found on disk.
    years:
        Window length. Ignored when ``full_history=True``.
    full_history:
        Restore the whole aligned history (start = latest first bar
        across pairs) instead of the last ``years`` years.

    Returns
    -------
    Window whose end is the min-of-max last bar and whose start is
    identical for every pair.
    """
    if years <= 0:
        raise ValueError(f"years must be positive, got {years}")
    root = _default_root(storage_root)
    if pairs is None:
        pairs = discover_pairs(root)
    if not pairs:
        raise ValueError(f"No pairs with bid+ask data found under {root}")

    bounds = {p: pair_bounds(p, root) for p in pairs}
    common_first = max(first for first, _ in bounds.values())
    end = min(last for _, last in bounds.values())

    if full_history:
        start = common_first
    else:
        start = max(common_first, end - pd.DateOffset(years=years))

    window = Window(start=start, end=end)
    logger.info(
        "common_window(%s pairs, years=%s, full_history=%s): %s -> %s",
        len(pairs),
        years,
        full_history,
        window.start.isoformat(),
        window.end.isoformat(),
    )
    return window


def slice_window(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Slice a DatetimeIndex-ed frame to [window.start, window.end] inclusive."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("slice_window requires a DatetimeIndex")
    return df[(df.index >= window.start) & (df.index <= window.end)]
