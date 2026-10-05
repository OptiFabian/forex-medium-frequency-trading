"""Persistence for market data.

Bars are stored as Parquet under a per-pair, per-side, per-month layout:

    <root>/<PAIR>/<SIDE>/<YYYY-MM>.parquet

Each file holds a DataFrame with columns:
    timestamp (UTC, tz-aware, unique) | open | high | low | close | volume

Monthly chunking keeps individual files small enough to load quickly while
avoiding tens of thousands of files over multi-year history. Within a file,
rows are sorted by timestamp and deduplicated on conflict (keep='last',
since re-imported data is assumed to be a correction).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BAR_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


def empty_bars_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical bar schema."""
    df = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a bars DataFrame to canonical dtypes + UTC tz-aware timestamps."""
    if df.empty:
        return empty_bars_df()
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Bars DataFrame missing required columns: {missing}")
    out = df.loc[:, list(BAR_COLUMNS)].copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = ts
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype("float64")
    return out


def merge_dataframes(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge two bars DataFrames, sort by timestamp, dedup keep='last'."""
    new = _normalize(new)
    if existing is None or existing.empty:
        combined = new
    else:
        combined = pd.concat([_normalize(existing), new], ignore_index=True)
    combined = combined.sort_values("timestamp", kind="mergesort")
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.reset_index(drop=True)
    return combined


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to Parquet at `path`, creating parent dirs.

    No merging — overwrites the file. Use `BarsStorage.append` for the
    merge-and-dedup workflow against existing files.
    """
    df = _normalize(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file written by `write_parquet`. Returns empty df if missing."""
    if not path.exists():
        return empty_bars_df()
    df = pd.read_parquet(path, engine="pyarrow")
    return _normalize(df)


class BarsStorage:
    """Filesystem-backed storage for one (pair, side) bar series.

    Layout: ``<root>/<pair>/<side>/<YYYY-MM>.parquet``
    """

    def __init__(self, root: Path, pair: str, side: str) -> None:
        self.root = Path(root)
        self.pair = pair
        self.side = side.lower()
        self.dir = self.root / pair / self.side

    def _month_path(self, year: int, month: int) -> Path:
        return self.dir / f"{year:04d}-{month:02d}.parquet"

    def _all_month_files(self) -> list[Path]:
        if not self.dir.exists():
            return []
        return sorted(self.dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].parquet"))

    def append(self, df: pd.DataFrame) -> int:
        """Merge `df` into per-month files. Returns total rows newly persisted.

        "Newly persisted" = rows in the merged result minus rows previously
        on disk for the affected months. A re-fetch of identical data
        returns 0.
        """
        df = _normalize(df)
        if df.empty:
            return 0

        rows_added = 0
        for (year, month), group in df.groupby(
            [df["timestamp"].dt.year, df["timestamp"].dt.month], sort=True
        ):
            path = self._month_path(int(year), int(month))
            existing = read_parquet(path)
            before = len(existing)
            merged = merge_dataframes(existing, group)
            write_parquet(merged, path)
            rows_added += len(merged) - before
        return rows_added

    def read_all(self) -> pd.DataFrame:
        """Read every month file for this (pair, side) into one DataFrame."""
        files = self._all_month_files()
        if not files:
            return empty_bars_df()
        parts = [read_parquet(p) for p in files]
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        return out

    def timestamp_range(self) -> tuple[datetime | None, datetime | None]:
        """Return (earliest, latest) timestamps on disk, or (None, None) if empty.

        Only inspects the first and last month files for speed — assumes
        files are written in sorted order (which `append` guarantees).
        """
        files = self._all_month_files()
        if not files:
            return None, None
        first = read_parquet(files[0])
        last = read_parquet(files[-1])
        if first.empty or last.empty:
            return None, None
        earliest = first["timestamp"].iloc[0].to_pydatetime()
        latest = last["timestamp"].iloc[-1].to_pydatetime()
        return earliest, latest

    def row_count(self) -> int:
        """Total rows on disk for this (pair, side)."""
        # Cheaper than read_all(): just sums row counts of each file.
        total = 0
        for path in self._all_month_files():
            total += len(read_parquet(path))
        return total
