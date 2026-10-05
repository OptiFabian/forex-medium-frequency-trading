"""Import your own 1-minute bars (one pair, one side) into the project's layout.

The pipeline needs, for every pair, 1-minute BID bars and 1-minute ASK bars
stored as monthly parquet files:

    data/raw/<PAIR>/bid/<YYYY-MM>.parquet
    data/raw/<PAIR>/ask/<YYYY-MM>.parquet

with columns `timestamp, open, high, low, close, volume`. `timestamp` is the
bar's START time; naive timestamps are interpreted as UTC (or pass --tz).
`volume` is optional and ignored by the pipeline; it is filled with -1 when
missing.

This script reads a CSV or parquet file with those columns (column names are
matched case-insensitively; `time`/`datetime`/`date` are accepted for
`timestamp`) and appends it to the store. Re-importing overlapping data is safe:
rows are de-duplicated on timestamp, keeping the newest.

    uv run python scripts/import_bars.py --pair EURUSD --side bid --file eurusd_bid.csv
    uv run python scripts/import_bars.py --pair EURUSD --side ask --file eurusd_ask.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fxalgo.data.storage import BarsStorage
from fxalgo.settings import PROJECT_ROOT

TIME_ALIASES = ("timestamp", "time", "datetime", "date")


def read_bars(path: Path, tz: str = "UTC") -> pd.DataFrame:
    """Read a CSV/parquet of 1-minute OHLC bars into the storage schema."""
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in TIME_ALIASES if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"no timestamp column (looked for {TIME_ALIASES}) in {path}")
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns {missing} in {path}")
    ts = pd.to_datetime(df[tcol])
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts
    out = pd.DataFrame({
        "timestamp": ts.dt.tz_convert("UTC"),
        "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
        "volume": df["volume"] if "volume" in df.columns else -1.0,
    })
    bad = (out["high"] < out[["open", "close"]].max(axis=1)) | (out["low"] > out[["open", "close"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have high/low inconsistent with open/close")
    return out.sort_values("timestamp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", required=True, help="e.g. EURUSD")
    ap.add_argument("--side", required=True, choices=("bid", "ask"))
    ap.add_argument("--file", required=True, type=Path)
    ap.add_argument("--tz", default="UTC", help="time zone of naive timestamps (default UTC)")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = ap.parse_args()
    bars = read_bars(args.file, tz=args.tz)
    n = BarsStorage(args.root, args.pair.upper(), args.side).append(bars)
    print(f"{args.pair.upper()} {args.side}: {len(bars):,} rows read, {n:,} new rows stored under {args.root}")


if __name__ == "__main__":
    main()
