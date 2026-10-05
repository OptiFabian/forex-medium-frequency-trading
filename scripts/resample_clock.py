"""Resample all six pairs to arbitrary CLOCK timeframes and build features.

Generalizes `scripts/resample_15min.py` (which is pinned to 15min) to any
fixed clock window. Used to add the 30-minute and 1-hour rungs to the
timeframe study; the discipline is identical to the 15-min and session
resamplers:

- Load each pair's 1-minute merged bid/ask over the standard 5-year
  cross-pair-aligned default window (NOT full history).
- Aggregate onto clean clock windows (:00/:30 for 30min, :00 for 1h),
  labeled at the window START and closed on the LEFT. A window is a fixed
  clock slot, so a bar can never span a market gap -- weekend / daily-reset
  / illiquid gaps are runs of EMPTY windows, which are dropped. Partial
  windows are kept and counted via `n_minutes`.
- Persist per-side bars to data/resampled/<freq>/ (BarsStorage layout);
  the 1-min raw data is untouched.
- Build the FULL feature pipeline FRESH on the resampled bars (a 20-period
  Bollinger on 30-min bars is a different indicator from one on 1-min bars,
  so features are never resampled) -> data/resampled/<freq>_features/.

Usage:
    uv run python scripts/resample_clock.py --freqs 30min 1h
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from fxalgo.data.resample import bars_per_window, resample_merged, side_bars
from fxalgo.data.storage import BarsStorage
from fxalgo.features.build import apply_pipeline, write_features
from fxalgo.features.loader import load_pair
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
RESAMPLE_ROOT = PROJECT_ROOT / "data" / "resampled"


def _aligned(index: pd.DatetimeIndex, freq: str) -> bool:
    """True when every label sits exactly on a `freq` clock boundary.

    Uses `floor(freq)` rather than integer arithmetic so the check is
    independent of the index's datetime resolution (parquet round-trips
    timestamps at microsecond unit, not nanosecond).
    """
    return bool((index.floor(freq) == index).all())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freqs", nargs="+", default=["30min", "1h"])
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")

    print(f"Resampling {len(args.pairs)} pairs to {', '.join(args.freqs)} bars "
          f"over the standard 5-year aligned window.\n")

    rows = []
    for pair in args.pairs:
        print(f"=== {pair} ===", flush=True)
        df1 = load_pair(pair)  # 1-min merged bid/ask, default 5-year window
        n1 = len(df1)
        print(f"  1-min bars in window: {n1:,}")

        for freq in args.freqs:
            n_per = bars_per_window(freq)
            out = resample_merged(df1, freq)
            n_out = len(out)
            nm = out["n_minutes"].to_numpy()

            for side in ("bid", "ask"):
                BarsStorage(RESAMPLE_ROOT / freq, pair, side).append(side_bars(out, side))

            feats = apply_pipeline(out.drop(columns=["n_minutes"]))
            path = write_features(feats, pair, root=RESAMPLE_ROOT / f"{freq}_features")

            checks = {
                "pair": pair,
                "freq": freq,
                "bars_1min": n1,
                "bars": n_out,
                "aligned": _aligned(out.index, freq),
                "minutes_covered": int(nm.sum()),
                "full": int((nm == n_per).sum()),
                "partial": int((nm < n_per).sum()),
                "cols": feats.shape[1],
            }
            rows.append(checks)
            print(
                f"  [{freq:>5}] bars: {n_out:,} (ratio {n1/n_out:.1f}x)  "
                f"aligned: {checks['aligned']}  "
                f"full: {checks['full']:,} ({checks['full']/n_out:.2%})  "
                f"partial: {checks['partial']:,}  "
                f"minutes covered: {checks['minutes_covered']:,}/{n1:,}\n"
                f"          features: {feats.shape[0]:,} rows x {feats.shape[1]} cols -> {path}"
            )

    summary = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    ok = (
        summary["aligned"].all()
        and (summary["minutes_covered"] == summary["bars_1min"]).all()
    )
    if not ok:
        print("WARNING: alignment or minute-coverage check failed", file=sys.stderr)
        return 1
    print("\nAll pairs resampled, persisted, feature-built; every 1-min bar accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
