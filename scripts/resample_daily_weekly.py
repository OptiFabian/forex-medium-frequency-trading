"""STEP 1+2: resample all six pairs to DAILY and WEEKLY bars, and build features.

Step 1 -- resample. Load each pair's 1-minute merged bid/ask over the standard
5-year cross-pair-aligned default window (NOT full history) and aggregate it
onto FX trading sessions:

  DAILY : 17:00 New York close convention -- each bar spans [D-1 17:00 NY,
          D 17:00 NY), the same anchor `fxalgo.data.quality` uses to classify
          daily-reset gaps. DST-correct (21:00 UTC in summer, 22:00 in winter).
  WEEKLY: Sunday 17:00 NY open through Friday 17:00 NY close.

Every 1-minute bar inside the window contributes to exactly one daily and one
weekly bar -- nothing is downsampled or dropped. Sessions with no minutes at
all (weekends, holidays) simply produce no bar; no bar spans a market gap.
Partial sessions (window edges, holiday half-days) are kept and counted via
`n_minutes`. Per-side bars are persisted to data/resampled/{daily,weekly}/
in the BarsStorage layout; the 1-min raw data is untouched.

Step 2 -- features. The FULL feature pipeline is computed FRESH on each
timeframe (a 20-period Bollinger on daily bars is a ~1-month band; on weekly
bars a ~5-month band -- these are different indicators from the 1-min one, so
we never resample features). Written to
data/resampled/{daily,weekly}_features/<PAIR>.parquet.

Usage:
    uv run python scripts/resample_daily_weekly.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from fxalgo.data.quality import NY_TZ
from fxalgo.data.resample import resample_sessions, side_bars
from fxalgo.data.storage import BarsStorage
from fxalgo.features.build import apply_pipeline, write_features
from fxalgo.features.loader import load_pair
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
PERIODS = ("daily", "weekly")
RESAMPLE_ROOT = PROJECT_ROOT / "data" / "resampled"
# A "thin" session is one holding less than this fraction of a full session's
# minutes -- reported, never dropped.
THIN_FRACTION = 0.25
NOMINAL_MINUTES = {"daily": 1440, "weekly": 5 * 1440}


def _root(period: str, features: bool = False) -> Path:
    name = f"{period}_features" if features else period
    return RESAMPLE_ROOT / name


def _validate(out: pd.DataFrame, period: str) -> dict:
    """Structural checks on a session-resampled frame."""
    labels_ny = out.index.tz_convert(NY_TZ)
    at_1700 = bool(((labels_ny.hour == 17) & (labels_ny.minute == 0)).all())
    # Daily labels may fall on any trading day; weekly labels must be Sundays.
    weekday_ok = bool((labels_ny.weekday == 6).all()) if period == "weekly" else True
    monotonic = bool(out.index.is_monotonic_increasing and out.index.is_unique)

    nm = out["n_minutes"]
    thin_cut = THIN_FRACTION * NOMINAL_MINUTES[period]
    return {
        "bars": len(out),
        "label_at_1700_ny": at_1700,
        "label_weekday_ok": weekday_ok,
        "monotonic_unique": monotonic,
        "min_minutes": int(nm.min()),
        "median_minutes": int(nm.median()),
        "max_minutes": int(nm.max()),
        "thin_sessions": int((nm < thin_cut).sum()),
        "start": out.index[0],
        "end": out.index[-1],
    }


def main() -> int:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")

    print("Resampling six pairs to DAILY (17:00 NY close) and WEEKLY (Sun 17:00 ->")
    print("Fri 17:00 NY) bars over the standard 5-year aligned window.\n")

    rows = []
    for pair in PAIRS:
        print(f"=== {pair} ===", flush=True)
        df1 = load_pair(pair)  # 1-min merged bid/ask, default 5-year window
        n1 = len(df1)
        print(f"  1-min bars in window: {n1:,}  ({df1.index[0]} -> {df1.index[-1]})")

        covered = 0
        for period in PERIODS:
            out = resample_sessions(df1, period)
            checks = _validate(out, period)
            covered = int(out["n_minutes"].sum())

            for side in ("bid", "ask"):
                BarsStorage(_root(period), pair, side).append(side_bars(out, side))

            feats = apply_pipeline(out.drop(columns=["n_minutes"]))
            path = write_features(feats, pair, root=_root(period, features=True))

            print(
                f"  [{period:<6}] bars: {checks['bars']:>6,}  "
                f"(ratio {n1 / checks['bars']:>7,.1f}x)  "
                f"minutes/bar min/med/max: {checks['min_minutes']:,}/"
                f"{checks['median_minutes']:,}/{checks['max_minutes']:,}  "
                f"thin: {checks['thin_sessions']}\n"
                f"           labels at 17:00 NY: {checks['label_at_1700_ny']}  "
                f"weekday ok: {checks['label_weekday_ok']}  "
                f"sorted/unique: {checks['monotonic_unique']}  "
                f"minutes covered: {covered:,}/{n1:,}\n"
                f"           range: {checks['start']} -> {checks['end']}\n"
                f"           features: {feats.shape[0]:,} rows x {feats.shape[1]} cols -> {path}"
            )
            rows.append({"pair": pair, "period": period, "bars_1min": n1,
                         "minutes_covered": covered, **checks})

    summary = pd.DataFrame(rows)

    print("\n=== BAR COUNTS ===")
    counts = summary.pivot(index="pair", columns="period", values="bars")
    counts.insert(0, "1min", summary.groupby("pair")["bars_1min"].first())
    print(counts.to_string())

    print("\n=== STRUCTURAL CHECKS ===")
    cols = ["pair", "period", "bars", "minutes_covered", "bars_1min",
            "label_at_1700_ny", "label_weekday_ok", "monotonic_unique",
            "min_minutes", "median_minutes", "max_minutes", "thin_sessions"]
    print(summary[cols].to_string(index=False))

    ok = (
        summary["label_at_1700_ny"].all()
        and summary["label_weekday_ok"].all()
        and summary["monotonic_unique"].all()
        and (summary["minutes_covered"] == summary["bars_1min"]).all()
    )
    csv_path = PROJECT_ROOT / "data" / "backtests" / "daily_weekly_resample_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    print(f"\nWrote summary to {csv_path}")

    if not ok:
        print("WARNING: a structural check failed (see table above)", file=sys.stderr)
        return 1
    print("All pairs resampled, persisted, feature-built; every 1-min bar accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
