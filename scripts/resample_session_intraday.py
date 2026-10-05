"""Build session-anchored 2-hour and 4-hour bars + features for all six pairs.

Needed by the higher-timeframe features (`features.htf`). Bars are cut
every 2 / 4 NY wall-clock hours from the 17:00 America/New_York session open
(see `fxalgo.data.resample.resample_sessions`), so they nest exactly inside the
daily session, stay pinned to it across DST, and never span the weekend or a
holiday gap. A plain clock resample would instead anchor to midnight UTC and
straddle the 17:00 rollover.

Bars are persisted to data/resampled/{2h,4h}/ (BarsStorage layout) and the FULL
feature pipeline is computed FRESH on them (a 20-period Bollinger on 4h bars is
a different indicator from one on 1h bars) into data/resampled/{2h,4h}_features/.

Usage:
    uv run python scripts/resample_session_intraday.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from fxalgo.data.quality import NY_TZ
from fxalgo.data.resample import (
    SESSION_INTRADAY_HOURS,
    resample_sessions,
    session_keys,
    side_bars,
)
from fxalgo.data.storage import BarsStorage
from fxalgo.features.build import apply_pipeline, write_features
from fxalgo.features.loader import load_pair
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
PERIODS = ("2h", "4h")
RESAMPLE_ROOT = PROJECT_ROOT / "data" / "resampled"


def _root(period: str, features: bool = False) -> Path:
    return RESAMPLE_ROOT / (f"{period}_features" if features else period)


def main() -> int:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")

    print("Building session-anchored 2h / 4h bars (17:00 NY grid) for six pairs\n")
    rows = []
    for pair in PAIRS:
        df1 = load_pair(pair)  # 1-min merged bid/ask, default 5-year window
        n1 = len(df1)
        sessions = session_keys(df1.index, "daily").nunique()
        print(f"=== {pair} === 1-min bars {n1:,} across {sessions:,} trading sessions")

        for period in PERIODS:
            hours = SESSION_INTRADAY_HOURS[period]
            out = resample_sessions(df1, period)
            labels_ny = out.index.tz_convert(NY_TZ)
            offsets = ((labels_ny.hour - 17) % 24)
            on_grid = bool((offsets % hours == 0).all())
            per_session = out.groupby(session_keys(out.index, "daily")).size()

            for side in ("bid", "ask"):
                BarsStorage(_root(period), pair, side).append(side_bars(out, side))
            feats = apply_pipeline(out.drop(columns=["n_minutes"]))
            path = write_features(feats, pair, root=_root(period, features=True))

            rows.append({
                "pair": pair, "period": period, "bars_1min": n1, "sessions": sessions,
                "bars": len(out), "on_grid": on_grid,
                "minutes_covered": int(out["n_minutes"].sum()),
                "bars_per_session_mode": int(per_session.mode().iloc[0]),
                "max_bars_per_session": int(per_session.max()),
                "cols": feats.shape[1],
            })
            print(
                f"  [{period}] bars {len(out):,}  (ratio {n1/len(out):.1f}x)  "
                f"on 17:00-NY grid: {on_grid}  "
                f"bars/session mode {int(per_session.mode().iloc[0])} "
                f"max {int(per_session.max())} (cap {24 // hours})  "
                f"minutes {int(out['n_minutes'].sum()):,}/{n1:,}\n"
                f"        features {feats.shape[0]:,} x {feats.shape[1]} -> {path.name}"
            )

    summary = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))

    ok = (
        summary["on_grid"].all()
        and (summary["minutes_covered"] == summary["bars_1min"]).all()
        and (summary["max_bars_per_session"] <= summary["period"].map(
            lambda p: 24 // SESSION_INTRADAY_HOURS[p])).all()
    )
    if not ok:
        print("WARNING: a structural check failed", file=sys.stderr)
        return 1
    print("\nAll pairs: bars on the 17:00-NY grid, every 1-min bar accounted for, "
          "no session over its bar cap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
