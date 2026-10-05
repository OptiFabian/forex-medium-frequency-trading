"""STEP 1+2: resample all six pairs to 15-min bars and build 15-min features.

Step 1 -- resample: load each pair's 1-minute merged bid/ask (windowed to the
5-year aligned default), aggregate to clean 15-min clock windows (:00/:15/
:30/:45) gap-safely (see fxalgo.data.resample), and PERSIST the per-side
bars to data/resampled/15min/<PAIR>/<side>/ (BarsStorage layout) -- the
1-min raw data is not touched.

Step 2 -- features: compute the FULL feature pipeline FRESH on the 15-min
bars (a 20-period Bollinger on 15-min bars is a different indicator than on
1-min -- we do NOT resample 1-min features), and write to
data/resampled/15min_features/<PAIR>.parquet.

Reports per pair: 15-min row count, boundary-alignment check, partial-bar
stats (n_minutes distribution), and the feature frame shape.

Usage:
    uv run python scripts/resample_15min.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fxalgo.data.resample import bars_per_window, resample_merged, side_bars
from fxalgo.data.storage import BarsStorage
from fxalgo.features.build import apply_pipeline, features_path, write_features
from fxalgo.features.loader import load_pair
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

FREQ = "15min"
PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
RESAMPLE_ROOT = PROJECT_ROOT / "data" / "resampled" / "15min"
FEAT15_ROOT = PROJECT_ROOT / "data" / "resampled" / "15min_features"


def main() -> int:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    n_per = bars_per_window(FREQ)

    print(f"Resampling six pairs to {FREQ} bars ({n_per} 1-min bars per window)")
    print(f"  bars  -> {RESAMPLE_ROOT}")
    print(f"  feats -> {FEAT15_ROOT}\n")

    summary = []
    for pair in PAIRS:
        print(f"=== {pair} ===", flush=True)
        df1 = load_pair(pair)  # windowed 1-min merged bid/ask
        n1 = len(df1)
        resampled = resample_merged(df1, FREQ)
        n15 = len(resampled)

        # Boundary-alignment check.
        mins = set(np.unique(resampled.index.minute).tolist())
        secs_ok = bool((resampled.index.second == 0).all())
        aligned = mins.issubset({0, 15, 30, 45}) and secs_ok

        # Partial-bar stats.
        nm = resampled["n_minutes"].to_numpy()
        n_full = int((nm == n_per).sum())
        n_partial = int((nm < n_per).sum())
        n_thin = int((nm < 5).sum())

        # Persist per-side bars.
        for side in ("bid", "ask"):
            BarsStorage(RESAMPLE_ROOT, pair, side).append(side_bars(resampled, side))

        # Build features fresh on the 15-min bars.
        feats = apply_pipeline(resampled.drop(columns=["n_minutes"]))
        path = write_features(feats, pair, root=FEAT15_ROOT)

        print(
            f"  1-min bars: {n1:,}  ->  {FREQ} bars: {n15:,}  "
            f"(ratio {n1/n15:.1f}x)\n"
            f"  boundaries clean: {aligned} (minutes seen: {sorted(mins)})\n"
            f"  full {n_per}-min bars: {n_full:,} ({n_full/n15:.1%})  "
            f"partial: {n_partial:,} ({n_partial/n15:.1%})  "
            f"thin(<5min): {n_thin:,} ({n_thin/n15:.2%})\n"
            f"  feature frame: {feats.shape[0]:,} rows x {feats.shape[1]} cols  -> {path.name}\n"
            f"  range: {resampled.index[0]} -> {resampled.index[-1]}"
        )
        summary.append(
            {"pair": pair, "bars_1min": n1, "bars_15min": n15, "aligned": aligned,
             "full": n_full, "partial": n_partial, "thin": n_thin, "cols": feats.shape[1]}
        )

    print("\n=== SUMMARY ===")
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    if not sdf["aligned"].all():
        print("WARNING: some pairs are not cleanly boundary-aligned!", file=sys.stderr)
        return 1
    print("\nAll pairs resampled, persisted, and feature-built; all boundary-aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
