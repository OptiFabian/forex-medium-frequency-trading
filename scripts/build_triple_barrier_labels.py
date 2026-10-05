"""Build and diagnose triple-barrier labels for EURUSD. LABELS ONLY.

No features, no event sampling, no train/test split, no model, no backtest.

Anchors on mid open(t+1) (the engine's fill price), barriers at
anchor +/- k * atr_at_entry for k in {1.0, 1.5, 2.0, 2.5, 3.0} with ATR taken
from the last 15-minute bar CLOSED at or before t, horizon 240 traded bars.
See `fxalgo.labels.triple_barrier` for the full contract.

Verifies the vectorised scan against the slow reference implementation on a
random sample before reporting anything.

Usage:
    uv run python scripts/build_triple_barrier_labels.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fxalgo.features.build import features_path
from fxalgo.labels.triple_barrier import (
    HORIZON,
    KS,
    _k_tag,
    build_labels,
    scan_reference,
)
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIR = "EURUSD"
ATR_SRC = PROJECT_ROOT / "data" / "resampled" / "15min_features" / f"{PAIR}.parquet"
OUT = PROJECT_ROOT / "data" / "labels" / f"{PAIR.lower()}_triple_barrier.parquet"
VERIFY_EVENTS = 10_000
TAGS = [_k_tag(k) for k in KS]


def main() -> int:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")

    bars = pd.read_parquet(features_path(PAIR), engine="pyarrow")
    atr15 = pd.read_parquet(ATR_SRC, engine="pyarrow")["atr_14"]
    n = len(bars)
    print(f"{PAIR} 1-min bars: {n:,}  ({bars.index[0]} -> {bars.index[-1]})")
    print(f"15-min ATR source: {len(atr15):,} bars\n")

    t0 = time.perf_counter()
    lab = build_labels(bars, atr15)
    elapsed = time.perf_counter() - t0
    n_events = n - HORIZON
    print("Scan: chunked numpy (sliding_window_view + cumulative forward max/min,")
    print(f"      one pass over the path, all {len(KS)} barrier sets evaluated per bar).")
    print(f"      {n_events:,} events x {HORIZON} bars = "
          f"{n_events * HORIZON / 1e6:,.0f}M bar-visits in {elapsed:,.1f}s wall clock.\n")

    # ---- verify the fast path against the slow reference ----
    rng = np.random.default_rng(20260809)
    events = rng.choice(n_events, size=VERIFY_EVENTS, replace=False)
    t1 = time.perf_counter()
    ref = scan_reference(
        bars["mid_high"].to_numpy(), bars["mid_low"].to_numpy(),
        lab["anchor_price"].to_numpy(), lab["atr_at_entry"].to_numpy(),
        KS, HORIZON, events=events)
    mismatches = 0
    for k in KS:
        tag = _k_tag(k)
        for col, key in ((f"label_{tag}", "label"), (f"bars_to_touch_{tag}", "bars"),
                         (f"ambiguous_{tag}", "ambiguous")):
            a = lab[col].to_numpy()[events]
            b = ref[k][key]
            if key == "ambiguous":
                mismatches += int((a.astype(bool) != b.astype(bool)).sum())
            else:
                mismatches += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
    print(f"VERIFICATION vs slow reference on {VERIFY_EVENTS:,} random events "
          f"({time.perf_counter() - t1:,.1f}s): "
          f"{'EXACT AGREEMENT' if mismatches == 0 else f'{mismatches} MISMATCHES'} "
          f"across {len(KS) * 3} label columns.\n")
    if mismatches:
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(OUT, engine="pyarrow")

    unlabelled = int(lab[f"label_{TAGS[0]}"].isna().sum())
    print(f"events emitted        : {n:,}")
    print(f"labelled              : {n - unlabelled:,}")
    print(f"NaN (window runs off the end of the data): {unlabelled:,}\n")

    lab_ok = lab[lab[f"label_{TAGS[0]}"].notna()]

    # ---- A) class distribution ----
    print("=" * 92)
    print("A) CLASS DISTRIBUTION PER k")
    print("=" * 92)
    print(f"  {'k':>5}{'+1':>12}{'-1':>12}{'0 (timeout)':>14}"
          f"{'+1 %':>9}{'-1 %':>9}{'0 %':>9}")
    for tag in TAGS:
        v = lab_ok[f"label_{tag}"]
        c = {x: int((v == x).sum()) for x in (1, -1, 0)}
        tot = len(v)
        print(f"  {tag:>5}{c[1]:>12,d}{c[-1]:>12,d}{c[0]:>14,d}"
              f"{c[1]/tot:>9.2%}{c[-1]/tot:>9.2%}{c[0]/tot:>9.2%}")

    # ---- B/C/D/H ----
    print("\n" + "=" * 92)
    print("B) TOUCH SPEED   C) AMBIGUITY   D) TIMEOUT   H) +1 vs -1 BALANCE")
    print("=" * 92)
    print(f"  {'k':>5}{'median bars':>14}{'mean bars':>12}{'ambiguous':>12}"
          f"{'ambig %':>10}{'timeout %':>12}{'+1/(+1+-1)':>13}{'imbalance':>12}")
    for tag in TAGS:
        v = lab_ok[f"label_{tag}"]
        b = lab_ok[f"bars_to_touch_{tag}"].dropna()
        amb = lab_ok[f"ambiguous_{tag}"]
        up, dn = int((v == 1).sum()), int((v == -1).sum())
        share = up / (up + dn) if up + dn else float("nan")
        flag = "  <-- >2%" if amb.mean() > 0.02 else ""
        print(f"  {tag:>5}{b.median():>14.1f}{b.mean():>12.1f}{int(amb.sum()):>12,d}"
              f"{amb.mean():>10.3%}{(v == 0).mean():>12.2%}{share:>13.4f}"
              f"{abs(share - 0.5) * 2:>12.2%}{flag}")

    # ---- E) by gap / friday ----
    print("\n" + "=" * 92)
    print("E) CLASS DISTRIBUTION BY spans_gap AND is_friday  (k = 2.0)")
    print("=" * 92)
    for col in ("spans_gap", "is_friday"):
        print(f"\n  by {col}:")
        print(f"    {col:>10}{'events':>12}{'+1 %':>9}{'-1 %':>9}{'0 %':>9}"
              f"{'median bars':>14}")
        for val, g in lab_ok.groupby(col, observed=True):
            v = g["label_2.0"]
            print(f"    {str(val):>10}{len(g):>12,d}{(v == 1).mean():>9.2%}"
                  f"{(v == -1).mean():>9.2%}{(v == 0).mean():>9.2%}"
                  f"{g['bars_to_touch_2.0'].median():>14.1f}")
    print("\n  gap_type breakdown (k = 2.0):")
    print(f"    {'gap_type':>12}{'events':>12}{'+1 %':>9}{'-1 %':>9}{'0 %':>9}")
    for val, g in lab_ok.groupby("gap_type", observed=True):
        v = g["label_2.0"]
        print(f"    {str(val):>12}{len(g):>12,d}{(v == 1).mean():>9.2%}"
              f"{(v == -1).mean():>9.2%}{(v == 0).mean():>9.2%}")

    # ---- F) by year ----
    print("\n" + "=" * 92)
    print("F) CLASS DISTRIBUTION BY CALENDAR YEAR (k = 2.0)")
    print("=" * 92)
    print(f"  {'year':>6}{'events':>12}{'+1 %':>9}{'-1 %':>9}{'0 %':>9}"
          f"{'median bars':>14}{'ambig %':>10}")
    by_year = lab_ok.groupby(lab_ok.index.year)
    for yr, g in by_year:
        v = g["label_2.0"]
        print(f"  {yr:>6}{len(g):>12,d}{(v == 1).mean():>9.2%}{(v == -1).mean():>9.2%}"
              f"{(v == 0).mean():>9.2%}{g['bars_to_touch_2.0'].median():>14.1f}"
              f"{g['ambiguous_2.0'].mean():>10.3%}")

    # ---- G) atr distribution and implied barrier widths ----
    print("\n" + "=" * 92)
    print("G) atr_at_entry DISTRIBUTION AND IMPLIED BARRIER WIDTH")
    print("=" * 92)
    atr = lab_ok["atr_at_entry"]
    anchor = lab_ok["anchor_price"]
    atr_bps = (atr / anchor * 1e4)
    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print("  atr_at_entry (price): " + "  ".join(
        f"p{int(q*100)}={atr.quantile(q):.6f}" for q in qs))
    print("  atr_at_entry (bps)  : " + "  ".join(
        f"p{int(q*100)}={atr_bps.quantile(q):.2f}" for q in qs))
    print("\n  implied HALF-WIDTH of the barrier in bps (k x atr / anchor):")
    print(f"  {'k':>5}{'p5':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}")
    for k, tag in zip(KS, TAGS, strict=True):
        w = atr_bps * k
        print(f"  {tag:>5}{w.quantile(0.05):>9.2f}{w.quantile(0.25):>9.2f}"
              f"{w.quantile(0.50):>9.2f}{w.quantile(0.75):>9.2f}{w.quantile(0.95):>9.2f}")

    # ---- realized vs clock-predicted gaps (diagnostic; see module docstring) ----
    steps = bars.index.to_series().diff()
    realized = int((steps > pd.Timedelta("1min")).sum())
    print(f"\n  realized index discontinuities: {realized:,} "
          f"({realized / n:.3%} of bars). The clock-derived gap columns predict "
          f"daily-reset and weekend closures;\n  unscheduled (holiday) closures are not "
          f"knowable at entry and are deliberately NOT encoded -- see module docstring.")

    print(f"\nWrote {OUT}")
    print(f"  rows {len(lab):,}  columns {lab.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
