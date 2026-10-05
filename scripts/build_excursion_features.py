"""Excursion history on the higher-TF band positions + a causal spread rank.

FEATURE BUILD ONLY -- no model, no training file, no predictiveness test, no
backtest.

WHICH FRAME THIS EXTENDS
------------------------
Source: data/resampled/15min_features_htf/  (140 columns)
Output: data/resampled/15min_features_exc/  (140 + 38 = 178 columns)

The chain stays linear: 15min_features (80) -> _moments (98) -> _htf (140) ->
_exc (178). No upstream directory is modified.

WHAT THE 38 ARE
---------------
36 excursion columns -- six timeframes x {consecutive run above / below,
occupancy of the last 20 closed bars above / below, bars since the last
inside->outside crossing, max depth of the current excursion} -- each computed
on the higher timeframe's OWN closed bars and then projected with
`align_completed_series`. Plus TWO trailing spread percentiles.

The spread percentiles exist to retire a hindsight hazard. `gate_stack`'s
spread ceiling is the FULL-SAMPLE median `spread_close`: the column is causal
given that constant, but the constant is not. These give the tree the
continuous quantity instead of a threshold chosen with knowledge of the whole
window. `gate_stack` is left untouched -- selection happens later.

BOTH percentile variants ship. `spread_pctile_trailing` (minute-of-week,
NY-anchored) is clean of the clock; `spread_pctile_trailing_rolling` (plain
23,710-bar window) carries ~20% time-of-day variance. Keeping the pair makes
that contamination measurable in feature importance downstream rather than only
in this one-off diagnostic.

Usage:
    uv run python scripts/build_excursion_features.py [--pairs EURUSD ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxalgo.features import excursion, htf
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
DIAG_PAIR = "EURUSD"
RESAMPLED = PROJECT_ROOT / "data" / "resampled"
SRC = RESAMPLED / "15min_features_htf"
DST = RESAMPLED / "15min_features_exc"
OUT_DIR = PROJECT_ROOT / "data" / "features"

NEW = excursion.feature_columns()
PCTILES = [0.01, 0.25, 0.50, 0.75, 0.99]
COLLINEAR_FLAG = 0.80
DUPLICATE_FLAG = 0.95
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_tokyo",
             "session_london", "session_ny", "session_london_ny_overlap"]
# Effective sample size at k=2.0, from data/labels/effective_sample_size.csv.
N_EFF = 13619.0


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _htf_frames(pair: str) -> dict[str, pd.DataFrame]:
    return {tf: _load(RESAMPLED / f"{tf}_features" / f"{pair}.parquet")
            for tf in htf.HIGHER_TIMEFRAMES}


def _describe(col: str, s: pd.Series) -> dict:
    v = s.astype("float64")
    d = v.dropna()
    q = d.quantile(PCTILES) if len(d) else pd.Series(index=PCTILES, dtype=float)
    return {
        "feature": col,
        "n_valid": int(len(d)),
        "nan_frac": float(v.isna().mean()),
        "first_valid": v.first_valid_index(),
        "mean": float(d.mean()) if len(d) else np.nan,
        "std": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
        "min": float(d.min()) if len(d) else np.nan,
        "p1": float(q.loc[0.01]) if len(d) else np.nan,
        "p25": float(q.loc[0.25]) if len(d) else np.nan,
        "p50": float(q.loc[0.50]) if len(d) else np.nan,
        "p75": float(q.loc[0.75]) if len(d) else np.nan,
        "p99": float(q.loc[0.99]) if len(d) else np.nan,
        "max": float(d.max()) if len(d) else np.nan,
    }


# ---------------------------------------------------------------- diagnostics


def _run_distribution(frames: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """UNCAPPED run lengths, on the higher-TF bars and on the aligned rows.

    The aligned-row count is what a model actually sees; the independent-window
    count divides it by the label overlap, and is what decides the cap.
    """
    from fxalgo.strategies.multi_timeframe import align_completed_series

    rows_per_window = len(index) / N_EFF
    out = []
    for tf in htf.HIGHER_TIMEFRAMES:
        pos = htf.bb_position(frames[tf])
        block = excursion.excursion_block(pos, run_cap=10**9)
        for side in ("above", "below"):
            src = block[f"bars_{side}_band"]
            aligned = align_completed_series(src, index, tf)
            bars = src.dropna().astype(int).value_counts()
            rows = aligned.dropna().astype(int).value_counts()
            for k in sorted(set(bars.index) | set(rows.index)):
                if k == 0:
                    continue
                out.append({
                    "timeframe": tf, "side": side, "run": int(k),
                    "htf_bars": int(bars.get(k, 0)),
                    "aligned_rows": int(rows.get(k, 0)),
                    "indep_windows": rows.get(k, 0) / rows_per_window,
                })
    return pd.DataFrame(out)


def _diagnostics(built: pd.DataFrame, existing: list[str],
                 frames: dict[str, pd.DataFrame]) -> None:
    n = len(built)
    print("\n" + "=" * 112)
    print(f"DIAGNOSTICS -- {DIAG_PAIR} 15-min, {n:,} bars "
          f"({built.index[0]} -> {built.index[-1]})")
    print("=" * 112)

    # ---- A) per-column summary
    diag = pd.DataFrame([_describe(c, built[c]) for c in NEW])
    print("\n=== A) PER-COLUMN SUMMARY (and warmup: first valid timestamp) ===")
    hdr = (f"{'feature':<34}{'NaN%':>7}{'valid':>10}{'mean':>9}{'std':>8}{'p1':>8}"
           f"{'p25':>8}{'p50':>8}{'p75':>8}{'p99':>8}{'min':>8}{'max':>8}"
           f"{'first valid':>22}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in diag.iterrows():
        print(f"{r['feature']:<34}{r['nan_frac']*100:>7.2f}{r['n_valid']:>10,d}"
              f"{r['mean']:>9.3f}{r['std']:>8.3f}{r['p1']:>8.3f}{r['p25']:>8.3f}"
              f"{r['p50']:>8.3f}{r['p75']:>8.3f}{r['p99']:>8.3f}{r['min']:>8.3f}"
              f"{r['max']:>8.3f}   {str(r['first_valid'])}")
    complete = built[NEW].notna().all(axis=1)
    binding = diag.loc[diag["nan_frac"].idxmax()]
    print(f"\n  rows with EVERY new column defined: {int(complete.sum()):,} of {n:,} "
          f"({complete.mean():.2%}); binding column = {binding['feature']} "
          f"(first valid {binding['first_valid']})")
    all_cols = built.notna().all(axis=1)
    print(f"  rows with every column of the FULL {built.shape[1]} defined: {int(all_cols.sum()):,} "
          f"({all_cols.mean():.2%})")

    # ---- B) uncapped run distribution -> the cap justification
    print("\n=== B) UNCAPPED RUN-LENGTH DISTRIBUTION (what justifies the cap) ===")
    dist = _run_distribution(frames, built.index)
    print("  per timeframe/side, run -> higher-TF bars (aligned 15-min rows):")
    for tf in htf.HIGHER_TIMEFRAMES:
        for side in ("above", "below"):
            g = dist[(dist["timeframe"] == tf) & (dist["side"] == side)]
            body = "  ".join(f"{int(r['run'])}:{int(r['htf_bars'])}({int(r['aligned_rows']):,})"
                             for _, r in g.iterrows())
            print(f"    {tf:>7} {side:<6} {body}")
    pooled = dist.groupby("run")[["htf_bars", "aligned_rows", "indep_windows"]].sum()
    print(f"\n  pooled over all 12 run columns "
          f"(N_eff={N_EFF:,.0f}, {n / N_EFF:.2f} rows per independent window):")
    print(f"  {'run':>5}{'htf bars':>12}{'aligned rows':>15}{'indep windows':>16}")
    for k, r in pooled.iterrows():
        print(f"  {int(k):>5}{int(r['htf_bars']):>12,d}{int(r['aligned_rows']):>15,d}"
              f"{r['indep_windows']:>16.0f}")
    over = dist[dist["run"] > excursion.RUN_CAP]
    print(f"\n  CAP = {excursion.RUN_CAP}. Longest run observed anywhere: "
          f"{int(dist['run'].max())}.")
    print(f"  Folded into the top bucket: {int(over['aligned_rows'].sum()):,} rows "
          f"across all 12 columns ({over['aligned_rows'].sum() / (12 * n):.4%} of "
          f"column-rows, ~{over['indep_windows'].sum():.0f} independent windows).")
    for tf in htf.HIGHER_TIMEFRAMES:
        g = over[over["timeframe"] == tf]
        print(f"    {tf:>7}: {int(g['aligned_rows'].sum()):>6,d} rows "
              f"({g['aligned_rows'].sum() / (2 * n):.4%} of that timeframe's two columns)")
    dist.to_csv(OUT_DIR / "exc_run_distribution.csv", index=False)

    # ---- C) run counter vs occupancy, same timeframe and side
    print("\n=== C) CORRELATION: run counter vs occupancy (same timeframe and side) ===")
    print(f"  {'timeframe':<12}{'side':<8}{'r':>9}   verdict")
    for tf in htf.HIGHER_TIMEFRAMES:
        for side in ("above", "below"):
            a = built[f"bars_{side}_band_{tf}"]
            b = built[excursion.column_name(f"frac_{side}_band", tf)]
            r = float(a.corr(b))
            verdict = ("DUPLICATE -- prune one" if abs(r) > DUPLICATE_FLAG
                       else "related, not duplicate")
            print(f"  {tf:<12}{side:<8}{r:>9.3f}   {verdict}")

    # ---- D) correlation against the existing 140
    print(f"\n=== D) CORRELATION OF EACH NEW COLUMN vs THE EXISTING {len(existing)} ===")
    frame = pd.concat([built[NEW].astype("float64"),
                       built[existing].astype("float64")], axis=1)
    block = frame.corr(numeric_only=True).loc[NEW, existing]
    stacked = (block.stack().rename("r").reset_index()
               .rename(columns={"level_0": "new_feature", "level_1": "existing_feature"}))
    stacked["abs_r"] = stacked["r"].abs()
    top = stacked.sort_values("abs_r", ascending=False)
    print(f"  top 15 of {len(stacked):,} pairings:")
    print(f"  {'new feature':<34}{'existing feature':<34}{'r':>9}")
    for _, r in top.head(15).iterrows():
        print(f"  {r['new_feature']:<34}{r['existing_feature']:<34}{r['r']:>+9.3f}")
    flagged = top[top["abs_r"] > COLLINEAR_FLAG]
    print(f"\n  pairings above |r| > {COLLINEAR_FLAG}: {len(flagged):,}")
    for _, r in flagged.iterrows():
        print(f"    {r['new_feature']:<34} ~ {r['existing_feature']:<30}{r['r']:>+8.3f}")
    stacked.to_csv(OUT_DIR / "exc_corr_vs_existing.csv", index=False)

    # ---- E) the spread percentile
    print("\n=== E) THE TWO SPREAD PERCENTILES ===")
    v = built[excursion.SPREAD_PCTILE_COL]
    rolling = built[excursion.SPREAD_PCTILE_ROLLING_COL]
    tod = pd.Series(built.index.tz_convert("America/New_York").hour * 60
                    + built.index.tz_convert("America/New_York").minute, index=built.index)

    def tod_share(x: pd.Series) -> float:
        d = pd.DataFrame({"x": x, "k": tod}).dropna()
        return float((d.groupby("k")["x"].transform("mean") - d["x"].mean()).var(ddof=0)
                     / d["x"].var(ddof=0))

    d = v.dropna()
    print(f"  valid {len(d):,} ({len(d)/n:.1%}), first valid {d.index[0]}")
    print(f"  distribution: mean {d.mean():.3f} std {d.std():.3f} | "
          + " ".join(f"p{int(p*100)} {d.quantile(p):.3f}" for p in PCTILES))
    print("  deciles: " + " ".join(
        f"{k}:{c/len(d):.1%}" for k, c in
        (d * 10).clip(upper=9.999).astype(int).value_counts().sort_index().items()))
    print("\n  BOTH VARIANTS ARE SHIPPED. The rolling one is the contaminated twin,")
    print("  kept so the contamination is measurable in feature importance later.")
    print("  NOTE the screening blind spot: 'max |r| vs a time column' does NOT")
    print("  detect it -- the diurnal spread pattern is a SPIKE at the 17:00 NY")
    print("  rollover, not a sinusoid, so no trig column tracks it. Read the")
    print("  variance share instead.")
    print(f"  {'variant':<28}{'valid':>10}{'first valid':>24}{'ToD var share':>15}"
          f"{'max |r| time':>14}{'which':>28}")
    for name, x in ((f"minute-of-week x{excursion.SPREAD_PCTILE_WEEKS} (NY)", v),
                    (f"plain rolling {excursion.ROLLING_PCTILE_WINDOW}", rolling)):
        cors = {c: abs(float(x.corr(built[c]))) for c in TIME_COLS}
        worst = max(cors, key=cors.get)
        dd = x.dropna()
        print(f"  {name:<28}{len(dd):>10,d}{str(dd.index[0]):>24}"
              f"{tod_share(x):>15.4f}{cors[worst]:>14.3f}{worst:>28}")
    print(f"  correlation between the two variants: {float(v.corr(rolling)):+.3f}")
    print("\n  correlation against the spread columns:")
    for c in ["spread_close", "spread_mean_15", "spread_ratio_15",
              "spread_mean_60", "spread_ratio_60", "spread_zscore_60"]:
        print(f"    {c:<20}{float(v.corr(built[c])):>+8.3f}")
    print("  correlation against the time columns:")
    for c in TIME_COLS:
        print(f"    {c:<28}{float(v.corr(built[c])):>+8.3f}")
    worst_time = max(abs(float(v.corr(built[c]))) for c in TIME_COLS)
    print(f"  -> max |r| vs a time column {worst_time:.3f} "
          f"({'OK' if worst_time <= COLLINEAR_FLAG else 'RECONSIDER THE WINDOW'}); "
          f"time-of-day variance share {tod_share(v):.4f}")

    # ---- F) how closely the causal column tracks the hindsight constant
    print("\n=== F) spread_pctile_trailing WHERE gate_stack IS TRUE ===")
    gate = built["gate_stack"] & v.notna()
    sub = v[gate]
    print(f"  gate_stack True and percentile defined: {len(sub):,} bars "
          f"(of {int(built['gate_stack'].sum()):,} gate_stack bars)")
    print(f"  mean {sub.mean():.3f} | "
          + " ".join(f"p{int(p*100)} {sub.quantile(p):.3f}" for p in PCTILES))
    print(f"  {'decile':>8}{'bars':>10}{'share':>10}{'cumulative':>12}")
    counts = (sub * 10).clip(upper=9.999).astype(int).value_counts().sort_index()
    cum = 0
    for k in range(10):
        c = int(counts.get(k, 0))
        cum += c
        print(f"  {f'{k/10:.1f}-{(k+1)/10:.1f}':>8}{c:>10,d}{c/len(sub):>10.1%}"
              f"{cum/len(sub):>12.1%}")
    below_med = float((sub < 0.5).mean())
    print(f"\n  gate_stack bars below the trailing MEDIAN of their own slot: {below_med:.1%}")
    print("  The two are NOT the same cut: the gate is an ABSOLUTE level fixed")
    print("  once for the whole window, the percentile is RELATIVE to the same")
    print("  minute-of-week. A structurally quiet slot can sit entirely below the")
    print("  global median, so a locally-wide spread there still passes the gate.")

    # ---- G) column count
    print("\n=== G) COLUMN COUNT ===")
    print(f"  existing {len(existing)} + new {len(NEW)} = {len(existing) + len(NEW)}")
    print(f"  frame actually carries {built.shape[1]} columns -- "
          f"{'CONFIRMED' if built.shape[1] == len(existing) + len(NEW) else 'MISMATCH'}")

    diag.to_csv(OUT_DIR / "exc_diagnostics.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    DST.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Excursion-history feature build")
    print(f"  source  {SRC}")
    print(f"  output  {DST}")
    print(f"  {len(NEW)} new columns | run cap {excursion.RUN_CAP} | occupancy window "
          f"{excursion.FRACTION_WINDOW} | spread percentile "
          f"{excursion.SPREAD_PCTILE_WEEKS} same-minute-of-week weeks (NY-anchored)\n")

    diag_state = None
    for pair in args.pairs:
        base = _load(SRC / f"{pair}.parquet")
        frames = _htf_frames(pair)
        interior = {
            tf: int(htf.bb_position(f).iloc[20:].isna().sum())
            for tf, f in frames.items()
        }
        built = excursion.add_features(base, frames)
        built.to_parquet(DST / f"{pair}.parquet", engine="pyarrow")
        holes = {k: v for k, v in interior.items() if v}
        print(f"{pair}: {len(base):,} bars, {base.shape[1]} -> {built.shape[1]} columns"
              + (f" | interior bb_position holes: {holes}" if holes else ""))
        if pair == DIAG_PAIR:
            diag_state = (built, list(base.columns), frames)

    if diag_state is None:
        print(f"\n{DIAG_PAIR} not built -- skipping diagnostics.")
        return 0

    built, existing, frames = diag_state
    _diagnostics(built, existing, frames)

    print("\n" + "-" * 112)
    print("CAVEATS carried by this build:")
    print("  * These columns are STEP FUNCTIONS. Between higher-TF closes every")
    print("    one is constant across the intervening 15-min rows (16 rows at 4h,")
    print("    a whole week at weekly). Their effective sample size is the number")
    print("    of higher-TF bars, not the row count.")
    print("  * The spread percentiles are ALTERNATIVES to gate_stack's hindsight")
    print("    threshold, not replacements -- gate_stack is untouched. Both the")
    print("    clean (minute-of-week) and contaminated (plain rolling) variants")
    print("    ship, so importance can measure the difference. Selection is")
    print("    downstream.")
    print("  * No predictiveness has been measured. These are inputs, not evidence.")
    print(f"\nWrote frames to {DST}")
    print(f"Wrote {OUT_DIR / 'exc_diagnostics.csv'}, "
          f"{OUT_DIR / 'exc_run_distribution.csv'}, "
          f"{OUT_DIR / 'exc_corr_vs_existing.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
