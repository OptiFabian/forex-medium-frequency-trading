"""Build the distributional-moments family onto the 15-min frames + diagnostics.

FEATURE BUILD ONLY -- no strategy, no backtest, no entry filter, no window
tuning. Measures what the new columns look like so the pruning decision before
the XGBoost phase has an evidence base.

Existing 15-min feature frames are NOT modified: the extended frames are written
to data/resampled/15min_features_moments/ so nothing already on disk changes.

Reports, per pair per feature: count, NaN fraction, first valid timestamp,
mean/std/min/p1/p25/p50/p75/p99/max, and the fraction of infinite and of
guarded (post-warmup NaN) values. Then three checks:
  A) collinearity of each new feature against every existing column
  B) internal redundancy among the new features
  C) stability of each new feature across calendar years

Usage:
    uv run python scripts/build_moments_features.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.data.resample import session_keys
from fxalgo.features import moments
from fxalgo.features.moments import add_features, feature_columns, gap_return_mask
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
SRC = PROJECT_ROOT / "data" / "resampled" / "15min_features"
DST = PROJECT_ROOT / "data" / "resampled" / "15min_features_moments"
OUT_CSV = PROJECT_ROOT / "data" / "features" / "moments_diagnostics.csv"
NEW = feature_columns()
COLLINEAR_FLAG = 0.80
REDUNDANT_FLAG = 0.95
PCTILES = [0.01, 0.25, 0.50, 0.75, 0.99]


def _null_sd(stat: str, n: int) -> float:
    """SD of the estimator under an iid-normal null, for window size n.

    Standard sampling variances of the bias-corrected estimators (Joanes &
    Gill): Var(G1) = 6n(n-1)/((n-2)(n+1)(n+3)) and
    Var(G2) = 24n(n-1)^2/((n-3)(n-2)(n+3)(n+5)).

    Dividing a feature's OBSERVED dispersion by this gives a noise yardstick:
    a ratio near 1 means the column is indistinguishable from what pure
    sampling error would produce on normal returns. FX returns are neither iid
    nor normal, so this is a reference point, not a significance test.
    """
    if stat == "skew":
        return float(np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))))
    return float(np.sqrt(24 * n * (n - 1) ** 2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))))


def _describe(pair: str, col: str, s: pd.Series) -> dict:
    vals = s.to_numpy(dtype="float64")
    finite = np.isfinite(vals)
    n_inf = int(np.isinf(vals).sum())
    first_valid = s.first_valid_index()
    if first_valid is not None:
        post = s.loc[first_valid:]
        guarded = float(post.isna().mean())
    else:
        guarded = float("nan")
    d = s.dropna()
    q = d.quantile(PCTILES) if len(d) else pd.Series(index=PCTILES, dtype=float)
    return {
        "pair": pair,
        "feature": col,
        "count": int(finite.sum()),
        "nan_frac": float(s.isna().mean()),
        "first_valid": first_valid,
        "mean": float(d.mean()) if len(d) else np.nan,
        "std": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
        "min": float(d.min()) if len(d) else np.nan,
        "p1": float(q.loc[0.01]) if len(d) else np.nan,
        "p25": float(q.loc[0.25]) if len(d) else np.nan,
        "p50": float(q.loc[0.50]) if len(d) else np.nan,
        "p75": float(q.loc[0.75]) if len(d) else np.nan,
        "p99": float(q.loc[0.99]) if len(d) else np.nan,
        "max": float(d.max()) if len(d) else np.nan,
        "inf_frac": n_inf / len(s) if len(s) else np.nan,
        "guarded_frac": guarded,
    }


def main() -> int:
    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    DST.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Distributional-moments feature build (15-min frames, six pairs)\n")
    print(f"percentile window = {moments.PERCENTILE_WINDOW_BARS:,} bars "
          f"({moments.PERCENTILE_TRADING_SESSIONS} sessions x "
          f"{moments.BARS_PER_SESSION_15MIN} bars/session)\n")

    rows, corr_rows, pooled = [], [], []
    n_existing = None
    for pair in PAIRS:
        src = pd.read_parquet(SRC / f"{pair}.parquet", engine="pyarrow")
        existing = list(src.columns)
        n_existing = len(existing)

        # Session structure check -- confirms the percentile-window constant.
        sessions = session_keys(src.index, "daily").nunique()
        bars_per_session = len(src) / sessions
        gmask = gap_return_mask(src.index)

        out = add_features(src)
        out.to_parquet(DST / f"{pair}.parquet", engine="pyarrow")

        print(f"{pair}: {len(src):,} bars x {n_existing} cols -> {out.shape[1]} cols  "
              f"| {sessions:,} sessions, {bars_per_session:.2f} bars/session  "
              f"| gap-masked returns {int(gmask.sum()):,} ({gmask.mean():.3%})")

        for col in NEW:
            rows.append(_describe(pair, col, out[col]))

        # A) collinearity against every EXISTING column, computed per pair
        # (many existing columns carry price units, so pooling them across
        # pairs would be meaningless).
        c = out[NEW + existing].corr(numeric_only=True)
        block = c.loc[NEW, [e for e in existing if e in c.columns]]
        stacked = block.stack().rename("r").reset_index()
        stacked.columns = ["new_feature", "existing_feature", "r"]
        stacked["pair"] = pair
        corr_rows.append(stacked)

        keep = out[NEW].copy()
        keep["pair"] = pair
        keep["year"] = out.index.year
        pooled.append(keep)

    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_CSV, index=False)
    allcorr = pd.concat(corr_rows, ignore_index=True)
    pooled_df = pd.concat(pooled, ignore_index=True)

    # ---------------- summary table ----------------
    print("\n\n=== PER-FEATURE SUMMARY (pooled over pairs; ranges are across the six) ===")
    hdr = (f"{'feature':<30}{'count':>10}{'NaN%':>8}{'guard%':>8}{'inf%':>7}"
           f"{'mean':>10}{'std':>9}{'p1':>10}{'p50':>9}{'p99':>10}{'first valid':>22}")
    print(hdr)
    print("-" * len(hdr))
    for col in NEW:
        g = diag[diag["feature"] == col]
        fv = pd.to_datetime(g["first_valid"]).max()
        print(f"{col:<30}{int(g['count'].mean()):>10,d}{g['nan_frac'].mean()*100:>8.2f}"
              f"{g['guarded_frac'].mean()*100:>8.3f}{g['inf_frac'].mean()*100:>7.2f}"
              f"{g['mean'].mean():>10.3f}{g['std'].mean():>9.3f}{g['p1'].mean():>10.3f}"
              f"{g['p50'].mean():>9.3f}{g['p99'].mean():>10.3f}"
              f"   {str(fv)}")

    # ---- sampling-noise yardstick for the moment estimators ----
    print("\n=== SAMPLING-NOISE YARDSTICK (moment estimators only) ===")
    print("observed SD vs the SD pure sampling error would give on iid-normal returns;")
    print("noise share = (null SD / observed SD)^2, the fraction of the column's")
    print("variance attributable to estimation error rather than real variation.")
    print(f"  {'feature':<18}{'n':>5}{'observed SD':>13}{'null SD':>10}"
          f"{'ratio':>8}{'noise share':>13}")
    for stat, w in (("skew", 96), ("kurt", 96), ("skew", 24), ("kurt", 24)):
        col = f"ret_{stat}_{w}"
        obs = float(diag[diag["feature"] == col]["std"].mean())
        null = _null_sd(stat, w)
        ratio = obs / null
        print(f"  {col:<18}{w:>5}{obs:>13.3f}{null:>10.3f}{ratio:>8.2f}"
              f"{1.0 / ratio**2:>12.1%}")

    # ---------------- A) collinearity ----------------
    print("\n\n=== A) COLLINEARITY vs EXISTING COLUMNS ===")
    agg = (allcorr.assign(absr=allcorr["r"].abs())
           .groupby(["new_feature", "existing_feature"])["absr"]
           .agg(mean_abs_r="mean", max_abs_r="max").reset_index()
           .sort_values("mean_abs_r", ascending=False))
    print(f"top 10 of {len(agg):,} (new x existing) pairings, by mean |r| across the six pairs:")
    print(f"  {'new feature':<28}{'existing feature':<22}{'mean |r|':>10}{'max |r|':>10}")
    for _, r in agg.head(10).iterrows():
        print(f"  {r['new_feature']:<28}{r['existing_feature']:<22}"
              f"{r['mean_abs_r']:>10.3f}{r['max_abs_r']:>10.3f}")
    flagged = agg[agg["max_abs_r"] > COLLINEAR_FLAG]
    if len(flagged):
        print(f"\n  REDUNDANCY CANDIDATES (|r| > {COLLINEAR_FLAG} on at least one pair):")
        for _, r in flagged.iterrows():
            print(f"    {r['new_feature']} ~ {r['existing_feature']}  "
                  f"mean |r| {r['mean_abs_r']:.3f}  max |r| {r['max_abs_r']:.3f}")
    else:
        print(f"\n  No new feature reaches |r| > {COLLINEAR_FLAG} against any existing column "
              f"on any pair.")

    # ---------------- B) internal redundancy ----------------
    print("\n\n=== B) INTERNAL REDUNDANCY (pooled across pairs; all 18 are scale-free) ===")
    icorr = pooled_df[NEW].corr()
    pairs_ = []
    for i, a in enumerate(NEW):
        for b in NEW[i + 1:]:
            pairs_.append({"a": a, "b": b, "r": icorr.loc[a, b]})
    ip = pd.DataFrame(pairs_).assign(absr=lambda d: d["r"].abs()).sort_values(
        "absr", ascending=False)
    print(f"  {'feature A':<28}{'feature B':<28}{'r':>8}")
    for _, r in ip.head(12).iterrows():
        mark = "   <-- one likely survives pruning" if r["absr"] > REDUNDANT_FLAG else ""
        print(f"  {r['a']:<28}{r['b']:<28}{r['r']:>+8.3f}{mark}")
    moment_twins = [
        ("ret_skew_96", "ret_bowley_skew_96"), ("ret_skew_24", "ret_bowley_skew_24"),
        ("ret_kurt_96", "ret_moors_kurt_96"), ("ret_kurt_24", "ret_moors_kurt_24"),
    ]
    print("\n  moment vs robust twin:")
    for a, b in moment_twins:
        r = icorr.loc[a, b]
        note = "  REDUNDANT" if abs(r) > REDUNDANT_FLAG else ""
        print(f"    {a:<24} ~ {b:<24} r = {r:+.3f}{note}")

    # ---------------- C) stability by year ----------------
    print("\n\n=== C) STABILITY BY CALENDAR YEAR (pooled across pairs) ===")
    ymean = pooled_df.groupby("year")[NEW].mean()
    ystd = pooled_df.groupby("year")[NEW].std(ddof=1)
    pooled_sd = pooled_df[NEW].std(ddof=1)
    drift = (ymean.max() - ymean.min()) / pooled_sd
    print("  yearly MEAN by feature:")
    print(ymean.T.to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\n  yearly STD by feature:")
    print(ystd.T.to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\n  drift = (max yearly mean - min yearly mean) / pooled SD:")
    for col in NEW:
        flag = "   <-- UNSTABLE (> 1 pooled SD)" if drift[col] > 1.0 else ""
        print(f"    {col:<30}{drift[col]:>7.3f}{flag}")

    print(f"\n\nWrote {OUT_CSV}")
    print(f"Extended frames -> {DST}  ({n_existing} existing + {len(NEW)} new = "
          f"{n_existing + len(NEW)} columns)")
    print("Existing 15-min feature frames were NOT modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
