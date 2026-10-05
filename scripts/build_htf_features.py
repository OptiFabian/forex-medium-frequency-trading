"""Higher-timeframe features aligned onto the 15-minute index + diagnostics.

FEATURE BUILD ONLY -- no model, no training file, no predictiveness test, no
backtest. The only strategy code invoked is `generate_signals`, used twice as a
CORRECTNESS CHECK that the two gate columns agree with the FSMs they encode.

WHICH FRAME THIS EXTENDS
------------------------
Source: data/resampled/15min_features_moments/  (98 columns -- the frame the
moments build wrote, NOT the 80-column main-path frame; the moments family was
deliberately kept out of `build.apply_pipeline` because its percentile window is
calibrated to 15-min bars).
Output: data/resampled/15min_features_htf/     (98 + 42 = 140 columns).

The chain is linear, not divergent: 15min_features -> 15min_features_moments ->
15min_features_htf. Neither upstream directory is modified.

HIGHER-TIMEFRAME SOURCES
------------------------
The per-timeframe features are computed on the feature frames ALREADY on disk
(data/resampled/{30min,1h,2h,4h,daily,weekly}_features/), so the bars are
exactly the ones every earlier study used and the Bollinger convention is
inherited rather than re-chosen: `volatility.add_features` -> 20-period,
2.0 std, on `mid_close`, with `atr_14` = Wilder's ATR(14) and `rsi_14` =
Wilder's RSI(14) on the same bars.

  30min / 1h : fixed CLOCK bars (`resample_merged`, scripts/resample_clock.py),
               labeled at the window start, closing at label + freq.
  2h / 4h    : SESSION-anchored sub-bars on the 17:00 NY grid
               (`resample_sessions`, scripts/resample_session_intraday.py).
  daily/weekly: FX trading sessions, 17:00 NY anchored.

Note that 30min and 1h are CLOCK-anchored, not session-anchored -- see the
`_report_close_convention` check, which verifies for every 2h/4h bar that the
session close instant equals label + 2h/4h, so handing the plain freq string to
`align_completed_series` is exact for all six timeframes.

WORKING-CONFIG PROVENANCE (decision F4)
---------------------------------------
regime_daily : `align_completed_signal(BollingerReversion().generate_signals(daily),
    index, "daily")`, i.e. the three-state daily Bollinger shadow (long at/below the
    lower band, flat at the mean, short at/above the upper band) projected with
    only CLOSED daily bars.
gate_cell_d  : cell D = `BollingerRsiConfluence(use_rsi=True,
    use_filter=False, use_regime=True)`. Per-bar entry condition: a band touch,
    RSI confirming the touch side (long <= 30, short >= 70), and
    `regime_daily != 0` -- the sign of the regime is deliberately ignored.
gate_stack   : "cost + RSI + spread<median": the entry condition of
    `BollingerRsiConfluence(use_rsi=True, use_filter=True)` plus an absolute
    spread ceiling `spread_close < median(spread_close)`, cost_multiple 2.0,
    notional 100k, default commission model.

Usage:
    uv run python scripts/build_htf_features.py [--pairs EURUSD ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxalgo.backtest.costs import DEFAULT_COMMISSION
from fxalgo.data.resample import session_close_utc
from fxalgo.features import htf
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.strategies.bollinger_rsi_confluence import BollingerRsiConfluence
from fxalgo.utils.logging import setup_logging

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY", "USDCAD"]
DIAG_PAIR = "EURUSD"
RESAMPLED = PROJECT_ROOT / "data" / "resampled"
SRC = RESAMPLED / "15min_features_moments"
DST = RESAMPLED / "15min_features_htf"
OUT_DIR = PROJECT_ROOT / "data" / "features"

NEW = htf.feature_columns()
PCTILES = [0.01, 0.25, 0.50, 0.75, 0.99]
NEAR_DUPLICATE = 0.95
COLLINEAR_FLAG = 0.80
# Recorded in the research log for the cell D work: of the 4,527 cell-A baseline
# entries, 57.5% passed the daily-regime gate (AGREE 1,300 + DISAGREE 1,301 of
# 4,527 in the mtf table). That is an ENTRY-DECISION-BAR rate, not a per-bar one.
STATUS_REGIME_PASS_AT_ENTRIES = 0.575


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _htf_frames(pair: str) -> dict[str, pd.DataFrame]:
    return {tf: _load(RESAMPLED / f"{tf}_features" / f"{pair}.parquet")
            for tf in htf.HIGHER_TIMEFRAMES}


# ---------------------------------------------------------------- pre-flight


def _report_close_convention(frames: dict[str, pd.DataFrame]) -> None:
    """The lag rule is only exact if each bar's close instant is what we claim."""
    print("Close-instant convention (what `align_completed_series` will assume):")
    for tf in htf.HIGHER_TIMEFRAMES:
        idx = frames[tf].index
        if tf in ("2h", "4h", "daily", "weekly"):
            sess = session_close_utc(idx, tf)
            if tf in ("2h", "4h"):
                # These are the two the primitive treats as fixed-width, so the
                # equality has to hold bar for bar or the lag is wrong.
                n_bad = int((sess.asi8 != (idx + pd.Timedelta(tf)).asi8).sum())
                note = f"session close == label+{tf} on {len(idx) - n_bad:,}/{len(idx):,} bars"
            else:
                n_bad = 0
                note = f"session close spans {(sess - idx).min()} .. {(sess - idx).max()}"
            print(f"  {tf:>6} session-anchored (17:00 NY): {note}")
            if tf in ("2h", "4h") and n_bad:
                raise SystemExit(f"{tf}: {n_bad} bars where label+{tf} != session close")
        else:
            print(f"  {tf:>6} clock-anchored: closes at label + {tf}")


def _dst_spot_check(built: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    """Around every US DST switch, is the daily value still the last CLOSED bar?"""
    idx = built.index
    switches = pd.DatetimeIndex(
        [t for t in pd.date_range("2021-06-03", "2026-06-03", freq="D", tz="UTC")
         if t.tz_convert("America/New_York").utcoffset()
         != (t - pd.Timedelta(days=1)).tz_convert("America/New_York").utcoffset()]
    )
    print(f"\nDST spot check -- {len(switches)} US transitions in the window:")
    worst = 0
    for tf in ("daily", "weekly"):
        closes = session_close_utc(frames[tf].index, tf)
        src = htf.htf_block(frames[tf])["bb_position"].to_numpy()
        got = built[f"bb_position_{tf}"].to_numpy()
        for s in switches:
            sel = (idx >= s - pd.Timedelta(days=3)) & (idx <= s + pd.Timedelta(days=3))
            pos = np.searchsorted(closes.to_numpy(), idx[sel].to_numpy(), side="right") - 1
            want = np.where(pos >= 0, src[np.clip(pos, 0, None)], np.nan)
            bad = int((~np.isclose(got[sel], want, equal_nan=True)).sum())
            worst = max(worst, bad)
    hours = sorted({int(h) for h in session_close_utc(frames["daily"].index, "daily").hour})
    print(f"  daily close instants occupy UTC hours {hours} (17:00 NY under EST/EDT)")
    print(f"  mismatches vs the last-closed-bar rule within +/-3 days of a switch: {worst}")
    if worst:
        raise SystemExit("DST alignment mismatch -- refusing to write the frames")


# ---------------------------------------------------------------- diagnostics


def _describe(col: str, s: pd.Series) -> dict:
    v = s.astype("float64")
    d = v.dropna()
    q = d.quantile(PCTILES) if len(d) else pd.Series(index=PCTILES, dtype=float)
    return {
        "feature": col,
        "dtype": str(s.dtype),
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


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype("float64")


def _validate_gates(built: pd.DataFrame, spread_gate_max: float) -> None:
    """Every entry the FSM takes must sit on a bar where the gate column is True.

    The columns are bar-local CONDITIONS; the strategies add path dependence
    (must be flat, and a skipped touch stays locked until price is back inside
    the bands). So entries are a strict SUBSET of gate-True bars, and the ratio
    is a property of the FSM, not a defect in the column.
    """
    print("\n=== GATE COLUMNS vs THE STRATEGY FSMs (correctness check) ===")
    configs = [
        ("gate_cell_d", BollingerRsiConfluence(
            use_rsi=True, use_filter=False, use_regime=True, regime_col=htf.REGIME_COL,
            rsi_lower=htf.RSI_LOWER, rsi_upper=htf.RSI_UPPER,
            notional=htf.NOTIONAL, commission=DEFAULT_COMMISSION)),
        # gate_stack adds an absolute spread ceiling on top of RSI + cost filter;
        # entries of the RSI + cost strategy taken on bars under the ceiling
        # must all sit on gate-True bars.
        ("gate_stack", BollingerRsiConfluence(
            use_rsi=True, use_filter=True,
            cost_multiple=htf.COST_MULTIPLE, rsi_lower=htf.RSI_LOWER,
            rsi_upper=htf.RSI_UPPER, notional=htf.NOTIONAL, commission=DEFAULT_COMMISSION)),
    ]
    for col, strat in configs:
        sig = strat.generate_signals(built).to_numpy()
        entries = (sig != 0) & (np.concatenate([[0], sig[:-1]]) == 0)
        if col == "gate_stack":
            entries &= built["spread_close"].to_numpy() < spread_gate_max
        gate = built[col].to_numpy()
        leaked = int((entries & ~gate).sum())
        print(f"  {col:<12} gate True on {gate.sum():>7,d} bars | FSM entries "
              f"{int(entries.sum()):>6,d} | entries NOT covered by the gate: {leaked}")
        if leaked:
            raise SystemExit(f"{col} misses {leaked} entries the FSM takes")


def _diagnostics(built: pd.DataFrame, existing: list[str], spread_gate_max: float) -> None:
    n = len(built)
    print("\n" + "=" * 110)
    print(f"DIAGNOSTICS -- {DIAG_PAIR} 15-min, {n:,} bars "
          f"({built.index[0]} -> {built.index[-1]})")
    print("=" * 110)

    # ---- A) per-column summary + warmup
    rows = [_describe(c, built[c]) for c in NEW]
    diag = pd.DataFrame(rows)
    print("\n=== A) PER-COLUMN SUMMARY (and warmup: first valid timestamp) ===")
    hdr = (f"{'feature':<26}{'NaN%':>7}{'valid rows':>12}{'mean':>10}{'std':>9}"
           f"{'p1':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p99':>9}{'min':>10}{'max':>10}"
           f"{'first valid':>22}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in diag.iterrows():
        print(f"{r['feature']:<26}{r['nan_frac']*100:>7.2f}{r['n_valid']:>12,d}"
              f"{r['mean']:>10.3f}{r['std']:>9.3f}{r['p1']:>9.3f}{r['p25']:>9.3f}"
              f"{r['p50']:>9.3f}{r['p75']:>9.3f}{r['p99']:>9.3f}{r['min']:>10.3f}"
              f"{r['max']:>10.3f}   {str(r['first_valid'])}")
    complete = built[NEW].notna().all(axis=1)
    binding = diag.loc[diag["nan_frac"].idxmax()]
    print(f"\n  rows with EVERY new column defined: {int(complete.sum()):,} of {n:,} "
          f"({complete.mean():.2%}); binding column = {binding['feature']} "
          f"(first valid {binding['first_valid']})")

    # ---- B) how often each timeframe is outside its bands
    print("\n=== B) FRACTION OF BARS OUTSIDE THE BANDS, per timeframe ===")
    print(f"  {'timeframe':<12}{'valid':>10}{'above 1.0':>12}{'below 0.0':>12}"
          f"{'either':>10}   note")
    for tf in htf.STRETCH_TIMEFRAMES:
        col = htf.BB_POSITION_15M if tf == "15m" else f"bb_position_{tf}"
        s = built[col].dropna()
        above, below = float((s > 1.0).mean()), float((s < 0.0).mean())
        either = above + below
        note = "LOW INFORMATION -- rarely outside" if either < 0.01 else ""
        print(f"  {tf:<12}{len(s):>10,d}{above:>12.3%}{below:>12.3%}{either:>10.3%}   {note}")

    # ---- C) correlation among the seven band positions
    print("\n=== C) CORRELATION AMONG THE SEVEN bb_position COLUMNS ===")
    pos_cols = [htf.BB_POSITION_15M] + [f"bb_position_{tf}" for tf in htf.HIGHER_TIMEFRAMES]
    labels = list(htf.STRETCH_TIMEFRAMES)
    c = built[pos_cols].corr()
    c.index = c.columns = labels
    print("  " + "".join(f"{x:>9}" for x in [""] + labels))
    for a in labels:
        print(f"  {a:<9}" + "".join(f"{c.loc[a, b]:>9.3f}" for b in labels))
    dupes = [(a, b, c.loc[a, b]) for i, a in enumerate(labels)
             for b in labels[i + 1:] if abs(c.loc[a, b]) > NEAR_DUPLICATE]
    print(f"\n  pairs above |r| > {NEAR_DUPLICATE}: "
          + (", ".join(f"{a}~{b} {r:+.3f}" for a, b, r in dupes) if dupes else "none"))

    # ---- D) correlation against the existing columns
    print(f"\n=== D) CORRELATION OF EACH NEW COLUMN vs THE EXISTING {len(existing)} ===")
    frame = pd.concat([_numeric(built, NEW), _numeric(built, existing)], axis=1)
    block = frame.corr(numeric_only=True).loc[NEW, existing]
    stacked = (block.stack().rename("r").reset_index()
               .rename(columns={"level_0": "new_feature", "level_1": "existing_feature"}))
    stacked["abs_r"] = stacked["r"].abs()
    top = stacked.sort_values("abs_r", ascending=False)
    print(f"  top 10 of {len(stacked):,} pairings:")
    print(f"  {'new feature':<26}{'existing feature':<26}{'r':>9}")
    for _, r in top.head(10).iterrows():
        print(f"  {r['new_feature']:<26}{r['existing_feature']:<26}{r['r']:>+9.3f}")
    flagged = top[top["abs_r"] > COLLINEAR_FLAG]
    print(f"\n  pairings above |r| > {COLLINEAR_FLAG}: {len(flagged):,}")
    for _, r in flagged.iterrows():
        print(f"    {r['new_feature']:<26} ~ {r['existing_feature']:<26}{r['r']:>+8.3f}")
    stacked.to_csv(OUT_DIR / "htf_corr_vs_existing.csv", index=False)

    # ---- E) stretch-count distribution
    print("\n=== E) DISTRIBUTION OF tf_stretch_count (0-7 timeframes outside their bands) ===")
    sc = built["tf_stretch_count"].dropna()
    print(f"  defined on {len(sc):,} bars ({len(sc)/n:.1%} of the frame)")
    print(f"  {'count':>7}{'bars':>12}{'share':>10}")
    for k in range(8):
        m = int((sc == k).sum())
        print(f"  {k:>7}{m:>12,d}{m/len(sc):>10.3%}")
    print(f"  mean {sc.mean():.3f}, max observed {int(sc.max())}")

    # ---- F) regime_daily_active
    print("\n=== F) regime_daily_active ===")
    active = built[htf.REGIME_ACTIVE_COL]
    print(f"  per-BAR fraction active: {active.mean():.3%} "
          f"({int(active.sum()):,} of {n:,} bars)")
    counts = built[htf.REGIME_COL].value_counts().sort_index()
    for k, v in counts.items():
        name = {-1: "short", 0: "flat", 1: "long"}[int(k)]
        print(f"    regime_daily {int(k):+d} ({name:<5}): {v:>9,d}  {v/n:>7.2%}")
    # The recorded figure is an ENTRY-DECISION-BAR rate, so reproduce that
    # population: the bar BEFORE each plain-Bollinger entry.
    from fxalgo.strategies.bollinger_reversion import BollingerReversion

    sig = BollingerReversion().generate_signals(built).to_numpy()
    entries = np.flatnonzero((sig != 0) & (np.concatenate([[0], sig[:-1]]) == 0))
    decision = entries - 1
    decision = decision[decision >= 0]
    at_entries = float((built[htf.REGIME_COL].to_numpy()[decision] != 0).mean())
    print(f"  at cell-A ENTRY DECISION bars: {at_entries:.3%} of {len(decision):,} candidates")
    print(f"  the research log records {STATUS_REGIME_PASS_AT_ENTRIES:.1%} "
          f"(pass daily-regime, cell-D factorial overlap table)")
    delta = at_entries - STATUS_REGIME_PASS_AT_ENTRIES
    verdict = "AGREES" if abs(delta) < 0.01 else "DISCREPANCY -- reported, not reconciled"
    print(f"  -> {verdict} (delta {delta:+.3%})")

    # ---- G) column count
    print("\n=== G) COLUMN COUNT ===")
    print(f"  existing {len(existing)} + new {len(NEW)} = {len(existing) + len(NEW)}")
    print(f"  frame actually carries {built.shape[1]} columns -- "
          f"{'CONFIRMED' if built.shape[1] == len(existing) + len(NEW) else 'MISMATCH'}")
    print(f"  gate_stack spread ceiling (median spread_close): {spread_gate_max:.8f} "
          f"({spread_gate_max / float(built['mid_close'].median()) * 1e4:.3f} bps of price)")

    diag.to_csv(OUT_DIR / "htf_diagnostics.csv", index=False)
    c.to_csv(OUT_DIR / "htf_bb_position_corr.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    DST.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Higher-timeframe feature build")
    print(f"  source  {SRC}")
    print(f"  output  {DST}")
    print(f"  {len(NEW)} new columns: {len(htf.HIGHER_TIMEFRAMES)} timeframes x "
          f"{len(htf.PER_TF_STEMS)} + {len(htf.SPREAD_TIMEFRAMES)} spreads + 2 stretch "
          f"+ 2 regime + 2 gates + 6 don_pos + 1 atr_bps_15m\n")

    diag_state = None
    for pair in args.pairs:
        base = _load(SRC / f"{pair}.parquet")
        frames = _htf_frames(pair)
        if pair == args.pairs[0]:
            _report_close_convention(frames)

        # The spread ceiling is a CONSTANT of the gate_stack config: the median
        # spread_close over the whole frame. It is therefore a FULL-SAMPLE statistic -- see the
        # caveat printed at the end.
        spread_gate_max = float(np.nanmedian(base["spread_close"].to_numpy()))
        built = htf.add_features(base, frames, spread_gate_max=spread_gate_max)
        built.to_parquet(DST / f"{pair}.parquet", engine="pyarrow")
        # Report the ceiling in bps of price, not "pips": a pip is 1e-4 for the
        # USD-quoted majors but 1e-2 for USDJPY, so a shared pip label would be
        # wrong for one of the six.
        ceiling_bps = spread_gate_max / float(base["mid_close"].median()) * 1e4
        print(f"\n{pair}: {len(base):,} bars, {base.shape[1]} -> {built.shape[1]} columns "
              f"| spread ceiling {spread_gate_max:.8f} ({ceiling_bps:.3f} bps) "
              f"| gate_cell_d {int(built['gate_cell_d'].sum()):,} bars "
              f"| gate_stack {int(built['gate_stack'].sum()):,} bars")
        if pair == DIAG_PAIR:
            diag_state = (built, list(base.columns), frames, spread_gate_max)

    if diag_state is None:
        print(f"\n{DIAG_PAIR} not built -- skipping diagnostics.")
        return 0

    built, existing, frames, spread_gate_max = diag_state
    _dst_spot_check(built, frames)
    _validate_gates(built, spread_gate_max)
    _diagnostics(built, existing, spread_gate_max)

    print("\n" + "-" * 110)
    print("CAVEATS carried by this build:")
    print("  * gate_stack's spread ceiling is the FULL-SAMPLE median spread. The")
    print("    column is causal given that constant, but the constant itself was")
    print("    chosen with knowledge of the whole window -- re-derive it on the")
    print("    training fold alone in any walk-forward.")
    print("  * gate_stack is NOT scale-free (decision F7). Its cost filter contains")
    print("    the commission model's absolute 2.00 per-order minimum, which does not move when")
    print("    prices are rescaled; every other new column is scale-invariant.")
    print("  * No predictiveness has been measured. These are inputs, not evidence.")
    print(f"\nWrote frames to {DST}")
    print(f"Wrote {OUT_DIR / 'htf_diagnostics.csv'}, "
          f"{OUT_DIR / 'htf_corr_vs_existing.csv'}, "
          f"{OUT_DIR / 'htf_bb_position_corr.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
