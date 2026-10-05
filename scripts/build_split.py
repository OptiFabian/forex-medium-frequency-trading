"""Blocked-by-time train/test split for the training dataset.

SPLIT ONLY -- no model, no training, no predictiveness measurement.

INPUTS
  data/training/eurusd_events.parquet       sampled events (balance criteria)
  data/training/eurusd_events_full.parquet  every event (the split domain)
  data/resampled/15min_features_exc/EURUSD.parquet  for the train-fold spread
                                            median and the gate recomputation

OUTPUT
  data/training/eurusd_split.parquet -- an ASSIGNMENT TABLE keyed by event
  timestamp: split / chunk_id / chunk_start / chunk_end / gate_stack_trainfold.
  No feature data is duplicated; it joins onto either events file, and because
  it is built on the FULL index the sampling stays reversible without a
  re-split.

Usage:
    uv run python scripts/build_split.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fxalgo.features import htf
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.strategies.multi_timeframe import align_completed_series
from fxalgo.training import split as sp
from fxalgo.utils.logging import setup_logging

PAIR = "EURUSD"
TRAINING = PROJECT_ROOT / "data" / "training"
SAMPLED = TRAINING / "eurusd_events.parquet"
FULL = TRAINING / "eurusd_events_full.parquet"
FEATURES = PROJECT_ROOT / "data" / "resampled" / "15min_features_exc" / f"{PAIR}.parquet"
OUT_PATH = TRAINING / "eurusd_split.parquet"

LABEL_COL = "label_2.0"
VOL_COL = "atr_bps_15m"
MEAN_WINDOW_BARS = 135.567521      # data/labels/effective_sample_size.csv, k=2.0
FULL_SAMPLE_CEILING = 0.00002      # the gate_stack constant baked into the frame


def _read(path, columns=None) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _usable_start(path) -> tuple[pd.Timestamp, str]:
    """First timestamp at which EVERY feature column is defined, and which
    column binds. Read in column batches so the whole file never materialises."""
    schema = pq.ParquetFile(path).schema_arrow
    skip = {"timestamp", *sp.SPLIT_VALUES} | set(
        pd.Index(["block_id", "event_offset", "sampled_flag"])
    )
    from fxalgo.training import dataset as ds

    skip |= set(ds.KEPT_LABEL_COLUMNS)
    cols = [c for c in schema.names if c not in skip]
    first: dict[str, pd.Timestamp] = {}
    for i in range(0, len(cols), 20):
        batch = _read(path, columns=["timestamp", *cols[i : i + 20]])
        for c in batch.columns:
            fv = batch[c].first_valid_index()
            if fv is not None:
                first[c] = fv
    binding = max(first, key=lambda c: first[c])
    return first[binding], binding


def _per_chunk_stats(events: pd.DataFrame, chunk_of: np.ndarray, n_chunks: int):
    """Everything the balance criteria need, precomputed once per chunk."""
    q = np.zeros((n_chunks, 4), dtype="int64")
    years = sorted(events.index.year.unique())
    y = np.zeros((n_chunks, len(years)), dtype="int64")
    lab = np.zeros((n_chunks, 3), dtype="int64")
    atr: list[np.ndarray] = []
    classes = [-1.0, 0.0, 1.0]
    for c in range(n_chunks):
        m = chunk_of == c
        sub = events[m]
        qi = (sub.index.quarter - 1)
        for k in range(4):
            q[c, k] = int((qi == k).sum())
        for j, yr in enumerate(years):
            y[c, j] = int((sub.index.year == yr).sum())
        for j, cl in enumerate(classes):
            lab[c, j] = int((sub[LABEL_COL] == cl).sum())
        atr.append(sub[VOL_COL].dropna().to_numpy())
    return q, y, lab, atr, years


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=int, default=sp.N_CANDIDATES)
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")

    print("Blocked-by-time split")
    print(f"  seed {sp.SPLIT_SEED} | {args.candidates} candidates | "
          f"chunk {sp.CHUNK_MONTHS} months | test fraction {sp.TEST_FRACTION:.0%}")
    print(f"  horizon {sp.LABEL_HORIZON} bars | embargo {sp.EMBARGO_BARS:,} bars\n")

    full_index = _read(FULL, columns=["timestamp"]).index
    sampled = _read(SAMPLED, columns=["timestamp", LABEL_COL, VOL_COL, "gate_stack"])
    print(f"full events    {len(full_index):,}  ({full_index[0]} -> {full_index[-1]})")
    print(f"sampled events {len(sampled):,}")

    start, binding = _usable_start(FULL)
    print(f"usable range starts {start} (binding column: {binding})\n")

    # ---------------------------------------------------------------- chunks
    chunks = sp.build_chunks(start, full_index[-1])
    print(f"=== CHUNK GRID: {len(chunks)} chunks of {sp.CHUNK_MONTHS} months ===")
    print("  boundaries fall on the 26th of Feb / May / Aug / Nov -- ~34 days from")
    print("  the nearest quarter end, so every chunk straddles two calendar")
    print("  quarters and no chunk IS a calendar quarter.\n")

    chunk_of_full = sp.chunk_of_event(full_index, chunks)
    chunk_of_samp = sp.chunk_of_event(sampled.index, chunks)
    in_grid = chunk_of_samp >= 0
    ev = sampled[in_grid]
    q, y, lab, atr, years = _per_chunk_stats(ev, chunk_of_samp[in_grid], len(chunks))

    # ------------------------------------------------------------- selection
    n_test = max(1, round(len(chunks) * sp.TEST_FRACTION))
    cands = sp.candidate_assignments(len(chunks), n_test, n_candidates=args.candidates,
                                     seed=sp.SPLIT_SEED)
    rows = []
    for cand in cands:
        s = sp.score_candidate(cand, n_chunks=len(chunks), quarter_counts=q,
                               year_counts=y, label_counts=lab, atr_by_chunk=atr)
        s["test_chunks"] = tuple(int(x) for x in cand)
        rows.append(s)
    scores = pd.DataFrame(rows)
    best_i, scored = sp.choose_assignment(scores)
    best = np.array(scores.loc[best_i, "test_chunks"], dtype="int64")

    print(f"=== SELECTION: {len(cands)} candidates, {n_test} of {len(chunks)} chunks "
          f"to test ({n_test/len(chunks):.2%}) ===")
    print("  criteria fixed before the draw; winner minimises the SUM OF RANKS")
    print(f"  {'criterion':<22}{'chosen':>12}{'median cand':>14}{'best cand':>12}"
          f"{'worst cand':>12}")
    for c in sp.CRITERIA:
        print(f"  {c:<22}{scores.loc[best_i, c]:>12.4f}{scores[c].median():>14.4f}"
              f"{scores[c].min():>12.4f}{scores[c].max():>12.4f}")
    print(f"  {'rank_sum':<22}{scored.loc[best_i, 'rank_sum']:>12.1f}"
          f"{scored['rank_sum'].median():>14.1f}{scored['rank_sum'].min():>12.1f}"
          f"{scored['rank_sum'].max():>12.1f}")
    print(f"\n  chosen test chunks: {best.tolist()}")

    # ------------------------------------------------- purge / embargo / table
    table = sp.split_table(full_index, chunks, best)
    split = table["split"].to_numpy()

    # ------------------------------------------------- step 4: the constant
    print("\n=== STEP 4: THE gate_stack CONSTANT ===")
    feat = _read(FEATURES)
    bar_chunk = sp.chunk_of_event(feat.index, chunks)
    is_test_chunk = np.zeros(len(chunks), dtype=bool)
    is_test_chunk[best] = True
    train_bars = (bar_chunk >= 0) & ~is_test_chunk[np.clip(bar_chunk, 0, None)]
    train_median = float(np.nanmedian(feat.loc[train_bars, "spread_close"].to_numpy()))
    full_median = float(np.nanmedian(feat["spread_close"].to_numpy()))
    px = float(feat["mid_close"].median())
    print(f"  full-sample ceiling  {full_median:.10f}  ({full_median / px * 1e4:.4f} bps)")
    print(f"  train-fold ceiling   {train_median:.10f}  ({train_median / px * 1e4:.4f} bps)")
    print(f"  difference           {train_median - full_median:+.10f}  "
          f"({(train_median / full_median - 1) * 100:+.4f}%)")
    print(f"  (frame was built with {FULL_SAMPLE_CEILING:.10f}; "
          f"recomputed full-sample median matches: "
          f"{np.isclose(full_median, FULL_SAMPLE_CEILING, rtol=1e-9)})")
    print(f"  15-min bars in train chunks: {int(train_bars.sum()):,} of {len(feat):,}")

    # WHY the two ceilings can coincide: the 15-min spread is heavily
    # discretised, so the median sits ON a tick and is robust to resampling.
    sc = feat["spread_close"].dropna()
    at_med = float((sc == full_median).mean())
    below = float((sc < full_median).mean())
    print(f"  spread_close exactly AT the median value: {at_med:.2%} of bars "
          f"({below:.2%} strictly below, {1 - at_med - below:.2%} strictly above)")
    print("  -> the median is a MASS POINT, not an interpolated quantile, so any "
          "subsample's\n     median lands on the same tick unless the mass point "
          "itself moves.")

    gate_train = htf.gate_stack_column(feat, spread_gate_max=train_median)
    aligned = align_completed_series(
        pd.Series(gate_train.astype("float64"), index=feat.index), full_index, "15min"
    )
    values = aligned.to_numpy()
    missing = np.isnan(values)
    gate_tf = pd.array(values != 0, dtype="boolean")
    gate_tf[missing] = pd.NA
    table["gate_stack_trainfold"] = pd.Series(gate_tf, index=full_index)

    old = _read(FULL, columns=["timestamp", "gate_stack"])["gate_stack"]
    both = old.notna() & table["gate_stack_trainfold"].notna()
    changed = both & (old.astype("boolean") != table["gate_stack_trainfold"])
    on = changed & table["gate_stack_trainfold"].fillna(False)
    off = changed & ~table["gate_stack_trainfold"].fillna(True)
    print(f"\n  events changing gate_stack under the train-fold ceiling: "
          f"{int(changed.sum()):,} of {int(both.sum()):,} ({changed.sum()/both.sum():.4%})")
    print(f"    False -> True (ceiling loosened): {int(on.sum()):,}")
    print(f"    True -> False (ceiling tightened): {int(off.sum()):,}")
    print(f"  gate_stack True: full-sample {int(old.fillna(False).sum()):,} -> "
          f"train-fold {int(table['gate_stack_trainfold'].fillna(False).sum()):,}")

    print("\n  OTHER FULL-SAMPLE-DERIVED CONSTANTS IN THE FRAME (audited, not assumed):")
    from fxalgo.features import excursion, moments

    at_cap = 0
    for tf in htf.HIGHER_TIMEFRAMES:
        for side in ("above", "below"):
            col = f"bars_{side}_band_{tf}"
            if col in feat.columns:
                at_cap += int((feat[col] == excursion.RUN_CAP).sum())
    print(f"    excursion.RUN_CAP = {excursion.RUN_CAP} -- chosen from the FULL-sample")
    print(f"      run-length distribution. Real but tiny: {at_cap:,} 15-min bar-values")
    print("      sit at the cap across the 12 run columns; a train-only "
          "distribution would move it at most one step.")
    print(f"    moments.PERCENTILE_WINDOW_BARS = {moments.PERCENTILE_WINDOW_BARS:,} -- "
          "derived from")
    print("      full-sample bars-per-session (94.84). A WINDOW LENGTH, not a "
          "threshold: it")
    print("      sets how much history a rank sees, not where a cut falls. "
          "Structural, not tuned.")
    print("    spread_pctile_trailing / _rolling -- trailing ranks, causal, NO "
          "full-sample constant.")
    print("    ret_skew_96_pctile / ret_kurt_96_pctile -- same, trailing ranks.")
    print("    gate_cell_d -- RSI 30/70 and cost_multiple 2.0 are A PRIORI "
          "constants, not")
    print("      derived from this data. They carry the project-wide in-sample "
          "SELECTION")
    print("      caveat, but nothing is computed from the test period to set them.")

    # ---------------------------------------------------------------- write
    table.to_parquet(OUT_PATH, engine="pyarrow")

    # ----------------------------------------------------------- diagnostics
    print("\n" + "=" * 100)
    print("DIAGNOSTICS")
    print("=" * 100)

    samp_split = table.loc[sampled.index, "split"].to_numpy()
    print("\n=== A) EVENT COUNTS ===")
    print(f"  {'split':<12}{'full':>14}{'full %':>10}{'sampled':>12}{'sampled %':>12}")
    for v in sp.SPLIT_VALUES:
        nf, ns = int((split == v).sum()), int((samp_split == v).sum())
        print(f"  {v:<12}{nf:>14,d}{nf/len(split):>10.2%}{ns:>12,d}{ns/len(samp_split):>12.2%}")
    print(f"  {'TOTAL':<12}{len(split):>14,d}{1:>10.2%}{len(samp_split):>12,d}{1:>12.2%}")
    graded = split != sp.WARMUP
    print(f"  within the split domain (excludes warmup): {int(graded.sum()):,} full, "
          f"{int((samp_split != sp.WARMUP).sum()):,} sampled")

    print("\n=== B) CHUNK TABLE ===")
    print(f"  {'id':>3}  {'start':<12}{'end':<12}{'assignment':<12}"
          f"{'full events':>13}{'sampled':>10}")
    for _, r in chunks.iterrows():
        cid = int(r["chunk_id"])
        assign = "TEST" if cid in set(best.tolist()) else "train"
        nf = int((chunk_of_full == cid).sum())
        ns = int((chunk_of_samp == cid).sum())
        print(f"  {cid:>3}  {str(r['chunk_start'].date()):<12}"
              f"{str(r['chunk_end'].date()):<12}{assign:<12}{nf:>13,d}{ns:>10,d}")

    print("\n=== C) BALANCE CRITERIA OF THE CHOSEN SPLIT ===")
    chosen = scores.loc[best_i]
    is_test_c = np.zeros(len(chunks), dtype=bool)
    is_test_c[best] = True
    qt, qtr = q[is_test_c].sum(axis=0), q[~is_test_c].sum(axis=0)
    print("  a) calendar-quarter coverage of TEST events")
    print(f"     {'quarter':<10}{'test':>10}{'share':>9}{'train share':>14}")
    for k in range(4):
        print(f"     Q{k+1:<9}{qt[k]:>10,d}{qt[k]/qt.sum():>9.2%}{qtr[k]/qtr.sum():>14.2%}")
    print(f"     max test share in one quarter: {chosen['quarter_max_share']:.2%} "
          f"(0.25 would be perfectly even)")
    yt, ytr = y[is_test_c].sum(axis=0), y[~is_test_c].sum(axis=0)
    print("  b) year coverage of TEST events")
    print(f"     {'year':<10}{'test':>10}{'share':>9}{'train share':>14}")
    for j, yr in enumerate(years):
        print(f"     {yr:<10}{yt[j]:>10,d}{yt[j]/yt.sum():>9.2%}{ytr[j]/ytr.sum():>14.2%}")
    print(f"     max test share in one year: {chosen['year_max_share']:.2%}")
    lt, ltr = lab[is_test_c].sum(axis=0), lab[~is_test_c].sum(axis=0)
    print("  c) label distribution")
    print(f"     {'label':>6}{'test':>12}{'test %':>10}{'train %':>10}{'delta pp':>11}")
    for j, cl in enumerate((-1, 0, 1)):
        pt, ptr = lt[j] / lt.sum(), ltr[j] / ltr.sum()
        print(f"     {cl:>6}{lt[j]:>12,d}{pt:>10.4%}{ptr:>10.4%}{(pt-ptr)*100:>+11.4f}")
    print(f"     max divergence: {chosen['label_max_div_pp']:.4f} pp")
    print("  d) volatility")
    print(f"     median atr_bps_15m  test {chosen['median_atr_test']:.4f}  "
          f"train {chosen['median_atr_train']:.4f}  "
          f"relative divergence {chosen['vol_rel_div']:.4%}")
    print("  e) spread across the window")
    print(f"     test chunk ids {best.tolist()}; adjacent pairs "
          f"{int(chosen['adjacent_pairs'])}; gaps between them "
          f"{np.diff(np.sort(best)).tolist()}")

    print("\n=== D) EFFECTIVE SAMPLE SIZE (the operative numbers) ===")
    print(f"  mean label window at k=2.0: {MEAN_WINDOW_BARS:.2f} one-minute bars")
    print(f"  {'side':<12}{'events':>14}{'independent windows':>22}")
    for v in (sp.TRAIN, sp.TEST):
        n = int((split == v).sum())
        print(f"  {v:<12}{n:>14,d}{n / MEAN_WINDOW_BARS:>22,.0f}")
    n_test_ev = int((split == sp.TEST).sum())
    n_train_ev = int((split == sp.TRAIN).sum())
    print("  full window reference: 1,846,315 events -> 13,619 windows")
    print(f"\n  >>> TEST SIDE: {n_test_ev / MEAN_WINDOW_BARS:,.0f} INDEPENDENT LABEL "
          f"WINDOWS <<<")
    print("  That, not the event count, is what any out-of-sample claim rests on.")
    print(f"  train / test ratio of independent windows: "
          f"{n_train_ev / max(n_test_ev, 1):.2f}")

    print("\n=== E) PURGE AND EMBARGO COST ===")
    n_purged = int((split == sp.PURGED).sum())
    n_emb = int((split == sp.EMBARGOED).sum())
    print(f"  purged   {n_purged:>10,d}  {n_purged/len(split):>8.4%} of all events, "
          f"{n_purged/max(graded.sum(),1):>8.4%} of the split domain")
    print(f"  embargoed{n_emb:>10,d}  {n_emb/len(split):>8.4%} of all events, "
          f"{n_emb/max(graded.sum(),1):>8.4%} of the split domain")
    print(f"  combined {n_purged+n_emb:>10,d}  "
          f"{(n_purged+n_emb)/max(graded.sum(),1):>8.4%} of the split domain")
    print(f"  embargo length {sp.EMBARGO_BARS:,} bars = "
          f"{sp.EMBARGO_BARS/sp.LABEL_HORIZON:.1f}x the {sp.LABEL_HORIZON}-bar maximum "
          f"window, {sp.EMBARGO_BARS/MEAN_WINDOW_BARS:.1f}x the mean")
    for mult in (2, 5):
        print(f"    at {mult}x the embargo the cost would be about "
              f"{n_emb*mult:,} events ({n_emb*mult/max(graded.sum(),1):.2%})")

    print("\n" + "-" * 100)
    print(f"Wrote {OUT_PATH}")
    print("  columns: split, chunk_id, chunk_start, chunk_end, gate_stack_trainfold")
    print("  keyed by event timestamp over the FULL index -- joins onto either")
    print("  events file; gate_stack is NOT overwritten anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
