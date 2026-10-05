"""Assemble the minute-event training dataset: labels x features.

BUILD ONLY -- no split, no model, no training, no predictiveness test.

INPUTS
  labels    data/labels/eurusd_triple_barrier.parquet      (1-minute events)
  features  data/resampled/15min_features_exc/EURUSD.parquet (178 columns)
  bands     data/resampled/{30min,1h,2h,4h,daily,weekly}_features/EURUSD.parquet
            -- the higher-timeframe Bollinger bands, which the 178-column frame
            does not carry (it carries only the band POSITIONS)
  raw       1-minute bid/ask bars, used ONLY to re-verify that the labels'
            anchor_price is still mid open(t+1)

OUTPUTS
  data/training/eurusd_events_full.parquet   every event, sampled_flag included
  data/training/eurusd_events.parquet        the sampled subset

COLUMN CONTRACT
--------------
The file carries only what is a feature or is needed to trade and diagnose:
the 178 feature columns MINUS the 43 that carry price units (decision F7),
`anchor_price`, the k=2.0 label family, `ret_at_time_barrier_bps`, the five
calendar columns, and three bookkeeping columns. The other four barrier widths
and `atr_at_entry` stay in data/labels/eurusd_triple_barrier.parquet and rejoin
on timestamp.

Higher-timeframe bands project DIRECTLY onto the event index at their own
timeframe's freq, so a 30-minute bar closing at 09:30 is visible at 09:31. The
mixed vintage against carried-forward columns is accepted -- see
`training.dataset.frozen_levels`.

UNIT NOTE ON dist_ema_*
-----------------------
The specification asked for `(anchor_price - ema_n) / ema_n, in bps`. Those two
clauses disagree: the formula is the FRACTIONAL distance, and bps would be the
same quantity times 1e4. This build emits the FRACTIONAL form, matching
`features.trend.add_features` exactly, because the column keeps its existing
name: emitting bps under `dist_ema_9` would put it on a different scale from
every prior study that used the column. Multiply by 1e4 downstream if bps are
wanted.

MEMORY
------
The full joined frame is ~1.85M rows x ~150 columns, so
the build streams in groups of BLOCKS (never splitting a 15-minute block across
a chunk) and writes parquet row-group by row-group. Chunking cannot change the
result: the block index and the sample mask are computed once over the whole
event index and sliced per chunk.

Usage:
    uv run python scripts/build_training_events.py [--blocks-per-chunk 10000]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fxalgo.data.resample import session_close_utc
from fxalgo.features.htf import HIGHER_TIMEFRAMES
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.training import dataset as ds
from fxalgo.utils.logging import setup_logging

PAIR = "EURUSD"
K = 2.0
LABEL_COL = f"label_{K}"
LABELS = PROJECT_ROOT / "data" / "labels" / "eurusd_triple_barrier.parquet"
RESAMPLED = PROJECT_ROOT / "data" / "resampled"
FEATURES = RESAMPLED / "15min_features_exc" / f"{PAIR}.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "training"
FULL_PATH = OUT_DIR / "eurusd_events_full.parquet"
SAMPLED_PATH = OUT_DIR / "eurusd_events.parquet"
N_EFF = 13619.0


def _load(path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


class Accumulator:
    """Streaming diagnostics -- everything that needs the FULL joined set.

    Correlations are accumulated as raw moments so check C is exact on all
    1.85M rows rather than estimated from the sample.
    """

    def __init__(self, recomputed: list[str]) -> None:
        self.recomputed = recomputed
        self.rows = 0
        self.notna: dict[str, int] = {}
        self.first_valid: dict[str, pd.Timestamp] = {}
        self.sampled_notna: dict[str, int] = {}
        self.sampled_rows = 0
        self.all_defined = 0
        self.first_all_defined: pd.Timestamp | None = None
        self.block_std_sum = dict.fromkeys(recomputed, 0.0)
        self.block_std_n = dict.fromkeys(recomputed, 0)
        self.block_zero_var = dict.fromkeys(recomputed, 0)
        self.moments = {c: np.zeros(6) for c in recomputed}

    def update(self, chunk: pd.DataFrame, feature_cols: list[str],
               frozen: pd.DataFrame) -> None:
        self.rows += len(chunk)
        notna = chunk.notna()
        for col in chunk.columns:
            self.notna[col] = self.notna.get(col, 0) + int(notna[col].sum())
            if col not in self.first_valid:
                fv = chunk[col].first_valid_index()
                if fv is not None:
                    self.first_valid[col] = fv

        defined = notna[feature_cols].all(axis=1)
        self.all_defined += int(defined.sum())
        if self.first_all_defined is None and defined.any():
            self.first_all_defined = chunk.index[defined.to_numpy().argmax()]

        sampled = chunk[chunk["sampled_flag"]]
        self.sampled_rows += len(sampled)
        s_notna = sampled.notna()
        for col in chunk.columns:
            self.sampled_notna[col] = self.sampled_notna.get(col, 0) + int(s_notna[col].sum())

        grouped = chunk.groupby("block_id")
        for col in self.recomputed:
            std = grouped[col].std()          # ddof=1, NaN for singleton blocks
            good = std.dropna()
            self.block_std_sum[col] += float(good.sum())
            self.block_std_n[col] += int(len(good))
            self.block_zero_var[col] += int((good == 0).sum())

            live = chunk[col]
            ref = frozen[col]
            ok = live.notna() & ref.notna()
            x = live[ok].to_numpy(dtype="float64")
            y = ref[ok].to_numpy(dtype="float64")
            self.moments[col] += np.array(
                [len(x), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum()]
            )

    def correlation(self, col: str) -> float:
        n, sx, sy, sxx, syy, sxy = self.moments[col]
        if n < 2:
            return float("nan")
        cov = sxy / n - (sx / n) * (sy / n)
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / np.sqrt(vx * vy))


def _verify_anchor(labels: pd.DataFrame) -> None:
    """Re-check the anchor against the RAW 1-minute bars before using it."""
    from fxalgo.features.loader import load_pair

    raw = load_pair(PAIR)
    common = labels.index.intersection(raw.index)
    a = labels.loc[common, ds.ANCHOR_COL]
    want = raw["mid_open"].shift(-1).reindex(common)
    ok = a.notna() & want.notna()
    exact = float((a[ok] == want[ok]).mean())
    print(f"anchor_price == raw minute mid_open(t+1) on {exact:.6%} of "
          f"{int(ok.sum()):,} comparable events")
    if exact < 1.0:
        raise SystemExit("anchor_price is not mid_open(t+1) -- refusing to build")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks-per-chunk", type=int, default=10_000)
    ap.add_argument("--skip-anchor-check", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = _load(LABELS)
    features = _load(FEATURES)
    htf_frames = {tf: _load(RESAMPLED / f"{tf}_features" / f"{PAIR}.parquet")
                  for tf in HIGHER_TIMEFRAMES}

    print("Training-event assembly")
    print(f"  labels    {len(labels):,} events x {labels.shape[1]} cols "
          f"({labels.index[0]} -> {labels.index[-1]})")
    print(f"  features  {len(features):,} bars x {features.shape[1]} cols")
    print(f"  seed {ds.SAMPLE_SEED} | {ds.EVENTS_PER_BLOCK} events per block | "
          f"min spacing {ds.MIN_SPACING_MINUTES} min\n")

    if not args.skip_anchor_check:
        _verify_anchor(labels)

    blocks = ds.block_id(labels[ds.BLOCK_KEY_COL])
    mask = ds.sample_blocks(blocks)
    starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
    stops = np.r_[starts[1:], len(blocks)]
    n_blocks = len(starts)
    sizes = stops - starts
    per_block = np.add.reduceat(mask.astype("int64"), starts)
    print(f"blocks {n_blocks:,} | sizes {dict(pd.Series(sizes).value_counts().sort_index())}")
    print(f"blocks yielding fewer than {ds.EVENTS_PER_BLOCK} events: "
          f"{int((per_block < ds.EVENTS_PER_BLOCK).sum()):,} "
          f"(sizes {sorted(set(sizes[per_block < ds.EVENTS_PER_BLOCK]))})")
    print(f"selected events: {int(mask.sum()):,}\n")

    recomputed = ds.recomputed_columns()
    kept_features = ds.kept_feature_columns(features)
    dropped = [c for c in features.columns if c not in set(kept_features)]
    print(f"price-unit feature columns dropped: {len(dropped)} "
          f"(kept {len(kept_features)} of {features.shape[1]})")
    for c in dropped:
        print(f"    - {c}")
    print()
    acc = Accumulator(recomputed)
    writer_full = writer_sampled = None
    dtypes = None
    feature_cols = kept_features

    group_starts = range(0, n_blocks, args.blocks_per_chunk)
    for gi, g0 in enumerate(group_starts, start=1):
        g1 = min(g0 + args.blocks_per_chunk, n_blocks)
        lo, hi = int(starts[g0]), int(stops[g1 - 1])
        chunk = ds.assemble(
            labels.iloc[lo:hi], features, htf_frames,
            sampled_mask=mask[lo:hi], blocks=blocks[lo:hi],
        )
        if dtypes is None:
            dtypes = chunk.dtypes.to_dict()
        else:
            chunk = chunk.astype(dtypes)
        # The un-recomputed 15-minute values, for diagnostic C only. Computed
        # here and discarded -- they are deliberately NOT written to any file.
        frozen = ds.join_features(chunk.index, features[recomputed])
        acc.update(chunk, feature_cols, frozen)

        flat = chunk.reset_index(names="timestamp")
        table = pa.Table.from_pandas(flat, preserve_index=False)
        if writer_full is None:
            writer_full = pq.ParquetWriter(FULL_PATH, table.schema, compression="snappy")
            writer_sampled = pq.ParquetWriter(SAMPLED_PATH, table.schema, compression="snappy")
        writer_full.write_table(table)
        sampled = flat[flat["sampled_flag"]]
        if len(sampled):
            writer_sampled.write_table(pa.Table.from_pandas(sampled, preserve_index=False,
                                                            schema=table.schema))
        print(f"  chunk {gi}/{len(group_starts)}: blocks {g0:,}-{g1 - 1:,} | "
              f"rows {hi - lo:,} | sampled {int(mask[lo:hi].sum()):,}", flush=True)

    writer_full.close()
    writer_sampled.close()

    # ------------------------------------------------------------ diagnostics
    n = acc.rows
    all_cols = list(dtypes)
    print("\n" + "=" * 112)
    print(f"DIAGNOSTICS -- {PAIR}, {n:,} events")
    print("=" * 112)

    print("\n=== A) ROW COUNTS AND WARMUP ===")
    print(f"  full joined events              : {n:,}")
    print(f"  sampled events                  : {acc.sampled_rows:,} "
          f"({acc.sampled_rows / n:.2%})")
    print(f"  events with all {len(feature_cols)} kept feature columns defined: "
          f"{acc.all_defined:,} ({acc.all_defined / n:.2%})")
    fv = {c: acc.first_valid.get(c) for c in feature_cols if c in acc.first_valid}
    binding = max(fv, key=lambda c: fv[c])
    print(f"  binding warmup column           : {binding} "
          f"(first valid {fv[binding]})")
    never = [c for c in feature_cols if c not in acc.first_valid]
    if never:
        print(f"  columns never valid             : {never}")
    print(f"  usable date range               : {acc.first_all_defined} -> "
          f"{labels.index[-1]}")

    print("\n=== B) WITHIN-BLOCK STANDARD DEVIATION OF EACH RECOMPUTED COLUMN ===")
    print("  Mean over blocks of the per-block std (ddof=1). This is how much the")
    print("  live-price recomputation actually moves the column inside one block.")
    print(f"  {'column':<30}{'blocks':>10}{'mean within-block sd':>22}"
          f"{'zero-variance blocks':>22}")
    for col in recomputed:
        cnt = acc.block_std_n[col]
        mean_sd = acc.block_std_sum[col] / cnt if cnt else float("nan")
        zero = acc.block_zero_var[col]
        flag = "   <-- NOT MOVING" if cnt and mean_sd == 0 else ""
        print(f"  {col:<30}{cnt:>10,d}{mean_sd:>22.8f}"
              f"{zero / cnt if cnt else float('nan'):>21.3%}{flag}")

    print("\n=== C) RECOMPUTED vs THE UN-RECOMPUTED 15-MINUTE VALUE ===")
    print("  Exact over all rows. The 15-minute value is reconstructed HERE for")
    print("  the measurement and written nowhere -- the _15m_frozen columns are")
    print("  gone from the output by design.")
    print(f"  {'column':<30}{'pairs':>12}{'r':>10}   verdict")
    for col in recomputed:
        r = acc.correlation(col)
        pairs = int(acc.moments[col][0])
        verdict = ("COSMETIC -- recomputation barely changes it" if r > 0.999
                   else "materially different")
        print(f"  {col:<30}{pairs:>12,d}{r:>10.5f}   {verdict}")

    print(f"\n=== D) LABEL DISTRIBUTION AT k={K}: SAMPLED vs FULL ===")
    full_lbl = labels[LABEL_COL]
    samp_lbl = full_lbl[mask]
    print(f"  {'label':>8}{'full':>14}{'full %':>10}{'sampled':>12}{'sampled %':>12}"
          f"{'delta pp':>10}")
    for v in (-1.0, 0.0, 1.0):
        fp = float((full_lbl == v).sum()) / float(full_lbl.notna().sum())
        sp = float((samp_lbl == v).sum()) / float(samp_lbl.notna().sum())
        print(f"  {int(v):>8}{int((full_lbl == v).sum()):>14,d}{fp:>10.4%}"
              f"{int((samp_lbl == v).sum()):>12,d}{sp:>12.4%}{(sp - fp) * 100:>+10.4f}")
    print(f"  NaN labels: full {int(full_lbl.isna().sum()):,}, "
          f"sampled {int(samp_lbl.isna().sum()):,}")

    print("\n=== E) NaN FRACTION PER COLUMN IN THE SAMPLED SET ===")
    rows_s = acc.sampled_rows
    nanfrac = {c: 1 - acc.sampled_notna.get(c, 0) / rows_s for c in all_cols}
    ranked = sorted(nanfrac.items(), key=lambda kv: -kv[1])
    print(f"  columns with ANY NaN: {sum(1 for v in nanfrac.values() if v > 0):,} "
          f"of {len(all_cols):,}")
    print(f"  {'column':<36}{'NaN%':>9}")
    for c, v in ranked[:25]:
        if v == 0:
            break
        print(f"  {c:<36}{v:>9.3%}")
    clean = [c for c, v in nanfrac.items() if v == 0]
    print(f"  fully-populated columns: {len(clean):,}")

    print("\n=== F) FRESHNESS GAINED BY PROJECTING BANDS DIRECTLY ===")
    print("  A higher-TF band now lands on the event index at its own close.")
    print("  Routing it through the 15-minute frame first would delay it; this")
    print("  measures how often, and by how much, that mattered.")
    ev = labels.index.to_numpy()
    b15_close = (features.index + pd.Timedelta(ds.BASE_FREQ)).to_numpy()
    b15 = np.searchsorted(b15_close, ev, side="right") - 1
    print(f"  {'timeframe':<12}{'events stale under 2-hop':>26}{'share':>10}"
          f"{'median age (min)':>19}{'p95 age (min)':>16}")
    for tf in HIGHER_TIMEFRAMES:
        frame = htf_frames[tf]
        closes = (session_close_utc(frame.index, tf)
                  if tf in ("2h", "4h", "daily", "weekly")
                  else frame.index + pd.Timedelta(tf)).to_numpy()
        direct = np.searchsorted(closes, ev, side="right") - 1
        on15 = np.searchsorted(closes, features.index.to_numpy(), side="right") - 1
        two_hop = np.where(b15 >= 0, on15[np.clip(b15, 0, None)], -1)
        differs = direct != two_hop
        newer = differs & (direct >= 0)
        delay = np.zeros(len(ev))
        delay[newer] = (ev[newer] - closes[direct[newer]]) / np.timedelta64(1, "m")
        # "age" = how long the newly-visible bar had ALREADY been closed at the
        # event. Median and p95, not mean/max: a bar that closes on Friday
        # evening is days old at Sunday's reopen, which says something about the
        # weekend, not about the projection.
        ages = delay[newer]
        print(f"  {tf:<12}{int(differs.sum()):>26,d}{differs.mean():>10.2%}"
              f"{(np.median(ages) if newer.any() else float('nan')):>19.2f}"
              f"{(np.percentile(ages, 95) if newer.any() else float('nan')):>16.2f}")

    print("\n=== G) COLUMN INVENTORY ===")
    groups = {
        "bookkeeping": list(ds.AUX_COLUMNS),
        "recomputed at the live minute price": recomputed,
        "carried forward from the 15-min frame": [
            c for c in feature_cols if c not in set(recomputed)],
        "label / trade": list(ds.KEPT_LABEL_COLUMNS),
    }
    total = sum(len(v) for v in groups.values())
    for name, cols in groups.items():
        print(f"  {name:<40}{len(cols):>5}")
    print(f"  {'TOTAL data columns':<40}{total:>5}   "
          f"(+ timestamp = {total + 1} in the file)")
    print(f"  frame actually carries {len(all_cols)} data columns -- "
          f"{'CONFIRMED' if total == len(all_cols) else 'MISMATCH'}")
    for name, cols in groups.items():
        print(f"\n  {name} ({len(cols)}):")
        for i in range(0, len(cols), 4):
            print("    " + "  ".join(f"{c:<32}" for c in cols[i:i + 4]).rstrip())

    print("\n" + "-" * 112)
    print("CAVEATS carried by this file:")
    print("  * Within a block only the PRICE moves. Every level, every history")
    print("    column and the gates are frozen at their own last closed bar.")
    print("  * MIXED VINTAGE, accepted: a higher-TF band position can describe a")
    print("    newer bar than the carried-forward columns beside it, for up to")
    print("    15 minutes. Freshness was preferred to internal tidiness.")
    print("  * gate_stack inherits the full-sample spread ceiling. It is causal")
    print("    given that constant; the constant is not. Re-derive per fold.")
    print("  * Sampling is reversible: eurusd_events_full.parquet holds every")
    print(f"    event with sampled_flag, seed {ds.SAMPLE_SEED}.")
    print("  * No split, no model, no predictiveness measured.")
    print(f"\nWrote {FULL_PATH}")
    print(f"Wrote {SAMPLED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
