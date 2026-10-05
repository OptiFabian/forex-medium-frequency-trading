"""Tests for the label/feature join and the live-price recomputation.

The two that carry the weight:

1. CORRUPTION over the WHOLE chain (`test_no_lookahead_across_the_join`). The
   join is where a 15-minute feature meets a 1-minute label, and an off-by-one
   there produces lookahead that presents as an excellent model rather than as
   a bug. Garbage every 1-minute bar after a cut, rebuild bars -> features ->
   higher-timeframe families -> labels -> join -> recompute, and demand
   bit-identical output at every event whose full 240-bar label window closed
   at or before the cut.

2. WITHIN-BLOCK VARIATION (`test_recomputed_columns_vary_within_a_block`).
   This is the test that proves step 2 actually happened. A build that
   silently carried every column forward would pass every other test in this
   file -- the join would be causal, the anchors consistent, the sampling
   spaced -- and would produce 15 identical feature vectors per block.

The synthetic feature frame runs the pipeline plus the higher-timeframe family,
which is what supplies every recomputed column. The moments and excursion
families are deliberately left out: they contribute only CARRIED-FORWARD
columns, they each already have their own corruption test, and building them
here would triple the runtime without testing anything new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.resample import (
    resample_merged,
    resample_sessions,
    session_close_utc,
    session_keys,
)
from fxalgo.features import htf
from fxalgo.features.build import apply_pipeline
from fxalgo.labels.triple_barrier import HORIZON, build_labels
from fxalgo.training import dataset as ds

CLOCK_TFS = ("30min", "1h")
SESSION_TFS = ("2h", "4h", "daily", "weekly")
PRICE_COLS = [
    "open_bid", "high_bid", "low_bid", "close_bid",
    "open_ask", "high_ask", "low_ask", "close_ask",
]
RECOMPUTED = ds.recomputed_columns()


# --------------------------------------------------------------- fixtures


def _minute_index(start: str, end: str) -> pd.DatetimeIndex:
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    keys = session_keys(idx, "daily")
    return idx[~np.isin(keys.weekday, (5, 6))]


def _synthetic_minutes(start="2024-01-01", end="2025-01-01", seed=23) -> pd.DataFrame:
    idx = _minute_index(start, end)
    rng = np.random.default_rng(seed)
    n = len(idx)
    mid_close = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 8e-5, n)))
    mid_open = np.concatenate([[mid_close[0]], mid_close[:-1]])
    wiggle = np.abs(rng.normal(0.0, 6e-5, n))
    mid_high = np.maximum(mid_open, mid_close) + wiggle
    mid_low = np.minimum(mid_open, mid_close) - wiggle
    spread = 0.00008 + np.abs(rng.normal(0.0, 0.00004, n))
    half = spread / 2.0
    df = pd.DataFrame(index=idx)
    for name, mid in (("open", mid_open), ("high", mid_high),
                      ("low", mid_low), ("close", mid_close)):
        df[f"{name}_bid"] = mid - half
        df[f"{name}_ask"] = mid + half
    df.index.name = "timestamp"
    return df


def _htf_frames(minutes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {tf: apply_pipeline(resample_merged(minutes, tf)) for tf in CLOCK_TFS}
    frames.update({tf: apply_pipeline(resample_sessions(minutes, tf)) for tf in SESSION_TFS})
    return frames


def _features(minutes: pd.DataFrame, *, spread_gate_max: float | None = None):
    base = apply_pipeline(resample_merged(minutes, "15min"))
    frames = _htf_frames(minutes)
    if spread_gate_max is None:
        spread_gate_max = float(np.nanmedian(base["spread_close"].to_numpy()))
    return htf.add_features(base, frames, spread_gate_max=spread_gate_max), frames


def _median_spread(minutes: pd.DataFrame) -> float:
    """The gate_stack ceiling, computed ONCE on clean data.

    It is a full-sample statistic, so re-deriving it from corrupted data would
    move `gate_stack` at earlier bars -- the known hazard that
    `test_htf.py::test_spread_gate_threshold_is_a_config_constant_not_a_feature`
    pins deliberately. Freezing it here keeps the corruption test measuring the
    JOIN rather than re-measuring that.
    """
    return float(np.nanmedian(
        apply_pipeline(resample_merged(minutes, "15min"))["spread_close"].to_numpy()))


def _minute_bars(minutes: pd.DataFrame) -> pd.DataFrame:
    """The 1-minute merged frame in the shape `build_labels` expects."""
    out = minutes.copy()
    for col in ("open", "high", "low", "close"):
        out[f"mid_{col}"] = (out[f"{col}_bid"] + out[f"{col}_ask"]) / 2.0
    out["spread_close"] = (out["close_ask"] - out["close_bid"]).clip(lower=0.0)
    return out


def _build(minutes: pd.DataFrame, *, spread_gate_max: float | None = None) -> pd.DataFrame:
    feats, frames = _features(minutes, spread_gate_max=spread_gate_max)
    bars = _minute_bars(minutes)
    labels = build_labels(bars, feats["atr_14"])
    blocks = ds.block_id(labels[ds.BLOCK_KEY_COL])
    mask = ds.sample_blocks(blocks)
    return ds.assemble(labels, feats, frames, sampled_mask=mask, blocks=blocks)


@pytest.fixture(scope="module")
def minutes() -> pd.DataFrame:
    return _synthetic_minutes()


@pytest.fixture(scope="module")
def built(minutes) -> pd.DataFrame:
    return _build(minutes)


# ------------------------------------------------------ 0. shape and contract


def test_all_recomputed_columns_are_present_and_no_frozen_twins_are(built):
    for col in RECOMPUTED:
        assert col in built.columns
    assert len(RECOMPUTED) == 25
    assert not [c for c in built.columns if c.endswith("_15m_frozen")]


def test_price_unit_columns_are_dropped(built):
    assert len(ds.PRICE_UNIT_COLUMNS) == 43
    assert not (set(ds.PRICE_UNIT_COLUMNS) & set(built.columns))


def test_only_the_kept_label_columns_survive(built):
    for col in ds.KEPT_LABEL_COLUMNS:
        assert col in built.columns
    for col in ("label_1.0", "label_3.0", "bars_to_touch_1.5",
                "touch_price_2.5", "ambiguous_3.0", "atr_at_entry"):
        assert col not in built.columns


def test_label_columns_survive_the_join(built):
    for col in ("anchor_price", "label_2.0", "bars_to_touch_2.0", "touch_price_2.0"):
        assert col in built.columns


def test_colliding_column_names_are_rejected(minutes):
    feats, frames = _features(minutes)
    bars = _minute_bars(minutes)
    labels = build_labels(bars, feats["atr_14"])
    clash = feats.assign(anchor_price=1.0)
    with pytest.raises(ValueError, match="collide"):
        ds.assemble(labels, clash, frames)


# ------------------------------------------- 1. CAUSALITY ACROSS THE JOIN


def _corrupt_minutes_after(minutes: pd.DataFrame, cut: pd.Timestamp, value=9.99):
    out = minutes.copy()
    mask = out.index >= cut
    assert mask.any() and not mask.all()
    for col in PRICE_COLS:
        out.loc[mask, col] = value + (0.5 if col.endswith("ask") else 0.0)
    return out


@pytest.mark.parametrize("frac", [0.6, 0.85])
def test_no_lookahead_across_the_join(minutes, built, frac):
    """Events whose 240-bar window closed before the cut must not move."""
    cut_pos = int(len(minutes) * frac)
    cut = minutes.index[cut_pos]
    corrupted = _build(_corrupt_minutes_after(minutes, cut),
                       spread_gate_max=_median_spread(minutes))

    # Event i scans bars i+1 .. i+HORIZON, so its window closes at bar i+HORIZON.
    # The last untouched bar is cut_pos - 1.
    safe = built.index[: max(cut_pos - HORIZON - 1, 0)]
    assert len(safe) > 10_000
    a = built.loc[safe]
    b = corrupted.loc[safe]
    assert a.index.equals(b.index)

    checked = 0
    for col in a.columns:
        if a[col].dtype == object or str(a[col].dtype) == "category":
            assert a[col].equals(b[col]), f"LOOKAHEAD in '{col}'"
            checked += 1
            continue
        x = pd.to_numeric(a[col], errors="coerce").to_numpy(dtype="float64")
        y = pd.to_numeric(b[col], errors="coerce").to_numpy(dtype="float64")
        assert not (np.isnan(x) != np.isnan(y)).any(), f"NaN pattern moved in '{col}'"
        m = ~np.isnan(x)
        assert np.array_equal(x[m], y[m]), f"LOOKAHEAD in '{col}'"
        checked += 1
    assert checked == len(a.columns)


def test_corruption_actually_bites(minutes, built):
    """Guard the guard: the corruption must move things AFTER the cut."""
    cut_pos = int(len(minutes) * 0.6)
    cut = minutes.index[cut_pos]
    corrupted = _build(_corrupt_minutes_after(minutes, cut),
                       spread_gate_max=_median_spread(minutes))
    tail = built.index[built.index > cut]
    for col in ("anchor_price", "bb_pct_b", "bb_position_daily"):
        x = built.loc[tail, col].dropna()
        y = corrupted.loc[tail, col].reindex(x.index)
        assert len(x) > 100, col
        assert not np.array_equal(x.to_numpy(), y.to_numpy()), col


# --------------------------- 2. THE TEST THAT PROVES STEP 2 HAPPENED


def test_recomputed_columns_vary_within_a_block(built):
    """Recomputed columns must MOVE inside a block; carried-forward must not."""
    # Take blocks from the TAIL: the first few hundred sit inside the weekly
    # Bollinger warmup, where the slower band positions are still NaN.
    full = built[built["block_id"].map(built["block_id"].value_counts()) == 15]
    sub = full.iloc[-200 * 15:]
    grouped = sub.groupby("block_id")

    for col in RECOMPUTED:
        nunique = grouped[col].nunique(dropna=True)
        usable = nunique[nunique > 0]
        assert len(usable) > 50, f"{col}: too few defined blocks to judge"
        assert usable.mean() > 1.5, (
            f"'{col}' is constant within its block -- the live-price "
            f"recomputation did not happen (mean distinct values {usable.mean():.2f})"
        )

    # Stronger than naming a handful of columns: EXACTLY the 25 recomputed
    # columns may vary inside a block, and every other feature column must be
    # frozen. This is the invariant the deleted `_15m_frozen` twins used to
    # demonstrate, asserted directly instead.
    aux_and_labels = set(ds.AUX_COLUMNS) | set(ds.KEPT_LABEL_COLUMNS)
    carried = [c for c in sub.columns
               if c not in RECOMPUTED and c not in aux_and_labels]
    assert len(carried) > 40, "expected many carried-forward feature columns"
    for col in carried:
        nunique = grouped[col].nunique(dropna=True)
        assert (nunique <= 1).all(), f"'{col}' moved inside a block but is carried forward"


def test_gates_are_frozen_by_decision(built):
    """Recorded decision: gates keep their 15-minute value (see module doc)."""
    full = built[built["block_id"].map(built["block_id"].value_counts()) == 15]
    sub = full.iloc[-400 * 15:]
    for col in ("gate_cell_d", "gate_stack", "regime_daily", "regime_daily_active"):
        nunique = sub.groupby("block_id")[col].nunique(dropna=True)
        assert (nunique <= 1).all(), f"{col} must not be recomputed at the minute price"


def test_cross_timeframe_columns_are_rebuilt_not_carried(built):
    """bb_pos_spread / tf_stretch must agree with the RECOMPUTED positions."""
    sub = built.dropna(subset=["bb_pct_b", "bb_position_weekly"]).iloc[:5000]
    for tf in htf.SPREAD_TIMEFRAMES:
        want = sub["bb_pct_b"] - sub[f"bb_position_{tf}"]
        np.testing.assert_allclose(sub[f"bb_pos_spread_15m_{tf}"], want, rtol=0, atol=0)
    cols = ["bb_pct_b"] + [f"bb_position_{tf}" for tf in htf.HIGHER_TIMEFRAMES]
    stretch = (sub[cols] - 0.5).abs()
    ok = stretch.notna().all(axis=1)
    np.testing.assert_array_equal(
        sub.loc[ok, "tf_stretch_count"].to_numpy(),
        (stretch.loc[ok] > 0.5).sum(axis=1).to_numpy().astype("float64"),
    )
    np.testing.assert_allclose(
        sub.loc[ok, "tf_stretch_max"].to_numpy(), stretch.loc[ok].max(axis=1).to_numpy()
    )


def test_band_position_is_not_clipped(built):
    pos = built["bb_pct_b"].dropna()
    assert (pos > 1.0).any() and (pos < 0.0).any()


# ------------------------------------------------------ 3. ANCHOR CONSISTENCY


def test_anchor_used_in_recomputation_is_the_label_anchor(minutes, built):
    feats, frames = _features(minutes)
    bars = _minute_bars(minutes)
    labels = build_labels(bars, feats["atr_14"])
    np.testing.assert_array_equal(
        built["anchor_price"].to_numpy(), labels["anchor_price"].to_numpy()
    )
    # And the recomputation really used it: invert bb_pct_b back to a price.
    levels = ds.frozen_levels(built.index, feats, frames)
    width = levels["bb_upper"] - levels["bb_lower"]
    implied = built["bb_pct_b"] * width + levels["bb_lower"]
    ok = implied.notna() & built["anchor_price"].notna()
    assert ok.sum() > 10_000
    np.testing.assert_allclose(
        implied[ok], built.loc[ok, "anchor_price"], rtol=1e-12, atol=1e-12
    )


# ------------------------------------------------------- 4. FROZEN LEVELS


@pytest.mark.parametrize("level,freq", [("bb_upper", "15min"), ("bb_lower", "15min"),
                                        ("ema_50", "15min"), ("don_high_60", "15min")])
def test_15min_levels_come_from_the_last_closed_bar(minutes, built, level, freq):
    feats, frames = _features(minutes)
    levels = ds.frozen_levels(built.index, feats, frames)
    closes = feats.index + pd.Timedelta(freq)
    pos = np.searchsorted(closes.to_numpy(), built.index.to_numpy(), side="right") - 1
    want = np.where(pos >= 0, feats[level].to_numpy()[np.clip(pos, 0, None)], np.nan)
    np.testing.assert_array_equal(levels[level].to_numpy(), want)

    covering = np.searchsorted(feats.index.to_numpy(), built.index.to_numpy(),
                               side="right") - 1
    assert (covering != pos).mean() > 0.9, "closed and covering bar barely differ"


@pytest.mark.parametrize("tf", htf.HIGHER_TIMEFRAMES)
def test_higher_tf_bands_come_from_that_timeframes_last_closed_bar(minutes, built, tf):
    """DIRECT projection: the last `tf` bar closed at or before the event.

    Not routed through the 15-minute frame -- a 30-minute bar closing at 09:30
    is visible at 09:31, not 09:45. The mixed vintage this creates against the
    carried-forward `rsi_14_30min` is an accepted trade-off; see
    `dataset.frozen_levels`.
    """
    feats, frames = _features(minutes)
    levels = ds.frozen_levels(built.index, feats, frames)
    closes = (session_close_utc(frames[tf].index, tf) if tf in SESSION_TFS
              else frames[tf].index + pd.Timedelta(tf))
    pos = np.searchsorted(closes.to_numpy(), built.index.to_numpy(), side="right") - 1
    for name in ("bb_upper", "bb_lower"):
        src = frames[tf][name].to_numpy()
        want = np.where(pos >= 0, src[np.clip(pos, 0, None)], np.nan)
        np.testing.assert_array_equal(levels[f"{name}_{tf}"].to_numpy(), want)


@pytest.mark.parametrize("tf", htf.HIGHER_TIMEFRAMES)
def test_a_bar_is_invisible_before_its_close_and_visible_after(minutes, built, tf):
    """The causality rule stated as a before/after pair around every close.

    For each higher-TF bar close C: the last event BEFORE C must still carry the
    previous bar's band, and the first event AT OR AFTER C must carry the bar
    that just closed. This is the check the freshness change is allowed to move
    -- and the one it must never break.

    For the intraday timeframes those neighbours are literally C-1min and
    C+1min, which the test counts separately: that is the "invisible at 09:29,
    visible at 09:31" claim. A weekly bar closes at Friday 17:00 NY, inside the
    weekend gap, so its neighbours are Friday's last minute and Sunday's first.
    """
    feats, frames = _features(minutes)
    levels = ds.frozen_levels(built.index, feats, frames)
    closes = (session_close_utc(frames[tf].index, tf) if tf in SESSION_TFS
              else frames[tf].index + pd.Timedelta(tf))
    band = frames[tf]["bb_upper"].to_numpy()
    got = levels[f"bb_upper_{tf}"]
    idx = got.index

    checked = adjacent = 0
    for i in range(1, len(closes) - 1):
        if not np.isfinite(band[i]) or not np.isfinite(band[i - 1]):
            continue
        if band[i] == band[i - 1]:
            continue
        after_pos = int(idx.searchsorted(closes[i], side="left"))
        if after_pos == 0 or after_pos >= len(idx):
            continue
        assert got.iloc[after_pos - 1] == band[i - 1], (
            f"{tf} bar closing {closes[i]} leaked BEFORE its close"
        )
        assert got.iloc[after_pos] == band[i], (
            f"{tf} bar closing {closes[i]} was still invisible AT its close"
        )
        checked += 1
        one = pd.Timedelta("1min")
        if idx[after_pos] - closes[i] < one and closes[i] - idx[after_pos - 1] <= one:
            adjacent += 1
    assert checked > 10, f"{tf}: too few closes exercised ({checked})"
    if tf != "weekly":
        assert adjacent > 20, (
            f"{tf}: the strict one-minute-either-side case was never exercised"
        )


def test_thirty_minute_band_is_fresh_within_the_quarter_hour(minutes, built):
    """The concrete case in the change request: 09:30 close, visible at 09:31.

    Under the previous two-hop routing the same event would still have been
    carrying the 09:00 bar until 09:45, so this asserts the freshness gain
    rather than just the causality.
    """
    feats, frames = _features(minutes)
    levels = ds.frozen_levels(built.index, feats, frames)
    thirty = frames["30min"]
    closes = thirty.index + pd.Timedelta("30min")
    # A close at :30 lands strictly inside the 15-minute block [:30, :45).
    mid_block = closes[(closes.minute == 30) & (closes.to_series().isin(built.index))]
    assert len(mid_block) > 100
    sample = mid_block[50:150]
    fresh = 0
    for close_at in sample:
        i = int(thirty.index.get_indexer([close_at - pd.Timedelta("30min")])[0])
        at = close_at + pd.Timedelta("1min")
        if at not in levels.index or not np.isfinite(thirty["bb_upper"].iloc[i]):
            continue
        assert levels.loc[at, "bb_upper_30min"] == thirty["bb_upper"].iloc[i]
        if i > 0 and thirty["bb_upper"].iloc[i] != thirty["bb_upper"].iloc[i - 1]:
            fresh += 1
    assert fresh > 20, "the freshness gain was never actually exercised"


def test_bands_and_close_come_from_the_same_higher_tf_bar(minutes, built):
    """Cross-check: bands and price from one bar reproduce that bar's position.

    Replaces the old comparison against the deleted `_15m_frozen` twin. Feeding
    the timeframe's OWN close through its OWN projected bands must reproduce
    that timeframe's `bb_pct_b`, projected the same way -- which fails if the
    bands and the close ever come from different bars.
    """
    from fxalgo.strategies.multi_timeframe import align_completed_series

    feats, frames = _features(minutes)
    levels = ds.frozen_levels(built.index, feats, frames)
    for tf in htf.HIGHER_TIMEFRAMES:
        close = align_completed_series(frames[tf]["mid_close"], built.index, tf)
        want = align_completed_series(frames[tf]["bb_pct_b"], built.index, tf)
        width = levels[f"bb_upper_{tf}"] - levels[f"bb_lower_{tf}"]
        implied = (close - levels[f"bb_lower_{tf}"]) / width.where(width != 0, np.nan)
        ok = implied.notna() & want.notna()
        assert ok.sum() > 1000, tf
        np.testing.assert_allclose(implied[ok], want[ok], rtol=1e-9, atol=1e-12)


# ----------------------------------------------------------- 5. SAMPLING


def test_sample_offsets_respect_the_spacing_and_cover_the_range():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(3000):
        off = ds.sample_offsets(rng, 15)
        assert len(off) == 3
        assert (np.diff(off) >= ds.MIN_SPACING_MINUTES).all()
        assert off.min() >= 0 and off.max() <= 14
        seen.add(tuple(off))
    # 3 positions from 15 with a minimum gap of 3 admits C(11,3) = 165 sets.
    assert len(seen) == 165, f"expected the full support, saw {len(seen)}"


@pytest.mark.parametrize("n,expected", [(15, 3), (11, 3), (7, 3), (6, 2), (4, 2),
                                        (3, 1), (1, 1)])
def test_short_blocks_take_what_the_spacing_allows(n, expected):
    rng = np.random.default_rng(1)
    for _ in range(200):
        off = ds.sample_offsets(rng, n)
        assert len(off) == expected
        assert off.max() < n
        if len(off) > 1:
            assert (np.diff(off) >= ds.MIN_SPACING_MINUTES).all()


def test_sampling_is_spaced_in_time_and_varies_across_blocks(built):
    sel = built[built["sampled_flag"]]
    counts = sel.groupby("block_id").size()
    assert counts.max() <= 3

    gaps = sel.groupby("block_id").apply(
        lambda g: g.index.to_series().diff().dropna().min(), include_groups=False
    ).dropna()
    assert (gaps >= pd.Timedelta(minutes=ds.MIN_SPACING_MINUTES)).all(), (
        "two selected events in one block are closer than the minimum spacing"
    )

    offsets = sel.groupby("block_id")["event_offset"].apply(tuple)
    assert offsets.nunique() > 50, "offsets look fixed rather than drawn per block"


def test_sampling_is_reproducible_under_the_seed(built):
    blocks = built["block_id"].to_numpy()
    a = ds.sample_blocks(blocks)
    b = ds.sample_blocks(blocks)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, built["sampled_flag"].to_numpy())
    c = ds.sample_blocks(blocks, seed=ds.SAMPLE_SEED + 1)
    assert not np.array_equal(a, c), "a different seed must give a different sample"


def test_every_block_is_represented(built):
    assert built.groupby("block_id")["sampled_flag"].any().all()
