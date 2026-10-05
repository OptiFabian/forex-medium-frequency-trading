"""Tests for the higher-timeframe feature family.

The load-bearing ones, in order of how badly they would hurt if absent:

1. CAUSALITY (`test_no_lookahead_*`). The whole build is a projection of
   coarse bars onto a fine index; an off-by-one in the daily or weekly lag
   would present as an excellent model. The corruption test runs the FULL
   chain -- 1-minute bars -> resample (clock and session) -> per-timeframe
   feature pipeline -> alignment -> new columns -- twice, garbaging everything
   after a cut, and demands bit-identical output at and before it.
2. ALIGNMENT (`test_aligned_value_comes_from_a_closed_bar`). States the
   invariant directly and exhaustively, including across both US DST
   transitions and exactly on bar boundaries.
3. SCALE INVARIANCE (decision F7), at 0.01x / 137.5x / 1000x as the moments
   module does -- with `gate_stack` singled out, since the absolute
   commission floor makes it genuinely price-level dependent.
4. WARMUP is NaN, never a partial window.

Everything runs on a synthetic year of 1-minute bars carrying REAL FX weekend
geometry (no minutes between Friday 17:00 NY and Sunday 17:00 NY), because a
continuous-clock synthetic would let a weekly bar aggregate Saturday data that
`session_close_utc` says was already closed on Friday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.backtest.costs import DEFAULT_COMMISSION
from fxalgo.data.resample import (
    resample_merged,
    resample_sessions,
    session_close_utc,
    session_keys,
)
from fxalgo.features import htf
from fxalgo.features.build import apply_pipeline
from fxalgo.strategies.cost_filter import passes_cost_filter

CLOCK_TFS = ("30min", "1h")
SESSION_TFS = ("2h", "4h", "daily", "weekly")
NEW_COLS = htf.feature_columns()
PRICE_COLS = [
    "open_bid", "high_bid", "low_bid", "close_bid",
    "open_ask", "high_ask", "low_ask", "close_ask",
]


# --------------------------------------------------------------- fixtures


def _minute_index(start: str, end: str) -> pd.DatetimeIndex:
    """1-minute UTC index with the FX weekend cut out.

    The sessions keyed Saturday and Sunday are exactly [Fri 17:00 NY, Sun
    17:00 NY) -- the closed span -- so dropping those two keys reproduces the
    real gap geometry, DST included, without hard-coding any UTC offset.
    """
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    keys = session_keys(idx, "daily")
    return idx[~np.isin(keys.weekday, (5, 6))]


def _synthetic_minutes(start="2024-01-01", end="2025-01-01", seed=7) -> pd.DataFrame:
    """A year of 1-minute merged bid/ask bars on a realistic session calendar."""
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


def _frames(minutes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = apply_pipeline(resample_merged(minutes, "15min"))
    frames = {tf: apply_pipeline(resample_merged(minutes, tf)) for tf in CLOCK_TFS}
    frames.update({tf: apply_pipeline(resample_sessions(minutes, tf)) for tf in SESSION_TFS})
    return base, frames


def _build(minutes: pd.DataFrame, *, spread_gate_max: float | None = None, **kw) -> pd.DataFrame:
    base, frames = _frames(minutes)
    if spread_gate_max is None:
        spread_gate_max = float(np.nanmedian(base["spread_close"].to_numpy()))
    return htf.add_features(base, frames, spread_gate_max=spread_gate_max, **kw)


@pytest.fixture(scope="module")
def minutes() -> pd.DataFrame:
    return _synthetic_minutes()


@pytest.fixture(scope="module")
def built(minutes) -> pd.DataFrame:
    return _build(minutes)


# ------------------------------------------------------- 0. shape & definitions


def test_adds_exactly_the_documented_columns(minutes, built):
    base, _ = _frames(minutes)
    added = [c for c in built.columns if c not in base.columns]
    assert added == NEW_COLS
    # 6 timeframes x 4 + 5 spreads + 2 stretch + 2 regime + 2 gates + 6 don_pos + 1 atr
    assert len(NEW_COLS) == 42


def test_bb_position_is_the_pipeline_bb_pct_b(built):
    """The 15-min band position is not recomputed differently from the pipeline's."""
    ours = htf.bb_position(built).to_numpy()
    theirs = built[htf.BB_POSITION_15M].to_numpy()
    assert np.array_equal(ours, theirs, equal_nan=True)


def test_bb_position_is_not_clipped(built):
    """Values outside [0, 1] must survive -- the overshoot IS the signal."""
    pos = built["bb_position_1h"].dropna().to_numpy()
    assert (pos > 1.0).any() and (pos < 0.0).any()


def test_commission_matches_scalar_model():
    prices = np.array([0.011, 1.0, 1.10, 137.5, 1100.0])
    vec = htf.commission_per_leg(prices, 100_000.0, DEFAULT_COMMISSION)
    scalar = [DEFAULT_COMMISSION.commission_for(float(p), 100_000.0) for p in prices]
    np.testing.assert_allclose(vec, scalar, rtol=0, atol=0)


def test_cost_filter_matches_scalar_implementation():
    rng = np.random.default_rng(3)
    price = 1.10 * np.exp(rng.normal(0, 0.05, 500))
    middle = price * (1.0 + rng.normal(0, 0.001, 500))
    spread = np.abs(rng.normal(0.00010, 0.00005, 500))
    vec = htf.passes_cost_filter_vec(
        price, middle, spread,
        notional=htf.NOTIONAL, commission=DEFAULT_COMMISSION, cost_multiple=htf.COST_MULTIPLE,
    )
    scalar = [
        passes_cost_filter(
            float(p), float(m), float(s),
            notional=htf.NOTIONAL, commission=DEFAULT_COMMISSION, cost_multiple=htf.COST_MULTIPLE,
        )
        for p, m, s in zip(price, middle, spread, strict=True)
    ]
    assert vec.tolist() == scalar


# ------------------------------------------------------------- 1. CAUSALITY


def _corrupt_minutes_after(minutes: pd.DataFrame, cut: pd.Timestamp, value=9.99) -> pd.DataFrame:
    """Garbage every 1-minute bar at or after `cut` (a 15-min bar boundary)."""
    out = minutes.copy()
    mask = out.index >= cut
    assert mask.any() and not mask.all()
    for col in PRICE_COLS:
        # Keep ask above bid so spread_close stays non-negative and the garbage
        # is a plausible-looking frame rather than a degenerate one.
        out.loc[mask, col] = value + (0.5 if col.endswith("ask") else 0.0)
    return out


@pytest.mark.parametrize("frac", [0.55, 0.8])
def test_no_lookahead_in_every_new_column(minutes, built, frac):
    """Corrupt the future, demand the past is bit-identical.

    The cut is a 15-min bar boundary T: every 1-minute bar from T onward is
    garbage, so the 15-min bars labeled < T (each of which closes at or before
    T) and every higher-timeframe bar that had already closed are untouched.
    Any column at a row < T that moves is reading the future.

    `spread_gate_max` is held FIXED across the two builds because it is a
    constant of the `gate_stack` config (the strategy takes it as a constructor
    argument), not a per-bar computation. What happens if it is re-derived from
    the data instead is the subject of the next test.
    """
    index = built.index
    cut = index[int(len(index) * frac)]
    base, _ = _frames(minutes)
    med = float(np.nanmedian(base["spread_close"].to_numpy()))
    corrupted = _build(_corrupt_minutes_after(minutes, cut), spread_gate_max=med)

    head_a = built.loc[built.index < cut, NEW_COLS]
    head_b = corrupted.loc[corrupted.index < cut, NEW_COLS]
    assert len(head_a) and head_a.index.equals(head_b.index)

    nan_moved = (head_a.isna() != head_b.isna()).any()
    assert not nan_moved.any(), (
        f"LOOKAHEAD: NaN pattern changed for {nan_moved[nan_moved].index.tolist()}"
    )
    for col in NEW_COLS:
        a, b = head_a[col].to_numpy(), head_b[col].to_numpy()
        if a.dtype == bool:
            assert np.array_equal(a, b), f"LOOKAHEAD in boolean column '{col}'"
            continue
        mask = ~pd.isna(a)
        assert np.array_equal(a[mask], b[mask]), (
            f"LOOKAHEAD in '{col}': values before {cut} changed when later "
            f"1-minute bars were corrupted"
        )


def test_spread_gate_threshold_is_a_config_constant_not_a_feature(minutes, built):
    """The gate_stack threshold is full-sample-derived: freezing it is REQUIRED.

    The research set the spread ceiling to the median `spread_close` over the
    whole frame. Recomputing that median from data the
    bar could not have seen moves the gate at EARLIER bars -- which is why
    `add_features` takes the threshold as an argument instead of deriving it,
    and why a walk-forward must re-derive it on the training fold alone.

    This test asserts the hazard is real rather than hypothetical.
    """
    index = built.index
    cut = index[int(len(index) * 0.55)]
    refit = _build(_corrupt_minutes_after(minutes, cut))  # threshold re-derived
    a = built.loc[built.index < cut, "gate_stack"].to_numpy()
    b = refit.loc[refit.index < cut, "gate_stack"].to_numpy()
    assert not np.array_equal(a, b), (
        "expected a full-sample threshold to contaminate the past; if this "
        "passes, the median has become insensitive and the warning is stale"
    )


def test_corruption_actually_bites(minutes, built):
    """Guard the guard: the corruption must change the columns AFTER the cut.

    Without this, a build that silently returned all-NaN would pass the
    lookahead test.
    """
    index = built.index
    cut = index[int(len(index) * 0.55)]
    corrupted = _build(_corrupt_minutes_after(minutes, cut))
    tail_a = built.loc[built.index > cut, "bb_position_daily"].dropna()
    tail_b = corrupted.loc[corrupted.index > cut, "bb_position_daily"].dropna()
    assert len(tail_a) > 100
    assert not np.array_equal(tail_a.to_numpy(), tail_b.reindex(tail_a.index).to_numpy())


# ------------------------------------------------------------- 2. ALIGNMENT


def _closes(frames: dict[str, pd.DataFrame], tf: str) -> pd.DatetimeIndex:
    if tf in ("daily", "weekly", "2h", "4h"):
        return session_close_utc(frames[tf].index, tf)
    return frames[tf].index + pd.Timedelta(tf)


@pytest.mark.parametrize("tf", htf.HIGHER_TIMEFRAMES)
def test_aligned_value_comes_from_a_closed_bar(minutes, built, tf):
    """For EVERY 15-min bar: the projected value is the last CLOSED bar's.

    Exhaustive rather than sampled -- the sampled version is the special case.
    Also asserts the negative: the value is never the one from the bar that
    merely COVERS the timestamp.
    """
    base, frames = _frames(minutes)
    block = htf.htf_block(frames[tf])
    closes = _closes(frames, tf)
    expected_pos = np.searchsorted(closes.to_numpy(), built.index.to_numpy(), side="right") - 1

    got = built[f"bb_position_{tf}"].to_numpy()
    src = block["bb_position"].to_numpy()
    valid = expected_pos >= 0
    want = np.full(len(built), np.nan)
    want[valid] = src[expected_pos[valid]]
    assert np.array_equal(got, want, equal_nan=True), f"{tf}: not the last closed bar"

    # The bar COVERING each timestamp must not be the one used (except where it
    # coincidentally equals the previous bar's value, which cannot happen for a
    # rolling band position on a random walk).
    covering = np.searchsorted(frames[tf].index.to_numpy(), built.index.to_numpy(), side="right") - 1
    differs = covering != expected_pos
    assert differs.mean() > 0.9, f"{tf}: covering bar and last-closed bar barely differ"


@pytest.mark.parametrize("tf", ["daily", "weekly", "4h"])
def test_alignment_is_exact_on_the_bar_boundary(minutes, built, tf):
    """A bar becomes visible at its close INSTANT, not one bar later or sooner."""
    base, frames = _frames(minutes)
    block = htf.htf_block(frames[tf])
    closes = _closes(frames, tf)
    got = built[f"bb_position_{tf}"]

    # The close instant does not always coincide with an existing 15-min bar:
    # a weekly bar closes at Friday 17:00 NY, when the market shuts, so the
    # first bar that can see it is Sunday's reopen. Compare across the
    # boundary rather than assuming a bar sits exactly on it.
    idx = got.index
    values = block["bb_position"].to_numpy()
    checked = 0
    for i, close_at in enumerate(closes[:-1]):
        if np.isnan(values[i]):
            continue
        pos = int(idx.searchsorted(close_at, side="left"))
        if pos == 0 or pos >= len(idx):
            continue
        assert got.iloc[pos] == values[i], (
            f"{tf} bar {frames[tf].index[i]} not visible at the first 15-min bar "
            f"at or after its close {close_at}"
        )
        if i > 0 and not np.isnan(values[i - 1]):
            assert got.iloc[pos - 1] == values[i - 1], (
                f"{tf} bar {frames[tf].index[i]} leaked BEFORE its close {close_at}"
            )
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("switch", ["2024-03-10", "2024-11-03"])
def test_alignment_survives_dst_transitions(minutes, built, switch):
    """Both US transitions: the daily/weekly grid moves in UTC, the lag does not.

    17:00 NY is 21:00 UTC under EDT and 22:00 UTC under EST, so the daily close
    instants shift by an hour across each switch. The values on the 15-min
    timeline must follow the NY-wall grid, not a fixed UTC offset.
    """
    base, frames = _frames(minutes)
    window = slice(pd.Timestamp(switch, tz="UTC") - pd.Timedelta(days=6),
                   pd.Timestamp(switch, tz="UTC") + pd.Timedelta(days=6))
    for tf in ("daily", "weekly"):
        closes = session_close_utc(frames[tf].index, tf)
        block = htf.htf_block(frames[tf])["bb_position"].to_numpy()
        sub = built.loc[window]
        assert len(sub) > 500
        pos = np.searchsorted(closes.to_numpy(), sub.index.to_numpy(), side="right") - 1
        want = np.where(pos >= 0, block[pos], np.nan)
        np.testing.assert_array_equal(sub[f"bb_position_{tf}"].to_numpy(), want)

    # And the grid really did move: the UTC hour of the daily closes differs
    # on the two sides of the switch.
    d_closes = session_close_utc(frames["daily"].index, "daily")
    around = d_closes[(d_closes > window.start) & (d_closes < window.stop)]
    assert len({int(h) for h in around.hour}) == 2, "DST shift not exercised"


# ------------------------------------------------- 3. SCALE INVARIANCE (F7)


def _scaled(minutes: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = minutes.copy()
    out[PRICE_COLS] = out[PRICE_COLS] * factor
    return out


SCALE_EXEMPT = ("gate_stack",)


@pytest.mark.parametrize("factor", [0.01, 137.5, 1000.0])
def test_every_new_column_is_scale_invariant(minutes, built, factor):
    """Multiplying the whole price series must not move any new column.

    `spread_gate_max` is a price-unit CONSTANT of the config, so it is scaled
    with the data -- otherwise the test would measure the threshold's units
    rather than the column's. `gate_stack` is exempt and handled below.
    """
    base, _ = _frames(minutes)
    med = float(np.nanmedian(base["spread_close"].to_numpy()))
    scaled = _build(_scaled(minutes, factor), spread_gate_max=med * factor)

    for col in NEW_COLS:
        if col in SCALE_EXEMPT:
            continue
        a = built[col].to_numpy()
        b = scaled[col].to_numpy()
        if a.dtype == bool:
            assert np.array_equal(a, b), f"{col} is not scale-invariant at {factor}x"
            continue
        assert not (pd.isna(a) != pd.isna(b)).any(), f"{col} NaN pattern moved at {factor}x"
        # Tolerance absorbs float cancellation only: bb_position divides a
        # ~1e-4 band width out of ~1e3 prices at 1000x, which costs ~1e-9
        # absolute. A genuine scale dependence would move values by `factor`
        # itself -- 8 to 11 orders of magnitude above this bar.
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-8, equal_nan=True,
            err_msg=f"{col} is not scale-invariant at {factor}x",
        )


@pytest.mark.parametrize("factor", [0.01, 137.5, 1000.0])
def test_gate_stack_is_scale_invariant_only_without_the_commission_floor(minutes, built, factor):
    """`gate_stack` fails F7 for exactly one reason -- and this proves which.

    Its cost filter compares a profit target against the round-trip cost,
    which contains an ABSOLUTE 2.00 per-order commission minimum. That floor does not
    move when prices are rescaled, so the gate legitimately depends on the
    price level. Drop the commission model and the invariance returns.
    """
    base, _ = _frames(minutes)
    med = float(np.nanmedian(base["spread_close"].to_numpy()))
    kw = dict(commission=None)
    a = _build(minutes, spread_gate_max=med, **kw)["gate_stack"].to_numpy()
    b = _build(_scaled(minutes, factor), spread_gate_max=med * factor, **kw)["gate_stack"].to_numpy()
    assert np.array_equal(a, b), "gate_stack should be scale-free once commission is removed"


# ------------------------------------------------------------------ 4. WARMUP


def test_warmup_is_nan_not_a_partial_window(built):
    """Every higher-TF column is NaN until its own timeframe has 20 bands."""
    for tf in htf.HIGHER_TIMEFRAMES:
        col = built[f"bb_position_{tf}"]
        first = col.first_valid_index()
        assert first is not None, f"{tf} never becomes valid"
        assert col.loc[:first].iloc[:-1].isna().all(), f"{tf} has a partial-window value"
    # The slowest input governs the summaries.
    slowest = max(
        built[f"bb_position_{tf}"].first_valid_index() for tf in htf.HIGHER_TIMEFRAMES
    )
    assert built["tf_stretch_count"].first_valid_index() >= slowest
    assert built["tf_stretch_max"].first_valid_index() >= slowest


def test_stretch_summaries_agree_with_their_inputs(built):
    """count and max are computed from the same seven positions, on defined rows."""
    cols = [htf.BB_POSITION_15M] + [f"bb_position_{tf}" for tf in htf.HIGHER_TIMEFRAMES]
    stretch = (built[cols] - 0.5).abs()
    ok = stretch.notna().all(axis=1)
    assert ok.sum() > 1000
    np.testing.assert_array_equal(
        built.loc[ok, "tf_stretch_count"].to_numpy(),
        (stretch.loc[ok] > 0.5).sum(axis=1).to_numpy().astype("float64"),
    )
    np.testing.assert_allclose(
        built.loc[ok, "tf_stretch_max"].to_numpy(), stretch.loc[ok].max(axis=1).to_numpy()
    )
    assert built.loc[~ok, "tf_stretch_count"].isna().all()


def test_regime_daily_is_the_daily_bollinger_shadow(minutes, built):
    """Decision F4: the column IS `align_completed_signal(shadow, index, 'daily')`."""
    from fxalgo.strategies.bollinger_reversion import BollingerReversion
    from fxalgo.strategies.multi_timeframe import align_completed_signal

    _, frames = _frames(minutes)
    shadow = BollingerReversion().generate_signals(frames["daily"])
    want = align_completed_signal(shadow, built.index, "daily")
    np.testing.assert_array_equal(built["regime_daily"].to_numpy(), want.to_numpy())
    assert set(np.unique(built["regime_daily"])) <= {-1, 0, 1}
    np.testing.assert_array_equal(
        built["regime_daily_active"].to_numpy(), want.to_numpy() != 0
    )


def test_gates_only_fire_on_a_band_touch(built):
    """Both gates require a touch, so they are a subset of the touch bars."""
    touch = (built["mid_close"] <= built["bb_lower"]) | (built["mid_close"] >= built["bb_upper"])
    for gate in ("gate_cell_d", "gate_stack"):
        assert built[gate].dtype == bool
        assert not (built[gate] & ~touch).any(), f"{gate} fired without a band touch"
        assert built[gate].sum() > 0, f"{gate} never fires on the synthetic year"
    # cell D additionally requires the daily shadow to hold a position.
    assert not (built["gate_cell_d"] & ~built["regime_daily_active"]).any()


def test_don_pos_is_unitless_and_guarded(built):
    from fxalgo.features.donchian import DONCHIAN_WINDOWS

    for w in DONCHIAN_WINDOWS:
        col = built[f"don_pos_{w}"]
        assert col.first_valid_index() is not None
        degenerate = built[f"don_high_{w}"] == built[f"don_low_{w}"]
        assert col[degenerate].isna().all(), f"don_pos_{w} not guarded at zero width"
        assert np.isfinite(col.dropna()).all()


def test_gate_stack_column_reproduces_the_built_column(minutes, built):
    """The extracted gate must be the column, not a lookalike.

    `gate_stack_column` exists so a walk-forward can re-derive the spread
    ceiling per training fold without rewriting the condition. If the two ever
    drift apart, the fold-specific gate stops being the same gate.
    """
    base, _ = _frames(minutes)
    med = float(np.nanmedian(base["spread_close"].to_numpy()))
    again = htf.gate_stack_column(built, spread_gate_max=med)
    np.testing.assert_array_equal(again, built["gate_stack"].to_numpy())


def test_gate_stack_column_is_monotone_in_the_ceiling(minutes, built):
    """A tighter ceiling can only ever admit fewer bars."""
    base, _ = _frames(minutes)
    med = float(np.nanmedian(base["spread_close"].to_numpy()))
    tight = htf.gate_stack_column(built, spread_gate_max=med * 0.5)
    loose = htf.gate_stack_column(built, spread_gate_max=med * 2.0)
    assert tight.sum() < built["gate_stack"].sum() < loose.sum()
    assert not (tight & ~loose).any()
