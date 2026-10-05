"""Tests for the distributional-moments feature family.

The load-bearing ones: strict causality (corruption test over EVERY new
column), scale invariance (what makes a pooled six-pair model possible), and
that the percentile rank is a TRAILING rank rather than a full-sample one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import moments
from fxalgo.features.moments import add_features, feature_columns, gap_return_mask

NEW_COLS = feature_columns()


def _frame(mids, freq="15min", start="2024-01-01", index=None):
    if index is None:
        index = pd.date_range(start, periods=len(mids), freq=freq, tz="UTC")
    df = pd.DataFrame({"mid_close": np.asarray(mids, dtype="float64")}, index=index)
    df.index.name = "timestamp"
    return df


def _random_frame(n=1200, seed=0, base=1.10, scale=0.0004):
    rng = np.random.default_rng(seed)
    mids = base * np.exp(np.cumsum(rng.normal(0, scale, n)))
    return _frame(mids)


# small percentile window so the rank columns are exercised in tests
SMALL_PCT = 200


def _add(df, **kw):
    kw.setdefault("percentile_window", SMALL_PCT)
    return add_features(df, **kw)


# ============================================================
# 1. Column set, warmup, and the estimator definitions.
# ============================================================


def test_adds_exactly_the_documented_columns():
    out = _add(_random_frame())
    added = [c for c in out.columns if c != "mid_close"]
    assert added == NEW_COLS
    # 7 per window x 2 windows + 2 divergences + 2 percentile ranks
    assert len(NEW_COLS) == 18


def test_warmup_is_nan_not_partial_window():
    out = _add(_random_frame(n=600))
    for w in (96, 24):
        for stem in ("ret_skew", "ret_kurt", "ret_bowley_skew", "ret_moors_kurt",
                     "ret_semidev_down", "ret_semidev_up", "ret_semidev_ratio"):
            col = out[f"{stem}_{w}"]
            assert col.iloc[:w].isna().all(), f"{stem}_{w} leaked a partial window"
            assert col.iloc[w:].notna().any(), f"{stem}_{w} never becomes valid"


def test_skew_and_kurt_match_the_documented_estimators():
    """pandas rolling.skew / rolling.kurt are the bias-corrected Fisher-Pearson
    G1 and the bias-corrected EXCESS kurtosis G2 (normal -> 0). Verified here
    against the textbook formulas so the docstring claim is reproducible."""
    rng = np.random.default_rng(3)
    n, w = 400, 96
    mids = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0005, n)))
    out = _add(_frame(mids))
    r = np.log(pd.Series(mids)).diff()

    def g1(x):
        k = len(x)
        m = x.mean()
        s = x.std(ddof=1)
        return k / ((k - 1) * (k - 2)) * (((x - m) / s) ** 3).sum()

    def g2(x):
        k = len(x)
        m = x.mean()
        s = x.std(ddof=1)
        num = k * (k + 1) / ((k - 1) * (k - 2) * (k - 3))
        return num * (((x - m) / s) ** 4).sum() - 3 * (k - 1) ** 2 / ((k - 2) * (k - 3))

    for t in (150, 250, 399):
        win = r.iloc[t - w + 1 : t + 1].dropna()
        assert out["ret_skew_96"].iloc[t] == pytest.approx(g1(win), rel=1e-9)
        assert out["ret_kurt_96"].iloc[t] == pytest.approx(g2(win), rel=1e-9)


def test_excess_kurtosis_of_gaussian_returns_is_near_zero():
    """Sanity on the sign convention: normal -> 0, not 3."""
    rng = np.random.default_rng(11)
    mids = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0005, 20_000)))
    out = _add(_frame(mids))
    assert abs(out["ret_kurt_96"].mean()) < 0.25
    assert abs(out["ret_skew_96"].mean()) < 0.10


def test_bowley_and_moors_have_the_documented_formulas():
    rng = np.random.default_rng(5)
    n, w = 300, 96
    mids = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.0006, n)))
    out = _add(_frame(mids))
    r = np.log(pd.Series(mids)).diff()

    t = 299
    win = r.iloc[t - w + 1 : t + 1].dropna()
    e = {q: np.quantile(win, q) for q in moments.OCTILES}
    iqr = e[0.75] - e[0.25]
    bowley = (e[0.75] + e[0.25] - 2 * e[0.5]) / iqr
    moors = ((e[0.875] - e[0.625]) + (e[0.375] - e[0.125])) / iqr
    assert out["ret_bowley_skew_96"].iloc[t] == pytest.approx(bowley, rel=1e-9)
    assert out["ret_moors_kurt_96"].iloc[t] == pytest.approx(moors, rel=1e-9)


def test_bowley_is_bounded_and_zero_for_symmetric_input():
    """A symmetric alternating return series has zero quartile skew."""
    n = 400
    steps = np.where(np.arange(n) % 2 == 0, 0.001, -0.001)
    mids = 1.10 * np.exp(np.cumsum(steps))
    out = _add(_frame(mids))
    b = out["ret_bowley_skew_96"].dropna()
    assert (b.abs() <= 1.0 + 1e-12).all()
    assert b.abs().max() < 1e-9


def test_flat_window_guards_to_nan_never_inf():
    """A constant price series has a zero IQR -- the quantile estimators must
    return NaN, and nothing anywhere may be infinite."""
    out = _add(_frame(np.full(400, 1.10)))
    for col in NEW_COLS:
        vals = out[col].to_numpy()
        assert not np.isinf(vals).any(), f"{col} produced an infinity"
    assert out["ret_bowley_skew_96"].iloc[200:].isna().all()
    assert out["ret_moors_kurt_96"].iloc[200:].isna().all()
    # semidev ratio: 0/0 must also be NaN, not inf
    assert out["ret_semidev_ratio_96"].iloc[200:].isna().all()
    # zero-variance windows leave skew/kurt UNDEFINED -- pandas would hand back
    # a finite -3.0 for kurt, which would read as a real thin-tail signal.
    assert out["ret_skew_96"].iloc[200:].isna().all()
    assert out["ret_kurt_96"].iloc[200:].isna().all()


def test_semideviation_is_in_bps_and_splits_direction():
    """A monotonically rising series has zero downside deviation and positive
    upside deviation; the magnitude is in bps."""
    n = 300
    step = 0.0005  # log-return of 5 bps per bar
    mids = 1.10 * np.exp(np.arange(n) * step)
    out = _add(_frame(mids))
    t = 299
    assert out["ret_semidev_down_96"].iloc[t] == pytest.approx(0.0, abs=1e-9)
    assert out["ret_semidev_up_96"].iloc[t] == pytest.approx(1e4 * step, rel=1e-6)
    assert np.isnan(out["ret_semidev_ratio_96"].iloc[t]) or \
        out["ret_semidev_ratio_96"].iloc[t] == pytest.approx(0.0, abs=1e-9)


# ============================================================
# 2. Scale invariance (decision F7 / pooled cross-pair model).
# ============================================================


@pytest.mark.parametrize("factor", [0.01, 137.5, 1000.0])
def test_every_column_is_invariant_to_price_scaling(factor):
    """USDJPY near 150 and EURUSD near 1.08 must produce identical features
    from identically-shaped return paths."""
    df = _random_frame(n=800, seed=21)
    base = _add(df)
    scaled = _add(_frame(df["mid_close"].to_numpy() * factor, index=df.index))
    for col in NEW_COLS:
        a = base[col].to_numpy()
        b = scaled[col].to_numpy()
        # atol dominates: ratio features (Bowley/Moors) have near-zero
        # numerators, so float noise of ~1e-12 absolute shows up as a large
        # RELATIVE difference while being physically meaningless. A genuine
        # scale dependence would move these by a factor of `factor`.
        np.testing.assert_allclose(
            a, b, rtol=1e-7, atol=1e-9, equal_nan=True,
            err_msg=f"{col} is not scale-invariant at factor {factor}",
        )


def test_no_column_carries_price_units():
    """A 10x price level must not move any feature's magnitude by ~10x."""
    df = _random_frame(n=600, seed=22)
    a = _add(df)
    b = _add(_frame(df["mid_close"].to_numpy() * 10.0, index=df.index))
    for col in NEW_COLS:
        x, y = a[col].dropna(), b[col].dropna()
        if len(x) == 0:
            continue
        scale = np.nanmax(np.abs(x.to_numpy())) or 1.0
        assert np.nanmax(np.abs(x.to_numpy() - y.to_numpy())) < 1e-6 * scale, col


# ============================================================
# 3. Causality: the corruption test, over EVERY new column.
# ============================================================


@pytest.mark.parametrize("mask_gaps", [True, False])
def test_no_lookahead_corrupting_the_future_leaves_the_past_identical(mask_gaps):
    """Overwrite ALL data after T with garbage; every feature value at or
    before T must be bit-identical. Covers all 20 columns."""
    df = _random_frame(n=1500, seed=31)
    base = _add(df, mask_gap_returns=mask_gaps)

    for cut in (300, 700, 1100):
        corrupted = df.copy()
        corrupted.iloc[cut + 1 :, corrupted.columns.get_loc("mid_close")] = 99.0
        got = _add(corrupted, mask_gap_returns=mask_gaps)
        for col in NEW_COLS:
            np.testing.assert_array_equal(
                base[col].to_numpy()[: cut + 1],
                got[col].to_numpy()[: cut + 1],
                err_msg=f"{col} leaked future information at T={cut}",
            )


def test_percentile_rank_is_trailing_not_full_sample():
    """The rank must use only the trailing window ending at t. A full-sample
    rank is the classic subtle lookahead: under it, an early extreme value
    would already know it is extreme relative to the whole history."""
    n = 900
    df = _random_frame(n=n, seed=41)
    out = _add(df)

    col = "ret_skew_96_pctile"
    src = out["ret_skew_96"]
    valid = out[col].dropna()
    assert len(valid) > 50

    # Recompute the trailing rank independently at a few points.
    for t in (SMALL_PCT + 150, SMALL_PCT + 400, n - 1):
        window = src.iloc[t - SMALL_PCT + 1 : t + 1]
        if window.isna().any():
            continue
        expected = (window <= window.iloc[-1]).mean()
        assert out[col].iloc[t] == pytest.approx(expected, abs=1e-9)

    # And prove it differs from a full-sample rank (which would be lookahead).
    full_sample = src.rank(pct=True)
    overlap = out[col].notna() & full_sample.notna()
    assert not np.allclose(
        out[col][overlap].to_numpy(), full_sample[overlap].to_numpy(), atol=1e-6
    ), "percentile equals the full-sample rank -- that would be lookahead"


@pytest.mark.parametrize("div,a,b", [
    ("skew_robust_divergence_96", "ret_skew_96", "ret_bowley_skew_96"),
    ("kurt_robust_divergence_96", "ret_kurt_96", "ret_moors_kurt_96"),
])
def test_divergence_standardizes_both_terms_before_differencing(div, a, b):
    """Both divergences must be z(a) - z(b), NOT a raw a - b. A raw difference
    is swamped by whichever term has the larger dispersion and degenerates into
    a copy of it -- the skew version originally measured r = +0.994 against
    ret_skew_96 for exactly this reason."""
    out = _add(_random_frame(n=1500, seed=61))
    valid = out[[div, a, b]].dropna()
    assert len(valid) > 100

    za = (valid[a] - valid[a].mean()) / valid[a].std(ddof=1)
    zb = (valid[b] - valid[b].mean()) / valid[b].std(ddof=1)
    # It tracks the difference of standardized terms far better than the raw one.
    r_std = np.corrcoef(valid[div], za - zb)[0, 1]
    r_raw = np.corrcoef(valid[div], valid[a] - valid[b])[0, 1]
    assert r_std > r_raw, f"{div} looks like a raw difference"

    # And it must not be a near-copy of either input.
    for col in (a, b):
        r = abs(np.corrcoef(valid[div], valid[col])[0, 1])
        assert r < 0.95, f"{div} is a near-copy of {col} (|r| = {r:.3f})"


def test_percentile_rank_is_bounded_and_warms_up_fully():
    out = _add(_random_frame(n=700, seed=42))
    for col in ("ret_skew_96_pctile", "ret_kurt_96_pctile"):
        v = out[col]
        assert v.iloc[: SMALL_PCT].isna().all()
        d = v.dropna()
        assert len(d) > 0
        assert d.between(0.0, 1.0).all()


# ============================================================
# 4. Gap masking.
# ============================================================


def test_gap_mask_flags_discontinuities_and_the_first_bar():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01 00:00", tz="UTC"),
         pd.Timestamp("2024-01-01 00:15", tz="UTC"),
         pd.Timestamp("2024-01-01 00:30", tz="UTC"),
         pd.Timestamp("2024-01-03 17:00", tz="UTC"),   # weekend-sized hole
         pd.Timestamp("2024-01-03 17:15", tz="UTC")]
    )
    mask = gap_return_mask(idx).to_numpy()
    np.testing.assert_array_equal(mask, [True, False, False, True, False])


def test_gap_spanning_return_is_excluded_from_the_window():
    """A huge jump across a gap must not reach the moments; the same jump
    WITHOUT a gap must."""
    n = 200
    mids = np.full(n, 1.10)
    mids[120:] = 1.30  # one enormous return at bar 120

    contiguous = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    gapped = contiguous.to_list()
    for i in range(120, n):  # push everything from the jump onward past a weekend
        gapped[i] = gapped[i] + pd.Timedelta(days=2, minutes=30)
    gapped = pd.DatetimeIndex(gapped)

    masked = _add(_frame(mids, index=gapped), mask_gap_returns=True)
    unmasked = _add(_frame(mids, index=gapped), mask_gap_returns=False)

    t = n - 1
    # With the gap return dropped the window is a flat series -> undefined.
    assert np.isnan(masked["ret_kurt_96"].iloc[t])
    # Keeping it leaves a single massive outlier -> large positive excess kurtosis.
    assert unmasked["ret_kurt_96"].iloc[t] > 50


def test_mask_flag_is_off_switchable_and_changes_nothing_without_gaps():
    """On a contiguous index the flag must be a no-op."""
    df = _random_frame(n=500, seed=51)
    on = _add(df, mask_gap_returns=True)
    off = _add(df, mask_gap_returns=False)
    for col in NEW_COLS:
        np.testing.assert_allclose(
            on[col].to_numpy(), off[col].to_numpy(),
            rtol=1e-12, atol=1e-15, equal_nan=True, err_msg=col,
        )


def test_requires_price_column_and_datetime_index():
    with pytest.raises(KeyError, match="mid_close"):
        _add(pd.DataFrame({"x": [1.0, 2.0]},
                          index=pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC")))
    with pytest.raises(TypeError):
        _add(pd.DataFrame({"mid_close": [1.0, 2.0]}))
