"""Tests for the purged validation carve and the subset definitions.

The validation carve is where a leak would do the most damage: early stopping
SELECTS a model by validation score, so a leaky validation block does not just
mis-measure, it hands the test set an overfit model that looks healthy.
`test_a_random_validation_split_would_leak` is the control -- it shows the
failure the carve prevents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.resample import session_keys
from fxalgo.models import fit as mf
from fxalgo.models import subsets as ss
from fxalgo.training.split import LABEL_HORIZON, window_reaches_test


def _index(start="2023-01-01", end="2024-01-01") -> pd.DatetimeIndex:
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    keys = session_keys(idx, "daily")
    return idx[~np.isin(keys.weekday, (5, 6))]


@pytest.fixture(scope="module")
def index() -> pd.DatetimeIndex:
    return _index()


@pytest.fixture(scope="module")
def carve(index):
    train = np.ones(len(index), dtype=bool)
    train[-5000:] = False               # a slab of "test" at the very end
    cutoff = index[int(len(index) * 0.85)]
    return index, train, cutoff, mf.purged_validation(index, train, cutoff)


# ------------------------------------------------------- the carve itself


def test_validation_is_a_contiguous_block_at_the_end(carve):
    index, _, cutoff, m = carve
    assert m.val.any() and m.fit.any()
    assert index[m.val].min() >= cutoff
    assert index[m.fit].max() < cutoff
    pos = np.flatnonzero(m.val)
    assert (np.diff(pos) == 1).all(), "validation must be one contiguous block"


def test_no_fit_label_window_reaches_the_validation_block(carve):
    _, _, _, m = carve
    reaches = window_reaches_test(m.val, LABEL_HORIZON)
    assert int((m.fit & reaches).sum()) == 0


def test_a_random_validation_split_would_leak(index):
    """The control: this is the failure the contiguous carve prevents."""
    rng = np.random.default_rng(0)
    val = rng.random(len(index)) < 0.15
    fit = ~val
    reaches = window_reaches_test(val, LABEL_HORIZON)
    leaked = int((fit & reaches).sum())
    assert leaked > 0.5 * len(index), (
        "a random validation split should leak on most fit rows; if it does "
        "not, the purge check has stopped meaning anything"
    )


def test_the_purge_and_embargo_both_actually_bite(carve):
    _, _, _, m = carve
    assert m.purged > 0
    assert m.embargoed > 0
    assert m.embargoed <= mf.EMBARGO_BARS


def test_the_gap_between_fit_and_validation_exceeds_the_embargo(carve):
    _, _, _, m = carve
    last_fit = int(np.flatnonzero(m.fit).max())
    first_val = int(np.flatnonzero(m.val).min())
    assert first_val - last_fit > mf.EMBARGO_BARS


def test_test_side_rows_are_never_used_for_fit_or_validation(carve):
    _, train, _, m = carve
    assert not (m.fit & ~train).any()
    assert not (m.val & ~train).any()


def test_fit_and_validation_are_disjoint(carve):
    _, _, _, m = carve
    assert not (m.fit & m.val).any()


# ------------------------------------------------------------ weights


def test_class_weights_equalise_the_three_classes():
    y = np.array([-1] * 600 + [0] * 300 + [1] * 100)
    w = mf.class_weights(y)
    assert w.mean() == pytest.approx(1.0)
    totals = [w[y == c].sum() for c in (-1, 0, 1)]
    assert totals[0] == pytest.approx(totals[1]) == pytest.approx(totals[2])
    assert w[y == 1][0] > w[y == -1][0], "the rare class must weigh more"


def test_class_weights_are_flat_when_classes_are_balanced():
    y = np.array([-1, 0, 1] * 100)
    np.testing.assert_allclose(mf.class_weights(y), 1.0)


# ------------------------------------------------------------ subsets


@pytest.fixture(scope="module")
def features() -> list[str]:
    tf = ["30min", "1h", "2h", "4h", "daily", "weekly"]
    cols = ["bb_pct_b", "rsi_14", "rsi_2", "atr_bps_15m", "regime_daily",
            "regime_daily_active", "gate_cell_d", "gate_stack",
            "spread_ratio_15", "spread_ratio_60", "spread_zscore_60",
            "spread_pctile_trailing", "spread_pctile_trailing_rolling",
            "tf_stretch_count", "tf_stretch_max"]
    cols += [f"bb_position_{t}" for t in tf]
    cols += [f"bb_width_atr_{t}" for t in tf]
    cols += [f"rsi_14_{t}" for t in tf]
    cols += [f"bb_pos_spread_15m_{t}" for t in tf[1:]]
    for stem in ("bars_above_band", "bars_below_band", "bars_since_band_cross",
                 "max_stretch_current_excursion"):
        cols += [f"{stem}_{t}" for t in tf]
    for stem in ("frac_above_band", "frac_below_band"):
        cols += [f"{stem}_{t}_20" for t in tf]
    cols += list(ss.TIME_COLUMNS)
    return cols


def test_the_four_subsets_are_the_four_committed_ones(features):
    built = ss.build(features)
    assert list(built) == ["A_all", "B_strategy_core", "C_mtf_bollinger", "D_no_time"]


def test_subset_a_is_everything_and_d_drops_exactly_the_clock(features):
    built = ss.build(features)
    assert built["A_all"] == features
    assert len(built["D_no_time"]) == len(features) - len(ss.TIME_COLUMNS)
    assert not (set(built["D_no_time"]) & set(ss.TIME_COLUMNS))


def test_subset_c_covers_the_band_structure_and_the_excursion_history(features):
    c = ss.build(features)["C_mtf_bollinger"]
    assert len([x for x in c if x == "bb_pct_b" or x.startswith("bb_position_")]) == 7
    assert len([x for x in c if x.startswith("bb_pos_spread_15m_")]) == 5
    assert len([x for x in c if x.startswith(ss._MTF_STEMS)]) == 36
    assert "tf_stretch_count" in c and "tf_stretch_max" in c
    assert len(set(c)) == len(c), "no duplicates"


def test_subset_c_carries_no_clock_column(features):
    assert not (set(ss.build(features)["C_mtf_bollinger"]) & set(ss.TIME_COLUMNS))


def test_strategy_core_is_a_subset_of_the_frame_and_rejects_a_missing_column(features):
    built = ss.build(features)
    assert set(built["B_strategy_core"]) <= set(features)
    with pytest.raises(KeyError, match="STRATEGY_CORE"):
        ss.build([c for c in features if c != "gate_cell_d"])
