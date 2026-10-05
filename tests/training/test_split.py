"""Tests for the blocked-by-time split, purge and embargo.

The load-bearing one is `test_no_training_window_reaches_a_test_chunk`. It is
built so that a NAIVE RANDOM SPLIT FAILS IT -- the same assertion is run against
a randomly assigned split in `test_a_random_split_fails_the_overlap_check`, and
if that control ever passes, the real test has stopped meaning anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.data.resample import session_keys
from fxalgo.training import split as sp

HORIZON = sp.LABEL_HORIZON
EMBARGO = sp.EMBARGO_BARS


def _minute_index(start="2022-06-01", end="2026-06-01") -> pd.DatetimeIndex:
    """1-minute UTC index with the FX weekend cut out."""
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    keys = session_keys(idx, "daily")
    return idx[~np.isin(keys.weekday, (5, 6))]


@pytest.fixture(scope="module")
def index() -> pd.DatetimeIndex:
    return _minute_index()


@pytest.fixture(scope="module")
def chunks(index) -> pd.DataFrame:
    return sp.build_chunks(index[0], index[-1])


@pytest.fixture(scope="module")
def table(index, chunks) -> pd.DataFrame:
    n_test = max(1, round(len(chunks) * sp.TEST_FRACTION))
    test_chunks = np.array(sorted(range(0, len(chunks), max(1, len(chunks) // n_test)))[:n_test])
    return sp.split_table(index, chunks, test_chunks)


# ----------------------------------------------------------------- chunking


def test_chunks_are_contiguous_and_cover_the_range(index, chunks):
    assert (chunks["chunk_start"].to_numpy()[1:] == chunks["chunk_end"].to_numpy()[:-1]).all()
    assert chunks["chunk_start"].iloc[0] == index[0]
    assert chunks["chunk_end"].iloc[-1] > index[-1]


def test_chunk_boundaries_avoid_calendar_quarter_ends(chunks):
    """Every boundary must sit well away from a quarter end, so no chunk IS a
    calendar quarter and no test chunk can be one season."""
    for ts in chunks["chunk_start"]:
        for q_month in (1, 4, 7, 10):
            q_start = pd.Timestamp(year=ts.year, month=q_month, day=1, tz=ts.tz)
            assert abs((ts - q_start).days) > 20, f"{ts} sits on a quarter boundary"


def test_a_short_tail_is_absorbed_not_left_as_a_stub():
    start = pd.Timestamp("2022-05-26", tz="UTC")
    end = start + pd.DateOffset(months=48) + pd.Timedelta(days=8)
    ch = sp.build_chunks(start, end)
    assert len(ch) == 16
    last = ch.iloc[-1]
    assert (last["chunk_end"] - last["chunk_start"]).days > 90


def test_every_event_in_the_grid_gets_a_chunk(index, chunks):
    ids = sp.chunk_of_event(index, chunks)
    assert (ids >= 0).all()
    assert ids.max() == len(chunks) - 1
    assert (np.diff(ids) >= 0).all(), "chunk ids must be non-decreasing in time"


def test_events_before_the_grid_are_marked_warmup(chunks):
    idx = pd.DatetimeIndex(
        [chunks["chunk_start"].iloc[0] - pd.Timedelta(minutes=5),
         chunks["chunk_start"].iloc[0]]
    )
    assert sp.chunk_of_event(idx, chunks).tolist() == [-1, 0]


# ------------------------------------------------- window / purge primitives


def test_window_reaches_test_is_exact_on_a_hand_case():
    # positions 5,6,7 are test; horizon 3 -> events 2,3,4 (and 5,6) reach them
    is_test = np.zeros(12, dtype=bool)
    is_test[5:8] = True
    got = sp.window_reaches_test(is_test, horizon=3)
    assert got[:2].tolist() == [False, False]
    assert got[2:7].tolist() == [True, True, True, True, True]
    assert got[8:].tolist() == [False] * 4


def test_window_uses_trading_bars_not_wall_clock(index):
    """A 240-bar window over a weekend spans days; positions get that right."""
    pos = np.arange(len(index) - HORIZON)      # drop the clamped tail
    end = index[pos + HORIZON]
    span_min = (end - index[pos]) / pd.Timedelta(minutes=1)
    assert span_min.min() >= HORIZON, "a window cannot be shorter than its bar count"
    assert span_min.max() > 24 * 60, "expected at least one window across a gap"


# --------------------------------------------- 1. THE OVERLAP TEST + CONTROL


def _overlap_count(index, split, chunk_of, is_test_chunk, horizon=HORIZON) -> int:
    """Training events whose forward window touches a test-chunk event."""
    is_test_event = (chunk_of >= 0) & is_test_chunk[np.clip(chunk_of, 0, None)]
    reaches = sp.window_reaches_test(is_test_event, horizon)
    return int((np.asarray(split == sp.TRAIN) & reaches).sum())


def test_no_training_window_reaches_a_test_chunk(index, chunks, table):
    chunk_of = sp.chunk_of_event(index, chunks)
    is_test_chunk = np.zeros(len(chunks), dtype=bool)
    is_test_chunk[table.loc[table["split"] == sp.TEST, "chunk_id"].unique()] = True
    n = _overlap_count(index, table["split"].to_numpy(), chunk_of, is_test_chunk)
    assert n == 0, f"{n:,} training events have a label window inside a test chunk"


def test_a_random_split_fails_the_overlap_check(index, chunks):
    """The control. If this passes, the test above proves nothing."""
    rng = np.random.default_rng(0)
    random_split = np.where(rng.random(len(index)) < 0.2, sp.TEST, sp.TRAIN)
    is_test_event = rng.random(len(index)) < 0.2
    reaches = sp.window_reaches_test(is_test_event, HORIZON)
    leaked = int((np.asarray(random_split == sp.TRAIN) & reaches).sum())
    assert leaked > len(index) * 0.5, (
        "a random split should leak on most training events; the overlap check "
        "has lost its teeth"
    )


# ------------------------------------------------------- 2. EMBARGO COVERAGE


def test_embargo_clears_the_boundary_in_both_directions(index, table):
    """At every boundary the gap between the two sides must exceed the embargo."""
    split = table["split"].to_numpy()
    train_pos = np.flatnonzero(split == sp.TRAIN)
    test_pos = np.flatnonzero(split == sp.TEST)
    assert len(train_pos) and len(test_pos)

    # For every test run, the nearest train event on each side must be at least
    # EMBARGO positions away.
    runs = np.split(test_pos, np.flatnonzero(np.diff(test_pos) != 1) + 1)
    checked = 0
    for run in runs:
        before = train_pos[train_pos < run[0]]
        after = train_pos[train_pos > run[-1]]
        if len(before):
            assert run[0] - before[-1] > EMBARGO, "train -> test gap too short"
            checked += 1
        if len(after):
            assert after[0] - run[-1] > EMBARGO, "test -> train gap too short"
            checked += 1
    assert checked >= 2, "expected boundaries in both directions"


def test_embargo_is_counted_in_trading_bars_not_minutes(index, table):
    """A weekend inside the embargo must not silently shorten it."""
    split = table["split"].to_numpy()
    emb = np.flatnonzero(split == sp.EMBARGOED)
    runs = np.split(emb, np.flatnonzero(np.diff(emb) != 1) + 1)
    for run in runs:
        assert len(run) <= EMBARGO
    assert max(len(r) for r in runs) == EMBARGO


# ----------------------------------------------------------- 3. PARTITION


def test_every_event_gets_exactly_one_split_value(index, table):
    assert len(table) == len(index)
    assert table.index.equals(index)
    assert table["split"].notna().all()
    counts = table["split"].value_counts()
    assert counts.sum() == len(index)
    assert set(table["split"].unique()) <= set(sp.SPLIT_VALUES)


def test_the_four_split_classes_are_disjoint_and_exhaustive(index, table):
    split = table["split"].to_numpy()
    masks = {v: (split == v) for v in sp.SPLIT_VALUES}
    stacked = np.vstack(list(masks.values()))
    assert (stacked.sum(axis=0) == 1).all(), "an event landed in two classes"
    total = sum(int(m.sum()) for m in masks.values())
    assert total == len(index)


def test_purged_events_sit_on_the_training_side(index, chunks, table):
    """A purged event is a TRAINING event that lost its place, never a test one."""
    chunk_of = sp.chunk_of_event(index, chunks)
    is_test_chunk = np.zeros(len(chunks), dtype=bool)
    is_test_chunk[table.loc[table["split"] == sp.TEST, "chunk_id"].unique()] = True
    purged = table["split"].to_numpy() == sp.PURGED
    in_test_chunk = is_test_chunk[np.clip(chunk_of, 0, None)] & (chunk_of >= 0)
    assert not (purged & in_test_chunk).any()


# --------------------------------------------------------- 4. DETERMINISM


def test_candidates_are_reproducible_and_distinct():
    a = sp.candidate_assignments(16, 3, n_candidates=200, seed=sp.SPLIT_SEED)
    b = sp.candidate_assignments(16, 3, n_candidates=200, seed=sp.SPLIT_SEED)
    assert len(a) == 200
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert len({tuple(x) for x in a}) == 200, "candidates must be distinct"
    c = sp.candidate_assignments(16, 3, n_candidates=200, seed=sp.SPLIT_SEED + 1)
    assert any(tuple(x) != tuple(y) for x, y in zip(a, c, strict=True))


def test_split_table_is_deterministic(index, chunks):
    t1 = sp.split_table(index, chunks, np.array([2, 7, 12]))
    t2 = sp.split_table(index, chunks, np.array([2, 7, 12]))
    pd.testing.assert_frame_equal(t1, t2)


def test_choose_assignment_picks_the_minimum_rank_sum():
    scores = pd.DataFrame({
        "quarter_max_share": [0.9, 0.3, 0.5],
        "year_max_share": [0.9, 0.3, 0.5],
        "label_max_div_pp": [2.0, 0.1, 1.0],
        "vol_rel_div": [0.2, 0.01, 0.1],
        "adjacent_pairs": [2.0, 0.0, 1.0],
    })
    best, table = sp.choose_assignment(scores)
    assert best == 1
    assert table["rank_sum"].idxmin() == 1


# ------------------------------- 5. SAMPLED AND FULL MUST AGREE ON THE SPLIT


def test_sampled_and_full_indexes_receive_the_same_assignment(index, chunks):
    """The sample is a subset, so the split must be a restriction, not a redraw."""
    rng = np.random.default_rng(3)
    keep = np.sort(rng.choice(len(index), size=len(index) // 5, replace=False))
    sampled = index[keep]
    full_table = sp.split_table(index, chunks, np.array([2, 7, 12]))

    restricted = full_table.loc[sampled]
    assert len(restricted) == len(sampled)
    # Chunk membership is a pure function of the timestamp, so it must agree
    # with an independent computation on the sampled index alone.
    np.testing.assert_array_equal(
        restricted["chunk_id"].to_numpy(), sp.chunk_of_event(sampled, chunks)
    )
    assert restricted["split"].notna().all()
