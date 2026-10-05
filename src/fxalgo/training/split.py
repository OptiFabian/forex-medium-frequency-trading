"""Blocked-by-time train/test split with purging and embargo.

SPLIT ONLY. Nothing here trains, scores or selects a model.

WHY NOT A RANDOM SPLIT
----------------------
Every label watches 240 forward bars. A randomly assigned split puts a training
event at 14:00 and a test event at 14:15, and the training event's outcome
window covers the test event whole. At a 240-bar horizon each test event has on
the order of 240 training events whose outcome windows overlap it. The model
scores well and the score means nothing.

Contiguous time chunks confine that to chunk BOUNDARIES, where it can be purged.

THE THREE MECHANISMS
--------------------
1. CHUNKS. Contiguous three-month blocks, boundaries deliberately offset from
   calendar quarter ends (see `build_chunks`).
2. PURGE. A training event whose 240-bar window reaches into a test chunk is
   removed from training. Computed from the actual horizon in TRADING bars, not
   from a wall-clock approximation -- a 240-bar window spans 240 minutes inside
   a session and up to 4,740 minutes across a weekend.
3. EMBARGO. A buffer of events on each side of every boundary, removed because
   markets are autocorrelated and adjacency leaks even where label windows do
   not literally overlap.

Nothing is deleted. Purged and embargoed events keep their own split value, so
the cost is visible and the decision is reversible without a rebuild.

EMBARGO LENGTH, AND A UNIT CORRECTION
-------------------------------------
The mean label window at k=2.0 is **135.57 ONE-minute bars** -- 9.04 fifteen-
minute blocks, about 2h16m of trading. (`data/labels/effective_sample_size.csv`
records it as `mean_window_bars`, and 1,846,315 / 135.57 = 13,619 is exactly the
recorded N_eff, which fixes the unit as one-minute bars.) It is NOT 135.6
fifteen-minute bars; that reading would overstate the window fifteen-fold.

The binding number for an embargo is not the mean anyway -- it is the MAXIMUM,
which is the horizon cap itself: 240 bars. `EMBARGO_BARS` is set to 1,440
trading bars, one full trading day:

  * 6.0x the 240-bar maximum window, so no label window can reach across it;
  * 10.6x the 135.57-bar mean;
  * one complete rotation of the Tokyo / London / New York session cycle, so
    session-of-day autocorrelation -- the reason an embargo exists at all, over
    and above window overlap -- is cleared rather than half-cleared.

Bars, not minutes: the embargo is counted in TRADING bars like the horizon, so a
weekend inside it does not silently shorten it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHUNK_MONTHS = 3
TEST_FRACTION = 0.20
N_CANDIDATES = 200
SPLIT_SEED = 20260815
LABEL_HORIZON = 240
EMBARGO_BARS = 1_440

TRAIN, TEST, PURGED, EMBARGOED, WARMUP = "train", "test", "purged", "embargoed", "warmup"
SPLIT_VALUES = (TRAIN, TEST, PURGED, EMBARGOED, WARMUP)

# The five balance criteria, FIXED BEFORE any candidate is drawn. Each candidate
# is ranked on each criterion across the whole candidate pool and the winner
# minimises the SUM OF RANKS -- equal weight, no thresholds to tune after the
# fact, no scale to argue about.
CRITERIA = (
    "quarter_max_share",   # a) largest share of test events in one calendar quarter
    "year_max_share",      # b) largest share of test events in one calendar year
    "label_max_div_pp",    # c) largest |p_test - p_train| over the three classes
    "vol_rel_div",         # d) |median atr_bps_15m test - train| / train
    "adjacent_pairs",      # e) count of test chunks that touch another test chunk
)


# ------------------------------------------------------------------- chunking


def build_chunks(
    start: pd.Timestamp, end: pd.Timestamp, *, months: int = CHUNK_MONTHS
) -> pd.DataFrame:
    """Contiguous `months`-long chunks covering [start, end].

    OFFSET. The grid is anchored at the first usable timestamp rather than at a
    calendar quarter. With the usable range starting 2022-05-26 the boundaries
    land on the 26th of February / May / August / November -- roughly 34 days
    from the nearest quarter end and 56 days after the nearest quarter start.

    That matters because a grid ON the quarters makes every chunk a calendar
    quarter, and three test chunks drawn from such a grid can be three fourth
    quarters: a test set made entirely of December holiday liquidity and
    year-end flow. Off the quarters, EVERY chunk straddles two calendar
    quarters, so no single test chunk can be one season and the concentration
    risk is structurally halved before any balance check runs.

    A trailing remainder shorter than half a chunk is absorbed into the final
    chunk rather than left as a stub, so no chunk is a fraction of the others.
    """
    edges = [start]
    while edges[-1] <= end:
        edges.append(edges[-1] + pd.DateOffset(months=months))
    if len(edges) > 2:
        tail = (end - edges[-2]) / pd.Timedelta(days=1)
        if tail < 30 * months / 2:
            edges.pop(-2)          # absorb the stub into the previous chunk
    rows = [
        {"chunk_id": i, "chunk_start": edges[i], "chunk_end": edges[i + 1]}
        for i in range(len(edges) - 1)
    ]
    out = pd.DataFrame(rows)
    # Report the last chunk's end at the data, not at the nominal grid line it
    # was extended to -- a chunk whose stated range runs months past the last
    # event reads as a gap in coverage that is not there.
    out.loc[out.index[-1], "chunk_end"] = min(out["chunk_end"].iloc[-1],
                                              end + pd.Timedelta(minutes=1))
    return out


def chunk_of_event(index: pd.DatetimeIndex, chunks: pd.DataFrame) -> np.ndarray:
    """Chunk id per event; -1 before the grid starts (feature warmup)."""
    starts = chunks["chunk_start"].to_numpy()
    pos = np.searchsorted(starts, index.to_numpy(), side="right") - 1
    return np.where(index.to_numpy() < starts[0], -1, pos)


# ------------------------------------------------- candidate draw and scoring


def candidate_assignments(
    n_chunks: int, n_test: int, *, n_candidates: int = N_CANDIDATES, seed: int = SPLIT_SEED
) -> list[np.ndarray]:
    """Distinct random test-chunk selections, drawn once under `seed`.

    With sixteen chunks the grid is coarse and a single draw lands lopsided
    easily, so the choice is made from a pool rather than taken first-come.
    """
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < n_candidates and attempts < n_candidates * 50:
        attempts += 1
        pick = tuple(sorted(rng.choice(n_chunks, size=n_test, replace=False)))
        if pick in seen:
            continue
        seen.add(pick)
        out.append(np.array(pick, dtype="int64"))
    return out


def score_candidate(
    test_chunks: np.ndarray,
    *,
    n_chunks: int,
    quarter_counts: np.ndarray,
    year_counts: np.ndarray,
    label_counts: np.ndarray,
    atr_by_chunk: list[np.ndarray],
) -> dict[str, float]:
    """The five balance criteria for one candidate. Lower is better throughout."""
    is_test = np.zeros(n_chunks, dtype=bool)
    is_test[test_chunks] = True

    q = quarter_counts[is_test].sum(axis=0)
    y = year_counts[is_test].sum(axis=0)
    lt = label_counts[is_test].sum(axis=0)
    ltr = label_counts[~is_test].sum(axis=0)
    p_test = lt / lt.sum()
    p_train = ltr / ltr.sum()

    med_test = float(np.median(np.concatenate([atr_by_chunk[i] for i in test_chunks])))
    train_ids = np.flatnonzero(~is_test)
    med_train = float(np.median(np.concatenate([atr_by_chunk[i] for i in train_ids])))

    adjacent = int(np.sum(np.diff(np.sort(test_chunks)) == 1))
    return {
        "quarter_max_share": float(q.max() / q.sum()),
        "year_max_share": float(y.max() / y.sum()),
        "label_max_div_pp": float(np.abs(p_test - p_train).max() * 100),
        "vol_rel_div": float(abs(med_test - med_train) / med_train),
        "adjacent_pairs": float(adjacent),
        "median_atr_test": med_test,
        "median_atr_train": med_train,
    }


def choose_assignment(scores: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Rank each candidate on each criterion; the winner minimises the rank sum.

    Rank-based rather than a weighted sum of raw values: the five criteria have
    incommensurable units (a share, a share, percentage points, a ratio, a
    count) and any weighting would be a knob to twiddle after seeing the
    results. Ranks need no scale and no threshold.
    """
    ranked = scores[list(CRITERIA)].rank(method="average")
    out = scores.copy()
    out["rank_sum"] = ranked.sum(axis=1)
    return int(out["rank_sum"].idxmin()), out


# --------------------------------------------------------- purge and embargo


def window_reaches_test(is_test_event: np.ndarray, horizon: int = LABEL_HORIZON) -> np.ndarray:
    """True where the event's forward window (positions i+1..i+horizon) hits test.

    Positions, not timestamps: the horizon is defined in TRADING bars, so this
    is exact across weekends and holidays where a wall-clock rule would not be.
    """
    n = len(is_test_event)
    prefix = np.concatenate([[0], np.cumsum(is_test_event.astype("int64"))])
    hi = np.minimum(np.arange(n) + horizon, n - 1)
    return np.asarray(prefix[hi + 1] - prefix[np.arange(n) + 1] > 0, dtype=bool)


def assign_splits(
    chunk_of: np.ndarray,
    is_test_chunk: np.ndarray,
    *,
    horizon: int = LABEL_HORIZON,
    embargo_bars: int = EMBARGO_BARS,
) -> np.ndarray:
    """Per-event split label: train / test / purged / embargoed / warmup.

    The embargo runs in BOTH directions at every boundary. Forward, it removes
    test events that sit right after a training chunk. Backward, it removes
    training events that sit right after a test chunk -- their labels look away
    from the test period, so purging never touches them, but their FEATURES are
    computed from history that runs through it.
    """
    n = len(chunk_of)
    in_grid = chunk_of >= 0
    is_test_event = in_grid & is_test_chunk[np.clip(chunk_of, 0, None)]

    split = np.where(in_grid, np.where(is_test_event, TEST, TRAIN), WARMUP).astype(object)

    purge = (split == TRAIN) & window_reaches_test(is_test_event, horizon)
    split[purge] = PURGED

    # Boundaries: positions where the event's side changes.
    side = np.where(is_test_event, 1, 0)
    side[~in_grid] = -1
    change = np.flatnonzero(np.diff(side) != 0) + 1
    embargo = np.zeros(n, dtype=bool)
    for p in change:
        if side[p] == -1 or side[p - 1] == -1:
            continue        # the warmup edge is not a train/test boundary
        embargo[p : p + embargo_bars] = True
    # Only the side we are protecting is removed; the purge decision on the
    # other side of the boundary stands.
    split[embargo & (split == TEST)] = EMBARGOED
    split[embargo & (split == TRAIN)] = EMBARGOED
    return split


def split_table(
    index: pd.DatetimeIndex,
    chunks: pd.DataFrame,
    test_chunks: np.ndarray,
    *,
    horizon: int = LABEL_HORIZON,
    embargo_bars: int = EMBARGO_BARS,
) -> pd.DataFrame:
    """The assignment table: one row per event, keyed by timestamp."""
    chunk_of = chunk_of_event(index, chunks)
    is_test_chunk = np.zeros(len(chunks), dtype=bool)
    is_test_chunk[test_chunks] = True
    split = assign_splits(chunk_of, is_test_chunk, horizon=horizon,
                          embargo_bars=embargo_bars)

    starts = chunks["chunk_start"].to_numpy()
    ends = chunks["chunk_end"].to_numpy()
    safe = np.clip(chunk_of, 0, None)
    return pd.DataFrame(
        {
            "split": pd.Categorical(split, categories=list(SPLIT_VALUES)),
            "chunk_id": np.where(chunk_of >= 0, chunk_of, -1).astype("int64"),
            "chunk_start": np.where(chunk_of >= 0, starts[safe], np.datetime64("NaT")),
            "chunk_end": np.where(chunk_of >= 0, ends[safe], np.datetime64("NaT")),
        },
        index=index,
    )
