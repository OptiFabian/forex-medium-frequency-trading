"""Tests for the default data window + cross-pair alignment (fxalgo.data.window).

Synthetic multi-pair storage: daily bars (the window logic is
timestamp-based, not frequency-based) with deliberately different first
and last bars per pair, so the max-of-min start and min-of-max end are
distinguishable.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fxalgo.data.storage import BarsStorage
from fxalgo.data.window import (
    DEFAULT_WINDOW_YEARS,
    Window,
    common_window,
    discover_pairs,
    pair_bounds,
    slice_window,
)
from fxalgo.features.loader import load_pair


def _write_pair(root, pair: str, start: str, end: str, *, ask_only=False, freq="1D") -> None:
    """Write identical bid+ask daily bars for `pair` spanning [start, end]."""
    idx = pd.date_range(start, end, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 1.10,
            "high": 1.101,
            "low": 1.099,
            "close": 1.10,
            "volume": -1.0,
        }
    )
    sides = ("ask",) if ask_only else ("bid", "ask")
    for side in sides:
        BarsStorage(root, pair, side).append(df)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


# ============================================================
# 1. The min-of-max end-date search.
# ============================================================


def test_common_end_is_min_of_last_bars(tmp_path):
    """The universal cutoff is the EARLIEST last-candle across pairs."""
    _write_pair(tmp_path, "AAA", "2024-01-01", "2025-06-30")
    _write_pair(tmp_path, "BBB", "2024-01-01", "2025-05-31")  # earliest tail
    _write_pair(tmp_path, "CCC", "2024-01-01", "2025-06-15")

    w = common_window(storage_root=tmp_path, full_history=True)
    assert w.end == _ts("2025-05-31")


def test_pair_bounds_respects_inner_join_of_sides(tmp_path):
    """A pair's merged series ends at the EARLIER of its two sides' last bars
    and starts at the LATER of its two sides' first bars."""
    _write_pair(tmp_path, "AAA", "2024-01-05", "2025-06-30", ask_only=True)
    # bid side starts earlier but ends earlier too
    idx = pd.date_range("2024-01-01", "2025-06-20", freq="1D", tz="UTC")
    bid = pd.DataFrame(
        {"timestamp": idx, "open": 1.1, "high": 1.1, "low": 1.1, "close": 1.1, "volume": -1.0}
    )
    BarsStorage(tmp_path, "AAA", "bid").append(bid)

    first, last = pair_bounds("AAA", tmp_path)
    assert first == _ts("2024-01-05")  # max of the two firsts
    assert last == _ts("2025-06-20")  # min of the two lasts


# ============================================================
# 2. The default N-year window start.
# ============================================================


def test_default_window_start_is_years_before_common_end(tmp_path):
    """With ample history, start = end - DEFAULT_WINDOW_YEARS years."""
    _write_pair(tmp_path, "AAA", "2015-01-01", "2026-06-30")
    _write_pair(tmp_path, "BBB", "2015-01-01", "2026-05-31")

    w = common_window(storage_root=tmp_path)
    assert w.end == _ts("2026-05-31")
    assert w.start == w.end - pd.DateOffset(years=DEFAULT_WINDOW_YEARS)


def test_window_start_clamps_to_common_first_bar(tmp_path):
    """A pair with less than N years of history clamps the start forward so the
    window never begins before every pair has data."""
    _write_pair(tmp_path, "AAA", "2015-01-01", "2026-06-30")
    _write_pair(tmp_path, "BBB", "2023-02-01", "2026-06-30")  # < 5 years

    w = common_window(storage_root=tmp_path)
    assert w.start == _ts("2023-02-01")  # max-of-min firsts, not end - 5y


def test_full_history_start_is_max_of_first_bars(tmp_path):
    """full_history aligns the start at the LATEST first-candle across pairs."""
    _write_pair(tmp_path, "AAA", "2018-01-01", "2026-06-30")
    _write_pair(tmp_path, "BBB", "2018-03-01", "2026-06-30")

    w = common_window(storage_root=tmp_path, full_history=True)
    assert w.start == _ts("2018-03-01")
    assert w.end == _ts("2026-06-30")


# ============================================================
# 3. All pairs share an identical [start, end] after alignment.
# ============================================================


def test_all_pairs_identical_start_end_after_alignment(tmp_path):
    """Loading each pair under the default window yields the same first and
    last timestamp for every pair (bars others lack are truncated)."""
    _write_pair(tmp_path, "AAA", "2019-01-01", "2026-06-30")
    _write_pair(tmp_path, "BBB", "2019-01-01", "2026-06-10")
    _write_pair(tmp_path, "CCC", "2019-01-01", "2026-06-20")

    frames = {p: load_pair(p, storage_root=tmp_path) for p in ("AAA", "BBB", "CCC")}
    firsts = {df.index[0] for df in frames.values()}
    lasts = {df.index[-1] for df in frames.values()}
    assert len(firsts) == 1, f"starts differ across pairs: {firsts}"
    assert len(lasts) == 1, f"ends differ across pairs: {lasts}"
    assert lasts.pop() == _ts("2026-06-10")  # min-of-max
    # Identical daily grid within one shared window -> identical row counts.
    counts = {p: len(df) for p, df in frames.items()}
    assert len(set(counts.values())) == 1, f"row counts differ: {counts}"


# ============================================================
# 4. Overrides restore full history / explicit bounds.
# ============================================================


def test_full_history_override_restores_all_rows(tmp_path):
    """load_pair(full_history=True) must return the pair's whole history even
    when the default window would truncate it."""
    _write_pair(tmp_path, "AAA", "2015-01-01", "2026-06-30")
    _write_pair(tmp_path, "BBB", "2015-01-01", "2026-01-31")  # drags cutoff back

    windowed = load_pair("AAA", storage_root=tmp_path)
    full = load_pair("AAA", storage_root=tmp_path, full_history=True)

    assert windowed.index[-1] == _ts("2026-01-31")  # default: aligned cutoff
    assert full.index[0] == _ts("2015-01-01")
    assert full.index[-1] == _ts("2026-06-30")  # override: own full history
    assert len(full) > len(windowed)


def test_explicit_start_end_bypass_default_window(tmp_path):
    """Explicit start/end are honored exactly, ignoring the default window."""
    _write_pair(tmp_path, "AAA", "2015-01-01", "2026-06-30")

    df = load_pair(
        "AAA",
        start=_ts("2016-02-01"),
        end=_ts("2016-03-01"),
        storage_root=tmp_path,
    )
    assert df.index[0] == _ts("2016-02-01")
    assert df.index[-1] == _ts("2016-03-01")


# ============================================================
# 5. Plumbing: discovery, validation, slicing.
# ============================================================


def test_discover_pairs_requires_both_sides(tmp_path):
    _write_pair(tmp_path, "AAA", "2024-01-01", "2024-02-01")
    _write_pair(tmp_path, "BBB", "2024-01-01", "2024-02-01", ask_only=True)
    (tmp_path / "not_a_pair.txt").write_text("x")

    assert discover_pairs(tmp_path) == ["AAA"]


def test_common_window_raises_on_empty_root(tmp_path):
    with pytest.raises(ValueError, match="No pairs"):
        common_window(storage_root=tmp_path)


def test_window_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="after end"):
        Window(start=_ts("2025-01-02"), end=_ts("2025-01-01"))


def test_slice_window_is_inclusive_both_ends(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"x": range(10)}, index=idx)
    w = Window(start=idx[2], end=idx[7])
    out = slice_window(df, w)
    assert out.index[0] == idx[2]
    assert out.index[-1] == idx[7]
    assert len(out) == 6
