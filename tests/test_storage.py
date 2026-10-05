"""Tests for `fxalgo.data.storage` — round-trip, dedup, month splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fxalgo.data.storage import (
    BAR_COLUMNS,
    BarsStorage,
    empty_bars_df,
    merge_dataframes,
    read_parquet,
    write_parquet,
)


def _make_bars(
    start: str = "2024-01-01 00:00",
    periods: int = 10,
    freq: str = "1min",
    base_price: float = 1.10,
) -> pd.DataFrame:
    """Build a synthetic bars DataFrame with strictly-increasing close prices."""
    ts = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [base_price + i * 0.0001 for i in range(periods)],
            "high": [base_price + i * 0.0001 + 0.0002 for i in range(periods)],
            "low": [base_price + i * 0.0001 - 0.0002 for i in range(periods)],
            "close": [base_price + i * 0.0001 + 0.00005 for i in range(periods)],
            "volume": [0.0] * periods,
        }
    )


def test_empty_bars_df_has_correct_schema() -> None:
    df = empty_bars_df()
    assert list(df.columns) == list(BAR_COLUMNS)
    assert df.empty
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    df = _make_bars(periods=5)
    path = tmp_path / "out.parquet"
    write_parquet(df, path)

    loaded = read_parquet(path)
    pd.testing.assert_frame_equal(df, loaded, check_dtype=True)


def test_read_missing_returns_empty(tmp_path: Path) -> None:
    loaded = read_parquet(tmp_path / "does_not_exist.parquet")
    assert loaded.empty
    assert list(loaded.columns) == list(BAR_COLUMNS)


def test_merge_dedups_keep_last() -> None:
    first = _make_bars(periods=10, base_price=1.10)
    # Overlap timestamps 5..9 with different prices; expect merge to keep `second`.
    second = _make_bars(start="2024-01-01 00:05", periods=10, base_price=2.00)

    merged = merge_dataframes(first, second)
    # 10 original + 10 new - 5 overlap = 15 unique timestamps
    assert len(merged) == 15
    assert merged["timestamp"].is_unique
    assert merged["timestamp"].is_monotonic_increasing

    # Timestamps 0..4 should retain first's prices, 5..14 should reflect second's.
    overlap_ts = first["timestamp"].iloc[5]
    overlap_close = merged.loc[merged["timestamp"] == overlap_ts, "close"].iloc[0]
    second_close = second.loc[second["timestamp"] == overlap_ts, "close"].iloc[0]
    assert overlap_close == pytest.approx(second_close)


def test_merge_with_none_existing() -> None:
    new = _make_bars(periods=3)
    merged = merge_dataframes(None, new)
    pd.testing.assert_frame_equal(merged, new, check_dtype=True)


def test_bars_storage_round_trip_single_month(tmp_path: Path) -> None:
    storage = BarsStorage(tmp_path, pair="EURUSD", side="bid")
    df = _make_bars(periods=20)
    added = storage.append(df)
    assert added == 20
    assert storage.row_count() == 20

    loaded = storage.read_all()
    pd.testing.assert_frame_equal(loaded, df, check_dtype=True)


def test_bars_storage_splits_by_month(tmp_path: Path) -> None:
    # 90 hourly bars starting late Jan -> spans Jan/Feb/March 2024
    df = pd.concat(
        [
            _make_bars(start="2024-01-30 00:00", periods=72, freq="1h"),  # Jan + Feb
            _make_bars(start="2024-03-01 00:00", periods=24, freq="1h"),  # March
        ],
        ignore_index=True,
    )
    storage = BarsStorage(tmp_path, pair="EURUSD", side="ask")
    storage.append(df)

    files = sorted(p.name for p in (tmp_path / "EURUSD" / "ask").iterdir())
    assert files == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet"]

    loaded = storage.read_all()
    assert len(loaded) == len(df)
    assert loaded["timestamp"].is_monotonic_increasing


def test_bars_storage_append_is_resumable(tmp_path: Path) -> None:
    """A second append with the same data should add zero new rows."""
    storage = BarsStorage(tmp_path, pair="EURUSD", side="bid")
    df = _make_bars(periods=30)
    added1 = storage.append(df)
    added2 = storage.append(df)
    assert added1 == 30
    assert added2 == 0
    assert storage.row_count() == 30


def test_bars_storage_timestamp_range(tmp_path: Path) -> None:
    storage = BarsStorage(tmp_path, pair="EURUSD", side="bid")
    assert storage.timestamp_range() == (None, None)

    df = _make_bars(start="2024-01-15 00:00", periods=100, freq="1h")  # spans Jan/Feb
    storage.append(df)
    earliest, latest = storage.timestamp_range()
    assert earliest == df["timestamp"].iloc[0].to_pydatetime()
    assert latest == df["timestamp"].iloc[-1].to_pydatetime()
