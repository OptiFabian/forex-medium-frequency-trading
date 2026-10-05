"""Tests for fxalgo.features.loader -- merge, mid, spread clipping."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import pytest

from fxalgo.data.storage import BarsStorage
from fxalgo.features.loader import load_pair


def _make_side_df(start: str, n: int, base: float):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": [base + 0.0001 * i for i in range(n)],
            "high": [base + 0.0001 * i + 0.0002 for i in range(n)],
            "low": [base + 0.0001 * i - 0.0002 for i in range(n)],
            "close": [base + 0.0001 * i + 0.00005 for i in range(n)],
            "volume": [-1.0] * n,
        }
    )


def test_loader_merges_bid_ask_and_adds_mid_spread(tmp_path):
    bid = _make_side_df("2024-01-01 00:00", 10, base=1.10)
    ask = _make_side_df("2024-01-01 00:00", 10, base=1.1002)  # ask = bid + 2 pips
    BarsStorage(tmp_path, "EURUSD", "bid").append(bid)
    BarsStorage(tmp_path, "EURUSD", "ask").append(ask)

    df = load_pair("EURUSD", storage_root=tmp_path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 10
    assert "mid_close" in df.columns
    assert "mid_open" in df.columns
    assert "spread_close" in df.columns
    assert "volume_bid" not in df.columns
    assert "volume_ask" not in df.columns
    # mid_close = (close_bid + close_ask) / 2
    expected_mid_close_first = (bid["close"].iloc[0] + ask["close"].iloc[0]) / 2.0
    assert df["mid_close"].iloc[0] == pytest.approx(expected_mid_close_first)
    # All spreads non-negative
    assert (df["spread_close"] >= 0).all()


def test_loader_clips_negative_spreads_and_logs_count(tmp_path, caplog):
    # Bid > Ask on row 3 and 7 (crossed quotes)
    bid_close = [1.1000, 1.1001, 1.1002, 1.1010, 1.1004, 1.1005, 1.1006, 1.1020]
    ask_close = [1.1002, 1.1003, 1.1004, 1.1005, 1.1006, 1.1007, 1.1008, 1.1009]
    n = len(bid_close)
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1min", tz="UTC")

    bid = pd.DataFrame(
        {
            "timestamp": idx,
            "open": bid_close,
            "high": bid_close,
            "low": bid_close,
            "close": bid_close,
            "volume": [-1.0] * n,
        }
    )
    ask = bid.copy()
    ask["open"] = ask["high"] = ask["low"] = ask["close"] = ask_close

    BarsStorage(tmp_path, "EURUSD", "bid").append(bid)
    BarsStorage(tmp_path, "EURUSD", "ask").append(ask)

    with caplog.at_level(logging.INFO, logger="fxalgo.features.loader"):
        df = load_pair("EURUSD", storage_root=tmp_path)

    # All output spreads must be >= 0 after clipping.
    assert (df["spread_close"] >= 0).all()
    # Two negative spreads were clipped (rows 3 and 7).
    assert "clipped 2 negative spreads" in caplog.text


def test_loader_date_range_filter(tmp_path):
    bid = _make_side_df("2024-01-01 00:00", 100, base=1.10)
    ask = _make_side_df("2024-01-01 00:00", 100, base=1.1002)
    BarsStorage(tmp_path, "EURUSD", "bid").append(bid)
    BarsStorage(tmp_path, "EURUSD", "ask").append(ask)

    start = datetime(2024, 1, 1, 0, 10, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
    df = load_pair("EURUSD", start=start, end=end, storage_root=tmp_path)
    assert df.index[0] == pd.Timestamp(start)
    assert df.index[-1] == pd.Timestamp(end)
    assert len(df) == 21  # 0:10 .. 0:30 inclusive


def test_loader_inner_join_drops_orphan_timestamps(tmp_path):
    # 10 bid rows starting at 0:00, 10 ask rows starting at 0:05 -> overlap 5
    bid = _make_side_df("2024-01-01 00:00", 10, base=1.10)
    ask = _make_side_df("2024-01-01 00:05", 10, base=1.1002)
    BarsStorage(tmp_path, "EURUSD", "bid").append(bid)
    BarsStorage(tmp_path, "EURUSD", "ask").append(ask)

    df = load_pair("EURUSD", storage_root=tmp_path)
    assert len(df) == 5  # only the overlap
