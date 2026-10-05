"""Tests for scripts/import_bars.py: plugging in your own 1-minute data."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from fxalgo.data.storage import BarsStorage
from fxalgo.settings import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("import_bars", PROJECT_ROOT / "scripts" / "import_bars.py")
import_bars = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_bars)


def _csv(tmp_path, rows, name="bars.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_reads_csv_with_naive_timestamps_as_utc_and_stores_monthly(tmp_path):
    p = _csv(tmp_path, {
        "Time": ["2024-01-31 23:59", "2024-02-01 00:00"],
        "Open": [1.10, 1.11], "High": [1.12, 1.12], "Low": [1.09, 1.10], "Close": [1.11, 1.115],
    })
    bars = import_bars.read_bars(p)
    assert str(bars["timestamp"].dt.tz) == "UTC"
    assert (bars["volume"] == -1.0).all()
    root = tmp_path / "raw"
    store = BarsStorage(root, "EURUSD", "bid")
    assert store.append(bars) == 2
    assert sorted(f.name for f in (root / "EURUSD" / "bid").iterdir()) == ["2024-01.parquet", "2024-02.parquet"]
    back = store.read_all()
    assert len(back) == 2 and back["close"].iloc[-1] == pytest.approx(1.115)


def test_converts_a_local_time_zone_to_utc(tmp_path):
    p = _csv(tmp_path, {"timestamp": ["2024-07-01 12:00"], "open": [1], "high": [1], "low": [1], "close": [1]})
    bars = import_bars.read_bars(p, tz="America/New_York")
    assert bars["timestamp"].iloc[0] == pd.Timestamp("2024-07-01 16:00", tz="UTC")


def test_rejects_missing_columns_and_inconsistent_ohlc(tmp_path):
    with pytest.raises(ValueError, match="timestamp"):
        import_bars.read_bars(_csv(tmp_path, {"open": [1], "high": [1], "low": [1], "close": [1]}))
    with pytest.raises(ValueError, match="OHLC"):
        import_bars.read_bars(_csv(tmp_path, {"timestamp": ["2024-01-01"], "open": [1], "close": [1]}))
    with pytest.raises(ValueError, match="inconsistent"):
        import_bars.read_bars(_csv(tmp_path, {"timestamp": ["2024-01-01"], "open": [1.2], "high": [1.1],
                                              "low": [1.0], "close": [1.05]}))
