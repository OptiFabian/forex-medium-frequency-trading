"""Tests for fxalgo.features.time_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.features import time_features
from tests.features.conftest import make_frame


def test_cyclic_hour_encoding_at_midnight_and_noon():
    df = make_frame([1.0], start="2024-03-04 00:00")
    out = time_features.add_features(df)
    # 0h UTC: sin(0)=0, cos(0)=1
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0)

    df = make_frame([1.0], start="2024-03-04 12:00")
    out = time_features.add_features(df)
    # 12h UTC: sin(pi) = 0, cos(pi) = -1
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["hour_cos"].iloc[0] == pytest.approx(-1.0)


def test_cyclic_dow_encoding_monday():
    # 2024-03-04 is a Monday.
    df = make_frame([1.0], start="2024-03-04 00:00")
    out = time_features.add_features(df)
    # Monday = 0: sin(0)=0, cos(0)=1
    assert out["dow_sin"].iloc[0] == pytest.approx(0.0)
    assert out["dow_cos"].iloc[0] == pytest.approx(1.0)


def test_session_flags_on_a_summer_weekday_during_dst():
    """2024-07-15 is a Monday in EDT (UTC-4) and BST (UTC+1)."""
    # NY session 08:00-17:00 EDT = 12:00-21:00 UTC. Pick 14:00 UTC -> 10:00 NY.
    # London session 08:00-17:00 BST = 07:00-16:00 UTC. 14:00 UTC -> 15:00 London. Active.
    # Tokyo session 09:00-18:00 JST (no DST) = 00:00-09:00 UTC. 14:00 UTC -> 23:00 Tokyo. Inactive.
    df = make_frame([1.0], start="2024-07-15 14:00")
    out = time_features.add_features(df)
    assert out["session_ny"].iloc[0] == 1
    assert out["session_london"].iloc[0] == 1
    assert out["session_tokyo"].iloc[0] == 0
    assert out["session_london_ny_overlap"].iloc[0] == 1


def test_session_flags_winter_dst_shift():
    """2024-01-15 is a Monday in EST (UTC-5) and GMT (UTC+0).

    NY 08:00-17:00 EST = 13:00-22:00 UTC. At 13:00 UTC -> NY 08:00 (start of session).
    London 08:00-17:00 GMT = 08:00-17:00 UTC. At 13:00 UTC -> London 13:00 (active).
    """
    df = make_frame([1.0], start="2024-01-15 13:00")
    out = time_features.add_features(df)
    assert out["session_ny"].iloc[0] == 1
    assert out["session_london"].iloc[0] == 1
    assert out["session_london_ny_overlap"].iloc[0] == 1


def test_session_flags_off_on_weekend():
    # 2024-07-13 is a Saturday.
    df = make_frame([1.0], start="2024-07-13 14:00")
    out = time_features.add_features(df)
    assert out["session_ny"].iloc[0] == 0
    assert out["session_london"].iloc[0] == 0
    assert out["session_tokyo"].iloc[0] == 0


def test_session_tokyo_active_at_japan_midday():
    """2024-07-15 03:00 UTC = Tokyo 12:00 JST."""
    df = make_frame([1.0], start="2024-07-15 03:00")
    out = time_features.add_features(df)
    assert out["session_tokyo"].iloc[0] == 1
    # London (BST UTC+1) at 03:00 UTC = 04:00 BST -> not yet open.
    assert out["session_london"].iloc[0] == 0


def test_naive_index_raises():
    n = 3
    df = make_frame([1.0, 1.1, 1.2])
    df.index = pd.date_range("2024-01-01", periods=n, freq="1min")  # naive
    with pytest.raises(ValueError, match="tz-aware"):
        time_features.add_features(df)
