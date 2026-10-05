"""Time + session features.

Calendar features come from the timestamp index; session flags are
anchored to the local exchange timezones so that DST is handled
automatically by `zoneinfo` (no fixed UTC hour assumptions).

Session definitions (local market hours, Mon-Fri only):

  Tokyo:    09:00-18:00  Asia/Tokyo       (no DST in Japan)
  London:   08:00-17:00  Europe/London    (GMT in winter, BST in summer)
  New York: 08:00-17:00  America/New_York (EST in winter, EDT in summer)

`session_london_ny_overlap` is the conjunction of the two -- this is the
deepest-liquidity window of the trading day.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TOKYO_TZ = ZoneInfo("Asia/Tokyo")
LONDON_TZ = ZoneInfo("Europe/London")
NEW_YORK_TZ = ZoneInfo("America/New_York")

# Local-time hours (inclusive start, exclusive end) for each session.
SESSION_START_HOUR = 8  # London / New York
TOKYO_START_HOUR = 9
SESSION_END_HOUR = 17  # all three sessions end at local 17:00
TOKYO_END_HOUR = 18


def _session_flag(idx: pd.DatetimeIndex, tz: ZoneInfo, start_hour: int, end_hour: int) -> np.ndarray:
    """Boolean array: True when local time is in [start_hour, end_hour) on a weekday."""
    local = idx.tz_convert(tz)
    hour = local.hour
    weekday = local.weekday  # Mon=0..Sun=6
    in_hours = (hour >= start_hour) & (hour < end_hour)
    in_week = weekday < 5
    return (in_hours & in_week).astype("int8")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append calendar + session columns.

    Columns added:
      - hour_sin, hour_cos: cyclic encoding of UTC hour (0..23)
      - dow_sin, dow_cos: cyclic encoding of UTC weekday (0..6, Mon=0)
      - session_tokyo, session_london, session_ny: int8 flags (0/1)
      - session_london_ny_overlap: int8 flag (London AND NY both active)
    """
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("time_features.add_features requires a DatetimeIndex")
    if idx.tz is None:
        raise ValueError("time_features.add_features requires a tz-aware index")

    hour = idx.hour
    dow = idx.weekday
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    out["session_tokyo"] = _session_flag(idx, TOKYO_TZ, TOKYO_START_HOUR, TOKYO_END_HOUR)
    out["session_london"] = _session_flag(idx, LONDON_TZ, SESSION_START_HOUR, SESSION_END_HOUR)
    out["session_ny"] = _session_flag(idx, NEW_YORK_TZ, SESSION_START_HOUR, SESSION_END_HOUR)
    out["session_london_ny_overlap"] = (
        out["session_london"].to_numpy() & out["session_ny"].to_numpy()
    ).astype("int8")
    return out
