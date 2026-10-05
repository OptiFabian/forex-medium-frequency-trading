"""Data-quality checks for 1-minute FX bar series.

Gaps in the timestamp sequence are classified into three categories:

- **weekend**: spans the FX weekend close. Anchored to ~Friday 17:00
  America/New_York through ~Sunday 17:00 New York. Using NY local time
  rather than a fixed UTC hour keeps the rule correct across daylight
  saving transitions (NY 17:00 == UTC 22:00 in winter / UTC 21:00 in
  summer).
- **daily_reset**: the FX daily settlement / rollover window, a
  ~15-20 minute hole each Monday-Thursday around NY 17:00. Friday's
  reset is absorbed into the weekend window (no double-counting).
- **active**: anything else. Intraday gaps during normal trading hours,
  mid-week missing bars, or unusually long weekend extensions. Only
  active gaps should be treated as potential data-quality concerns.

The tolerances below are deliberately generous on the "after" side
because liquidity often returns a few minutes after the reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

ONE_MIN = pd.Timedelta(minutes=1)
NY_TZ = ZoneInfo("America/New_York")

# Forex daily close / weekly close are both anchored to NY 17:00.
FX_CLOSE_HOUR_NY = 17

# How far before / after NY 17:00 we still recognize the daily reset.
DAILY_RESET_BEFORE = timedelta(minutes=5)
DAILY_RESET_AFTER = timedelta(minutes=20)

# Weekend tolerance: the Sunday reopen is itself delayed by the daily
# reset, so allow ~20 min past Sun 17:00 NY.
WEEKEND_TOLERANCE_BEFORE = timedelta(minutes=5)
WEEKEND_TOLERANCE_AFTER = timedelta(minutes=20)

# Minimum duration for a gap to be considered weekend (rules out a
# hypothetical Friday-evening gap that would otherwise fit the NY bounds).
MIN_WEEKEND_DURATION = timedelta(hours=24)

GapKind = Literal["weekend", "daily_reset", "active"]


@dataclass
class Gap:
    """A single discontinuity in the timestamp series."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta
    kind: GapKind


@dataclass
class QualitySummary:
    """Per-series data-quality summary."""

    row_count: int = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    weekend_gaps: int = 0
    daily_reset_gaps: int = 0
    active_gaps: int = 0
    largest_active_gap: pd.Timedelta | None = None
    sample_active_gaps: list[Gap] = field(default_factory=list)

    def format(self, label: str) -> str:
        """Human-readable multi-line summary."""
        if self.row_count == 0:
            return f"  {label}: (empty)"
        span = (
            f"{self.earliest.isoformat()} -> {self.latest.isoformat()}"
            if self.earliest and self.latest
            else "(unknown range)"
        )
        lines = [
            f"  {label}:",
            f"    rows: {self.row_count:,}",
            f"    range: {span}",
            (
                f"    gaps: {self.weekend_gaps} weekend, "
                f"{self.daily_reset_gaps} daily_reset, "
                f"{self.active_gaps} active"
            ),
        ]
        if self.largest_active_gap is not None:
            lines.append(f"    largest active gap: {self.largest_active_gap}")
        for g in self.sample_active_gaps[:5]:
            lines.append(
                f"      - {g.start.isoformat()} -> {g.end.isoformat()} ({g.duration})"
            )
        return "\n".join(lines)


def _classify_gap(start: pd.Timestamp, end: pd.Timestamp) -> GapKind:
    """Classify a discontinuity in a 1-min bar series.

    `start` is the last bar before the gap, `end` is the first bar after.
    The "missing" period is therefore [start+1min, end-1min].

    Order of checks:
      1. weekend (requires gap >= 24h and bounds inside Fri17 NY -> Sun17 NY)
      2. daily_reset (Mon-Thu only, within ~NY 17:00 +/- tolerance, same day)
      3. otherwise active
    """
    missing_start = start + ONE_MIN
    missing_end = end - ONE_MIN
    duration = end - start

    ms_ny = missing_start.tz_convert(NY_TZ)
    me_ny = missing_end.tz_convert(NY_TZ)

    # --- Weekend ---
    if duration >= MIN_WEEKEND_DURATION:
        # Anchor to the Friday 17:00 NY at or before missing_start.
        days_since_fri = (ms_ny.weekday() - 4) % 7
        fri_close_ny = (ms_ny - timedelta(days=days_since_fri)).replace(
            hour=FX_CLOSE_HOUR_NY, minute=0, second=0, microsecond=0
        )
        sun_open_ny = fri_close_ny + timedelta(days=2)
        if (
            ms_ny >= fri_close_ny - WEEKEND_TOLERANCE_BEFORE
            and me_ny <= sun_open_ny + WEEKEND_TOLERANCE_AFTER
        ):
            return "weekend"

    # --- Daily reset (Mon-Thu only; Fri's reset belongs to the weekend) ---
    if ms_ny.weekday() in (0, 1, 2, 3) and ms_ny.date() == me_ny.date():
        reset_anchor = ms_ny.replace(
            hour=FX_CLOSE_HOUR_NY, minute=0, second=0, microsecond=0
        )
        if (
            ms_ny >= reset_anchor - DAILY_RESET_BEFORE
            and me_ny <= reset_anchor + DAILY_RESET_AFTER
        ):
            return "daily_reset"

    return "active"


def analyze(df: pd.DataFrame, sample_size: int = 5) -> QualitySummary:
    """Compute a `QualitySummary` for a bars DataFrame.

    Parameters
    ----------
    df:
        DataFrame with a `timestamp` column (UTC, tz-aware).
    sample_size:
        Number of active-gap examples to retain in the summary.
    """
    if df.empty:
        return QualitySummary()

    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values().reset_index(drop=True)
    diffs = ts.diff()

    summary = QualitySummary(
        row_count=len(ts),
        earliest=ts.iloc[0].to_pydatetime(),
        latest=ts.iloc[-1].to_pydatetime(),
    )

    gap_mask = diffs > ONE_MIN
    if not gap_mask.any():
        return summary

    largest_active: pd.Timedelta | None = None
    samples: list[Gap] = []
    for idx in diffs.index[gap_mask]:
        start = ts.iloc[idx - 1]
        end = ts.iloc[idx]
        duration = diffs.iloc[idx]
        kind = _classify_gap(start, end)
        if kind == "weekend":
            summary.weekend_gaps += 1
        elif kind == "daily_reset":
            summary.daily_reset_gaps += 1
        else:
            summary.active_gaps += 1
            if largest_active is None or duration > largest_active:
                largest_active = duration
            if len(samples) < sample_size:
                samples.append(Gap(start=start, end=end, duration=duration, kind="active"))

    summary.largest_active_gap = largest_active
    summary.sample_active_gaps = samples
    return summary
