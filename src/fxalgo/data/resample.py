"""Resample 1-minute merged bid/ask bars to a coarser timeframe, gap-safely.

Two resamplers live here:

- `resample_merged(df, freq)`  -- fixed CLOCK windows (15min, 1h, ...).
- `resample_sessions(df, period)` -- FX TRADING SESSIONS: daily bars on the
  17:00 New York close convention, weekly bars Sunday 17:00 NY -> Friday
  17:00 NY. See "Session bars" below.

Clean-boundary rule (this is the whole point)
---------------------------------------------
Each output bar is a fixed clock window ``[t, t + freq)`` labeled at its
START ``t`` and closed on the LEFT. Its OHLC is the first / max / min / last
of the 1-minute bars whose timestamp falls in that window (per side). So a
15-minute bar is anchored to :00 / :15 / :30 / :45.

Because every output bar is a fixed clock slot, a bar can NEVER aggregate
across a market gap: a weekend or daily-reset gap is simply a run of empty
clock windows. Windows containing NO 1-minute bars are DROPPED, and the bars
on either side of the gap stay separate and correctly bounded. There is no
special-casing of weekends needed -- the boundary alignment does the work
(the same gaps the `quality` module classifies as weekend / daily_reset /
active show up here as empty windows).

Partial bars
------------
A window at a session edge or in thin liquidity may contain fewer than
``freq/1min`` one-minute bars. These PARTIAL bars are KEPT (they carry a
valid OHLC and a valid closing spread); dropping them would discard most
session-edge and off-peak data. Each output bar records ``n_minutes`` (the
count of source minutes) so partial bars can be measured and, if a caller
wishes, filtered downstream.

Causality: an output bar's OHLC depends only on 1-minute bars inside its own
window (all at timestamps <= the window end), so resampling introduces no
lookahead. The bar carries the window-start label; the backtest engine then
executes a signal from bar t-1 at the OPEN of bar t (the first price of the
next window), preserving the t+1 execution rule.

Session bars (daily / weekly)
-----------------------------
A calendar-day boundary is wrong for FX: the market runs continuously from
Sunday evening to Friday evening New York time, and the industry day boundary
is the **17:00 America/New_York close** -- the same anchor the `quality`
module already uses to classify daily-reset and weekend gaps. `resample_sessions`
therefore assigns every 1-minute bar to a trading session by shifting NY local
wall time forward by ``24 - 17 = 7`` hours, so 17:00 NY becomes the next
session's 00:00:

  daily  : [D-1 17:00 NY, D 17:00 NY)  -- labeled at its START (in UTC)
  weekly : [Sun 17:00 NY, Sat 17:00 NY) -- the shifted Mon-anchored week,
           which in practice is Sunday's open through Friday's 17:00 close
           because the market is shut for the remainder of that span.
  2h / 4h: SUB-bars that nest inside a daily session, cut every 2 (or 4) NY
           wall-clock hours from the 17:00 NY open. A full session holds
           exactly 12 (2h) or 6 (4h) of them, with boundaries at 17:00,
           19:00, 21:00 ... NY (2h) or 17:00, 21:00, 01:00, 05:00, 09:00,
           13:00 NY (4h). Because the cut is on NY WALL time, the grid stays
           pinned to the session across DST: the sub-bar spanning a
           spring-forward holds one real hour, the one spanning a fall-back
           holds three, and neither boundary drifts off the session open.
           A plain clock resample (`resample_merged(df, "4h")`) would instead
           anchor to midnight UTC, letting bars straddle the 17:00 rollover
           and shift against the session twice a year.

Using NY *wall* time (not a fixed UTC offset) makes the boundary DST-correct
automatically: 17:00 NY is 21:00 UTC in summer and 22:00 UTC in winter, and
the labels follow.

Gaps: sessions are formed by GROUPING the timestamps that exist, so a session
with no bars at all (holiday, weekend) simply never appears -- no bar is ever
fabricated across a gap, and no bar ever spans one, because the weekend is
the [Fri 17:00, Sun 17:00) part of a week that contains no minutes. As with
the clock resampler, PARTIAL sessions (a holiday half-day, or the first/last
session of a data window) are KEPT and counted via ``n_minutes``.
"""

from __future__ import annotations

import pandas as pd

from fxalgo.data.quality import FX_CLOSE_HOUR_NY, NY_TZ

SIDE_OHLC: dict[str, tuple[str, str, str, str]] = {
    "bid": ("open_bid", "high_bid", "low_bid", "close_bid"),
    "ask": ("open_ask", "high_ask", "low_ask", "close_ask"),
}

# Shift that moves the FX close (17:00 NY) onto the next session's midnight.
SESSION_SHIFT = pd.Timedelta(hours=24 - FX_CLOSE_HOUR_NY)
SESSION_PERIODS: tuple[str, ...] = ("daily", "weekly")
# Session-anchored INTRADAY periods: sub-bars nested inside a daily session,
# cut every N NY wall-clock hours from the 17:00 NY open. Must divide 24.
SESSION_INTRADAY_HOURS: dict[str, int] = {"2h": 2, "4h": 4}
# Everything `resample_sessions` / `session_keys` accept.
ALL_SESSION_PERIODS: tuple[str, ...] = SESSION_PERIODS + tuple(SESSION_INTRADAY_HOURS)
# How long after its start label each session RUNS, in NY wall-clock time: a
# daily bar ends at the next 17:00 NY; a weekly bar opens Sunday 17:00 NY and
# ends at Friday 17:00 NY, five wall days later; an intraday sub-bar ends N
# wall hours after its own label.
SESSION_SPAN_DAYS: dict[str, int] = {"daily": 1, "weekly": 5}
SESSION_SPAN: dict[str, pd.Timedelta] = {
    **{p: pd.Timedelta(days=d) for p, d in SESSION_SPAN_DAYS.items()},
    **{p: pd.Timedelta(hours=h) for p, h in SESSION_INTRADAY_HOURS.items()},
}


def bars_per_window(freq: str) -> int:
    """Number of 1-minute bars in one `freq` window (e.g. 15 for '15min')."""
    return int(pd.Timedelta(freq) / pd.Timedelta("1min"))


def resample_merged(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Resample a 1-min merged bid/ask frame (loader output) to `freq`.

    Input: DatetimeIndex-ed frame with the eight side-OHLC columns
    (open_bid..close_ask). Output: the same eight columns aggregated onto
    clean `freq` windows, plus recomputed mid_open/high/low/close and
    spread_close, and an `n_minutes` count. Empty windows are dropped.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_merged requires a DatetimeIndex")
    missing = [c for s in SIDE_OHLC.values() for c in s if c not in df.columns]
    if missing:
        raise KeyError(f"resample_merged requires side-OHLC columns; missing {missing}")

    g = df.resample(freq, label="left", closed="left")
    data: dict[str, pd.Series] = {}
    for _side, (o, h, l, c) in SIDE_OHLC.items():
        data[o] = g[o].first()
        data[h] = g[h].max()
        data[l] = g[l].min()
        data[c] = g[c].last()
    out = pd.DataFrame(data)
    out["n_minutes"] = g[SIDE_OHLC["bid"][3]].count().astype("int64")
    out = out[out["n_minutes"] > 0].copy()

    for col in ("open", "high", "low", "close"):
        out[f"mid_{col}"] = (out[f"{col}_bid"] + out[f"{col}_ask"]) / 2.0
    out["spread_close"] = (out["close_ask"] - out["close_bid"]).clip(lower=0.0)

    out.index.name = df.index.name or "timestamp"
    return out


def session_keys(index: pd.DatetimeIndex, period: str = "daily") -> pd.DatetimeIndex:
    """Trading-session key for every timestamp (naive, NY-shifted).

    Each timestamp is converted to New York wall time and shifted forward by
    7 hours, so the 17:00 NY close lands on the next session's midnight. The
    key is then that shifted day (daily), the Monday of that shifted week
    (weekly, = the Sunday 17:00 NY open), or the shifted day plus the
    N-wall-hour bucket within it (the "2h" / "4h" intraday periods).

    Returned keys are NAIVE timestamps -- they are grouping labels, not
    instants. Use `session_label_utc` to convert them back to real UTC
    session-start timestamps.
    """
    if period not in ALL_SESSION_PERIODS:
        raise ValueError(f"period must be one of {ALL_SESSION_PERIODS}, got {period!r}")
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("session_keys requires a DatetimeIndex")
    if index.tz is None:
        raise ValueError("session_keys requires a tz-aware index (UTC)")

    shifted = index.tz_convert(NY_TZ).tz_localize(None) + SESSION_SHIFT
    day = shifted.normalize()
    if period == "daily":
        return day
    if period == "weekly":
        return day - pd.to_timedelta(day.weekday, unit="D")
    # Intraday: bucket the wall-clock offset into the session into N-hour steps.
    hours = SESSION_INTRADAY_HOURS[period]
    offset_h = (shifted - day) // pd.Timedelta(hours=1)
    return day + pd.to_timedelta((offset_h // hours) * hours, unit="h")


def session_label_utc(keys: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert naive session keys back to their UTC session-START timestamps.

    The inverse of the shift in `session_keys`: key midnight - 7h = 17:00 NY
    on the previous calendar day, localized to New York and expressed in UTC
    (21:00 UTC in EDT, 22:00 UTC in EST).

    A 17:00 label is never inside a DST transition. The intraday grids do put
    a boundary on the transition hours (a 2h grid has an 01:00 NY label), so
    the localization resolves an ambiguous label to the FIRST (DST) occurrence
    -- the true start of that bar -- and shifts a nonexistent one forward.
    In FX practice these never carry data anyway: both US transitions happen
    on a Sunday morning, inside the Sat-17:00 session, when the market is shut.
    """
    ny_wall = pd.DatetimeIndex(keys) - SESSION_SHIFT
    return ny_wall.tz_localize(
        NY_TZ, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")


def session_close_utc(labels: pd.DatetimeIndex, period: str = "daily") -> pd.DatetimeIndex:
    """The UTC instant at which each labeled session CLOSES.

    A daily bar labeled at 17:00 NY closes at the next 17:00 NY (one wall-clock
    day later); a weekly bar labeled Sunday 17:00 NY closes at Friday 17:00 NY,
    five wall-clock days later; a 2h/4h sub-bar closes that many wall hours
    after its own label. Arithmetic is done in NY WALL time so a DST
    transition inside the session does not move the close off the session grid.

    This is what makes a session bar usable downstream without lookahead: its
    signal must not be visible on a finer timeline before this instant.
    """
    if period not in ALL_SESSION_PERIODS:
        raise ValueError(f"period must be one of {ALL_SESSION_PERIODS}, got {period!r}")
    idx = pd.DatetimeIndex(labels)
    if idx.tz is None:
        raise ValueError("session_close_utc requires a tz-aware index (UTC)")
    ny_wall = idx.tz_convert(NY_TZ).tz_localize(None)
    closes = ny_wall + SESSION_SPAN[period]
    return closes.tz_localize(
        NY_TZ, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")


def resample_sessions(df: pd.DataFrame, period: str = "daily") -> pd.DataFrame:
    """Resample a 1-min merged bid/ask frame onto FX trading sessions.

    `period="daily"` builds one bar per 17:00-NY-to-17:00-NY trading day;
    `period="weekly"` builds one bar per Sunday-17:00-NY-to-Friday-17:00-NY
    trading week; `period="2h"` / `"4h"` build sub-bars nested inside the
    daily session, cut every 2 / 4 NY wall-clock hours from its 17:00 open
    (12 / 6 per full session). Output columns match `resample_merged`: the
    eight side-OHLC columns, recomputed mid OHLC and `spread_close` (at the
    bar's close), and `n_minutes`. The index is the bar START in UTC.

    Windows with no data (weekends, holidays, the daily-reset hole) never
    appear -- no bar is fabricated and none spans a gap, because bars are
    formed by grouping the timestamps that exist. Partial windows are kept
    and counted via `n_minutes`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample_sessions requires a DatetimeIndex")
    missing = [c for s in SIDE_OHLC.values() for c in s if c not in df.columns]
    if missing:
        raise KeyError(f"resample_sessions requires side-OHLC columns; missing {missing}")

    df = df.sort_index()  # first/last aggregation is order-dependent
    keys = session_keys(df.index, period)
    g = df.groupby(keys, sort=True)

    data: dict[str, pd.Series] = {}
    for _side, (o, h, l, c) in SIDE_OHLC.items():
        data[o] = g[o].first()
        data[h] = g[h].max()
        data[l] = g[l].min()
        data[c] = g[c].last()
    out = pd.DataFrame(data)
    out["n_minutes"] = g.size().astype("int64")

    for col in ("open", "high", "low", "close"):
        out[f"mid_{col}"] = (out[f"{col}_bid"] + out[f"{col}_ask"]) / 2.0
    out["spread_close"] = (out["close_ask"] - out["close_bid"]).clip(lower=0.0)

    out.index = session_label_utc(pd.DatetimeIndex(out.index))
    out.index.name = df.index.name or "timestamp"
    return out


def side_bars(resampled: pd.DataFrame, side: str) -> pd.DataFrame:
    """Extract one side's OHLC as a storage-ready frame.

    Returns columns [timestamp, open, high, low, close, volume] matching the
    `BarsStorage` schema (volume = -1.0, the "no volume" placeholder). Used to
    persist the resampled bars separately from the 1-min raw data.
    """
    if side not in SIDE_OHLC:
        raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
    o, h, l, c = SIDE_OHLC[side]
    out = (
        resampled[[o, h, l, c]]
        .rename(columns={o: "open", h: "high", l: "low", c: "close"})
        .copy()
    )
    out["volume"] = -1.0
    out = out.reset_index()
    ts_col = "timestamp" if "timestamp" in out.columns else out.columns[0]
    out = out.rename(columns={ts_col: "timestamp"})
    return out[["timestamp", "open", "high", "low", "close", "volume"]]
