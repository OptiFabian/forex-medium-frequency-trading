"""Triple-barrier labels: which of an upper, lower or time barrier is hit first.

LABELS ONLY. This module builds the target; it does not build features, sample
events, split data, or model anything.

Event and anchor
----------------
One event per 1-minute bar. Event ``t`` is anchored at **mid open(t+1)** -- the
same price the backtest engine fills at. Anchoring on ``close(t)`` while the
engine fills at ``open(t+1)`` would introduce a systematic bias, so the two are
deliberately identical.

Barriers
--------
``atr_at_entry`` is ``atr_14`` from the **last 15-minute bar CLOSED at or
before t**. A 15-min bar merely COVERING t is still in progress and must not be
used -- an off-by-one here yields a label that partly knows its own future.
This is enforced by `strategies.multi_timeframe.align_completed_series`, the
same causality primitive the daily-bias code uses, rather than a reimplementation.

The ATR is read at the DECISION bar t, while the anchor price is open(t+1):
decide on information available at t, fill at t+1.

``atr_at_entry`` is FIXED for the life of the window -- it does not update as
the window progresses::

    upper_k = anchor + k * atr_at_entry
    lower_k = anchor - k * atr_at_entry

for k in {1.0, 1.5, 2.0, 2.5, 3.0}. All barriers are on MID prices. No bid/ask
and no cost adjustment at the label layer: costs are enforced downstream at the
trade layer by prior decision.

Path scan
---------
Bars ``t+1 .. t+HORIZON`` inclusive (HORIZON = 240), so the anchor bar itself is
exposed to its own range. ``label = +1`` if the upper barrier is touched first,
``-1`` if the lower is, ``0`` if neither is touched inside the horizon.

``bars_to_touch`` is 1-based: 1 means the touch happened on the anchor bar.

THE HORIZON IS 240 BARS OF TRADING, NOT 240 WALL-CLOCK MINUTES. A window
opening on Friday afternoon continues into Sunday's session until 240 traded
bars have elapsed; windows are allowed to span gaps and such events are never
excluded.

Same-bar ambiguity
------------------
When one minute bar's high crosses ``upper_k`` AND its low crosses ``lower_k``,
the touch ORDER is unknowable from OHLC. Those rows are flagged in
``ambiguous_k`` and labelled ``-1`` -- the ADVERSE outcome -- so the dataset is
never silently optimistic. They are kept, not dropped, so they can be excluded
or relabelled later.

Gap and calendar columns
------------------------
These record THAT a gap will occur, which is known from the calendar at entry.
They are derived from the CLOCK ALONE -- never from the future index -- so they
introduce no lookahead. They deliberately say nothing about how long a gap
lasts, because closure length is not always knowable in advance.

``gap_type`` is one of ``none`` / ``daily_reset`` / ``weekend``. **A `holiday`
category was specified but is NOT emitted**: an unscheduled market closure is
not knowable from the calendar at entry, so producing it would require reading
the future index and would be exactly the lookahead this module is built to
avoid. Realized-vs-predicted gap incidence is reported as a diagnostic instead.

``bars_to_gap`` equals ``minutes_to_close`` by construction on a 1-minute
series (one traded bar per elapsed minute); both are kept because both were
specified, and they would diverge only on a coarser bar series.

Tail of the sample
------------------
The final ``HORIZON`` events cannot be labelled -- their windows run off the end
of the data. They are emitted as NaN rather than silently truncated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.data.quality import FX_CLOSE_HOUR_NY, NY_TZ
from fxalgo.data.resample import session_keys
from fxalgo.strategies.multi_timeframe import align_completed_series

HORIZON = 240
KS: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)
ATR_COL = "atr_14"
ATR_FREQ = "15min"
REQUIRED = ("mid_open", "mid_high", "mid_low", "mid_close")


def _k_tag(k: float) -> str:
    """Column suffix for a barrier width: 1.0 -> '1.0', 2.5 -> '2.5'."""
    return f"{k:g}" if k != int(k) else f"{int(k)}.0"


def label_columns(ks: tuple[float, ...] = KS) -> list[str]:
    """Every column `build_labels` emits, in order."""
    cols = ["anchor_price", "atr_at_entry"]
    for k in ks:
        tag = _k_tag(k)
        cols += [f"label_{tag}", f"bars_to_touch_{tag}",
                 f"touch_price_{tag}", f"ambiguous_{tag}"]
    cols += ["ret_at_time_barrier_bps", "spans_gap", "gap_type",
             "bars_to_gap", "is_friday", "minutes_to_close"]
    return cols


def atr_at_entry(index: pd.DatetimeIndex, atr_15m: pd.Series) -> pd.Series:
    """ATR from the last 15-minute bar CLOSED at or before each 1-min timestamp."""
    return align_completed_series(atr_15m, index, ATR_FREQ)


def next_ny_close(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The next 17:00 America/New_York instant STRICTLY after each timestamp.

    Pure clock arithmetic in NY wall time, so it is DST-correct and needs no
    knowledge of which bars exist in the future.
    """
    ny = index.tz_convert(NY_TZ)
    wall = ny.tz_localize(None)
    same_day_close = wall.normalize() + pd.Timedelta(hours=FX_CLOSE_HOUR_NY)
    # If we are at or past today's 17:00, the next close is tomorrow's.
    close_wall = pd.DatetimeIndex(
        np.where(wall < same_day_close, same_day_close,
                 same_day_close + pd.Timedelta(days=1))
    )
    return close_wall.tz_localize(
        NY_TZ, ambiguous=True, nonexistent="shift_forward"
    ).tz_convert("UTC")


def calendar_columns(
    anchor_time: pd.DatetimeIndex, horizon: int = HORIZON
) -> pd.DataFrame:
    """Clock-only gap/calendar columns for each anchor timestamp (see module doc)."""
    closes = next_ny_close(anchor_time)
    minutes_to_close = ((closes - anchor_time) / pd.Timedelta(minutes=1)).astype("int64")
    # The session an entry belongs to runs 17:00->17:00 NY; a "Friday" session
    # ends at Friday 17:00 NY and is the one followed by the weekend.
    session_day = session_keys(anchor_time, "daily")
    is_friday = session_day.weekday == 4
    spans = minutes_to_close < horizon
    gap_type = np.where(~spans, "none", np.where(is_friday, "weekend", "daily_reset"))
    return pd.DataFrame(
        {
            "spans_gap": spans,
            "gap_type": pd.Categorical(gap_type, categories=["none", "daily_reset", "weekend"]),
            "bars_to_gap": minutes_to_close,
            "is_friday": is_friday,
            "minutes_to_close": minutes_to_close,
        },
        index=anchor_time,
    )


def scan_reference(
    mid_high: np.ndarray, mid_low: np.ndarray, anchor: np.ndarray,
    atr: np.ndarray, ks: tuple[float, ...] = KS, horizon: int = HORIZON,
    events: np.ndarray | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Slow, obviously-correct reference scan. Used only to verify the fast path.

    Walks each event's window bar by bar in Python. `events` selects a subset of
    event indices; None means all (very slow on a full dataset).
    """
    n = len(anchor)
    if events is None:
        events = np.arange(n - horizon)
    out = {k: {"label": np.full(len(events), np.nan),
               "bars": np.full(len(events), np.nan),
               "ambiguous": np.zeros(len(events), dtype=bool)} for k in ks}
    for row, i in enumerate(events):
        a, v = anchor[i], atr[i]
        if not np.isfinite(a) or not np.isfinite(v):
            continue
        for k in ks:
            up, dn = a + k * v, a - k * v
            lab, bars, amb = 0, np.nan, False
            for j in range(1, horizon + 1):
                hit_up = mid_high[i + j] >= up
                hit_dn = mid_low[i + j] <= dn
                if hit_up and hit_dn:
                    lab, bars, amb = -1, j, True
                    break
                if hit_up:
                    lab, bars = 1, j
                    break
                if hit_dn:
                    lab, bars = -1, j
                    break
            out[k]["label"][row] = lab
            out[k]["bars"][row] = bars
            out[k]["ambiguous"][row] = amb
    return out


def _scan_fast(
    mid_high: np.ndarray, mid_low: np.ndarray, anchor: np.ndarray, atr: np.ndarray,
    ks: tuple[float, ...], horizon: int, chunk: int,
) -> dict[float, dict[str, np.ndarray]]:
    """Vectorised scan: ONE pass over the path, all k evaluated per bar.

    Uses a sliding-window VIEW of the 1-min highs/lows (no copy), then a
    cumulative forward max/min along the window axis. Because the running max
    is non-decreasing, the first bar whose running max clears `upper_k` IS the
    first touch of the upper barrier; likewise the running min and `lower_k`.
    That turns "first touch" into one argmax over a boolean, and lets a single
    pair of accumulations serve all five barrier widths.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n_events = len(anchor) - horizon
    sw_high = sliding_window_view(mid_high, horizon)
    sw_low = sliding_window_view(mid_low, horizon)

    res = {k: {"label": np.full(n_events, np.nan),
               "bars": np.full(n_events, np.nan),
               "ambiguous": np.zeros(n_events, dtype=bool)} for k in ks}

    for start in range(0, n_events, chunk):
        stop = min(start + chunk, n_events)
        # window for event i is bars i+1 .. i+horizon  ->  sw[i+1]
        cmax = np.maximum.accumulate(sw_high[start + 1 : stop + 1], axis=1)
        cmin = np.minimum.accumulate(sw_low[start + 1 : stop + 1], axis=1)
        a = anchor[start:stop]
        v = atr[start:stop]
        defined = np.isfinite(a) & np.isfinite(v)

        for k in ks:
            up = (a + k * v)[:, None]
            dn = (a - k * v)[:, None]
            hit_up = cmax >= up
            hit_dn = cmin <= dn
            # cmax/cmin are monotone, so the last column answers "ever touched"
            any_up, any_dn = hit_up[:, -1], hit_dn[:, -1]
            j_up = np.where(any_up, hit_up.argmax(axis=1), horizon + 1)
            j_dn = np.where(any_dn, hit_dn.argmax(axis=1), horizon + 1)

            amb = any_up & any_dn & (j_up == j_dn)
            lab = np.where(j_up < j_dn, 1, np.where(j_dn < j_up, -1, 0))
            lab = np.where(amb, -1, lab)          # ambiguous -> adverse
            touched = any_up | any_dn
            lab = np.where(touched, lab, 0)
            bars = np.where(touched, np.minimum(j_up, j_dn) + 1, np.nan)

            sl = slice(start, stop)
            res[k]["label"][sl] = np.where(defined, lab, np.nan)
            res[k]["bars"][sl] = np.where(defined, bars, np.nan)
            res[k]["ambiguous"][sl] = amb & defined
    return res


def build_labels(
    bars: pd.DataFrame,
    atr_15m: pd.Series,
    *,
    ks: tuple[float, ...] = KS,
    horizon: int = HORIZON,
    chunk: int = 50_000,
) -> pd.DataFrame:
    """Build the triple-barrier label set. One row per input bar.

    `bars` needs mid_open/high/low/close on a sorted 1-minute DatetimeIndex;
    `atr_15m` is ATR indexed by 15-MINUTE bar START (its close-time lag is
    applied here). The last `horizon` rows are emitted with NaN labels.
    """
    missing = [c for c in REQUIRED if c not in bars.columns]
    if missing:
        raise KeyError(f"build_labels requires columns {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("build_labels requires a DatetimeIndex")
    if not bars.index.is_monotonic_increasing:
        raise ValueError("bars must be sorted by timestamp")

    n = len(bars)
    idx = bars.index
    mid_open = bars["mid_open"].to_numpy(dtype="float64")
    mid_high = bars["mid_high"].to_numpy(dtype="float64")
    mid_low = bars["mid_low"].to_numpy(dtype="float64")
    mid_close = bars["mid_close"].to_numpy(dtype="float64")

    # Anchor: mid open of t+1 (the engine's fill price). Last bar has none.
    anchor = np.full(n, np.nan)
    anchor[: n - 1] = mid_open[1:]
    # ATR from the last 15-min bar closed at or before t (the DECISION bar).
    atr = atr_at_entry(idx, atr_15m).to_numpy(dtype="float64")

    out = pd.DataFrame(index=idx)
    out.index.name = idx.name or "timestamp"
    out["anchor_price"] = anchor
    out["atr_at_entry"] = atr

    n_events = n - horizon
    scan = _scan_fast(mid_high, mid_low, anchor, atr, ks, horizon, chunk)
    for k in ks:
        tag = _k_tag(k)
        lab = np.full(n, np.nan)
        bts = np.full(n, np.nan)
        amb = np.zeros(n, dtype=bool)
        lab[:n_events] = scan[k]["label"]
        bts[:n_events] = scan[k]["bars"]
        amb[:n_events] = scan[k]["ambiguous"]
        out[f"label_{tag}"] = lab
        out[f"bars_to_touch_{tag}"] = bts
        # Touch price is the BARRIER LEVEL reached -- the idealised resolution
        # price, not a modelled fill (a bar can gap through it; execution
        # modelling is downstream by prior decision). NaN when nothing touched.
        touch = np.where(lab == 1, anchor + np.array([k]) * atr,
                         np.where(lab == -1, anchor - np.array([k]) * atr, np.nan))
        out[f"touch_price_{tag}"] = np.where(np.isnan(lab), np.nan, touch)
        out[f"ambiguous_{tag}"] = amb

    ret = np.full(n, np.nan)
    if n_events > 0:
        end_close = mid_close[horizon : horizon + n_events]
        ret[:n_events] = (end_close - anchor[:n_events]) / anchor[:n_events] * 1e4
    out["ret_at_time_barrier_bps"] = ret

    # Calendar columns are keyed to the ANCHOR time (t+1), the entry instant.
    anchor_time = idx.copy()
    if n > 1:
        anchor_time = pd.DatetimeIndex(np.concatenate([idx.to_numpy()[1:], idx.to_numpy()[-1:]]))
        anchor_time = anchor_time.tz_localize(None).tz_localize("UTC") if anchor_time.tz is None else anchor_time
    cal = calendar_columns(anchor_time, horizon)
    for col in ("spans_gap", "gap_type", "bars_to_gap", "is_friday", "minutes_to_close"):
        out[col] = cal[col].to_numpy()

    return out[label_columns(ks)]
