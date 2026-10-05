"""Cross-timeframe alignment that only sees CLOSED higher-timeframe bars.

`align_completed_series` projects any higher-timeframe series (indexed by bar
START) onto a finer timeline, re-stamping each value at the instant its bar
CLOSES and forward-filling from there. A bar merely covering time t is still
in progress and is invisible at t, so nothing from the future leaks in.

`align_completed_signal` wraps it for {-1, 0, +1} signals -- used to project
the daily Bollinger "shadow" onto 15-minute bars as the regime filter.

Fixed clock widths ("15min", "30min", "1h", ...) close at start + width;
session bars ("daily", "weekly", and the session-anchored "2h"/"4h") close at
the 17:00 New York rollover grid, DST-correct (`data.resample.session_close_utc`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxalgo.data.resample import SESSION_PERIODS, session_close_utc


def align_completed_series(
    htf_series: pd.Series,
    target_index: pd.DatetimeIndex,
    freq: str,
) -> pd.Series:
    """Project ANY higher-TF series onto `target_index` using only CLOSED bars.

    This is the causality primitive: each higher-TF value is re-stamped at the
    instant its bar CLOSES and forward-filled from there, so the value at
    target timestamp `t` comes from the last bar that finished at or before
    `t`. A bar merely COVERING `t` is still in progress and is invisible.

    Dtype-preserving and NOT filled: timestamps before the first completed bar
    are NaN. `align_completed_signal` wraps this for {-1, 0, +1} signals;
    label/feature code uses it directly for float series such as ATR.
    """
    if not isinstance(htf_series.index, pd.DatetimeIndex):
        raise TypeError("align_completed_series requires a DatetimeIndex on htf_series")
    if not isinstance(target_index, pd.DatetimeIndex):
        raise TypeError("align_completed_series requires a DatetimeIndex target")
    if not htf_series.index.is_monotonic_increasing:
        raise ValueError("htf_series must be sorted by timestamp")

    if freq in SESSION_PERIODS:
        closed_at = session_close_utc(htf_series.index, freq)
    else:
        closed_at = htf_series.index + pd.Timedelta(freq)
    at_close = pd.Series(htf_series.to_numpy(), index=closed_at)
    return at_close.reindex(target_index, method="ffill")


def align_completed_signal(
    htf_signal: pd.Series,
    target_index: pd.DatetimeIndex,
    freq: str,
) -> pd.Series:
    """Project a higher-TF SIGNAL onto `target_index` using only CLOSED bars.

    Parameters
    ----------
    htf_signal:
        Signal series indexed by higher-TF bar START timestamps (the label
        convention of `fxalgo.data.resample`), values in {-1, 0, +1}.
    target_index:
        The fine timeline (e.g. the 1-minute feature frame's index).
    freq:
        The higher timeframe's bar width -- either a fixed clock width
        ("15min", "30min", "1h") or a trading session ("daily", "weekly").
        Used to turn each bar's start label into its close instant.

    Returns
    -------
    int8 Series on `target_index`: at minute t, the signal of the last
    higher-TF bar whose window closed at or before t. 0 before the first
    completed bar.
    """
    aligned = align_completed_series(htf_signal, target_index, freq)
    return pd.Series(
        aligned.fillna(0).to_numpy().astype(np.int8),
        index=target_index,
        name=htf_signal.name or "htf_bias",
        dtype=np.int8,
    )
