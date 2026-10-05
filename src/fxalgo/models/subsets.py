"""The four PRE-COMMITTED feature subsets. Fixed before any model ran.

Four subsets is four looks at the same 2,015 independent test observations. The
list is closed deliberately: adding a fifth after seeing four results, or
re-cutting one that looked promising, makes the winner more biased than its
error bar admits.

  A) ALL              every feature column.
  B) STRATEGY_CORE    the decomposition of the two working configs. Asks the
                      central question of the ML phase: does a booster beat the
                      hand-built rules given the SAME inputs?
  C) MTF_BOLLINGER    the multi-timeframe band structure plus excursion history.
                      Asks whether the one measured structural finding -- 15-min
                      and daily band position orthogonal at r = -0.002 -- carries
                      a model on its own.
  D) NO_TIME          everything except the eight clock columns. Asks how much
                      apparent performance is the calendar. Close to A means the
                      clock is not doing the work; far below A means it is.
"""

from __future__ import annotations

import pandas as pd

TIME_COLUMNS: tuple[str, ...] = (
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "session_tokyo", "session_london", "session_ny", "session_london_ny_overlap",
)

# B) The two working configs, decomposed. The first eleven are the specified
# core; the last two are added because the configs cannot otherwise be
# expressed -- see `STRATEGY_CORE_NOTES`.
STRATEGY_CORE: tuple[str, ...] = (
    "bb_pct_b",                 # 15-min band position: the band touch itself
    "bb_position_daily",        # the daily shadow's continuous position
    "rsi_14",                   # RSI confluence, 15-min
    "rsi_14_daily",
    "regime_daily",             # the shadow's three-state signal
    "regime_daily_active",      # ... and the non-directional form cell D uses
    "gate_cell_d",              # the whole cell D entry condition
    "gate_stack",               # the whole filter-stack entry condition
    "spread_pctile_trailing",   # causal stand-in for the spread level
    "atr_bps_15m",              # scale-free volatility at the decision bar
    "bb_width_atr_daily",
    "spread_ratio_60",          # the spread the cost filter and gate act on
    "spread_zscore_60",
)

STRATEGY_CORE_NOTES = """\
ADDED beyond the specified eleven: spread_ratio_60 and spread_zscore_60. The
filter stack's cost filter compares the band-to-mean target against a
round-trip cost that is dominated by the SPREAD, and its spread gate is an
absolute spread ceiling. Neither is expressible from the other eleven: the
absolute spread level (`spread_close`) is a price-unit column and was dropped
from the training file, so the only surviving views of it are the two relative
measures and `spread_pctile_trailing`. Without them the subset cannot express
the config it is supposed to decompose.

NOT added, deliberately: `bb_pos_spread_15m_daily`. It is the explicit
15m-vs-daily band-position difference, and while cell D does combine those two
timescales, the combination is already carried by `gate_cell_d`. Handing the
model a pre-formed cross-timeframe interaction would be giving it MORE than the
hand-built rule had, which is the opposite of what this subset asks.

NOT available: `bb_width_atr_15m` does not exist -- `features.htf` computes
`bb_width_atr` for the six HIGHER timeframes only. The 15-min band width in
price units (`bb_upper - bb_lower`) is a price-unit column and was dropped.
`atr_bps_15m` is the closest scale-free 15-minute width proxy in the frame.
"""

_MTF_STEMS = ("bars_above_band", "bars_below_band", "frac_above_band",
              "frac_below_band", "bars_since_band_cross",
              "max_stretch_current_excursion")


def _mtf_bollinger(features: list[str]) -> list[str]:
    """C) band structure across timeframes, plus the excursion history."""
    cols = [c for c in features if c == "bb_pct_b" or c.startswith("bb_position_")]
    cols += [c for c in features if c.startswith("bb_width_atr_")]
    cols += [c for c in features if c.startswith("bb_pos_spread_15m_")]
    cols += [c for c in features if c in ("tf_stretch_count", "tf_stretch_max")]
    cols += [c for c in features if c.startswith(_MTF_STEMS)]
    return cols


def build(features: list[str]) -> dict[str, list[str]]:
    """The four subsets, resolved against the columns actually present."""
    missing = [c for c in STRATEGY_CORE if c not in features]
    if missing:
        raise KeyError(f"STRATEGY_CORE names columns not in the frame: {missing}")
    return {
        "A_all": list(features),
        "B_strategy_core": list(STRATEGY_CORE),
        "C_mtf_bollinger": _mtf_bollinger(features),
        "D_no_time": [c for c in features if c not in set(TIME_COLUMNS)],
    }


def feature_columns(frame: pd.DataFrame, non_features: set[str]) -> list[str]:
    """Feature columns of the training frame, in frame order."""
    return [c for c in frame.columns if c not in non_features]
