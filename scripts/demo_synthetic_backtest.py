"""Run the full feature pipeline and backtest engine on SYNTHETIC data.

No market data or broker connection is needed: this generates a random
15-minute bid/ask series (a mean-reverting random walk with a constant
spread), builds the same feature frame used in the research, and runs the
baseline Bollinger strategy and the RSI-confluence variant through the
same engine (fills at the next bar's open, real bid/ask, the default
commission model). It exists to show the code runs end to end; the numbers mean
nothing about real markets.

    uv run python scripts/demo_synthetic_backtest.py --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxalgo.backtest import engine, metrics
from fxalgo.features.build import apply_pipeline
from fxalgo.strategies.bollinger_reversion import BollingerReversion
from fxalgo.strategies.bollinger_rsi_confluence import BollingerRsiConfluence


def synthetic_bars(n: int, seed: int) -> pd.DataFrame:
    """Merged bid/ask OHLC frame in the loader's column layout."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    idx = idx[idx.dayofweek < 5][: n]  # weekdays only, like FX
    m = len(idx)
    # Ornstein-Uhlenbeck-style log price around 1.10: mild mean reversion.
    x = np.zeros(m)
    for t in range(1, m):
        x[t] = 0.995 * x[t - 1] + rng.normal(0, 0.0006)
    close = 1.10 * np.exp(x)
    open_ = np.r_[close[0], close[:-1]]
    wiggle = np.abs(rng.normal(0, 0.0003, m))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    half = 0.00004  # 0.8 pip total spread
    df = pd.DataFrame(index=idx)
    for side, sgn in (("bid", -1), ("ask", +1)):
        df[f"open_{side}"] = open_ + sgn * half
        df[f"high_{side}"] = high + sgn * half
        df[f"low_{side}"] = low + sgn * half
        df[f"close_{side}"] = close + sgn * half
    for col in ("open", "high", "low", "close"):
        df[f"mid_{col}"] = (df[f"{col}_bid"] + df[f"{col}_ask"]) / 2.0
    df["spread_close"] = df["close_ask"] - df["close_bid"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bars", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    feats = apply_pipeline(synthetic_bars(args.bars, args.seed))
    print(f"synthetic frame: {len(feats):,} bars x {feats.shape[1]} columns (seed {args.seed})")
    for label, strat in [
        ("Bollinger baseline", BollingerReversion()),
        ("Bollinger + RSI confluence", BollingerRsiConfluence(use_rsi=True)),
    ]:
        result = engine.run(feats, strat)
        rep = metrics.compute(result, feats)
        print(
            f"{label:28s} trades={rep.n_trades:5d}  win={100 * rep.win_rate:5.1f}%  "
            f"return={100 * rep.return_pct_of_notional:+7.2f}% of notional  sharpe={rep.sharpe:+.2f}  "
            f"maxDD={100 * rep.max_drawdown_pct_of_notional:.2f}% of notional"
        )


if __name__ == "__main__":
    main()
