"""Draw the README figures from the head-to-head result tables.

Reads `results/test_pooled_cumulative_pct.csv` and `results/test_metrics.csv`
(written by `scripts/head_to_head.py`) and writes two PNGs next to them. Units
are percent of position value and basis points; no currency amounts.

    uv run python scripts/make_figures.py --results results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "USDJPY", "POOLED"]
COLORS = {"Bollinger": "tab:blue", "XGBoost": "tab:orange"}


def equity_curves(results: Path) -> None:
    cum = pd.read_csv(results / "test_pooled_cumulative_pct.csv", parse_dates=["date"])
    x = np.arange(len(cum))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for m in ("Bollinger", "XGBoost"):
        ax.plot(x, cum[m], label=m, color=COLORS[m], lw=1.6)
    # the test set is three separate quarters: mark where each one starts
    gaps = np.flatnonzero(cum["date"].diff().dt.days.fillna(0).to_numpy() > 7)
    starts = np.r_[0, gaps]
    for i, s in enumerate(starts):
        if i:
            ax.axvline(s - 0.5, color="grey", lw=0.8, ls=":")
        end = (starts[i + 1] - 1) if i + 1 < len(starts) else len(cum) - 1
        ax.text((s + end) / 2, 0.98, f"{cum['date'][s]:%b %Y} - {cum['date'][end]:%b %Y}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="grey")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("trading days in the three held-out quarters")
    ax.set_ylabel("cumulative net return\n(% of position value)")
    ax.set_title("Out-of-sample, six pairs equally weighted, after spread and commission", fontsize=10)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(results / "fig1_out_of_sample_equity.png", dpi=150)
    plt.close(fig)


def per_pair(results: Path) -> None:
    m = pd.read_csv(results / "test_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.38
    x = np.arange(len(PAIRS))
    for k, method in enumerate(("Bollinger", "XGBoost")):
        d = m[m.method == method].set_index("pair").loc[PAIRS]
        se = d["mean_net_bps"] / d["t_stat"]
        ax.bar(x + (k - 0.5) * width, d["mean_net_bps"], width, yerr=1.96 * se.abs(),
               capsize=3, label=method, color=COLORS[method])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, PAIRS)
    ax.set_ylabel("mean net return per trade (bps)")
    ax.set_title("Out-of-sample net return per trade, with 95% confidence intervals", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results / "fig2_per_pair_net_bps.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    equity_curves(args.results)
    per_pair(args.results)
    print(f"wrote figures to {args.results}")


if __name__ == "__main__":
    main()
