"""Out-of-sample head-to-head: filtered Bollinger mean reversion vs XGBoost.

Both methods trade the same six pairs on the same 15-minute bars, through the
same engine (signal at the close of bar t, fill at the open of bar t+1 at the
bid/ask, the same commission model on every order leg, 100,000 base units per
trade), and are scored on the same held-out bars.

THE SPLIT is the one fixed before any model was trained (16 three-month chunks
from 2022-05-26; chunks 2, 7 and 12 are the test set). Every free choice is made
on the training side only:

  * Bollinger: the configuration is re-selected on the TRAINING chunks from a
    small grid declared below (the 2x2 regime x RSI factorial, two band widths,
    two RSI thresholds), by the t-statistic of the pooled per-trade net return.
  * XGBoost: the model is the A_all booster (lowest validation log-loss of the
    four pre-committed subsets), trained on the training chunks only, used
    unchanged. Its trading threshold is chosen on the VALIDATION chunk (15),
    which was used for early stopping but never for fitting.

The test chunks are touched exactly once, after both selections are frozen.

Inputs (read-only), under --data-root:
    resampled/15min_features_exc/<PAIR>.parquet   15-min bars + features
    models/A_all.json                              the trained booster
    training/eurusd_split.parquet                  (optional) cross-check of chunk dates

    uv run python scripts/head_to_head.py --data-root data --out results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fxalgo.backtest import engine
from fxalgo.backtest.costs import DEFAULT_COMMISSION
from fxalgo.backtest.level_exits import AtrBarrier
from fxalgo.features.bollinger_utils import rescale_bollinger
from fxalgo.strategies.bollinger_rsi_confluence import BollingerRsiConfluence

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "USDJPY"]
NOTIONAL = 100_000.0
WARMUP = 600
BARS_PER_DAY = 96          # one trading day of 15-minute bars: the embargo
ANNUAL_DAYS = 252

# The pre-committed split: chunk boundaries (UTC) and roles.
CHUNK_EDGES = pd.to_datetime([
    "2022-05-26 00:15", "2022-08-26 00:15", "2022-11-26 00:15", "2023-02-26 00:15",
    "2023-05-26 00:15", "2023-08-26 00:15", "2023-11-26 00:15", "2024-02-26 00:15",
    "2024-05-26 00:15", "2024-08-26 00:15", "2024-11-26 00:15", "2025-02-26 00:15",
    "2025-05-26 00:15", "2025-08-26 00:15", "2025-11-26 00:15", "2026-02-26 00:15",
    "2026-06-03 00:15",
], utc=True)
TEST_CHUNKS = (2, 7, 12)
VALIDATION_CHUNK = 15

# Bollinger grid, declared before looking at any result.
BOLL_GRID = [
    # (label, use_regime, use_rsi, band_k, rsi_lower, rsi_upper)
    (f"{cell} k={k} rsi={lo:g}/{hi:g}" if rsi else f"{cell} k={k}", reg, rsi, k, lo, hi)
    for cell, reg, rsi in (("A", False, False), ("B", True, False), ("C", False, True), ("D", True, True))
    for k in (2.0, 2.5)
    for lo, hi in (((30.0, 70.0), (25.0, 75.0)) if rsi else ((30.0, 70.0),))
]
BOLL_MIN_TRAIN_TRADES = 500
# The research's cell D, chosen on the FULL sample (test included): reported on the
# test set for reference only, never used for selection.
RESEARCH_CONFIG = "D k=2.0 rsi=30/70"

# XGBoost trading rule: enter when |P(up) - P(down)| > theta, hold for the label
# horizon (16 bars = 240 minutes) with take-profit and stop at +/- 2 ATR of the
# decision bar -- the triple barrier the model was trained to predict.
XGB_THETAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
XGB_MIN_VAL_TRADES = 100
XGB_HOLD_BARS = 16
XGB_BARRIER_K = 2.0


# ----------------------------------------------------------------- periods

def period_labels(index: pd.DatetimeIndex) -> np.ndarray:
    """'train' / 'test' / 'val' / 'embargo' / 'pre' for every bar.

    A bar's chunk is the one containing its START. Bars within one trading day
    (96 bars) of a test-chunk boundary, on either side, are embargoed: neither
    side may select or score on them.
    """
    chunk = np.searchsorted(CHUNK_EDGES.values, index.values, side="right") - 1
    lab = np.full(len(index), "train", dtype=object)
    lab[chunk < 0] = "pre"
    lab[chunk >= len(CHUNK_EDGES) - 1] = "pre"
    is_test = np.isin(chunk, TEST_CHUNKS)
    lab[is_test] = "test"
    lab[chunk == VALIDATION_CHUNK] = "val"
    # embargo around every test boundary
    edges = np.flatnonzero(np.diff(is_test.astype(int)) != 0) + 1
    for e in edges:
        lab[max(0, e - BARS_PER_DAY): e + BARS_PER_DAY] = "embargo"
    return lab


# ----------------------------------------------------------------- trades

def trade_frame(result, labels: pd.Series, pair: str, method: str) -> pd.DataFrame:
    t = result.trades.copy()
    if t.empty:
        return pd.DataFrame(columns=["pair", "method", "entry_time", "exit_time", "net_bps", "period"])
    tv = NOTIONAL * t["entry_price"].to_numpy()
    t["net_bps"] = (t["pnl"].to_numpy() - t["commission"].to_numpy()) / tv * 1e4
    t["gross_bps"] = (t["pnl"].to_numpy() + t["spread_cost"].to_numpy()) / tv * 1e4
    t["cost_bps"] = t["gross_bps"] - t["net_bps"]
    # the decision bar is the bar BEFORE the entry bar
    pos = labels.index.get_indexer(t["entry_time"]) - 1
    t["period"] = labels.to_numpy()[np.clip(pos, 0, None)]
    t["pair"], t["method"] = pair, method
    return t[["pair", "method", "direction", "entry_time", "exit_time", "bars_held",
              "exit_reason", "gross_bps", "cost_bps", "net_bps", "period"]]


def bollinger_trades(frame: pd.DataFrame, labels: pd.Series, pair: str, cfg) -> pd.DataFrame:
    label, use_regime, use_rsi, k, lo, hi = cfg
    upper, lower = "bb_upper", "bb_lower"
    if k != 2.0:
        upper, lower = f"bb_upper_k{k}", f"bb_lower_k{k}"
        if upper not in frame.columns:
            rescale_bollinger(frame, k, upper, lower)
    strat = BollingerRsiConfluence(
        use_rsi=use_rsi, use_regime=use_regime, regime_col="regime_daily",
        rsi_lower=lo, rsi_upper=hi, upper_col=upper, lower_col=lower,
        notional=NOTIONAL, commission=DEFAULT_COMMISSION,
    )
    res = engine.run(frame, strat, warmup=WARMUP, notional=NOTIONAL, commission=DEFAULT_COMMISSION)
    return trade_frame(res, labels, pair, f"bollinger[{label}]")


def xgb_signals(score: np.ndarray, valid: np.ndarray, theta: float) -> pd.Series:
    """Enter on |score| > theta, then hold the direction for XGB_HOLD_BARS bars."""
    n = len(score)
    sig = np.zeros(n, dtype=np.int8)
    remaining, d = 0, 0
    for t in range(n):
        if remaining > 0:
            sig[t] = d
            remaining -= 1
            continue
        if valid[t] and abs(score[t]) > theta:
            d = 1 if score[t] > 0 else -1
            sig[t] = d
            remaining = XGB_HOLD_BARS - 1
    return pd.Series(sig, index=None)


def xgb_trades(frame, labels, pair, score, valid, theta) -> pd.DataFrame:
    sig = xgb_signals(score, valid, theta)
    sig.index = frame.index
    res = engine.simulate(
        frame, sig, warmup=WARMUP, notional=NOTIONAL, commission=DEFAULT_COMMISSION,
        level_policy=AtrBarrier(k=XGB_BARRIER_K, atr_col="atr_decision"), max_hold_bars=XGB_HOLD_BARS,
    )
    return trade_frame(res, labels, pair, f"xgboost[theta={theta:g}]")


# ----------------------------------------------------------------- metrics

def _norm_p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def session_date(ts) -> pd.DatetimeIndex:
    """FX trading day: the New York 17:00 rollover starts the next day (Mon-Fri only)."""
    ts = pd.DatetimeIndex(pd.to_datetime(ts))
    return (ts.tz_convert("America/New_York") + pd.Timedelta(hours=7)).tz_localize(None).normalize()


def daily_returns(trades: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Net return per trading day (fraction of position value), by exit day; 0 on no-exit days."""
    if trades.empty:
        return pd.Series(0.0, index=days)
    d = trades.groupby(session_date(trades["exit_time"]))["net_bps"].sum() / 1e4
    return d.reindex(days, fill_value=0.0)


def summarize(trades: pd.DataFrame, daily: pd.Series) -> dict:
    x = trades["net_bps"].to_numpy(dtype=float)
    cum = daily.cumsum()
    sd = daily.std(ddof=1)
    return {
        "trades": int(len(x)),
        "mean_net_bps": float(x.mean()) if len(x) else float("nan"),
        "t_stat": tstat(x),
        "sharpe_ann": float(daily.mean() / sd * math.sqrt(ANNUAL_DAYS)) if sd > 0 else float("nan"),
        "max_dd_pct": float(100 * (cum - cum.cummax()).min()) if len(cum) else float("nan"),
        "total_return_pct": float(100 * cum.iloc[-1]) if len(cum) else float("nan"),
        "hit_rate_pct": float(100 * (x > 0).mean()) if len(x) else float("nan"),
        "mean_gross_bps": float(trades["gross_bps"].mean()) if len(x) else float("nan"),
        "mean_cost_bps": float(trades["cost_bps"].mean()) if len(x) else float("nan"),
    }


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return float(z), _norm_p_two_sided(z)


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    d = (a - b).to_numpy(dtype=float)
    z = tstat(d)
    return z, _norm_p_two_sided(z)


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--save-trades", action="store_true", help="also write every test trade to test_trades.csv")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    split_path = args.data_root / "training" / "eurusd_split.parquet"
    if split_path.exists():  # cross-check the hard-coded chunk dates against the saved split
        s = pd.read_parquet(split_path, columns=["chunk_id", "chunk_start", "split"])
        starts = s[s.chunk_id >= 0].groupby("chunk_id")["chunk_start"].first().sort_index()
        assert (starts.values == CHUNK_EDGES[:-1].values).all(), "chunk dates differ from the saved split"
        tests = sorted(s[s.split == "test"].chunk_id.unique().tolist())
        assert tuple(tests) == TEST_CHUNKS, tests
        print("chunk dates and test chunks match the saved split")

    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(args.data_root / "models" / "A_all.json"))
    feats = booster.feature_names
    best_it = int(booster.attr("best_iteration")) if booster.attr("best_iteration") else None

    frames, labels, scores, valids = {}, {}, {}, {}
    for pair in PAIRS:
        f = pd.read_parquet(args.data_root / "resampled" / "15min_features_exc" / f"{pair}.parquet")
        f["atr_decision"] = f["atr_14"].shift(1)
        frames[pair] = f
        labels[pair] = pd.Series(period_labels(f.index), index=f.index)
        x = f[feats]
        kw = {"iteration_range": (0, best_it + 1)} if best_it is not None else {}
        p = booster.predict(xgb.DMatrix(x, feature_names=feats), **kw)
        scores[pair] = p[:, 2] - p[:, 0]          # classes are (-1, 0, +1)
        valids[pair] = np.isfinite(x.to_numpy(dtype=float)).all(axis=1) & np.isfinite(f["atr_decision"].to_numpy())
        print(f"{pair}: {len(f):,} bars; periods {labels[pair].value_counts().to_dict()}")

    # ---------------- 1. Bollinger: select on TRAIN
    rows, boll_all = [], {}
    for cfg in BOLL_GRID:
        tr = pd.concat([bollinger_trades(frames[p], labels[p], p, cfg) for p in PAIRS])
        boll_all[cfg[0]] = tr
        sel = tr[tr.period.isin(["train", "val"])]   # the whole training side
        rows.append({"config": cfg[0], "train_trades": len(sel),
                     "train_mean_net_bps": sel.net_bps.mean(), "train_t": tstat(sel.net_bps)})
        print(f"  bollinger {cfg[0]:<22} train n={len(sel):5d} mean={sel.net_bps.mean():+.3f} t={tstat(sel.net_bps):+.2f}")
    grid = pd.DataFrame(rows)
    eligible = grid[grid.train_trades >= BOLL_MIN_TRAIN_TRADES]
    boll_choice = eligible.sort_values("train_t", ascending=False).iloc[0]["config"]
    grid["selected"] = grid.config == boll_choice
    grid.round(4).to_csv(out / "selection_bollinger_train.csv", index=False)
    print(f"SELECTED Bollinger (train): {boll_choice}")

    # ---------------- 2. XGBoost: threshold on VALIDATION
    rows, xgb_all = [], {}
    for th in XGB_THETAS:
        tr = pd.concat([xgb_trades(frames[p], labels[p], p, scores[p], valids[p], th) for p in PAIRS])
        xgb_all[th] = tr
        sel = tr[tr.period == "val"]
        rows.append({"theta": th, "val_trades": len(sel),
                     "val_mean_net_bps": sel.net_bps.mean(), "val_t": tstat(sel.net_bps)})
        print(f"  xgboost theta={th:<5} val n={len(sel):5d} mean={sel.net_bps.mean():+.3f} t={tstat(sel.net_bps):+.2f}")
    vgrid = pd.DataFrame(rows)
    elig = vgrid[vgrid.val_trades >= XGB_MIN_VAL_TRADES]
    theta = float(elig.sort_values("val_mean_net_bps", ascending=False).iloc[0]["theta"])
    vgrid["selected"] = vgrid.theta == theta
    vgrid.round(4).to_csv(out / "selection_xgboost_validation.csv", index=False)
    print(f"SELECTED XGBoost theta (validation): {theta}")

    # ---------------- 3. TEST, touched once
    boll_tr = boll_all[boll_choice]
    xgb_tr = xgb_all[theta]
    boll_test, xgb_test = boll_tr[boll_tr.period == "test"].copy(), xgb_tr[xgb_tr.period == "test"].copy()
    ref_tr = boll_all[RESEARCH_CONFIG]
    ref_test = ref_tr[ref_tr.period == "test"].copy()
    test_days = pd.DatetimeIndex(sorted(set().union(*[
        set(session_date(labels[p].index[labels[p].to_numpy() == "test"])) for p in PAIRS])))

    res_rows, dailies = [], {}
    sets = [("Bollinger", boll_test), ("XGBoost", xgb_test)]
    if boll_choice != RESEARCH_CONFIG:
        sets.append(("Bollinger_research_cellD_reference", ref_test))
    for method, tr in sets:
        per_pair = {}
        for p in PAIRS:
            tp = tr[tr.pair == p]
            d = daily_returns(tp, test_days)
            per_pair[p] = d
            res_rows.append({"method": method, "pair": p, **summarize(tp, d)})
        pooled_daily = pd.concat(per_pair, axis=1).mean(axis=1)   # equal weight across pairs
        dailies[method] = pooled_daily
        res_rows.append({"method": method, "pair": "POOLED", **summarize(tr, pooled_daily)})
    res = pd.DataFrame(res_rows)

    diff_rows = []
    for p in PAIRS + ["POOLED"]:
        a = boll_test if p == "POOLED" else boll_test[boll_test.pair == p]
        b = xgb_test if p == "POOLED" else xgb_test[xgb_test.pair == p]
        zw, pw = welch(a.net_bps, b.net_bps)
        if p == "POOLED":
            zd, pd_ = paired(dailies["Bollinger"], dailies["XGBoost"])
        else:
            zd, pd_ = paired(daily_returns(a, test_days), daily_returns(b, test_days))
        diff_rows.append({"pair": p, "diff_mean_net_bps": a.net_bps.mean() - b.net_bps.mean(),
                          "welch_z_per_trade": zw, "welch_p": pw,
                          "paired_daily_t": zd, "paired_daily_p": pd_})
    diff = pd.DataFrame(diff_rows)

    # The same frozen choices, scored on the side they were selected on.
    side_rows = []
    for method, tr, per in (("Bollinger", boll_tr, ("train", "val")), ("XGBoost", xgb_tr, ("val",))):
        tr = tr[tr.period.isin(per)]
        side_days = pd.DatetimeIndex(sorted(set().union(*[
            set(session_date(labels[p].index[labels[p].isin(per).to_numpy()])) for p in PAIRS])))
        per_pair = {}
        for p in PAIRS:
            tp = tr[tr.pair == p]
            per_pair[p] = daily_returns(tp, side_days)
            side_rows.append({"method": method, "period": "+".join(per), "pair": p,
                              **summarize(tp, per_pair[p])})
        side_rows.append({"method": method, "period": "+".join(per), "pair": "POOLED",
                          **summarize(tr, pd.concat(per_pair, axis=1).mean(axis=1))})
    pd.DataFrame(side_rows).round(4).to_csv(out / "selection_side_metrics.csv", index=False)
    exits = pd.concat([boll_test, xgb_test]).groupby(["method", "exit_reason"]).agg(
        trades=("net_bps", "size"), mean_net_bps=("net_bps", "mean"), mean_bars=("bars_held", "mean"))
    exits.round(3).to_csv(out / "test_exit_reasons.csv")
    print(pd.DataFrame(side_rows).round(3).to_string(index=False))
    print(exits.round(3).to_string())

    # Per test quarter: is the result spread over the held-out period or concentrated?
    q_rows = []
    for method, tr in (("Bollinger", boll_test), ("XGBoost", xgb_test)):
        chunk = np.searchsorted(CHUNK_EDGES.values, pd.to_datetime(tr["entry_time"]).values, side="right") - 1
        for c in TEST_CHUNKS:
            x = tr["net_bps"].to_numpy()[chunk == c]
            q_rows.append({"method": method, "test_chunk": c,
                           "start": str(CHUNK_EDGES[c].date()), "end": str(CHUNK_EDGES[c + 1].date()),
                           "trades": len(x), "mean_net_bps": x.mean() if len(x) else float("nan"),
                           "t_stat": tstat(x)})
    pd.DataFrame(q_rows).round(4).to_csv(out / "test_by_quarter.csv", index=False)
    print(pd.DataFrame(q_rows).round(3).to_string(index=False))

    res.round(4).to_csv(out / "test_metrics.csv", index=False)
    diff.round(4).to_csv(out / "test_difference.csv", index=False)
    if args.save_trades:
        pd.concat([boll_test, xgb_test]).to_csv(out / "test_trades.csv", index=False)
    curves = pd.DataFrame({m: 100 * dailies[m].cumsum() for m in dailies}).round(4)
    curves.index.name = "date"
    curves.to_csv(out / "test_pooled_cumulative_pct.csv")
    meta = {
        "train_chunks": [i for i in range(len(CHUNK_EDGES) - 1) if i not in TEST_CHUNKS and i != VALIDATION_CHUNK],
        "validation_chunk": VALIDATION_CHUNK,
        "test_chunks": list(TEST_CHUNKS),
        "chunk_edges_utc": [str(e) for e in CHUNK_EDGES],
        "embargo_bars_each_side_of_test_boundaries": BARS_PER_DAY,
        "bollinger_selected": boll_choice,
        "xgboost_model": "A_all", "xgboost_theta": theta, "xgboost_best_iteration": best_it,
        "test_days": len(test_days),
    }
    (out / "selection.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(diff.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
