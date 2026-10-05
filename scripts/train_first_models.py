"""First XGBoost models on the four PRE-COMMITTED feature subsets.

Read-only on features, labels and the split. Fits four models, writes them and
their predictions, reports, and stops.

THE PRE-COMMITTED READING, recorded before this run and not to be softened
after it. The test side carries 2,015 INDEPENDENT LABEL WINDOWS, not 54,645
rows. At 2,015 observations a true edge near 55% accuracy is clearly
detectable, 53% is marginal, and 52% or below is NOT resolvable. Realistic FX
edges sit at 51-53%. A result at 51.5% is INSUFFICIENT RESOLUTION -- neither
evidence of an edge nor evidence against one.

THE READING TRAP. Overall accuracy is inflated by the timeout class: windows
spanning the 17:00 NY rollover time out far more often than clean ones, so a
model can score well predicting timeouts from the calendar. Real, learnable,
and not tradeable. The number that matters is DIRECTIONAL ACCURACY ON TOUCHED
EVENTS, reported against its own base rate rather than against 50%.

NO TUNING. Hyperparameters are identical across all four subsets. Tuning would
make this a comparison of hyperparameters instead of feature sets, and every
pass is another look at the same 2,015 observations.

Usage:
    uv run python scripts/train_first_models.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fxalgo.models import fit as mf
from fxalgo.models import metrics as mx
from fxalgo.models import subsets as ss
from fxalgo.settings import PROJECT_ROOT, get_settings
from fxalgo.training import dataset as ds
from fxalgo.utils.logging import setup_logging

TRAINING = PROJECT_ROOT / "data" / "training"
EVENTS = TRAINING / "eurusd_events.parquet"
SPLIT = TRAINING / "eurusd_split.parquet"
MODELS = PROJECT_ROOT / "data" / "models"
PREDS = MODELS / "predictions"

LABEL = "label_2.0"
CLASSES = (-1, 0, 1)
# Independent label windows per side, from the split build. NOT row counts.
N_EFF_TEST = 2015.0
N_EFF_TRAIN = 8857.0
PCT = [0.01, 0.25, 0.5, 0.75, 0.99]


def _read(path, columns=None) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df


def _numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """float64 view; pd.NA becomes NaN, which XGBoost handles natively."""
    return frame[cols].astype("float64")


def _dir_stats(y: np.ndarray, proba: np.ndarray, n_eff: float, base: float) -> dict:
    acc, n = mx.directional_accuracy(y, proba, CLASSES)
    se = mx.binomial_se(acc, n_eff)
    lo, hi = mx.wald_ci(acc, n_eff)
    return {"dir_acc": acc, "dir_n": n, "dir_se": se, "dir_lo": lo, "dir_hi": hi,
            "dir_z": mx.z_against(acc, base, n_eff)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perm-repeats", type=int, default=3)
    args = ap.parse_args()

    settings = get_settings()
    setup_logging(level=settings.log_level, log_dir=PROJECT_ROOT / "logs")
    MODELS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)

    split = _read(SPLIT)
    events = _read(EVENTS)
    full_index = split.index
    print("First models -- four pre-committed subsets, identical hyperparameters")
    print(f"  events {len(events):,} sampled | split table {len(split):,} rows")

    # ---------------------------------------------------- fit / val / test
    train_full = (split["split"] == "train").to_numpy()
    test_full = (split["split"] == "test").to_numpy()
    train_chunks = sorted(split.loc[split["split"] == "train", "chunk_id"].unique())
    val_chunk = int(max(train_chunks))
    val_start = split.loc[split["chunk_id"] == val_chunk, "chunk_start"].iloc[0]
    masks = mf.purged_validation(full_index, train_full, val_start)

    role = pd.Series("unused", index=full_index, dtype=object)
    role[masks.fit] = "fit"
    role[masks.val] = "val"
    role[test_full] = "test"
    ev_role = role.loc[events.index].to_numpy()

    print("\n=== VALIDATION CARVE (purged, not random) ===")
    print(f"  validation = the LAST training chunk, id {val_chunk}, from {val_start}")
    print("  contiguous block at the end of the training side; the boundary gets")
    print("  the same treatment split.py gives every train/test boundary.")
    print(f"  purged from fit (window reaches validation): {masks.purged:,} full events")
    print(f"  embargoed from validation ({mf.EMBARGO_BARS:,} bars): "
          f"{masks.embargoed:,} full events")
    for name in ("fit", "val", "test"):
        n_full = int((role == name).sum())
        n_ev = int((ev_role == name).sum())
        print(f"  {name:<5} {n_full:>10,d} full events | {n_ev:>8,d} sampled rows")
    assert not (masks.fit & test_full).any() and not (masks.val & test_full).any()

    y_all = events[LABEL].to_numpy(dtype="float64")
    ok = ~np.isnan(y_all)
    fit_rows = (ev_role == "fit") & ok
    val_rows = (ev_role == "val") & ok
    test_rows = (ev_role == "test") & ok
    y_fit = y_all[fit_rows].astype("int64")
    y_val = y_all[val_rows].astype("int64")
    y_test = y_all[test_rows].astype("int64")

    non_features = set(ds.AUX_COLUMNS) | set(ds.KEPT_LABEL_COLUMNS)
    features = ss.feature_columns(events, non_features)
    subsets = ss.build(features)

    base_test = mx.directional_base_rate(y_test)
    base_fit = mx.directional_base_rate(y_fit)
    print("\n=== THE POPULATION ===")
    print(f"  test rows {len(y_test):,} | touched {int((y_test != 0).sum()):,} "
          f"({(y_test != 0).mean():.2%})")
    print("  test label mix: " + "  ".join(
        f"{c:+d}: {(y_test == c).mean():.4%}" for c in CLASSES))
    print(f"  DIRECTIONAL BASE RATE on touched test events: {base_test:.4%}")
    print(f"  N_eff test {N_EFF_TEST:,.0f} independent label windows "
          f"(rows would give {len(y_test):,} -- "
          f"{np.sqrt(len(y_test)/N_EFF_TEST):.1f}x too confident)")

    # the reading trap, verified rather than assumed
    gap = events.loc[test_rows, "spans_gap"].to_numpy(dtype=bool)
    print("\n  TIMEOUT RATE by window type (the reading trap, on test):")
    print(f"    spans a 17:00 NY rollover : {(y_test[gap] == 0).mean():.2%} "
          f"time out  ({int(gap.sum()):,} rows)")
    print(f"    clean window              : {(y_test[~gap] == 0).mean():.2%} "
          f"time out  ({int((~gap).sum()):,} rows)")

    print("\n=== SUBSETS (pre-committed, closed) ===")
    for name, cols in subsets.items():
        print(f"  {name:<18}{len(cols):>4} columns")
    print("\n" + ss.STRATEGY_CORE_NOTES)

    print("=== HYPERPARAMETERS (identical across all four; not tuned) ===")
    for k, v in mf.DEFAULT_PARAMS.items():
        print(f"  {k:<20}{v}")
    print(f"  {'num_boost_round':<20}{mf.NUM_BOOST_ROUND} (cap)")
    print(f"  {'early_stopping':<20}{mf.EARLY_STOPPING_ROUNDS} rounds on the "
          f"purged validation block")
    print(f"  {'sample weights':<20}inverse class frequency, mean 1 (not resampling)")
    print(f"  {'seed':<20}{mf.SEED}")

    # ------------------------------------------------------------- fitting
    rows, store = [], {}
    for name, cols in subsets.items():
        x_fit = _numeric(events.loc[fit_rows], cols)
        x_val = _numeric(events.loc[val_rows], cols)
        x_test = _numeric(events.loc[test_rows], cols)
        print(f"\nfitting {name} ({len(cols)} features) ...", flush=True)
        booster, evals = mf.fit_model(x_fit, y_fit, x_val, y_val, classes=CLASSES)
        best = booster.best_iteration
        p_test = mf.predict_proba(booster, x_test)
        p_fit = mf.predict_proba(booster, x_fit)
        booster.save_model(str(MODELS / f"{name}.json"))
        for tag, idx, proba in (("test", events.index[test_rows], p_test),
                                ("fit", events.index[fit_rows], p_fit)):
            pd.DataFrame(
                {"p_-1": proba[:, 0], "p_0": proba[:, 1], "p_+1": proba[:, 2]},
                index=idx,
            ).to_parquet(PREDS / f"{name}_{tag}.parquet", engine="pyarrow")

        pred_test = mx.predicted_class(p_test, CLASSES)
        pred_fit = mx.predicted_class(p_fit, CLASSES)
        d_test = _dir_stats(y_test, p_test, N_EFF_TEST, base_test)
        d_fit = _dir_stats(y_fit, p_fit, N_EFF_TRAIN, base_fit)
        pr = mx.precision_recall(y_test, pred_test, CLASSES)
        rows.append({
            "subset": name, "n_features": len(cols), "best_iteration": int(best),
            "val_mlogloss": float(evals["val"]["mlogloss"][best]),
            "test_acc": float((pred_test == y_test).mean()),
            "test_auc": mx.roc_auc_ovr_macro(y_test, p_test, CLASSES),
            "fit_acc": float((pred_fit == y_fit).mean()),
            "fit_auc": mx.roc_auc_ovr_macro(y_fit, p_fit, CLASSES),
            **{f"test_{k}": v for k, v in d_test.items()},
            **{f"fit_{k}": v for k, v in d_fit.items()},
            **{f"prec_{c:+d}": pr.loc[c, "precision"] for c in CLASSES},
            **{f"rec_{c:+d}": pr.loc[c, "recall"] for c in CLASSES},
        })
        store[name] = {"proba_test": p_test, "pred_test": pred_test,
                       "x_test": x_test, "booster": booster, "cols": cols}
        print(f"  stopped at iteration {best} | test dir acc "
              f"{d_test['dir_acc']:.4%}")

    res = pd.DataFrame(rows).set_index("subset")
    res.to_csv(MODELS / "results.csv")

    # ------------------------------------------------------------- report
    print("\n" + "=" * 118)
    print("RESULTS -- all error bars on N_eff = 2,015 INDEPENDENT WINDOWS, never row counts")
    print("=" * 118)
    print(f"{'subset':<18}{'feat':>5}{'iters':>7}{'overall acc':>13}{'macro AUC':>11}"
          f"{'DIR ACC (touched)':>19}{'SE':>8}{'95% CI':>18}{'z vs base':>11}")
    for name, r in res.iterrows():
        ci = f"[{r['test_dir_lo']:.2%}, {r['test_dir_hi']:.2%}]"
        print(f"{name:<18}{int(r['n_features']):>5}{int(r['best_iteration']):>7}"
              f"{r['test_acc']:>13.4%}{r['test_auc']:>11.4f}"
              f"{r['test_dir_acc']:>19.4%}{r['test_dir_se']:>8.4f}{ci:>18}"
              f"{r['test_dir_z']:>11.2f}")
    print(f"\n  directional base rate = {base_test:.4%}; "
          f"SE at N_eff=2,015 is {mx.binomial_se(base_test, N_EFF_TEST):.4%} "
          f"(={mx.binomial_se(base_test, N_EFF_TEST)*100:.2f} pp)")
    print(f"  a 95% CI is +/- {1.96*mx.binomial_se(base_test, N_EFF_TEST)*100:.2f} pp, "
          f"so anything inside "
          f"[{base_test-1.96*mx.binomial_se(base_test, N_EFF_TEST):.2%}, "
          f"{base_test+1.96*mx.binomial_se(base_test, N_EFF_TEST):.2%}] "
          f"is indistinguishable from the base rate")
    print("\n  NOTE the touched subset is ~72% of test events, so its own N_eff is")
    print(f"  nearer {N_EFF_TEST*float((y_test != 0).mean()):,.0f} than 2,015 -- the "
          f"error bars above are if anything OPTIMISTIC.")

    print("\n=== PER-CLASS PRECISION / RECALL (test) ===")
    print(f"{'subset':<18}" + "".join(
        f"{'P(' + f'{c:+d}' + ')':>10}{'R(' + f'{c:+d}' + ')':>10}" for c in CLASSES))
    for name, r in res.iterrows():
        print(f"{name:<18}" + "".join(
            f"{r[f'prec_{c:+d}']:>10.4f}{r[f'rec_{c:+d}']:>10.4f}" for c in CLASSES))

    print("\n=== A) CONFUSION MATRICES (test) ===")
    for name in subsets:
        print(f"\n  {name}:")
        cm = mx.confusion(y_test, store[name]["pred_test"], CLASSES)
        print("    " + cm.to_string().replace("\n", "\n    "))

    print("\n=== B) DIRECTIONAL ACCURACY BY WINDOW TYPE (spans_gap) ===")
    print(f"{'subset':<18}{'clean acc':>12}{'clean n':>10}{'gap acc':>12}"
          f"{'gap n':>10}{'difference':>13}")
    for name in subsets:
        p = store[name]["proba_test"]
        out = {}
        for tag, m in (("clean", ~gap), ("gap", gap)):
            a, n = mx.directional_accuracy(y_test[m], p[m], CLASSES)
            out[tag] = (a, n)
        print(f"{name:<18}{out['clean'][0]:>12.4%}{out['clean'][1]:>10,d}"
              f"{out['gap'][0]:>12.4%}{out['gap'][1]:>10,d}"
              f"{out['gap'][0]-out['clean'][0]:>+13.4%}")

    print("\n=== C) DIRECTIONAL ACCURACY BY TEST CHUNK ===")
    chunk_test = split.loc[events.index[test_rows], "chunk_id"].to_numpy()
    chunk_ids = sorted(set(chunk_test.tolist()))
    n_eff_chunk = N_EFF_TEST / len(chunk_ids)
    print(f"  per-chunk N_eff ~ {n_eff_chunk:,.0f} windows -> SE ~ "
          f"{mx.binomial_se(0.5, n_eff_chunk)*100:.2f} pp, 95% CI +/- "
          f"{1.96*mx.binomial_se(0.5, n_eff_chunk)*100:.2f} pp")
    print(f"{'subset':<18}" + "".join(f"{'chunk ' + str(c):>16}" for c in chunk_ids))
    for name in subsets:
        p = store[name]["proba_test"]
        cells = []
        for c in chunk_ids:
            m = chunk_test == c
            a, n = mx.directional_accuracy(y_test[m], p[m], CLASSES)
            cells.append(f"{a:>10.4%}({n:,})".rjust(16))
        print(f"{name:<18}" + "".join(cells))

    print("\n=== D) TRAIN vs TEST (the overfitting gap) ===")
    print(f"{'subset':<18}{'fit acc':>10}{'test acc':>10}{'gap':>9}"
          f"{'fit AUC':>10}{'test AUC':>10}{'fit dir':>10}{'test dir':>10}{'dir gap':>10}")
    for name, r in res.iterrows():
        print(f"{name:<18}{r['fit_acc']:>10.4%}{r['test_acc']:>10.4%}"
              f"{r['fit_acc']-r['test_acc']:>+9.2%}{r['fit_auc']:>10.4f}"
              f"{r['test_auc']:>10.4f}{r['fit_dir_acc']:>10.4%}"
              f"{r['test_dir_acc']:>10.4%}{r['fit_dir_acc']-r['test_dir_acc']:>+10.2%}")

    print("\n=== E) PERMUTATION IMPORTANCE, subset A, TEST set, top 20 ===")
    print("  permutation not split gain: split gain is biased toward")
    print("  high-cardinality continuous columns and, with overlapping labels, a")
    print("  column can rank high for tracking the CALENDAR rather than the market.")
    a = store["A_all"]

    def _predict(frame: pd.DataFrame) -> np.ndarray:
        return mf.predict_proba(a["booster"], frame)

    imp = mx.permutation_importance(
        _predict, a["x_test"], y_test,
        metrics={
            "acc": lambda yt, pr: float((mx.predicted_class(pr, CLASSES) == yt).mean()),
            "dir": lambda yt, pr: mx.directional_accuracy(yt, pr, CLASSES)[0],
        },
        n_repeats=args.perm_repeats, seed=mf.SEED,
    )
    imp = imp.sort_values("acc_drop", ascending=False)
    imp.to_csv(MODELS / "permutation_importance_A.csv")
    print(f"  baseline: overall acc {imp.attrs['baseline']['acc']:.4%}, "
          f"dir acc {imp.attrs['baseline']['dir']:.4%}  "
          f"({args.perm_repeats} repeats)")
    print(f"  {'rank':>4}  {'feature':<36}{'acc drop':>11}{'+/-':>8}{'dir drop':>11}"
          f"{'+/-':>8}")
    for i, (feat, r) in enumerate(imp.head(20).iterrows(), start=1):
        print(f"  {i:>4}  {feat:<36}{r['acc_drop']:>+11.4%}{r['acc_sd']:>8.4f}"
              f"{r['dir_drop']:>+11.4%}{r['dir_sd']:>8.4f}")
    time_rank = [i for i, f in enumerate(imp.index, 1) if f in ss.TIME_COLUMNS]
    print(f"\n  the 8 clock columns rank {time_rank} of {len(imp)} by accuracy drop")
    print(f"  their combined accuracy drop: "
          f"{imp.loc[list(ss.TIME_COLUMNS), 'acc_drop'].sum():+.4%} "
          f"vs top-20 total {imp.head(20)['acc_drop'].sum():+.4%}")

    print("\n=== F) CALIBRATION OF P(+1), subset A, test ===")
    cal = mx.calibration_table(a["proba_test"][:, 2], (y_test == 1).astype(int))
    cal.to_csv(MODELS / "calibration_A.csv", index=False)
    print(f"  {'bucket':<14}{'n':>9}{'mean predicted':>16}{'observed':>11}{'gap':>10}")
    for _, r in cal.iterrows():
        if r["n"] == 0:
            print(f"  {r['bucket']:<14}{0:>9}{'--':>16}{'--':>11}{'--':>10}")
            continue
        print(f"  {r['bucket']:<14}{int(r['n']):>9,d}{r['mean_predicted']:>16.4f}"
              f"{r['observed']:>11.4f}{r['gap']:>+10.4f}")
    filled = cal[cal["n"] > 0]
    print(f"  predicted P(+1) spans [{a['proba_test'][:, 2].min():.3f}, "
          f"{a['proba_test'][:, 2].max():.3f}]; "
          f"{int((filled['n'] > 0).sum())} of 10 buckets populated")
    print(f"  max |gap| on buckets with n > 100: "
          f"{filled[filled['n'] > 100]['gap'].abs().max():.4f}")

    (MODELS / "run.json").write_text(json.dumps({
        "seed": mf.SEED, "params": {k: str(v) for k, v in mf.DEFAULT_PARAMS.items()},
        "num_boost_round": mf.NUM_BOOST_ROUND,
        "early_stopping_rounds": mf.EARLY_STOPPING_ROUNDS,
        "validation_chunk": val_chunk, "validation_start": str(val_start),
        "n_eff_test": N_EFF_TEST, "n_eff_train": N_EFF_TRAIN,
        "directional_base_rate": base_test,
        "subsets": {k: v for k, v in subsets.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote models to {MODELS}, predictions to {PREDS}")
    print(f"Wrote {MODELS / 'results.csv'}, {MODELS / 'run.json'}, "
          f"{MODELS / 'permutation_importance_A.csv'}, {MODELS / 'calibration_A.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
