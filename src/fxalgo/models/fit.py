"""Fitting an XGBoost classifier under the same purge/embargo discipline as the split.

THE VALIDATION SLICE IS THE DANGEROUS PART. Early stopping selects a model by
validation score, so a leaky validation split does not merely mis-measure -- it
picks an overfit model and hands it to the test set looking healthy.

A RANDOM validation split leaks by construction: each label watches 240 forward
bars, so a training event at 14:00 and a validation event at 14:15 share almost
their whole outcome window. `purged_validation` therefore carves a CONTIGUOUS
block from the end of the training side and applies the same two mechanisms
`training.split` applies at every train/test boundary -- purge the fit side,
embargo the validation side -- using the same primitive, not a reimplementation.

HYPERPARAMETERS ARE FIXED, NOT TUNED. Four subsets is already four looks at the
same 2,015 independent test observations; tuning would turn this into a
comparison of hyperparameters and would multiply the looks. `DEFAULT_PARAMS` is
set once, conservatively, for a problem with ~8,857 independent training
windows -- not for 1.2 million rows, which is what the row count would suggest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fxalgo.training.split import EMBARGO_BARS, LABEL_HORIZON, window_reaches_test

SEED = 20260816

# Conservative and deliberately untuned. `min_child_weight` is the one that
# carries the effective-sample thinking: with softprob the per-row hessian is
# about p(1-p) ~ 0.22, so 400 is roughly 1,800 rows, and at ~136 rows per
# independent label window that is ~13 independent observations per leaf. A
# default of 1 would happily build leaves describing a single label window.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 400,
    "reg_lambda": 5.0,
    "gamma": 1.0,
    "tree_method": "hist",
    "seed": SEED,
}
NUM_BOOST_ROUND = 1500
EARLY_STOPPING_ROUNDS = 50


@dataclass
class Masks:
    """Row masks over the FULL event index."""

    fit: np.ndarray
    val: np.ndarray
    purged: int = 0
    embargoed: int = 0
    boundary: pd.Timestamp | None = None
    notes: dict[str, object] = field(default_factory=dict)


def purged_validation(
    index: pd.DatetimeIndex,
    train_mask: np.ndarray,
    val_start: pd.Timestamp,
    *,
    horizon: int = LABEL_HORIZON,
    embargo_bars: int = EMBARGO_BARS,
) -> Masks:
    """Carve a contiguous validation block from the END of the training side.

    `index` must be the FULL event index, because a label window is measured in
    positions on that index -- a window starting inside the fit block reaches
    across whatever lies between, sampled or not.

    Purge: fit events whose 240-bar window touches a validation event.
    Embargo: the first `embargo_bars` positions from the boundary, removed from
    validation, so adjacency cannot leak the other way.
    """
    val = train_mask & np.asarray(index >= val_start)
    fit = train_mask & np.asarray(index < val_start)

    reaches = window_reaches_test(val, horizon)
    purged = int((fit & reaches).sum())
    fit = fit & ~reaches

    # pandas' own searchsorted, not numpy's: converting a tz-aware timestamp to
    # np.datetime64 silently drops the zone.
    first = int(index.searchsorted(val_start, side="left"))
    before = int(val.sum())
    val = val.copy()
    val[first : first + embargo_bars] = False
    embargoed = before - int(val.sum())

    return Masks(fit=fit, val=val, purged=purged, embargoed=embargoed,
                 boundary=val_start)


def class_weights(y: np.ndarray, classes: tuple[int, ...] = (-1, 0, 1)) -> np.ndarray:
    """Inverse-frequency sample weights, mean 1.

    Weights rather than resampling: resampling would duplicate or drop rows,
    and with overlapping label windows a duplicated row is a duplicated
    *window*, which corrupts the effective sample size the error bars depend on.
    """
    w = np.ones(len(y), dtype="float64")
    for c in classes:
        m = y == c
        n = int(m.sum())
        if n:
            w[m] = len(y) / (len(classes) * n)
    return np.asarray(w / w.mean(), dtype='float64')


def fit_model(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    params: dict[str, object] | None = None,
    num_boost_round: int = NUM_BOOST_ROUND,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    classes: tuple[int, ...] = (-1, 0, 1),
) -> tuple[Any, dict[str, dict[str, list[float]]]]:
    """Fit with early stopping on the purged validation block."""
    import xgboost as xgb

    p = dict(DEFAULT_PARAMS if params is None else params)
    codes = {c: i for i, c in enumerate(classes)}
    dfit = xgb.DMatrix(x_fit, label=np.vectorize(codes.get)(y_fit),
                       weight=class_weights(y_fit, classes),
                       feature_names=list(x_fit.columns))
    dval = xgb.DMatrix(x_val, label=np.vectorize(codes.get)(y_val),
                       weight=class_weights(y_val, classes),
                       feature_names=list(x_val.columns))
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        p, dfit, num_boost_round=num_boost_round,
        evals=[(dfit, "fit"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result, verbose_eval=False,
    )
    return booster, evals_result


def predict_proba(booster: Any, x: pd.DataFrame) -> np.ndarray:
    """Class probabilities at the early-stopped iteration."""
    import xgboost as xgb

    d = xgb.DMatrix(x, feature_names=list(x.columns))
    best = getattr(booster, "best_iteration", None)
    kw = {"iteration_range": (0, best + 1)} if best is not None else {}
    return np.asarray(booster.predict(d, **kw), dtype="float64")
