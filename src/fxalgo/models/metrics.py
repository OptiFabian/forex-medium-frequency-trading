"""Evaluation metrics, with error bars sized to INDEPENDENT observations.

THE RULE THIS MODULE EXISTS TO ENFORCE. Every standard error, confidence
interval and significance claim about the test side must use the number of
INDEPENDENT LABEL WINDOWS (2,015), never the row count (54,645 sampled /
273,225 full). Overlapping label windows make rows ~5.2x more correlated than
independent, so a row-count error bar is ~5.2x too narrow -- narrow enough to
call noise a discovery. `binomial_se` therefore takes `n_eff` as a required
positional argument; there is no default to forget.

THE READING TRAP this module also exists to sidestep. Overall accuracy is
inflated by the timeout class: windows spanning the 17:00 NY rollover time out
far more often than clean ones, so a model can score well by predicting
timeouts from the calendar. That is real and learnable but it is mechanical,
not tradeable. `directional_accuracy` is the number that matters -- P(+1) vs
P(-1) among events where a barrier was actually touched.

scikit-learn is not a dependency; AUC, permutation importance and calibration
are implemented here and unit-tested against hand-worked cases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

CLASSES: tuple[int, ...] = (-1, 0, 1)


# ------------------------------------------------------------- error bars


def binomial_se(p: float, n_eff: float) -> float:
    """Standard error of a proportion at `n_eff` INDEPENDENT observations."""
    if n_eff <= 0:
        raise ValueError(f"n_eff must be positive, got {n_eff}")
    return float(np.sqrt(p * (1.0 - p) / n_eff))


def wald_ci(p: float, n_eff: float, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wald interval at `n_eff` independent observations."""
    se = binomial_se(p, n_eff)
    return (p - z * se, p + z * se)


def z_against(p: float, base_rate: float, n_eff: float) -> float:
    """Z statistic of `p` against a fixed base rate, at `n_eff` observations.

    The base rate is the honest null here, NOT 0.5: the touched population is
    not balanced, so testing against 0.5 would credit the model with the class
    imbalance it did not have to find.
    """
    se = binomial_se(base_rate, n_eff)
    return float((p - base_rate) / se)


# ------------------------------------------------------------------ AUC


def _auc_binary(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1, dtype="float64")
    # average ranks within tied score groups
    s = scores[order]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_auc_ovr_macro(y_true: np.ndarray, proba: np.ndarray,
                      classes: Sequence[int] = CLASSES) -> float:
    """Macro-averaged one-vs-rest AUC over `classes`."""
    aucs = [_auc_binary(proba[:, k], y_true == c) for k, c in enumerate(classes)]
    return float(np.nanmean(aucs))


# ----------------------------------------------------- the headline metric


def directional_mask(y_true: np.ndarray) -> np.ndarray:
    """Events where a barrier was actually touched (label != 0)."""
    return np.asarray(y_true != 0, dtype=bool)


def directional_accuracy(y_true: np.ndarray, proba: np.ndarray,
                         classes: Sequence[int] = CLASSES) -> tuple[float, int]:
    """P(+1) vs P(-1) accuracy among TOUCHED events. Returns (accuracy, n rows).

    The timeout class is excluded from both the prediction and the truth: the
    question is which barrier was hit, given that one was. `P(0)` is ignored
    rather than renormalised -- ranking between the two directional classes is
    unchanged by renormalising, and ignoring it keeps the metric a pure
    two-way comparison.
    """
    i_neg, i_pos = list(classes).index(-1), list(classes).index(1)
    m = directional_mask(y_true)
    if not m.any():
        return float("nan"), 0
    pred = np.where(proba[m, i_pos] >= proba[m, i_neg], 1, -1)
    return float((pred == y_true[m]).mean()), int(m.sum())


def directional_base_rate(y_true: np.ndarray) -> float:
    """The majority share among touched events -- the null a model must beat."""
    m = directional_mask(y_true)
    if not m.any():
        return float("nan")
    share_up = float((y_true[m] == 1).mean())
    return max(share_up, 1.0 - share_up)


# --------------------------------------------------------- classification


def predicted_class(proba: np.ndarray, classes: Sequence[int] = CLASSES) -> np.ndarray:
    return np.asarray(np.asarray(classes)[proba.argmax(axis=1)])


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              classes: Sequence[int] = CLASSES) -> pd.DataFrame:
    m = pd.DataFrame(0, index=[f"true {c:+d}" for c in classes],
                     columns=[f"pred {c:+d}" for c in classes], dtype="int64")
    for i, ct in enumerate(classes):
        for j, cp in enumerate(classes):
            m.iat[i, j] = int(((y_true == ct) & (y_pred == cp)).sum())
    return m


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     classes: Sequence[int] = CLASSES) -> pd.DataFrame:
    rows = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        pp = int((y_pred == c).sum())
        ap = int((y_true == c).sum())
        rows.append({
            "class": c,
            "precision": tp / pp if pp else float("nan"),
            "recall": tp / ap if ap else float("nan"),
            "support": ap,
            "predicted": pp,
        })
    return pd.DataFrame(rows).set_index("class")


# ---------------------------------------------------------- calibration


def calibration_table(prob: np.ndarray, outcome: np.ndarray,
                      n_buckets: int = 10) -> pd.DataFrame:
    """Observed frequency per predicted-probability bucket.

    Fixed-width buckets over [0, 1], not quantile buckets: the question is
    whether a stated probability of 0.7 happens 70% of the time, which is a
    statement about the probability SCALE. Quantile buckets would hide a model
    whose probabilities are all crammed into a narrow band.
    """
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_buckets - 1)
    rows = []
    for b in range(n_buckets):
        m = idx == b
        rows.append({
            "bucket": f"[{edges[b]:.1f},{edges[b+1]:.1f})",
            "n": int(m.sum()),
            "mean_predicted": float(prob[m].mean()) if m.any() else float("nan"),
            "observed": float(outcome[m].mean()) if m.any() else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["gap"] = out["observed"] - out["mean_predicted"]
    return out


# -------------------------------------------------- permutation importance


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Drop in each metric when one column is shuffled, averaged over repeats.

    Permutation, NOT split gain. Split-gain importance is biased toward
    high-cardinality continuous features, and with overlapping labels a column
    can rank highly for tracking the CALENDAR rather than the market. Shuffling
    asks the only question that matters: how much worse does the model get
    without this column's information, holding the fitted model fixed.

    Computed on the TEST set: importance on the training set measures what the
    model memorised, not what generalises.
    """
    rng = np.random.default_rng(seed)
    base_pred = predict(x)
    base = {name: fn(y, base_pred) for name, fn in metrics.items()}
    rows = []
    for col in x.columns:
        drops: dict[str, list[float]] = {name: [] for name in metrics}
        original = x[col].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = x.copy()
            shuffled[col] = rng.permutation(original)
            pred = predict(shuffled)
            for name, fn in metrics.items():
                drops[name].append(base[name] - fn(y, pred))
        row = {"feature": col}
        for name in metrics:
            row[f"{name}_drop"] = float(np.mean(drops[name]))
            row[f"{name}_sd"] = float(np.std(drops[name], ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows).set_index("feature")
    out.attrs["baseline"] = base
    return out
