"""Tests for the evaluation metrics.

scikit-learn is not a dependency, so AUC, calibration and permutation
importance are hand-implemented -- which means they need hand-worked cases,
not a reference implementation to lean on.

The two that matter most:
  * `binomial_se` must take n_eff explicitly. A row-count error bar on this
    dataset is ~5.2x too narrow, which is the difference between "no signal"
    and a spurious discovery.
  * `directional_accuracy` must ignore the timeout class entirely. Overall
    accuracy is inflated by predictable rollover timeouts; the directional
    metric is the one that is supposed to be hard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxalgo.models import metrics as mx

# ------------------------------------------------------------ error bars


def test_binomial_se_matches_the_closed_form():
    assert mx.binomial_se(0.5, 2015) == pytest.approx(np.sqrt(0.25 / 2015))
    assert mx.binomial_se(0.5, 2015) == pytest.approx(0.011139, abs=1e-5)


def test_row_count_error_bars_are_much_narrower_than_n_eff_ones():
    """The whole reason n_eff is a required argument."""
    se_rows = mx.binomial_se(0.5, 54_645)
    se_eff = mx.binomial_se(0.5, 2_015)
    assert se_eff / se_rows == pytest.approx(np.sqrt(54_645 / 2_015), rel=1e-9)
    assert se_eff / se_rows > 5.0


def test_binomial_se_rejects_a_nonpositive_sample():
    with pytest.raises(ValueError):
        mx.binomial_se(0.5, 0)


def test_wald_ci_is_symmetric_about_p():
    lo, hi = mx.wald_ci(0.52, 2015)
    assert (lo + hi) / 2 == pytest.approx(0.52)
    assert hi - lo == pytest.approx(2 * 1.96 * mx.binomial_se(0.52, 2015))


def test_z_is_measured_against_the_base_rate_not_half():
    """Testing against 0.5 would credit the model with the class imbalance."""
    z_base = mx.z_against(0.52, 0.506, 2015)
    z_half = mx.z_against(0.52, 0.5, 2015)
    assert z_base < z_half
    assert z_base == pytest.approx((0.52 - 0.506) / mx.binomial_se(0.506, 2015))


# ------------------------------------------------------------------ AUC


def test_auc_is_one_for_a_perfect_ranking_and_half_for_none():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert mx._auc_binary(scores, np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert mx._auc_binary(scores, np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert mx._auc_binary(np.ones(4), np.array([0, 1, 0, 1])) == pytest.approx(0.5)


def test_auc_hand_case_with_a_tie():
    # positives at scores 0.5 and 0.5, negatives at 0.1 and 0.5
    # pairs: (0.5,0.1)=1, (0.5,0.5)=0.5 -> for each positive: 1 + 0.5 = 1.5
    # AUC = (1.5 + 1.5) / (2 * 2) = 0.75
    scores = np.array([0.5, 0.5, 0.1, 0.5])
    positive = np.array([1, 1, 0, 0])
    assert mx._auc_binary(scores, positive) == pytest.approx(0.75)


def test_macro_ovr_auc_averages_the_three_one_vs_rest_values():
    y = np.array([-1, 0, 1, -1, 0, 1])
    proba = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    assert mx.roc_auc_ovr_macro(y, proba) == pytest.approx(1.0)


# --------------------------------------------------- the headline metric


def test_directional_accuracy_ignores_the_timeout_class():
    y = np.array([1, -1, 0, 0, 1])
    # P(0) is huge everywhere; it must not affect the directional call
    proba = np.array([
        [0.1, 0.8, 0.1],   # pred +? p(+1)=0.1 < p(-1)=0.1 -> tie goes to +1
        [0.3, 0.6, 0.1],   # p(-1)=0.3 > p(+1)=0.1 -> -1, true -1  OK
        [0.4, 0.5, 0.1],   # timeout, excluded
        [0.1, 0.5, 0.4],   # timeout, excluded
        [0.1, 0.7, 0.2],   # p(+1)=0.2 > p(-1)=0.1 -> +1, true +1  OK
    ])
    acc, n = mx.directional_accuracy(y, proba)
    assert n == 3, "only the three touched events count"
    assert acc == pytest.approx(3 / 3)


def test_directional_accuracy_is_unchanged_by_the_timeout_probability():
    """Rescaling P(0) must not move the metric -- it is a two-way comparison."""
    rng = np.random.default_rng(0)
    y = rng.choice([-1, 0, 1], size=500)
    raw = rng.random((500, 3))
    a, _ = mx.directional_accuracy(y, raw / raw.sum(axis=1, keepdims=True))
    bumped = raw.copy()
    bumped[:, 1] *= 9.0
    b, _ = mx.directional_accuracy(y, bumped / bumped.sum(axis=1, keepdims=True))
    assert a == pytest.approx(b)


def test_directional_base_rate_is_the_majority_share():
    y = np.array([1, 1, -1, 0, 0, 0])
    assert mx.directional_base_rate(y) == pytest.approx(2 / 3)


def test_a_model_that_always_says_up_scores_the_base_rate():
    y = np.array([1] * 51 + [-1] * 49 + [0] * 30)
    proba = np.tile([0.1, 0.5, 0.9], (len(y), 1))
    acc, n = mx.directional_accuracy(y, proba)
    assert n == 100
    assert acc == pytest.approx(0.51)


# ------------------------------------------------------- classification


def test_confusion_and_precision_recall_on_a_hand_case():
    y = np.array([-1, -1, 0, 1, 1, 1])
    p = np.array([-1, 0, 0, 1, 1, -1])
    cm = mx.confusion(y, p)
    assert cm.loc["true -1", "pred -1"] == 1
    assert cm.loc["true -1", "pred +0"] == 1
    assert cm.loc["true +1", "pred +1"] == 2
    assert cm.to_numpy().sum() == len(y)

    pr = mx.precision_recall(y, p)
    assert pr.loc[1, "precision"] == pytest.approx(1.0)
    assert pr.loc[1, "recall"] == pytest.approx(2 / 3)
    assert pr.loc[-1, "precision"] == pytest.approx(0.5)
    assert pr.loc[-1, "recall"] == pytest.approx(0.5)


# --------------------------------------------------------- calibration


def test_calibration_recovers_a_perfectly_calibrated_model():
    rng = np.random.default_rng(4)
    prob = rng.random(200_000)
    outcome = (rng.random(200_000) < prob).astype(int)
    tab = mx.calibration_table(prob, outcome)
    filled = tab[tab["n"] > 100]
    assert len(filled) == 10
    assert filled["gap"].abs().max() < 0.02


def test_calibration_flags_an_overconfident_model():
    rng = np.random.default_rng(5)
    prob = rng.random(50_000)
    outcome = (rng.random(50_000) < 0.5).astype(int)   # truth is always 50/50
    tab = mx.calibration_table(prob, outcome)
    assert tab.loc[tab.index[-1], "gap"] < -0.3, "top bucket should be far too confident"
    assert tab.loc[tab.index[0], "gap"] > 0.3


def test_calibration_uses_fixed_width_not_quantile_buckets():
    """A model with all probabilities in a narrow band must show empty buckets."""
    prob = np.full(1000, 0.42)
    tab = mx.calibration_table(prob, np.zeros(1000, dtype=int))
    assert int((tab["n"] > 0).sum()) == 1


# -------------------------------------------------- permutation importance


def test_permutation_importance_finds_the_column_that_matters():
    rng = np.random.default_rng(6)
    n = 4000
    signal = rng.normal(size=n)
    x = pd.DataFrame({"useful": signal, "noise": rng.normal(size=n)})
    y = np.where(signal > 0, 1, -1)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        p_up = (frame["useful"].to_numpy() > 0).astype(float) * 0.9 + 0.05
        return np.column_stack([1 - p_up, np.zeros(len(frame)), p_up])

    imp = mx.permutation_importance(
        predict, x, y,
        metrics={"acc": lambda yt, pr: mx.directional_accuracy(yt, pr)[0]},
        n_repeats=3, seed=1,
    )
    assert imp.loc["useful", "acc_drop"] > 0.3
    assert abs(imp.loc["noise", "acc_drop"]) < 0.02
    assert imp.attrs["baseline"]["acc"] == pytest.approx(1.0)
