"""Tests for the online-model-retraining ACI variant
(`RetrainingSlidingWindowAdaptiveConformalForecaster` in
`forecasting/adaptive.py`) -- the README's long-open "What's next" item:
every ACI/EnbPI variant shipped before this one keeps the point model
frozen at its initial fit for the whole test window; only the conformal
layer adapts. This module checks the actual new mechanism (periodic
retraining on a growing, leak-free history) by hand first, mirroring
`tests/test_sliding_window.py`'s discipline, before
`tests/test_online_retraining_experiment.py` wires it to the real LSTM and
a bundled dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.adaptive import (
    RetrainingSlidingWindowAdaptiveConformalForecaster,
    SlidingWindowAdaptiveConformalForecaster,
    evaluate_adaptive_coverage,
)
from forecasting.models import NaiveForecaster


class MeanModel:
    """A trivial, hand-checkable model: predicts the mean of whatever `y`
    it was last fit on. Deliberately cheap (no real training), so these
    tests check the *retraining mechanism* (cadence, history growth, no
    leakage) in isolation from any neural-network cost or randomness --
    `tests/test_online_retraining_experiment.py` covers the real LSTM case.
    """

    def __init__(self):
        self.value: float | None = None
        self.n_fit_calls = 0
        self.fit_sizes: list[int] = []

    def fit(self, X, y):
        self.n_fit_calls += 1
        self.fit_sizes.append(len(X))
        self.value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full((X.shape[0], 1), self.value, dtype=np.float64)


def _factory():
    return MeanModel()


def test_validates_retrain_every():
    with pytest.raises(ValueError):
        RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, retrain_every=0)
    with pytest.raises(ValueError):
        RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, retrain_every=-5)
    # retrain_every=None (retraining disabled entirely) and any positive int are both fine.
    RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, retrain_every=None)
    RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, retrain_every=3)


def test_predict_sequential_before_fit_raises():
    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, alpha=0.1, retrain_every=2)
    with pytest.raises(RuntimeError):
        forecaster.predict_sequential(np.zeros((3, 3)), np.zeros(3))


def test_predict_sequential_requires_matching_lengths():
    X = np.zeros((10, 3))
    y = np.zeros((10, 1))
    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, alpha=0.1, retrain_every=2).fit(
        X, y, X, y
    )
    with pytest.raises(ValueError):
        forecaster.predict_sequential(np.zeros((5, 3)), np.zeros(4))


def test_fit_stores_train_plus_cal_as_the_initial_retrain_history():
    n_train, n_cal = 6, 4
    X_train = np.zeros((n_train, 3))
    y_train = np.zeros((n_train, 1))
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.zeros((n_cal, 1))

    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, retrain_every=5).fit(
        X_train, y_train, X_cal, y_cal
    )

    assert forecaster._X_history.shape[0] == n_train + n_cal
    assert forecaster._y_history.shape[0] == n_train + n_cal
    # The very first fit (inside .fit(), via the parent class) happened on
    # X_train/y_train alone, matching every other ACI variant's own fit
    # contract -- retraining only changes what happens *after* fit().
    assert forecaster.model.n_fit_calls == 1
    assert forecaster.model.fit_sizes == [n_train]


def test_retrain_cadence_and_no_future_leakage():
    """The actual mechanism this feature adds: with `retrain_every=3` and
    10 test steps, a retrain should fire right before scoring steps
    3, 6, 9 -- never step 0 (the initial fit from `.fit()` already covers
    it) -- and each retrain's training set must be exactly
    `n_train + n_cal` original windows plus the test windows *strictly
    before* the current step (never including the current step's own,
    not-yet-revealed y).
    """
    n_train, n_cal, n_test = 5, 5, 10
    X_train = np.zeros((n_train, 3))
    y_train = np.zeros((n_train, 1))  # all-zero history
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.zeros((n_cal, 1))

    # `model_factory` constructs a *fresh* model instance per retrain (see
    # the class docstring), so `forecaster.model` at the end only reflects
    # the last one -- a shared log, captured across every instance the
    # factory produces, is what actually lets this test see every fit call.
    fit_log: list[int] = []

    class LoggingMeanModel(MeanModel):
        def fit(self, X, y):
            fit_log.append(len(X))
            return super().fit(X, y)

    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(
        LoggingMeanModel, alpha=0.2, window=None, retrain_every=3
    ).fit(X_train, y_train, X_cal, y_cal)

    X_test = np.zeros((n_test, 3))
    y_test = np.arange(n_test, dtype=float)  # 0, 1, 2, ..., 9

    pred = forecaster.predict_sequential(X_test, y_test)

    assert pred.n_retrains == 3
    assert list(pred.retrained_at_steps) == [3, 6, 9]

    # fit_log[0] is the initial fit from .fit() itself (n_train=5); the
    # three retrains should have grown the training set by exactly the
    # number of test steps revealed so far: 5+5+3=13, 5+5+6=16, 5+5+9=19.
    assert fit_log == [n_train, n_train + n_cal + 3, n_train + n_cal + 6, n_train + n_cal + 9]

    # Hand-check the no-leakage claim directly: at the retrain before step 3,
    # the model was refit on y=[0,0,0,0,0,0,0,0,0,0,0,1,2] (train+cal all
    # zero, plus y_test[0:3]=[0,1,2]) -- mean = 3/13 -- so point[3] (what the
    # model predicts *using* that retrain) must reflect it, not y_test[3]=3
    # itself (which would leak the very value being predicted).
    expected_mean_before_step_3 = (n_train + n_cal) * 0.0 + 0.0 + 1.0 + 2.0
    expected_mean_before_step_3 /= n_train + n_cal + 3
    assert pred.point[3, 0] == pytest.approx(expected_mean_before_step_3)
    # In particular, it must NOT equal a mean that includes y_test[3]=3.
    leaked_mean = (expected_mean_before_step_3 * (n_train + n_cal + 3) + 3.0) / (n_train + n_cal + 4)
    assert pred.point[3, 0] != pytest.approx(leaked_mean)


def test_retrain_every_none_matches_the_frozen_sliding_pool_variant_exactly():
    """`retrain_every=None` should disable retraining entirely and reduce to
    exactly `SlidingWindowAdaptiveConformalForecaster`'s own behavior --
    this is the regression test that guarantees the new class is a strict
    superset/generalization, not an accidentally-different reimplementation
    of the frozen case.
    """
    n_train, n_cal, n_test = 5, 5, 12
    X_train = np.zeros((n_train, 3))
    y_train = np.full((n_train, 1), 2.0)
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.full((n_cal, 1), 2.0)
    X_test = np.zeros((n_test, 3))
    y_test = np.linspace(0, 5, n_test)

    frozen = SlidingWindowAdaptiveConformalForecaster(MeanModel(), alpha=0.1, gamma=0.05, window=7).fit(
        X_train, y_train, X_cal, y_cal
    )
    frozen_pred = frozen.predict_sequential(X_test, y_test)

    retraining_disabled = RetrainingSlidingWindowAdaptiveConformalForecaster(
        _factory, alpha=0.1, gamma=0.05, window=7, retrain_every=None
    ).fit(X_train, y_train, X_cal, y_cal)
    disabled_pred = retraining_disabled.predict_sequential(X_test, y_test)

    assert disabled_pred.n_retrains == 0
    assert list(disabled_pred.retrained_at_steps) == []
    np.testing.assert_allclose(disabled_pred.point, frozen_pred.point)
    np.testing.assert_allclose(disabled_pred.lower, frozen_pred.lower)
    np.testing.assert_allclose(disabled_pred.upper, frozen_pred.upper)
    np.testing.assert_allclose(disabled_pred.q_hat, frozen_pred.q_hat)
    np.testing.assert_array_equal(disabled_pred.pool_size, frozen_pred.pool_size)


def test_window_still_caps_the_pool_independently_of_retraining():
    """Retraining the model and sliding the residual pool are independent
    mechanisms -- this checks the pool-capping behavior already proven for
    the frozen variant (`tests/test_sliding_window.py`) still holds exactly
    when retraining is also happening.
    """
    n_train, n_cal, n_test, window = 3, 5, 10, 6
    X_train = np.zeros((n_train, 3))
    y_train = np.zeros((n_train, 1))
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.zeros((n_cal, 1))
    X_test = np.zeros((n_test, 3))
    y_test = np.arange(n_test, dtype=float)

    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(
        _factory, alpha=0.2, window=window, retrain_every=4
    ).fit(X_train, y_train, X_cal, y_cal)
    pred = forecaster.predict_sequential(X_test, y_test)

    expected = [min(n_cal + t, window) for t in range(n_test)]
    assert list(pred.pool_size) == expected
    assert pred.pool_size.max() == window
    assert pred.n_retrains == 2  # fires at t=4 and t=8
    assert list(pred.retrained_at_steps) == [4, 8]


def test_evaluate_adaptive_coverage_accepts_the_retraining_prediction_type():
    """`evaluate_adaptive_coverage` was written generically against the base
    `AdaptiveConformalPrediction` fields -- confirm the new subclass works
    with it unmodified, the same way `SlidingWindowAdaptivePrediction`
    already does.
    """
    n_train, n_test = 5, 8
    X_train = np.zeros((n_train, 3))
    y_train = np.zeros((n_train, 1))
    X_test = np.zeros((n_test, 3))
    y_test = np.zeros(n_test)

    forecaster = RetrainingSlidingWindowAdaptiveConformalForecaster(_factory, alpha=0.1, retrain_every=3).fit(
        X_train, y_train, X_train, y_train
    )
    pred = forecaster.predict_sequential(X_test, y_test)
    result = evaluate_adaptive_coverage(pred, y_test.reshape(-1, 1))

    assert result["empirical_coverage"] == pytest.approx(1.0)  # all-zero model, all-zero truth, zero residuals
    assert result["n_test"] == n_test
