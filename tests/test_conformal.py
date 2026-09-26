"""The most important test file in this repo.

A conformal-prediction library is only worth anything if its intervals
actually achieve close to their advertised coverage. These tests don't
just check that the code runs -- they check the *statistical claim*.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.data import generate_synthetic, make_windows
from forecasting.models import NaiveForecaster


class ConstantModel:
    """A trivial model that always predicts a fixed constant.

    Used to test the conformal wrapper in isolation, independent of any
    real model's ability to fit the data well (a bad point forecast just
    means wider intervals -- coverage should still hold).
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full((X.shape[0], 1), self.value, dtype=np.float64)


def test_quantile_finite_sample_correction_matches_hand_computation():
    # 9 calibration residuals, alpha=0.1 -> ceil(10 * 0.9) = 9 -> the 9th of
    # 9 sorted values (index 8) -> the max.
    residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    q = SplitConformalForecaster._quantile_with_finite_sample_correction(residuals, alpha=0.1)
    assert q == 9.0


def test_conformal_interval_is_symmetric_around_point_forecast():
    X_train = np.zeros((20, 3))
    y_train = np.zeros((20, 1))
    X_cal = np.zeros((20, 3))
    y_cal = np.array([[float(i % 5)] for i in range(20)])  # residuals 0..4 around ConstantModel(0)

    wrapped = SplitConformalForecaster(ConstantModel(0.0), alpha=0.2).fit(X_train, y_train, X_cal, y_cal)
    pred = wrapped.predict(np.zeros((5, 3)))
    width = pred.upper - pred.lower
    np.testing.assert_allclose(width, 2 * pred.q_hat)
    np.testing.assert_allclose(pred.upper - pred.point, pred.point - pred.lower)


def test_conformal_alpha_validation():
    with pytest.raises(ValueError):
        SplitConformalForecaster(NaiveForecaster(), alpha=0.0)
    with pytest.raises(ValueError):
        SplitConformalForecaster(NaiveForecaster(), alpha=1.0)


@pytest.mark.parametrize("alpha,seed", [(0.1, 0), (0.1, 1), (0.2, 0), (0.2, 1)])
def test_empirical_coverage_tracks_nominal_on_synthetic_data(alpha, seed):
    """The core claim of this whole project, checked directly.

    Synthetic data has *known* i.i.d. Gaussian noise, so residuals from a
    reasonable point forecaster are close to exchangeable and split
    conformal's coverage guarantee should approximately hold. We use a
    generous tolerance (+/- 8 percentage points) because with a finite
    test set (~160 points here) the empirical coverage is itself a random
    variable -- this test is checking "is calibration working at all",
    not "is it exact to 3 decimal places", which no finite sample could
    show anyway.
    """
    series = generate_synthetic(n=1000, period=24, noise_std=1.0, seed=seed)
    lookback = 24
    X, y = make_windows(series, lookback=lookback, horizon=1)

    n = len(X)
    n_train = int(n * 0.5)
    n_cal = int(n * 0.3)
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
    X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

    model = NaiveForecaster()  # cheap, deterministic, no training variance to control for
    wrapped = SplitConformalForecaster(model, alpha=alpha).fit(X_train, y_train, X_cal, y_cal)
    pred = wrapped.predict(X_test)
    result = evaluate_coverage(pred, y_test)

    assert abs(result["coverage_gap"]) < 0.08, (
        f"empirical coverage {result['empirical_coverage']:.3f} strayed too far from "
        f"nominal {result['nominal_coverage']:.3f} (alpha={alpha}, seed={seed})"
    )


def test_predict_before_fit_raises():
    wrapped = SplitConformalForecaster(NaiveForecaster(), alpha=0.1)
    with pytest.raises(RuntimeError):
        wrapped.predict(np.zeros((3, 3)))
