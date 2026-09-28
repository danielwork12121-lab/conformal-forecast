"""Tests for Adaptive Conformal Inference (`forecasting/adaptive.py`).

Mirrors the discipline of `tests/test_conformal.py`: these don't just check
that the code runs, they check the actual statistical claims -- both that
ACI behaves sensibly in the easy (exchangeable) case, and, more importantly,
that it *measurably helps* in exactly the kind of non-exchangeable,
drifting-error scenario that motivated building it (the real LSTM finding
documented in the README and in `forecasting/conformal.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.adaptive import (
    ALPHA_T_CLIP_HIGH,
    ALPHA_T_CLIP_LOW,
    AdaptiveConformalForecaster,
    evaluate_adaptive_coverage,
)
from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.data import generate_synthetic, make_windows
from forecasting.models import NaiveForecaster


class ConstantModel:
    """Same trivial fixture as `tests/test_conformal.py`: predicts a fixed
    constant, so calibration residuals are just |y_cal| and test residuals
    are just |y_test|, and every number in these tests can be hand-checked.
    """

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full((X.shape[0], 1), self.value, dtype=np.float64)


def test_adaptive_validates_alpha_and_gamma():
    with pytest.raises(ValueError):
        AdaptiveConformalForecaster(NaiveForecaster(), alpha=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalForecaster(NaiveForecaster(), alpha=1.0)
    with pytest.raises(ValueError):
        AdaptiveConformalForecaster(NaiveForecaster(), alpha=0.1, gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalForecaster(NaiveForecaster(), alpha=0.1, gamma=-0.01)


def test_predict_sequential_before_fit_raises():
    forecaster = AdaptiveConformalForecaster(NaiveForecaster(), alpha=0.1)
    with pytest.raises(RuntimeError):
        forecaster.predict_sequential(np.zeros((3, 3)), np.zeros(3))


def test_predict_sequential_requires_matching_lengths():
    X_cal = np.zeros((10, 3))
    y_cal = np.zeros((10, 1))
    forecaster = AdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.1).fit(X_cal, y_cal, X_cal, y_cal)
    with pytest.raises(ValueError):
        forecaster.predict_sequential(np.zeros((5, 3)), np.zeros(4))


def test_alpha_t_update_matches_hand_computation():
    """Hand-verify the exact ACI update rule on a short, deterministic run.

    Calibration residuals are exactly [0, 1, 2, 3, 4] (as in
    `test_conformal.py`'s quantile test), so q_hat at any alpha level is
    easy to compute by hand: for n=5, level = ceil(6*(1-alpha_t))/5, and
    q_hat is the sorted residual at that level.

    With alpha=0.2, gamma=0.1, the very first step uses alpha_t = alpha =
    0.2 exactly -> level = ceil(6*0.8)/5 = ceil(4.8)/5 = 5/5 = 1.0 -> the
    max residual, 4. The constant model predicts 0, so the interval is
    [-4, 4]. y_test[0] = 10 misses (10 > 4) -> err=1 -> alpha_t updates to
    0.2 + 0.1*(0.2 - 1) = 0.2 - 0.08 = 0.12 for step 2.
    """
    X_cal = np.zeros((5, 3))
    y_cal = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    forecaster = AdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.2, gamma=0.1).fit(X_cal, y_cal, X_cal, y_cal)

    X_test = np.zeros((2, 3))
    y_test = np.array([10.0, 0.5])  # step 0 misses badly, step 1 should be well inside

    pred = forecaster.predict_sequential(X_test, y_test)

    assert pred.alpha_t[0] == pytest.approx(0.2)
    assert pred.q_hat[0] == pytest.approx(4.0)
    assert pred.errs[0] == 1.0
    assert pred.alpha_t[1] == pytest.approx(0.12)
    # level = ceil(6 * 0.88) / 5 = ceil(5.28) / 5 = 6/5 -> clipped to 1.0 -> still the max (4).
    assert pred.q_hat[1] == pytest.approx(4.0)
    assert pred.errs[1] == 0.0


def test_alpha_t_stays_within_clip_bounds_under_a_long_run_of_misses():
    X_cal = np.zeros((5, 3))
    y_cal = np.array([[0.0], [0.0], [0.0], [0.0], [1.0]])
    forecaster = AdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.1, gamma=0.2).fit(X_cal, y_cal, X_cal, y_cal)

    n = 50
    X_test = np.zeros((n, 3))
    y_test = np.full(n, 100.0)  # every step misses badly -> alpha_t driven down every step

    pred = forecaster.predict_sequential(X_test, y_test)

    assert np.all(pred.alpha_t >= ALPHA_T_CLIP_LOW)
    assert np.all(pred.alpha_t <= ALPHA_T_CLIP_HIGH)
    assert pred.alpha_t[-1] == pytest.approx(ALPHA_T_CLIP_LOW)


@pytest.mark.parametrize("alpha,seed", [(0.1, 0), (0.1, 1), (0.2, 0)])
def test_adaptive_coverage_tracks_nominal_on_stationary_synthetic_data(alpha, seed):
    """Sanity check: on ordinary exchangeable data (the same synthetic
    generator `test_conformal.py` uses), ACI should do about as well as
    static split conformal -- it shouldn't make the easy case worse.
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

    model = NaiveForecaster()
    forecaster = AdaptiveConformalForecaster(model, alpha=alpha, gamma=0.05).fit(X_train, y_train, X_cal, y_cal)
    pred = forecaster.predict_sequential(X_test, y_test.reshape(-1))
    result = evaluate_adaptive_coverage(pred, y_test)

    assert abs(result["coverage_gap"]) < 0.1, (
        f"ACI empirical coverage {result['empirical_coverage']:.3f} strayed too far from "
        f"nominal {result['nominal_coverage']:.3f} on stationary data (alpha={alpha}, seed={seed})"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_adaptive_meaningfully_reduces_coverage_gap_under_distribution_drift(seed):
    """The actual point of this module, checked directly, not asserted.

    Construct a genuinely non-exchangeable scenario, structurally like the
    real LSTM finding in the README: calibration-window residuals come from
    a stable regime, but the true value drifts further from the model's
    (constant) prediction as the test window goes on, so the model's own
    absolute error grows between calibration and test -- split conformal's
    exchangeability assumption is directly violated by construction.

    Static split conformal, calibrated once on the easy regime, should
    under-cover badly and consistently. ACI, which reacts to each miss by
    widening (up to the calibration pool's own max residual -- it cannot
    invent data it never saw, which is an honest, documented limitation of
    this "fixed calibration pool" variant, not a bug), should recover a
    real, substantial fraction of that lost coverage. It is not expected to
    fully close the gap to nominal here -- that would need a wider
    calibration pool or a sliding-window variant, both future work -- so
    this test checks for a large, robust *improvement*, not a perfect fix.
    """
    rng = np.random.default_rng(seed)
    n_cal = 150
    cal_residuals = np.abs(rng.normal(0, 2.0, size=n_cal))
    X_cal = np.zeros((n_cal, 3))
    y_cal = cal_residuals.reshape(-1, 1)

    n_test = 200
    drift = np.linspace(0, 6.0, n_test)  # the model's true error grows over the test window
    noise = rng.normal(0, 0.5, size=n_test)
    y_test = (drift + noise).reshape(-1, 1)
    X_test = np.zeros((n_test, 3))

    alpha = 0.1

    static = SplitConformalForecaster(ConstantModel(0.0), alpha=alpha).fit(X_cal, y_cal, X_cal, y_cal)
    static_pred = static.predict(X_test)
    static_result = evaluate_coverage(static_pred, y_test)

    adaptive = AdaptiveConformalForecaster(ConstantModel(0.0), alpha=alpha, gamma=0.03).fit(X_cal, y_cal, X_cal, y_cal)
    adaptive_pred = adaptive.predict_sequential(X_test, y_test.reshape(-1))
    adaptive_result = evaluate_adaptive_coverage(adaptive_pred, y_test)

    # Sanity check the scenario actually does what it claims: static conformal
    # should visibly under-cover here (this isn't the point being tested, but
    # if it stops being true the rest of the assertions are meaningless).
    assert static_result["coverage_gap"] < -0.2, "test scenario no longer produces real under-coverage"

    # The actual claim: ACI closes a large, robust fraction of that gap.
    assert abs(adaptive_result["coverage_gap"]) < abs(static_result["coverage_gap"]) * 0.7, (
        f"seed={seed}: static coverage_gap={static_result['coverage_gap']:.3f}, "
        f"adaptive coverage_gap={adaptive_result['coverage_gap']:.3f} "
        "-- ACI should reduce the gap by a large, robust margin under this drift scenario"
    )
