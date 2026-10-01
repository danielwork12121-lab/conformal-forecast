"""Tests for the sliding/growing-pool ACI variant (`SlidingWindowAdaptiveConformalForecaster`
in `forecasting/adaptive.py`).

Mirrors `tests/test_adaptive.py`'s discipline: hand-verified mechanics first,
then the actual statistical claim -- that folding observed test-time
residuals into the pool removes the fixed-pool variant's "can't propose an
interval wider than the calibration set's own max residual" ceiling, and
that this measurably closes *more* of a real coverage gap than the
fixed-pool variant does, under the same kind of distribution drift that
motivated building ACI in the first place.
"""
from __future__ import annotations

import numpy as np
import pytest

from forecasting.adaptive import (
    AdaptiveConformalForecaster,
    SlidingWindowAdaptiveConformalForecaster,
    evaluate_adaptive_coverage,
)
from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.models import NaiveForecaster


class ConstantModel:
    """Same trivial, hand-checkable fixture as `tests/test_adaptive.py`."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full((X.shape[0], 1), self.value, dtype=np.float64)


def test_sliding_window_validates_window():
    with pytest.raises(ValueError):
        SlidingWindowAdaptiveConformalForecaster(NaiveForecaster(), window=0)
    with pytest.raises(ValueError):
        SlidingWindowAdaptiveConformalForecaster(NaiveForecaster(), window=-3)
    # window=None (unbounded growing buffer) and any positive int are both fine.
    SlidingWindowAdaptiveConformalForecaster(NaiveForecaster(), window=None)
    SlidingWindowAdaptiveConformalForecaster(NaiveForecaster(), window=10)


def test_predict_sequential_before_fit_raises():
    forecaster = SlidingWindowAdaptiveConformalForecaster(NaiveForecaster(), alpha=0.1)
    with pytest.raises(RuntimeError):
        forecaster.predict_sequential(np.zeros((3, 3)), np.zeros(3))


def test_predict_sequential_requires_matching_lengths():
    X_cal = np.zeros((10, 3))
    y_cal = np.zeros((10, 1))
    forecaster = SlidingWindowAdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.1).fit(X_cal, y_cal, X_cal, y_cal)
    with pytest.raises(ValueError):
        forecaster.predict_sequential(np.zeros((5, 3)), np.zeros(4))


def test_pool_grows_by_exactly_one_each_step_when_unbounded():
    """With window=None, the pool should never shrink and should gain exactly
    one residual per test step -- and, critically, `pool_size[t]` (the size
    *used* to score step t) must equal n_cal + t, never n_cal + t + 1: if
    step t's own residual leaked into the pool before being scored, this
    would be off by one.
    """
    n_cal, n_test = 5, 10
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.zeros((n_cal, 1))
    forecaster = SlidingWindowAdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.2, window=None).fit(
        X_cal, y_cal, X_cal, y_cal
    )
    X_test = np.zeros((n_test, 3))
    y_test = np.arange(n_test, dtype=float)  # arbitrary, nonzero so residuals aren't trivially 0

    pred = forecaster.predict_sequential(X_test, y_test)

    assert list(pred.pool_size) == [n_cal + t for t in range(n_test)]


def test_pool_size_caps_at_window_when_sliding():
    """With window=k, the pool should grow by one per step until it hits k,
    then stay exactly at k (oldest residual evicted each further step).
    """
    n_cal, n_test, window = 5, 20, 8
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.zeros((n_cal, 1))
    forecaster = SlidingWindowAdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.2, window=window).fit(
        X_cal, y_cal, X_cal, y_cal
    )
    X_test = np.zeros((n_test, 3))
    y_test = np.arange(n_test, dtype=float)

    pred = forecaster.predict_sequential(X_test, y_test)

    expected = [min(n_cal + t, window) for t in range(n_test)]
    assert list(pred.pool_size) == expected
    assert pred.pool_size.max() == window


def test_window_smaller_than_seed_pool_shrinks_immediately():
    """Regression test for a real bug found and fixed in this run (v0.4):
    if `window` is smaller than the calibration pool it's seeded from, the
    pool must be truncated down to `window` *before* the first test step,
    not left to `predict_sequential`'s per-step append/evict loop to shrink
    it over time.

    Before the fix, each test step did exactly one append + (if over
    capacity) one evict -- a net change of zero once the pool was already
    at or above `window`. A pool seeded above `window` (e.g. 20 calibration
    residuals with `window=8`) therefore never shrank at all: `pool_size[t]`
    stayed at 20 for every single test step, and `window=8` silently behaved
    identically to `window=20` (or to `window=None`) for the entire test
    run. Caught via `scratch_window_sweep.py` during this run's own
    exploration (window=10/15/20 all gave byte-identical results on the
    airline dataset, which has 28 calibration windows), root-caused to this
    seeding bug, and fixed by truncating the seed pool to the most recent
    `window` calibration residuals in `predict_sequential` itself.
    """
    n_cal, n_test, window = 20, 5, 8
    X_cal = np.zeros((n_cal, 3))
    y_cal = np.arange(n_cal, dtype=float).reshape(-1, 1)  # distinct residuals, not all zero
    forecaster = SlidingWindowAdaptiveConformalForecaster(ConstantModel(0.0), alpha=0.2, window=window).fit(
        X_cal, y_cal, X_cal, y_cal
    )
    X_test = np.zeros((n_test, 3))
    y_test = np.arange(n_test, dtype=float)

    pred = forecaster.predict_sequential(X_test, y_test)

    # The pool must be at `window`, not `n_cal`, from the very first step --
    # and must never exceed `window` afterward either.
    assert pred.pool_size[0] == window
    assert list(pred.pool_size) == [window] * n_test
    assert pred.pool_size.max() == window

    # The truncation must keep the *most recent* `window` calibration
    # residuals (matching a real sliding window's semantics), not an
    # arbitrary subset: y_cal here is [0..19], so residuals are [0..19]
    # too (ConstantModel always predicts 0), and the most recent 8 are
    # [12..19] -- q_hat at step 0 (alpha=0.2, n=8) should reflect that
    # narrower, larger-valued pool, not the full [0..19] range.
    from forecasting.conformal import SplitConformalForecaster

    expected_seed_pool = np.arange(12, 20, dtype=float)  # last 8 of [0..19]
    expected_q_hat_0 = SplitConformalForecaster._quantile_with_finite_sample_correction(expected_seed_pool, 0.2)
    assert pred.q_hat[0] == pytest.approx(expected_q_hat_0)


def test_sliding_pool_hand_computation_matches_fixed_pool_until_pool_changes():
    """Hand-verify the exact quantile computation once the pool has grown,
    using the same deterministic setup as
    `test_adaptive.py::test_alpha_t_update_matches_hand_computation`
    (calibration residuals [0, 1, 2, 3, 4], alpha=0.2, gamma=0.1) so the
    numbers can be checked by hand, not just "the code ran".

    Step 0 is identical to the fixed-pool variant (the pool hasn't changed
    yet): q_hat = max(pool) = 4, y_test[0]=10 misses.
    After step 0, the sliding variant folds residual |10-0|=10 into the pool
    -> pool becomes [0,1,2,3,4,10] (n=6), which the *fixed*-pool variant
    never does. At step 1: level = ceil(7*0.88)/6 = ceil(6.16)/6 = 7/6 ->
    clipped to 1.0 -> q_hat = max(pool) = 10 (not 4, as the fixed-pool
    variant would use) -- this is exactly the mechanism that gives the
    sliding variant more headroom.
    """
    X_cal = np.zeros((5, 3))
    y_cal = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    forecaster = SlidingWindowAdaptiveConformalForecaster(
        ConstantModel(0.0), alpha=0.2, gamma=0.1, window=None
    ).fit(X_cal, y_cal, X_cal, y_cal)

    X_test = np.zeros((2, 3))
    y_test = np.array([10.0, 0.5])

    pred = forecaster.predict_sequential(X_test, y_test)

    assert pred.pool_size[0] == 5
    assert pred.q_hat[0] == pytest.approx(4.0)
    assert pred.errs[0] == 1.0

    assert pred.pool_size[1] == 6
    assert pred.alpha_t[1] == pytest.approx(0.12)
    assert pred.q_hat[1] == pytest.approx(10.0)  # the fixed-pool variant would give 4.0 here
    assert pred.errs[1] == 0.0


@pytest.mark.parametrize("alpha,seed", [(0.1, 0), (0.1, 1), (0.2, 0)])
def test_sliding_pool_coverage_tracks_nominal_on_stationary_synthetic_data(alpha, seed):
    """Sanity check mirroring `test_adaptive.py`'s equivalent: on ordinary
    exchangeable data, folding test residuals into the pool shouldn't make
    the easy case worse than the fixed-pool variant already handles well.
    """
    from forecasting.data import generate_synthetic, make_windows

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
    forecaster = SlidingWindowAdaptiveConformalForecaster(model, alpha=alpha, gamma=0.05, window=None).fit(
        X_train, y_train, X_cal, y_cal
    )
    pred = forecaster.predict_sequential(X_test, y_test.reshape(-1))
    result = evaluate_adaptive_coverage(pred, y_test)

    assert abs(result["coverage_gap"]) < 0.1, (
        f"sliding-pool ACI empirical coverage {result['empirical_coverage']:.3f} strayed too far from "
        f"nominal {result['nominal_coverage']:.3f} on stationary data (alpha={alpha}, seed={seed})"
    )


def test_sliding_pool_closes_more_of_a_real_coverage_gap_than_fixed_pool():
    """The actual point of this module, checked directly across several
    seeds, not asserted from one lucky run.

    Reuses the exact non-exchangeable drift scenario from
    `test_adaptive.py::test_adaptive_meaningfully_reduces_coverage_gap_under_distribution_drift`
    (calibration residuals from a stable regime, test-window truth drifting
    linearly away from a constant model's prediction) so the comparison is
    apples-to-apples with the fixed-pool variant's own documented result.

    The fixed-pool variant is capped at the calibration pool's own maximum
    residual -- it recovers *some* coverage by widening alpha_t, but can
    never propose an interval wider than that ceiling. Folding each step's
    own observed (post-drift, larger) residual into the pool removes that
    ceiling entirely. This is checked as a robust, aggregate improvement
    across seeds (the fixed-pool variant sometimes already gets close enough
    on a given seed that a single-seed comparison is noise-dominated -- see
    seed 3 below in exploration), not as a per-seed guarantee.
    """
    seeds = [0, 1, 2, 3, 4, 5]
    static_gaps, fixed_gaps, sliding_gaps = [], [], []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        n_cal = 150
        cal_residuals = np.abs(rng.normal(0, 2.0, size=n_cal))
        X_cal = np.zeros((n_cal, 3))
        y_cal = cal_residuals.reshape(-1, 1)

        n_test = 200
        drift = np.linspace(0, 6.0, n_test)
        noise = rng.normal(0, 0.5, size=n_test)
        y_test = (drift + noise).reshape(-1, 1)
        X_test = np.zeros((n_test, 3))
        alpha = 0.1

        static = SplitConformalForecaster(ConstantModel(0.0), alpha=alpha).fit(X_cal, y_cal, X_cal, y_cal)
        static_result = evaluate_coverage(static.predict(X_test), y_test)

        fixed = AdaptiveConformalForecaster(ConstantModel(0.0), alpha=alpha, gamma=0.03).fit(X_cal, y_cal, X_cal, y_cal)
        fixed_pred = fixed.predict_sequential(X_test, y_test.reshape(-1))
        fixed_result = evaluate_adaptive_coverage(fixed_pred, y_test)

        sliding = SlidingWindowAdaptiveConformalForecaster(
            ConstantModel(0.0), alpha=alpha, gamma=0.03, window=None
        ).fit(X_cal, y_cal, X_cal, y_cal)
        sliding_pred = sliding.predict_sequential(X_test, y_test.reshape(-1))
        sliding_result = evaluate_adaptive_coverage(sliding_pred, y_test)

        static_gaps.append(abs(static_result["coverage_gap"]))
        fixed_gaps.append(abs(fixed_result["coverage_gap"]))
        sliding_gaps.append(abs(sliding_result["coverage_gap"]))

        # Per-seed: the sliding variant should always be a large, robust
        # improvement over doing nothing (static), same bar the fixed-pool
        # variant's own test applies.
        assert sliding_gaps[-1] < static_gaps[-1] * 0.3, (
            f"seed={seed}: static gap={static_gaps[-1]:.3f}, sliding gap={sliding_gaps[-1]:.3f} "
            "-- sliding-pool ACI should reduce the gap by a large margin under this drift scenario"
        )

    # Aggregate: across seeds, the sliding variant should close substantially
    # more of the gap than the fixed-pool variant does -- the actual claim
    # this module makes over the one already shipped in v0.2.
    mean_fixed_gap = float(np.mean(fixed_gaps))
    mean_sliding_gap = float(np.mean(sliding_gaps))
    assert mean_sliding_gap < mean_fixed_gap * 0.6, (
        f"mean |coverage_gap| across {len(seeds)} seeds: fixed-pool ACI={mean_fixed_gap:.3f}, "
        f"sliding-pool ACI={mean_sliding_gap:.3f} -- expected sliding-pool ACI to close substantially "
        "more of the gap on average, not just match the fixed-pool variant"
    )
