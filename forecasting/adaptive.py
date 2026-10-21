"""Adaptive Conformal Inference (ACI) -- online coverage correction.

Split conformal prediction (`forecasting.conformal`) assumes the calibration
and test residuals are *exchangeable*. The README documents a case where
that assumption visibly breaks: the from-scratch LSTM's calibration-window
residuals are systematically smaller than its test-window residuals (its own
error grows over the test period), so a 90%-nominal interval only achieves
~63% empirical coverage on the airline and synthetic-LSTM benchmarks.

Adaptive Conformal Inference (Gibbs & Candes, 2021, "Adaptive Conformal
Inference Under Distribution Shift") fixes exactly this, without retraining
the model and without assuming exchangeability at all: it treats the
miscoverage rate itself as a variable that adapts online, based only on
whether the *most recent* interval covered the truth.

    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

where err_t = 1 if step t's interval missed, else 0. Miss too often -> alpha_t
shrinks -> the interval widens (uses a higher quantile of the residual pool).
Cover too often -> alpha_t grows back -> the interval tightens toward the
target width. The paper proves (Prop 4.1) that this drives the long-run
*average* miscoverage rate to the nominal alpha, deterministically, no matter
how the underlying data (or a model's error pattern) shifts over time -- a
distribution-free guarantee that split conformal's exchangeability-based
guarantee cannot offer.

This module ships two variants:

- `AdaptiveConformalForecaster` -- the "fixed calibration pool" variant (see
  e.g. Zaffran et al. 2022, "Adaptive Conformal Predictions for Time
  Series"): only alpha_t moves each step; the residual pool used to look up
  a quantile is always the original calibration-window residuals. Simple and
  a small extension of `SplitConformalForecaster`, at the cost of a hard
  ceiling: it can never propose an interval wider than the *calibration
  pool's own maximum residual*, no matter how badly the test-time error has
  grown. That ceiling is exactly why this variant barely moves the needle on
  the airline benchmark (see its own docstring and the README): with only 28
  calibration windows, the finite-sample-corrected quantile at alpha=0.1 is
  already within ~2 residuals of that ceiling, so there's almost no headroom
  left to adapt into.
- `SlidingWindowAdaptiveConformalForecaster` (new) -- folds each step's own
  *observed* residual into the pool after scoring that step, so later
  quantile lookups see residuals from the test period itself, including any
  that are larger than anything seen during calibration. `window=None` keeps
  every observed residual forever (a "growing buffer"); `window=k` keeps
  only the most recent `k` (a true "sliding window", bounded memory, and
  better suited to a genuinely non-stationary process whose error scale
  might shrink again later, not just grow). This is a direct, honest attempt
  to remove the fixed-pool ceiling responsible for the airline shortfall --
  see `forecasting/experiment.py`'s `run_sliding_window_comparison` and the
  README for whether it actually does, measured on real data, not assumed.
- `RetrainingSlidingWindowAdaptiveConformalForecaster` (new) -- every
  variant above keeps the point model frozen at its initial fit for the
  entire test window; only the conformal layer adapts. This variant
  periodically *retrains* the point model itself from a growing history
  (original train+calibration windows plus every test-set window whose
  true value has been revealed so far), on top of the same sliding
  residual pool. See `forecasting/experiment.py`'s
  `run_online_retraining_comparison` and the README for whether letting
  the model itself learn online actually helps, measured on real data.

This module ships `evaluate_adaptive_coverage`, matching the "don't trust
the guarantee, measure it" discipline of `forecasting/conformal.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecasting.conformal import SplitConformalForecaster

ALPHA_T_CLIP_LOW = 1e-3
ALPHA_T_CLIP_HIGH = 1 - 1e-3


@dataclass
class AdaptiveConformalPrediction:
    point: np.ndarray  # (n, 1)
    lower: np.ndarray  # (n, 1)
    upper: np.ndarray  # (n, 1)
    q_hat: np.ndarray  # (n,) -- threshold actually used at each step (varies)
    alpha_t: np.ndarray  # (n,) -- the adapted miscoverage rate used at each step
    alpha: float  # nominal target, fixed
    gamma: float
    errs: np.ndarray  # (n,) -- 1.0 if that step's interval missed, else 0.0


@dataclass
class SlidingWindowAdaptivePrediction(AdaptiveConformalPrediction):
    pool_size: np.ndarray = None  # (n,) -- size of the residual pool used *at* each step


@dataclass
class RetrainingSlidingWindowAdaptivePrediction(SlidingWindowAdaptivePrediction):
    n_retrains: int = 0  # how many times the point model was actually refit during this test window
    retrained_at_steps: np.ndarray = None  # (n_retrains,) -- which test-set step indices triggered a refit


class AdaptiveConformalForecaster:
    """Wraps a point model + calibration residual pool with online ACI.

    Unlike `SplitConformalForecaster.predict`, this needs the *true* y
    values during prediction: ACI is inherently sequential, since alpha_t for
    step t+1 depends on whether step t's interval actually covered the
    truth. This models a realistic deployment where the actual value becomes
    known before the next forecast is due -- true of every dataset in this
    repo (you learn this month's real passenger count before forecasting
    next month's; you learn today's real temperature before forecasting
    tomorrow's).
    """

    def __init__(self, model, alpha: float = 0.1, gamma: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self.model = model
        self.alpha = alpha
        self.gamma = gamma
        self._residuals: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "AdaptiveConformalForecaster":
        self.model.fit(X_train, y_train)
        cal_pred = self.model.predict(X_cal)
        self._residuals = np.abs(np.asarray(y_cal) - cal_pred).reshape(-1)
        return self

    def predict_sequential(self, X_test: np.ndarray, y_test: np.ndarray) -> AdaptiveConformalPrediction:
        if self._residuals is None:
            raise RuntimeError("AdaptiveConformalForecaster.predict_sequential called before fit()")

        X_test = np.asarray(X_test)
        y_test_flat = np.asarray(y_test).reshape(-1)
        n = X_test.shape[0]
        if n != len(y_test_flat):
            raise ValueError("X_test and y_test must have the same length")
        if n == 0:
            raise ValueError("Need at least one test point to run ACI sequentially")

        point = self.model.predict(X_test).reshape(-1)

        alpha_t = self.alpha
        alphas = np.empty(n)
        q_hats = np.empty(n)
        lower = np.empty(n)
        upper = np.empty(n)
        errs = np.empty(n)

        for t in range(n):
            clipped_alpha = min(max(alpha_t, ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)
            q_hat = SplitConformalForecaster._quantile_with_finite_sample_correction(
                self._residuals, clipped_alpha
            )
            lo, hi = point[t] - q_hat, point[t] + q_hat
            err = 0.0 if (lo <= y_test_flat[t] <= hi) else 1.0

            alphas[t] = clipped_alpha
            q_hats[t] = q_hat
            lower[t] = lo
            upper[t] = hi
            errs[t] = err

            # Update *before* clipping-for-use next iteration; clip the
            # running state too so a long streak of misses/hits can't drive
            # alpha_t arbitrarily far from a usable range.
            alpha_t = min(max(alpha_t + self.gamma * (self.alpha - err), ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)

        return AdaptiveConformalPrediction(
            point=point.reshape(-1, 1),
            lower=lower.reshape(-1, 1),
            upper=upper.reshape(-1, 1),
            q_hat=q_hats,
            alpha_t=alphas,
            alpha=self.alpha,
            gamma=self.gamma,
            errs=errs,
        )


class SlidingWindowAdaptiveConformalForecaster(AdaptiveConformalForecaster):
    """ACI whose residual pool is updated online with each step's own
    observed residual, instead of staying frozen at the calibration set.

    `window=None` (default): a *growing buffer* -- every observed residual
    (calibration + all test steps seen so far) stays in the pool forever, so
    the pool can only ever grow and the achievable quantile can only ever
    rise to match the largest residual actually observed anywhere so far.

    `window=k` (positive int): a true *sliding window* of the most recent
    `k` residuals (calibration residuals seed it, then test residuals push
    the oldest ones out once the pool reaches size `k`). This additionally
    lets the achievable interval *shrink back down* if the process becomes
    easier again later, which a growing buffer cannot do (once a large
    residual enters a growing buffer it never leaves).

    Ordering per step, kept strict so no step ever sees its own answer early:
    1. Compute q_hat / the interval from the pool *as it stood before this
       step*.
    2. Score the interval against the true y_test[t] (this is what alpha_t's
       update already did in the base class).
    3. Only now fold this step's own residual into the pool, for use by
       step t+1 onward.
    """

    def __init__(self, model, alpha: float = 0.1, gamma: float = 0.05, window: int | None = None):
        super().__init__(model, alpha=alpha, gamma=gamma)
        if window is not None and window <= 0:
            raise ValueError("window must be a positive integer, or None for an unbounded growing buffer")
        self.window = window

    def predict_sequential(self, X_test: np.ndarray, y_test: np.ndarray) -> SlidingWindowAdaptivePrediction:
        if self._residuals is None:
            raise RuntimeError(
                "SlidingWindowAdaptiveConformalForecaster.predict_sequential called before fit()"
            )

        X_test = np.asarray(X_test)
        y_test_flat = np.asarray(y_test).reshape(-1)
        n = X_test.shape[0]
        if n != len(y_test_flat):
            raise ValueError("X_test and y_test must have the same length")
        if n == 0:
            raise ValueError("Need at least one test point to run ACI sequentially")

        point = self.model.predict(X_test).reshape(-1)

        # A mutable working copy -- the original calibration pool (self._residuals)
        # is never mutated, so re-running predict_sequential (e.g. from a test)
        # always starts from the same seed pool.
        #
        # If window is smaller than the calibration pool itself, seed with
        # only the most recent `window` calibration residuals. Without this
        # truncation, a bounded window would never actually bind: each test
        # step appends exactly one residual and (once over capacity) evicts
        # exactly one, a net change of zero, so a pool that starts *above*
        # `window` stays at that starting size forever instead of shrinking
        # down to it. (This was a real bug in the first cut of this class --
        # `window=10` silently behaved identically to `window=30` on a
        # dataset with a 28-point calibration set, because the pool never
        # got the chance to shrink. Caught via `scratch_window_sweep.py`
        # during this run's own exploration, fixed here, and pinned down by
        # `tests/test_sliding_window.py::test_window_smaller_than_seed_pool_shrinks_immediately`
        # so it can't silently regress.)
        residuals = list(self._residuals)
        if self.window is not None and len(residuals) > self.window:
            residuals = residuals[-self.window :]
        pool: list[float] = residuals

        alpha_t = self.alpha
        alphas = np.empty(n)
        q_hats = np.empty(n)
        lower = np.empty(n)
        upper = np.empty(n)
        errs = np.empty(n)
        pool_sizes = np.empty(n, dtype=int)

        for t in range(n):
            pool_sizes[t] = len(pool)
            clipped_alpha = min(max(alpha_t, ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)
            q_hat = SplitConformalForecaster._quantile_with_finite_sample_correction(
                np.asarray(pool), clipped_alpha
            )
            lo, hi = point[t] - q_hat, point[t] + q_hat
            err = 0.0 if (lo <= y_test_flat[t] <= hi) else 1.0

            alphas[t] = clipped_alpha
            q_hats[t] = q_hat
            lower[t] = lo
            upper[t] = hi
            errs[t] = err

            # Fold this step's own residual in *after* scoring it, so q_hat
            # at step t never has access to y_test[t] itself.
            pool.append(float(abs(y_test_flat[t] - point[t])))
            if self.window is not None and len(pool) > self.window:
                pool.pop(0)

            alpha_t = min(max(alpha_t + self.gamma * (self.alpha - err), ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)

        return SlidingWindowAdaptivePrediction(
            point=point.reshape(-1, 1),
            lower=lower.reshape(-1, 1),
            upper=upper.reshape(-1, 1),
            q_hat=q_hats,
            alpha_t=alphas,
            alpha=self.alpha,
            gamma=self.gamma,
            errs=errs,
            pool_size=pool_sizes,
        )


class RetrainingSlidingWindowAdaptiveConformalForecaster(SlidingWindowAdaptiveConformalForecaster):
    """ACI whose point *model* is periodically retrained online, on top of
    the sliding-pool mechanism `SlidingWindowAdaptiveConformalForecaster`
    already provides for the residual pool.

    This is the README's long-open "What's next" item: every ACI/EnbPI
    variant shipped so far adapts *the calibration* to a model that stays
    frozen at its initial fit for the entire test window (the LSTM that
    predicted January 1960's airline passengers is the exact same LSTM,
    with the exact same weights, used for every test step through
    December 1960). This class instead lets the *model itself* keep
    learning from newly-revealed test-set truth, not just the conformal
    layer wrapped around it.

    Honest scope limitation, stated plainly rather than hidden: retraining
    a from-scratch neural model (Adam + early stopping, up to
    `max_epochs` epochs) at *every* test step would be correct but far too
    slow to run on `temperature`'s ~650-point test window in this
    CPU-constrained sandbox, and would also mean most "retrains" happen on
    an almost-unchanged training set (one extra point) for no measurable
    benefit. `retrain_every` instead retrains periodically -- a real,
    honest design trade-off (coarser-grained online learning, not
    continuous), not a disguised version of the frozen-model case.
    `retrain_every=1` *is* the literal "retrain at every step" case for a
    dataset/budget where that's actually affordable (used by the test
    suite's small synthetic fixtures); real benchmark runs pick a coarser
    value (see `forecasting/experiment.py::run_online_retraining_comparison`).

    `model_factory` -- a zero-argument callable returning a *fresh*,
    unfitted model instance (e.g. `lambda: DeltaWrapper(LSTMForecaster(seed=0))`)
    -- is required (rather than accepting an already-fitted model instance,
    as every other class in this module does) because retraining a from-
    scratch model means constructing a brand new one each time, not
    calling `.fit()' again on an object that has already-trained weights
    sitting in it as a (non-guaranteed) starting point; a fresh instance
    per retrain keeps every retrain's optimization identical in kind to
    the very first fit, so "did retraining help" isn't entangled with "did
    warm-starting from old weights help instead."

    Ordering per step, same strictness as the parent class: compute this
    step's interval from the model/pool *as they stood before this step*,
    score it against the true y, fold the residual into the pool, and only
    *after* scoring does a newly-revealed y become eligible to retrain on
    (so a retrain at step t uses y_test[0:t], never y_test[t] itself).
    """

    def __init__(
        self,
        model_factory,
        alpha: float = 0.1,
        gamma: float = 0.05,
        window: int | None = None,
        retrain_every: int | None = None,
    ):
        if retrain_every is not None and retrain_every <= 0:
            raise ValueError("retrain_every must be a positive integer, or None to disable retraining entirely")
        self.model_factory = model_factory
        self.retrain_every = retrain_every
        self.n_retrains = 0
        self._X_history: np.ndarray | None = None
        self._y_history: np.ndarray | None = None
        super().__init__(model_factory(), alpha=alpha, gamma=gamma, window=window)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "RetrainingSlidingWindowAdaptiveConformalForecaster":
        super().fit(X_train, y_train, X_cal, y_cal)
        # Everything the model is allowed to retrain on at step 0: the
        # original train+calibration windows. Test-set windows are appended
        # to this history one at a time, only once their true y is known.
        self._X_history = np.concatenate([np.asarray(X_train), np.asarray(X_cal)], axis=0)
        self._y_history = np.concatenate([np.asarray(y_train), np.asarray(y_cal)], axis=0)
        return self

    def predict_sequential(self, X_test: np.ndarray, y_test: np.ndarray) -> RetrainingSlidingWindowAdaptivePrediction:
        if self._residuals is None or self._X_history is None:
            raise RuntimeError(
                "RetrainingSlidingWindowAdaptiveConformalForecaster.predict_sequential called before fit()"
            )

        X_test = np.asarray(X_test)
        y_test_flat = np.asarray(y_test).reshape(-1)
        n = X_test.shape[0]
        if n != len(y_test_flat):
            raise ValueError("X_test and y_test must have the same length")
        if n == 0:
            raise ValueError("Need at least one test point to run ACI sequentially")
        # y_history/y_cal/y_train all have shape (m, horizon); this module
        # (like the rest of the repo's bundled-dataset path) only ever
        # exercises horizon=1, so a single test-set column reshapes cleanly
        # onto the same (m, 1) shape for history concatenation.
        y_test_col = y_test_flat.reshape(-1, 1)

        residuals = list(self._residuals)
        if self.window is not None and len(residuals) > self.window:
            residuals = residuals[-self.window :]
        pool: list[float] = residuals

        alpha_t = self.alpha
        alphas = np.empty(n)
        q_hats = np.empty(n)
        lower = np.empty(n)
        upper = np.empty(n)
        errs = np.empty(n)
        points = np.empty(n)
        pool_sizes = np.empty(n, dtype=int)
        retrained_at_steps: list[int] = []

        for t in range(n):
            if self.retrain_every is not None and t > 0 and t % self.retrain_every == 0:
                X_hist = np.concatenate([self._X_history, X_test[:t]], axis=0)
                y_hist = np.concatenate([self._y_history, y_test_col[:t]], axis=0)
                self.model = self.model_factory()
                self.model.fit(X_hist, y_hist)
                self.n_retrains += 1
                retrained_at_steps.append(t)

            pool_sizes[t] = len(pool)
            point_t = float(self.model.predict(X_test[t : t + 1]).reshape(-1)[0])
            points[t] = point_t

            clipped_alpha = min(max(alpha_t, ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)
            q_hat = SplitConformalForecaster._quantile_with_finite_sample_correction(np.asarray(pool), clipped_alpha)
            lo, hi = point_t - q_hat, point_t + q_hat
            err = 0.0 if (lo <= y_test_flat[t] <= hi) else 1.0

            alphas[t] = clipped_alpha
            q_hats[t] = q_hat
            lower[t] = lo
            upper[t] = hi
            errs[t] = err

            pool.append(float(abs(y_test_flat[t] - point_t)))
            if self.window is not None and len(pool) > self.window:
                pool.pop(0)

            alpha_t = min(max(alpha_t + self.gamma * (self.alpha - err), ALPHA_T_CLIP_LOW), ALPHA_T_CLIP_HIGH)

        return RetrainingSlidingWindowAdaptivePrediction(
            point=points.reshape(-1, 1),
            lower=lower.reshape(-1, 1),
            upper=upper.reshape(-1, 1),
            q_hat=q_hats,
            alpha_t=alphas,
            alpha=self.alpha,
            gamma=self.gamma,
            errs=errs,
            pool_size=pool_sizes,
            n_retrains=self.n_retrains,
            retrained_at_steps=np.array(retrained_at_steps, dtype=int),
        )


def evaluate_adaptive_coverage(prediction: AdaptiveConformalPrediction, y_true: np.ndarray) -> dict:
    """Same discipline as `forecasting.conformal.evaluate_coverage`: measure,
    don't assert. Also reports `mean_error_rate`, which is the exact
    quantity ACI's Prop 4.1 guarantee is about (it should track `alpha`
    more reliably, over a long enough run, than plain empirical coverage
    tracks nominal coverage under distribution shift -- these two numbers
    can legitimately differ slightly since one is 1 - the other only when
    intervals are computed the same way throughout, which ACI deliberately
    does not do).
    """
    from forecasting.metrics import empirical_coverage, mean_interval_width

    y_true_col = np.asarray(y_true).reshape(-1, 1)
    nominal = 1 - prediction.alpha
    empirical = empirical_coverage(y_true_col, prediction.lower, prediction.upper)
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "mean_interval_width": mean_interval_width(prediction.lower, prediction.upper),
        "mean_q_hat": float(np.mean(prediction.q_hat)),
        "mean_error_rate": float(np.mean(prediction.errs)),
        "final_alpha_t": float(prediction.alpha_t[-1]),
        "n_test": int(y_true_col.shape[0]),
    }
