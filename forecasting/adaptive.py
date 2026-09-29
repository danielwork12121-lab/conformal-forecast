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
        pool: list[float] = list(self._residuals)

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
