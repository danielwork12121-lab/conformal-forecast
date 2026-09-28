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

This implementation is the "fixed calibration pool" variant (see e.g.
Zaffran et al. 2022, "Adaptive Conformal Predictions for Time Series"):
rather than maintaining a growing/sliding buffer of nonconformity scores, it
reuses the same calibration-residual pool from `SplitConformalForecaster`
and only lets alpha_t (and therefore which quantile of that fixed pool is
used) move each step. This keeps the implementation a small, honest
extension of the existing split-conformal machinery rather than a separate
system, at the cost of not adapting to a genuinely shifting *residual scale*
mid-test the way a sliding-window variant would -- a reasonable next step,
not implemented here (see README).

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
