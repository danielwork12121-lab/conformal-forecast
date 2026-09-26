"""Split conformal prediction for point-forecast models.

Most time-series libraries only ever hand back a point forecast. This
module wraps *any* model implementing the `fit(X, y)` / `predict(X)`
interface in `forecasting.models` and produces a calibrated prediction
interval alongside the point forecast -- and, critically, this file also
contains the machinery to actually *check* that the calibration worked
(empirical coverage on held-out data), rather than just asserting it.

How it works (split conformal, Lei et al. 2018 / Vovk et al. 2005):
    1. The wrapped model is fit on the *training* split only.
    2. Absolute residuals |y - y_hat| are computed on a separate
       *calibration* split the model never trained on.
    3. For a desired miscoverage rate alpha (e.g. 0.1 for 90% intervals),
       q_hat is the ceil((n_cal + 1) * (1 - alpha)) / n_cal empirical
       quantile of those residuals (the "+1" finite-sample correction is
       what gives the finite-sample marginal coverage guarantee, not just
       an asymptotic one).
    4. Every new prediction gets interval [y_hat - q_hat, y_hat + q_hat].

Honesty note on the guarantee: the textbook conformal coverage guarantee
assumes the calibration and test residuals are *exchangeable*. For a
genuinely non-stationary time series (residual variance drifting over
time, structural breaks, etc.) that assumption is only approximate, not
exact -- this is a real, known limitation of applying split conformal to
time series "as is" (as opposed to time-series-specific variants such as
EnbPI or adaptive conformal inference, which are natural follow-ups for
this project, not yet implemented here). That's exactly why this module
ships `evaluate_coverage`: don't trust the guarantee, measure it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalPrediction:
    point: np.ndarray  # (n, horizon)
    lower: np.ndarray  # (n, horizon)
    upper: np.ndarray  # (n, horizon)
    q_hat: float
    alpha: float


class SplitConformalForecaster:
    """Wraps a point-forecast model to also emit calibrated intervals."""

    def __init__(self, model, alpha: float = 0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.model = model
        self.alpha = alpha
        self.q_hat: float | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "SplitConformalForecaster":
        self.model.fit(X_train, y_train)
        cal_pred = self.model.predict(X_cal)
        residuals = np.abs(y_cal - cal_pred).reshape(-1)
        self.q_hat = self._quantile_with_finite_sample_correction(residuals, self.alpha)
        return self

    @staticmethod
    def _quantile_with_finite_sample_correction(residuals: np.ndarray, alpha: float) -> float:
        n = len(residuals)
        if n == 0:
            raise ValueError("Need at least one calibration residual")
        # ceil((n+1)(1-alpha)) / n, clipped to 1.0 -> take the max residual
        # (can't promise better than "the interval containing everything
        # seen so far" once n is small relative to (1-alpha)).
        level = math.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        sorted_r = np.sort(residuals)
        idx = min(int(math.ceil(level * n)) - 1, n - 1)
        idx = max(idx, 0)
        return float(sorted_r[idx])

    def predict(self, X: np.ndarray) -> ConformalPrediction:
        if self.q_hat is None:
            raise RuntimeError("SplitConformalForecaster.predict called before fit()")
        point = self.model.predict(X)
        lower = point - self.q_hat
        upper = point + self.q_hat
        return ConformalPrediction(point=point, lower=lower, upper=upper, q_hat=self.q_hat, alpha=self.alpha)


def evaluate_coverage(prediction: ConformalPrediction, y_true: np.ndarray) -> dict:
    """Compute empirical coverage + width and compare against the nominal target.

    Returns a plain dict (not just a bool) so a caller can see *how far
    off* an approximate guarantee actually was, not just pass/fail.
    """
    from forecasting.metrics import empirical_coverage, mean_interval_width

    nominal = 1 - prediction.alpha
    empirical = empirical_coverage(y_true, prediction.lower, prediction.upper)
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal,
        "mean_interval_width": mean_interval_width(prediction.lower, prediction.upper),
        "q_hat": prediction.q_hat,
        "n_test": int(np.asarray(y_true).shape[0]),
    }
