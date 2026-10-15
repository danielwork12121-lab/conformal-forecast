"""EnbPI -- ensemble batch prediction intervals for online conformal forecasting.

(Xu & Xie, 2021, "Conformal Prediction Interval for Dynamic Time-Series.")

`forecasting.conformal`'s own module docstring names EnbPI directly as a
"natural follow-up for this project, not yet implemented here" -- this
module is that follow-up, and the README's "What's next" section flagged
it as the same open item. Like `forecasting.adaptive`'s two ACI variants,
this is a second, independent answer to split conformal's exchangeability
problem, but via a structurally different mechanism:

- ACI (`forecasting.adaptive`) keeps a *single* fitted model and adapts the
  *significance level* alpha_t online, based on whether the most recent
  interval covered the truth.
- EnbPI keeps a *fixed* significance level alpha, and instead trains an
  *ensemble* of B models, each on an independent bootstrap resample of the
  training data. Every ensemble member has some training points it never
  saw (its bootstrap draw happened not to include them) -- call those its
  "out-of-bag" points. Averaging the predictions of only the out-of-bag
  models for a given training point gives a genuinely out-of-sample
  ("leave-one-out", LOO) prediction for it, with a residual that is honest
  about that model family's real test-time error, not optimistic the way
  an in-sample residual would be. Those LOO residuals seed an online
  residual pool; as each test point is scored and then revealed, its own
  observed residual is folded into the pool (optionally evicting the
  oldest one, exactly as `forecasting.adaptive`'s sliding-window ACI
  already does), so the quantile used for tomorrow's interval always
  reflects a recent window of real, out-of-sample error -- all without
  ever retraining a model or touching an "alpha_t" at all.

Two deliberate simplifications versus the original paper, both disclosed
here rather than silently assumed:

1. The paper's own default construction keeps the residual pool at a
   *constant* size -- every new test residual evicts the single oldest one,
   starting from the moment the pool is seeded. This module instead
   defaults `window=None` (an unbounded, ever-growing pool), matching this
   repo's own `SlidingWindowAdaptiveConformalForecaster` convention, so the
   two "does a bounded window help" questions stay directly comparable
   with the same calling convention. Pass an explicit `window=<pool size
   at fit() time>` to reproduce the paper's own constant-memory behavior.
2. The paper computes two one-sided residual quantiles from *signed*
   residuals, giving an asymmetric interval. This module uses this
   project's existing house convention instead (absolute residuals, one
   symmetric quantile -- see `forecasting.conformal`'s
   `_quantile_with_finite_sample_correction`), so EnbPI's interval-width
   numbers are directly comparable to static split conformal's and ACI's
   in the same README tables. This is a real, acknowledged departure from
   the paper's exact method, not an attempt to reproduce it bit-for-bit.

Calibration-split handling: unlike `SplitConformalForecaster`/
`AdaptiveConformalForecaster`, EnbPI's own residual pool doesn't strictly
need a separate calibration split at all -- the bootstrap ensemble's
out-of-bag structure already gives genuinely out-of-sample residuals from
the training set alone. This module still accepts `(X_cal, y_cal)` in
`fit()`, purely so `EnbPIForecaster` exposes the same
`fit(X_train, y_train, X_cal, y_cal)` shape every other model/wrapper in
this repo does (for drop-in use in `forecasting.experiment`'s comparison
functions) -- and folds the calibration residuals in too, since every
ensemble member is trivially out-of-bag for calibration data (none of
them ever trained on it), giving an even stronger out-of-sample residual
than the training LOO ones. Nothing about the algorithm requires this;
it's just a way to spend the same held-out data budget the other methods
in this comparison already spend, honestly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forecasting.conformal import SplitConformalForecaster

ModelFactory = Callable[[int], object]  # (seed) -> fresh model instance implementing fit/predict


@dataclass
class EnbPIPrediction:
    point: np.ndarray  # (n, 1) -- full-ensemble mean prediction at each test step
    lower: np.ndarray  # (n, 1)
    upper: np.ndarray  # (n, 1)
    q_hat: np.ndarray  # (n,) -- threshold used at each step (fixed alpha, but pool moves)
    alpha: float  # nominal target, fixed throughout (no alpha_t adaptation, unlike ACI)
    errs: np.ndarray  # (n,) -- 1.0 if that step's interval missed, else 0.0
    pool_size: np.ndarray  # (n,) -- residual pool size *at* each step
    n_bootstrap: int
    n_loo_skipped: int  # training points with no out-of-bag model at all (excluded, not biased in)


class EnbPIForecaster:
    """Ensemble batch prediction intervals (Xu & Xie, 2021).

    `model_factory(seed) -> model` must return a fresh, unfitted model
    instance implementing this repo's usual `fit(X, y)` / `predict(X)`
    interface (see `forecasting.models`) each time it's called, so the
    `n_bootstrap` ensemble members are independently initialized as well as
    independently resampled -- two different random LSTM weight inits
    trained on the same bootstrap draw would still only be "one source of
    randomness" apart, which understates the ensemble's real diversity.
    """

    def __init__(
        self,
        model_factory: ModelFactory,
        n_bootstrap: int = 20,
        alpha: float = 0.1,
        window: int | None = None,
        seed: int = 0,
    ):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2 (need >=1 out-of-bag model per point)")
        if window is not None and window <= 0:
            raise ValueError("window must be a positive integer, or None for an unbounded growing buffer")
        self.model_factory = model_factory
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.window = window
        self.seed = seed
        self._models: list = []
        self._residuals: list[float] | None = None
        self.n_loo_skipped_: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "EnbPIForecaster":
        rng = np.random.default_rng(self.seed)
        X_train = np.asarray(X_train)
        y_train_flat = np.asarray(y_train).reshape(-1)
        n = X_train.shape[0]

        self._models = []
        in_bag = np.zeros((self.n_bootstrap, n), dtype=bool)
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            in_bag[b, np.unique(idx)] = True
            model = self.model_factory(self.seed * 1000 + b + 1)
            model.fit(X_train[idx], y_train[idx])
            self._models.append(model)
        out_of_bag = ~in_bag  # (n_bootstrap, n) -- True where model b never saw training point i

        # Out-of-bag ("leave-one-out") residuals on the training set: for
        # each training point, average only the predictions of models that
        # never trained on it.
        train_preds = np.stack([m.predict(X_train).reshape(-1) for m in self._models], axis=0)  # (B, n)
        loo_residuals: list[float] = []
        n_skipped = 0
        for i in range(n):
            oob_for_i = out_of_bag[:, i]
            if not oob_for_i.any():
                # Every bootstrap draw happened to include this point --
                # possible but rare for reasonable (n_bootstrap, n). Skip
                # it rather than fall back to an in-sample (optimistic)
                # prediction, which would quietly bias the residual pool
                # toward looking better-calibrated than the ensemble
                # actually is out-of-sample.
                n_skipped += 1
                continue
            loo_pred_i = float(train_preds[oob_for_i, i].mean())
            loo_residuals.append(abs(y_train_flat[i] - loo_pred_i))
        self.n_loo_skipped_ = n_skipped

        # Calibration residuals: every ensemble member is out-of-bag for
        # *all* of X_cal (none of them ever trained on it), so the
        # full-ensemble mean prediction on X_cal is already a valid,
        # genuinely out-of-sample residual -- no leave-one-out bookkeeping
        # needed here, unlike the training-set loop above.
        cal_preds = np.stack([m.predict(X_cal).reshape(-1) for m in self._models], axis=0).mean(axis=0)
        cal_residuals = np.abs(np.asarray(y_cal).reshape(-1) - cal_preds)

        self._residuals = list(loo_residuals) + [float(r) for r in cal_residuals]
        return self

    def predict_sequential(self, X_test: np.ndarray, y_test: np.ndarray) -> EnbPIPrediction:
        if self._residuals is None:
            raise RuntimeError("EnbPIForecaster.predict_sequential called before fit()")

        X_test = np.asarray(X_test)
        y_test_flat = np.asarray(y_test).reshape(-1)
        n = X_test.shape[0]
        if n != len(y_test_flat):
            raise ValueError("X_test and y_test must have the same length")
        if n == 0:
            raise ValueError("Need at least one test point to run EnbPI sequentially")

        # Full-ensemble mean prediction -- unlike the training-set LOO
        # residuals, every model contributes here (there's no "out-of-bag"
        # distinction left to make once bootstrap-fitting is done; the
        # out-of-bag structure is only ever used to build honest residuals,
        # never to decide which models get to vote on an actual forecast).
        test_preds = np.stack([m.predict(X_test).reshape(-1) for m in self._models], axis=0)  # (B, n)
        point = test_preds.mean(axis=0)

        # A mutable working copy -- fit()'s own stored pool is never
        # mutated, so re-running predict_sequential always starts fresh.
        #
        # If `window` is smaller than the seeded (LOO + calibration) pool,
        # truncate to the most recent `window` residuals *before* the
        # online loop starts. Without this, a bounded window would never
        # actually bind: each test step appends exactly one residual and
        # (once over capacity) evicts exactly one, a net change of zero, so
        # a pool that starts *above* `window` would stay at that starting
        # size forever instead of shrinking down to it. This is the exact
        # same bug `forecasting.adaptive.SlidingWindowAdaptiveConformalForecaster`
        # documents finding and fixing in its own first cut -- caught here
        # by `tests/test_enbpi.py::test_window_bounds_the_residual_pool_size`
        # before this module ever shipped with it.
        pool: list[float] = list(self._residuals)
        if self.window is not None and len(pool) > self.window:
            pool = pool[-self.window :]
        q_hats = np.empty(n)
        lower = np.empty(n)
        upper = np.empty(n)
        errs = np.empty(n)
        pool_sizes = np.empty(n, dtype=int)

        for t in range(n):
            pool_sizes[t] = len(pool)
            q_hat = SplitConformalForecaster._quantile_with_finite_sample_correction(
                np.asarray(pool), self.alpha
            )
            lo, hi = point[t] - q_hat, point[t] + q_hat
            err = 0.0 if (lo <= y_test_flat[t] <= hi) else 1.0

            q_hats[t] = q_hat
            lower[t] = lo
            upper[t] = hi
            errs[t] = err

            # Fold this step's own observed residual in *after* scoring it
            # (same ordering discipline as the sliding-window ACI variant),
            # so q_hat at step t never has access to y_test[t] itself.
            pool.append(float(abs(y_test_flat[t] - point[t])))
            if self.window is not None and len(pool) > self.window:
                pool.pop(0)

        return EnbPIPrediction(
            point=point.reshape(-1, 1),
            lower=lower.reshape(-1, 1),
            upper=upper.reshape(-1, 1),
            q_hat=q_hats,
            alpha=self.alpha,
            errs=errs,
            pool_size=pool_sizes,
            n_bootstrap=self.n_bootstrap,
            n_loo_skipped=self.n_loo_skipped_,
        )


def evaluate_enbpi_coverage(prediction: EnbPIPrediction, y_true: np.ndarray) -> dict:
    """Same "measure, don't assert" discipline as `evaluate_coverage` /
    `evaluate_adaptive_coverage`. There is no `alpha_t` to report (EnbPI
    never adapts the significance level), so this reports the fixed
    `alpha` plus the pool-size trajectory in its place.
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
        "final_pool_size": int(prediction.pool_size[-1]),
        "n_test": int(y_true_col.shape[0]),
    }
