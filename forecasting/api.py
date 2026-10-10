"""Minimal FastAPI serving layer around the exact pipeline the CLI already uses.

Deliberately thin: no new forecasting logic lives here. Every endpoint
calls straight into `experiment._prepare_split` + `conformal.SplitConformalForecaster`
-- the same code path `run_experiment` (and 125 existing tests) already
exercise -- so the calibrated forecasts this repo produces can be called by
another program instead of only read off a markdown table.

v0.12: fitted-forecaster caching. `/forecast`'s expensive step is fitting
the model (trivial for naive/seasonal_naive, real training for lstm) --
`alpha` never affects that fit, only the conformal quantile computed from
it, which is a cheap O(n_cal log n_cal) sort. So the cache key below is
`(dataset, model, seed, lookback, period)`, deliberately excluding `alpha`:
two requests that differ only in `alpha` reuse one fit and each get their
own correctly-recomputed interval. `_model_cache` is a small bounded LRU
(`_CACHE_MAX_SIZE` entries, oldest-used evicted first) guarded by a lock,
inspectable via `GET /cache/stats` and resettable via `DELETE /cache` --
both exist mainly so the test suite can verify cache behavior through the
same HTTP interface every other test uses, rather than by reaching into
module internals, but they're also genuinely useful operationally (e.g.
forcing a refit after noticing something odd).

Honest scope, stated here and in the README: this is still a demo-quality
serving layer. No auth, no rate limiting, no persistence across process
restarts (the cache is in-memory only) -- but the LSTM no longer retrains
on every request for a (dataset, model, seed, lookback, period) combo
that's already been seen, which was v0.11's own most obvious gap. See the
README's "Serving it as an API" section and "What's next" for what's still
not done (accepting a user-supplied series instead of only the 3 bundled
dataset names).

Endpoints
---------
GET    /health       -- liveness check.
GET    /datasets     -- static metadata for the 3 bundled datasets.
POST   /forecast     -- run one model on one dataset through the conformal
                         pipeline; returns every test point's actual value,
                         point forecast, calibrated interval, and whether
                         the interval covered the actual, plus summary
                         stats. Reuses a cached fit when one exists.
GET    /cache/stats  -- current cache size, max size, and cached keys.
DELETE /cache        -- clear the cache (forces the next matching request
                         to refit).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.experiment import _prepare_split
from forecasting.metrics import mae as _mae
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster, SeasonalNaiveForecaster

app = FastAPI(
    title="conformal-forecast API",
    description=(
        "Calibrated time-series forecasts (split conformal prediction) over "
        "a fixed set of bundled datasets. Demo-quality: see the repo README "
        "for honest scope notes."
    ),
    version="0.12",
)

# --- Fitted-forecaster cache (v0.12) ---------------------------------------
# Keyed on everything that affects model *fitting*; `alpha` is excluded on
# purpose (see module docstring). A plain OrderedDict + a lock is enough for
# a demo-quality server -- no eviction policy more elaborate than "least
# recently used" is needed at this scale.
_CACHE_MAX_SIZE = 16
_model_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(req: "ForecastRequest") -> tuple:
    return (req.dataset, req.model, req.seed, req.lookback, req.period)


def _get_or_fit(req: "ForecastRequest") -> dict:
    """Return the fitted-model bundle for `req`'s (dataset, model, seed,
    lookback, period), fitting and caching it on first use.

    Raises HTTPException(422) for a dataset/lookback/period combination
    `_prepare_split` rejects (too few windows) -- same behavior `forecast()`
    had before caching existed, just moved here since this is now the one
    place that calls `_prepare_split`.
    """
    key = _cache_key(req)
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
            return cached

    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(
            req.dataset, req.lookback, req.period
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if req.model == "naive":
        model = NaiveForecaster()
    elif req.model == "seasonal_naive":
        model = SeasonalNaiveForecaster(period=period)
    else:
        model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=req.seed))

    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "normalizer": normalizer,
        "period": period,
    }
    with _cache_lock:
        # Another request for the same key may have raced this one and won
        # -- keep whichever bundle is already there rather than clobbering
        # it with a second, redundant fit's result.
        bundle = _model_cache.setdefault(key, bundle)
        _model_cache.move_to_end(key)
        while len(_model_cache) > _CACHE_MAX_SIZE:
            _model_cache.popitem(last=False)
    return bundle


DATASET_DESCRIPTIONS = {
    "airline": (
        "Classic Box & Jenkins monthly airline passenger counts, 1949-1960 "
        "(144 points, strong trend + yearly seasonality, almost no noise)."
    ),
    "temperature": (
        "Daily minimum temperatures in Melbourne, Australia, 1981-1990 "
        "(3650 points, strong yearly seasonality, substantial day-to-day noise)."
    ),
    "synthetic": (
        "Trend + seasonality + i.i.d. Gaussian noise with a known std, used "
        "to validate the conformal-prediction machinery itself."
    ),
}

DatasetName = Literal["airline", "temperature", "synthetic"]
ModelName = Literal["naive", "seasonal_naive", "lstm"]


class ForecastRequest(BaseModel):
    dataset: DatasetName = Field(..., description="one of the 3 bundled datasets")
    model: ModelName = Field("naive", description="point-forecast model to calibrate")
    alpha: float = Field(0.1, gt=0.0, lt=1.0, description="miscoverage rate, e.g. 0.1 -> 90% intervals")
    seed: int = Field(0, description="LSTM init/training seed (ignored for naive/seasonal_naive)")
    lookback: int | None = Field(None, gt=0, description="override the dataset's default lookback window")
    period: int | None = Field(None, gt=0, description="override the dataset's default seasonal period")
    max_points: int | None = Field(
        None, gt=0, description="return only the most recent N test points in `points` (default: all)"
    )


class ForecastPoint(BaseModel):
    index: int
    actual: float
    point_forecast: float
    lower: float
    upper: float
    covered: bool


class ForecastSummary(BaseModel):
    mae: float
    nominal_coverage: float
    empirical_coverage: float
    coverage_gap: float
    mean_interval_width: float
    q_hat: float
    n_test: int


class ForecastResponse(BaseModel):
    dataset: DatasetName
    model: ModelName
    alpha: float
    seed: int
    points: list[ForecastPoint]
    summary: ForecastSummary


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/datasets")
def datasets() -> dict:
    return {
        "datasets": [
            {"name": name, "description": desc} for name, desc in DATASET_DESCRIPTIONS.items()
        ]
    }


@app.get("/cache/stats")
def cache_stats() -> dict:
    with _cache_lock:
        keys = list(_model_cache.keys())
    return {
        "size": len(keys),
        "max_size": _CACHE_MAX_SIZE,
        "keys": [
            {"dataset": k[0], "model": k[1], "seed": k[2], "lookback": k[3], "period": k[4]} for k in keys
        ],
    }


@app.delete("/cache")
def clear_cache() -> dict:
    with _cache_lock:
        n = len(_model_cache)
        _model_cache.clear()
    return {"cleared": n}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    bundle = _get_or_fit(req)
    model = bundle["model"]
    X_cal, y_cal = bundle["X_cal"], bundle["y_cal"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]
    normalizer = bundle["normalizer"]

    # The fit (possibly reused from cache) never depends on alpha -- only
    # this quantile does, so it's always recomputed fresh per request.
    conformal = SplitConformalForecaster(model, alpha=req.alpha)
    cal_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - cal_pred).reshape(-1)
    conformal.q_hat = SplitConformalForecaster._quantile_with_finite_sample_correction(residuals, req.alpha)
    pred = conformal.predict(X_test)

    point = normalizer.inverse_transform(pred.point).reshape(-1)
    lower = normalizer.inverse_transform(pred.lower).reshape(-1)
    upper = normalizer.inverse_transform(pred.upper).reshape(-1)
    actual = normalizer.inverse_transform(y_test).reshape(-1)

    # Summary stats always describe the *full* test window, even when
    # `max_points` truncates the per-point list below -- truncation is a
    # display convenience, not a different evaluation.
    cov = evaluate_coverage(pred, y_test)
    mae_value = _mae(actual, point)

    n = len(actual)
    start = 0 if req.max_points is None else max(0, n - req.max_points)
    points = [
        ForecastPoint(
            index=i,
            actual=float(actual[i]),
            point_forecast=float(point[i]),
            lower=float(lower[i]),
            upper=float(upper[i]),
            covered=bool(lower[i] <= actual[i] <= upper[i]),
        )
        for i in range(start, n)
    ]

    summary = ForecastSummary(
        mae=mae_value,
        nominal_coverage=cov["nominal_coverage"],
        empirical_coverage=cov["empirical_coverage"],
        coverage_gap=cov["coverage_gap"],
        mean_interval_width=cov["mean_interval_width"],
        q_hat=cov["q_hat"],
        n_test=cov["n_test"],
    )

    return ForecastResponse(
        dataset=req.dataset, model=req.model, alpha=req.alpha, seed=req.seed, points=points, summary=summary
    )
