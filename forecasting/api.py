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

v0.13: accepting a user-supplied `series` instead of only the 3 bundled
dataset names. `ForecastRequest` now accepts exactly one of `dataset` or
`series` (enforced by a model-level validator); a custom series shares the
*exact* same split/window/normalize/validate core as the bundled-dataset
path via `experiment._prepare_split_from_series` (extracted from
`_prepare_split` this same version -- see that function's docstring). A
custom series has no name to key the fit cache on, so `_cache_key()` now
returns a 6-tuple tagged `"dataset"` or `"custom"`, with the custom tag's
discriminator being a truncated SHA-256 of the series' float64 bytes
(`_series_cache_id()`) -- two requests for the identical series (even with
different `alpha`) still share one fit, and two different series never
collide. `GET /cache/stats` emits the *same* 5-field dict as before for
dataset-path entries (so the existing exact-dict-equality test keeps
passing byte-for-byte) and a 6-field dict (adds `series_hash`) for
custom-series entries.

v0.14: a "dataset handle" for a user-supplied series, so a repeat caller
doesn't have to resend the whole series on every request (v0.13's own
honestly-flagged gap). `POST /series` registers a series once and returns
its `series_id` -- which is *exactly* `_series_cache_id(series)`, the same
hash v0.13 already computed internally, so a request using `series_id`
collides on the fit cache with an equivalent inline-`series` request for
the byte-identical series, by construction, not by coincidence. The store
is a second small bounded LRU (`_SERIES_STORE_MAX_SIZE`, independently
sized from the fit cache), inspectable via `GET /series/{series_id}` and
removable via `DELETE /series/{series_id}`. `ForecastRequest` now accepts
exactly one of `dataset`/`series`/`series_id` (the validator below grew a
third branch); and -- opportunistically, so a client doesn't have to
separately call `POST /series` first just to get reuse -- every inline-
`series` `/forecast` call also registers that series, so its `series_id`
comes back in the response (`ForecastResponse.series_id`) ready to reuse
on the next call without resending the series at all. A `series_id` whose
fit is still warm in the model cache works even if it's since been evicted
from the series store -- the cache key is the id string itself, not a
lookup, so the raw series is only ever fetched from the store on an actual
cache miss.

Honest scope, stated here and in the README: this is still a demo-quality
serving layer. No auth, no rate limiting, no persistence across process
restarts -- both the fit cache and the series store are in-memory only and
vanish on restart or eviction, so `series_id` is a same-process convenience,
not a durable handle a client can rely on indefinitely. But the LSTM no
longer retrains on every request for a combo that's already been seen
(v0.12), the API isn't limited to 3 bundled dataset names (v0.13), and a
repeat caller no longer has to resend a whole series it already sent once
(v0.14, this version's fix).

Endpoints
---------
GET    /health              -- liveness check.
GET    /datasets            -- static metadata for the 3 bundled datasets.
POST   /series               -- register a series, get back its `series_id`
                                 (idempotent: registering the same series
                                 twice returns the same id).
GET    /series/{series_id}   -- check whether a `series_id` is still in the
                                 store, and its length.
DELETE /series/{series_id}   -- remove a series from the store.
POST   /forecast             -- run one model on one dataset OR one
                                 user-supplied series OR a previously
                                 registered `series_id` through the
                                 conformal pipeline; returns every test
                                 point's actual value, point forecast,
                                 calibrated interval, and whether the
                                 interval covered the actual, plus summary
                                 stats. Reuses a cached fit when one exists.
GET    /cache/stats         -- current fit-cache size, max size, and keys.
DELETE /cache                -- clear the fit cache (forces the next
                                 matching request to refit).
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.experiment import DEFAULT_LOOKBACKS, _prepare_split, _prepare_split_from_series
from forecasting.metrics import mae as _mae
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster, SeasonalNaiveForecaster

app = FastAPI(
    title="conformal-forecast API",
    description=(
        "Calibrated time-series forecasts (split conformal prediction) over "
        "a fixed set of bundled datasets, your own series, or a series "
        "you've registered once and referenced by id. "
        "Demo-quality: see the repo README for honest scope notes."
    ),
    version="0.14",
)

# Generic defaults used for a user-supplied series, which has no dataset
# name to look up DEFAULT_LOOKBACKS/DEFAULT_PERIODS with -- the same
# fallback value `_prepare_split` already uses for an unlisted dataset
# name (see `forecasting.experiment.DEFAULT_LOOKBACKS/DEFAULT_PERIODS`).
_CUSTOM_SERIES_DEFAULT_LOOKBACK = 12
_CUSTOM_SERIES_DEFAULT_PERIOD = 12

# --- Fitted-forecaster cache (v0.12) ---------------------------------------
# Keyed on everything that affects model *fitting*; `alpha` is excluded on
# purpose (see module docstring). A plain OrderedDict + a lock is enough for
# a demo-quality server -- no eviction policy more elaborate than "least
# recently used" is needed at this scale.
_CACHE_MAX_SIZE = 16
_model_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _series_cache_id(series: list[float]) -> str:
    """A stable cache-key discriminator for a user-supplied series, which
    has no dataset name to key on. Truncated (16 hex chars = 64 bits --
    ample to avoid an accidental collision at this cache's scale of at
    most `_CACHE_MAX_SIZE` entries) SHA-256 of the series' float64 bytes,
    so two requests for the byte-identical series always hash identically
    and two different series practically never collide.
    """
    arr = np.asarray(series, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


# --- Dataset-handle series store (v0.14) -----------------------------------
# A second, independently-sized bounded LRU, storing raw series by the exact
# same id `_series_cache_id()` produces -- so a series registered via
# `POST /series` and the same series sent inline to `/forecast` always agree
# on their id, and the two paths' fit-cache entries collide by construction.
_SERIES_STORE_MAX_SIZE = 32
_series_store: "OrderedDict[str, list[float]]" = OrderedDict()
_series_lock = threading.Lock()


def _register_series(series: list[float]) -> str:
    """Store `series` under its content hash (idempotent -- registering the
    same series twice just refreshes its LRU position) and return the id.
    """
    series_id = _series_cache_id(series)
    with _series_lock:
        _series_store[series_id] = series
        _series_store.move_to_end(series_id)
        while len(_series_store) > _SERIES_STORE_MAX_SIZE:
            _series_store.popitem(last=False)
    return series_id


def _resolve_series_id(series_id: str) -> list[float]:
    """Look up a previously registered series by id, bumping its LRU
    position. Raises HTTPException(404) if it's not (or no longer) in the
    store -- the store is a bounded LRU, so an id can legitimately age out.
    """
    with _series_lock:
        series = _series_store.get(series_id)
        if series is not None:
            _series_store.move_to_end(series_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"no series registered with id {series_id!r}")
    return series


def _cache_key(req: "ForecastRequest", series_id: str | None) -> tuple:
    """Everything that affects model *fitting*, tagged by source so a
    dataset-path key and a custom-series-path key can never collide even
    if a coincidence made their other fields match. `alpha` is excluded on
    purpose (see module docstring) in both cases. `series_id` is resolved by
    the caller (`_get_or_fit`) from either `req.series` (hashed) or
    `req.series_id` (used as-is) -- this function never touches `_series_store`,
    so a cache hit never needs the raw series at all, only its id string.
    """
    if series_id is not None:
        return ("custom", series_id, req.model, req.seed, req.lookback, req.period)
    return ("dataset", req.dataset, req.model, req.seed, req.lookback, req.period)


def _get_or_fit(req: "ForecastRequest") -> dict:
    """Return the fitted-model bundle for `req`'s cache key (see
    `_cache_key`), fitting and caching it on first use.

    Raises HTTPException(404) if `req.series_id` doesn't name a series
    still in `_series_store` *and* the fit isn't already cached (a warm
    cache entry never needs the store at all -- see `_cache_key`).
    Raises HTTPException(422) for a dataset/series/lookback/period
    combination `_prepare_split`/`_prepare_split_from_series` rejects (too
    few windows) -- same behavior `forecast()` had before caching existed,
    just moved here since this is now the one place that calls either.
    """
    # An inline `series` is opportunistically registered *before* the cache
    # check, every time -- cheap (just the hash it would need anyway for the
    # cache key) and it keeps the series store's LRU position fresh so a
    # repeatedly-used inline series doesn't fall out of the store between
    # calls. A bare `series_id` is used as-is here; it's only actually
    # resolved against the store below, and only on a cache miss.
    if req.series is not None:
        series_id = _register_series(req.series)
    else:
        series_id = req.series_id

    key = _cache_key(req, series_id)
    with _cache_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            _model_cache.move_to_end(key)
            return cached

    try:
        if series_id is not None:
            series_list = req.series if req.series is not None else _resolve_series_id(req.series_id)
            lookback = req.lookback or _CUSTOM_SERIES_DEFAULT_LOOKBACK
            period = req.period or _CUSTOM_SERIES_DEFAULT_PERIOD
            series_arr = np.asarray(series_list, dtype=np.float64)
            X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split_from_series(
                series_arr, lookback, period, label="series"
            )
        else:
            # Mirrors the lookback resolution `_prepare_split` does internally
            # (idempotent, no side effects) so `n_input_points` below can be
            # computed the same way for both paths without changing
            # `_prepare_split`'s own return signature.
            lookback = req.lookback or DEFAULT_LOOKBACKS.get(req.dataset, 12)
            X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(
                req.dataset, req.lookback, req.period
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Raw input-series length, derived rather than re-fetched: make_windows
    # produces exactly len(series) - lookback windows, and X_train/X_cal/
    # X_test exactly partition those (X_test absorbs the remainder) -- so
    # this holds for both the bundled-dataset path and the custom-series
    # path without needing either prepare-split function to return it.
    n_input_points = lookback + len(X_train) + len(X_cal) + len(X_test)

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
        "n_input_points": n_input_points,
        "series_id": series_id,
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
    dataset: DatasetName | None = Field(
        None, description="one of the 3 bundled datasets -- exactly one of `dataset`/`series`/`series_id` is required"
    )
    series: list[float] | None = Field(
        None,
        min_length=30,
        description=(
            "a user-supplied series instead of a bundled dataset (v0.13) -- exactly one of "
            "`dataset`/`series`/`series_id` is required. 30 points is a practical floor: "
            "`_prepare_split_from_series` needs at least 5 train + 5 calibration + 5 test windows after "
            "the default lookback=12, and a shorter series will likely still fail that check with a 422, "
            "just later and less clearly. Registered opportunistically under its own `series_id` (see "
            "`ForecastResponse.series_id`) so a later request can reuse it without resending it."
        ),
    )
    series_id: str | None = Field(
        None,
        description=(
            "the id of a series registered earlier, via `POST /series` or returned from a previous "
            "`/forecast` response (v0.14) -- exactly one of `dataset`/`series`/`series_id` is required. "
            "A 404 means the id isn't (or is no longer; the store is a bounded LRU) known."
        ),
    )
    model: ModelName = Field("naive", description="point-forecast model to calibrate")
    alpha: float = Field(0.1, gt=0.0, lt=1.0, description="miscoverage rate, e.g. 0.1 -> 90% intervals")
    seed: int = Field(0, description="LSTM init/training seed (ignored for naive/seasonal_naive)")
    lookback: int | None = Field(
        None, gt=0, description="override the default lookback window (the dataset's default, or 12 for a series)"
    )
    period: int | None = Field(
        None, gt=0, description="override the default seasonal period (the dataset's default, or 12 for a series)"
    )
    max_points: int | None = Field(
        None, gt=0, description="return only the most recent N test points in `points` (default: all)"
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ForecastRequest":
        provided = sum(x is not None for x in (self.dataset, self.series, self.series_id))
        if provided != 1:
            raise ValueError("exactly one of `dataset`, `series`, or `series_id` is required")
        return self


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
    dataset: DatasetName | Literal["custom"]
    model: ModelName
    alpha: float
    seed: int
    n_input_points: int
    series_id: str | None = Field(
        None,
        description=(
            "set whenever `dataset` is \"custom\" (i.e. the request used `series` or `series_id`), to "
            "this series' registered id -- reuse it as `series_id` on a later request instead of "
            "resending the series. None for a bundled-dataset request."
        ),
    )
    points: list[ForecastPoint]
    summary: ForecastSummary


class SeriesRegisterRequest(BaseModel):
    series: list[float] = Field(..., min_length=30, description="the series to register; see `ForecastRequest.series`")


class SeriesRegisterResponse(BaseModel):
    series_id: str
    length: int


class SeriesInfoResponse(BaseModel):
    series_id: str
    length: int


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


@app.post("/series", response_model=SeriesRegisterResponse)
def register_series(req: SeriesRegisterRequest) -> SeriesRegisterResponse:
    series_id = _register_series(req.series)
    return SeriesRegisterResponse(series_id=series_id, length=len(req.series))


@app.get("/series/{series_id}", response_model=SeriesInfoResponse)
def series_info(series_id: str) -> SeriesInfoResponse:
    series = _resolve_series_id(series_id)
    return SeriesInfoResponse(series_id=series_id, length=len(series))


@app.delete("/series/{series_id}")
def delete_series(series_id: str) -> dict:
    with _series_lock:
        existed = _series_store.pop(series_id, None) is not None
    return {"deleted": existed}


def _cache_key_to_dict(key: tuple) -> dict:
    """Render one `_cache_key()` tuple for `GET /cache/stats`.

    Dataset-path keys (`("dataset", name, model, seed, lookback, period)`)
    render as the exact same 5-field dict shape v0.12 always produced
    (preserving the existing exact-dict-equality test byte-for-byte);
    custom-series keys (`("custom", series_id, model, seed, lookback,
    period)`) render as a 6-field dict that additionally identifies the
    series by its cache-key hash rather than a name it doesn't have.
    """
    source, discriminator, model, seed, lookback, period = key
    if source == "dataset":
        return {"dataset": discriminator, "model": model, "seed": seed, "lookback": lookback, "period": period}
    return {
        "dataset": "custom",
        "series_hash": discriminator,
        "model": model,
        "seed": seed,
        "lookback": lookback,
        "period": period,
    }


@app.get("/cache/stats")
def cache_stats() -> dict:
    with _cache_lock:
        keys = list(_model_cache.keys())
    return {
        "size": len(keys),
        "max_size": _CACHE_MAX_SIZE,
        "keys": [_cache_key_to_dict(k) for k in keys],
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

    response_dataset = "custom" if bundle["series_id"] is not None else req.dataset

    return ForecastResponse(
        dataset=response_dataset,
        model=req.model,
        alpha=req.alpha,
        seed=req.seed,
        n_input_points=bundle["n_input_points"],
        series_id=bundle["series_id"],
        points=points,
        summary=summary,
    )
