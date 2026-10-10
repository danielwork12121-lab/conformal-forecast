"""Minimal FastAPI serving layer around the exact pipeline the CLI already uses.

Deliberately thin: no new forecasting logic lives here. Every endpoint
calls straight into `experiment._prepare_split` + `conformal.SplitConformalForecaster`
-- the same code path `run_experiment` (and 115 existing tests) already
exercise -- so the calibrated forecasts this repo produces can be called by
another program instead of only read off a markdown table.

Honest scope, stated here and in the README: this is a demo-quality
serving layer. No auth, no rate limiting, no response caching -- the LSTM
is retrained from scratch on every `/forecast` request that asks for it,
which is fine for a request every few seconds on these small bundled
datasets and would not be fine in production. See the README's "Serving
it as an API" section and "What's next" for the honest follow-ups
(caching a fitted forecaster; accepting a user-supplied series).

Endpoints
---------
GET  /health     -- liveness check.
GET  /datasets   -- static metadata for the 3 bundled datasets.
POST /forecast   -- run one model on one dataset through the conformal
                    pipeline; returns every test point's actual value,
                    point forecast, calibrated interval, and whether the
                    interval covered the actual, plus summary stats.
"""
from __future__ import annotations

from typing import Literal

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
    version="0.10",
)

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


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
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

    conformal = SplitConformalForecaster(model, alpha=req.alpha)
    conformal.fit(X_train, y_train, X_cal, y_cal)
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
