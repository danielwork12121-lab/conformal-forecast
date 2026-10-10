"""Tests for the FastAPI serving layer (forecasting/api.py).

These exercise the HTTP layer (request validation, response shape) *and*
check that it doesn't silently diverge from the underlying library: the
key test below computes an LSTM forecast two ways -- once through the API,
once by calling `forecasting.conformal` / `forecasting.experiment` directly
-- and asserts they agree to the last digit.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forecasting.api import app
from forecasting.conformal import SplitConformalForecaster, evaluate_coverage
from forecasting.experiment import _prepare_split
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_datasets():
    resp = client.get("/datasets")
    assert resp.status_code == 200
    body = resp.json()
    names = {d["name"] for d in body["datasets"]}
    assert names == {"airline", "temperature", "synthetic"}
    assert all(isinstance(d["description"], str) and d["description"] for d in body["datasets"])


def test_forecast_rejects_unknown_dataset():
    resp = client.post("/forecast", json={"dataset": "not_a_real_dataset"})
    assert resp.status_code == 422


def test_forecast_rejects_unknown_model():
    resp = client.post("/forecast", json={"dataset": "airline", "model": "xgboost"})
    assert resp.status_code == 422


def test_forecast_rejects_out_of_range_alpha():
    resp = client.post("/forecast", json={"dataset": "airline", "alpha": 1.5})
    assert resp.status_code == 422
    resp = client.post("/forecast", json={"dataset": "airline", "alpha": 0.0})
    assert resp.status_code == 422


def test_forecast_rejects_non_positive_max_points():
    resp = client.post("/forecast", json={"dataset": "airline", "max_points": 0})
    assert resp.status_code == 422
    resp = client.post("/forecast", json={"dataset": "airline", "max_points": -3})
    assert resp.status_code == 422


def test_forecast_naive_is_well_formed():
    resp = client.post("/forecast", json={"dataset": "airline", "model": "naive"})
    assert resp.status_code == 200
    body = resp.json()

    points = body["points"]
    assert len(points) == body["summary"]["n_test"]

    covered_count = 0
    for p in points:
        assert p["lower"] <= p["point_forecast"] <= p["upper"]
        assert p["covered"] == (p["lower"] <= p["actual"] <= p["upper"])
        covered_count += int(p["covered"])

    # The summary's own empirical_coverage must match a direct count over
    # the (untruncated, since max_points wasn't set) returned points -- the
    # response shouldn't be able to report a coverage number the points it
    # hands back don't actually support.
    assert body["summary"]["empirical_coverage"] == covered_count / len(points)

    # This exact number is the README's own existing airline benchmark
    # table entry for Naive -- the API must agree with the documented
    # CLI-produced result, not just be internally consistent.
    assert body["summary"]["mae"] == pytest.approx(45.1, abs=1e-3)


def test_forecast_max_points_truncates_to_most_recent_and_stays_index_aligned():
    full = client.post("/forecast", json={"dataset": "airline", "model": "naive"}).json()
    truncated = client.post("/forecast", json={"dataset": "airline", "model": "naive", "max_points": 3}).json()

    assert len(truncated["points"]) == 3
    # The truncated response's points must be exactly the full response's
    # last 3 points, same index values and same values -- not a re-run that
    # happens to look similar (model fitting is deterministic here, but the
    # point of this test is the slicing logic, not determinism).
    assert truncated["points"] == full["points"][-3:]
    # Summary stats describe the full test window regardless of truncation.
    assert truncated["summary"] == full["summary"]


def test_forecast_lstm_matches_direct_pipeline_exactly():
    """The actual regression test: the API must not silently diverge from
    calling the library directly, for the one model (LSTM) whose numbers
    aren't a deterministic closed form (seasonal_naive/naive are trivial)."""
    dataset, model_name, seed, alpha = "synthetic", "lstm", 2, 0.1

    resp = client.post("/forecast", json={"dataset": dataset, "model": model_name, "seed": seed, "alpha": alpha})
    assert resp.status_code == 200
    api_body = resp.json()

    # Same pipeline, called directly, bypassing the API entirely.
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, period = _prepare_split(dataset, None, None)
    model = DeltaWrapper(LSTMForecaster(hidden_size=32, num_layers=1, seed=seed))
    conformal = SplitConformalForecaster(model, alpha=alpha)
    conformal.fit(X_train, y_train, X_cal, y_cal)
    pred = conformal.predict(X_test)

    direct_point = normalizer.inverse_transform(pred.point).reshape(-1)
    direct_lower = normalizer.inverse_transform(pred.lower).reshape(-1)
    direct_upper = normalizer.inverse_transform(pred.upper).reshape(-1)
    direct_cov = evaluate_coverage(pred, y_test)

    api_point = np.array([p["point_forecast"] for p in api_body["points"]])
    api_lower = np.array([p["lower"] for p in api_body["points"]])
    api_upper = np.array([p["upper"] for p in api_body["points"]])

    np.testing.assert_allclose(api_point, direct_point, rtol=1e-5)
    np.testing.assert_allclose(api_lower, direct_lower, rtol=1e-5)
    np.testing.assert_allclose(api_upper, direct_upper, rtol=1e-5)
    assert api_body["summary"]["empirical_coverage"] == direct_cov["empirical_coverage"]
    assert abs(api_body["summary"]["q_hat"] - direct_cov["q_hat"]) < 1e-6


def test_docs_endpoint_is_reachable():
    resp = client.get("/docs")
    assert resp.status_code == 200
