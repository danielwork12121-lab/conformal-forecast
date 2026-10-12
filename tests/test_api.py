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
from forecasting.experiment import _prepare_split, _prepare_split_from_series
from forecasting.models import DeltaWrapper, LSTMForecaster, NaiveForecaster

client = TestClient(app)


def _make_synthetic_series(n: int = 250, seed: int = 0) -> list[float]:
    """A deterministic synthetic series for the user-supplied-series tests
    below (v0.13) -- trend + seasonality + noise, same shape of signal as
    the repo's own bundled `synthetic` dataset generator, but built here so
    these tests don't depend on `forecasting.data`'s synthetic generator
    changing underneath them."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (50 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, n)).tolist()


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


# --- Fitted-forecaster cache (v0.12) ---------------------------------------


def test_cache_starts_empty_after_clear():
    resp = client.delete("/cache")
    assert resp.status_code == 200
    stats = client.get("/cache/stats").json()
    assert stats == {"size": 0, "max_size": 16, "keys": []}


def test_cache_reuses_fitted_model_across_requests_that_only_differ_by_alpha(monkeypatch):
    """The actual regression test for caching: two /forecast calls with the
    same (dataset, model, seed, lookback, period) but different `alpha`
    must fit the LSTM exactly once, not twice."""
    client.delete("/cache")
    from forecasting import models as models_module

    calls = {"n": 0}
    original_fit = models_module.LSTMForecaster.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(models_module.LSTMForecaster, "fit", counting_fit)

    r1 = client.post("/forecast", json={"dataset": "synthetic", "model": "lstm", "seed": 11, "alpha": 0.1})
    r2 = client.post("/forecast", json={"dataset": "synthetic", "model": "lstm", "seed": 11, "alpha": 0.2})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["n"] == 1, "second request should have reused the cached fit, not refit"

    stats = client.get("/cache/stats").json()
    assert stats["size"] == 1
    assert stats["keys"][0] == {"dataset": "synthetic", "model": "lstm", "seed": 11, "lookback": None, "period": None}

    client.delete("/cache")


def test_cache_key_distinguishes_different_seeds(monkeypatch):
    """A different `seed` must NOT reuse another seed's cached fit -- that
    would silently serve the wrong model's forecast."""
    client.delete("/cache")
    from forecasting import models as models_module

    calls = {"n": 0}
    original_fit = models_module.LSTMForecaster.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(models_module.LSTMForecaster, "fit", counting_fit)

    client.post("/forecast", json={"dataset": "synthetic", "model": "lstm", "seed": 1})
    client.post("/forecast", json={"dataset": "synthetic", "model": "lstm", "seed": 2})
    assert calls["n"] == 2

    stats = client.get("/cache/stats").json()
    assert stats["size"] == 2

    client.delete("/cache")


def test_cached_fit_gives_identical_point_forecast_across_different_alpha():
    """Correctness, not just call-counting: whether or not the fit is
    served from cache, the point forecast for one (dataset, model, seed)
    must not change with alpha -- only the interval should."""
    client.delete("/cache")
    r1 = client.post("/forecast", json={"dataset": "airline", "model": "naive", "alpha": 0.1}).json()
    r2 = client.post("/forecast", json={"dataset": "airline", "model": "naive", "alpha": 0.4}).json()

    assert [p["point_forecast"] for p in r1["points"]] == [p["point_forecast"] for p in r2["points"]]
    assert [p["actual"] for p in r1["points"]] == [p["actual"] for p in r2["points"]]
    # A smaller nominal coverage (alpha=0.4 -> 60%) must give a narrower (or
    # equal) interval than a larger one (alpha=0.1 -> 90%), never wider.
    assert r2["summary"]["q_hat"] <= r1["summary"]["q_hat"]
    assert r1["summary"]["mae"] == r2["summary"]["mae"] == pytest.approx(45.1, abs=1e-3)

    client.delete("/cache")


def test_cache_evicts_least_recently_used_beyond_max_size(monkeypatch):
    client.delete("/cache")
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_CACHE_MAX_SIZE", 2)

    # 3 distinct keys (naive ignores `seed` numerically, but it still keys
    # the cache separately -- that's fine, this test is about eviction
    # mechanics, not about naive actually using the seed).
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 1})
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 2})
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 3})

    stats = client.get("/cache/stats").json()
    assert stats["size"] == 2
    seeds_cached = {k["seed"] for k in stats["keys"]}
    assert seeds_cached == {2, 3}, "seed=1 was least-recently-used and should have been evicted first"

    client.delete("/cache")


def test_cache_access_moves_entry_to_most_recently_used(monkeypatch):
    """Re-requesting an existing key must count as a fresh 'use' for LRU
    purposes -- otherwise a popular key could still be evicted just because
    it happened to be cached first."""
    client.delete("/cache")
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_CACHE_MAX_SIZE", 2)

    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 1})
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 2})
    # Touch seed=1 again so it becomes the most-recently-used of the two.
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 1})
    # A third, new key should now evict seed=2 (least-recently-used), not seed=1.
    client.post("/forecast", json={"dataset": "synthetic", "model": "naive", "seed": 3})

    stats = client.get("/cache/stats").json()
    seeds_cached = {k["seed"] for k in stats["keys"]}
    assert seeds_cached == {1, 3}

    client.delete("/cache")


# --- User-supplied series (v0.13) ------------------------------------------


def test_forecast_rejects_both_dataset_and_series():
    series = _make_synthetic_series()
    resp = client.post("/forecast", json={"dataset": "airline", "series": series})
    assert resp.status_code == 422


def test_forecast_rejects_neither_dataset_nor_series():
    resp = client.post("/forecast", json={"model": "naive"})
    assert resp.status_code == 422


def test_forecast_rejects_series_shorter_than_field_minimum():
    """Pydantic's own `min_length=30` on the field -- a series shorter than
    that is rejected before `_prepare_split_from_series` ever runs."""
    resp = client.post("/forecast", json={"series": list(range(10))})
    assert resp.status_code == 422


def test_forecast_rejects_series_too_short_for_requested_lookback():
    """Distinct from the field-level check above: a series that clears
    `min_length=30` can still be too short once a (user-requested) lookback
    is applied -- this actually exercises
    `_prepare_split_from_series`'s own 'not enough windows' ValueError, not
    just the pydantic field check."""
    series = _make_synthetic_series(n=35)
    resp = client.post("/forecast", json={"series": series, "lookback": 30})
    assert resp.status_code == 422


def test_forecast_series_naive_is_well_formed():
    series = _make_synthetic_series(n=250)
    resp = client.post("/forecast", json={"series": series, "model": "naive"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["dataset"] == "custom"
    assert body["n_input_points"] == len(series)

    points = body["points"]
    assert len(points) == body["summary"]["n_test"]
    for p in points:
        assert p["lower"] <= p["point_forecast"] <= p["upper"]
        assert p["covered"] == (p["lower"] <= p["actual"] <= p["upper"])


def test_forecast_series_lstm_matches_direct_pipeline_exactly():
    """Same discipline as `test_forecast_lstm_matches_direct_pipeline_exactly`,
    but for the series path: the API must not silently diverge from calling
    `_prepare_split_from_series` directly."""
    series = _make_synthetic_series(n=250, seed=3)
    model_name, seed, alpha, lookback, period = "lstm", 4, 0.1, 12, 12

    resp = client.post(
        "/forecast", json={"series": series, "model": model_name, "seed": seed, "alpha": alpha}
    )
    assert resp.status_code == 200
    api_body = resp.json()

    series_arr = np.asarray(series, dtype=np.float64)
    X_train, y_train, X_cal, y_cal, X_test, y_test, normalizer, _ = _prepare_split_from_series(
        series_arr, lookback, period, label="series"
    )
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


def test_cache_reuses_fitted_model_for_series_across_alpha(monkeypatch):
    """The series-path equivalent of
    `test_cache_reuses_fitted_model_across_requests_that_only_differ_by_alpha`:
    two /forecast calls for the byte-identical series, differing only in
    `alpha`, must fit the LSTM exactly once."""
    client.delete("/cache")
    from forecasting import models as models_module

    calls = {"n": 0}
    original_fit = models_module.LSTMForecaster.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(models_module.LSTMForecaster, "fit", counting_fit)

    series = _make_synthetic_series(n=250, seed=7)
    r1 = client.post("/forecast", json={"series": series, "model": "lstm", "seed": 5, "alpha": 0.1})
    r2 = client.post("/forecast", json={"series": series, "model": "lstm", "seed": 5, "alpha": 0.3})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["n"] == 1, "second request should have reused the cached fit, not refit"

    stats = client.get("/cache/stats").json()
    assert stats["size"] == 1
    cached_key = stats["keys"][0]
    assert cached_key["dataset"] == "custom"
    assert cached_key["model"] == "lstm"
    assert cached_key["seed"] == 5
    assert "series_hash" in cached_key

    client.delete("/cache")


def test_cache_distinguishes_different_series(monkeypatch):
    """Two different series, otherwise identical request fields, must never
    share a cache entry -- that would silently serve one series' fit for
    another's forecast."""
    client.delete("/cache")
    from forecasting import models as models_module

    calls = {"n": 0}
    original_fit = models_module.LSTMForecaster.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(models_module.LSTMForecaster, "fit", counting_fit)

    series_a = _make_synthetic_series(n=250, seed=1)
    series_b = _make_synthetic_series(n=250, seed=2)
    client.post("/forecast", json={"series": series_a, "model": "lstm", "seed": 9})
    client.post("/forecast", json={"series": series_b, "model": "lstm", "seed": 9})
    assert calls["n"] == 2

    stats = client.get("/cache/stats").json()
    assert stats["size"] == 2
    hashes = {k["series_hash"] for k in stats["keys"]}
    assert len(hashes) == 2, "two different series must not collide on the same cache key"

    client.delete("/cache")
