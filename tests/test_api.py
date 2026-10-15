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


def test_cache_series_id_response_matches_registration_id():
    """The `series_id` an inline-`series` `/forecast` call hands back must be
    the exact id a direct `POST /series` registration of the same series
    would produce -- the two paths are supposed to agree by construction,
    not by coincidence."""
    client.delete("/cache")
    series = _make_synthetic_series(n=250, seed=42)

    forecast_resp = client.post("/forecast", json={"series": series, "model": "naive"})
    assert forecast_resp.status_code == 200
    forecast_series_id = forecast_resp.json()["series_id"]
    assert forecast_series_id is not None

    register_resp = client.post("/series", json={"series": series})
    assert register_resp.status_code == 200
    assert register_resp.json()["series_id"] == forecast_series_id

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


# --- Dataset handle: POST/GET/DELETE /series (v0.14) -----------------------


def test_register_series_returns_id_and_length():
    series = _make_synthetic_series(n=60, seed=20)
    resp = client.post("/series", json={"series": series})
    assert resp.status_code == 200
    body = resp.json()
    assert body["length"] == 60
    assert isinstance(body["series_id"], str) and len(body["series_id"]) == 16


def test_register_series_rejects_shorter_than_minimum():
    resp = client.post("/series", json={"series": list(range(10))})
    assert resp.status_code == 422


def test_register_series_is_idempotent():
    """Registering the byte-identical series twice must return the same id
    -- the id is a content hash, not a sequence number."""
    series = _make_synthetic_series(n=60, seed=21)
    id_1 = client.post("/series", json={"series": series}).json()["series_id"]
    id_2 = client.post("/series", json={"series": series}).json()["series_id"]
    assert id_1 == id_2


def test_series_info_reports_registered_series():
    series = _make_synthetic_series(n=70, seed=22)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]
    resp = client.get(f"/series/{series_id}")
    assert resp.status_code == 200
    assert resp.json() == {"series_id": series_id, "length": 70}


def test_series_info_404s_for_unknown_id():
    resp = client.get("/series/not-a-real-id")
    assert resp.status_code == 404


def test_delete_series_removes_it_and_is_idempotent():
    series = _make_synthetic_series(n=60, seed=23)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]

    resp = client.delete(f"/series/{series_id}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}

    # Now gone.
    assert client.get(f"/series/{series_id}").status_code == 404
    # Deleting again is a no-op, not an error.
    resp = client.delete(f"/series/{series_id}")
    assert resp.json() == {"deleted": False}


def test_forecast_rejects_dataset_and_series_id_together():
    series = _make_synthetic_series(n=60, seed=24)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]
    resp = client.post("/forecast", json={"dataset": "airline", "series_id": series_id})
    assert resp.status_code == 422


def test_forecast_rejects_series_and_series_id_together():
    series = _make_synthetic_series(n=60, seed=25)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]
    resp = client.post("/forecast", json={"series": series, "series_id": series_id})
    assert resp.status_code == 422


def test_forecast_rejects_unknown_series_id():
    resp = client.post("/forecast", json={"series_id": "not-a-real-id", "model": "naive"})
    assert resp.status_code == 404


def test_forecast_by_series_id_matches_forecast_by_inline_series():
    """The whole point of the feature: a `series_id` request must give the
    identical forecast a byte-identical inline `series` request would, not
    merely a well-formed one."""
    client.delete("/cache")
    series = _make_synthetic_series(n=250, seed=26)

    inline_resp = client.post("/forecast", json={"series": series, "model": "lstm", "seed": 3, "alpha": 0.1})
    assert inline_resp.status_code == 200
    inline_body = inline_resp.json()

    series_id = client.post("/series", json={"series": series}).json()["series_id"]
    handle_resp = client.post("/forecast", json={"series_id": series_id, "model": "lstm", "seed": 3, "alpha": 0.1})
    assert handle_resp.status_code == 200
    handle_body = handle_resp.json()

    assert handle_body["points"] == inline_body["points"]
    assert handle_body["summary"] == inline_body["summary"]
    assert handle_body["series_id"] == inline_body["series_id"] == series_id
    assert handle_body["n_input_points"] == inline_body["n_input_points"] == 250

    client.delete("/cache")


def test_forecast_by_series_id_reuses_cached_fit_without_touching_series_store(monkeypatch):
    """A `series_id` whose fit is already warm in the model cache must not
    need the series store at all -- so it keeps working even after the raw
    series has been deleted from the store (or aged out of it)."""
    client.delete("/cache")
    from forecasting import models as models_module

    calls = {"n": 0}
    original_fit = models_module.LSTMForecaster.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(models_module.LSTMForecaster, "fit", counting_fit)

    series = _make_synthetic_series(n=250, seed=27)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]

    r1 = client.post("/forecast", json={"series_id": series_id, "model": "lstm", "seed": 8, "alpha": 0.1})
    assert r1.status_code == 200
    assert calls["n"] == 1

    # Delete the series from the store -- the fit is still cached.
    client.delete(f"/series/{series_id}")
    assert client.get(f"/series/{series_id}").status_code == 404

    r2 = client.post("/forecast", json={"series_id": series_id, "model": "lstm", "seed": 8, "alpha": 0.1})
    assert r2.status_code == 200
    assert calls["n"] == 1, "should have reused the cached fit, not refit, and never needed the deleted series"
    assert r2.json()["points"] == r1.json()["points"]

    # A different `alpha` is still a cache hit on the fit (still no second
    # fit, still no need for the deleted series) -- just a different
    # interval, same point forecasts, exactly like the inline-series cache
    # tests above.
    r3 = client.post("/forecast", json={"series_id": series_id, "model": "lstm", "seed": 8, "alpha": 0.3})
    assert r3.status_code == 200
    assert calls["n"] == 1
    assert [p["point_forecast"] for p in r3.json()["points"]] == [p["point_forecast"] for p in r1.json()["points"]]

    client.delete("/cache")


def test_forecast_by_series_id_404s_on_cache_miss_after_eviction(monkeypatch):
    """Unlike the warm-cache case above, a cache *miss* for a `series_id`
    that's no longer in the store has no raw series to fit from -- it must
    404, not silently fall back to anything else."""
    client.delete("/cache")
    series = _make_synthetic_series(n=60, seed=28)
    series_id = client.post("/series", json={"series": series}).json()["series_id"]
    client.delete(f"/series/{series_id}")

    resp = client.post("/forecast", json={"series_id": series_id, "model": "naive"})
    assert resp.status_code == 404

    client.delete("/cache")


def test_forecast_inline_series_is_reusable_by_id_without_resending_it():
    """An inline `series` call opportunistically registers the series, so a
    later request can reference it by `series_id` alone -- this is the
    feature's main promise: resend the series once, reuse it by id after."""
    client.delete("/cache")
    series = _make_synthetic_series(n=250, seed=29)

    first = client.post("/forecast", json={"series": series, "model": "naive"})
    assert first.status_code == 200
    series_id = first.json()["series_id"]

    second = client.post("/forecast", json={"series_id": series_id, "model": "seasonal_naive"})
    assert second.status_code == 200
    assert second.json()["dataset"] == "custom"
    assert second.json()["n_input_points"] == 250

    client.delete("/cache")


# --- Listing registered series (v0.15) --------------------------------------
#
# The whole store is module-level shared state, re-registered-into by every
# test above (and never fully cleared, since there was no bulk-clear
# endpoint before this feature). Rather than asserting exact contents against
# that shared history, every test below first swaps in a fresh, empty
# `_series_store` via `monkeypatch.setattr` (auto-reverted after the test,
# same technique the cache-eviction tests above already use for
# `_CACHE_MAX_SIZE`) so each test's listing assertions are exact and order
# does not depend on what ran before it.


def _fresh_series_store(monkeypatch) -> None:
    import forecasting.api as api_module
    from collections import OrderedDict

    monkeypatch.setattr(api_module, "_series_store", OrderedDict())
    # Also give the idle-TTL bookkeeping (v0.17) a clean slate, for the same
    # reason: it's module-level shared state that would otherwise carry
    # leftover entries (and thus leftover last-access timestamps) across
    # tests, even though those entries belong to different series ids than
    # whatever a given test registers.
    monkeypatch.setattr(api_module, "_series_last_access", {})


def test_list_series_reflects_empty_store(monkeypatch):
    _fresh_series_store(monkeypatch)
    resp = client.get("/series")
    assert resp.status_code == 200
    body = resp.json()
    assert body["series"] == []
    assert body["count"] == 0
    import forecasting.api as api_module

    assert body["max_size"] == api_module._SERIES_STORE_MAX_SIZE


def test_list_series_includes_registered_entries_with_correct_lengths(monkeypatch):
    _fresh_series_store(monkeypatch)
    id_a = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=30)}).json()["series_id"]
    id_b = client.post("/series", json={"series": _make_synthetic_series(n=55, seed=31)}).json()["series_id"]

    body = client.get("/series").json()
    assert body["count"] == 2
    lengths_by_id = {s["series_id"]: s["length"] for s in body["series"]}
    assert lengths_by_id == {id_a: 40, id_b: 55}


def test_list_series_orders_most_recently_used_first(monkeypatch):
    _fresh_series_store(monkeypatch)
    id_a = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=32)}).json()["series_id"]
    id_b = client.post("/series", json={"series": _make_synthetic_series(n=41, seed=33)}).json()["series_id"]
    id_c = client.post("/series", json={"series": _make_synthetic_series(n=42, seed=34)}).json()["series_id"]
    # Registration order so far (oldest -> newest use): a, b, c. Touching a
    # via GET bumps it to most-recently-used, without changing b/c's relative
    # order -- the same recency semantics /forecast's cache miss path and a
    # repeat GET /series/{id} already rely on.
    client.get(f"/series/{id_a}")

    body = client.get("/series").json()
    assert [s["series_id"] for s in body["series"]] == [id_a, id_c, id_b]


def test_list_series_respects_store_capacity_after_eviction(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_MAX_SIZE", 2)

    client.post("/series", json={"series": _make_synthetic_series(n=40, seed=35)})
    id_b = client.post("/series", json={"series": _make_synthetic_series(n=41, seed=36)}).json()["series_id"]
    id_c = client.post("/series", json={"series": _make_synthetic_series(n=42, seed=37)}).json()["series_id"]

    body = client.get("/series").json()
    assert body["max_size"] == 2
    assert body["count"] == 2
    assert {s["series_id"] for s in body["series"]} == {id_b, id_c}, (
        "the first-registered series was least-recently-used and should have been evicted"
    )


def test_list_series_reflects_deletion(monkeypatch):
    _fresh_series_store(monkeypatch)
    id_a = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=38)}).json()["series_id"]
    id_b = client.post("/series", json={"series": _make_synthetic_series(n=41, seed=39)}).json()["series_id"]
    client.delete(f"/series/{id_a}")

    body = client.get("/series").json()
    assert body["count"] == 1
    assert [s["series_id"] for s in body["series"]] == [id_b]


def test_list_series_includes_opportunistically_registered_inline_series(monkeypatch):
    """The feature's actual motivation: an inline `series` /forecast call
    registers a series "silently" -- this is the one place that id becomes
    discoverable again without already having kept the forecast response."""
    _fresh_series_store(monkeypatch)
    client.delete("/cache")
    series = _make_synthetic_series(n=250, seed=40)

    resp = client.post("/forecast", json={"series": series, "model": "naive"})
    series_id = resp.json()["series_id"]

    body = client.get("/series").json()
    assert body["count"] == 1
    import forecasting.api as api_module

    assert body["series"] == [
        {
            "series_id": series_id,
            "length": 250,
            "ttl_remaining_seconds": pytest.approx(
                api_module._SERIES_STORE_TTL_SECONDS, abs=30.0
            ),
        }
    ]

    client.delete("/cache")


# --- Idle-TTL eviction for the series store (v0.17) -------------------------
#
# A series can now age out two independent ways: capacity (tested above,
# unchanged) or idle time. `_clock` is monkeypatched to a fake, test-
# controlled clock rather than sleeping for real, the same reason
# `_SERIES_STORE_MAX_SIZE` is monkeypatched above instead of registering 32
# real series to force a real eviction.


def _fake_clock(monkeypatch, start: float = 1_000.0):
    """Install a fake, test-advanceable clock in place of `_clock` and
    return a function that advances it by `seconds`.
    """
    import forecasting.api as api_module

    state = {"now": start}
    monkeypatch.setattr(api_module, "_clock", lambda: state["now"])

    def advance(seconds: float) -> None:
        state["now"] += seconds

    return advance


def test_series_survives_within_ttl_window(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=41)}).json()["series_id"]
    advance(50.0)  # well under the 100s TTL

    assert client.get(f"/series/{series_id}").status_code == 200
    assert client.get("/series").json()["count"] == 1


def test_series_expires_after_ttl_and_disappears_from_listing(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=42)}).json()["series_id"]
    advance(150.0)  # past the 100s TTL, nothing else touched it in between

    resp = client.get(f"/series/{series_id}")
    assert resp.status_code == 404

    body = client.get("/series").json()
    assert body["count"] == 0
    assert body["series"] == []


def test_resolving_a_series_resets_its_ttl(monkeypatch):
    """Touching a series (via GET /series/{id}, or the fit-cache path that
    calls the same resolver) refreshes its idle clock -- the same recency
    semantics the LRU-capacity eviction already relies on, now also
    protecting against idle-TTL eviction."""
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=43)}).json()["series_id"]
    advance(80.0)  # under the TTL
    assert client.get(f"/series/{series_id}").status_code == 200  # touches it, resets the idle clock

    advance(80.0)  # another 80s -- 160s since registration, but only 80s since the touch above
    assert client.get(f"/series/{series_id}").status_code == 200


def test_ttl_expiry_does_not_affect_unexpired_siblings(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    id_old = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=44)}).json()["series_id"]
    advance(90.0)
    id_new = client.post("/series", json={"series": _make_synthetic_series(n=41, seed=45)}).json()["series_id"]
    advance(20.0)  # id_old is now 110s idle (expired); id_new is only 20s idle (fine)

    body = client.get("/series").json()
    assert body["count"] == 1
    assert [s["series_id"] for s in body["series"]] == [id_new]
    assert client.get(f"/series/{id_old}").status_code == 404
    assert client.get(f"/series/{id_new}").status_code == 200


def test_ttl_eviction_does_not_leak_last_access_bookkeeping(monkeypatch):
    """An id evicted by TTL (or by capacity -- see the LRU-capacity test
    above) must not leave a dangling entry in `_series_last_access`; that
    dict's size should track the store's real size, not grow unbounded as
    series come and go."""
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    client.post("/series", json={"series": _make_synthetic_series(n=40, seed=46)})
    advance(150.0)
    client.get("/series")  # triggers the sweep

    assert len(api_module._series_last_access) == 0
    assert len(api_module._series_store) == 0


def test_capacity_eviction_also_cleans_up_last_access_bookkeeping(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_MAX_SIZE", 1)

    client.post("/series", json={"series": _make_synthetic_series(n=40, seed=47)})
    client.post("/series", json={"series": _make_synthetic_series(n=41, seed=48)})  # evicts the first by capacity

    assert len(api_module._series_store) == 1
    assert len(api_module._series_last_access) == 1


# --- Per-entry ttl_remaining_seconds in GET /series (v0.18) -----------------


def test_list_series_reports_full_ttl_for_a_freshly_registered_entry(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    _fake_clock(monkeypatch)  # freeze the clock; no time passes before the GET below

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=49)}).json()["series_id"]

    body = client.get("/series").json()
    assert body["series"] == [
        {"series_id": series_id, "length": 40, "ttl_remaining_seconds": 100.0}
    ]


def test_list_series_ttl_remaining_decreases_with_idle_time(monkeypatch):
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=50)}).json()["series_id"]
    advance(37.0)

    body = client.get("/series").json()
    assert body["series"][0]["series_id"] == series_id
    assert body["series"][0]["ttl_remaining_seconds"] == pytest.approx(63.0)


def test_list_series_ttl_remaining_resets_on_touch(monkeypatch):
    """Resolving a series (e.g. GET /series/{id}) resets its idle clock --
    ttl_remaining_seconds in a later listing should reflect time since that
    touch, not time since the original registration."""
    _fresh_series_store(monkeypatch)
    import forecasting.api as api_module

    monkeypatch.setattr(api_module, "_SERIES_STORE_TTL_SECONDS", 100.0)
    advance = _fake_clock(monkeypatch)

    series_id = client.post("/series", json={"series": _make_synthetic_series(n=40, seed=51)}).json()["series_id"]
    advance(80.0)
    client.get(f"/series/{series_id}")  # touches it, resets the idle clock
    advance(10.0)

    body = client.get("/series").json()
    assert body["series"][0]["ttl_remaining_seconds"] == pytest.approx(90.0)


