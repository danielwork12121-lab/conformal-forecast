"""Integration test: `run_experiment`'s model_factories include the new
Transformer family (v0.20) end-to-end, on a real bundled dataset.

`tests/test_models.py` already proves TransformerForecaster's own fit/predict
mechanics in isolation. This file checks the thing a reader of the README
benchmark tables actually cares about: that wiring it into `run_experiment`
(the same function `python -m forecasting.cli benchmark` calls) produces a
real `ModelResult` with sane, in-range numbers on a real dataset -- not just
that the class itself works standalone.
"""
from __future__ import annotations

import math

from forecasting.experiment import run_experiment


def test_run_experiment_includes_the_transformer_model_with_sane_results():
    results = run_experiment("synthetic", alpha=0.1, seed=0)
    by_name = {r.name: r for r in results}

    assert "Transformer (delta)" in by_name
    transformer_result = by_name["Transformer (delta)"]

    # Real numbers, not NaN/inf -- a broken forward pass or a shape bug
    # would show up here before it ever reached a README table.
    assert math.isfinite(transformer_result.mae)
    assert math.isfinite(transformer_result.rmse)
    assert math.isfinite(transformer_result.mape)
    assert 0.0 <= transformer_result.coverage["empirical_coverage"] <= 1.0
    assert transformer_result.coverage["mean_interval_width"] > 0.0
    assert transformer_result.coverage["n_test"] == by_name["Naive"].coverage["n_test"]

    # All four models ran on the exact same test split (same n_test),
    # so their coverage/width/MAE numbers are directly comparable in a
    # README table -- not an artifact of different models seeing
    # different amounts of test data.
    n_test_values = {r.coverage["n_test"] for r in results}
    assert len(n_test_values) == 1


def test_run_experiment_transformer_and_lstm_are_independently_seeded():
    # Both neural models share `seed` end-to-end (same torch.manual_seed
    # call inside their own fit()), but they're different architectures
    # with different parameter counts/initialization schemes -- their
    # predictions on the same data should not come out identical, which
    # would indicate one of them isn't actually training (e.g. silently
    # falling back to returning its input unchanged).
    results = run_experiment("synthetic", alpha=0.1, seed=0)
    by_name = {r.name: r for r in results}
    lstm_pred = by_name["LSTM (delta)"].point_pred
    transformer_pred = by_name["Transformer (delta)"].point_pred
    assert lstm_pred.shape == transformer_pred.shape
    assert not (lstm_pred == transformer_pred).all()


def test_run_experiment_includes_the_nbeats_model_with_sane_results():
    results = run_experiment("synthetic", alpha=0.1, seed=0)
    by_name = {r.name: r for r in results}

    assert "N-BEATS (delta)" in by_name
    nbeats_result = by_name["N-BEATS (delta)"]

    assert math.isfinite(nbeats_result.mae)
    assert math.isfinite(nbeats_result.rmse)
    assert math.isfinite(nbeats_result.mape)
    assert 0.0 <= nbeats_result.coverage["empirical_coverage"] <= 1.0
    assert nbeats_result.coverage["mean_interval_width"] > 0.0
    assert nbeats_result.coverage["n_test"] == by_name["Naive"].coverage["n_test"]


def test_run_experiment_all_three_neural_models_are_independently_seeded():
    # LSTM, Transformer, and N-BEATS all share `seed` end-to-end, but are
    # three structurally distinct architectures (recurrence / attention /
    # stacked-FC-with-residual-basis-expansion) -- their predictions on the
    # same data should not come out identical, which would indicate one of
    # them isn't actually training.
    results = run_experiment("synthetic", alpha=0.1, seed=0)
    by_name = {r.name: r for r in results}
    lstm_pred = by_name["LSTM (delta)"].point_pred
    transformer_pred = by_name["Transformer (delta)"].point_pred
    nbeats_pred = by_name["N-BEATS (delta)"].point_pred
    assert lstm_pred.shape == transformer_pred.shape == nbeats_pred.shape
    assert not (lstm_pred == transformer_pred).all()
    assert not (lstm_pred == nbeats_pred).all()
    assert not (transformer_pred == nbeats_pred).all()
