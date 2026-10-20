"""Tests for `run_num_harmonics_sweep_comparison`/`format_num_harmonics_sweep_comparison`
(v0.26) -- the direct follow-up `run_nbeats_basis_sweep_comparison`'s own
docstring named but did not attempt: sweeping `num_harmonics` (the
seasonality basis's own hyperparameter) while holding `trend_degree` at
its shipped default (2), mirroring the `trend_degree` sweep (v0.25) for a
different hyperparameter of the same model family.
"""
from __future__ import annotations

from forecasting.experiment import (
    NumHarmonicsSweepComparison,
    format_num_harmonics_sweep_comparison,
    run_num_harmonics_sweep_comparison,
    run_experiment,
)


def test_sweep_runs_every_requested_value_and_reports_generic_reference():
    values = [1, 2, 4]
    comparison = run_num_harmonics_sweep_comparison("synthetic", num_harmonics_values=values, alpha=0.1, seed=0)

    assert isinstance(comparison, NumHarmonicsSweepComparison)
    assert [r.num_harmonics for r in comparison.results] == values
    assert comparison.shipped_default == 1  # horizon=1 everywhere in this project -> max(1, 1 // 2) == 1
    assert 0.0 <= comparison.generic["empirical_coverage"] <= 1.0
    assert comparison.generic["mae"] > 0.0
    for r in comparison.results:
        assert 0.0 <= r.interpretable["empirical_coverage"] <= 1.0
        assert r.interpretable["mae"] > 0.0
        assert r.interpretable["n_test"] == comparison.generic["n_test"]


def test_shipped_default_in_the_sweep_matches_run_experiment_for_the_same_seed():
    """`num_harmonics=1` in the sweep must be numerically identical to
    `run_experiment`'s own "N-BEATS interpretable (delta)" row for the
    same seed, since that row's `NBEATSInterpretableForecaster` call leaves
    `num_harmonics=None`, which `NBEATSSeasonalityBlock` itself resolves to
    `max(1, horizon // 2) == 1` at this project's horizon (1). Checked as a
    real numeric equality (not just assumed from both using the same
    constructor defaults), the same guard `test_nbeats_basis_sweep.py`
    already applies to `trend_degree=2`.
    """
    sweep = run_num_harmonics_sweep_comparison("airline", num_harmonics_values=[1], alpha=0.1, seed=1)
    direct = [r for r in run_experiment("airline", alpha=0.1, seed=1) if r.name == "N-BEATS interpretable (delta)"][0]

    swept_cov = sweep.results[0].interpretable
    assert swept_cov["empirical_coverage"] == direct.coverage["empirical_coverage"]
    assert swept_cov["mean_interval_width"] == direct.coverage["mean_interval_width"]
    assert swept_cov["mae"] == direct.mae


def test_higher_num_harmonics_changes_point_predictions_not_just_coverage():
    """A direct regression guard, mirroring `test_nbeats_basis_sweep.py`'s
    analogous `trend_degree` check: `num_harmonics` changes
    `NBEATSSeasonalityBlock`'s own basis and `theta_proj` output size, so a
    1-harmonic model and a 5-harmonic model on the same data/seed should
    not land on the same MAE. If this function accidentally fit one model
    and relabeled it for every value, every row's MAE would be identical.
    """
    comparison = run_num_harmonics_sweep_comparison("airline", num_harmonics_values=[1, 5], alpha=0.1, seed=0)
    low, high = comparison.results[0].interpretable, comparison.results[1].interpretable
    assert low["mae"] != high["mae"]


def test_format_includes_every_value_the_generic_reference_and_the_default_marker():
    comparison = run_num_harmonics_sweep_comparison("synthetic", num_harmonics_values=[1, 2, 3], alpha=0.1, seed=0)
    text = format_num_harmonics_sweep_comparison(comparison)
    assert "generic N-BEATS (reference)" in text
    assert "| 2 |" in text
    assert "| 3 |" in text
    assert "1 **<- shipped default**" in text
