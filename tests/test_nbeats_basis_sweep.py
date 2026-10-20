"""Tests for `run_nbeats_basis_sweep_comparison`/`format_nbeats_basis_sweep_comparison`
(v0.25) -- the follow-up v0.24's own README "What's next" named but did not
attempt: "sweeping trend_degree/num_harmonics rather than using the paper's
own defaults throughout." This covers the `trend_degree` half of that item.
"""
from __future__ import annotations

from forecasting.experiment import (
    NBEATSBasisSweepComparison,
    format_nbeats_basis_sweep_comparison,
    run_nbeats_basis_sweep_comparison,
    run_experiment,
)


def test_sweep_runs_every_requested_degree_and_reports_generic_reference():
    degrees = [1, 2, 4]
    comparison = run_nbeats_basis_sweep_comparison("synthetic", trend_degrees=degrees, alpha=0.1, seed=0)

    assert isinstance(comparison, NBEATSBasisSweepComparison)
    assert [r.trend_degree for r in comparison.results] == degrees
    assert 0.0 <= comparison.generic["empirical_coverage"] <= 1.0
    assert comparison.generic["mae"] > 0.0
    for r in comparison.results:
        assert 0.0 <= r.interpretable["empirical_coverage"] <= 1.0
        assert r.interpretable["mae"] > 0.0
        assert r.interpretable["n_test"] == comparison.generic["n_test"]


def test_degree_two_in_the_sweep_matches_run_experiment_for_the_same_seed():
    """The whole point of including `trend_degree=2` in the swept list
    (rather than training it separately as a second reference) is that it
    must be the SAME model `run_experiment`'s own "N-BEATS interpretable
    (delta)" row trains -- same architecture defaults, same training
    discipline, same seed. Checked directly here as a real numeric
    equality, not just assumed from both using the same constructor
    defaults: if a future edit drifted the sweep's hyperparameters away
    from `run_experiment`'s model_factories entry, this would catch it.
    """
    sweep = run_nbeats_basis_sweep_comparison("airline", trend_degrees=[2], alpha=0.1, seed=1)
    direct = [r for r in run_experiment("airline", alpha=0.1, seed=1) if r.name == "N-BEATS interpretable (delta)"][0]

    swept_cov = sweep.results[0].interpretable
    assert swept_cov["empirical_coverage"] == direct.coverage["empirical_coverage"]
    assert swept_cov["mean_interval_width"] == direct.coverage["mean_interval_width"]
    assert swept_cov["mae"] == direct.mae


def test_higher_trend_degree_changes_point_predictions_not_just_coverage():
    """A direct regression guard that each swept degree really gets its own
    independent fit (unlike the EnbPI/ACI window sweeps, which deliberately
    *reuse* one fit): `trend_degree` changes `NBEATSTrendBlock`'s own basis
    and `theta_proj` output size, so a degree-1 model and a degree-5 model
    on the same data/seed should not land on the same MAE. If this function
    accidentally fit one model and relabeled it for every degree, every
    row's MAE would be identical.
    """
    comparison = run_nbeats_basis_sweep_comparison("airline", trend_degrees=[1, 5], alpha=0.1, seed=0)
    low, high = comparison.results[0].interpretable, comparison.results[1].interpretable
    assert low["mae"] != high["mae"]


def test_format_includes_every_degree_the_generic_reference_and_the_default_marker():
    comparison = run_nbeats_basis_sweep_comparison("synthetic", trend_degrees=[1, 2, 3], alpha=0.1, seed=0)
    text = format_nbeats_basis_sweep_comparison(comparison)
    assert "generic N-BEATS (reference)" in text
    assert "| 1 |" in text
    assert "| 3 |" in text
    assert "2 **<- shipped default**" in text
