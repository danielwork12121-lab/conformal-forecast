"""Tests for `run_nbeats_joint_basis_sweep_comparison`/
`format_nbeats_joint_basis_sweep_comparison` (v0.28) -- the real follow-up
both single-hyperparameter basis sweeps' own README "What's next" named but
neither attempted: whether `trend_degree` and `num_harmonics` interact,
rather than only ever being measured one at a time (v0.25 swept
`trend_degree` alone holding `num_harmonics` at its default; v0.26 swept
`num_harmonics` alone holding `trend_degree` at 2).
"""
from __future__ import annotations

from forecasting.experiment import (
    NBEATSJointBasisSweepComparison,
    format_nbeats_joint_basis_sweep_comparison,
    run_nbeats_basis_sweep_comparison,
    run_nbeats_joint_basis_sweep_comparison,
    run_num_harmonics_sweep_comparison,
)


def test_sweep_runs_every_requested_cell_and_reports_real_numbers():
    degrees = [1, 2, 3]
    harmonics = [1, 2]
    comparison = run_nbeats_joint_basis_sweep_comparison(
        "synthetic", trend_degrees=degrees, num_harmonics_values=harmonics, alpha=0.1, seed=0
    )

    assert isinstance(comparison, NBEATSJointBasisSweepComparison)
    assert len(comparison.cells) == len(degrees) * len(harmonics)
    seen = {(c.trend_degree, c.num_harmonics) for c in comparison.cells}
    assert seen == {(d, h) for d in degrees for h in harmonics}
    for cell in comparison.cells:
        assert cell.mean_interval_width > 0.0
        assert cell.mae > 0.0


def test_single_cell_matches_both_single_hyperparameter_sweeps_exactly():
    """The whole point of this joint sweep reusing the exact same model
    construction as both single-hyperparameter sweeps is that one grid cell
    (trend_degree=2, the shipped default's own num_harmonics) must
    numerically match what BOTH `run_nbeats_basis_sweep_comparison` (at
    degree=2) and `run_num_harmonics_sweep_comparison` (at the shipped
    default) report for the same seed -- checked as a real numeric
    equality, not just assumed from both calling the same underlying class.
    """
    joint = run_nbeats_joint_basis_sweep_comparison(
        "airline", trend_degrees=[2], num_harmonics_values=[1], alpha=0.1, seed=1
    )
    degree_sweep = run_nbeats_basis_sweep_comparison("airline", trend_degrees=[2], alpha=0.1, seed=1)
    harmonics_sweep = run_num_harmonics_sweep_comparison("airline", num_harmonics_values=[1], alpha=0.1, seed=1)

    cell = joint.cells[0]
    assert cell.coverage_gap == degree_sweep.results[0].interpretable["coverage_gap"]
    assert cell.mae == degree_sweep.results[0].interpretable["mae"]
    assert cell.coverage_gap == harmonics_sweep.results[0].interpretable["coverage_gap"]
    assert cell.mae == harmonics_sweep.results[0].interpretable["mae"]


def test_every_cell_gets_its_own_independent_fit():
    """Direct regression guard that this is a real 2D grid of independent
    fits, not one fit relabeled across cells: changing *either*
    hyperparameter alone, holding the other fixed, must change the MAE --
    mirroring each single-hyperparameter sweep's own equivalent guard.
    """
    comparison = run_nbeats_joint_basis_sweep_comparison(
        "airline", trend_degrees=[1, 5], num_harmonics_values=[1, 5], alpha=0.1, seed=0
    )
    by_pos = {(c.trend_degree, c.num_harmonics): c for c in comparison.cells}
    # Varying trend_degree alone (num_harmonics fixed at 1) changes MAE.
    assert by_pos[(1, 1)].mae != by_pos[(5, 1)].mae
    # Varying num_harmonics alone (trend_degree fixed at 1) changes MAE.
    assert by_pos[(1, 1)].mae != by_pos[(1, 5)].mae


def test_shipped_default_harmonics_is_horizon_derived_not_hardcoded():
    comparison = run_nbeats_joint_basis_sweep_comparison(
        "synthetic", trend_degrees=[2], num_harmonics_values=[1], alpha=0.1, seed=0
    )
    # At this project's horizon=1 everywhere _prepare_split is used, max(1, horizon // 2) == 1.
    assert comparison.shipped_default_harmonics == 1


def test_format_includes_every_cell_the_generic_reference_and_the_default_marker():
    comparison = run_nbeats_joint_basis_sweep_comparison(
        "synthetic", trend_degrees=[1, 2], num_harmonics_values=[1, 2], alpha=0.1, seed=0
    )
    text = format_nbeats_joint_basis_sweep_comparison(comparison)
    assert "generic N-BEATS" in text
    assert "| 1 | 1" in text
    assert "| 2 | 2" in text
    assert "shipped default" in text
    assert "best |gap| in this grid" in text
