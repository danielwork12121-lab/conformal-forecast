"""Tests for `run_nbeats_block_count_sweep_comparison`/
`format_nbeats_block_count_sweep_comparison` (v0.29) -- the real follow-up
every prior N-BEATS interpretable-basis sweep (v0.25 `trend_degree`, v0.26
`num_harmonics`, v0.28's joint sweep of the two) left open: those all swept
a basis's own per-block *shape*, holding stack *depth*
(`num_trend_blocks`/`num_seasonality_blocks`) fixed at the constructor
default (2, 2) throughout. This sweeps depth instead, holding shape
(`trend_degree`/`num_harmonics`) at its shipped default.
"""
from __future__ import annotations

from forecasting.experiment import (
    NBEATSBlockCountSweepComparison,
    format_nbeats_block_count_sweep_comparison,
    run_nbeats_block_count_sweep_comparison,
    run_nbeats_joint_basis_sweep_comparison,
)


def test_sweep_runs_every_requested_cell_and_reports_real_numbers():
    trend_blocks = [1, 2, 3]
    season_blocks = [1, 2]
    comparison = run_nbeats_block_count_sweep_comparison(
        "synthetic",
        num_trend_blocks_values=trend_blocks,
        num_seasonality_blocks_values=season_blocks,
        alpha=0.1,
        seed=0,
    )

    assert isinstance(comparison, NBEATSBlockCountSweepComparison)
    assert len(comparison.cells) == len(trend_blocks) * len(season_blocks)
    seen = {(c.num_trend_blocks, c.num_seasonality_blocks) for c in comparison.cells}
    assert seen == {(t, s) for t in trend_blocks for s in season_blocks}
    for cell in comparison.cells:
        assert cell.mean_interval_width > 0.0
        assert cell.mae > 0.0


def test_shipped_default_combo_matches_v028_joint_basis_sweep_exactly():
    """At (num_trend_blocks=2, num_seasonality_blocks=2) -- the constructor
    default this project has used in every prior N-BEATS sweep -- this
    sweep's model construction is identical to v0.28's own joint basis
    sweep evaluated at (trend_degree=2, num_harmonics=shipped default):
    both build `NBEATSInterpretableForecaster(num_trend_blocks=2,
    num_seasonality_blocks=2, trend_degree=2, num_harmonics=<shipped>,
    seed=<same>)`. Checked as a real numeric equality against the
    already-published v0.28 sweep, not just assumed from the shared
    construction.
    """
    block_sweep = run_nbeats_block_count_sweep_comparison(
        "airline",
        num_trend_blocks_values=[2],
        num_seasonality_blocks_values=[2],
        alpha=0.1,
        seed=1,
    )
    shipped_harmonics = block_sweep.num_harmonics
    basis_sweep = run_nbeats_joint_basis_sweep_comparison(
        "airline", trend_degrees=[2], num_harmonics_values=[shipped_harmonics], alpha=0.1, seed=1
    )

    block_cell = block_sweep.cells[0]
    basis_cell = basis_sweep.cells[0]
    assert block_cell.coverage_gap == basis_cell.coverage_gap
    assert block_cell.mae == basis_cell.mae
    assert block_cell.mean_interval_width == basis_cell.mean_interval_width


def test_every_cell_gets_its_own_independent_fit():
    """Direct regression guard that this is a real 2D grid of independent
    fits, not one fit relabeled across cells: changing *either*
    hyperparameter alone, holding the other fixed, must change the MAE.
    """
    comparison = run_nbeats_block_count_sweep_comparison(
        "airline",
        num_trend_blocks_values=[1, 4],
        num_seasonality_blocks_values=[1, 4],
        alpha=0.1,
        seed=0,
    )
    by_pos = {(c.num_trend_blocks, c.num_seasonality_blocks): c for c in comparison.cells}
    # Varying num_trend_blocks alone (num_seasonality_blocks fixed at 1) changes MAE.
    assert by_pos[(1, 1)].mae != by_pos[(4, 1)].mae
    # Varying num_seasonality_blocks alone (num_trend_blocks fixed at 1) changes MAE.
    assert by_pos[(1, 1)].mae != by_pos[(1, 4)].mae


def test_trend_degree_and_num_harmonics_are_shipped_defaults_not_hardcoded_wrong():
    comparison = run_nbeats_block_count_sweep_comparison(
        "synthetic",
        num_trend_blocks_values=[2],
        num_seasonality_blocks_values=[2],
        alpha=0.1,
        seed=0,
    )
    assert comparison.trend_degree == 2
    # At this project's horizon=1 everywhere _prepare_split is used, max(1, horizon // 2) == 1.
    assert comparison.num_harmonics == 1


def test_format_includes_every_cell_the_generic_reference_and_the_default_marker():
    comparison = run_nbeats_block_count_sweep_comparison(
        "synthetic",
        num_trend_blocks_values=[1, 2],
        num_seasonality_blocks_values=[1, 2],
        alpha=0.1,
        seed=0,
    )
    text = format_nbeats_block_count_sweep_comparison(comparison)
    assert "generic N-BEATS" in text
    assert "| 1 | 1" in text
    assert "| 2 | 2" in text
    assert "shipped default" in text
    assert "best |gap| in this grid" in text
